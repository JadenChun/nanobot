"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import weakref
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain
from nanobot.utils.prompt_budget import (
    AsyncPortableFileLock,
    PromptBudget,
    measure_prompt,
    portable_file_lock,
)

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_save_memory_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None

_TOOL_CHOICE_EXPLICIT_MARKERS = (
    'should be ["none", "auto"]',
    "tool_choice parameter does not support",
    "forced tool_choice",
    "tool_choice not supported",
)

_TOOL_CHOICE_NAME_MARKERS = ("tool_choice", "toolchoice")


def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported.

    Requires either an explicit known phrase, or a mention of `tool_choice`
    combined with a rejection keyword — this avoids false positives from
    unrelated errors like "model does not support reasoning_effort".
    """
    text = (content or "").lower()
    if not text:
        return False
    if any(m in text for m in _TOOL_CHOICE_EXPLICIT_MARKERS):
        return True
    mentions_tool_choice = any(m in text for m in _TOOL_CHOICE_NAME_MARKERS)
    rejection_hint = any(
        m in text for m in ("does not support", "unsupported", "not allowed", "invalid")
    )
    return mentions_tool_choice and rejection_hint


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3
    _MAX_RAW_ARCHIVE_CHARS = 8_000
    _MAX_RENDERED_PROMPT_CHARS = 256_000
    _RECEIPT_VERSION = 1
    _RECEIPT_FILE_NAME = ".consolidation.pending.json"
    _COMMIT_LOCK_FILE_NAME = ".consolidation.lock"
    _TRANSACTION_LOCK_FILE_NAME = ".consolidation.transaction.lock"
    _HISTORY_MARKER_PREFIX = "<!-- nanobot-consolidation:"
    _CONSOLIDATION_SYSTEM_PROMPT = (
        "You are a memory consolidation agent. Summarize conversations into concise, "
        "actionable memory. Focus on preserving facts and decisions, not verbatim dialogue. "
        "Call the save_memory tool with your consolidation."
    )

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        # The receipt is intentionally a dot-file under the workspace memory
        # directory.  It contains no prompt text beyond the bounded generated
        # summary and is safe to recover on the next process start.
        self.pending_receipt_file = self.memory_dir / self._RECEIPT_FILE_NAME
        self.pending_file = self.pending_receipt_file  # compatibility alias
        self.pending_receipt_path = self.pending_receipt_file
        self.receipt_file = self.pending_receipt_file
        self.commit_lock_file = self.memory_dir / self._COMMIT_LOCK_FILE_NAME
        self.commit_lock_path = self.commit_lock_file
        self.transaction_lock_file = self.memory_dir / self._TRANSACTION_LOCK_FILE_NAME
        self.transaction_lock_path = self.transaction_lock_file
        self._last_receipt_conflict = False
        self._consecutive_failures = 0
        # Cache of (provider class, model) pairs known to reject forced tool_choice,
        # so we skip the forced attempt and go straight to "auto" on later calls.
        self._forced_tool_choice_unsupported: set[tuple[str, str]] = set()
        # A MemoryConsolidator may install a request budget on this store so
        # direct callers retain the historical ``consolidate`` signature.
        self.prompt_budget: PromptBudget | None = None

        # Complete any memory/history half-commit left by a prior process.
        # Session offsets, when present in the receipt, are completed by the
        # owning MemoryConsolidator which can supply its SessionManager.
        try:
            self.recover_pending_receipt()
        except Exception:
            logger.exception("Failed to recover pending memory consolidation")

    @staticmethod
    def _provider_cache_key(provider: LLMProvider, model: str) -> tuple[str, str]:
        return (type(provider).__name__, model)

    def _read_long_term_unlocked(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def read_long_term(self) -> str:
        """Read a complete MEMORY.md snapshot without waiting on consolidation."""
        # Consolidation holds the transaction lock across provider I/O.  The
        # memory file itself is replaced atomically, so readers can safely see
        # either the old or the new complete snapshot without joining that
        # long-running transaction.
        return self._read_long_term_unlocked()

    def write_long_term(self, content: str) -> None:
        # Direct writes must serialize with an in-flight consolidation's
        # read/LLM/commit transaction or a receipt CAS would overwrite them.
        with portable_file_lock(self.transaction_lock_file):
            self._atomic_write_text(self.memory_file, content)

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        """Replace a text file atomically and force its contents to disk."""
        ensure_dir(path.parent)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _history_entry_with_source(entry: str, source: str | None = None) -> str:
        """Apply the optional source tag without changing existing markers."""
        if source and "[source=" not in entry:
            if entry.startswith("[") and "] " in entry:
                bracket_end = entry.index("] ") + 2
                entry = f"{entry[:bracket_end]}[source={source}] {entry[bracket_end:]}"
            else:
                entry = f"[source={source}] {entry}"
        return entry

    def _append_history_unlocked(self, entry: str) -> None:
        """Append and fsync one history record while the commit lock is held."""
        ensure_dir(self.history_file.parent)
        with open(self.history_file, "a", encoding="utf-8") as handle:
            handle.write(entry.rstrip() + "\n\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append_history(self, entry: str, source: str | None = None) -> None:
        """Append an entry to HISTORY.md.

        Args:
            entry: History entry (may include timestamp prefix)
            source: Optional source tag (e.g., "cron", "chat") for filtering
        """
        entry = self._history_entry_with_source(entry, source)
        with portable_file_lock(self.commit_lock_file):
            self._append_history_unlocked(entry)

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    async def compress_to_target(
        self,
        target_tokens: int,
        provider: LLMProvider,
        model: str,
    ) -> str:
        """Compress MEMORY.md to target token count.

        Uses LLM to compress memory while preserving key facts and recent events.

        Args:
            target_tokens: Target token count for compressed memory
            provider: LLM provider for compression
            model: Model to use for compression

        Returns:
            Compressed memory content (does not write to disk)
        """
        current = self.read_long_term()
        if not current:
            return ""

        # Rough token estimate (4 chars per token)
        current_tokens = len(current) // 4

        if current_tokens <= target_tokens:
            return current

        logger.info(
            "Compressing memory from ~{} to ~{} tokens",
            current_tokens,
            target_tokens,
        )

        prompt = f"""Compress this agent memory to ~{target_tokens} tokens.

Preserve:
- Key facts, decisions, and user preferences
- Recent events and current task state
- Important file names and paths
- Errors and how they were resolved

Remove:
- Redundant details and verbose descriptions
- Old events that are no longer relevant
- Verbatim dialogue (keep summaries instead)

Current memory:
{current}

Return compressed memory only (no explanation)."""

        try:
            response = await provider.chat_with_retry(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=target_tokens + 200,
            )

            compressed = response.content or ""

            logger.info(
                "Compressed memory from {} to {} chars",
                len(current),
                len(compressed),
            )
            return compressed

        except Exception as e:
            logger.error("Failed to compress memory: {}", e)
            # Fallback: truncate to max tokens
            max_chars = target_tokens * 4
            if len(current) > max_chars:
                truncated = current[:max_chars] + "\n\n[Truncated due to size]"
                return truncated
            return current

    @classmethod
    def _format_messages(
        cls,
        messages: list[dict],
        *,
        max_chars: int | None = None,
    ) -> str:
        """Render persisted messages into a bounded consolidation transcript."""
        lines: list[str] = []
        used = 0
        limit = max_chars if max_chars is not None else cls._MAX_RENDERED_PROMPT_CHARS
        for message in messages:
            content = message.get("content")
            if not content:
                continue
            tools = (
                f" [tools: {', '.join(str(item) for item in message['tools_used'])}]"
                if message.get("tools_used")
                else ""
            )
            timestamp = str(message.get("timestamp", "?"))[:16]
            line = f"[{timestamp}] {str(message.get('role', '?')).upper()}{tools}: {content}"
            remaining = limit - used
            if remaining <= 0:
                break
            if len(line) > remaining:
                lines.append(line[:remaining].rstrip() + "\n... (transcript truncated)")
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    def render_consolidation_prompt(
        self,
        messages: list[dict],
        *,
        current_memory: str | None = None,
        max_chars: int | None = None,
    ) -> list[dict[str, Any]]:
        """Render the exact system/user payload used for consolidation.

        The rendered transcript is bounded before token measurement.  The
        measured payload and ``_SAVE_MEMORY_TOOL`` are the same values passed
        to the provider, preventing selection from relying on a different
        approximation than the actual auxiliary request.
        """
        if current_memory is None:
            current_memory = self.read_long_term()
        render_limit = (
            self._MAX_RENDERED_PROMPT_CHARS
            if max_chars is None
            else max(0, int(max_chars))
        )
        memory_text = str(current_memory or "(empty)")
        # Reserve room for the transcript and fixed instructions.  This keeps
        # the helper bounded even when MEMORY.md itself grew unexpectedly.
        memory_limit = max(0, render_limit // 2)
        if len(memory_text) > memory_limit:
            memory_text = memory_text[:memory_limit].rstrip() + "\n... (memory truncated)"
        transcript = self._format_messages(
            messages,
            max_chars=max(0, render_limit - len(memory_text)),
        )
        prompt = f"""Summarize this conversation for continuity. Preserve:
- Key facts, decisions, and user preferences
- Files examined or modified (names and paths, not full contents)
- Errors encountered and how they were resolved
- Current task state and next steps
- Any learnings useful for future conversations

Call the save_memory tool with your consolidation.

## Current Long-term Memory
{memory_text}

## Conversation to Process
{transcript}"""
        return [
            {"role": "system", "content": self._CONSOLIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    # Compatibility/descriptive aliases for callers that use ``build`` or
    # ``render`` terminology for this request-local helper.
    build_consolidation_prompt = render_consolidation_prompt
    build_consolidation_messages = render_consolidation_prompt
    render_consolidation_messages = render_consolidation_prompt
    render_prompt = render_consolidation_prompt

    def measure_consolidation_prompt(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
        budget: PromptBudget | None = None,
        *,
        current_memory: str | None = None,
        max_chars: int | None = None,
    ):
        """Measure the fully rendered auxiliary request including its tool."""
        rendered = self.render_consolidation_prompt(
            messages,
            current_memory=current_memory,
            max_chars=max_chars,
        )
        return measure_prompt(provider, model, rendered, _SAVE_MEMORY_TOOL, budget)

    measure_prompt = measure_consolidation_prompt

    @staticmethod
    def _digest_messages(messages: list[dict]) -> str:
        payload = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _digest_text(value: str) -> str:
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()

    def _history_contains_marker(self, marker_id: str) -> bool:
        if not marker_id or not self.history_file.exists():
            return False
        marker = f"{self._HISTORY_MARKER_PREFIX}{marker_id} -->"
        try:
            with self.history_file.open(encoding="utf-8") as handle:
                return any(marker in line for line in handle)
        except OSError:
            return False

    def _history_contains_entry(self, entry: str) -> bool:
        if not entry or not self.history_file.exists():
            return False
        try:
            return entry in self.history_file.read_text(encoding="utf-8")
        except OSError:
            return False

    def _write_receipt_unlocked(self, receipt: dict[str, Any]) -> None:
        self._atomic_write_text(
            self.pending_receipt_file,
            json.dumps(receipt, ensure_ascii=False, sort_keys=True),
        )

    def _read_receipt_unlocked(self) -> dict[str, Any] | None:
        if not self.pending_receipt_file.exists():
            return None
        try:
            value = json.loads(self.pending_receipt_file.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _apply_session_offset(
        self,
        receipt: dict[str, Any],
        session_manager: SessionManager | None,
    ) -> bool:
        session_key = receipt.get("session_key")
        end_offset = receipt.get("end_offset")
        if not session_key or not isinstance(end_offset, int):
            return True
        if session_manager is None:
            # Leave the receipt for the owning consolidator to finish.  This
            # is important when a new MemoryStore is constructed before the
            # SessionManager has been initialized during process startup.
            return False
        session = session_manager.get_or_create(str(session_key))
        expected_generation = receipt.get(
            "session_generation",
            receipt.get("generation", session.generation),
        )
        try:
            expected_generation = int(expected_generation)
        except (TypeError, ValueError):
            expected_generation = session.generation
        return session_manager.advance_last_consolidated(
            session,
            expected_generation,
            str(receipt.get("chunk_digest") or ""),
            start_offset=(
                int(receipt["start_offset"])
                if isinstance(receipt.get("start_offset"), int)
                else None
            ),
            end_offset=end_offset,
        )

    def _apply_receipt_unlocked(
        self,
        receipt: dict[str, Any],
        session_manager: SessionManager | None = None,
    ) -> bool:
        """Apply a receipt idempotently; return whether it is complete."""
        self._last_receipt_conflict = False
        marker_id = str(receipt.get("history_marker_id") or "")
        entry = str(receipt.get("history_entry") or "").strip()
        if entry:
            if marker_id:
                if not self._history_contains_marker(marker_id):
                    marker = f"{self._HISTORY_MARKER_PREFIX}{marker_id} -->"
                    self._append_history_unlocked(f"{marker}\n{entry}")
            elif not self._history_contains_entry(entry):
                self._append_history_unlocked(entry)

        memory_output = receipt.get("memory_output", receipt.get("memory_update"))
        if memory_output is not None:
            update = _ensure_text(memory_output)
            current = self._read_long_term_unlocked()
            current_digest = self._digest_text(current)
            output_digest = str(
                receipt.get("output_memory_digest")
                or receipt.get("memory_output_digest")
                or ""
            )
            if output_digest and current_digest == output_digest:
                # The output was already applied before a process interruption.
                pass
            elif receipt.get("base_memory_digest") or receipt.get("memory_base_digest"):
                base_digest = str(
                    receipt.get("base_memory_digest")
                    or receipt.get("memory_base_digest")
                )
                if current_digest != base_digest:
                    # Keep the current memory chosen by another writer.  The
                    # history entry is still useful for auditability, but this
                    # obsolete receipt must never advance the session offset.
                    self._last_receipt_conflict = True
                    logger.warning("Discarding stale memory consolidation receipt")
                    return True
                self._atomic_write_text(self.memory_file, update)
            elif current != update:
                # Legacy receipts predate CAS digests; preserve their original
                # idempotent recovery behavior.
                self._atomic_write_text(self.memory_file, update)

        offset_applied = self._apply_session_offset(receipt, session_manager)
        if not offset_applied and session_manager is not None:
            # A receipt tied to an unavailable/different session generation is
            # obsolete.  Do not leave it to retry forever against a newly
            # cleared stream; memory remains durable while the new generation
            # retains its own offset.
            self._last_receipt_conflict = True
            logger.warning("Discarding stale consolidation session offset")
            return True
        return offset_applied

    def _finish_receipt_unlocked(
        self,
        receipt: dict[str, Any],
        session_manager: SessionManager | None = None,
    ) -> bool:
        if not self._apply_receipt_unlocked(receipt, session_manager):
            return False
        if self._last_receipt_conflict:
            # Disposing the obsolete receipt prevents it from repeatedly
            # retrying against a deliberately preserved newer memory value.
            try:
                self.pending_receipt_file.unlink()
            except FileNotFoundError:
                pass
            return True
        try:
            self.pending_receipt_file.unlink()
        except FileNotFoundError:
            pass
        return True

    def recover_pending_receipt(
        self,
        session_manager: SessionManager | None = None,
    ) -> bool:
        """Recover a prior half-commit, returning whether it was completed."""
        # Lock order is transaction -> receipt commit -> session file.  The
        # synchronous recovery path is used during startup, before any LLM
        # request is made.
        with portable_file_lock(self.transaction_lock_file):
            with portable_file_lock(self.commit_lock_file):
                receipt = self._read_receipt_unlocked()
                if receipt is None:
                    return True
                try:
                    return self._finish_receipt_unlocked(receipt, session_manager)
                except Exception:
                    logger.exception("Failed to apply pending memory receipt")
                    return False

    async def _commit_receipt(
        self,
        receipt: dict[str, Any],
        session_manager: SessionManager | None = None,
    ) -> bool:
        """Durably apply one receipt under the transaction and commit locks."""
        async with AsyncPortableFileLock(self.transaction_lock_file):
            return await self._commit_receipt_locked(receipt, session_manager)

    async def _commit_receipt_locked(
        self,
        receipt: dict[str, Any],
        session_manager: SessionManager | None = None,
    ) -> bool:
        """Apply a receipt when the workspace transaction lock is held."""
        async with AsyncPortableFileLock(self.commit_lock_file):
            # A concurrent process may have left an older receipt.  Finish it
            # before writing this one so progress is monotonic.
            pending = self._read_receipt_unlocked()
            if pending is not None:
                if not self._finish_receipt_unlocked(pending, session_manager):
                    return False
            self._write_receipt_unlocked(receipt)
            try:
                return self._finish_receipt_unlocked(receipt, session_manager)
            except Exception:
                logger.exception("Memory consolidation commit interrupted")
                return False

    def _make_receipt(
        self,
        *,
        messages: list[dict],
        history_entry: str,
        memory_output: Any = None,
        source: str | None = None,
        session_key: str | None = None,
        start_offset: int | None = None,
        end_offset: int | None = None,
        history_marker_id: str | None = None,
        reference_only: bool = False,
        base_memory: str | None = None,
        base_memory_digest: str | None = None,
        session_generation: int | None = None,
    ) -> dict[str, Any]:
        """Build the bounded, JSON-safe receipt for one consolidation chunk."""
        digest = self._digest_messages(messages)
        if base_memory_digest is None:
            if base_memory is None:
                base_memory = self.read_long_term()
            base_memory_digest = self._digest_text(base_memory)
        output_text = _ensure_text(memory_output) if memory_output is not None else None
        marker_id = history_marker_id
        if marker_id is None and session_key:
            marker_id = digest[:24]
        entry = self._history_entry_with_source(str(history_entry).strip(), source)
        receipt: dict[str, Any] = {
            "version": self._RECEIPT_VERSION,
            "chunk_digest": digest,
            "history_marker_id": marker_id or "",
            "history_entry": entry[: self._MAX_RAW_ARCHIVE_CHARS],
            "memory_output": output_text,
            "base_memory_digest": str(base_memory_digest),
            "output_memory_digest": (
                self._digest_text(output_text) if output_text is not None else None
            ),
            "reference_only": bool(reference_only),
            "created_at": datetime.now().isoformat(),
        }
        if session_key:
            receipt["session_key"] = str(session_key)
        if start_offset is not None:
            receipt["start_offset"] = int(start_offset)
        if end_offset is not None:
            receipt["end_offset"] = int(end_offset)
        if session_generation is not None:
            receipt["session_generation"] = int(session_generation)
        # Keep the explicit field name used by older recovery scripts too.
        receipt["memory_update"] = receipt["memory_output"]
        # Descriptive aliases retained for recovery tooling that used the
        # longer names before the canonical fields were documented.
        receipt["memory_base_digest"] = receipt["base_memory_digest"]
        receipt["memory_output_digest"] = receipt["output_memory_digest"]
        return receipt

    async def _commit_reference(
        self,
        messages: list[dict],
        *,
        reason: str,
        source: str | None = None,
        session_key: str | None = None,
        start_offset: int | None = None,
        end_offset: int | None = None,
        history_marker_id: str | None = None,
        session_manager: SessionManager | None = None,
        session_generation: int | None = None,
        base_memory: str | None = None,
        _transaction_held: bool = False,
    ) -> bool:
        """Durably archive a bounded reference instead of raw oversized text."""
        digest = self._digest_messages(messages)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = (
            f"[{ts}] [REFERENCE] Memory consolidation skipped ({reason}); "
            f"messages={len(messages)} digest={digest[:24]}"
        )
        receipt = self._make_receipt(
            messages=messages,
            history_entry=entry,
            source=source,
            session_key=session_key,
            start_offset=start_offset,
            end_offset=end_offset,
            history_marker_id=history_marker_id,
            reference_only=True,
            base_memory=base_memory,
            session_generation=session_generation,
        )
        if _transaction_held:
            return await self._commit_receipt_locked(receipt, session_manager)
        return await self._commit_receipt(receipt, session_manager)

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
        max_output_tokens: int | None = None,
        source: str | None = None,
        prompt_budget: PromptBudget | None = None,
        session_key: str | None = None,
        start_offset: int | None = None,
        end_offset: int | None = None,
        history_marker_id: str | None = None,
        session_manager: SessionManager | None = None,
        reference_only: bool = False,
        session_generation: int | None = None,
    ) -> bool:
        """Run one complete consolidation as a workspace transaction."""
        if not messages:
            return True
        async with AsyncPortableFileLock(self.transaction_lock_file):
            return await self._consolidate_locked(
                messages,
                provider,
                model,
                max_output_tokens=max_output_tokens,
                source=source,
                prompt_budget=prompt_budget,
                session_key=session_key,
                start_offset=start_offset,
                end_offset=end_offset,
                history_marker_id=history_marker_id,
                session_manager=session_manager,
                reference_only=reference_only,
                session_generation=session_generation,
            )

    async def _consolidate_locked(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
        max_output_tokens: int | None = None,
        source: str | None = None,
        prompt_budget: PromptBudget | None = None,
        session_key: str | None = None,
        start_offset: int | None = None,
        end_offset: int | None = None,
        history_marker_id: str | None = None,
        session_manager: SessionManager | None = None,
        reference_only: bool = False,
        session_generation: int | None = None,
    ) -> bool:
        """Consolidate the provided message chunk into MEMORY.md + HISTORY.md.

        Args:
            messages: Message list to consolidate
            provider: LLM provider for consolidation
            model: Model to use
            max_output_tokens: Max tokens for output
            source: Optional source tag for HISTORY.md entries (e.g., "cron", "chat")
        """
        if not messages:
            return True

        # The transaction lock is already held here.  Recovery acquires only
        # the nested receipt lock, preserving transaction -> receipt ->
        # session ordering without recursively taking the transaction lock.
        with portable_file_lock(self.commit_lock_file):
            pending = self._read_receipt_unlocked()
            if pending is not None:
                try:
                    self._finish_receipt_unlocked(pending, session_manager)
                except Exception:
                    logger.exception("Failed to apply pending memory receipt")
                    return False
        current_memory = self._read_long_term_unlocked()
        budget = prompt_budget or self.prompt_budget
        chat_messages = self.render_consolidation_prompt(
            messages,
            current_memory=current_memory,
        )

        try:
            if reference_only:
                return await self._commit_reference(
                    messages,
                    reason="raw message cap exceeded",
                    source=source,
                    session_key=session_key,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    history_marker_id=history_marker_id,
                    session_manager=session_manager,
                    session_generation=session_generation,
                    base_memory=current_memory,
                    _transaction_held=True,
                )
            if budget is not None:
                measurement = measure_prompt(
                    provider,
                    model,
                    chat_messages,
                    _SAVE_MEMORY_TOOL,
                    budget,
                )
                if not measurement.fits:
                    return await self._commit_reference(
                        messages,
                        reason="auxiliary prompt budget exceeded",
                        source=source,
                        session_key=session_key,
                        start_offset=start_offset,
                        end_offset=end_offset,
                        history_marker_id=history_marker_id,
                        session_manager=session_manager,
                        session_generation=session_generation,
                        base_memory=current_memory,
                        _transaction_held=True,
                    )

            cache_key = self._provider_cache_key(provider, model)
            forced_supported = cache_key not in self._forced_tool_choice_unsupported
            # "required" tells the provider "must call any tool" — since we only
            # register save_memory, this is equivalent to forcing that specific tool,
            # but is far more portable than the OpenAI object form
            # {"type": "function", "function": {"name": ...}}, which many providers
            # and gateways reject.
            tool_choice: str = "required" if forced_supported else "auto"
            chat_kwargs: dict[str, Any] = {
                "messages": chat_messages,
                "tools": _SAVE_MEMORY_TOOL,
                "model": model,
                "tool_choice": tool_choice,
            }
            if max_output_tokens is not None:
                chat_kwargs["max_tokens"] = max_output_tokens
            response = await provider.chat_with_retry(**chat_kwargs)

            if (
                forced_supported
                and response.finish_reason == "error"
                and _is_tool_choice_unsupported(response.content)
            ):
                logger.warning(
                    "Forced tool_choice unsupported for {}; falling back to auto "
                    "and remembering for future consolidations",
                    cache_key,
                )
                self._forced_tool_choice_unsupported.add(cache_key)
                response = await provider.chat_with_retry(
                    **{**chat_kwargs, "tool_choice": "auto"}
                )

            if not response.has_tool_calls:
                if response.finish_reason == "length":
                    logger.warning(
                        "Memory consolidation: LLM hit the output cap before calling "
                        "save_memory (finish_reason=length, max_output_tokens={}, "
                        "content_len={}, content_preview={}). Consider raising "
                        "max_tokens.output, lowering or disabling model reasoning if "
                        "supported, or reducing the consolidation chunk size.",
                        (
                            max_output_tokens
                            if max_output_tokens is not None
                            else "provider default"
                        ),
                        len(response.content or ""),
                        (response.content or "")[:200],
                    )
                else:
                    logger.warning(
                        "Memory consolidation: LLM did not call save_memory "
                        "(finish_reason={}, content_len={}, content_preview={})",
                        response.finish_reason,
                        len(response.content or ""),
                        (response.content or "")[:200],
                    )
                return await self._handle_failure(
                    messages,
                    source=source,
                    session_key=session_key,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    history_marker_id=history_marker_id,
                    session_manager=session_manager,
                    session_generation=session_generation,
                    base_memory=current_memory,
                    _transaction_held=True,
                )

            args = _normalize_save_memory_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Memory consolidation: unexpected save_memory arguments")
                return await self._handle_failure(
                    messages,
                    source=source,
                    session_key=session_key,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    history_marker_id=history_marker_id,
                    session_manager=session_manager,
                    session_generation=session_generation,
                    base_memory=current_memory,
                    _transaction_held=True,
                )

            if "history_entry" not in args or "memory_update" not in args:
                logger.warning("Memory consolidation: save_memory payload missing required fields")
                return await self._handle_failure(
                    messages,
                    source=source,
                    session_key=session_key,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    history_marker_id=history_marker_id,
                    session_manager=session_manager,
                    session_generation=session_generation,
                    base_memory=current_memory,
                    _transaction_held=True,
                )

            entry = args["history_entry"]
            update = args["memory_update"]

            if entry is None or update is None:
                logger.warning("Memory consolidation: save_memory payload contains null required fields")
                return await self._handle_failure(
                    messages,
                    source=source,
                    session_key=session_key,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    history_marker_id=history_marker_id,
                    session_manager=session_manager,
                    session_generation=session_generation,
                    base_memory=current_memory,
                    _transaction_held=True,
                )

            entry = _ensure_text(entry).strip()
            if not entry:
                logger.warning("Memory consolidation: history_entry is empty after normalization")
                return await self._handle_failure(
                    messages,
                    source=source,
                    session_key=session_key,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    history_marker_id=history_marker_id,
                    session_manager=session_manager,
                    session_generation=session_generation,
                    base_memory=current_memory,
                    _transaction_held=True,
                )

            update = _ensure_text(update)
            receipt = self._make_receipt(
                messages=messages,
                history_entry=entry,
                memory_output=update,
                source=source,
                session_key=session_key,
                start_offset=start_offset,
                end_offset=end_offset,
                history_marker_id=history_marker_id,
                base_memory=current_memory,
                session_generation=session_generation,
            )
            if not await self._commit_receipt_locked(receipt, session_manager):
                return False

            self._consecutive_failures = 0
            logger.info("Memory consolidation done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return await self._handle_failure(
                messages,
                source=source,
                session_key=session_key,
                start_offset=start_offset,
                end_offset=end_offset,
                history_marker_id=history_marker_id,
                session_manager=session_manager,
                session_generation=session_generation,
                base_memory=current_memory,
                _transaction_held=True,
            )

    async def _handle_failure(
        self,
        messages: list[dict],
        *,
        source: str | None = None,
        session_key: str | None = None,
        start_offset: int | None = None,
        end_offset: int | None = None,
        history_marker_id: str | None = None,
        session_manager: SessionManager | None = None,
        session_generation: int | None = None,
        base_memory: str | None = None,
        _transaction_held: bool = False,
    ) -> bool:
        """Retry failures, then make bounded durable progress."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._consecutive_failures = 0
        if session_key:
            return await self._commit_reference(
                messages,
                reason="consolidation failed",
                source=source,
                session_key=session_key,
                start_offset=start_offset,
                end_offset=end_offset,
                history_marker_id=history_marker_id,
                session_manager=session_manager,
                session_generation=session_generation,
                base_memory=base_memory,
                _transaction_held=_transaction_held,
            )
        self._raw_archive(messages)
        return True

    def _fail_or_raw_archive(self, messages: list[dict]) -> bool:
        """Increment failure count; after threshold, raw-archive messages and return True."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict]) -> None:
        """Fallback: append a bounded raw excerpt to HISTORY.md."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        digest = self._digest_messages(messages)
        transcript = self._format_messages(
            messages,
            max_chars=max(0, self._MAX_RAW_ARCHIVE_CHARS - 180),
        )
        entry = (
            f"[{ts}] [RAW] {len(messages)} messages digest={digest[:24]}\n"
            f"{transcript}"
        )
        self.append_history(entry[: max(0, self._MAX_RAW_ARCHIVE_CHARS - 2)])
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5
    _MAX_CONSOLIDATION_MESSAGES = 200

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
        consolidation_trigger_ratio: float = 0.5,
        consolidation_target_ratio: float = 0.3,
    ):
        self.store = MemoryStore(workspace)
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self.consolidation_trigger_ratio = consolidation_trigger_ratio
        self.consolidation_target_ratio = consolidation_target_ratio
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._session_transactions: dict[int, dict[str, Any]] = {}
        self.store.prompt_budget = self._auxiliary_prompt_budget()

    def _auxiliary_prompt_budget(self) -> PromptBudget | None:
        if self.context_window_tokens <= 0:
            return None
        return PromptBudget(
            total_tokens=self.context_window_tokens,
            completion_reserve=max(0, self.max_completion_tokens),
            safety_buffer=max(0, self._SAFETY_BUFFER),
        )

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive a selected message chunk into persistent memory."""
        self.store.prompt_budget = self._auxiliary_prompt_budget()
        transaction = self._session_transactions.pop(id(messages), None)
        kwargs: dict[str, Any] = {
            "max_output_tokens": self.max_completion_tokens,
        }
        if transaction is not None:
            kwargs.update(transaction)
        return await self.store.consolidate(
            messages,
            self.provider,
            self.model,
            **kwargs,
        )

    @staticmethod
    def _turn_end(messages: list[dict[str, object]], start: int) -> int:
        """Return the next user boundary, preserving a complete user-led turn."""
        end = min(max(0, start) + 1, len(messages))
        while end < len(messages) and messages[end].get("role") != "user":
            end += 1
        return end

    def _select_consolidation_chunk(
        self,
        session: Session,
    ) -> tuple[int, int, list[dict[str, object]]] | None:
        """Select the largest fitting prefix of complete turns.

        At most 200 raw JSONL messages are sent to the auxiliary model.  If
        the first complete turn itself exceeds the rendered auxiliary budget,
        it is returned for bounded-reference archival so consolidation still
        makes monotonic progress rather than retrying an impossible request.
        """
        start = max(0, int(session.last_consolidated))
        if start >= len(session.messages):
            return None
        messages = session.messages
        end = start
        best_end: int | None = None
        budget = self._auxiliary_prompt_budget()

        while end < len(messages):
            candidate_end = self._turn_end(messages, end)
            if candidate_end - start > self._MAX_CONSOLIDATION_MESSAGES:
                # A first complete turn that itself exceeds the raw-message
                # cap must make bounded reference progress.  Once a fitting
                # prefix exists, however, do not swallow that prefix into an
                # oversized next turn: return the best complete prefix and
                # let the next round handle the large turn.
                if best_end is not None:
                    break
                return start, candidate_end, messages[start:candidate_end]
            candidate = messages[start:candidate_end]
            if budget is not None:
                measurement = self.store.measure_consolidation_prompt(
                    candidate,
                    self.provider,
                    self.model,
                    budget,
                )
                if not measurement.fits:
                    if best_end is not None:
                        break
                    return start, candidate_end, candidate
            best_end = candidate_end
            end = candidate_end
            if end >= len(messages) or end - start >= self._MAX_CONSOLIDATION_MESSAGES:
                break

        if best_end is None:
            return None
        return start, best_end, messages[start:best_end]

    def _legacy_archive_override(self) -> bool:
        """Detect tests/integrations replacing the archive hook in-place."""
        method = getattr(self, "consolidate_messages")
        return getattr(method, "__func__", None) is not MemoryConsolidator.consolidate_messages

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_model_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive messages with guaranteed persistence (retries until raw-dump fallback)."""
        if not messages:
            return True
        for _ in range(self.store._MAX_FAILURES_BEFORE_RAW_ARCHIVE):
            if await self.consolidate_messages(messages):
                return True
        return True

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Loop: archive old messages until prompt fits within safe budget.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.
        """
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            self.store.recover_pending_receipt(self.sessions)
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            trigger = int(budget * self.consolidation_trigger_ratio)
            target = int(budget * self.consolidation_target_ratio)
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            if estimated < trigger:
                logger.debug(
                    "Token consolidation idle {}: {}/{} (trigger={}) via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    trigger,
                    source,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                if self._legacy_archive_override():
                    boundary = self.pick_consolidation_boundary(
                        session,
                        max(1, estimated - target),
                    )
                    selected = (
                        (session.last_consolidated, boundary[0],
                         session.messages[session.last_consolidated:boundary[0]])
                        if boundary is not None
                        else None
                    )
                else:
                    selected = self._select_consolidation_chunk(session)
                if selected is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                start_idx, end_idx, chunk = selected
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                # ``_select_consolidation_chunk`` returns the first complete
                # turn even when that turn is too large for the rendered
                # auxiliary request.  Mark that case explicitly so the
                # transaction cannot accidentally transport a small (<200
                # message) but irreducibly oversized turn.  The selector has
                # already measured this candidate; repeating the exact
                # rendered measurement here keeps the transaction metadata
                # correct for compatibility/custom selection paths too.
                reference_only = len(chunk) > self._MAX_CONSOLIDATION_MESSAGES
                if not reference_only:
                    auxiliary_budget = self._auxiliary_prompt_budget()
                    if auxiliary_budget is not None:
                        reference_only = not self.store.measure_consolidation_prompt(
                            chunk,
                            self.provider,
                            self.model,
                            auxiliary_budget,
                        ).fits
                self._session_transactions[id(chunk)] = {
                    "session_key": session.key,
                    "start_offset": start_idx,
                    "end_offset": end_idx,
                    "history_marker_id": self.store._digest_messages(chunk)[:24],
                    "session_manager": self.sessions,
                    "session_generation": session.generation,
                    "reference_only": reference_only,
                }
                try:
                    archived = await self.consolidate_messages(chunk)
                finally:
                    self._session_transactions.pop(id(chunk), None)
                if not archived:
                    return
                # The store advances this offset as part of the durable
                # receipt.  Keep the compatibility path for mocked stores
                # and older integrations, but never advance before success.
                if session.last_consolidated < end_idx:
                    session.last_consolidated = end_idx
                    self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return
