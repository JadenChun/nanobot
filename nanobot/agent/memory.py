"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import json
import weakref
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

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

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self._consecutive_failures = 0
        # Cache of (provider class, model) pairs known to reject forced tool_choice,
        # so we skip the forced attempt and go straight to "auto" on later calls.
        self._forced_tool_choice_unsupported: set[tuple[str, str]] = set()

    @staticmethod
    def _provider_cache_key(provider: LLMProvider, model: str) -> tuple[str, str]:
        return (type(provider).__name__, model)

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str, source: str | None = None) -> None:
        """Append an entry to HISTORY.md.
        
        Args:
            entry: History entry (may include timestamp prefix)
            source: Optional source tag (e.g., "cron", "chat") for filtering
        """
        # Add source tag if provided and entry doesn't already have one
        if source and "[source=" not in entry:
            # Try to insert source tag after timestamp if present
            # Format: [YYYY-MM-DD HH:MM] content -> [YYYY-MM-DD HH:MM] [source=X] content
            if entry.startswith("[") and "] " in entry:
                bracket_end = entry.index("] ") + 2
                entry = f"{entry[:bracket_end]}[source={source}] {entry[bracket_end:]}"
            else:
                entry = f"[source={source}] {entry}"
        
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

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

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
        max_output_tokens: int | None = None,
        source: str | None = None,
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

        current_memory = self.read_long_term()
        prompt = f"""Summarize this conversation for continuity. Preserve:
- Key facts, decisions, and user preferences
- Files examined or modified (names and paths, not full contents)
- Errors encountered and how they were resolved
- Current task state and next steps
- Any learnings useful for future conversations

Call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{self._format_messages(messages)}"""

        chat_messages = [
            {"role": "system", "content": "You are a memory consolidation agent. Summarize conversations into concise, actionable memory. Focus on preserving facts and decisions, not verbatim dialogue. Call the save_memory tool with your consolidation."},
            {"role": "user", "content": prompt},
        ]

        try:
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
                return self._fail_or_raw_archive(messages)

            args = _normalize_save_memory_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Memory consolidation: unexpected save_memory arguments")
                return self._fail_or_raw_archive(messages)

            if "history_entry" not in args or "memory_update" not in args:
                logger.warning("Memory consolidation: save_memory payload missing required fields")
                return self._fail_or_raw_archive(messages)

            entry = args["history_entry"]
            update = args["memory_update"]

            if entry is None or update is None:
                logger.warning("Memory consolidation: save_memory payload contains null required fields")
                return self._fail_or_raw_archive(messages)

            entry = _ensure_text(entry).strip()
            if not entry:
                logger.warning("Memory consolidation: history_entry is empty after normalization")
                return self._fail_or_raw_archive(messages)

            self.append_history(entry, source=source)
            update = _ensure_text(update)
            if update != current_memory:
                self.write_long_term(update)

            self._consecutive_failures = 0
            logger.info("Memory consolidation done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return self._fail_or_raw_archive(messages)

    def _fail_or_raw_archive(self, messages: list[dict]) -> bool:
        """Increment failure count; after threshold, raw-archive messages and return True."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to HISTORY.md without LLM summarization."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.append_history(
            f"[{ts}] [RAW] {len(messages)} messages\n"
            f"{self._format_messages(messages)}"
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5

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

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive a selected message chunk into persistent memory."""
        return await self.store.consolidate(
            messages,
            self.provider,
            self.model,
            max_output_tokens=self.max_completion_tokens,
        )

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
        history = session.get_history(max_messages=0)
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

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated:end_idx]
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
                if not await self.consolidate_messages(chunk):
                    return
                session.last_consolidated = end_idx
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return
