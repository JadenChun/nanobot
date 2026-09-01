"""Session management for conversation history."""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.config.paths import get_legacy_sessions_dir
from nanobot.utils.helpers import ensure_dir, safe_filename
from nanobot.utils.prompt_budget import portable_file_lock


class SessionWriteConflict(Exception):  # noqa: N818 - public conflict contract
    """Raised when a stale session attempts a replacement write.

    The exception text is intentionally fixed and content-free: session
    messages may contain secrets and must never be copied into an error.
    """

    DEFAULT_MESSAGE = "Session changed concurrently; reload and retry."

    def __init__(self) -> None:
        super().__init__(self.DEFAULT_MESSAGE)


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Important: Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md/HISTORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to files
    # Additive optimistic-concurrency metadata.  Legacy files which do not
    # carry these fields load as generation/revision zero and are upgraded on
    # their next atomic save; construction itself never rewrites them.
    revision: int = 0
    generation: int = 0
    _replace_messages_on_save: bool = field(default=False, repr=False, compare=False)
    _expected_revision_on_save: int | None = field(default=None, repr=False, compare=False)

    SCHEDULED_RING_KEY = "_recent_scheduled_runs"
    SCHEDULED_RING_LIMIT = 20
    SCHEDULED_EXCERPT_CHARS = 240
    SCHEDULED_PROJECTION_CHARS = 6000

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session.

        Internal turn metadata such as ``_run_id`` is intentionally accepted
        and persisted.  :meth:`get_history` removes it before messages are
        supplied to a model.
        """
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def add_run_message(self, run_id: str, role: str, content: str, **kwargs: Any) -> None:
        """Add a detail message tagged for a specific run."""
        self.add_message(role, content, _run_id=run_id, **kwargs)

    def get_run_messages(self, run_id: str) -> list[dict[str, Any]]:
        """Return persisted detail messages belonging to ``run_id``.

        Unlike :meth:`get_history`, this is a trace-facing API and therefore
        retains the internal tag and any other persisted detail fields.
        """
        matching = [dict(message) for message in self.messages if message.get("_run_id") == run_id]
        if matching:
            return matching
        # Compatibility for scheduled links stored in the bounded metadata
        # ring.  The full detail remains in the cron execution session; this
        # synthetic view keeps older trace readers source-compatible without
        # appending a second instruction/result pair to visible history.
        for item in self.recent_scheduled_runs():
            if item.get("run_id") == run_id:
                return [
                    {"role": "user", "content": item.get("instruction", ""), "_run_id": run_id},
                    {"role": "assistant", "content": item.get("result", ""), "_run_id": run_id},
                ]
        return []

    # Descriptive alias used by trace readers.
    messages_for_run = get_run_messages

    @staticmethod
    def _strip_internal_metadata(value: Any) -> Any:
        """Remove internal keys recursively from model-facing message data."""
        if isinstance(value, dict):
            return {
                key: Session._strip_internal_metadata(item)
                for key, item in value.items()
                if not str(key).startswith("_")
            }
        if isinstance(value, list):
            return [Session._strip_internal_metadata(item) for item in value]
        if isinstance(value, tuple):
            return tuple(Session._strip_internal_metadata(item) for item in value)
        return value

    @staticmethod
    def _find_legal_start(messages: list[dict[str, Any]]) -> int:
        """Find first index where every tool result has a matching assistant tool_call."""
        declared: set[str] = set()
        start = 0
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        declared.add(str(tc["id"]))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid and str(tid) not in declared:
                    start = i + 1
                    declared.clear()
                    for prev in messages[start:i + 1]:
                        if prev.get("role") == "assistant":
                            for tc in prev.get("tool_calls") or []:
                                if isinstance(tc, dict) and tc.get("id"):
                                    declared.add(str(tc["id"]))
        return start

    def _history_messages(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return unconsolidated messages for LLM input, aligned to a legal tool-call boundary."""
        unconsolidated = self.messages[self.last_consolidated:]
        sliced = unconsolidated[-max_messages:] if max_messages > 0 else list(unconsolidated)

        # Drop leading non-user messages to avoid starting mid-turn when possible.
        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[i:]
                break

        # Some providers reject orphan tool results if the matching assistant
        # tool_calls message fell outside the fixed-size history window.
        start = self._find_legal_start(sliced)
        if start:
            sliced = sliced[start:]

        out: list[dict[str, Any]] = []
        for message in sliced:
            entry: dict[str, Any] = {"role": message["role"], "content": message.get("content", "")}
            for key in ("tool_calls", "tool_call_id", "name"):
                if key in message:
                    entry[key] = message[key]
            out.append(self._strip_internal_metadata(entry))
        return out

    @staticmethod
    def _is_scheduled_wrapper(message: dict[str, Any]) -> bool:
        content = message.get("content")
        return (
            message.get("role") == "user"
            and isinstance(content, str)
            and content.startswith(("[Scheduled task from cron:", "[Scheduled task approval from cron:"))
        )

    def recent_scheduled_runs(self) -> list[dict[str, Any]]:
        """Return the bounded scheduled-awareness metadata ring."""
        value = self.metadata.get(self.SCHEDULED_RING_KEY, [])
        if not isinstance(value, list):
            return []
        # Entries written before approval metadata was introduced remain
        # readable; callers and the projection use ``False`` for the missing
        # field without rewriting the legacy metadata.
        return [dict(item) for item in value if isinstance(item, dict)][-self.SCHEDULED_RING_LIMIT:]

    def add_scheduled_run(
        self,
        *,
        job_id: str,
        run_id: str,
        instruction: str,
        result: str,
        detail_session_key: str,
        status: str = "completed",
        approval_granted: bool = False,
        occurrence_id: str | None = None,
        stop_reason: str | None = None,
    ) -> None:
        """Record a bounded scheduled result reference without chat mirroring."""
        def excerpt(value: Any) -> str:
            text = str(value or "").strip()
            if len(text) > self.SCHEDULED_EXCERPT_CHARS:
                suffix = "..."
                limit = max(0, self.SCHEDULED_EXCERPT_CHARS)
                return text[: max(0, limit - len(suffix))].rstrip() + suffix
            return text

        runs = self.recent_scheduled_runs()
        runs = [item for item in runs if item.get("run_id") != str(run_id)]
        canonical_status = getattr(status, "value", str(status))
        runs.append({
            "job_id": str(job_id),
            "run_id": str(run_id),
            "detail_ref": {
                "session_key": str(detail_session_key),
                "run_id": str(run_id),
            },
            "instruction": excerpt(instruction),
            "result": excerpt(result),
            "status": canonical_status,
            "run_status": canonical_status,
            "approval_granted": bool(approval_granted),
            **({"occurrence_id": str(occurrence_id)} if occurrence_id else {}),
            **({"stop_reason": str(stop_reason)} if stop_reason else {}),
            "timestamp": datetime.now().isoformat(),
        })
        self.metadata[self.SCHEDULED_RING_KEY] = runs[-self.SCHEDULED_RING_LIMIT:]
        self.updated_at = datetime.now()

    def _scheduled_projection(self) -> dict[str, Any] | None:
        """Build a bounded model-facing scheduled-awareness message."""
        runs = self.recent_scheduled_runs()
        if not runs:
            return None
        lines = [
            "[Recent Scheduled Runs]",
            "Recent scheduled activity is metadata only; retrieve full detail by run reference.",
        ]
        for item in reversed(runs):
            ref = item.get("detail_ref") or {}
            ref_text = f"{ref.get('session_key', '')}/{ref.get('run_id', item.get('run_id', ''))}"
            lines.append(
                f"- job={item.get('job_id', '')} run={item.get('run_id', '')} "
                f"status={item.get('status', '')} "
                f"approval_granted={str(bool(item.get('approval_granted', False))).lower()} "
                f"ref={ref_text}"
            )
            if item.get("stop_reason"):
                lines.append(f"  stop_reason: {item.get('stop_reason')}")
            lines.append(f"  instruction: {item.get('instruction', '')}")
            lines.append(f"  result: {item.get('result', '')}")
        content = "\n".join(lines)
        if len(content) > self.SCHEDULED_PROJECTION_CHARS:
            suffix = "\n..."
            limit = max(0, self.SCHEDULED_PROJECTION_CHARS)
            content = content[: max(0, limit - len(suffix))].rstrip() + suffix
        return {"role": "user", "content": content}

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return compatibility history, including bounded scheduled link views."""
        out = self._history_messages(max_messages)
        # Keep the old trace-reader behavior bounded by the same 20-entry ring.
        for item in self.recent_scheduled_runs():
            out.extend([
                {"role": "user", "content": item.get("instruction", "")},
                {"role": "assistant", "content": item.get("result", "")},
            ])
        return out

    def get_model_history(self, max_messages: int = 0) -> list[dict[str, Any]]:
        """Return the request projection used by the agent model.

        Legacy visible-chat scheduled pairs are omitted without rewriting the
        underlying JSONL.  New scheduled runs are represented by one bounded
        metadata projection and remain resolvable through their run reference.
        """
        history = self._history_messages(max_messages)
        collapsed: list[dict[str, Any]] = []
        index = 0
        while index < len(history):
            if self._is_scheduled_wrapper(history[index]):
                if index + 1 < len(history) and history[index + 1].get("role") == "assistant":
                    index += 2
                    continue
            collapsed.append(history[index])
            index += 1
        projection = self._scheduled_projection()
        return ([projection] if projection else []) + collapsed

    def clear(self, *, expected_revision: int | None = None) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.last_consolidated = 0
        # A clear is an intentional replacement, not an append-only stale
        # suffix.  SessionManager.save() uses this transient flag to prevent
        # cross-process merge logic from resurrecting the old conversation.
        self._replace_messages_on_save = True
        self._expected_revision_on_save = (
            self.revision if expected_revision is None else int(expected_revision)
        )
        self.generation = max(0, int(self.generation)) + 1
        self.updated_at = datetime.now()

    def retain_recent_legal_suffix(
        self,
        max_messages: int,
        *,
        expected_revision: int | None = None,
    ) -> None:
        """Keep a legal recent suffix, mirroring get_history boundary rules."""
        if max_messages <= 0:
            self.clear(expected_revision=expected_revision)
            return
        if len(self.messages) <= max_messages:
            return

        start_idx = max(0, len(self.messages) - max_messages)

        # If the cutoff lands mid-turn, extend backward to the nearest user turn.
        while start_idx > 0 and self.messages[start_idx].get("role") != "user":
            start_idx -= 1

        retained = self.messages[start_idx:]

        # Mirror get_history(): avoid persisting orphan tool results at the front.
        start = self._find_legal_start(retained)
        if start:
            retained = retained[start:]

        dropped = len(self.messages) - len(retained)
        self.messages = retained
        self.last_consolidated = max(0, self.last_consolidated - dropped)
        self._replace_messages_on_save = True
        self._expected_revision_on_save = (
            self.revision if expected_revision is None else int(expected_revision)
        )
        self.generation = max(0, int(self.generation)) + 1
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self.legacy_sessions_dir = get_legacy_sessions_dir()
        self._cache: dict[str, Session] = {}

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def _get_legacy_session_path(self, key: str) -> Path:
        """Legacy global session path (~/.nanobot/sessions/)."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.legacy_sessions_dir / f"{safe_key}.jsonl"

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)
        if not path.exists():
            legacy_path = self._get_legacy_session_path(key)
            if legacy_path.exists():
                try:
                    shutil.move(str(legacy_path), str(path))
                    logger.info("Migrated session {} from legacy path", key)
                except Exception:
                    logger.exception("Failed to migrate session {}", key)

        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None
            last_consolidated = 0
            revision = 0
            generation = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                        last_consolidated = data.get("last_consolidated", 0)
                        revision = max(0, int(data.get("revision", 0)))
                        generation = max(0, int(data.get("generation", 0)))
                    else:
                        messages.append(data)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated,
                revision=revision,
                generation=generation,
            )
        except Exception as e:
            logger.warning("Failed to load session {}: {}", key, e)
            return None

    def save(self, session: Session, *, expected_revision: int | None = None) -> None:
        """Save a session to disk atomically.

        The temporary file is created beside the target so ``os.replace`` is
        atomic on the filesystem used for the sessions directory.  A failed
        or interrupted write leaves an existing session file untouched.
        """
        path = self._get_session_path(session.key)
        with portable_file_lock(f"{path}.lock"):
            disk = self._read_session_file(path, session.key) if path.exists() else None
            is_replacement = bool(session._replace_messages_on_save)
            expected = expected_revision
            if expected is None and is_replacement:
                expected = session._expected_revision_on_save

            # Replacement operations (/new and heartbeat trimming) are
            # compare-and-swap writes.  Never merge a stale replacement with
            # a newer stream: doing so can resurrect a cleared conversation.
            disk_revision = disk.revision if disk is not None else 0
            if is_replacement and expected is not None and disk_revision != expected:
                raise SessionWriteConflict()

            # Ordinary appends may merge stale same-generation suffixes, but a
            # different generation means another writer performed a
            # replacement.  Reject the append instead of reviving old text.
            if not is_replacement and disk is not None and disk.generation != session.generation:
                raise SessionWriteConflict()

            if disk is not None and not is_replacement:
                common = 0
                while (
                    common < len(disk.messages)
                    and common < len(session.messages)
                    and disk.messages[common] == session.messages[common]
                ):
                    common += 1
                # Keep both writers' suffixes.  Identical messages are valid
                # user turns; deduplicating by value would silently lose them.
                merged = [dict(message) for message in disk.messages]
                merged.extend(dict(message) for message in session.messages[common:])
                session.messages = merged
                session.last_consolidated = max(
                    int(session.last_consolidated), int(disk.last_consolidated)
                )
                merged_metadata = dict(disk.metadata)
                merged_metadata.update(session.metadata)
                # Preserve both processes' scheduled ring entries by run ID.
                ring: list[dict[str, Any]] = []
                for item in [
                    *disk.recent_scheduled_runs(),
                    *session.recent_scheduled_runs(),
                ]:
                    if not any(existing.get("run_id") == item.get("run_id") for existing in ring):
                        ring.append(item)
                if ring:
                    merged_metadata[Session.SCHEDULED_RING_KEY] = ring[-Session.SCHEDULED_RING_LIMIT:]
                session.metadata = merged_metadata
                session.revision = disk.revision
                session.generation = disk.generation
                session.created_at = disk.created_at
                if disk.updated_at > session.updated_at:
                    session.updated_at = disk.updated_at

            next_revision = max(0, int(disk_revision), int(session.revision)) + 1

            self._write_session_file_unlocked(session, path, next_revision)

            session.revision = next_revision
            session._replace_messages_on_save = False
            session._expected_revision_on_save = None

        self._cache[session.key] = session

    @staticmethod
    def _write_session_file_unlocked(
        session: Session,
        path: Path,
        revision: int,
    ) -> None:
        """Atomically write one session while its per-file lock is held."""
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                metadata_line = {
                    "_type": "metadata",
                    "key": session.key,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "metadata": session.metadata,
                    "last_consolidated": session.last_consolidated,
                    "revision": int(revision),
                    "generation": max(0, int(session.generation)),
                }
                f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
                for msg in session.messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _read_session_file(path: Path, key: str) -> Session | None:
        """Read a session file without touching the manager cache."""
        try:
            messages: list[dict[str, Any]] = []
            metadata: dict[str, Any] = {}
            created_at: datetime | None = None
            updated_at: datetime | None = None
            last_consolidated = 0
            revision = 0
            generation = 0
            with path.open(encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = (
                            datetime.fromisoformat(data["created_at"])
                            if data.get("created_at")
                            else None
                        )
                        updated_at = (
                            datetime.fromisoformat(data["updated_at"])
                            if data.get("updated_at")
                            else None
                        )
                        last_consolidated = max(0, int(data.get("last_consolidated", 0)))
                        revision = max(0, int(data.get("revision", 0)))
                        generation = max(0, int(data.get("generation", 0)))
                    elif isinstance(data, dict):
                        messages.append(data)
            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or datetime.now(),
                metadata=metadata if isinstance(metadata, dict) else {},
                last_consolidated=last_consolidated,
                revision=revision,
                generation=generation,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def advance_last_consolidated(
        self,
        session_or_key: Session | str,
        expected_generation: int,
        chunk_digest: str,
        start_offset: int | None = None,
        end_offset: int | None = None,
    ) -> bool:
        """Advance a consolidation offset using a generation-checked CAS.

        The session file is read and rewritten under its lock.  A receipt from
        a previous generation therefore cannot advance a freshly cleared or
        trimmed session.  When offsets are supplied, the exact chunk digest is
        checked as an additional guard against a rewritten same-generation
        stream.
        """
        if isinstance(session_or_key, Session):
            key = session_or_key.key
            target = session_or_key
        else:
            key = str(session_or_key)
            target = self._cache.get(key)

        infer_end = end_offset is None
        if end_offset is not None and end_offset < 0:
            return False
        if start_offset is not None and start_offset < 0:
            return False

        path = self._get_session_path(key)
        with portable_file_lock(f"{path}.lock"):
            disk = self._read_session_file(path, key) if path.exists() else None
            if disk is None:
                # A newly-created session may be consolidated before its
                # first ordinary turn save.  Treat the caller's in-memory
                # generation-zero snapshot as the compare-and-swap base.
                if target is None or target.generation != int(expected_generation):
                    return False
                disk = Session(
                    key=target.key,
                    messages=[dict(message) for message in target.messages],
                    created_at=target.created_at,
                    updated_at=target.updated_at,
                    metadata=dict(target.metadata),
                    last_consolidated=target.last_consolidated,
                    revision=target.revision,
                    generation=target.generation,
                )
            elif disk.generation != int(expected_generation):
                return False
            if infer_end:
                search_start = (
                    start_offset
                    if start_offset is not None
                    else max(0, int(disk.last_consolidated))
                )
                end_offset = None
                for candidate_end in range(search_start + 1, len(disk.messages) + 1):
                    import hashlib

                    payload = json.dumps(
                        disk.messages[search_start:candidate_end],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ).encode("utf-8")
                    if hashlib.sha256(payload).hexdigest() == str(chunk_digest):
                        start_offset = search_start
                        end_offset = candidate_end
                        break
                if end_offset is None:
                    return False
            if end_offset is None or end_offset < 0:
                return False
            if end_offset > len(disk.messages):
                return False
            if start_offset is not None:
                if start_offset > end_offset:
                    return False
                # Import lazily to keep session manager's normal dependency
                # surface small and identical to MemoryStore's digest rule.
                import hashlib

                payload = json.dumps(
                    disk.messages[start_offset:end_offset],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
                if hashlib.sha256(payload).hexdigest() != str(chunk_digest):
                    return False

            if disk.last_consolidated >= end_offset:
                if target is not None:
                    target.messages = [dict(message) for message in disk.messages]
                    target.last_consolidated = disk.last_consolidated
                    target.revision = disk.revision
                    target.generation = disk.generation
                return True

            disk.last_consolidated = int(end_offset)
            disk.updated_at = datetime.now()
            next_revision = max(0, int(disk.revision)) + 1
            self._write_session_file_unlocked(disk, path, next_revision)
            disk.revision = next_revision
            if target is not None:
                target.messages = [dict(message) for message in disk.messages]
                target.last_consolidated = disk.last_consolidated
                target.revision = disk.revision
                target.generation = disk.generation
                target.updated_at = disk.updated_at
            self._cache[key] = target or disk
            return True

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        sessions = []

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            key = data.get("key") or path.stem.replace("_", ":", 1)
                            sessions.append({
                                "key": key,
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": str(path)
                            })
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
