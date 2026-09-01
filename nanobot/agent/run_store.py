"""Atomic, sanitized persistence for per-turn run records."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.turn import RunRecord, RunStatus
from nanobot.session.manager import SessionManager
from nanobot.utils.helpers import ensure_dir


class RunStore:
    """Store one small JSON record per run under ``workspace/runs``.

    Session JSONL files remain the source of detailed trace data.  A run record
    only keeps a reference to that session and bounded status metadata.
    """

    def __init__(self, workspace: Path, session_manager: SessionManager | None = None) -> None:
        self.workspace = Path(workspace)
        self.runs_dir = ensure_dir(self.workspace / "runs")
        self.session_manager = session_manager

    def path_for(self, run_id: str) -> Path:
        """Return the canonical path for a run ID and reject path traversal."""
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be a non-empty filename-safe value")
        return self.runs_dir / f"{run_id}.json"

    # Compatibility alias for callers that call this a record path.
    record_path = path_for

    @staticmethod
    def _coerce_record(record: RunRecord | Mapping[str, Any]) -> RunRecord:
        if isinstance(record, RunRecord):
            return record
        return RunRecord.from_dict(record)

    def save(self, record: RunRecord | Mapping[str, Any]) -> RunRecord:
        """Atomically persist a sanitized run record and return it."""
        record = self._coerce_record(record)
        path = self.path_for(record.run_id)
        payload = record.to_dict()

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            # Replacing a file is atomic when source and destination share a
            # directory.  In particular, an interrupted temp write cannot
            # truncate or otherwise replace a previously valid record.
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
        return record

    create = save

    def load(self, run_id: str) -> RunRecord | None:
        """Load one record, returning ``None`` when it does not exist/parse."""
        path = self.path_for(run_id)
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            return RunRecord.from_dict(payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load run record {}: {}", run_id, exc)
            return None

    get = load

    def update(self, run_id: str, **changes: Any) -> RunRecord:
        """Apply schema fields to an existing record and persist atomically."""
        record = self.load(run_id)
        if record is None:
            raise KeyError(f"Unknown run: {run_id}")

        allowed = {
            "status",
            "source",
            "session_ref",
            "started_at",
            "completed_at",
            "stop_reason",
            "error",
            "metadata",
            "delivery_target",
            "scheduled_link",
            "session_key",
        }
        for key, value in changes.items():
            if key not in allowed:
                continue
            setattr(record, key, value)
        record.updated_at = datetime.now(timezone.utc).isoformat()
        # Re-run normalization after mutable field updates, including metadata
        # sanitization and conversion of enum/string compatibility values.
        record.__post_init__()
        return self.save(record)

    def finalize(
        self,
        run_id: str,
        status: RunStatus = RunStatus.COMPLETED,
        *,
        completed_at: str | datetime | None = None,
        **changes: Any,
    ) -> RunRecord:
        """Mark a run terminal and persist its final sanitized state."""
        changes["status"] = status
        changes["completed_at"] = completed_at or datetime.now(timezone.utc).isoformat()
        return self.update(run_id, **changes)

    def load_trace(self, run_id: str) -> list[dict[str, Any]]:
        """Resolve a record's detail session and return only matching messages."""
        record = self.load(run_id)
        if record is None or record.session_ref is None:
            return []
        manager = self.session_manager or SessionManager(self.workspace)
        session = manager.get_or_create(record.session_ref.session_key)
        return session.get_run_messages(run_id)
