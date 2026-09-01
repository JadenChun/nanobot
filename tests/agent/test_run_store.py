import json
import os
from pathlib import Path

import pytest

from nanobot.agent.run_store import RunStore
from nanobot.agent.turn import (
    RunRecord,
    RunStatus,
    SessionRunRef,
    TurnSource,
)
from nanobot.session.manager import SessionManager


def _record(run_id: str = "run-1") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        source=TurnSource.DIRECT,
        status=RunStatus.RUNNING,
        session_ref=SessionRunRef(session_key="direct:chat", run_id=run_id),
        metadata={
            "channel": "cli",
            "safe": {"count": 1},
            "prompt": "must not be persisted",
            "tool_args": {"path": "must not be persisted"},
            "api_key": "must not be persisted",
        },
    )


def test_run_record_round_trips_sanitized_schema_atomically(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    record = _record()

    store.save(record)
    path = tmp_path / "runs" / "run-1.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["run_id"] == "run-1"
    assert payload["metadata"] == {"channel": "cli", "safe": {"count": 1}}
    assert "prompt" not in json.dumps(payload)
    assert "api_key" not in json.dumps(payload)
    assert "tool_args" not in json.dumps(payload)

    loaded = store.load("run-1")
    assert loaded is not None
    assert loaded.to_dict() == record.to_dict()


def test_interrupted_atomic_write_keeps_previous_record(tmp_path: Path, monkeypatch) -> None:
    store = RunStore(tmp_path)
    store.save(_record())
    path = tmp_path / "runs" / "run-1.json"
    original = path.read_text()

    def interrupted_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        raise KeyboardInterrupt("simulated interruption")

    monkeypatch.setattr("nanobot.agent.run_store.os.replace", interrupted_replace)
    with pytest.raises(KeyboardInterrupt):
        store.save(_record())

    assert path.read_text() == original


def test_update_and_finalize_persist_status(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.save(_record())

    updated = store.update("run-1", metadata={"safe": True})
    assert updated.status is RunStatus.RUNNING
    finalized = store.finalize("run-1", status=RunStatus.COMPLETED, stop_reason="done")

    assert finalized.status is RunStatus.COMPLETED
    assert finalized.completed_at is not None
    assert store.get("run-1") == finalized


def test_load_trace_selects_only_requested_run_messages(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("direct:chat")
    session.messages.extend(
        [
            {"role": "user", "content": "one", "_run_id": "run-1"},
            {"role": "assistant", "content": "two", "_run_id": "run-2"},
            {"role": "tool", "content": "three", "_run_id": "run-1"},
        ]
    )
    manager.save(session)

    store = RunStore(tmp_path, session_manager=manager)
    store.save(_record())

    trace = store.load_trace("run-1")
    assert [message["content"] for message in trace] == ["one", "three"]
    assert all(message.get("_run_id") == "run-1" for message in trace)
