from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import (
    ApprovalGrant,
    HistoryMode,
    RunStatus,
    ToolOutcome,
    TraceMode,
    TurnContext,
    TurnRequest,
    TurnSource,
)
from nanobot.session.manager import Session


def test_turn_request_is_frozen_and_enums_are_typed() -> None:
    request = TurnRequest(
        prompt="hello",
        source=TurnSource.DIRECT,
        history_mode=HistoryMode.FRESH,
        trace_mode=TraceMode.SANITIZED,
    )

    assert request.source.value == "direct"
    assert request.history_mode.value == "fresh"
    assert request.trace_mode.value == "sanitized"
    assert RunStatus.RUNNING.value == "running"
    with pytest.raises(FrozenInstanceError):
        request.prompt = "changed"


def test_turn_context_is_mutable_per_run_state() -> None:
    context = TurnContext(request=TurnRequest(prompt="hello"), run_id="run-1")
    context.iteration = 2
    context.cancelled = True

    assert context.iteration == 2
    assert context.cancelled is True


class _ContextTool(Tool):
    @property
    def name(self) -> str:
        return "context_tool"

    @property
    def description(self) -> str:
        return "test context plumbing"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, context: TurnContext | None = None, **_: Any) -> Any:
        return ToolOutcome(
            content={"run_id": context.run_id if context is not None else ""},
            stop_reason="done",
            policy_metadata={"allowed": True},
        )


@pytest.mark.asyncio
async def test_registry_context_path_returns_tool_outcome_and_legacy_path_stays_bare() -> None:
    registry = ToolRegistry()
    registry.register(_ContextTool())
    context = TurnContext(request=TurnRequest(prompt="hello"), run_id="run-2")

    contextual = await registry.execute("context_tool", {}, context=context)
    legacy = await registry.execute("context_tool", {})

    assert isinstance(contextual, ToolOutcome)
    assert contextual.content == {"run_id": "run-2"}
    assert contextual.stop_reason == "done"
    assert legacy == {"run_id": ""}


def test_session_history_omits_internal_metadata_recursively() -> None:
    session = Session(key="direct:test")
    session.messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello", "_run_id": "run-1"}],
            "_run_id": "run-1",
            "_internal": "secret",
        }
    )
    session.messages.append(
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [{"id": "call-1", "_run_id": "run-1"}],
            "_run_id": "run-1",
        }
    )

    history = session.get_history()

    assert history == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": "ok", "tool_calls": [{"id": "call-1"}]},
    ]
    assert all("_run_id" not in repr(message) for message in history)


def test_approval_grant_is_available_as_a_typed_value() -> None:
    grant = ApprovalGrant(approved=True, grant_id="approval-1")
    assert grant.approved is True
    assert grant.grant_id == "approval-1"
