"""Acceptance checks for the canonical per-turn execution boundary."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.policy import RiskyActionPolicy
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.turn import (
    DelegationBudget,
    DeliveryTarget,
    RunStatus,
    ToolOutcome,
    TurnCallbacks,
    TurnContext,
    TurnRequest,
    TurnSource,
)
from nanobot.agent.write_guard import FileLockRegistry
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ExecToolConfig
from nanobot.providers.base import LLMResponse, ToolCallRequest


class _BarrierProvider:
    def __init__(self) -> None:
        self.started: dict[str, asyncio.Event] = {}
        self.release: dict[str, asyncio.Event] = {}
        self.calls: list[str] = []

    def get_default_model(self) -> str:
        return "test-model"

    async def chat_with_retry(self, *, messages, **kwargs):
        raw_prompt = next(
            (
                str(message.get("content", ""))
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        prompt = next(
            (marker for marker in ("one", "two", "a", "b") if raw_prompt.rstrip().endswith(marker)),
            raw_prompt,
        )
        self.calls.append(prompt)
        started = self.started.setdefault(prompt, asyncio.Event())
        release = self.release.setdefault(prompt, asyncio.Event())
        started.set()
        await release.wait()
        return LLMResponse(content=f"done: {prompt}", tool_calls=[], usage={})


class _StoppingTool(Tool):
    @property
    def name(self) -> str:
        return "stopping_tool"

    @property
    def description(self) -> str:
        return "test tool that requests a terminal turn stop"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **_: object) -> str:
        return "legacy result"

    async def execute_with_context(self, context, **_: object) -> ToolOutcome:
        return ToolOutcome(
            content="stop now",
            stop_reason=RunStatus.POLICY_BLOCKED.value,
            policy_metadata={"reason": "test stop"},
        )


class _ToolErrorTool(_StoppingTool):
    @property
    def name(self) -> str:
        return "tool_error"

    async def execute_with_context(self, context, **_: object) -> ToolOutcome:
        return ToolOutcome(
            content="tool failed",
            stop_reason=RunStatus.TOOL_ERROR.value,
            policy_metadata={"reason": "test error"},
        )


@pytest.mark.asyncio
async def test_execute_turn_serializes_same_session(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = _BarrierProvider()
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    provider.started["one"] = asyncio.Event()
    provider.started["two"] = asyncio.Event()
    provider.release["one"] = asyncio.Event()
    provider.release["two"] = asyncio.Event()

    first = asyncio.create_task(loop.execute_turn(TurnRequest(
        content="one",
        source=TurnSource.DIRECT,
        session_key="cli:same",
        route=DeliveryTarget(channel="cli", chat_id="same"),
    )))
    await provider.started["one"].wait()

    second = asyncio.create_task(loop.execute_turn(TurnRequest(
        content="two",
        source=TurnSource.DIRECT,
        session_key="cli:same",
        route=DeliveryTarget(channel="cli", chat_id="same"),
    )))
    await asyncio.sleep(0)
    assert provider.calls == ["one"]

    provider.release["one"].set()
    await provider.started["two"].wait()
    provider.release["two"].set()
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.content == "done: one"
    assert second_result.content == "done: two"
    assert provider.calls == ["one", "two"]
    assert first_result.status is RunStatus.COMPLETED
    assert second_result.status is RunStatus.COMPLETED
    assert first_result.record is not None
    assert all(
        message.get("_run_id") == first_result.run_id
        for message in loop.run_store.load_trace(first_result.run_id)
    )


@pytest.mark.asyncio
async def test_execute_turn_allows_different_sessions_to_overlap(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = _BarrierProvider()
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    provider.started["a"] = asyncio.Event()
    provider.started["b"] = asyncio.Event()
    provider.release["a"] = asyncio.Event()
    provider.release["b"] = asyncio.Event()

    first = asyncio.create_task(loop.execute_turn(TurnRequest(
        content="a",
        session_key="cli:a",
        route=DeliveryTarget(channel="cli", chat_id="a"),
    )))
    second = asyncio.create_task(loop.execute_turn(TurnRequest(
        content="b",
        session_key="cli:b",
        route=DeliveryTarget(channel="cli", chat_id="b"),
    )))
    await asyncio.gather(provider.started["a"].wait(), provider.started["b"].wait())

    provider.release["a"].set()
    provider.release["b"].set()
    await asyncio.gather(first, second)
    assert set(provider.calls) == {"a", "b"}


@pytest.mark.asyncio
async def test_system_compat_adapter_warns_and_preserves_historical_route(tmp_path, monkeypatch) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="system result", tool_calls=[], usage={})
    )
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    warning = MagicMock()
    monkeypatch.setattr("nanobot.agent.loop.logger.warning", warning)

    with pytest.warns(DeprecationWarning, match="system message"):
        result = await loop._process_message(
            InboundMessage(
                channel="system",
                sender_id="scheduler",
                chat_id="telegram:room:thread",
                content="run the task",
            )
        )

    assert result is not None
    assert result.channel == "telegram"
    assert result.chat_id == "room:thread"
    records = list((tmp_path / "runs").glob("*.json"))
    assert len(records) == 1
    record = loop.run_store.load(records[0].stem)
    assert record is not None
    assert record.source.value == "system_compat"
    warning.assert_any_call(
        "Deprecated system message compatibility adapter used",
        event="system_compat",
        channel="telegram",
        chat_id="room:thread",
        session_key="telegram:room:thread",
    )


@pytest.mark.asyncio
async def test_contextual_tool_stop_does_not_make_an_extra_model_call(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(
            id="stop-1",
            name="stopping_tool",
            arguments={},
        )],
        usage={},
    ))
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.tools.register(_StoppingTool())

    result = await loop.execute_turn(TurnRequest(
        content="stop after tool",
        session_key="cli:stop",
        tool_names=("stopping_tool",),
    ))

    assert provider.chat_with_retry.await_count == 1
    assert result.status is RunStatus.POLICY_BLOCKED
    assert result.stop_reason == RunStatus.POLICY_BLOCKED.value
    assert result.content == "stop now"


@pytest.mark.asyncio
async def test_nested_policy_stop_skips_later_write_in_same_outer_batch(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[
            ToolCallRequest(
                id="plan-1",
                name="plan_task",
                arguments={"objective": "Inspect before writing"},
            ),
            ToolCallRequest(
                id="write-1",
                name="write_file",
                arguments={"path": "should-not-exist.txt", "content": "must not write"},
            ),
        ],
        usage={},
    ))
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.delegation.run_plan = AsyncMock(return_value=ToolOutcome(
        content="nested policy blocked",
        stop_reason=RunStatus.POLICY_BLOCKED.value,
        policy_metadata={"requires_approval": False, "nested": True},
    ))

    result = await loop.execute_turn(TurnRequest(
        content="inspect then write",
        session_key="cli:nested-stop",
        tool_names=("plan_task", "write_file"),
    ))

    assert provider.chat_with_retry.await_count == 1
    assert result.status is RunStatus.POLICY_BLOCKED
    assert result.stop_reason == RunStatus.POLICY_BLOCKED.value
    assert result.content == "nested policy blocked"
    assert result.policy_metadata == {"requires_approval": False, "nested": True}
    assert not (tmp_path / "should-not-exist.txt").exists()
    assert result.messages is not None
    skipped = [
        message for message in result.messages
        if message.get("role") == "tool" and message.get("tool_call_id") == "write-1"
    ]
    assert len(skipped) == 1
    assert "skipped" in str(skipped[0]["content"]).lower()
    assert loop._pending_approvals == {}


@pytest.mark.asyncio
async def test_contextual_tool_error_is_terminal_without_an_extra_model_call(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="error-1", name="tool_error", arguments={})],
        usage={},
    ))
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.tools.register(_ToolErrorTool())

    result = await loop.execute_turn(TurnRequest(
        content="stop on error",
        session_key="cli:error",
        tool_names=("tool_error",),
    ))

    assert provider.chat_with_retry.await_count == 1
    assert result.status is RunStatus.TOOL_ERROR
    assert result.stop_reason == RunStatus.TOOL_ERROR.value
    assert result.content == "tool failed"


@pytest.mark.asyncio
async def test_execute_turn_finalizes_unexpected_error_and_notifies_callback(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop._execute_turn_locked = AsyncMock(side_effect=RuntimeError("boom"))
    on_error = AsyncMock()

    result = await loop.execute_turn(TurnRequest(
        content="fail",
        session_key="cli:fail",
        callbacks=TurnCallbacks(on_error=on_error),
    ))

    assert result.status is RunStatus.ERROR
    assert result.error == "Error: RuntimeError: boom"
    on_error.assert_awaited_once_with(result)
    stored = loop.run_store.load(result.run_id or "")
    assert stored is not None
    assert stored.status is RunStatus.ERROR


@pytest.mark.asyncio
async def test_contextual_message_delivery_does_not_cross_route(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.tools.message import MessageTool

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.estimate_prompt_tokens.return_value = (1, "test")
    provider.chat_with_retry = AsyncMock()
    provider.chat_with_retry.return_value = LLMResponse(content="ok", tool_calls=[], usage={})
    sent = []
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    message_tool = loop.tools.get("message")
    assert isinstance(message_tool, MessageTool)
    message_tool.set_send_callback(AsyncMock(side_effect=sent.append))

    first, second = await asyncio.gather(
        loop.execute_turn(TurnRequest(
            content="A",
            session_key="telegram:A",
            route=DeliveryTarget(channel="telegram", chat_id="A"),
        )),
        loop.execute_turn(TurnRequest(
            content="B",
            session_key="telegram:B",
            route=DeliveryTarget(channel="telegram", chat_id="B"),
        )),
    )

    assert first.record is not None and first.record.session_ref.session_key == "telegram:A"
    assert second.record is not None and second.record.session_ref.session_key == "telegram:B"


@pytest.mark.asyncio
async def test_message_context_uses_route_local_delivery_ledger() -> None:
    sent = []
    tool = MessageTool(send_callback=AsyncMock(side_effect=sent.append))
    first = TurnContext(
        request=TurnRequest(
            content="A",
            session_key="telegram:A",
            route=DeliveryTarget(channel="telegram", chat_id="A"),
        ),
        run_id="run-a",
    )
    second = TurnContext(
        request=TurnRequest(
            content="B",
            session_key="telegram:B",
            route=DeliveryTarget(channel="telegram", chat_id="B"),
        ),
        run_id="run-b",
    )

    await tool.execute_with_context(first, content="for A")
    await tool.execute_with_context(second, content="for B")

    assert [message.chat_id for message in sent] == ["A", "B"]
    assert [message.chat_id for message in first.delivery.sent_messages] == ["A"]
    assert [message.chat_id for message in second.delivery.sent_messages] == ["B"]
    assert tool.sent_messages_in_turn == ()


@pytest.mark.asyncio
async def test_auxiliary_message_does_not_mark_primary_delivery_complete() -> None:
    tool = MessageTool(send_callback=AsyncMock())
    context = TurnContext(
        request=TurnRequest(
            content="send elsewhere",
            session_key="telegram:primary",
            route=DeliveryTarget(channel="telegram", chat_id="primary"),
        ),
        run_id="run-aux",
    )

    await tool.execute_with_context(
        context,
        content="auxiliary",
        channel="discord",
        chat_id="other",
    )

    assert context.delivery.sent_messages[0].channel == "discord"
    assert context.delivery.delivered is False


@pytest.mark.asyncio
async def test_approval_alias_waits_for_detail_lock_and_preserves_refs(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="continued", tool_calls=[], usage={})
    )
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    await loop._store_pending_approval(
        session_key="cron:job",
        channel="telegram",
        chat_id="room",
        summary="remove files",
    )
    detail_lock = loop._session_locks.setdefault("cron:job", asyncio.Lock())
    await detail_lock.acquire()
    continuation = asyncio.create_task(loop.execute_turn(TurnRequest(
        content="yes",
        session_key="telegram:room",
        route=DeliveryTarget(channel="telegram", chat_id="room"),
    )))
    await asyncio.sleep(0)
    assert provider.chat_with_retry.await_count == 0

    detail_lock.release()
    result = await continuation

    assert provider.chat_with_retry.await_count == 1
    assert result.record is not None
    assert result.record.detail_ref is not None
    assert result.record.detail_ref.session_key == "cron:job"
    assert result.record.visible_session_key == "telegram:room"
    visible = loop.sessions.get_or_create("telegram:room")
    assert not any(
        message.get("role") == "assistant" and message.get("content") == "continued"
        for message in visible.messages
    )
    runs = visible.recent_scheduled_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == result.run_id
    assert runs[0]["detail_ref"] == {
        "session_key": "cron:job",
        "run_id": result.run_id,
    }
    assert runs[0]["result"] == "continued"


@pytest.mark.asyncio
async def test_concurrent_approval_aliases_grant_exactly_one_run(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="continued", tool_calls=[], usage={})
    )
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    await loop._store_pending_approval(
        session_key="cron:job",
        channel="telegram",
        chat_id="room",
        summary="remove files",
    )
    detail_lock = loop._session_locks.setdefault("cron:job", asyncio.Lock())
    await detail_lock.acquire()
    first = asyncio.create_task(loop.execute_turn(TurnRequest(
        content="yes",
        session_key="telegram:room",
        route=DeliveryTarget(channel="telegram", chat_id="room"),
    )))
    second = asyncio.create_task(loop.execute_turn(TurnRequest(
        content="yes",
        session_key="telegram:room",
        route=DeliveryTarget(channel="telegram", chat_id="room"),
    )))
    await asyncio.sleep(0)
    detail_lock.release()
    results = await asyncio.gather(first, second)

    assert provider.chat_with_retry.await_count == 1
    assert sorted(result.status.value for result in results) == ["cancelled", "completed"]
    assert any(result.stop_reason == "approval_stale" for result in results)
    assert loop._pending_approvals == {}


@pytest.mark.asyncio
async def test_stale_approval_cas_keeps_replacement_pending(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="continued", tool_calls=[], usage={})
    )
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    old = await loop._store_pending_approval(
        session_key="cron:old",
        channel="telegram",
        chat_id="room",
        summary="old action",
    )
    detail_lock = loop._session_locks.setdefault("cron:old", asyncio.Lock())
    await detail_lock.acquire()
    continuation = asyncio.create_task(loop.execute_turn(TurnRequest(
        content="yes",
        session_key="telegram:room",
        route=DeliveryTarget(channel="telegram", chat_id="room"),
    )))
    await asyncio.sleep(0)
    replacement = await loop._store_pending_approval(
        session_key="cron:new",
        channel="telegram",
        chat_id="room",
        summary="new action",
    )
    detail_lock.release()
    result = await continuation

    assert provider.chat_with_retry.await_count == 0
    assert result.status is RunStatus.CANCELLED
    assert result.stop_reason == "approval_stale"
    assert loop._pending_approvals["telegram:room"] is replacement
    assert loop._pending_approvals["cron:old"] is old


@pytest.mark.asyncio
async def test_cron_context_uses_route_and_rejects_nested_scheduling(tmp_path) -> None:
    service = MagicMock()
    service.add_job.return_value = SimpleNamespace(name="job", id="job-1")
    tool = CronTool(service)
    context = TurnContext(
        request=TurnRequest(
            content="schedule",
            source=TurnSource.DIRECT,
            session_key="telegram:A",
            route=DeliveryTarget(channel="telegram", chat_id="A"),
        ),
        run_id="run-a",
    )

    result = await tool.execute_with_context(
        context,
        action="add",
        message="remind me",
        every_seconds=60,
        additional_destinations=[{"channel": "discord", "to": "B"}],
    )
    assert "Created job" in result.content
    assert service.add_job.call_args.kwargs["channel"] == "telegram"
    assert service.add_job.call_args.kwargs["to"] == "A"
    assert service.add_job.call_args.kwargs["additional_destinations"][0].channel == "discord"
    assert service.add_job.call_args.kwargs["additional_destinations"][0].to == "B"

    cron_context = TurnContext(
        request=TurnRequest(
            content="nested",
            source=TurnSource.CRON,
            session_key="cron:job-1",
            route=DeliveryTarget(channel="telegram", chat_id="A"),
        ),
        run_id="run-cron",
    )
    nested = await tool.execute_with_context(
        cron_context,
        action="add",
        message="not allowed",
        every_seconds=60,
    )
    assert "cannot schedule" in nested.content
    assert service.add_job.call_count == 1


@pytest.mark.asyncio
async def test_filesystem_context_uses_lock_owner(tmp_path) -> None:
    target = tmp_path / "draft.md"
    target.write_text("hello world", encoding="utf-8")
    locks = FileLockRegistry()
    await locks.acquire(target, "owner-a")
    tool = EditFileTool(workspace=tmp_path, lock_registry=locks)
    context = TurnContext(
        request=TurnRequest(content="edit", session_key="cli:B"),
        run_id="run-b",
        lock_owner="owner-b",
    )
    try:
        result = await tool.execute_with_context(
            context,
            path="draft.md",
            old_text="world",
            new_text="earth",
        )
    finally:
        await locks.release(target, "owner-a")
    assert "owner-a" in result
    assert target.read_text(encoding="utf-8") == "hello world"


def test_worker_capability_boundary_and_context_local_budget(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    manager = ForegroundAgentManager(
        provider=provider,
        workspace=tmp_path,
        exec_config=ExecToolConfig(enable=True),
    )
    scopes = manager._normalize_scopes(tmp_path, ["src/"])
    worker_tools = manager._worker_tools("call-1", scopes)
    assert set(worker_tools.tool_names) == {"read_file", "list_dir", "write_file", "edit_file"}

    first = TurnContext(
        request=TurnRequest(content="one", session_key="cli:one"),
        run_id="one",
        delegation_budget=DelegationBudget(max_calls=1, max_worker_corrections=1),
    )
    second = TurnContext(
        request=TurnRequest(content="two", session_key="cli:two"),
        run_id="two",
        delegation_budget=DelegationBudget(max_calls=1, max_worker_corrections=1),
    )
    assert manager._consume_call("planner", first) is None
    assert "limit reached" in (manager._consume_call("reviewer", first) or "")
    assert manager._consume_call("reviewer", second) is None


@pytest.mark.asyncio
async def test_browser_policy_allows_navigation_and_requires_approval_for_mutation(tmp_path) -> None:
    from nanobot.providers.base import ToolCallRequest

    policy = RiskyActionPolicy(workspace=tmp_path)
    read = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(id="read", name="agent_browser", arguments={"args": ["open", "https://example.com"]})],
    )
    mutate = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(id="mutate", name="agent_browser", arguments={"args": ["click", "@e1"]})],
    )
    unknown = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(id="unknown", name="agent_browser", arguments={"args": ["future-command"]})],
    )
    assert read.action == "allow"
    assert mutate.stop_reason == "approval_required"
    assert unknown.stop_reason == "approval_required"
