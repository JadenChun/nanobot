from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.tools.delegation import DelegateTaskTool, PlanTaskTool, ReviewWorkTool
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="A normal human response.",
        tool_calls=[],
        usage={},
    ))
    return provider


@pytest.mark.asyncio
async def test_simple_request_uses_one_main_agent_call_without_delegation(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = _provider()
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)

    response = await loop.process_direct("What products do we cover?")

    assert response is not None
    assert response.content == "A normal human response."
    assert provider.chat_with_retry.await_count == 1


@pytest.mark.asyncio
async def test_cron_turn_waits_for_context_repo_sync(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    loop = AgentLoop(bus=MessageBus(), provider=_provider(), workspace=tmp_path)
    loop._sync_context_repos = AsyncMock(return_value=True)

    response = await loop.process_direct(
        "Generate daily research memory",
        session_key="cron:daily-trends",
        channel="telegram",
        chat_id="123",
    )

    assert response is not None
    loop._sync_context_repos.assert_awaited_once()


@pytest.mark.asyncio
async def test_injected_memory_does_not_duplicate_or_orphan_history(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = _provider()
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.context.memory.write_long_term("remember this")
    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "reading",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "read_file", "content": "ok"},
    ]
    loop.sessions.save(session)

    await loop.process_direct("hello", session_key="cli:test")

    saved = loop.sessions.get_or_create("cli:test").messages
    assert [message["role"] for message in saved[-2:]] == ["user", "assistant"]
    assert saved.count(session.messages[2]) == 1


def test_main_loop_exposes_optional_foreground_roles_without_background_tools(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    loop = AgentLoop(bus=MessageBus(), provider=_provider(), workspace=tmp_path)

    assert loop.tools.get("plan_task") is not None
    assert loop.tools.get("delegate_task") is not None
    assert loop.tools.get("review_work") is not None
    assert loop.tools.get("explore") is not None
    assert loop.tools.get("spawn") is None
    assert loop.tools.get("spawn_pipeline") is None


@pytest.mark.asyncio
async def test_foreground_tools_wait_for_manager_results() -> None:
    manager = MagicMock(spec=ForegroundAgentManager)
    manager.run_plan = AsyncMock(return_value="plan")
    manager.run_worker = AsyncMock(return_value="worker result")
    manager.run_review = AsyncMock(return_value="PASS")

    assert await PlanTaskTool(manager).execute("Plan this") == "plan"
    assert await DelegateTaskTool(manager).execute("# Contract\nDo the work", ["report/"]) == "worker result"
    assert await ReviewWorkTool(manager).execute("Goal", "It works") == "PASS"


@pytest.mark.asyncio
async def test_orchestrator_can_call_planner_then_continue_to_final_answer(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = _provider()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="I will inspect the sequence.",
            tool_calls=[ToolCallRequest(
                id="plan-1",
                name="plan_task",
                arguments={"objective": "Prepare the weekly report", "context": "Products: dry food, freeze-dried, tofu litter"},
            )],
            usage={},
        ),
        LLMResponse(content="The report is ready. Please check it for details.", tool_calls=[], usage={}),
    ])
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.delegation.run_plan = AsyncMock(return_value="1. Research\n2. Write\n3. Check sources")

    response = await loop.process_direct("Prepare the weekly report")

    assert response is not None
    assert response.content == "The report is ready. Please check it for details."
    loop.delegation.run_plan.assert_awaited_once()
    assert provider.chat_with_retry.await_count == 2


@pytest.mark.asyncio
async def test_main_task_does_not_run_automatic_verification(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    loop = AgentLoop(bus=MessageBus(), provider=_provider(), workspace=tmp_path)
    loop._run_agent = AsyncMock(return_value=AgentRunResult(
        final_content="Done",
        messages=[{"role": "assistant", "content": "Done"}],
        tools_used=["edit_file"],
    ))

    result = await loop._run_main_task(
        [{"role": "user", "content": "update it"}],
        channel="cli",
        chat_id="direct",
        message_id=None,
    )

    assert result.final_content == "Done"
    loop._run_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_is_foreground_scoped_and_sequential(tmp_path) -> None:
    manager = ForegroundAgentManager(provider=_provider(), workspace=tmp_path)
    manager.runner.run = AsyncMock(return_value=AgentRunResult(
        final_content="complete",
        messages=[],
    ))

    result = await manager.run_worker(
        contract="# Contract\nUpdate the report template and test it.",
        write_scope=["reports/"],
    )

    assert result == "complete"
    spec = manager.runner.run.await_args.args[0]
    assert spec.concurrent_tools is False
    assert spec.tools.get("write_file") is not None
    assert spec.tools.get("message") is None
    assert spec.tools.get("delegate_task") is None


def test_delegated_read_only_roles_have_no_shell_tool(tmp_path) -> None:
    manager = ForegroundAgentManager(provider=_provider(), workspace=tmp_path)

    assert manager._read_only_tools().get("exec") is None
    assert all(
        manager._tools_for_role(role).get("exec") is None
        for role in ("planner", "reviewer", "explorer", "crawler")
    )


def test_delegation_budget_resets_each_main_turn(tmp_path) -> None:
    manager = ForegroundAgentManager(
        provider=_provider(),
        workspace=tmp_path,
        max_calls_per_turn=1,
    )

    assert manager._consume_call("planner") is None
    assert "limit reached" in manager._consume_call("reviewer")
    manager.start_turn()
    assert manager._consume_call("reviewer") is None


@pytest.mark.asyncio
async def test_worker_rejects_write_scope_outside_workspace(tmp_path) -> None:
    manager = ForegroundAgentManager(provider=_provider(), workspace=tmp_path)

    result = await manager.run_worker(
        contract="# Contract\nChange an external file.",
        write_scope=["../outside.txt"],
    )

    assert result == "Error: write_scope entries must stay inside the workspace"


def test_orchestrator_prompt_hides_internal_state_and_explains_optional_routing(tmp_path) -> None:
    from nanobot.agent.context import ContextBuilder

    prompt = ContextBuilder(tmp_path).build_system_prompt(
        tool_names={"plan_task", "delegate_task", "review_work", "explore"}
    )

    assert "main orchestrator" in prompt
    assert "small cohesive tasks directly" in prompt
    assert "Never expose role names" in prompt
    assert "complete usable result" in prompt
    assert "direction to check another report" in prompt
    assert "concise Markdown" in prompt
    assert "improvement-loop memory" in prompt
    assert "raw research trails" in prompt
    assert "fixed planner" not in prompt.lower()
