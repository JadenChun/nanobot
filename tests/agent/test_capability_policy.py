from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.policy import DelegatedReadOnlyPolicy, RiskyActionPolicy
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.tools.agent_browser import AgentBrowserTool
from nanobot.agent.tools.agent_device import AgentDeviceTool
from nanobot.agent.tools.filesystem import SearchFilesTool
from nanobot.agent.tools.social_crawl import SocialCrawlTool
from nanobot.agent.turn import RunStatus, ToolOutcome, TurnRequest
from nanobot.context_repo import ResourceAccessPolicy
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock()
    return provider


def test_role_tool_profiles_are_exact_and_worker_has_no_shell(tmp_path) -> None:
    manager = ForegroundAgentManager(provider=_provider(), workspace=tmp_path)

    assert set(manager._tools_for_role("planner").tool_names) == {
        "read_file",
        "list_dir",
        "search_files",
        "web_search",
        "web_fetch",
    }
    assert set(manager._tools_for_role("reviewer").tool_names) == {
        "read_file",
        "list_dir",
        "search_files",
        "web_search",
        "web_fetch",
        "agent_browser",
    }
    assert set(manager._tools_for_role("explorer").tool_names) == {
        "read_file",
        "list_dir",
        "search_files",
        "web_search",
        "web_fetch",
        "agent_browser",
        "agent_device",
    }
    assert set(manager._tools_for_role("crawler").tool_names) == {"social_crawl"}
    assert set(manager._worker_tools("worker", manager._normalize_scopes(tmp_path, ["src/"])).tool_names) == {
        "read_file",
        "list_dir",
        "write_file",
        "edit_file",
    }
    assert all("exec" not in manager._tools_for_role(role).tool_names for role in (
        "planner", "reviewer", "explorer", "crawler"
    ))


@pytest.mark.asyncio
async def test_delegated_read_only_policy_fails_closed_and_is_not_elevatable() -> None:
    policy = DelegatedReadOnlyPolicy(
        allowed_tools={"read_file", "agent_browser", "agent_device", "social_crawl"},
        approval_granted=True,
    )

    safe = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(
            id="safe",
            name="agent_browser",
            arguments={"args": ["snapshot"]},
        )],
    )
    unsafe = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(
            id="unsafe",
            name="agent_browser",
            arguments={"args": ["click", "@e1"]},
        )],
    )
    mixed = await policy.evaluate(
        messages=[],
        tool_calls=[
            ToolCallRequest(id="read", name="read_file", arguments={"path": "README.md"}),
            ToolCallRequest(id="unknown", name="future_tool", arguments={}),
        ],
    )

    assert safe.action == "allow"
    assert unsafe.action == "respond"
    assert unsafe.stop_reason == RunStatus.POLICY_BLOCKED.value
    assert unsafe.requires_approval is False
    assert unsafe.metadata["requires_approval"] is False
    assert "approval" not in (unsafe.response or "").lower()
    assert mixed.stop_reason == RunStatus.POLICY_BLOCKED.value
    assert mixed.requires_approval is False

    forged_writer = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(
            id="writer",
            name="write_file",
            arguments={"path": "out.txt", "content": "x"},
        )],
    )
    assert forged_writer.stop_reason == RunStatus.POLICY_BLOCKED.value


@pytest.mark.asyncio
async def test_delegated_browser_device_screenshot_and_cwd_overrides_are_blocked() -> None:
    policy = DelegatedReadOnlyPolicy(
        allowed_tools={"agent_browser", "agent_device"},
        role="explorer",
    )

    browser_screenshot = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(
            id="browser-shot",
            name="agent_browser",
            arguments={"args": ["screenshot", "/tmp/page.png"]},
        )],
    )
    browser_cwd = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(
            id="browser-cwd",
            name="agent_browser",
            arguments={"args": ["snapshot"], "working_dir": "/tmp/out"},
        )],
    )
    device_screenshot = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(
            id="device-shot",
            name="agent_device",
            arguments={"args": ["screenshot", "screen.png"]},
        )],
    )
    device_cwd = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(
            id="device-cwd",
            name="agent_device",
            arguments={"args": ["snapshot"], "working_dir": "/tmp/out"},
        )],
    )

    for decision in (browser_screenshot, browser_cwd, device_screenshot, device_cwd):
        assert decision.stop_reason == RunStatus.POLICY_BLOCKED.value
        assert decision.requires_approval is False


@pytest.mark.asyncio
async def test_main_policy_requires_approval_for_browser_device_screenshots() -> None:
    policy = RiskyActionPolicy(workspace=Path("."))
    for name in ("agent_browser", "agent_device"):
        for args in (["screenshot", "screen.png"], ["open", "https://example.com", "--output-path", "screen.png"]):
            decision = await policy.evaluate(
                messages=[],
                tool_calls=[ToolCallRequest(
                    id=f"{name}-shot",
                    name=name,
                    arguments={"args": args},
                )],
            )
            assert decision.stop_reason == RunStatus.APPROVAL_REQUIRED.value


@pytest.mark.asyncio
async def test_delegated_unsafe_browser_call_is_blocked_before_tool_execution() -> None:
    provider = _provider()
    provider.chat_with_retry.return_value = LLMResponse(
        content="I will click it.",
        tool_calls=[ToolCallRequest(
            id="click",
            name="agent_browser",
            arguments={"args": ["click", "@e1"]},
        )],
        usage={},
    )
    tool = MagicMock(spec=AgentBrowserTool)
    tool.name = "agent_browser"
    tool.supports_parallel_calls = False
    tool.execute = AsyncMock(return_value="executed")
    from nanobot.agent.runner import AgentRunner, AgentRunSpec
    from nanobot.agent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(tool)
    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[],
        tools=registry,
        model="test-model",
        max_iterations=2,
        tool_policy=DelegatedReadOnlyPolicy(
            allowed_tools={"agent_browser"},
            approval_granted=True,
        ),
    ))

    assert result.stop_reason == RunStatus.POLICY_BLOCKED.value
    tool.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_nested_read_only_stop_reaches_outer_turn_without_approval_or_extra_call(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    provider = _provider()
    provider.chat_with_retry.return_value = LLMResponse(
        content="I will inspect this first.",
        tool_calls=[ToolCallRequest(
            id="plan-1",
            name="plan_task",
            arguments={"objective": "Inspect the repository"},
        )],
        usage={},
    )
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.delegation.runner.run = AsyncMock(return_value=AgentRunResult(
        final_content="Blocked by delegated capability policy.",
        messages=[],
        stop_reason=RunStatus.POLICY_BLOCKED.value,
        policy_metadata={"action_class": "state_changing", "requires_approval": False},
    ))

    result = await loop.execute_turn(TurnRequest(content="Inspect the repository"))

    assert provider.chat_with_retry.await_count == 1
    assert result.status is RunStatus.POLICY_BLOCKED
    assert result.stop_reason == RunStatus.POLICY_BLOCKED.value
    assert result.policy_metadata["requires_approval"] is False
    assert loop._pending_approvals == {}
    assert "approval" not in str(result.content).lower()


def test_main_policy_requires_run_wide_approval_for_device_mutations(tmp_path) -> None:
    policy = RiskyActionPolicy(workspace=tmp_path)
    safe = policy._risky_reason(ToolCallRequest(
        id="safe",
        name="agent_device",
        arguments={"args": ["snapshot"]},
    ))
    unsafe = policy._risky_reason(ToolCallRequest(
        id="unsafe",
        name="agent_device",
        arguments={"args": ["press", "@e1"]},
    ))

    assert safe is None
    assert unsafe is not None


def test_browser_device_are_sequential_and_crawler_schema_has_no_click() -> None:
    assert AgentBrowserTool().supports_parallel_calls is False
    assert AgentDeviceTool().supports_parallel_calls is False
    assert "click" not in SocialCrawlTool.parameters["properties"]["action"]["enum"]
    assert {"follow_link", "expand", "paginate"} <= set(
        SocialCrawlTool.parameters["properties"]["action"]["enum"]
    )


@pytest.mark.asyncio
async def test_social_crawl_defense_in_depth_blocks_click_without_backend_request(monkeypatch) -> None:
    request = AsyncMock()
    monkeypatch.setattr(
        "nanobot.agent.tools.social_crawl.SocialCrawlClient.request_async",
        request,
    )

    result = await SocialCrawlTool().execute(
        "click",
        selector="button.like",
    )

    assert "blocked" in result.lower()
    request.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_only_browser_and_device_guards_do_not_start_processes(monkeypatch) -> None:
    browser_process = AsyncMock()
    device_process = AsyncMock()
    monkeypatch.setattr("nanobot.agent.tools.agent_browser.run_owned_process", browser_process)
    monkeypatch.setattr("nanobot.agent.tools.agent_device.run_owned_process", device_process)

    browser_result = await AgentBrowserTool(read_only=True).execute(args=["click", "@e1"])
    device_result = await AgentDeviceTool(read_only=True).execute(args=["press", "@e1"])

    assert isinstance(browser_result, ToolOutcome)
    assert isinstance(device_result, ToolOutcome)
    assert browser_result.stop_reason == RunStatus.POLICY_BLOCKED.value
    assert device_result.stop_reason == RunStatus.POLICY_BLOCKED.value
    browser_process.assert_not_awaited()
    device_process.assert_not_awaited()


@pytest.mark.asyncio
async def test_structured_search_honors_workspace_ignore_protected_binary_and_limits(tmp_path) -> None:
    (tmp_path / "visible.txt").write_text("needle visible\nother", encoding="utf-8")
    (tmp_path / "also-visible.txt").write_text("needle second", encoding="utf-8")
    ignored = tmp_path / "node_modules" / "ignored.txt"
    ignored.parent.mkdir()
    ignored.write_text("needle ignored", encoding="utf-8")
    (tmp_path / ".env").write_text("needle secret", encoding="utf-8")
    (tmp_path / "binary.dat").write_bytes(b"needle\x00binary")
    policy = ResourceAccessPolicy(workspace=tmp_path, restrict_to_workspace=True)
    tool = SearchFilesTool(
        workspace=tmp_path,
        allowed_dir=tmp_path,
        resource_policy=policy,
    )

    result = await tool.execute(query="needle", path=".", max_results=1)
    traversal = await tool.execute(query="needle", path="../")

    assert "visible.txt" in result
    assert "node_modules" not in result
    assert ".env" not in result
    assert "binary.dat" not in result
    assert "showing first 1" in result.lower() or "truncated" in result.lower()
    assert "outside allowed" in traversal.lower() or "outside" in traversal.lower()
    assert (tmp_path / "visible.txt").read_text(encoding="utf-8") == "needle visible\nother"
