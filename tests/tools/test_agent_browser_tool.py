from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nanobot.agent.tools.agent_browser import AgentBrowserTool, classify_browser_action
from nanobot.agent.turn import RunStatus, ToolOutcome


class _FakeProcess:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self):
        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_agent_browser_requires_args() -> None:
    tool = AgentBrowserTool()
    result = await tool.execute(args=[])
    assert result == "Error: args must include at least one CLI argument"


@pytest.mark.asyncio
async def test_agent_browser_reports_missing_npx() -> None:
    tool = AgentBrowserTool()
    with patch("nanobot.agent.tools.agent_browser._resolve_npx", return_value=None):
        result = await tool.execute(args=["--version"])

    assert "npx" in result
    assert "Install Node.js/npm first" in result


@pytest.mark.asyncio
async def test_agent_browser_returns_json_payload_for_success() -> None:
    tool = AgentBrowserTool(working_dir="/tmp/work")
    fake_process = _FakeProcess(stdout=b"agent-browser 0.25.5\n", stderr=b"", returncode=0)
    npx_path = r"C:\Program Files\nodejs\npx.cmd"

    with patch(
        "nanobot.agent.tools.agent_browser._resolve_npx",
        return_value=npx_path,
    ), patch(
        "nanobot.agent.tools.agent_browser.asyncio.create_subprocess_exec",
        return_value=fake_process,
    ) as mock_exec:
        result = await tool.execute(args=["--version"])

    payload = json.loads(result)
    assert payload["command"] == [npx_path, "--yes", "agent-browser", "--version"]
    assert payload["cwd"] == "/tmp/work"
    assert payload["exitCode"] == 0
    assert payload["stdout"] == "agent-browser 0.25.5\n"
    assert payload["stderr"] == ""
    assert payload["action_class"] == "read_navigation"
    mock_exec.assert_called_once()


def test_agent_browser_classifies_navigation_and_mutation() -> None:
    assert classify_browser_action(["--headed", "open", "https://example.com"]) == "read_navigation"
    assert classify_browser_action(["click", "@e1"]) == "state_changing"
    assert classify_browser_action(["type", "@e1", "hello"]) == "state_changing"
    assert classify_browser_action(["unknown-command"]) == "state_changing"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args, working_dir",
    [
        (["screenshot", "/tmp/page.png"], None),
        (["open", "https://example.com", "--screenshot", "/tmp/page.png"], None),
        (["snapshot"], "/tmp/caller-selected"),
    ],
)
async def test_agent_browser_read_only_rejects_screenshot_output_and_caller_cwd(
    args: list[str], working_dir: str | None, monkeypatch
) -> None:
    process = AsyncMock()
    monkeypatch.setattr("nanobot.agent.tools.agent_browser.run_owned_process", process)

    result = await AgentBrowserTool(
        read_only=True,
        working_dir="/tmp/configured",
    ).execute(args=args, working_dir=working_dir)

    assert isinstance(result, ToolOutcome)
    assert result.stop_reason == RunStatus.POLICY_BLOCKED.value
    process.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_browser_read_only_allows_snapshot_without_override(monkeypatch) -> None:
    process = AsyncMock(return_value=SimpleNamespace(
        timed_out=False,
        stdout=b"ok\n",
        stderr=b"",
        returncode=0,
    ))
    monkeypatch.setattr("nanobot.agent.tools.agent_browser._resolve_npx", lambda: "npx")
    monkeypatch.setattr("nanobot.agent.tools.agent_browser.run_owned_process", process)

    result = await AgentBrowserTool(read_only=True, working_dir="/tmp/configured").execute(
        args=["snapshot"],
    )

    assert json.loads(result)["action_class"] == "read_navigation"
    process.assert_awaited_once()
