"""Tool wrapper for the agent-device CLI."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Sequence
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.process import run_owned_process
from nanobot.agent.turn import ToolOutcome

READ_ONLY_DEVICE_ACTIONS = frozenset({
    "devices",
    "inspect",
    "snapshot",
    "status",
    "logs",
})
_FILE_OUTPUT_OPTIONS = frozenset({
    "--file",
    "--filename",
    "--output",
    "--out",
    "--path",
    "--save",
    "-o",
})


def classify_device_action(args: Sequence[str]) -> str:
    """Classify agent-device args for policy consumers.

    Device inspection is read-only; commands that can interact with an app or
    simulator are conservatively marked state-changing.
    """

    if not isinstance(args, Sequence) or isinstance(args, (str, bytes)):
        return "state_changing"
    if any(not isinstance(arg, str) for arg in args):
        return "state_changing"
    normalized_args = [arg.strip().lower() for arg in args if isinstance(arg, str)]
    option_names = [arg.split("=", 1)[0] for arg in normalized_args]
    # A screenshot can write a caller-selected path, so every screenshot form
    # is state-changing for both delegated and main-agent policy decisions.
    if "screenshot" in normalized_args or any(
        arg.startswith("--screenshot") for arg in normalized_args
    ):
        return "state_changing"
    if any(
        option in _FILE_OUTPUT_OPTIONS
        or option.startswith(("--output", "--out", "--path", "--file", "--save"))
        for option in option_names
    ):
        return "state_changing"
    action = next((arg for arg in normalized_args if arg and not arg.startswith("-")), "")
    if action not in READ_ONLY_DEVICE_ACTIONS:
        return "state_changing"
    action_names = READ_ONLY_DEVICE_ACTIONS | frozenset({
        "open", "launch", "close", "press", "tap", "type", "input", "submit", "evaluate",
        "app", "settings", "install", "uninstall", "screenshot-save",
    })
    try:
        action_index = next(
            index for index, arg in enumerate(normalized_args)
            if arg == action
        )
    except StopIteration:
        return "state_changing"
    if any(arg in action_names for arg in normalized_args[action_index + 1:]):
        return "state_changing"
    return "read_navigation"


def device_action_metadata(args: Sequence[str]) -> dict[str, Any]:
    """Return machine-readable device action classification metadata."""

    action_class = classify_device_action(args)
    return {
        "action_class": action_class,
        "read_navigation": action_class == "read_navigation",
        "state_changing": action_class == "state_changing",
    }


def _resolve_npx() -> str | None:
    """Resolve the runnable npx command, including Windows .cmd launchers."""

    return shutil.which("npx") or shutil.which("npx.cmd")


class AgentDeviceTool(Tool):
    """Execute local simulator/emulator automation tasks via agent-device."""

    def __init__(
        self,
        package: str = "agent-device",
        timeout: int = 180,
        max_output_chars: int = 12000,
        working_dir: str | None = None,
        read_only: bool = False,
    ):
        self.package = package
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.working_dir = working_dir
        self.read_only = read_only

    @property
    def supports_parallel_calls(self) -> bool:
        """Simulator/emulator state is shared and must be serialized."""
        return False

    @property
    def name(self) -> str:
        return "agent_device"

    @property
    def description(self) -> str:
        return (
            "Run agent-device CLI for local iOS simulator and Android emulator automation. "
            "Pass CLI args as a string array (for example ['devices', '--platform', 'ios'], "
            "['open', 'Settings', '--platform', 'ios'], ['snapshot', '-i'], ['press', '@e2'], "
            "['screenshot', 'artifacts/mobile-tests/run-1/screen.png'], or ['close'])."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "args": {
                    "type": "array",
                    "description": "CLI arguments passed to agent-device.",
                    "items": {"type": "string"},
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (5-900).",
                    "minimum": 5,
                    "maximum": 900,
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command.",
                },
            },
            "required": ["args"],
        }

    async def execute(
        self,
        args: list[str],
        timeout: int | None = None,
        working_dir: str | None = None,
        **kwargs: Any,
    ) -> str | ToolOutcome:
        if not args:
            return "Error: args must include at least one CLI argument"
        if len(args) > 120:
            return "Error: too many CLI arguments (max 120)"
        if any(len(arg) > 2000 for arg in args):
            return "Error: one or more CLI arguments exceed max length (2000 chars)"
        if self.read_only and working_dir is not None:
            return ToolOutcome(
                content="Error: caller-supplied working_dir is blocked for delegated read-only device actions",
                stop_reason="policy_blocked",
                policy_metadata={
                    "policy": "delegated_read_only",
                    "reason": "caller working_dir override",
                    "requires_approval": False,
                },
            )
        if self.read_only and classify_device_action(args) != "read_navigation":
            return ToolOutcome(
                content="Error: state-changing or unknown device action is blocked",
                stop_reason="policy_blocked",
                policy_metadata={
                    "policy": "delegated_read_only",
                    "action_class": "state_changing",
                    "requires_approval": False,
                },
            )

        run_timeout = timeout if timeout is not None else self.timeout
        cwd = working_dir or self.working_dir or os.getcwd()

        npx_command = _resolve_npx()
        if not npx_command:
            return (
                "Error: 'npx' was not found in PATH. "
                "Install Node.js/npm first, then retry."
            )

        command = [npx_command, "--yes", self.package, *args]
        try:
            result = await run_owned_process(
                command,
                timeout=run_timeout,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except Exception as e:
            return f"Error: failed to start agent-device: {e}"

        if result.timed_out:
            return json.dumps(
                {
                    "error": f"agent-device timed out after {run_timeout} seconds",
                    "command": command,
                    "cwd": cwd,
                    "action_class": classify_device_action(args),
                },
                ensure_ascii=False,
            )

        stdout_text = result.stdout.decode("utf-8", errors="replace")
        stderr_text = result.stderr.decode("utf-8", errors="replace")

        stdout_truncated = len(stdout_text) > self.max_output_chars
        stderr_truncated = len(stderr_text) > self.max_output_chars
        if stdout_truncated:
            stdout_text = stdout_text[: self.max_output_chars]
        if stderr_truncated:
            stderr_text = stderr_text[: self.max_output_chars]

        return json.dumps(
            {
                "command": command,
                "cwd": cwd,
                "exitCode": result.returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "stdoutTruncated": stdout_truncated,
                "stderrTruncated": stderr_truncated,
                "action_class": classify_device_action(args),
            },
            ensure_ascii=False,
        )


__all__ = [
    "AgentDeviceTool",
    "READ_ONLY_DEVICE_ACTIONS",
    "classify_device_action",
    "device_action_metadata",
]
