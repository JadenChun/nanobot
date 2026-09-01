"""Shell execution tool."""

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.process import run_owned_process


class ExecTool(Tool):
    """Tool to execute shell commands."""

    _DESKTOP_MUTATION_COMMAND_RE = re.compile(
        r"(?:^|(?:&&|\|\||[;|&()\n]))\s*"
        r"(?:(?:command|exec|nohup|sudo)\s+)*"
        r"(?:env\s+(?:[A-Za-z_][A-Za-z0-9_]*=[^\s]+\s+)*)?"
        r"(?:[A-Za-z_][A-Za-z0-9_]*=[^\s]+\s+)*"
        r"(?:/(?:usr/)?bin/)?(?:open|osascript|xdotool|wmctrl)(?=\s|$)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        path_append: str = "",
        resource_policy: Any | None = None,
        desktop_use_active: bool = False,
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        # When desktop_use is enabled, shell-based desktop mutation is blocked
        # by _guard_command. Read-only inspection remains available for tasks
        # that explicitly request it (for example, querying lsappinfo).
        self.desktop_use_active = bool(desktop_use_active)
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",          # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",              # del /f, del /q
            r"\brmdir\s+/s\b",               # rmdir /s
            r"(?:^|[;&|]\s*)format\b",       # format (as standalone command only)
            r"\b(mkfs|diskpart)\b",          # disk operations
            r"\bdd\s+if=",                   # dd
            r">\s*/dev/sd",                  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",          # fork bomb
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append
        self.resource_policy = resource_policy

    @property
    def name(self) -> str:
        return "exec"

    _MAX_TIMEOUT = 600
    _MAX_OUTPUT = 10_000
    _GIT_NETWORK_OPS = ("push", "pull", "fetch", "clone", "ls-remote")

    @property
    def description(self) -> str:
        base = "Execute a shell command and return its output. Use with caution."
        if self.desktop_use_active:
            return (
                base
                + " desktop_use is enabled, so shell-based desktop/window/app mutations "
                "are blocked \u2014 no `osascript`, `AppleScript`, `open -a`, `wmctrl`, "
                "`xdotool`, or similar. Route UI actions through `desktop_use`; if it "
                "fails, report the failure instead of using a shell fallback."
            )
        return base

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Timeout in seconds. Increase for long-running commands "
                        "like compilation or installation (default 60, max 600)."
                    ),
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            "required": ["command"],
        }

    async def execute(
        self, command: str, working_dir: str | None = None,
        timeout: int | None = None, **kwargs: Any,
    ) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        effective_timeout = min(timeout or self.timeout, self._MAX_TIMEOUT)

        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append
        self._apply_git_noninteractive_env(command, env)

        try:
            result = await run_owned_process(
                command,
                shell=True,
                timeout=effective_timeout,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            if result.timed_out:
                return f"Error: Command timed out after {effective_timeout} seconds"

            output_parts = []

            if result.stdout:
                output_parts.append(result.stdout.decode("utf-8", errors="replace"))

            if result.stderr:
                stderr_text = result.stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            output_parts.append(f"\nExit code: {result.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            # Head + tail truncation to preserve both start and end of output
            max_len = self._MAX_OUTPUT
            if len(result) > max_len:
                half = max_len // 2
                result = (
                    result[:half]
                    + f"\n\n... ({len(result) - max_len:,} chars truncated) ...\n\n"
                    + result[-half:]
                )

            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"

    @classmethod
    def _apply_git_noninteractive_env(cls, command: str, env: dict[str, str]) -> None:
        """Prevent git network commands from hanging on interactive prompts."""
        if not cls._is_git_network_command(command):
            return
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        if "GIT_SSH_COMMAND" not in env:
            env["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes -o ConnectTimeout=10"

    @classmethod
    def _is_git_network_command(cls, command: str) -> bool:
        lower = command.lower()
        for op in cls._GIT_NETWORK_OPS:
            # Only match within one shell segment to reduce false positives.
            if re.search(rf"\bgit\b[^|;&\n]*\b{re.escape(op)}\b", lower):
                return True
        return False

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.desktop_use_active and self._DESKTOP_MUTATION_COMMAND_RE.search(cmd):
            return (
                "Error: Command blocked because desktop_use is enabled. "
                "Shell-based desktop mutation is not permitted; use desktop_use or "
                "report its failure without a fallback."
            )

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        from nanobot.security.network import contains_internal_url
        if contains_internal_url(cmd):
            return "Error: Command blocked by safety guard (internal/private URL detected)"

        if self.resource_policy:
            if policy_error := self.resource_policy.validate_exec(command, cwd):
                return policy_error
            return None

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return "Error: Command blocked by safety guard (path traversal detected)"

            cwd_path = Path(cwd).resolve()

            for raw in self._extract_absolute_paths(cmd):
                try:
                    expanded = os.path.expandvars(raw.strip())
                    p = Path(expanded).expanduser().resolve()
                except Exception:
                    continue
                if p.is_absolute() and cwd_path not in p.parents and p != cwd_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"

        return None

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]+", command)   # Windows: C:\...
        posix_paths = re.findall(r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command) # POSIX: /absolute only
        home_paths = re.findall(r"(?:^|[\s|>'\"])(~[^\s\"'>;|<]*)", command) # POSIX/Windows home shortcut: ~
        return win_paths + posix_paths + home_paths
