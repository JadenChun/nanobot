"""Internal execution policy for the main agent harness."""

from __future__ import annotations

import difflib
import re
import shlex
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanobot.agent.tools.filesystem import _find_match
from nanobot.context_repo import ContextRepoManager
from nanobot.providers.base import ToolCallRequest

_APPROVAL_REQUIRED = "approval_required"


@dataclass(slots=True)
class ToolPolicyDecision:
    """Decision returned before executing a tool batch."""

    action: str = "allow"
    stop_reason: str | None = None
    response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # ``True`` is reserved for the main agent's run-wide approval flow.  A
    # delegated read-only policy explicitly returns ``False`` so the caller
    # cannot accidentally create a pending approval or ask the user to elevate
    # an immutable role.
    requires_approval: bool = False


class ToolPolicy:
    """Base policy surface for tool-batch interception."""

    async def evaluate(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_calls: list[ToolCallRequest],
    ) -> ToolPolicyDecision:
        return ToolPolicyDecision()


@dataclass(slots=True)
class DelegatedReadOnlyPolicy(ToolPolicy):
    """Non-elevatable policy used by planner/reviewer/explorer/crawler roles.

    Role registries are the first boundary, but a model can still emit a
    forged or stale tool call.  This policy validates the complete batch before
    execution.  Any disallowed, unknown, malformed, or state-changing call
    blocks the entire batch; ``approval_granted`` is intentionally ignored so a
    main-agent approval cannot turn a read-only delegate into a writer.
    """

    allowed_tools: Collection[str] = field(default_factory=frozenset)
    role: str = "read_only"
    # Accepted for construction compatibility with generic policy factories.
    # It must not affect evaluation.
    approval_granted: bool = False

    _NEVER_READ_ONLY = frozenset({
        "write_file",
        "edit_file",
        "exec",
        "cron",
        "message",
        "delegate_task",
        "plan_task",
        "review_work",
        "explore",
        "crawl_research",
        "desktop_use",
    })

    def __post_init__(self) -> None:
        if not self.allowed_tools:
            try:
                from nanobot.agent.capabilities import role_capabilities

                self.allowed_tools = role_capabilities(self.role).tools
            except ValueError:
                # An unregistered role with no explicit tools remains empty
                # and therefore fails closed for every call.
                self.allowed_tools = frozenset()
        self.allowed_tools = frozenset(
            name for name in self.allowed_tools if isinstance(name, str) and name
        )

    async def evaluate(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_calls: list[ToolCallRequest],
    ) -> ToolPolicyDecision:
        _ = messages
        if not tool_calls:
            return ToolPolicyDecision()

        reasons: list[str] = []
        blocked_tools: list[str] = []
        for tool_call in tool_calls:
            reason = self._blocked_reason(tool_call)
            if reason:
                reasons.append(reason)
                if isinstance(tool_call.name, str) and tool_call.name not in blocked_tools:
                    blocked_tools.append(tool_call.name)

        if not reasons:
            return ToolPolicyDecision()

        # Keep this response deliberately free of approval language.  The
        # outer loop treats policy_blocked as terminal and only creates pending
        # approval state for approval_required.
        summary = "; ".join(dict.fromkeys(reasons))
        return ToolPolicyDecision(
            action="respond",
            stop_reason="policy_blocked",
            response=f"Blocked by delegated read-only policy: {summary}.",
            metadata={
                "policy": "delegated_read_only",
                "role": self.role,
                "blocked_tools": blocked_tools,
                "reasons": list(dict.fromkeys(reasons)),
                "requires_approval": False,
            },
            requires_approval=False,
        )

    def _blocked_reason(self, tool_call: ToolCallRequest) -> str | None:
        name = tool_call.name if isinstance(tool_call.name, str) else ""
        if name not in self.allowed_tools:
            return f"tool {name or '<unknown>'!r} is not permitted"
        if name in self._NEVER_READ_ONLY:
            return f"tool {name!r} is state-changing and not permitted"

        arguments = tool_call.arguments
        if not isinstance(arguments, dict):
            return f"tool {name!r} supplied malformed arguments"

        if name == "agent_browser":
            from nanobot.agent.tools.agent_browser import classify_browser_action

            if arguments.get("working_dir") is not None:
                return "agent_browser caller working_dir override is not permitted"
            args = arguments.get("args")
            if not isinstance(args, (list, tuple)):
                return "agent_browser action is unknown"
            if classify_browser_action(args) != "read_navigation":
                return "agent_browser action is state-changing or unknown"
        elif name == "agent_device":
            from nanobot.agent.tools.agent_device import classify_device_action

            if arguments.get("working_dir") is not None:
                return "agent_device caller working_dir override is not permitted"
            args = arguments.get("args")
            if not isinstance(args, (list, tuple)):
                return "agent_device action is unknown"
            if classify_device_action(args) != "read_navigation":
                return "agent_device action is state-changing or unknown"
        elif name == "social_crawl":
            from nanobot.agent.tools.social_crawl import classify_crawler_action

            action = arguments.get("action")
            if not isinstance(action, str):
                return "social_crawl action is unknown"
            if classify_crawler_action(action) != "read_navigation":
                return "social_crawl action is state-changing or unknown"
        return None


@dataclass(slots=True)
class RiskyActionPolicy(ToolPolicy):
    """Require approval for risky or hard-to-undo tool batches."""

    workspace: Path
    approval_granted: bool = False
    context_manager: ContextRepoManager = field(default_factory=ContextRepoManager)

    _RISKY_EXEC_PATTERNS = (
        (re.compile(r"\brm\s+-[rf]{1,2}\b"), "delete files or directories"),
        (re.compile(r"\bgit\s+reset\b"), "reset git history"),
        (re.compile(r"\bgit\s+clean\b"), "remove untracked files"),
        (re.compile(r"\bgit\b[^|;&\n]*\bbranch\b[^|;&\n]*(?:\s-D\b|\s-d\b|--delete\b)"), "delete a git branch"),
        (re.compile(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*\s--delete\b"), "delete a git branch"),
        (re.compile(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*(?:^|\s)\+?:[^\s|;&]+"), "delete a git branch"),
        (re.compile(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*\s--mirror\b"), "delete git branches through a mirror push"),
        (re.compile(r"\bgit\s+push\b"), "push commits to a remote repository"),
        (re.compile(r"\bgit\s+checkout\b[^|;&\n]*\s+-f\b"), "force checkout changes"),
        (re.compile(r"\bdrop\s+table\b"), "drop database tables"),
        (re.compile(r"\bdelete\s+from\b"), "delete database rows"),
        (re.compile(r"\btruncate\s+table\b"), "truncate database tables"),
        (re.compile(r"\b(kubectl|helm)\b[^|;&\n]*\b(apply|delete|upgrade|rollback)\b"), "change deployed infrastructure"),
        (re.compile(r"\b(npm|pnpm|yarn|bun)\b[^|;&\n]*\bpublish\b"), "publish a package"),
    )

    def mutating_tool_names(self) -> set[str]:
        return {"write_file", "edit_file", "exec", "cron", "agent_browser", "agent_device"}

    async def evaluate(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_calls: list[ToolCallRequest],
    ) -> ToolPolicyDecision:
        if self.approval_granted or not tool_calls:
            return ToolPolicyDecision()

        reasons: list[str] = []
        for tool_call in tool_calls:
            reason = self._risky_reason(tool_call)
            if reason:
                reasons.append(reason)

        if not reasons:
            return ToolPolicyDecision()

        summary = "; ".join(dict.fromkeys(reasons))
        response = (
            "I'm about to take a risky action and want your approval first. "
            f"Planned action: {summary}. Reply `yes` to continue or `no` to cancel."
        )
        return ToolPolicyDecision(
            action="respond",
            stop_reason=_APPROVAL_REQUIRED,
            response=response,
            metadata={"reasons": reasons, "summary": summary, "requires_approval": True},
            requires_approval=True,
        )

    def batch_has_mutation(self, tool_calls: list[ToolCallRequest]) -> bool:
        return any(tc.name in self.mutating_tool_names() for tc in tool_calls)

    def _risky_reason(self, tool_call: ToolCallRequest) -> str | None:
        if tool_call.name == "exec":
            return self._risky_exec_reason(
                str(tool_call.arguments.get("command") or ""),
                str(tool_call.arguments.get("working_dir") or ""),
            )
        if tool_call.name == "write_file":
            return self._risky_write_reason(
                str(tool_call.arguments.get("path") or ""),
                str(tool_call.arguments.get("content") or ""),
            )
        if tool_call.name == "edit_file":
            return self._risky_edit_reason(
                str(tool_call.arguments.get("path") or ""),
                str(tool_call.arguments.get("old_text") or ""),
                str(tool_call.arguments.get("new_text") or ""),
                bool(tool_call.arguments.get("replace_all")),
            )
        if tool_call.name == "cron":
            return "schedule a recurring automation"
        if tool_call.name == "agent_browser":
            # Browser classification is deliberately conservative: only the
            # WP2 read/navigation primitives are autonomous.  Unknown and
            # generic actions (click/type/submit/evaluate/etc.) require the
            # same run-wide approval as other side effects.
            from nanobot.agent.tools.agent_browser import classify_browser_action

            args = tool_call.arguments.get("args") or []
            if classify_browser_action(args) != "read_navigation":
                return "perform a state-changing browser action"
        if tool_call.name == "agent_device":
            # Device inspection is safe; interaction, app/system controls, and
            # unknown commands require the main run-wide approval.
            from nanobot.agent.tools.agent_device import classify_device_action

            args = tool_call.arguments.get("args") or []
            if classify_device_action(args) != "read_navigation":
                return "perform a state-changing device action"
        return None

    def _risky_exec_reason(self, command: str, working_dir: str | None = None) -> str | None:
        lower = command.strip().lower()
        if not lower:
            return None
        if self._is_autonomous_context_git_command(command, working_dir):
            return None
        for pattern, label in self._RISKY_EXEC_PATTERNS:
            if pattern.search(lower):
                return label
        return None

    def _command_working_dir(self, command: str, working_dir: str | None) -> Path:
        base = self._resolve_workspace_path(working_dir or ".")
        git_c_path = self._extract_git_c_path(command)
        if git_c_path:
            candidate = Path(git_c_path).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            return candidate.resolve(strict=False)
        return base

    def _extract_git_c_path(self, command: str) -> str | None:
        try:
            tokens = shlex.split(command, posix=False)
        except ValueError:
            return None
        cleaned = [token.strip("'\"") for token in tokens]
        lowered = [token.lower() for token in cleaned]
        for index, token in enumerate(lowered):
            if token != "git":
                continue
            if index + 2 < len(tokens) and cleaned[index + 1] == "-C":
                return cleaned[index + 2]
        return None

    def _is_autonomous_context_git_command(self, command: str, working_dir: str | None) -> bool:
        lower = command.strip().lower()
        if not re.search(r"\bgit\b", lower):
            return False
        if self._deletes_git_branch(lower):
            return False

        cwd = self._command_working_dir(command, working_dir)
        target = self.context_manager.find_target_repo_for_path(cwd)
        repo = self.context_manager.find_repo_for_path(cwd)
        if target and (repo is None or len(str(target.path or "")) >= len(str(repo.path))):
            return False
        return bool(repo and repo.auto_push_enabled())

    def _deletes_git_branch(self, lower_command: str) -> bool:
        return bool(
            re.search(r"\bgit\b[^|;&\n]*\bbranch\b[^|;&\n]*(?:\s-D\b|\s-d\b|--delete\b)", lower_command)
            or re.search(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*\s--delete\b", lower_command)
            or re.search(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*(?:^|\s)\+?:[^\s|;&]+", lower_command)
            or re.search(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*\s--mirror\b", lower_command)
        )

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        return candidate.resolve()

    def _changed_line_count(self, before: str, after: str) -> tuple[int, int]:
        before_lines = before.splitlines()
        after_lines = after.splitlines()
        matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines)
        changed = 0
        total = max(len(before_lines), len(after_lines), 1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                changed += max(i2 - i1, j2 - j1)
        return changed, total

    # Files at or below this many lines are considered trivial state; a full
    # rewrite is not "risky" because there is nothing of substance to lose.
    # The absolute ``changed > 200`` guard still catches genuinely large edits.
    _SMALL_FILE_LINE_THRESHOLD = 10

    def _is_agent_state_file(self, raw_path: str) -> bool:
        """Memory files the agent manages freely — no size checks."""
        p = Path(raw_path)
        return "memory" in p.parts

    def _is_autonomous_managed_path(self, raw_path: str) -> bool:
        """Writable managed paths already have finer-grained repo policy checks."""
        if not raw_path:
            return False
        path = self._resolve_workspace_path(raw_path)

        target = self.context_manager.find_target_repo_for_path(path)
        repo = self.context_manager.find_repo_for_path(path)
        if target and (repo is None or len(str(target.path or "")) >= len(str(repo.path))):
            rel = target.rel_path(path)
            return bool(
                rel
                and not target.is_protected(rel)
                and not target.requires_proposal(rel)
                and target.is_writable(rel)
            )

        if repo:
            rel = repo.rel_path(path)
            return bool(
                rel
                and not repo.read_only
                and not repo.is_protected(rel)
                and not repo.requires_proposal(rel)
                and not repo.blocks_direct_store_edit(rel)
                and repo.is_writable(rel)
            )

        return False

    def _risky_write_reason(self, raw_path: str, new_content: str) -> str | None:
        if not raw_path:
            return None
        if self._is_agent_state_file(raw_path) or self._is_autonomous_managed_path(raw_path):
            return None
        path = self._resolve_workspace_path(raw_path)
        if not path.exists() or not path.is_file():
            return None
        try:
            old_content = path.read_text(encoding="utf-8")
        except Exception:
            return f"overwrite existing file {path.name}"
        changed, total = self._changed_line_count(old_content, new_content)
        if changed > 200:
            return f"overwrite a large portion of {path.name}"
        if total <= self._SMALL_FILE_LINE_THRESHOLD:
            return None
        if changed / total > 0.5:
            return f"overwrite a large portion of {path.name}"
        return None

    def _risky_edit_reason(
        self,
        raw_path: str,
        old_text: str,
        new_text: str,
        replace_all: bool,
    ) -> str | None:
        if not raw_path:
            return None
        if self._is_agent_state_file(raw_path) or self._is_autonomous_managed_path(raw_path):
            return None
        if replace_all:
            return f"apply a bulk replace in {Path(raw_path).name}"

        touched_lines = max(len(old_text.splitlines()), len(new_text.splitlines()))
        if touched_lines > 200:
            return f"edit more than 200 lines in {Path(raw_path).name}"

        path = self._resolve_workspace_path(raw_path)
        if not path.exists() or not path.is_file():
            return None

        try:
            content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        except Exception:
            return None

        match, count = _find_match(content, old_text.replace("\r\n", "\n"))
        if count > 1:
            return f"replace repeated content in {path.name}"
        if match is not None:
            changed, total = self._changed_line_count(match, new_text.replace("\r\n", "\n"))
            if changed > 200 or changed / total > 0.5:
                return f"rewrite most of the matched block in {path.name}"
        return None
