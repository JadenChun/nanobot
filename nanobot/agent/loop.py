"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import AsyncExitStack, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.policy import RiskyActionPolicy
from nanobot.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec, clear_old_tool_results
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.agent_browser import AgentBrowserTool
from nanobot.agent.tools.agent_device import AgentDeviceTool
from nanobot.agent.tools.desktop_use import DesktopUseTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.explore import ExploreTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.image import (
    ImageGenerationTool,
    image_generation_available,
)
from nanobot.agent.tools.mcp import is_read_only_mcp_tool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.pipeline import SpawnPipelineTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.agent.write_guard import FileLockRegistry
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.helpers import build_assistant_message, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.config.schema import (
        AgentBrowserConfig,
        AgentDeviceConfig,
        ChannelsConfig,
        DesktopUseConfig,
        ExecToolConfig,
        ImageConfig,
        MaxTokensConfig,
        WebSearchConfig,
    )
    from nanobot.cron.service import CronService


class _LoopHook(AgentHook):
    """Core lifecycle hook for the main agent loop.

    Handles streaming delta relay, progress reporting, tool-call logging,
    and think-tag stripping for the built-in agent path.
    """

    def __init__(
        self,
        agent_loop: AgentLoop,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        session_key: str | None = None,
    ) -> None:
        self._loop = agent_loop
        self._on_progress = on_progress
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._channel = channel
        self._chat_id = chat_id
        self._message_id = message_id
        self._session_key = session_key
        self._stream_buf = ""

    def wants_streaming(self) -> bool:
        return self._on_stream is not None

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        self._stream_buf += delta
        if self._on_stream:
            await self._on_stream(delta)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        # Deltas are forwarded live during streaming; no need to re-send the
        # full buffer at the end (that would duplicate the content).
        # Think-tag stripping is handled by the renderer's _render().
        if self._on_stream_end:
            await self._on_stream_end(resuming=resuming)
        self._stream_buf = ""

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        if self._on_progress:
            if not self._on_stream:
                thought = self._loop._strip_think(
                    context.response.content if context.response else None
                )
                if thought:
                    await self._on_progress(thought)
            tool_hint = self._loop._strip_think(self._loop._tool_hint(context.tool_calls))
            await self._on_progress(tool_hint, tool_hint=True)
        for tc in context.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            logger.info("Tool call: {}({})", tc.name, args_str[:200])
        self._loop._set_tool_context(self._channel, self._chat_id, self._message_id, self._session_key)

    async def after_iteration(self, context: AgentHookContext) -> None:
        for event in context.tool_events:
            if event.get("status") == "error":
                logger.error(
                    "Tool error: {} -> {}",
                    event.get("name", "unknown"),
                    event.get("detail", "(no detail)"),
                )

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return self._loop._strip_think(content)


class _LoopHookChain(AgentHook):
    """Run the core loop hook first, then best-effort extra hooks.

    This preserves the historical failure behavior of ``_LoopHook`` while still
    letting user-supplied hooks opt into ``CompositeHook`` isolation.
    """

    __slots__ = ("_primary", "_extras")

    def __init__(self, primary: AgentHook, extra_hooks: list[AgentHook]) -> None:
        self._primary = primary
        self._extras = CompositeHook(extra_hooks)

    def wants_streaming(self) -> bool:
        return self._primary.wants_streaming() or self._extras.wants_streaming()

    async def before_iteration(self, context: AgentHookContext) -> None:
        await self._primary.before_iteration(context)
        await self._extras.before_iteration(context)

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        await self._primary.on_stream(context, delta)
        await self._extras.on_stream(context, delta)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        await self._primary.on_stream_end(context, resuming=resuming)
        await self._extras.on_stream_end(context, resuming=resuming)

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        await self._primary.before_execute_tools(context)
        await self._extras.before_execute_tools(context)

    async def after_iteration(self, context: AgentHookContext) -> None:
        await self._primary.after_iteration(context)
        await self._extras.after_iteration(context)

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        content = self._primary.finalize_content(context, content)
        return self._extras.finalize_content(context, content)


@dataclass(slots=True)
class _PendingApproval:
    """Pending risky action awaiting an explicit yes/no reply."""

    summary: str
    created_at: float
    session_key: str


@dataclass(slots=True)
class _PlanDecision:
    """Internal planner output."""

    decision: str = "execute"
    response: str | None = None
    action_summary: str = ""
    review_goal: str = ""
    references: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_handoff(self) -> bool:
        return bool(self.action_summary or self.review_goal or self.references)


@dataclass(slots=True)
class _VerificationResult:
    """Structured verifier outcome."""

    verdict: str = "PASS"
    issues: list[str] = field(default_factory=list)
    feedback: str = ""


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000
    _TOOL_RESULT_CLEARING_SAFETY_BUFFER = 1024
    _TOOL_RESULT_CLEAR_TRIGGER_RATIO = 0.8
    _TOOL_RESULT_CLEAR_TARGET_RATIO = 0.6
    _DEFAULT_PLANNER_MAX_ITERATIONS = 12
    _DEFAULT_PLANNER_EXPLORE_MAX_ITERATIONS = 100
    _DEFAULT_PLANNER_MAX_PARALLEL_EXPLORES = 2
    _PLANNER_HANDOFF_METADATA_KEY = "planner_handoff"

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_tokens: "MaxTokensConfig | None" = None,
        max_iterations: int = 200,
        context_window_tokens: int | None = None,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        agent_browser_config: AgentBrowserConfig | None = None,
        agent_device_config: AgentDeviceConfig | None = None,
        desktop_use_config: "DesktopUseConfig | None" = None,
        exec_config: ExecToolConfig | None = None,
        image_config: ImageConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        hooks: list[AgentHook] | None = None,
        context_paths: list[Path] | None = None,
        context_repos: list[Any] | None = None,
        planning_mode: str = "agent",
        planner_max_iterations: int = _DEFAULT_PLANNER_MAX_ITERATIONS,
        planner_explore_subagent_max_iterations: int = _DEFAULT_PLANNER_EXPLORE_MAX_ITERATIONS,
        planner_max_parallel_explore_agents: int = _DEFAULT_PLANNER_MAX_PARALLEL_EXPLORES,
        tool_result_clearing_keep: int = 3,
        consolidation_trigger_ratio: float = 0.5,
        consolidation_target_ratio: float = 0.3,
    ):
        from nanobot.config.schema import (
            AgentBrowserConfig,
            AgentDeviceConfig,
            DesktopUseConfig,
            ExecToolConfig,
            ImageConfig,
            MaxTokensConfig,
            WebSearchConfig,
        )

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.tool_result_clearing_keep = tool_result_clearing_keep
        self.consolidation_trigger_ratio = consolidation_trigger_ratio
        self.consolidation_target_ratio = consolidation_target_ratio
        self.max_tokens = max_tokens or MaxTokensConfig()
        if context_window_tokens is not None and context_window_tokens > 0:
            self.max_tokens.input = context_window_tokens
        self.context_window_tokens = self.max_tokens.input
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.agent_browser_config = agent_browser_config or AgentBrowserConfig()
        self.agent_device_config = agent_device_config or AgentDeviceConfig()
        self.desktop_use_config = desktop_use_config or DesktopUseConfig()
        self.exec_config = exec_config or ExecToolConfig()
        self.image_config = image_config or ImageConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []
        self.planning_mode = planning_mode
        self.planner_max_iterations = max(1, planner_max_iterations)
        self.planner_explore_subagent_max_iterations = max(1, planner_explore_subagent_max_iterations)
        self.planner_max_parallel_explore_agents = max(1, planner_max_parallel_explore_agents)

        self.context_manager = ContextRepoManager.from_config(
            context_paths=context_paths,
            context_repos=context_repos,
        )
        self.resource_policy = ResourceAccessPolicy(
            workspace=workspace,
            context_manager=self.context_manager,
            restrict_to_workspace=restrict_to_workspace,
        )

        self.context = ContextBuilder(
            workspace,
            timezone=timezone,
            context_manager=self.context_manager,
            planning_mode=planning_mode,
        )
        self.context_paths = self.context_manager.paths
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.runner = AgentRunner(provider)
        self._file_lock_registry = FileLockRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            agent_browser_config=self.agent_browser_config,
            agent_device_config=self.agent_device_config,
            desktop_use_config=self.desktop_use_config,
            exec_config=self.exec_config,
            image_config=self.image_config,
            context_paths=self.context_paths,
            context_manager=self.context_manager,
            restrict_to_workspace=restrict_to_workspace,
            mcp_servers=mcp_servers or {},
            planner_max_parallel_explore_agents=self.planner_max_parallel_explore_agents,
            file_lock_registry=self._file_lock_registry,
            on_completion=self._record_subagent_completion,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._pending_approvals: dict[str, _PendingApproval] = {}
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=self.max_tokens.input,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=self.max_tokens.output,
            consolidation_trigger_ratio=self.consolidation_trigger_ratio,
            consolidation_target_ratio=self.consolidation_target_ratio,
        )
        self._register_default_tools()
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    def _context_skill_paths(self) -> list[Path]:
        """Return skill directories from configured context repositories."""
        return self.context_manager.skill_roots()

    def _extra_read_dirs(self, allowed_dir: Path | None) -> list[Path] | None:
        """Extra read roots needed when tool access is workspace-restricted."""
        if not allowed_dir:
            return None
        return [BUILTIN_SKILLS_DIR, *self.resource_policy.extra_read_dirs()]

    def _extra_write_dirs(self, allowed_dir: Path | None) -> list[Path] | None:
        """Extra write roots needed when tool access is workspace-restricted."""
        if not allowed_dir:
            return None
        return self.resource_policy.extra_write_dirs()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = self._extra_read_dirs(allowed_dir)
        extra_write = self._extra_write_dirs(allowed_dir)
        self.tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read, resource_policy=self.resource_policy))
        self.tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read, resource_policy=self.resource_policy))
        self.tools.register(WriteFileTool(
            workspace=self.workspace,
            allowed_dir=allowed_dir,
            extra_allowed_dirs=extra_write,
            resource_policy=self.resource_policy,
            lock_registry=self._file_lock_registry,
        ))
        self.tools.register(EditFileTool(
            workspace=self.workspace,
            allowed_dir=allowed_dir,
            extra_allowed_dirs=extra_write,
            resource_policy=self.resource_policy,
            lock_registry=self._file_lock_registry,
        ))
        if self.exec_config.enable:
            self.tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
                resource_policy=self.resource_policy,
                desktop_use_active=self.desktop_use_config.enabled,
            ))
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        if image_generation_available(self.image_config):
            self.tools.register(ImageGenerationTool(config=self.image_config, workspace=self.workspace))
        if self.agent_browser_config.enabled:
            self.tools.register(AgentBrowserTool(
                package=self.agent_browser_config.package,
                timeout=self.agent_browser_config.timeout,
                max_output_chars=self.agent_browser_config.max_output_chars,
                working_dir=str(self.workspace),
            ))
        if self.agent_device_config.enabled:
            self.tools.register(AgentDeviceTool(
                package=self.agent_device_config.package,
                timeout=self.agent_device_config.timeout,
                max_output_chars=self.agent_device_config.max_output_chars,
                working_dir=str(self.workspace),
            ))
        if self.desktop_use_config.enabled:
            self.tools.register(DesktopUseTool(
                screenshot_delay=self.desktop_use_config.screenshot_delay,
                typing_delay_ms=self.desktop_use_config.typing_delay_ms,
                scaling_enabled=self.desktop_use_config.scaling_enabled,
                max_output_chars=self.desktop_use_config.max_output_chars,
                working_dir=str(self.workspace),
            ))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        self.tools.register(SpawnPipelineTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )

    def _build_read_only_tools(self) -> ToolRegistry:
        """Build a read-only tool registry for planning and verification."""
        tools = ToolRegistry()
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = self._extra_read_dirs(allowed_dir)
        tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read, resource_policy=self.resource_policy))
        tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read, resource_policy=self.resource_policy))
        if self.exec_config.enable:
            tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
                resource_policy=self.resource_policy,
                desktop_use_active=self.desktop_use_config.enabled,
            ))
        tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        tools.register(WebFetchTool(proxy=self.web_proxy))
        for tool in self.tools.iter_tools():
            if is_read_only_mcp_tool(tool):
                tools.register(tool)
        return tools

    def _build_planner_tools(self) -> ToolRegistry:
        """Build planner tools, including foreground explore delegation."""
        tools = self._build_read_only_tools()
        tools.register(ExploreTool(
            self.subagents,
            max_iterations=self.planner_explore_subagent_max_iterations,
        ))
        return tools

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(
        self,
        channel: str,
        chat_id: str,
        message_id: str | None = None,
        session_key: str | None = None,
    ) -> None:
        """Update context for all tools that need routing info."""
        effective_session_key = session_key or f"{channel}:{chat_id}"
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id, message_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id, effective_session_key)

        if pipeline_tool := self.tools.get("spawn_pipeline"):
            if isinstance(pipeline_tool, SpawnPipelineTool):
                pipeline_tool.set_context(channel, chat_id, effective_session_key)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

        for tool in self.tools.iter_tools():
            if hasattr(tool, "set_lock_owner"):
                tool.set_lock_owner(f"main:{effective_session_key}")

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from nanobot.utils.helpers import strip_think
        return strip_think(text) or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _find_legal_message_start(messages: list[dict[str, Any]]) -> int:
        """Find the first index where tool results have matching assistant tool_calls."""
        declared: set[str] = set()
        start = 0
        for index, message in enumerate(messages):
            role = message.get("role")
            if role == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    if isinstance(tool_call, dict) and tool_call.get("id"):
                        declared.add(str(tool_call["id"]))
            elif role == "tool":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id and str(tool_call_id) not in declared:
                    start = index + 1
                    declared.clear()
                    for previous in messages[start:index + 1]:
                        if previous.get("role") == "assistant":
                            for tool_call in previous.get("tool_calls") or []:
                                if isinstance(tool_call, dict) and tool_call.get("id"):
                                    declared.add(str(tool_call["id"]))
        return start

    @classmethod
    def _recent_legal_messages(
        cls,
        messages: list[dict[str, Any]],
        *,
        max_messages: int,
    ) -> list[dict[str, Any]]:
        """Return a recent suffix that does not start with orphaned tool results."""
        sliced = messages[-max_messages:] if max_messages > 0 else list(messages)
        start = cls._find_legal_message_start(sliced)
        return sliced[start:] if start else sliced

    @staticmethod
    def _flatten_unknown_tool_exchanges(
        messages: list[dict[str, Any]],
        known_tool_names: set[str],
    ) -> list[dict[str, Any]]:
        """Rewrite assistant/tool exchanges referencing unknown tools into plain text.

        When the verifier (or any read-only secondary pass) replays a session
        history that called tools not present in *its* tool registry, strict
        providers (notably the GitHub Copilot OpenAI-compatible proxy) reject
        the request with HTTP 400 because the history references tool names
        the model wasn't told about.

        This collapses any assistant message whose ``tool_calls`` reference at
        least one unknown tool into a plain assistant text message, and
        rewrites the matching ``tool`` result messages into ``user`` text
        notes with the same content. Known-tool exchanges are left intact so
        the verifier can still reason about them naturally.
        """
        if not messages:
            return messages

        # Pass 1: figure out which tool_call_ids belong to "unknown" assistant calls.
        unknown_tool_call_ids: set[str] = set()
        rewritten_assistant: dict[int, dict[str, Any]] = {}
        for idx, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                continue
            has_unknown = False
            names: list[str] = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                name = (tc.get("function") or {}).get("name") or tc.get("name") or "?"
                names.append(str(name))
                if name not in known_tool_names:
                    has_unknown = True
                    if tc.get("id"):
                        unknown_tool_call_ids.add(str(tc["id"]))
            if not has_unknown:
                continue
            # Build a narrative replacement for this assistant turn.
            original_content = msg.get("content")
            if isinstance(original_content, list):
                text_parts = [
                    b.get("text", "")
                    for b in original_content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                base_text = "\n".join(t for t in text_parts if t).strip()
            else:
                base_text = (original_content or "").strip() if isinstance(original_content, str) else ""
            summary = f"[Prior tool calls: {', '.join(names)}]"
            new_text = f"{base_text}\n\n{summary}".strip() if base_text else summary
            rewritten_assistant[idx] = {"role": "assistant", "content": new_text}

        if not rewritten_assistant and not unknown_tool_call_ids:
            return messages

        # Pass 2: emit rewritten history.
        result: list[dict[str, Any]] = []
        for idx, msg in enumerate(messages):
            if idx in rewritten_assistant:
                result.append(rewritten_assistant[idx])
                continue
            if msg.get("role") == "tool":
                tcid = msg.get("tool_call_id")
                if tcid and str(tcid) in unknown_tool_call_ids:
                    # Convert orphaned tool result into a plain user note so
                    # the verifier still sees the output without referencing
                    # an unknown tool schema. Preserve image_url blocks so the
                    # verifier can still see screenshots (image pruner upstream
                    # trims older ones).
                    raw_content = msg.get("content")
                    name = msg.get("name") or "tool"
                    new_blocks: list[Any] = []
                    if isinstance(raw_content, list):
                        text_parts: list[str] = []
                        image_blocks: list[dict[str, Any]] = []
                        for b in raw_content:
                            if not isinstance(b, dict):
                                continue
                            if b.get("type") == "text":
                                text_parts.append(b.get("text") or "")
                            elif b.get("type") == "image_url":
                                image_blocks.append(b)
                        text = "\n".join(t for t in text_parts if t).strip() or "(empty tool result)"
                        new_blocks.append({"type": "text", "text": f"[Result of prior {name} call]\n{text}"})
                        new_blocks.extend(image_blocks)
                    elif isinstance(raw_content, str):
                        new_blocks = [{"type": "text", "text": f"[Result of prior {name} call]\n{raw_content}"}]
                    else:
                        new_blocks = [{"type": "text", "text": f"[Result of prior {name} call]\n(empty tool result)"}]

                    if len(new_blocks) == 1 and new_blocks[0].get("type") == "text":
                        result.append({"role": "user", "content": new_blocks[0]["text"]})
                    else:
                        result.append({"role": "user", "content": new_blocks})
                    continue
            result.append(msg)
        return result

    @staticmethod
    def _summarize_history_for_verifier(
        messages: list[dict[str, Any]],
        *,
        max_image_blocks: int = 1,
        max_text_chars: int = 6000,
    ) -> list[dict[str, Any]]:
        """Collapse an action conversation into a single narrative user message.

        The verifier runs against strict providers (e.g. GitHub Copilot's
        OpenAI-compat proxy) that reject conversation histories whose
        ``tool_calls`` reference tool names not registered in the current
        request. Even when we try to flatten unknown-tool exchanges, the
        replayed history is large enough — and shaped specifically enough —
        that some proxies still return HTTP 400.

        To sidestep all of that, build a single ``user`` message that:

        * narrates each assistant text turn,
        * lists the tool calls that happened with a short args preview,
        * inlines short tool-result text,
        * keeps at most ``max_image_blocks`` of the most recent screenshots
          attached as ``image_url`` blocks (so vision-capable verifiers can
          still inspect the final state).

        Returns a list with zero or one message ready to splice after the
        verifier's system/user instructions.
        """
        if not messages:
            return []

        narrative_lines: list[str] = ["[Action-phase transcript follows]"]
        # Walk from oldest to newest, collecting text. Collect image blocks
        # separately and we'll keep the last N at the end.
        image_blocks: list[dict[str, Any]] = []

        def _truncate(text: str, limit: int = 500) -> str:
            text = (text or "").strip()
            if len(text) <= limit:
                return text
            return text[:limit] + "…"

        def _arg_preview(args: Any) -> str:
            if isinstance(args, str):
                return _truncate(args, 120)
            try:
                import json as _json
                return _truncate(_json.dumps(args, ensure_ascii=False), 120)
            except Exception:
                return _truncate(str(args), 120)

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                # Skip system messages; the verifier has its own.
                continue
            if role == "user":
                if isinstance(content, str) and content.strip():
                    narrative_lines.append(f"USER: {_truncate(content, 1500)}")
                elif isinstance(content, list):
                    for b in content:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") == "text" and b.get("text"):
                            narrative_lines.append(f"USER: {_truncate(b['text'], 1500)}")
                        elif b.get("type") == "image_url":
                            image_blocks.append(b)
            elif role == "assistant":
                text_part = ""
                if isinstance(content, str):
                    text_part = content
                elif isinstance(content, list):
                    text_part = "\n".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                text_part = text_part.strip()
                if text_part:
                    narrative_lines.append(f"ASSISTANT: {_truncate(text_part, 1500)}")
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    name = (tc.get("function") or {}).get("name") or tc.get("name") or "?"
                    raw_args = (tc.get("function") or {}).get("arguments") or tc.get("arguments")
                    narrative_lines.append(f"  → tool_call {name}({_arg_preview(raw_args)})")
            elif role == "tool":
                name = msg.get("name") or "tool"
                if isinstance(content, str):
                    narrative_lines.append(f"  ← {name} result: {_truncate(content, 600)}")
                elif isinstance(content, list):
                    text_parts: list[str] = []
                    for b in content:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") == "text" and b.get("text"):
                            text_parts.append(b["text"])
                        elif b.get("type") == "image_url":
                            image_blocks.append(b)
                    joined = "\n".join(text_parts).strip()
                    if joined:
                        narrative_lines.append(f"  ← {name} result: {_truncate(joined, 600)}")
                    else:
                        narrative_lines.append(f"  ← {name} result: [non-text content]")

        narrative = "\n".join(narrative_lines)
        if len(narrative) > max_text_chars:
            # Keep the tail (most recent activity is most relevant for verification).
            narrative = "[…truncated earlier history…]\n" + narrative[-max_text_chars:]

        # Keep only the most recent N image blocks.
        tail_images = image_blocks[-max_image_blocks:] if max_image_blocks > 0 else []
        if tail_images:
            blocks: list[dict[str, Any]] = [{"type": "text", "text": narrative}]
            blocks.extend(tail_images)
            return [{"role": "user", "content": blocks}]
        return [{"role": "user", "content": narrative}]


    @staticmethod
    def _is_affirmative(text: str) -> bool:
        return bool(re.fullmatch(r"\s*(yes|y|approve|approved|go ahead|continue|do it|run it)\s*[.!]?\s*", text, re.I))

    @staticmethod
    def _is_negative(text: str) -> bool:
        return bool(re.fullmatch(r"\s*(no|n|cancel|stop|don't|do not)\s*[.!]?\s*", text, re.I))

    def _clear_pending_approval(
        self,
        key: str,
        pending: _PendingApproval | None = None,
    ) -> None:
        """Clear a pending approval and any chat alias pointing at it."""
        pending = pending or self._pending_approvals.get(key)
        if pending is None:
            self._pending_approvals.pop(key, None)
            return
        for approval_key, approval in list(self._pending_approvals.items()):
            if approval is pending or approval.session_key == pending.session_key:
                self._pending_approvals.pop(approval_key, None)

    def _store_pending_approval(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        summary: str,
    ) -> None:
        """Store approval by task session and by the user-visible chat."""
        pending = _PendingApproval(
            summary=summary,
            created_at=time.time(),
            session_key=session_key,
        )
        self._pending_approvals[session_key] = pending
        visible_key = f"{channel}:{chat_id}"
        if visible_key != session_key:
            self._pending_approvals[visible_key] = pending

    @staticmethod
    def _should_mirror_task_session(session_key: str, visible_key: str) -> bool:
        return (
            session_key != visible_key
            and session_key.startswith(("cron:", "heartbeat"))
        )

    def _mirror_task_session_to_visible_chat(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        task_text: str,
        response_text: str,
        approval_granted: bool,
    ) -> None:
        """Mirror scheduled-task context into the chat users continue from."""
        visible_key = f"{channel}:{chat_id}"
        if not self._should_mirror_task_session(session_key, visible_key):
            return

        visible_session = self.sessions.get_or_create(visible_key)
        label = "Scheduled task approval" if approval_granted else "Scheduled task"
        task_snippet = task_text.strip()
        if len(task_snippet) > 4000:
            task_snippet = task_snippet[:4000].rstrip() + "\n... (truncated)"

        visible_session.add_message(
            "user",
            (
                f"[{label} from {session_key}]\n"
                f"{task_snippet}"
            ),
        )
        visible_session.add_message("assistant", response_text)
        self.sessions.save(visible_session)

    @staticmethod
    def _truncate_ledger_text(text: str, max_chars: int = 4000) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n... (truncated)"

    def _subagent_completion_record(self, event: dict[str, Any]) -> str:
        status = "completed" if event.get("status") == "ok" else "failed"
        notify = "yes" if event.get("notify") else "no"
        lines = [
            f"[Background subagent {status}]",
            f"Label: {event.get('label') or 'subagent'}",
            f"Task id: {event.get('task_id') or 'unknown'}",
            f"Notify user: {notify}",
            "",
            "Task:",
            self._truncate_ledger_text(str(event.get("task") or "")),
            "",
            "Result:",
            self._truncate_ledger_text(str(event.get("result") or "")),
        ]
        review_meta = event.get("review_meta")
        if isinstance(review_meta, dict):
            approved = "yes" if review_meta.get("approved") else "no"
            confidence = review_meta.get("confidence", "?")
            lines.extend(["", f"Review approved: {approved}", f"Review confidence: {confidence}%"])
            issues = review_meta.get("issues")
            if isinstance(issues, list) and issues:
                lines.append(f"Review issues: {'; '.join(str(issue) for issue in issues)}")
        return "\n".join(lines)

    async def _record_subagent_completion(self, event: dict[str, Any]) -> None:
        """Persist subagent outcome for the main agent's orchestration context."""
        origin = event.get("origin") if isinstance(event.get("origin"), dict) else {}
        channel = str(origin.get("channel") or "cli")
        chat_id = str(origin.get("chat_id") or "direct")
        session_key = str(event.get("session_key") or f"{channel}:{chat_id}")
        record = self._subagent_completion_record(event)

        self._record_session_ledger(
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            record=record,
        )

    def _record_session_ledger(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        record: str,
    ) -> None:
        """Persist an orchestration ledger entry and mirror task sessions to chat."""
        visible_key = f"{channel}:{chat_id}"
        session = self.sessions.get_or_create(session_key)
        session.add_message("user", record)
        self.sessions.save(session)

        if self._should_mirror_task_session(session_key, visible_key):
            visible_session = self.sessions.get_or_create(visible_key)
            visible_session.add_message("user", record)
            self.sessions.save(visible_session)

    def record_task_cancellation(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        active_tasks: int,
        subagent_tasks: int,
    ) -> None:
        total = active_tasks + subagent_tasks
        record = "\n".join([
            "[Background task cancelled]",
            f"Stopped tasks: {total}",
            f"Active main-agent tasks: {active_tasks}",
            f"Background subagent tasks: {subagent_tasks}",
        ])
        self._record_session_ledger(
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            record=record,
        )

    def record_task_failure(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        label: str,
        task: str,
        error: str,
    ) -> None:
        record = "\n".join([
            "[Background task failed]",
            f"Label: {label}",
            "",
            "Task:",
            self._truncate_ledger_text(task),
            "",
            "Error:",
            self._truncate_ledger_text(error),
        ])
        self._record_session_ledger(
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            record=record,
        )

    @staticmethod
    def _is_simple_conversation(text: str) -> bool:
        cleaned = text.strip().lower()
        if not cleaned:
            return True
        if len(cleaned) > 120:
            return False
        if re.search(r"\b(fix|update|implement|change|edit|write|create|add|remove|delete|refactor|debug|test|run|search|check|plan)\b", cleaned):
            return False
        return cleaned.endswith("?")

    def _should_plan(self, text: str) -> bool:
        return self._should_plan_with_mode(text, self.planning_mode)

    def _should_plan_with_mode(self, text: str, mode: str) -> bool:
        if mode == "off":
            return False
        if self._is_simple_conversation(text):
            return False
        if mode == "on":
            return True
        return bool(re.search(
            r"\b(fix|update|implement|change|edit|write|create|add|remove|delete|refactor|debug|test|run|search|check|review|plan)\b",
            text,
            re.I,
        ) or len(text.strip()) > 160)

    def _should_verify(self, text: str, tools_used: list[str], planned: _PlanDecision | None) -> bool:
        if self.planning_mode == "on" and not self._is_simple_conversation(text):
            return True
        return any(name in {"write_file", "edit_file", "exec", "spawn", "spawn_pipeline", "desktop_use"} for name in tools_used)

    def _tool_result_clear_thresholds(self) -> tuple[int | None, int | None]:
        """Return prompt-token thresholds for compacting stale tool results."""
        if self.max_tokens.input <= 0:
            return None, None
        budget = (
            self.max_tokens.input
            - self.max_tokens.output
            - self._TOOL_RESULT_CLEARING_SAFETY_BUFFER
        )
        if budget <= 0:
            budget = self.max_tokens.input
        if budget <= 0:
            return None, None
        trigger = int(budget * self._TOOL_RESULT_CLEAR_TRIGGER_RATIO)
        target = int(budget * self._TOOL_RESULT_CLEAR_TARGET_RATIO)
        return trigger or None, target or None

    def _channel_task_update_mode(self, channel: str) -> str:
        """Return the task update mode for chat channels.

        CLI intentionally keeps its current progress behavior regardless of
        channel reply mode settings.
        """
        if channel == "cli" or self.channels_config is None:
            return "verbose"
        return self.channels_config.task_update_mode

    @staticmethod
    def _planner_user_message(planned: _PlanDecision) -> str | None:
        """Build the short user-facing plan summary for plan_result mode."""
        summary = planned.action_summary.strip()
        if not summary:
            return None
        return f"Plan: {summary}"

    def _planner_prompt(self) -> str:
        planner_tool_names = sorted(self._build_planner_tools().tool_names)
        executor_only = sorted(set(self.tools.tool_names) - set(planner_tool_names))
        executor_hint = (
            ", ".join(executor_only) if executor_only else "(none — planner has direct access to all tools)"
        )
        return f"""# Internal Planner

You are nanobot's internal planner. Stay focused on coordination: gather enough evidence, draft the execution handoff, and sanity-check it before action begins. Do not modify files or external systems.

## Workspace
{self.workspace}

## Action-phase capabilities
You do NOT call these directly, but the action phase WILL have them in addition to your read-only tools. Choose `execute` whenever the user's request needs any of them — never refuse a request just because you cannot call the tool yourself:
{executor_hint}

## Workflow
Follow this internal loop until the handoff is ready or you run out of planner iterations:

1. Explore
- Use direct read-only tools for narrow targeted checks.
- Use the `explore` tool for broad or repeated investigation that would otherwise bloat your context.
- At most {self.planner_max_parallel_explore_agents} `explore` calls may run in parallel in one response. Use fewer when one is enough.
- Keep gathering evidence when key facts are still missing.

2. Plan
- Draft the concrete work the action phase should do.
- Draft a specific review goal that the verification phase can check.
- Build a references list with evidence-backed findings. Include the ACTUAL VALUES you gathered (file contents, command outputs) so the action phase can use them directly without re-reading.

3. Review
- Ask yourself whether the current handoff is ready for action and review without repeating the same research.
- If evidence is missing, go back to Explore.
- If evidence is enough but the plan is weak or vague, refine the handoff.
- Stop once action and review can proceed directly from your handoff.

## Decision Rules
- Use `answer` when no mutating or multi-step work is needed.
- Use `clarify` only when multiple plausible interpretations would change execution and inspection cannot resolve them.
- Use `execute` only when your inspected evidence is sufficient for action and review to proceed.
- Do not guess when the task depends on evidence you have not checked.

## Output
End your response with exactly:

---PLAN---
{{"decision":"answer_or_clarify_or_execute","response":"user-facing text when decision is answer/clarify","action_summary":"short concrete execution summary","review_goal":"what the review phase should verify","references":[{{"finding":"concise description of what was found","value":"the actual value gathered (file content, command output, etc.)","source":"tool_name: arguments or description of how this was gathered","references":["file: /abs/path/example.py","function: helper_name"],"open_question":"optional unresolved gap"}}]}}
---END---

For `execute`, `action_summary`, `review_goal`, and `references` are required.
IMPORTANT: Include `value` in each reference so the action phase can use the gathered data directly without re-reading files or re-running commands.
Keep every finding concise, concrete, and tied to exact references."""

    @staticmethod
    def _parse_plan_decision(result: str | None) -> _PlanDecision:
        if not result:
            return _PlanDecision()
        match = re.search(r"---PLAN---\s*\n(.*?)\n\s*---END---", result, re.DOTALL)
        if not match:
            return _PlanDecision()
        try:
            payload = json.loads(match.group(1).strip())
        except Exception:
            return _PlanDecision()
        decision = str(payload.get("decision") or "execute").strip().lower()
        if decision not in {"answer", "clarify", "execute"}:
            decision = "execute"
        references = AgentLoop._normalize_plan_references(payload.get("references"))
        return _PlanDecision(
            decision=decision,
            response=str(payload.get("response") or "").strip() or None,
            action_summary=str(payload.get("action_summary") or payload.get("summary") or "").strip(),
            review_goal=str(payload.get("review_goal") or "").strip(),
            references=references,
        )

    @staticmethod
    def _normalize_plan_references(payload: Any) -> list[dict[str, Any]]:
        """Normalize planner references into a stable list of finding entries."""
        if not isinstance(payload, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, str):
                finding = item.strip()
                if finding:
                    normalized.append({"finding": finding, "references": []})
                continue
            if not isinstance(item, dict):
                continue
            finding = str(item.get("finding") or "").strip()
            value = item.get("value")
            source = str(item.get("source") or "").strip() or None
            refs_raw = item.get("references") or []
            references = (
                [str(ref).strip() for ref in refs_raw if str(ref).strip()]
                if isinstance(refs_raw, list)
                else ([str(refs_raw).strip()] if str(refs_raw).strip() else [])
            )
            open_question = str(item.get("open_question") or "").strip() or None
            if not finding and not references and not open_question and value is None:
                continue
            entry: dict[str, Any] = {
                "finding": finding,
                "references": references,
            }
            if value is not None:
                entry["value"] = value
            if source:
                entry["source"] = source
            if open_question:
                entry["open_question"] = open_question
            normalized.append(entry)
        return normalized

    @staticmethod
    def _planner_handoff_message(planned: _PlanDecision) -> str:
        """Build the planner handoff injected into action/review phases."""
        parts = [
            "Internal planner handoff. The following data was gathered with read-only investigation tools.",
            "IMPORTANT: TRUST these values as ground truth. Use them directly without re-reading files or re-running commands. "
            "Only gather new information if the task requires data NOT listed in the references below."
        ]
        if planned.action_summary:
            parts.append(f"Action Summary:\n{planned.action_summary}")
        if planned.review_goal:
            parts.append(f"Review Goal:\n{planned.review_goal}")
        if planned.references:
            rendered: list[str] = []
            for idx, item in enumerate(planned.references, start=1):
                finding = item.get('finding') or '(unspecified)'
                value = item.get('value')
                source = item.get('source')
                entry = [f"{idx}. {finding}"]
                if value is not None:
                    entry.append(f"   Value: {value}")
                if source:
                    entry.append(f"   Source: {source}")
                refs = item.get("references") or []
                if refs:
                    entry.append("   References:")
                    entry.extend(f"   - {ref}" for ref in refs)
                if item.get("open_question"):
                    entry.append(f"   Open Question: {item['open_question']}")
                rendered.append("\n".join(entry))
            parts.append("Gathered Context:\n" + "\n".join(rendered))
        return "\n\n".join(parts)

    @staticmethod
    def _planner_verification_goal(task_text: str, planned: _PlanDecision | None) -> str:
        """Build the concrete review goal from the planner handoff."""
        if planned is None:
            return task_text

        parts: list[str] = []
        if planned.review_goal:
            parts.append(planned.review_goal)
        elif planned.action_summary:
            parts.append(planned.action_summary)
        else:
            parts.append(task_text)

        if planned.references:
            lines: list[str] = []
            for idx, item in enumerate(planned.references, start=1):
                finding = str(item.get("finding") or "").strip()
                value = item.get("value")
                source = item.get("source")
                refs = item.get("references") or []
                if finding:
                    lines.append(f"{idx}. {finding}")
                if value is not None:
                    lines.append(f"   - Value: {value}")
                if source:
                    lines.append(f"   - Source: {source}")
                for ref in refs:
                    lines.append(f"   - {ref}")
                if item.get("open_question"):
                    lines.append(f"   - Open question: {item['open_question']}")
            if lines:
                parts.append("Planner References:\n" + "\n".join(lines))

        return "\n\n".join(part for part in parts if part.strip())

    @staticmethod
    def _log_preview(text: str | None, limit: int = 120) -> str:
        """Render a short single-line preview for lifecycle logs."""
        if not text:
            return ""
        collapsed = re.sub(r"\s+", " ", text.strip())
        if len(collapsed) <= limit:
            return collapsed
        return collapsed[:limit] + "..."

    @classmethod
    def _serialize_planner_handoff(cls, task_text: str, planned: _PlanDecision) -> dict[str, Any]:
        """Persist a planner handoff in session metadata while the task is active."""
        return {
            "task_text": task_text,
            "decision": planned.decision,
            "response": planned.response,
            "action_summary": planned.action_summary,
            "review_goal": planned.review_goal,
            "references": planned.references,
        }

    @classmethod
    def _store_planner_handoff(cls, session: Session, task_text: str, planned: _PlanDecision) -> None:
        session.metadata[cls._PLANNER_HANDOFF_METADATA_KEY] = cls._serialize_planner_handoff(task_text, planned)

    @classmethod
    def _clear_planner_handoff(cls, session: Session) -> None:
        session.metadata.pop(cls._PLANNER_HANDOFF_METADATA_KEY, None)

    def _verifier_prompt(self, goal: str) -> str:
        return f"""# Internal Verifier

You are nanobot's internal verification pass.

## Goal
{goal}

## Rules
- You MUST NOT modify files, repositories, or external systems.
- Verify claims with concrete files, commands, or artifacts when possible.
- For video, timeline, or media-editing tasks, you MUST inspect the actual timeline/previews with read-only MCP tools when they are available.
- If a media-editing claim would require timeline/preview inspection and you cannot inspect it, do not return PASS.
- If work is incomplete or risky, say so directly.

## Output
End your response with exactly:

---VERIFY---
{{"verdict":"PASS_or_FAIL_or_PARTIAL","issues":["issue1"],"feedback":"brief actionable feedback"}}
---END---"""

    @staticmethod
    def _parse_verification_result(result: str | None) -> _VerificationResult:
        if not result:
            return _VerificationResult(verdict="PARTIAL", issues=["Verifier produced no output"], feedback="Verifier produced no output.")
        match = re.search(r"---VERIFY---\s*\n(.*?)\n\s*---END---", result, re.DOTALL)
        if not match:
            return _VerificationResult(verdict="PARTIAL", issues=["Verifier response was not structured"], feedback=result[-500:])
        try:
            payload = json.loads(match.group(1).strip())
        except Exception:
            return _VerificationResult(verdict="PARTIAL", issues=["Failed to parse verifier output"], feedback=result[-500:])
        verdict = str(payload.get("verdict") or "PARTIAL").strip().upper()
        if verdict not in {"PASS", "FAIL", "PARTIAL"}:
            verdict = "PARTIAL"
        issues = payload.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        return _VerificationResult(
            verdict=verdict,
            issues=[str(issue) for issue in issues],
            feedback=str(payload.get("feedback") or "").strip(),
        )

    async def _run_internal_planner(
        self,
        messages: list[dict[str, Any]],
        *,
        channel: str,
        chat_id: str,
    ) -> _PlanDecision:
        logger.info(
            "Planner phase started for {}:{} (max_iterations={})",
            channel,
            chat_id,
            self.planner_max_iterations,
        )
        planner_messages = [{"role": "system", "content": self._planner_prompt()}, *messages]
        result = await self._run_agent(
            planner_messages,
            tools=self._build_planner_tools(),
            max_iterations=self.planner_max_iterations,
            preserve_tool_results=True,
            channel=channel,
            chat_id=chat_id,
        )
        planned = self._parse_plan_decision(result.final_content)
        logger.info(
            "Planner phase completed for {}:{} with decision={} handoff={} stop_reason={}",
            channel,
            chat_id,
            planned.decision or "unknown",
            planned.has_handoff,
            result.stop_reason or "completed",
        )
        return planned

    async def _run_internal_verifier(
        self,
        messages: list[dict[str, Any]],
        *,
        goal: str,
        channel: str,
        chat_id: str,
    ) -> _VerificationResult:
        logger.info(
            "Verification phase started for {}:{} goal={}",
            channel,
            chat_id,
            self._log_preview(goal, limit=140),
        )
        recent_messages = self._recent_legal_messages(messages, max_messages=12)
        verifier_tools = self._build_read_only_tools()
        # Strict providers (e.g. GitHub Copilot's OpenAI-compat proxy) return
        # HTTP 400 when conversation history references tool names or tool_call
        # ids the model wasn't told about. Collapse the prior action transcript
        # into a single narrative user message so the verifier sees a clean,
        # schema-safe context with at most one recent screenshot attached.
        summarized = self._summarize_history_for_verifier(recent_messages)
        verify_messages = [
            {"role": "system", "content": self._verifier_prompt(goal)},
            {"role": "user", "content": (
                "Review the completed work against the goal below.\n\n"
                f"Goal:\n{goal}\n\n"
                "Read any changed files or artifacts and verify the result."
            )},
            *summarized,
        ]
        result = await self._run_agent(
            verify_messages,
            tools=verifier_tools,
            max_iterations=10,
            channel=channel,
            chat_id=chat_id,
        )
        verification = self._parse_verification_result(result.final_content)
        logger.info(
            "Verification phase completed for {}:{} verdict={} issues={}",
            channel,
            chat_id,
            verification.verdict,
            len(verification.issues),
        )
        return verification

    async def _run_agent(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        tools: ToolRegistry | None = None,
        tool_policy: RiskyActionPolicy | None = None,
        max_iterations: int | None = None,
        preserve_tool_results: bool = False,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        session_key: str | None = None,
    ) -> AgentRunResult:
        """Run a shared agent iteration loop and return the full result.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.
        """
        loop_hook = _LoopHook(
            self,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            session_key=session_key,
        )
        hook: AgentHook = (
            _LoopHookChain(loop_hook, self._extra_hooks)
            if self._extra_hooks
            else loop_hook
        )

        result = await self.runner.run(AgentRunSpec(
            initial_messages=initial_messages,
            tools=tools or self.tools,
            model=self.model,
            max_iterations=max_iterations or self.max_iterations,
            hook=hook,
            error_message="Sorry, I encountered an error calling the AI model.",
            concurrent_tools=True,
            tool_result_clearing_keep=(
                None if preserve_tool_results else self.tool_result_clearing_keep
            ),
            tool_result_clear_trigger_tokens=(
                None if preserve_tool_results else self._tool_result_clear_thresholds()[0]
            ),
            tool_result_clear_target_tokens=(
                None if preserve_tool_results else self._tool_result_clear_thresholds()[1]
            ),
            tool_policy=tool_policy,
        ))
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", max_iterations or self.max_iterations)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        tools: ToolRegistry | None = None,
        tool_policy: RiskyActionPolicy | None = None,
        max_iterations: int | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        session_key: str | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        result = await self._run_agent(
            initial_messages,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            tools=tools,
            tool_policy=tool_policy,
            max_iterations=max_iterations,
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            session_key=session_key,
        )
        return result.final_content, result.tools_used, result.messages

    async def _run_main_task(
        self,
        initial_messages: list[dict[str, Any]],
        *,
        task_text: str,
        session_key: str = "cli:direct",
        channel: str,
        chat_id: str,
        message_id: str | None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        approval_granted: bool = False,
        planned: _PlanDecision | None = None,
        skip_verification: bool = False,
    ) -> AgentRunResult:
        policy = RiskyActionPolicy(
            workspace=self.workspace,
            approval_granted=approval_granted,
            context_manager=self.context_manager,
        )
        verification_goal = self._planner_verification_goal(task_text, planned)
        logger.info(
            "Action phase started for {}:{} planned_handoff={} approval_granted={}",
            channel,
            chat_id,
            bool(planned and planned.has_handoff),
            approval_granted,
        )
        result = await self._run_agent(
            initial_messages,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            tool_policy=policy,
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            session_key=session_key,
        )
        if result.stop_reason == "approval_required":
            logger.info("Action phase paused for approval on {}:{}", channel, chat_id)
            return result

        if skip_verification or not self._should_verify(task_text, result.tools_used, planned):
            logger.info(
                "Verification skipped for {}:{} tools_used={} planned_handoff={}",
                channel,
                chat_id,
                len(result.tools_used),
                bool(planned and planned.has_handoff),
            )
            return result

        verification = await self._run_internal_verifier(
            result.messages,
            goal=verification_goal,
            channel=channel,
            chat_id=chat_id,
        )
        if verification.verdict == "PASS":
            logger.info("Action phase accepted after initial verification on {}:{}", channel, chat_id)
            return result

        logger.info(
            "Revision phase started for {}:{} after verifier verdict={}",
            channel,
            chat_id,
            verification.verdict,
        )
        revision_messages = list(result.messages)
        revision_messages.append({
            "role": "user",
            "content": (
                "[Internal verification feedback]\n"
                f"Verdict: {verification.verdict}\n"
                f"Issues: {'; '.join(verification.issues) or 'None'}\n"
                f"Feedback: {verification.feedback or 'Revise and verify your work carefully.'}\n\n"
                "Please revise the work and produce the final response."
            ),
        })
        revised = await self._run_agent(
            revision_messages,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            tool_policy=RiskyActionPolicy(workspace=self.workspace, approval_granted=True),
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            session_key=session_key,
        )
        if revised.stop_reason == "approval_required":
            return revised

        final_verification = await self._run_internal_verifier(
            revised.messages,
            goal=verification_goal,
            channel=channel,
            chat_id=chat_id,
        )
        if final_verification.verdict == "PASS":
            logger.info("Revision phase accepted after final verification on {}:{}", channel, chat_id)
            return revised

        logger.warning(
            "Revision phase ended with non-pass verification on {}:{} verdict={} issues={}",
            channel,
            chat_id,
            final_verification.verdict,
            len(final_verification.issues),
        )
        note = (
            "\n\nVerification status: "
            f"{final_verification.verdict}. "
            f"{final_verification.feedback or 'Some issues may remain.'}"
        )
        revised.final_content = (revised.final_content or "").rstrip() + note
        if revised.messages and revised.messages[-1].get("role") == "assistant" and not revised.messages[-1].get("tool_calls"):
            revised.messages[-1] = build_assistant_message(revised.final_content)
        else:
            revised.messages.append(build_assistant_message(revised.final_content))
        revised.stop_reason = "verification_partial"
        revised.policy_metadata = {
            **revised.policy_metadata,
            "verification": {
                "verdict": final_verification.verdict,
                "issues": final_verification.issues,
            },
        }
        return revised

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        # Connect MCP servers in background so agent accepts messages immediately
        asyncio.create_task(self._connect_mcp())
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                # Preserve real task cancellation so shutdown can complete cleanly.
                # Only ignore non-task CancelledError signals that may leak from integrations.
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, loop=self)
                result = await self.commands.dispatch_priority(ctx)
                if result:
                    await self.bus.publish_outbound(result)
                continue
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        lock = self._session_locks.setdefault(msg.session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()
        async with lock, gate:
            try:
                on_stream = on_stream_end = None
                stream_content_published = False
                if msg.metadata.get("_wants_stream"):
                    # Split one answer into distinct stream segments.
                    stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                    stream_segment = 0
                    buffered_stream = ""
                    stream_progress_enabled = True
                    defer_terminal_stream_until_completion = False
                    if msg.channel != "cli":
                        task_update_mode = self._channel_task_update_mode(msg.channel)
                        stream_progress_enabled = (task_update_mode == "verbose")
                        defer_terminal_stream_until_completion = (task_update_mode == "plan_result")

                    def _current_stream_id() -> str:
                        return f"{stream_base_id}:{stream_segment}"

                    async def _publish_stream_delta(delta: str) -> None:
                        nonlocal stream_content_published
                        meta = dict(msg.metadata or {})
                        meta["_stream_delta"] = True
                        meta["_stream_id"] = _current_stream_id()
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content=delta,
                            metadata=meta,
                        ))
                        if delta:
                            stream_content_published = True

                    async def _publish_stream_end(*, resuming: bool) -> None:
                        # Acknowledgement event lets in-process consumers (e.g., the
                        # CLI renderer) signal that they've finished writing this
                        # segment's terminating newline before we return and allow
                        # subsequent logger.* calls to hit stderr. Without this sync,
                        # log lines race ahead of the terminating newline and visually
                        # stick to the tail of the streamed response.
                        rendered_ack = asyncio.Event()
                        meta = dict(msg.metadata or {})
                        meta["_stream_end"] = True
                        meta["_resuming"] = resuming
                        meta["_stream_id"] = _current_stream_id()
                        meta["_stream_render_ack"] = rendered_ack
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="",
                            metadata=meta,
                        ))
                        # Bounded wait: external channels (slack/matrix/etc.) don't
                        # set the ack, so fall through on timeout rather than block
                        # the agent loop.
                        try:
                            await asyncio.wait_for(rendered_ack.wait(), timeout=0.2)
                        except asyncio.TimeoutError:
                            pass

                    async def on_stream(delta: str) -> None:
                        nonlocal buffered_stream
                        if stream_progress_enabled:
                            await _publish_stream_delta(delta)
                        else:
                            buffered_stream += delta

                    async def on_stream_end(*, resuming: bool = False) -> None:
                        nonlocal stream_segment, buffered_stream
                        if stream_progress_enabled:
                            await _publish_stream_end(resuming=resuming)
                            stream_segment += 1
                            return

                        if resuming:
                            buffered_stream = ""
                            stream_segment += 1
                            return

                        if defer_terminal_stream_until_completion:
                            stream_segment += 1
                            return

                        if buffered_stream:
                            await _publish_stream_delta(buffered_stream)
                            buffered_stream = ""
                            await _publish_stream_end(resuming=resuming)
                        stream_segment += 1

                response = await self._process_message(
                    msg, on_stream=on_stream, on_stream_end=on_stream_end,
                )
                if response is not None:
                    if response.metadata.get("_streamed") and not stream_content_published:
                        response.metadata = dict(response.metadata)
                        response.metadata.pop("_streamed", None)
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    def cancel_active_tasks(self) -> None:
        """Cancel all active message processing and background tasks."""
        for tasks in self._active_tasks.values():
            for task in tasks:
                if not task.done():
                    task.cancel()
        self._active_tasks.clear()
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        self._background_tasks.clear()

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        planning_mode_override: str | None = None,
        skip_verification: bool = False,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"), key)
            history = session.get_history(max_messages=0)
            current_role = "assistant" if msg.sender_id == "subagent" else "user"
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                current_role=current_role,
            )

            if self.max_tokens.input > 0:
                try:
                    tools = self.tools.get_definitions()
                    tokens, _ = estimate_prompt_tokens_chain(self.provider, self.model, messages, tools)
                    if tokens > self.max_tokens.input:
                        logger.warning(
                            "System context size ({}) exceeds maxTokens.input ({}). Trimming oldest turns.",
                            tokens,
                            self.max_tokens.input,
                        )
                        while tokens > self.max_tokens.input and history:
                            history.pop(0)
                            messages = self.context.build_messages(
                                history=history,
                                current_message=msg.content,
                                channel=channel,
                                chat_id=chat_id,
                                current_role=current_role,
                            )
                            tokens, _ = estimate_prompt_tokens_chain(
                                self.provider,
                                self.model,
                                messages,
                                tools,
                            )
                except Exception as e:
                    logger.error("Failed to check system token count: {}", e)

            final_content, _, all_msgs = await self._run_agent_loop(
                messages, channel=channel, chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
                session_key=key,
            )
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
            self._maybe_sync_context_repo()
            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=final_content or "Background task completed.",
            )

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        if msg.channel != "system" and self._PLANNER_HANDOFF_METADATA_KEY in session.metadata:
            logger.info("Clearing stale planner handoff for session {}", key)
            self._clear_planner_handoff(session)
            self.sessions.save(session)

        # Slash commands
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if result := await self.commands.dispatch(ctx):
            return result

        pending = self._pending_approvals.get(key)
        approval_note: str | None = None
        if pending:
            if self._is_affirmative(raw):
                if pending.session_key != key:
                    key = pending.session_key
                    session = self.sessions.get_or_create(key)
                approval_note = (
                    "The user approved the previously blocked risky action. "
                    f"Resume the task that required approval: {pending.summary}."
                )
                self._clear_pending_approval(key, pending)
            elif self._is_negative(raw):
                self._clear_pending_approval(key, pending)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Cancelled the pending risky action.",
                    metadata=msg.metadata or {},
                )
            else:
                # A new request supersedes the older pending approval.
                self._clear_pending_approval(key, pending)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"), key)
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        current_message = msg.content
        if approval_note:
            current_message = f"{approval_note}\n\nOriginal approval reply: {msg.content}"
        initial_messages = self.context.build_messages(
            history=history,
            current_message=current_message,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        tools = self.tools.get_definitions()
        clear_trigger_tokens, clear_target_tokens = self._tool_result_clear_thresholds()

        # Compact old tool results only when prompt size is approaching the budget.
        clear_old_tool_results(
            initial_messages,
            keep_last=self.tool_result_clearing_keep,
            provider=self.provider,
            model=self.model,
            tools=tools,
            trigger_tokens=clear_trigger_tokens,
            target_tokens=clear_target_tokens,
        )

        # Safety check: trim oldest turns if this specific request still exceeds maxTokens.input.
        if self.max_tokens.input > 0:
            try:
                tokens, _ = estimate_prompt_tokens_chain(
                    self.provider,
                    self.model,
                    initial_messages,
                    tools,
                )
                if tokens > self.max_tokens.input:
                    logger.warning(
                        "Context size ({}) exceeds maxTokens.input ({}). Trimming oldest turns.",
                        tokens,
                        self.max_tokens.input,
                    )
                    while tokens > self.max_tokens.input and history:
                        history.pop(0)
                        initial_messages = self.context.build_messages(
                            history=history,
                            current_message=current_message,
                            media=msg.media if msg.media else None,
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                        )
                        tokens, _ = estimate_prompt_tokens_chain(
                            self.provider,
                            self.model,
                            initial_messages,
                            tools,
                        )
            except Exception as e:
                logger.error("Failed to check token count: {}", e)

        planned: _PlanDecision | None = None
        effective_planning_mode = planning_mode_override if planning_mode_override is not None else self.planning_mode
        if self._should_plan_with_mode(current_message, effective_planning_mode):
            planned = await self._run_internal_planner(
                initial_messages,
                channel=msg.channel,
                chat_id=msg.chat_id,
            )
            if planned.decision in {"answer", "clarify"} and planned.response:
                all_msgs = [*initial_messages, build_assistant_message(planned.response)]
                self._save_turn(session, all_msgs, 1 + len(history))
                self.sessions.save(session)
                self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
                self._maybe_sync_context_repo()
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=planned.response,
                    metadata=msg.metadata or {},
                )
            if planned.decision == "execute" and planned.has_handoff:
                logger.info(
                    "Planner handoff stored for session {} with {} reference items",
                    key,
                    len(planned.references),
                )
                self._store_planner_handoff(session, current_message, planned)
                self.sessions.save(session)
                handoff_text = self._planner_handoff_message(planned)
                if initial_messages and initial_messages[0].get("role") == "system":
                    # Append to the existing system message instead of inserting a new one.
                    # Inserting a second system-role message causes the Anthropic provider to
                    # overwrite the first (it takes the last system message seen), which strips
                    # the agent's identity, workspace context, and tool knowledge.
                    existing = initial_messages[0].get("content") or ""
                    initial_messages[0] = {
                        **initial_messages[0],
                        "content": f"{existing}\n\n{handoff_text}",
                    }
                else:
                    initial_messages.insert(0, {"role": "system", "content": handoff_text})
                logger.info("Planner handoff injected into action context for session {}", key)
                if (
                    approval_note is None
                    and self._channel_task_update_mode(msg.channel) == "plan_result"
                ):
                    plan_message = self._planner_user_message(planned)
                    if plan_message:
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=plan_message,
                            metadata={
                                **dict(msg.metadata or {}),
                                "_intermediate": True,
                            },
                        ))

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        result = await self._run_main_task(
            initial_messages,
            task_text=current_message,
            session_key=key,
            on_progress=on_progress or _bus_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=msg.channel, chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
            approval_granted=approval_note is not None,
            planned=planned,
            skip_verification=skip_verification,
        )
        final_content = result.final_content
        all_msgs = result.messages

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        if planned and planned.decision == "execute":
            logger.info("Clearing planner handoff after action/review cycle for session {}", key)
            self._clear_planner_handoff(session)
        self.sessions.save(session)
        self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
        self._maybe_sync_context_repo()

        if result.stop_reason == "approval_required":
            summary = str(result.policy_metadata.get("summary") or "risky action")
            self._store_pending_approval(
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                summary=summary,
            )

        self._mirror_task_session_to_visible_chat(
            session_key=key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            task_text=current_message,
            response_text=final_content,
            approval_granted=approval_note is not None,
        )

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if on_stream is not None and result.stop_reason != "approval_required":
            meta["_streamed"] = True
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=meta,
        )

    @staticmethod
    def _image_placeholder(block: dict[str, Any]) -> dict[str, str]:
        """Convert an inline image block into a compact text placeholder."""
        path = (block.get("_meta") or {}).get("path", "")
        return {"type": "text", "text": f"[image: {path}]" if path else "[image]"}

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if (
                block.get("type") == "image_url"
                and block.get("image_url", {}).get("url", "").startswith("data:image/")
            ):
                filtered.append(self._image_placeholder(block))
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if truncate_text and len(text) > self._TOOL_RESULT_MAX_CHARS:
                    text = text[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "system":
                continue
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(entry.get("content"), str) and entry["content"].startswith("[Internal verification feedback]"):
                    continue
                if (
                    isinstance(entry.get("content"), str)
                    and "Original approval reply:" in entry["content"]
                    and entry["content"].startswith("The user approved the previously blocked risky action.")
                ):
                    entry["content"] = entry["content"].split("Original approval reply:", 1)[1].strip()
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session: Session, archive_all: bool = False) -> bool:
        """Compatibility shim for background consolidation paths."""
        if archive_all:
            chunk = session.messages[session.last_consolidated:]
            if not chunk:
                return True
            archived = await self.memory_consolidator.archive_messages(chunk)
            if archived:
                session.last_consolidated = len(session.messages)
                self.sessions.save(session)
            return archived
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)
        return True

    def _maybe_sync_context_repo(self) -> None:
        """If context repos are configured, schedule a background git sync for each."""
        repos = [repo for repo in self.context_manager.repos if repo.auto_sync]
        if not repos:
            return
        from nanobot.utils.git_sync import async_sync_context_repo

        async def _sync() -> None:
            try:
                await asyncio.gather(
                    *(
                        async_sync_context_repo(
                            repo.path,
                            include_paths=repo.sync_include_patterns(),
                            exclude_paths=repo.sync_exclude_patterns(),
                            message=f"nanobot: sync {repo.name} context updates",
                        )
                        for repo in repos
                    ),
                    return_exceptions=False
                )
            except Exception:
                logger.exception("Context repo sync failed")

        self._schedule_background(_sync())

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        planning_mode: str | None = None,
        skip_verification: bool = False,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        return await self._process_message(
            msg, session_key=session_key, on_progress=on_progress,
            on_stream=on_stream, on_stream_end=on_stream_end,
            planning_mode_override=planning_mode,
            skip_verification=skip_verification,
        )
