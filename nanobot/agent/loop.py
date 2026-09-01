"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import warnings
from contextlib import AsyncExitStack, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.context_budget import ContextBudget, ContextBudgetManager
from nanobot.agent.delegation import (
    ForegroundAgentManager,
    reset_current_turn_context,
    set_current_turn_context,
)
from nanobot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.policy import RiskyActionPolicy
from nanobot.agent.run_store import RunStore
from nanobot.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.tools.agent_browser import AgentBrowserTool
from nanobot.agent.tools.agent_device import AgentDeviceTool
from nanobot.agent.tools.crawler import CrawlResearchTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.delegation import DelegateTaskTool, PlanTaskTool, ReviewWorkTool
from nanobot.agent.tools.desktop_use import DesktopUseTool
from nanobot.agent.tools.explore import ExploreTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.image import (
    ImageGenerationTool,
    image_generation_available,
)
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.social_crawl import crawl_tools_enabled
from nanobot.agent.tools.trend_vpn import (
    TrendVpnBrowserFetchTool,
    TrendVpnFetchTool,
    TrendVpnSessionCloseTool,
    TrendVpnSessionStartTool,
    vpn_tools_enabled,
)
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.agent.turn import (
    ApprovalGrant,
    DeliveryState,
    DeliveryTarget,
    HistoryMode,
    RunPolicyContext,
    RunRecord,
    RunStatus,
    SessionRunRef,
    ToolOutcome,
    TurnCallbacks,
    TurnContext,
    TurnRequest,
    TurnResult,
    TurnSource,
)
from nanobot.agent.write_guard import FileLockRegistry
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.prompt_budget import PromptBudgetExceeded

if TYPE_CHECKING:
    from nanobot.config.schema import (
        AgentBrowserConfig,
        AgentDeviceConfig,
        ChannelsConfig,
        CrawlerAgentConfig,
        DesktopUseConfig,
        ExecToolConfig,
        ImageConfig,
        MaxTokensConfig,
        WebSearchConfig,
    )
    from nanobot.cron.service import CronService


class _TurnStopError(Exception):
    """Internal signal used to stop the runner before its next model call."""

    def __init__(
        self,
        stop_reason: str,
        content: Any = None,
        policy_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(stop_reason)
        self.stop_reason = stop_reason
        self.content = content
        self.policy_metadata = policy_metadata or {}


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
        turn_context: TurnContext | None = None,
    ) -> None:
        self._loop = agent_loop
        self._on_progress = on_progress
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._channel = channel
        self._chat_id = chat_id
        self._message_id = message_id
        self._session_key = session_key
        self._turn_context = turn_context
        self._stream_buf = ""

    def wants_streaming(self) -> bool:
        return self._on_stream is not None

    async def before_iteration(self, context: AgentHookContext) -> None:
        if self._turn_context is None:
            return
        stop_reason = self._turn_context.metadata.get("_tool_stop_reason")
        if stop_reason:
            raise _TurnStopError(
                str(stop_reason),
                self._turn_context.metadata.get("_tool_stop_content"),
                dict(self._turn_context.metadata.get("_tool_policy_metadata") or {}),
            )

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

    async def after_iteration(self, context: AgentHookContext) -> None:
        if self._turn_context is not None:
            self._turn_context.messages = context.messages
            if context.tool_calls:
                self._turn_context.tools_used.extend(
                    tool_call.name for tool_call in context.tool_calls
                )
            if context.tool_results:
                self._turn_context.tool_results.extend(context.tool_results)
            self._turn_context.metadata["_turn_usage"] = dict(context.usage)
            self._turn_context.metadata["_turn_tool_events"] = [
                *self._turn_context.metadata.get("_turn_tool_events", []),
                *context.tool_events,
            ]
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


class _ContextualToolRegistry:
    """Registry-shaped adapter binding one runner invocation to one turn.

    ``AgentRunner`` intentionally remains product-agnostic and asks its
    registry for bare model-facing values.  This adapter invokes the WP1
    contextual registry API, records any terminal policy envelope on the
    ``TurnContext``, and unwraps only the JSON-safe content sent back to the
    model.  No per-turn routing state is written to shared tool instances.
    """

    def __init__(self, registry: ToolRegistry, context: TurnContext) -> None:
        self._registry = registry
        self.context = context

    def get_definitions(self) -> list[dict[str, Any]]:
        return self._registry.get_definitions()

    def get(self, name: str) -> Any:
        return self._registry.get(name)

    def iter_tools(self):
        return self._registry.iter_tools()

    @property
    def tool_names(self) -> list[str]:
        return self._registry.tool_names

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        outcome = await self._registry.execute(name, params, context=self.context)
        if isinstance(outcome, ToolOutcome):
            if outcome.stop_reason in {
                RunStatus.APPROVAL_REQUIRED.value,
                RunStatus.POLICY_BLOCKED.value,
                RunStatus.TOOL_ERROR.value,
                RunStatus.CANCELLED.value,
            }:
                self.context.metadata["_tool_stop_reason"] = outcome.stop_reason
                self.context.metadata["_tool_stop_content"] = outcome.content
                self.context.metadata["_tool_policy_metadata"] = dict(outcome.policy_metadata)
            # Preserve the shared envelope for AgentRunner.  It owns the
            # product-neutral terminal/skip behavior and must see the first
            # stop before a later outer tool can start.
            return outcome
        return ToolOutcome(content=outcome)

    async def execute_with_context(self, name: str, params: dict[str, Any]) -> ToolOutcome:
        outcome = await self._registry.execute(name, params, context=self.context)
        return outcome if isinstance(outcome, ToolOutcome) else ToolOutcome(content=outcome)


def target_message_id(target: DeliveryTarget | None) -> str | None:
    """Return a message identifier from a normalized delivery target."""
    return target.message_id if target is not None else None


@dataclass(slots=True)
class _PendingApproval:
    """Pending risky action awaiting an explicit yes/no reply."""

    summary: str
    created_at: float
    session_key: str


@dataclass(frozen=True, slots=True)
class _ApprovalSnapshot:
    """Approval state observed atomically before choosing the detail lock."""

    alias_key: str
    pending: _PendingApproval | None
    action: str
    effective_key: str


_STALE_APPROVAL_MESSAGE = (
    "That approval request is no longer current. "
    "Please reply to the latest approval request."
)


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
    _DEFAULT_EXPLORE_MAX_ITERATIONS = 100
    _DEFAULT_MAX_PARALLEL_EXPLORES = 2

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
        tool_result_clearing_keep: int = 3,
        consolidation_trigger_ratio: float = 0.5,
        consolidation_target_ratio: float = 0.3,
        crawler_agent_config: "CrawlerAgentConfig | None" = None,
        crawler_provider: LLMProvider | None = None,
    ):
        from nanobot.config.schema import (
            AgentBrowserConfig,
            AgentDeviceConfig,
            CrawlerAgentConfig,
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
        self.crawler_agent_config = crawler_agent_config or CrawlerAgentConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []
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
        )
        self.context_paths = self.context_manager.paths
        self.sessions = session_manager or SessionManager(workspace)
        self.run_store = RunStore(workspace, session_manager=self.sessions)
        self.tools = ToolRegistry()
        self.runner = AgentRunner(provider)

        # Unified context budget management
        # Always created when max_tokens.input is set. Handles all context reduction:
        # - Compact old tool results
        # - Drop old turn pairs
        # - Compress memory ([Past Knowledge] message)
        self.budget_manager: ContextBudgetManager | None = None
        if self.max_tokens.input > 0:
            self.budget_manager = ContextBudgetManager(
                budget=ContextBudget(
                    max_tokens=self.max_tokens.input,
                    output_reserve=self.max_tokens.output,
                ),
                memory_store=self.context.memory,
                provider=provider,
                model=self.model,
                tool_registry=self.tools,
            )
            logger.info(
                "Context budget manager initialized: max_tokens={}, available_budget={}",
                self.budget_manager.budget.max_tokens,
                self.budget_manager.budget.available_budget,
            )
        self._file_lock_registry = FileLockRegistry()
        self.delegation = ForegroundAgentManager(
            provider=provider,
            workspace=workspace,
            model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            agent_browser_config=self.agent_browser_config,
            agent_device_config=self.agent_device_config,
            exec_config=self.exec_config,
            context_paths=self.context_paths,
            context_manager=self.context_manager,
            restrict_to_workspace=restrict_to_workspace,
            mcp_servers=mcp_servers or {},
            max_parallel_explore_agents=self._DEFAULT_MAX_PARALLEL_EXPLORES,
            crawler_provider=crawler_provider,
            crawler_model=self.crawler_agent_config.model,
            crawler_max_iterations=self.crawler_agent_config.max_tool_iterations,
            crawler_max_input_tokens=self.crawler_agent_config.max_tokens.input,
            crawler_max_output_tokens=self.crawler_agent_config.max_tokens.output,
            crawler_reasoning_effort=self.crawler_agent_config.reasoning_effort,
            file_lock_registry=self._file_lock_registry,
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
        self._approval_guard = asyncio.Lock()
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
        if vpn_tools_enabled():
            self.tools.register(TrendVpnSessionStartTool())
            self.tools.register(TrendVpnFetchTool())
            self.tools.register(TrendVpnBrowserFetchTool())
            self.tools.register(TrendVpnSessionCloseTool())
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
                humanize_typing=self.desktop_use_config.humanize_typing,
            ))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(PlanTaskTool(manager=self.delegation))
        self.tools.register(DelegateTaskTool(manager=self.delegation))
        self.tools.register(ReviewWorkTool(manager=self.delegation))
        self.tools.register(ExploreTool(
            self.delegation,
            max_iterations=self._DEFAULT_EXPLORE_MAX_ITERATIONS,
        ))
        if self.crawler_agent_config.enabled and crawl_tools_enabled():
            self.tools.register(CrawlResearchTool(manager=self.delegation))
        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )

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

    def _channel_task_update_mode(self, channel: str) -> str:
        """Return the configured delivery mode for chat-channel task updates."""
        if channel == "cli" or self.channels_config is None:
            return "verbose"
        return self.channels_config.task_update_mode

    @staticmethod
    def _is_affirmative(text: str) -> bool:
        return bool(re.fullmatch(r"\s*(yes|y|approve|approved|go ahead|continue|do it|run it)\s*[.!]?\s*", text, re.I))

    @staticmethod
    def _is_negative(text: str) -> bool:
        return bool(re.fullmatch(r"\s*(no|n|cancel|stop|don't|do not)\s*[.!]?\s*", text, re.I))

    async def _snapshot_approval(self, key: str, content: str) -> _ApprovalSnapshot:
        """Snapshot and classify a pending approval without holding session locks."""
        async with self._approval_guard:
            pending = self._pending_approvals.get(key)
            if pending is None:
                return _ApprovalSnapshot(
                    alias_key=key,
                    pending=None,
                    action="none",
                    effective_key=key,
                )

            raw = content.strip()
            if self._is_affirmative(raw):
                action = "affirmative"
                effective_key = pending.session_key
            elif self._is_negative(raw):
                action = "negative"
                effective_key = key
            elif raw.startswith("/"):
                # Slash commands are routed independently and must not consume
                # an approval merely because one is pending in this chat.
                action = "command"
                effective_key = key
            else:
                action = "other"
                effective_key = key

            return _ApprovalSnapshot(
                alias_key=key,
                pending=pending,
                action=action,
                effective_key=effective_key,
            )

    def _remove_pending_approval_identity_locked(self, pending: _PendingApproval) -> bool:
        """Remove only aliases whose value is exactly ``pending``.

        Callers must already hold ``_approval_guard``.  Comparing object
        identity is deliberate: a newer approval may reuse the same detail
        session key while replacing the visible-chat alias.
        """
        removed = False
        for approval_key, approval in list(self._pending_approvals.items()):
            if approval is pending:
                self._pending_approvals.pop(approval_key, None)
                removed = True
        return removed

    async def _clear_pending_approval(
        self,
        key: str,
        pending: _PendingApproval | None = None,
    ) -> bool:
        """Clear the current approval alias and all aliases of that object."""
        async with self._approval_guard:
            candidate = pending or self._pending_approvals.get(key)
            if candidate is None or self._pending_approvals.get(key) is not candidate:
                return False
            return self._remove_pending_approval_identity_locked(candidate)

    def _store_pending_approval_locked(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        summary: str,
    ) -> _PendingApproval:
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
        return pending

    async def _store_pending_approval(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        summary: str,
    ) -> _PendingApproval:
        """Store an approval under the short-lived approval-map guard."""
        async with self._approval_guard:
            return self._store_pending_approval_locked(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                summary=summary,
            )

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
        run_id: str | None = None,
        status: str = "completed",
        stop_reason: str | None = None,
        occurrence_id: str | None = None,
    ) -> None:
        """Record bounded scheduled-task metadata for the chat users continue from."""
        visible_key = f"{channel}:{chat_id}"
        if not self._should_mirror_task_session(session_key, visible_key):
            return

        # Scheduled execution has its complete trace in the detail session.
        # Keep the visible chat's model-facing history bounded by storing only
        # a compact run reference and excerpts in the metadata ring.  The
        # compatibility ``get_history`` view can still synthesize the old
        # instruction/result pair for trace readers without persisting another
        # full turn in the visible JSONL.
        visible_session = self.sessions.get_or_create(visible_key)
        detail_id = session_key.split(":", 1)[1] if ":" in session_key else session_key
        visible_session.add_scheduled_run(
            job_id=detail_id,
            run_id=run_id or f"legacy:{detail_id}",
            instruction=task_text,
            result=response_text,
            detail_session_key=session_key,
            status=status,
            approval_granted=approval_granted,
            stop_reason=stop_reason,
            occurrence_id=occurrence_id,
        )
        self.sessions.save(visible_session)

    @staticmethod
    def _truncate_ledger_text(text: str, max_chars: int = 4000) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n... (truncated)"

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
    ) -> None:
        record = "\n".join([
            "[Task cancelled]",
            f"Stopped tasks: {active_tasks}",
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
        turn_context: TurnContext | None = None,
        extra_hooks: list[AgentHook] | tuple[AgentHook, ...] | None = None,
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
            turn_context=turn_context,
        )
        all_extra_hooks = [*self._extra_hooks, *(extra_hooks or [])]
        hook: AgentHook = (
            _LoopHookChain(loop_hook, all_extra_hooks)
            if all_extra_hooks
            else loop_hook
        )

        base_tools = tools if tools is not None else self.tools
        runner_tools = (
            _ContextualToolRegistry(base_tools, turn_context)
            if turn_context is not None
            else base_tools
        )
        try:
            result = await self.runner.run(AgentRunSpec(
                initial_messages=initial_messages,
                tools=runner_tools,
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
                max_input_tokens=self.max_tokens.input if self.max_tokens.input > 0 else None,
                budget_manager=self.budget_manager,  # Pass unified budget manager for mid-loop enforcement
            ))
        except _TurnStopError as stop:
            context_messages = turn_context.messages if turn_context is not None else initial_messages
            context_tools = turn_context.tools_used if turn_context is not None else []
            context_usage = (
                turn_context.metadata.get("_turn_usage", {})
                if turn_context is not None
                else {}
            )
            context_events = (
                turn_context.metadata.get("_turn_tool_events", [])
                if turn_context is not None
                else []
            )
            result = AgentRunResult(
                final_content=(
                    stop.content
                    if isinstance(stop.content, str)
                    else str(stop.content or "")
                ),
                messages=context_messages,
                tools_used=list(context_tools),
                usage=dict(context_usage),
                stop_reason=stop.stop_reason,
                tool_events=list(context_events),
                policy_metadata=dict(stop.policy_metadata),
            )
        self._last_usage = result.usage
        if turn_context is not None:
            turn_context.messages = result.messages
            turn_context.tools_used = list(result.tools_used)
            turn_context.tool_results = list(result.tool_events)
            marker = turn_context.metadata.get("_tool_stop_reason")
            if marker and result.stop_reason in {"completed", "max_iterations"}:
                result.stop_reason = str(marker)
                stop_content = turn_context.metadata.get("_tool_stop_content")
                if stop_content is not None:
                    result.final_content = (
                        stop_content
                        if isinstance(stop_content, str)
                        else str(stop_content)
                    )
                result.policy_metadata = dict(
                    turn_context.metadata.get("_tool_policy_metadata") or {}
                )
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
        turn_context: TurnContext | None = None,
        extra_hooks: list[AgentHook] | tuple[AgentHook, ...] | None = None,
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
            turn_context=turn_context,
            extra_hooks=extra_hooks,
        )
        return result.final_content, result.tools_used, result.messages

    async def _run_main_task(
        self,
        initial_messages: list[dict[str, Any]],
        *,
        session_key: str = "cli:direct",
        channel: str,
        chat_id: str,
        message_id: str | None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        approval_granted: bool = False,
        tools: ToolRegistry | None = None,
        turn_context: TurnContext | None = None,
        extra_hooks: list[AgentHook] | tuple[AgentHook, ...] | None = None,
    ) -> AgentRunResult:
        """Run the single main-agent orchestrator loop."""
        policy = RiskyActionPolicy(
            workspace=self.workspace,
            approval_granted=approval_granted,
            context_manager=self.context_manager,
        )
        logger.info(
            "Orchestrator run started for {}:{} approval_granted={}",
            channel,
            chat_id,
            approval_granted,
        )
        return await self._run_agent(
            initial_messages,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            tools=tools,
            tool_policy=policy,
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            session_key=session_key,
            turn_context=turn_context,
            extra_hooks=extra_hooks,
        )

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        # Connect MCP servers in background so agent accepts messages immediately
        asyncio.create_task(self._connect_mcp())
        # Start keep-alive ping for local providers (if configured)
        if hasattr(self.provider, "start_keep_alive"):
            self.provider.start_keep_alive()
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
                    if msg.channel == "cli":
                        result.metadata = dict(result.metadata or {})
                        result.metadata["_cli_turn_complete"] = True
                    await self.bus.publish_outbound(result)
                continue
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        # ``execute_turn`` owns both the authoritative session lock and the
        # cross-session concurrency gate.  Keep this adapter's indentation and
        # publication behavior while avoiding a second lock around the runner.
        # A direct test/integration replacement of ``_process_message`` predates
        # the canonical adapter and still needs the compatibility lock.
        process_impl = getattr(self._process_message, "__func__", None)
        if process_impl is AgentLoop._process_message:
            dispatch_lock = nullcontext()
            dispatch_gate = nullcontext()
        else:
            dispatch_lock = self._session_locks.setdefault(msg.session_key, asyncio.Lock())
            dispatch_gate = self._concurrency_gate or nullcontext()
        async with dispatch_lock, dispatch_gate:
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
                        # Result mode exposes only the completed turn.
                        defer_terminal_stream_until_completion = task_update_mode == "result"

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
                    if msg.channel == "cli":
                        response.metadata = dict(response.metadata or {})
                        response.metadata["_cli_turn_complete"] = True
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    meta = dict(msg.metadata or {})
                    meta["_cli_turn_complete"] = True
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=meta,
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                meta = {"_cli_turn_complete": True} if msg.channel == "cli" else {}
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                    metadata=meta,
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
        await self.delegation.close()

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(lambda t: self._background_tasks.remove(t) if t in self._background_tasks else None)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    def cancel_active_tasks(self) -> None:
        """Cancel all active message processing and background tasks."""
        # Stop keep-alive ping first
        if hasattr(self.provider, "stop_keep_alive"):
            self.provider.stop_keep_alive()
        for tasks in self._active_tasks.values():
            for task in tasks:
                if not task.done():
                    task.cancel()
        self._active_tasks.clear()
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        self._background_tasks.clear()

    @staticmethod
    def _request_route(request: TurnRequest) -> tuple[str, str, str, DeliveryTarget]:
        """Resolve one request's session key and visible delivery target."""
        target = request.delivery_target or request.route
        channel = target.channel if target is not None else ""
        chat_id = (
            target.chat_id or target.to or target.recipient
            if target is not None
            else ""
        )
        key = request.session_key
        if not key:
            key = f"{channel}:{chat_id}" if channel and chat_id else "cli:direct"
        if not channel or not chat_id:
            inferred_channel, separator, inferred_chat = key.partition(":")
            channel = inferred_channel if separator else "cli"
            chat_id = inferred_chat if separator else key
        if target is None or not target.channel or not (
            target.chat_id or target.to or target.recipient
        ):
            target = DeliveryTarget(
                channel=channel,
                chat_id=chat_id,
                message_id=target.message_id if target is not None else None,
            )
        return key, channel, chat_id, target

    @staticmethod
    def _new_run_id(requested: str | None) -> str:
        if requested and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", requested):
            return requested
        import uuid

        return uuid.uuid4().hex

    @staticmethod
    def _run_status(stop_reason: str | None) -> RunStatus:
        return {
            "completed": RunStatus.COMPLETED,
            "approval_required": RunStatus.APPROVAL_REQUIRED,
            "policy_blocked": RunStatus.POLICY_BLOCKED,
            "tool_error": RunStatus.TOOL_ERROR,
            "max_iterations": RunStatus.MAX_ITERATIONS,
            "cancelled": RunStatus.CANCELLED,
            "error": RunStatus.ERROR,
        }.get(stop_reason or "completed", RunStatus.COMPLETED)

    def _finalize_run_record(
        self,
        record: RunRecord,
        context: TurnContext,
        result: TurnResult,
        *,
        status: RunStatus | None = None,
    ) -> RunRecord:
        """Persist a terminal, sanitized record and attach it to the result."""
        record.status = status or result.status
        record.updated_at = datetime.now(timezone.utc).isoformat()
        record.completed_at = datetime.now(timezone.utc).isoformat()
        record.stop_reason = result.stop_reason
        record.tools_used = tuple(result.tools_used or context.tools_used)
        record.usage = dict(result.usage or {})
        if result.error:
            record.error_type = "error"
        if context.session_key and (
            record.session_ref is None or record.session_ref.session_key != context.session_key
        ):
            record.session_ref = SessionRunRef(session_key=context.session_key, run_id=record.run_id)
            record.detail_ref = record.session_ref
            record.session_key = context.session_key
        record = self.run_store.save(record)
        result.record = record
        return record

    async def execute_turn(
        self,
        request: TurnRequest,
        *,
        _tools: ToolRegistry | None = None,
    ) -> TurnResult:
        """Execute exactly one canonical, serialized agent turn."""
        if not isinstance(request, TurnRequest):
            raise TypeError("execute_turn expects a TurnRequest")

        requested_key, channel, chat_id, target = self._request_route(request)
        approval = await self._snapshot_approval(requested_key, request.content)
        key = approval.effective_key
        run_id = self._new_run_id(request.run_id)
        started_at = datetime.now(timezone.utc)
        context = TurnContext(
            request=request,
            run_id=run_id,
            started_at=started_at,
            session_key=key,
            delegation_budget=self.delegation.new_budget(),
            policy=RunPolicyContext(
                approval_grant=request.approval_grant or (
                    request.approval if request.approval.granted else None
                ),
                workspace=self.workspace,
                approval_granted=bool(request.approval.granted),
                source=request.source,
                context_manager=self.context_manager,
            ),
            delivery=DeliveryState(primary=target, target=target),
            lock_owner=f"main:{key}",
            metadata=dict(request.metadata or {}),
        )
        # ``TurnRequest`` carries approval under both compatibility names; the
        # explicit grant wins when present.
        if request.approval_grant is not None:
            context.policy.approval_granted = bool(request.approval_grant.granted)
        elif request.approval.granted:
            context.policy.approval_granted = True

        record = RunRecord(
            run_id=run_id,
            status=RunStatus.RUNNING,
            source=request.source,
            detail_ref=SessionRunRef(session_key=key, run_id=run_id),
            session_ref=SessionRunRef(session_key=key, run_id=run_id),
            visible_session_key=request.scheduled_link.visible_session_key
            if request.scheduled_link is not None and request.scheduled_link.visible_session_key
            else f"{channel}:{chat_id}",
            resumed_run_id=(
                request.approval_grant.resumed_run_id
                if request.approval_grant is not None
                else None
            ),
            started_at=started_at.isoformat(),
            updated_at=started_at.isoformat(),
            metadata=dict(request.metadata or {}),
            delivery_target=target,
            scheduled_link=request.scheduled_link,
        )
        self.run_store.save(record)

        lock = self._session_locks.setdefault(key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()
        token = set_current_turn_context(context)
        result: TurnResult | None = None
        try:
            async with lock, gate:
                if approval.pending is not None and approval.action in {
                    "affirmative",
                    "negative",
                    "other",
                }:
                    async with self._approval_guard:
                        current = self._pending_approvals.get(approval.alias_key)
                        if current is not approval.pending:
                            result = TurnResult(
                                run_id=run_id,
                                status=RunStatus.CANCELLED,
                                content=_STALE_APPROVAL_MESSAGE,
                                final_content=_STALE_APPROVAL_MESSAGE,
                                stop_reason="approval_stale",
                                outbound=OutboundMessage(
                                    channel=channel,
                                    chat_id=chat_id,
                                    content=_STALE_APPROVAL_MESSAGE,
                                    metadata=dict(request.metadata or {}),
                                ),
                                messages=[],
                            )
                        else:
                            self._remove_pending_approval_identity_locked(approval.pending)
                            if approval.action == "negative":
                                result = TurnResult(
                                    run_id=run_id,
                                    status=RunStatus.CANCELLED,
                                    content="Cancelled the pending risky action.",
                                    final_content="Cancelled the pending risky action.",
                                    stop_reason="cancelled",
                                    outbound=OutboundMessage(
                                        channel=channel,
                                        chat_id=chat_id,
                                        content="Cancelled the pending risky action.",
                                        metadata=dict(request.metadata or {}),
                                    ),
                                    messages=[],
                                )
                            elif approval.action == "affirmative":
                                context.policy.approval_granted = True

                if result is None:
                    await self._connect_mcp()
                    result = await self._execute_turn_locked(
                        request,
                        context,
                        record,
                        channel=channel,
                        chat_id=chat_id,
                        tools=_tools,
                        approval_action=approval.action,
                        approval_pending=approval.pending,
                    )
        except asyncio.CancelledError:
            context.cancelled = True
            cancelled = TurnResult(
                run_id=run_id,
                status=RunStatus.CANCELLED,
                stop_reason="cancelled",
                error="turn cancelled",
                content=None,
                final_content=None,
                sent_messages=tuple(context.delivery.sent_messages),
            )
            self._finalize_run_record(record, context, cancelled, status=RunStatus.CANCELLED)
            raise
        except PromptBudgetExceeded as exc:
            # Budget failures are deterministic local validation outcomes. Do
            # not log a traceback here: traceback formatting can expose the
            # oversized message payload through local-variable rendering.
            result = TurnResult(
                run_id=run_id,
                status=RunStatus.ERROR,
                content=None,
                final_content=None,
                stop_reason="prompt_budget_exceeded",
                error=str(exc),
                outbound=None,
                sent_messages=tuple(context.delivery.sent_messages),
            )
            self._finalize_run_record(record, context, result, status=RunStatus.ERROR)
            callback = request.callbacks.on_error
            if callback is not None:
                try:
                    callback_result = callback(result)
                    if hasattr(callback_result, "__await__"):
                        await callback_result
                except Exception:
                    logger.exception("Turn callback failed for {}", run_id)
        except Exception as exc:
            logger.exception("Error executing canonical turn {}", run_id)
            error = f"Error: {type(exc).__name__}: {exc}"
            outbound = OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content="Sorry, I encountered an error.",
                metadata=dict(request.metadata or {}),
            )
            result = TurnResult(
                run_id=run_id,
                status=RunStatus.ERROR,
                content=outbound.content,
                final_content=outbound.content,
                stop_reason="error",
                error=error,
                outbound=outbound,
                sent_messages=tuple(context.delivery.sent_messages),
            )
            self._finalize_run_record(record, context, result, status=RunStatus.ERROR)
            callback = request.callbacks.on_error
            if callback is not None:
                try:
                    callback_result = callback(result)
                    if hasattr(callback_result, "__await__"):
                        await callback_result
                except Exception:
                    logger.exception("Turn callback failed for {}", run_id)
        else:
            if (
                result is not None
                and result.status is not RunStatus.COMMAND
                and result.stop_reason not in {"cancelled", "approval_stale"}
                and self._should_mirror_task_session(
                    context.session_key or key,
                    f"{channel}:{chat_id}",
                )
            ):
                visible_key = f"{channel}:{chat_id}"
                visible_lock = self._session_locks.setdefault(visible_key, asyncio.Lock())
                async with visible_lock:
                    self._mirror_task_session_to_visible_chat(
                        session_key=context.session_key or key,
                        channel=channel,
                        chat_id=chat_id,
                        task_text=str(
                            context.metadata.get("_mirror_task_text", request.content)
                        ),
                        response_text=result.content or result.final_content or "",
                        approval_granted=bool(context.policy.approval_granted),
                        run_id=context.run_id,
                        status=getattr(result.status, "value", str(result.status)),
                        stop_reason=result.stop_reason,
                        occurrence_id=(
                            request.scheduled_link.occurrence_id
                            if request.scheduled_link is not None
                            else None
                        ),
                    )
            assert result is not None
            result.run_id = run_id
            result.sent_messages = tuple(context.delivery.sent_messages)
            result.policy_metadata = result.policy_metadata or context.policy.metadata
            self._finalize_run_record(record, context, result)
            callback = request.callbacks.on_error if result.error else request.callbacks.on_complete
            if callback is not None:
                try:
                    callback_result = callback(result)
                    if hasattr(callback_result, "__await__"):
                        await callback_result
                except Exception:
                    logger.exception("Turn callback failed for {}", run_id)
        finally:
            reset_current_turn_context(token)
        return result

    async def _execute_turn_locked(
        self,
        request: TurnRequest,
        context: TurnContext,
        record: RunRecord,
        *,
        channel: str,
        chat_id: str,
        tools: ToolRegistry | None = None,
        approval_action: str = "none",
        approval_pending: _PendingApproval | None = None,
    ) -> TurnResult:
        """Run the canonical body after ``execute_turn`` acquires its locks."""
        msg = InboundMessage(
            channel=channel,
            sender_id=request.sender_id,
            chat_id=chat_id,
            content=request.content,
            media=list(request.media),
            metadata=dict(request.metadata or {}),
            session_key_override=context.session_key
            if context.session_key != f"{channel}:{chat_id}"
            else None,
        )

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)
        key = context.session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        raw = msg.content.strip()
        command_context = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if command_result := await self.commands.dispatch(command_context):
            return TurnResult(
                run_id=context.run_id,
                status=RunStatus.COMMAND,
                content=command_result.content,
                final_content=command_result.content,
                outbound=command_result,
                messages=[],
            )

        history = (
            []
            if request.history_mode is HistoryMode.FRESH
            else session.get_model_history(max_messages=0)
        )
        current_message = msg.content
        if approval_action == "affirmative" and approval_pending is not None:
            current_message = (
                "The user approved the previously blocked risky action. "
                f"Resume the task that required approval: {approval_pending.summary}."
                f"\n\nOriginal approval reply: {msg.content}"
            )
        context.metadata["_mirror_task_text"] = current_message

        effective_tools = tools if tools is not None else (
            self.tools.filtered(list(request.tool_names))
            if request.tool_names is not None
            else self.tools
        )
        tool_names_set = (
            set(effective_tools.tool_names) if effective_tools is not self.tools else None
        )
        initial_messages = self.context.build_messages(
            history=history,
            current_message=current_message,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            tool_names=tool_names_set,
            inject_memory=True,
        )
        if self.budget_manager is not None:
            initial_messages = await self.budget_manager.enforce_budget(
                initial_messages,
                preserve_last_n_turns=self.tool_result_clearing_keep,
            )
        save_from = len(initial_messages) - 1

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=content,
                metadata=meta,
            ))

        result = await self._run_main_task(
            initial_messages,
            session_key=key,
            on_progress=request.callbacks.on_progress or _bus_progress,
            on_stream=request.callbacks.on_stream,
            on_stream_end=request.callbacks.on_stream_end,
            channel=msg.channel,
            chat_id=msg.chat_id,
            message_id=target_message_id(context.delivery.primary),
            approval_granted=bool(context.policy.approval_granted),
            tools=effective_tools,
            turn_context=context,
            extra_hooks=list(request.hooks),
        )
        final_content = result.final_content or "I've completed processing but have no response to give."
        context.tools_used = list(result.tools_used)
        self._save_turn(session, result.messages, save_from, run_id=context.run_id)
        self.sessions.save(session)
        self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
        if key.startswith("cron:"):
            await self._sync_context_repos()
        else:
            self._maybe_sync_context_repo()

        if result.stop_reason == "approval_required":
            summary = str(result.policy_metadata.get("summary") or "risky action")
            await self._store_pending_approval(
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                summary=summary,
            )
        primary = context.delivery.primary or context.delivery.target
        primary_chat = primary.chat_id or primary.to or primary.recipient if primary else None
        final_text = final_content.strip() if isinstance(final_content, str) else ""
        sent_to_primary = any(
            sent.channel == primary.channel
            and sent.chat_id == primary_chat
            and isinstance(sent.content, str)
            and bool(sent.content.strip())
            and bool(final_text)
            and sent.content.strip() == final_text
            for sent in context.delivery.sent_messages
        ) if primary else False
        outbound: OutboundMessage | None
        if sent_to_primary:
            outbound = None
        else:
            meta = dict(msg.metadata or {})
            if request.callbacks.on_stream is not None and result.stop_reason != "approval_required":
                meta["_streamed"] = True
            outbound = OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=final_content,
                metadata=meta,
            )
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, final_content[:120])
        return TurnResult(
            run_id=context.run_id,
            status=self._run_status(result.stop_reason),
            content=final_content,
            final_content=final_content,
            stop_reason=result.stop_reason,
            error=result.error,
            usage=result.usage,
            tools_used=list(result.tools_used),
            outbound=outbound,
            policy_metadata=result.policy_metadata,
            messages=result.messages,
        )

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        approval_granted: bool = False,
        tools: ToolRegistry | None = None,
    ) -> OutboundMessage | None:
        """Compatibility adapter around :meth:`execute_turn`."""
        source = TurnSource.SYSTEM_COMPAT if msg.channel == "system" else TurnSource.GATEWAY
        if msg.channel == "system":
            if ":" in msg.chat_id:
                route_channel, route_chat = msg.chat_id.split(":", 1)
            else:
                route_channel, route_chat = "cli", msg.chat_id
            compat_session_key = session_key or f"{route_channel}:{route_chat}"
            warnings.warn(
                "Deprecated system message compatibility adapter used; "
                "use execute_turn(TurnRequest(...)) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.warning(
                "Deprecated system message compatibility adapter used",
                event="system_compat",
                channel=route_channel,
                chat_id=route_chat,
                session_key=compat_session_key,
            )
        else:
            route_channel, route_chat = msg.channel, msg.chat_id
            compat_session_key = session_key or msg.session_key
        route = DeliveryTarget(
            channel=route_channel,
            chat_id=route_chat,
            message_id=msg.metadata.get("message_id"),
        )
        grant = ApprovalGrant(approved=True, source="compat") if approval_granted else None
        request = TurnRequest(
            content=msg.content,
            source=source,
            session_key=compat_session_key,
            route=route,
            sender_id=msg.sender_id,
            media=tuple(msg.media),
            approval_grant=grant,
            approval=grant or ApprovalGrant(),
            callbacks=TurnCallbacks(
                on_progress=on_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
            ),
            metadata=dict(msg.metadata or {}),
            tool_names=tuple(tools.tool_names) if tools is not None else None,
        )
        result = await self.execute_turn(request, _tools=tools)
        # Keep the legacy inspection properties useful for callers that still
        # invoke ``_process_message`` directly.  Canonical execution itself
        # uses only ``TurnResult.sent_messages`` and never reads these fields.
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool._sent_messages_in_turn = list(result.sent_messages)
            message_tool._sent_in_turn = result.outbound is None and bool(result.sent_messages)
        return result.outbound

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

    def _save_turn(
        self,
        session: Session,
        messages: list[dict],
        skip: int,
        *,
        run_id: str | None = None,
    ) -> None:
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
            if run_id:
                # SessionManager strips this internal marker from model history
                # while RunStore can resolve the detailed trace by run ID.
                entry.pop("role", None)
                entry.pop("_run_id", None)
                session.add_run_message(
                    run_id,
                    str(role or "assistant"),
                    entry.pop("content", ""),
                    **entry,
                )
            else:
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

    async def _sync_context_repos(self) -> bool:
        """Sync configured context repos and report whether every sync succeeded."""
        repos = [repo for repo in self.context_manager.repos if repo.auto_sync]
        if not repos:
            return True
        from nanobot.utils.git_sync import async_sync_context_repo

        try:
            results = await asyncio.gather(
                *(
                    async_sync_context_repo(
                        repo.path,
                        include_paths=repo.sync_include_patterns(),
                        exclude_paths=repo.sync_exclude_patterns(),
                        message=f"nanobot: sync {repo.name} context updates",
                    )
                    for repo in repos
                )
            )
        except Exception:
            logger.exception("Context repo sync failed")
            return False

        failed = [repo.name for repo, succeeded in zip(repos, results) if not succeeded]
        if failed:
            logger.error("Context repo sync did not complete for: {}", ", ".join(failed))
            return False
        return True

    def _maybe_sync_context_repo(self) -> None:
        """Schedule a background context-repository sync for interactive turns."""
        if any(repo.auto_sync for repo in self.context_manager.repos):
            self._schedule_background(self._sync_context_repos())

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        approval_granted: bool = False,
        tool_names: list[str] | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload.

        *tool_names*: optional whitelist of tool names to expose.  When set,
        only the listed tools are available to the agent, which reduces the
        system prompt token count — useful for lightweight scheduled tasks on
        resource-constrained devices.
        """
        tools = self.tools.filtered(tool_names) if tool_names else None
        grant = ApprovalGrant(approved=True, source="direct") if approval_granted else None
        request = TurnRequest(
            content=content,
            source=TurnSource.DIRECT,
            session_key=session_key,
            route=DeliveryTarget(channel=channel, chat_id=chat_id),
            sender_id="user",
            approval_grant=grant,
            approval=grant or ApprovalGrant(),
            callbacks=TurnCallbacks(
                on_progress=on_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
            ),
            tool_names=tuple(tool_names) if tool_names else None,
        )
        result = await self.execute_turn(request, _tools=tools)
        return result.outbound
