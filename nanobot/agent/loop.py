"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import AsyncExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.context_budget import ContextBudget, ContextBudgetManager
from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.policy import RiskyActionPolicy
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
from nanobot.agent.write_guard import FileLockRegistry
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.helpers import estimate_prompt_tokens_chain

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
            max_input_tokens=self.max_tokens.input if self.max_tokens.input > 0 else None,
            budget_manager=self.budget_manager,  # Pass unified budget manager for mid-loop enforcement
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
        session_key: str = "cli:direct",
        channel: str,
        chat_id: str,
        message_id: str | None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        approval_granted: bool = False,
        tools: ToolRegistry | None = None,
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
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                current_role="user",
            )

            if self.max_tokens.input > 0:
                try:
                    sys_tool_defs = self.tools.get_definitions()
                    tokens, _ = estimate_prompt_tokens_chain(self.provider, self.model, messages, sys_tool_defs)
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
                                current_role="user",
                            )
                            tokens, _ = estimate_prompt_tokens_chain(
                                self.provider,
                                self.model,
                                messages,
                                sys_tool_defs,
                            )
                except Exception as e:
                    logger.error("Failed to check system token count: {}", e)

            save_from = len(messages) - 1
            final_content, _, all_msgs = await self._run_agent_loop(
                messages, channel=channel, chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
                session_key=key,
            )
            self._save_turn(session, all_msgs, save_from)
            self.sessions.save(session)
            self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
            if key.startswith("cron:"):
                await self._sync_context_repos()
            else:
                self._maybe_sync_context_repo()
            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=final_content or "Task completed.",
            )

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

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
        self.delegation.start_turn()

        history = session.get_history(max_messages=0)
        current_message = msg.content
        if approval_note:
            current_message = f"{approval_note}\n\nOriginal approval reply: {msg.content}"

        # Use filtered tool set if provided, otherwise full set.
        effective_tools = tools if tools is not None else self.tools
        tool_names_set = set(effective_tools.tool_names) if tools is not None else None

        # Unified context model: memory injected as [Past Knowledge] message
        initial_messages = self.context.build_messages(
            history=history,
            current_message=current_message,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
            tool_names=tool_names_set,
            inject_memory=True,
        )

        # Enforce budget using unified manager (handles all reduction strategies)
        if self.budget_manager is not None:
            initial_messages = await self.budget_manager.enforce_budget(
                initial_messages,
                preserve_last_n_turns=self.tool_result_clearing_keep,
            )
        # The current user message is always the final initial message. Saving
        # from this index avoids persisting injected memory or duplicating the
        # last history entry when [Past Knowledge] is present.
        save_from = len(initial_messages) - 1

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        result = await self._run_main_task(
            initial_messages,
            session_key=key,
            on_progress=on_progress or _bus_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=msg.channel, chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
            approval_granted=approval_granted or approval_note is not None,
            tools=tools,
        )
        final_content = result.final_content
        all_msgs = result.messages

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, save_from)
        self.sessions.save(session)
        self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
        if key.startswith("cron:"):
            await self._sync_context_repos()
        else:
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
            approval_granted=approval_granted or approval_note is not None,
        )

        # A successful direct send is already the user-facing result, including
        # scheduled turns. Do not publish the model's trailing completion recap.
        if (
            (mt := self.tools.get("message"))
            and isinstance(mt, MessageTool)
            and mt._sent_in_turn
        ):
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
        await self._connect_mcp()
        tools = self.tools.filtered(tool_names) if tool_names else None
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        return await self._process_message(
            msg, session_key=session_key, on_progress=on_progress,
            on_stream=on_stream, on_stream_end=on_stream_end,
            approval_granted=approval_granted,
            tools=tools,
        )
