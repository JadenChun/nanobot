"""Shared lifecycle hook primitives for agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMResponse, ToolCallRequest


@dataclass(slots=True)
class AgentHookContext:
    """Mutable per-iteration state exposed to runner hooks."""

    iteration: int
    messages: list[dict[str, Any]]
    response: LLMResponse | None = None
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)
    tool_events: list[dict[str, str]] = field(default_factory=list)
    final_content: str | None = None
    stop_reason: str | None = None
    error: str | None = None


class AgentHook:
    """Minimal lifecycle surface for shared runner customization."""

    def wants_streaming(self) -> bool:
        return False

    async def before_iteration(self, context: AgentHookContext) -> None:
        pass

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        pass

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        pass

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        pass

    async def after_iteration(self, context: AgentHookContext) -> None:
        pass

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return content


class CompositeHook(AgentHook):
    """Fan-out hook that delegates to an ordered list of hooks.

    Error isolation: hook exceptions are caught and logged per hook
    so a faulty custom hook cannot crash the agent loop. ``finalize_content``
    remains an ordered pipeline, retaining the last successful content when a
    transform fails.
    """

    __slots__ = ("_hooks",)

    def __init__(self, hooks: list[AgentHook]) -> None:
        self._hooks = list(hooks)

    def wants_streaming(self) -> bool:
        wants_streaming = False
        for h in self._hooks:
            try:
                if h.wants_streaming():
                    wants_streaming = True
            except Exception as exc:
                logger.error(
                    "AgentHook.wants_streaming error in {} ({})",
                    type(h).__name__,
                    type(exc).__name__,
                )
        return wants_streaming

    async def before_iteration(self, context: AgentHookContext) -> None:
        for h in self._hooks:
            try:
                await h.before_iteration(context)
            except Exception:
                logger.exception("AgentHook.before_iteration error in {}", type(h).__name__)

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        for h in self._hooks:
            try:
                await h.on_stream(context, delta)
            except Exception:
                logger.exception("AgentHook.on_stream error in {}", type(h).__name__)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        for h in self._hooks:
            try:
                await h.on_stream_end(context, resuming=resuming)
            except Exception:
                logger.exception("AgentHook.on_stream_end error in {}", type(h).__name__)

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        for h in self._hooks:
            try:
                await h.before_execute_tools(context)
            except Exception:
                logger.exception("AgentHook.before_execute_tools error in {}", type(h).__name__)

    async def after_iteration(self, context: AgentHookContext) -> None:
        for h in self._hooks:
            try:
                await h.after_iteration(context)
            except Exception:
                logger.exception("AgentHook.after_iteration error in {}", type(h).__name__)

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        for h in self._hooks:
            try:
                transformed = h.finalize_content(context, content)
            except Exception as exc:
                logger.error(
                    "AgentHook.finalize_content error in {} ({})",
                    type(h).__name__,
                    type(exc).__name__,
                )
            else:
                content = transformed
        return content
