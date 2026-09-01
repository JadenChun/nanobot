"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.message_content import compact_content, content_to_text
from nanobot.agent.policy import ToolPolicy, ToolPolicyDecision
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import ToolOutcome
from nanobot.providers.base import LLMProvider, ToolCallRequest
from nanobot.utils.prompt_budget import (
    PromptBudget,
    assert_prompt_fits,
    reduce_messages_to_budget,
)
from nanobot.utils.helpers import build_assistant_message, estimate_prompt_tokens_chain

_DEFAULT_MAX_ITERATIONS_MESSAGE = (
    "I reached the maximum number of tool call iterations ({max_iterations}) "
    "without completing the task. You can try breaking the task into smaller steps."
)
_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."
_CLEARED_PLACEHOLDER = "[cleared to save context]"
_COMPACTED_PLACEHOLDER = "[compacted to save context]"
_COMPACTED_HEAD_LINES = 6
_COMPACTED_TAIL_LINES = 4
_COMPACTED_MAX_CHARS = 700
_TERMINAL_TOOL_STOP_REASONS = frozenset({
    "approval_required",
    "policy_blocked",
    "tool_error",
    "cancelled",
})


@dataclass(slots=True)
class _ToolBatchResult:
    """Results from one model tool batch, including its first terminal stop."""

    results: list[Any]
    events: list[dict[str, str]]
    fatal_error: BaseException | None = None
    terminal_outcome: ToolOutcome | None = None

    def __iter__(self):
        """Keep the historical three-value private helper unpacking intact."""
        yield self.results
        yield self.events
        yield self.fatal_error


def _extract_tool_result_text(content: Any) -> str:
    """Best-effort text extraction for compacting tool results."""
    return content_to_text(content)


def _compact_tool_result_content(content: Any) -> str:
    """Preserve a small but useful trace of an old tool result."""
    return compact_content(
        content,
        compacted_placeholder=_COMPACTED_PLACEHOLDER,
        cleared_placeholder=_CLEARED_PLACEHOLDER,
        head_lines=_COMPACTED_HEAD_LINES,
        tail_lines=_COMPACTED_TAIL_LINES,
        max_chars=_COMPACTED_MAX_CHARS,
    )


def clear_old_tool_results(
    messages: list[dict[str, Any]],
    keep_last: int = 3,
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    trigger_tokens: int | None = None,
    target_tokens: int | None = None,
) -> None:
    """Shrink old tool result content, keeping the last N intact.

    Modelled on Anthropic's ``clear_tool_uses_20250919`` strategy: the assistant
    ``tool_calls`` block is left intact so the model still knows *what* it called,
    but older ``tool`` result content is compacted only when needed to free
    context tokens. If compaction is insufficient, the oldest results are fully
    cleared as a last resort.
    """
    if keep_last <= 0:
        return

    # Collect indices of all tool-result messages.
    tool_indices: list[int] = [
        i for i, m in enumerate(messages) if m.get("role") == "tool"
    ]

    # Nothing to clear if there aren't more results than we want to keep.
    if len(tool_indices) <= keep_last:
        return

    to_clear = tool_indices[:-keep_last]

    # Preserve full results unless we are actually approaching the prompt budget.
    if (
        provider is not None
        and tools is not None
        and trigger_tokens is not None
        and target_tokens is not None
        and trigger_tokens > 0
        and target_tokens > 0
    ):
        estimated, _ = estimate_prompt_tokens_chain(provider, model, messages, tools)
        if estimated <= 0 or estimated < trigger_tokens:
            return

        for idx in to_clear:
            compacted = _compact_tool_result_content(messages[idx].get("content"))
            if compacted == messages[idx].get("content"):
                continue
            messages[idx] = {**messages[idx], "content": compacted}
            estimated, _ = estimate_prompt_tokens_chain(provider, model, messages, tools)
            if 0 < estimated <= target_tokens:
                return

        # If compacted snippets still leave us over budget, fall back to clearing.
        for idx in to_clear:
            if messages[idx].get("content") == _CLEARED_PLACEHOLDER:
                continue
            messages[idx] = {**messages[idx], "content": _CLEARED_PLACEHOLDER}
            estimated, _ = estimate_prompt_tokens_chain(provider, model, messages, tools)
            if 0 < estimated <= target_tokens:
                return
        return

    # Legacy behavior for callers that do not provide prompt-budget context.
    for idx in to_clear:
        messages[idx] = {**messages[idx], "content": _CLEARED_PLACEHOLDER}


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    hook: AgentHook | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    concurrent_tools: bool = False
    fail_on_tool_error: bool = False
    tool_result_clearing_keep: int | None = None
    tool_result_clear_trigger_tokens: int | None = None
    tool_result_clear_target_tokens: int | None = None
    tool_policy: ToolPolicy | None = None
    # Hard ceiling on prompt tokens.  When set, the runner will trim the
    # oldest conversation turns (assistant + tool pairs) at the top of each
    # iteration to keep the context within budget.  This prevents the
    # gradual context growth that causes inference slowdown on resource-
    # constrained devices.
    max_input_tokens: int | None = None
    # Unified context budget manager. When provided, the runner uses this
    # for mid-loop context enforcement instead of the scattered tool result
    # clearing and turn trimming logic.
    budget_manager: Any | None = None
    # Explicit per-request budget for delegated callers.  Main-agent callers
    # normally provide ``budget_manager``; this field keeps auxiliary runners
    # on the same final provider-boundary invariant.
    prompt_budget: PromptBudget | None = None


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    policy_metadata: dict[str, Any] = field(default_factory=dict)


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    @staticmethod
    def _truncate_detail(detail: str, max_chars: int = 120) -> str:
        compact = detail.replace("\n", " ").strip()
        if not compact:
            return "(empty)"
        if len(compact) > max_chars:
            return compact[:max_chars] + "..."
        return compact

    @classmethod
    def _summarize_tool_result(cls, result: Any) -> tuple[str, str]:
        """Return (status, detail) for a tool result.

        Status is "error" for explicit error strings and for structured JSON outputs
        that report process failures (for example {"exitCode": 1, ...}).
        """
        if result is None:
            return "ok", "(empty)"

        if isinstance(result, dict):
            err = result.get("error")
            if err:
                return "error", cls._truncate_detail(content_to_text(err))
            exit_code = result.get("exitCode")
            if isinstance(exit_code, int) and exit_code != 0:
                stderr = result.get("stderr")
                if isinstance(stderr, str) and stderr.strip():
                    first = stderr.strip().splitlines()[0]
                    return "error", cls._truncate_detail(f"exitCode {exit_code}: {first}")
                return "error", f"exitCode {exit_code}"

        if isinstance(result, str):
            text = result.strip()
            if text.startswith("Error"):
                return "error", cls._truncate_detail(text)

            parsed: dict[str, Any] | None = None
            if text.startswith("{") and text.endswith("}"):
                try:
                    raw = json.loads(text)
                    if isinstance(raw, dict):
                        parsed = raw
                except Exception:
                    parsed = None

            if parsed is not None:
                err = parsed.get("error")
                if err:
                    return "error", cls._truncate_detail(str(err))

                exit_code = parsed.get("exitCode")
                if isinstance(exit_code, int) and exit_code != 0:
                    stderr = parsed.get("stderr")
                    if isinstance(stderr, str) and stderr.strip():
                        first = stderr.strip().splitlines()[0]
                        return "error", cls._truncate_detail(f"exitCode {exit_code}: {first}")
                    return "error", f"exitCode {exit_code}"

        return "ok", cls._truncate_detail(content_to_text(result) or "(empty)")

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        # Request-local reductions and runner appends must never mutate the
        # caller's persisted/session-owned dictionaries.
        messages = copy.deepcopy(spec.initial_messages)
        final_content: str | None = None
        tools_used: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        error: str | None = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []
        latest_context: AgentHookContext | None = None
        terminal_stream_end_emitted = False

        async def emit_terminal_stream_end() -> None:
            """Close the run's stream exactly once, including on exceptions."""
            nonlocal terminal_stream_end_emitted
            if terminal_stream_end_emitted or not hook.wants_streaming():
                return
            terminal_stream_end_emitted = True
            context = latest_context or AgentHookContext(
                iteration=0,
                messages=messages,
            )
            await hook.on_stream_end(context, resuming=False)

        def provider_error_result(exc: Exception, context: AgentHookContext) -> AgentRunResult:
            """Normalize provider failures while preserving hook failures."""
            provider_error = f"Error: {type(exc).__name__}: {exc}"
            context_error_content = spec.error_message or _DEFAULT_ERROR_MESSAGE
            context.final_content = context_error_content
            context.error = provider_error
            context.stop_reason = "error"
            return AgentRunResult(
                final_content=context_error_content,
                messages=messages,
                tools_used=tools_used,
                usage=usage,
                stop_reason="error",
                error=provider_error,
                tool_events=tool_events,
            )

        try:
            for iteration in range(spec.max_iterations):
                # Mid-loop context enforcement
                if iteration > 0:
                    if spec.budget_manager is not None:
                        # Unified context model: use budget manager for all enforcement
                        await spec.budget_manager.enforce_budget(messages)
                    else:
                        # Legacy model: scattered trimming
                        # Hard ceiling: drop the oldest assistant+tool turn pairs if the
                        # accumulated context exceeds max_input_tokens.  Run this BEFORE
                        # clear_old_tool_results so we don't waste effort clearing
                        # messages that are about to be deleted.
                        if spec.max_input_tokens and spec.max_input_tokens > 0:
                            self._trim_context_to_budget(
                                messages,
                                spec=spec,
                            )

                        # Clear old tool results each iteration to prevent context overflow
                        # during long-running tasks, but only once prompt size actually needs it.
                        if spec.tool_result_clearing_keep is not None:
                            clear_old_tool_results(
                                messages,
                                keep_last=spec.tool_result_clearing_keep,
                                provider=self.provider,
                                model=spec.model,
                                tools=spec.tools.get_definitions(),
                                trigger_tokens=spec.tool_result_clear_trigger_tokens,
                                target_tokens=spec.tool_result_clear_target_tokens,
                            )

                context = AgentHookContext(iteration=iteration, messages=messages)
                latest_context = context
                await hook.before_iteration(context)

                # Final request-local reduction and hard preflight happen
                # after all hooks, immediately before provider transport.  A
                # provider-boundary check below is still applied by the base
                # provider as defense in depth.
                budget = spec.prompt_budget
                if budget is None and spec.budget_manager is not None:
                    budget = getattr(spec.budget_manager.budget, "prompt_budget", None)
                if budget is None:
                    total = spec.max_input_tokens
                    if not isinstance(total, int) or total <= 0:
                        total = getattr(
                            getattr(self.provider, "generation", None),
                            "context_window_tokens",
                            0,
                        )
                    if isinstance(total, int) and total > 0:
                        reserve = spec.max_tokens
                        if reserve is None:
                            reserve = getattr(
                                getattr(self.provider, "generation", None),
                                "max_tokens",
                                0,
                            )
                        budget = PromptBudget(
                            total_tokens=total,
                            completion_reserve=int(reserve or 0),
                        )

                if spec.budget_manager is not None:
                    await spec.budget_manager.enforce_budget(
                        messages,
                        preserve_last_n_turns=spec.tool_result_clearing_keep or 2,
                    )
                elif budget is not None:
                    messages[:] = reduce_messages_to_budget(
                        messages,
                        self.provider,
                        spec.model,
                        spec.tools.get_definitions(),
                        budget,
                        preserve_last_n_turns=spec.tool_result_clearing_keep or 2,
                    )
                context.messages = messages
                kwargs: dict[str, Any] = {
                    "messages": messages,
                    "tools": spec.tools.get_definitions(),
                    "model": spec.model,
                }
                if spec.temperature is not None:
                    kwargs["temperature"] = spec.temperature
                if spec.max_tokens is not None:
                    kwargs["max_tokens"] = spec.max_tokens
                if spec.reasoning_effort is not None:
                    kwargs["reasoning_effort"] = spec.reasoning_effort

                if budget is not None:
                    assert_prompt_fits(
                        self.provider,
                        spec.model,
                        messages,
                        spec.tools.get_definitions(),
                        budget,
                    )

                if hook.wants_streaming():
                    async def _stream(delta: str) -> None:
                        await hook.on_stream(context, delta)

                    try:
                        response = await self.provider.chat_stream_with_retry(
                            **kwargs,
                            on_content_delta=_stream,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        return provider_error_result(exc, context)
                else:
                    try:
                        response = await self.provider.chat_with_retry(**kwargs)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        return provider_error_result(exc, context)

                raw_usage = response.usage or {}
                usage = {
                    "prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
                }
                context.response = response
                context.usage = usage
                context.tool_calls = list(response.tool_calls)

                if response.has_tool_calls:
                    decision = await self._evaluate_tool_policy(spec, context, response.tool_calls)
                    if decision.action != "allow":
                        stop_reason = decision.stop_reason or "policy_blocked"
                        final_content = decision.response
                        proposed_approach = hook.finalize_content(context, response.content)
                        if (
                            (
                                getattr(decision, "requires_approval", False)
                                or stop_reason == "approval_required"
                            )
                            and isinstance(final_content, str)
                            and proposed_approach
                            and proposed_approach.strip()
                            and proposed_approach.strip() not in final_content
                        ):
                            final_content = (
                                f"{final_content}\n\n"
                                f"Proposed approach before approval:\n{proposed_approach.strip()}"
                            )
                        messages.append(build_assistant_message(
                            final_content,
                            reasoning_content=response.reasoning_content,
                            thinking_blocks=response.thinking_blocks,
                            image_calls=response.image_calls,
                        ))
                        context.final_content = final_content
                        context.stop_reason = stop_reason
                        await hook.after_iteration(context)
                        return AgentRunResult(
                            final_content=final_content,
                            messages=messages,
                            tools_used=tools_used,
                            usage=usage,
                            stop_reason=stop_reason,
                            error=error,
                            tool_events=tool_events,
                            policy_metadata=decision.metadata,
                        )

                    if hook.wants_streaming():
                        # This is an intermediate segment.  The terminal segment
                        # is emitted by the common finalizer below.
                        await hook.on_stream_end(context, resuming=True)

                    messages.append(build_assistant_message(
                        response.content or "",
                        tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                        image_calls=response.image_calls,
                    ))
                    tools_used.extend(tc.name for tc in response.tool_calls)

                    await hook.before_execute_tools(context)

                    batch = await self._execute_tools(spec, response.tool_calls)
                    results, new_events, fatal_error = batch
                    tool_events.extend(new_events)
                    context.tool_results = list(results)
                    context.tool_events = list(new_events)
                    for tool_call, result in zip(response.tool_calls, results):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        })

                    # A terminal ToolOutcome ends this iteration immediately.
                    # In the sequential path, _execute_tools has already
                    # represented every unstarted call as a skipped result, so
                    # provider history remains a legal assistant/tool pair.
                    if batch.terminal_outcome is not None:
                        terminal = batch.terminal_outcome
                        stop_reason = terminal.stop_reason or "tool_error"
                        final_content = (
                            terminal.content
                            if isinstance(terminal.content, str)
                            else str(terminal.content or "")
                        )
                        context.final_content = final_content
                        context.stop_reason = stop_reason
                        if stop_reason == "tool_error":
                            context.error = final_content
                        await hook.after_iteration(context)
                        return AgentRunResult(
                            final_content=final_content,
                            messages=messages,
                            tools_used=tools_used,
                            usage=usage,
                            stop_reason=stop_reason,
                            error=context.error,
                            tool_events=tool_events,
                            policy_metadata=dict(terminal.policy_metadata),
                        )
                    if fatal_error is not None:
                        error = f"Error: {type(fatal_error).__name__}: {fatal_error}"
                        stop_reason = "tool_error"
                        context.error = error
                        context.stop_reason = stop_reason
                        await hook.after_iteration(context)
                        break
                    await hook.after_iteration(context)
                    continue

                clean = hook.finalize_content(context, response.content)
                if response.finish_reason == "error":
                    final_content = clean or spec.error_message or _DEFAULT_ERROR_MESSAGE
                    stop_reason = "error"
                    error = final_content
                    context.final_content = final_content
                    context.error = error
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    break

                # If the model returned no visible text (e.g. only <think> blocks stripped)
                # but has already used tools, nudge it to produce a real response instead of
                # silently finishing — common with local/reasoning models.
                if (not clean or not clean.strip()) and tools_used and iteration < spec.max_iterations:
                    messages.append(build_assistant_message(
                        "",
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                        image_calls=response.image_calls,
                    ))
                    messages.append({
                        "role": "user",
                        "content": "Please provide a text response summarizing what you found and accomplished.",
                    })
                    await hook.after_iteration(context)
                    continue

                messages.append(build_assistant_message(
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                    image_calls=response.image_calls,
                ))
                final_content = clean
                context.final_content = final_content
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                break
            else:
                stop_reason = "max_iterations"
                template = spec.max_iterations_message or _DEFAULT_MAX_ITERATIONS_MESSAGE
                final_content = template.format(max_iterations=spec.max_iterations)
        except asyncio.CancelledError:
            # Child tools own their process cleanup; preserve cancellation after
            # the stream finalizer in ``finally`` has closed the visible stream.
            raise
        finally:
            if hook.wants_streaming():
                try:
                    await emit_terminal_stream_end()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A failing stream sink must not prevent the runner from
                    # returning its completed/error result.
                    logger.exception("Agent stream finalizer failed")

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
        )

    async def _evaluate_tool_policy(
        self,
        spec: AgentRunSpec,
        context: AgentHookContext,
        tool_calls: list[ToolCallRequest],
    ) -> ToolPolicyDecision:
        policy = spec.tool_policy
        if policy is None:
            return ToolPolicyDecision()
        return await policy.evaluate(messages=context.messages, tool_calls=tool_calls)

    async def _execute_tools(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
    ) -> _ToolBatchResult:
        should_run_concurrently = spec.concurrent_tools and all(
            (tool is None or tool.supports_parallel_calls)
            for tool in (spec.tools.get(tool_call.name) for tool_call in tool_calls)
        )

        if should_run_concurrently:
            raw_results = await asyncio.gather(*(
                self._run_tool(spec, tool_call)
                for tool_call in tool_calls
            ))
        else:
            raw_results = []
            for tool_call in tool_calls:
                raw_result = await self._run_tool(spec, tool_call)
                raw_results.append(raw_result)
                execution = self._normalize_tool_execution(raw_result)
                if execution[3] is not None or execution[2] is not None:
                    # A terminal outcome/error must prevent every later
                    # sequential call from starting.  Fill the result list
                    # with explicit skipped entries so the provider sees a
                    # result for each assistant tool_call id.
                    results = [
                        self._normalize_tool_execution(item)[0]
                        for item in raw_results
                    ]
                    events = [
                        self._normalize_tool_execution(item)[1]
                        for item in raw_results
                    ]
                    terminal = execution[3]
                    fatal_error = execution[2]
                    if terminal is None and fatal_error is not None:
                        terminal = ToolOutcome(
                            content=f"Error: {type(fatal_error).__name__}: {fatal_error}",
                            stop_reason="tool_error",
                        )
                    for skipped_call in tool_calls[len(raw_results):]:
                        skipped_detail = (
                            f"Skipped: not executed after terminal {terminal.stop_reason}"
                            if terminal is not None
                            else "Skipped: not executed after a tool error"
                        )
                        results.append(skipped_detail)
                        events.append({
                            "name": skipped_call.name,
                            "status": "skipped",
                            "detail": skipped_detail,
                            "tool_call_id": skipped_call.id,
                        })
                    return _ToolBatchResult(
                        results=results,
                        events=events,
                        fatal_error=None if terminal is not None else fatal_error,
                        terminal_outcome=terminal,
                    )

            # No terminal result occurred in the sequential path.
            tool_results = raw_results

        # Concurrent calls have all been launched by this point.  We still
        # preserve the first terminal outcome in model order, but cannot claim
        # that later calls were prevented from running.
        tool_results = raw_results

        results: list[Any] = []
        events: list[dict[str, str]] = []
        fatal_error: BaseException | None = None
        terminal_outcome: ToolOutcome | None = None
        for raw_result in tool_results:
            result, event, error, outcome = self._normalize_tool_execution(raw_result)
            results.append(result)
            events.append(event)
            if outcome is None and error is not None:
                outcome = ToolOutcome(
                    content=f"Error: {type(error).__name__}: {error}",
                    stop_reason="tool_error",
                )
            if terminal_outcome is None and outcome is not None:
                terminal_outcome = outcome
            if error is not None and fatal_error is None:
                fatal_error = error
        if terminal_outcome is None and fatal_error is not None:
            terminal_outcome = ToolOutcome(
                content=f"Error: {type(fatal_error).__name__}: {fatal_error}",
                stop_reason="tool_error",
            )
        return _ToolBatchResult(
            results=results,
            events=events,
            fatal_error=fatal_error if terminal_outcome is None else None,
            terminal_outcome=terminal_outcome,
        )

    @staticmethod
    def _normalize_tool_execution(
        raw_result: tuple[Any, ...],
    ) -> tuple[Any, dict[str, str], BaseException | None, ToolOutcome | None]:
        """Normalize current and legacy _run_tool return shapes."""
        if len(raw_result) == 4:
            result, event, error, outcome = raw_result
            if isinstance(outcome, ToolOutcome):
                result = outcome.content
                if outcome.stop_reason in _TERMINAL_TOOL_STOP_REASONS:
                    return result, {
                        **event,
                        "status": outcome.stop_reason,
                    }, error, outcome
            return result, event, error, None
        result, event, error = raw_result
        if isinstance(result, ToolOutcome):
            return result.content, {
                **event,
                "status": result.stop_reason or event.get("status", "ok"),
            }, error, (
                result if result.stop_reason in _TERMINAL_TOOL_STOP_REASONS else None
            )
        return result, event, error, None

    async def _run_tool(
        self,
        spec: AgentRunSpec,
        tool_call: ToolCallRequest,
    ) -> tuple[Any, dict[str, str], BaseException | None, ToolOutcome | None]:
        try:
            raw_result = await spec.tools.execute(tool_call.name, tool_call.arguments)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
            }
            if spec.fail_on_tool_error:
                return f"Error: {type(exc).__name__}: {exc}", event, exc, None
            return f"Error: {type(exc).__name__}: {exc}", event, None, None

        outcome = raw_result if isinstance(raw_result, ToolOutcome) else None
        result = outcome.content if outcome is not None else raw_result
        if outcome is not None and outcome.stop_reason in _TERMINAL_TOOL_STOP_REASONS:
            return result, {
                "name": tool_call.name,
                "status": outcome.stop_reason,
                "detail": self._truncate_detail(content_to_text(result) or "(empty)"),
            }, None, outcome

        status, detail = self._summarize_tool_result(result)
        return result, {
            "name": tool_call.name,
            "status": status,
            "detail": detail,
        }, None, None

    def _trim_context_to_budget(
        self,
        messages: list[dict[str, Any]],
        *,
        spec: AgentRunSpec,
    ) -> None:
        """Drop the oldest assistant+tool turn pairs to stay within *max_input_tokens*.

        The system message and the first user message are never removed.
        Assistant messages that contain tool_calls are kept together with
        their subsequent tool-result messages so the conversation remains
        well-formed for the LLM API.
        """
        max_tokens = spec.max_input_tokens
        if not max_tokens:
            return

        tool_defs = spec.tools.get_definitions()
        tokens, _ = estimate_prompt_tokens_chain(
            self.provider, spec.model, messages, tool_defs,
        )
        if tokens <= max_tokens:
            return

        # Identify turn boundaries: an assistant message (possibly with
        # tool_calls) followed by its tool results forms one "turn".
        turns: list[tuple[int, int]] = []  # (start_idx, end_idx) inclusive
        i = 1  # skip system message at index 0
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") == "assistant":
                start = i
                i += 1
                # Consume subsequent tool-result messages belonging to this turn.
                while i < len(messages) and messages[i].get("role") == "tool":
                    i += 1
                turns.append((start, i - 1))
            else:
                i += 1

        # Drop oldest turns until we are within budget (keep at least 1 turn).
        # Iterate from the END so that deletions do not invalidate earlier indices.
        droppable = list(reversed(turns[:-1]))
        dropped_indices: list[tuple[int, int]] = []
        for start, end in droppable:
            del messages[start:end + 1]
            dropped_indices.append((start, end))
            tokens, _ = estimate_prompt_tokens_chain(
                self.provider, spec.model, messages, tool_defs,
            )
            if tokens <= max_tokens:
                break

        dropped = len(dropped_indices)
        if dropped:
            logger.info(
                "Mid-loop context trim: dropped {} old turn(s) to reach {} tokens (budget {})",
                dropped, tokens, max_tokens,
            )
