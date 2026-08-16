"""Tests for the shared agent runner and its integration contracts."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.providers.base import LLMResponse, ToolCallRequest


class _TokenCountingProvider:
    def estimate_prompt_tokens(self, messages, tools, model):
        total = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total += len(content)
            else:
                total += len(json.dumps(content, ensure_ascii=False))
        return total, "test_counter"


def _make_loop(tmp_path):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.ForegroundAgentManager"):
        loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path)
    return loop


@pytest.mark.asyncio
async def test_runner_preserves_reasoning_fields_and_tool_results():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    captured_second_call: list[dict] = []
    call_count = {"n": 0}

    async def chat_with_retry(*, messages, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="thinking",
                tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
                reasoning_content="hidden reasoning",
                thinking_blocks=[{"type": "thinking", "thinking": "step"}],
                usage={"prompt_tokens": 5, "completion_tokens": 3},
            )
        captured_second_call[:] = messages
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="tool result")

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=3,
    ))

    assert result.final_content == "done"
    assert result.tools_used == ["list_dir"]
    assert result.tool_events == [
        {"name": "list_dir", "status": "ok", "detail": "tool result"}
    ]

    assistant_messages = [
        msg for msg in captured_second_call
        if msg.get("role") == "assistant" and msg.get("tool_calls")
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["reasoning_content"] == "hidden reasoning"
    assert assistant_messages[0]["thinking_blocks"] == [{"type": "thinking", "thinking": "step"}]
    assert any(
        msg.get("role") == "tool" and msg.get("content") == "tool result"
        for msg in captured_second_call
    )


def test_clear_old_tool_results_preserves_history_when_below_budget():
    from nanobot.agent.runner import clear_old_tool_results

    messages = [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "inspect"}}]},
        {"role": "tool", "content": "first result with useful detail"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "function": {"name": "preview"}}]},
        {"role": "tool", "content": "latest result"},
    ]

    clear_old_tool_results(
        messages,
        keep_last=1,
        provider=_TokenCountingProvider(),
        model="test-model",
        tools=[],
        trigger_tokens=10_000,
        target_tokens=8_000,
    )

    assert messages[2]["content"] == "first result with useful detail"


def test_clear_old_tool_results_compacts_before_full_clear():
    from nanobot.agent.runner import clear_old_tool_results

    messages = [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "inspect"}}]},
        {"role": "tool", "content": "\n".join(f"line {i} " + ("x" * 40) for i in range(40))},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "function": {"name": "preview"}}]},
        {"role": "tool", "content": "latest result"},
    ]

    clear_old_tool_results(
        messages,
        keep_last=1,
        provider=_TokenCountingProvider(),
        model="test-model",
        tools=[],
        trigger_tokens=100,
        target_tokens=900,
    )

    assert messages[2]["content"].startswith("[compacted to save context]")
    assert messages[2]["content"] != "[cleared to save context]"
    assert "line 0" in messages[2]["content"]


def test_clear_old_tool_results_falls_back_to_full_clear_if_needed():
    from nanobot.agent.runner import clear_old_tool_results

    messages = [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "inspect"}}]},
        {"role": "tool", "content": "\n".join(f"line {i} " + ("x" * 60) for i in range(80))},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "function": {"name": "preview"}}]},
        {"role": "tool", "content": "latest result"},
    ]

    clear_old_tool_results(
        messages,
        keep_last=1,
        provider=_TokenCountingProvider(),
        model="test-model",
        tools=[],
        trigger_tokens=100,
        target_tokens=40,
    )

    assert messages[2]["content"] == "[cleared to save context]"


@pytest.mark.asyncio
async def test_runner_calls_hooks_in_order():
    from nanobot.agent.hook import AgentHook, AgentHookContext
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    call_count = {"n": 0}
    events: list[tuple] = []

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="thinking",
                tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
            )
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="tool result")

    class RecordingHook(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            events.append(("before_iteration", context.iteration))

        async def before_execute_tools(self, context: AgentHookContext) -> None:
            events.append((
                "before_execute_tools",
                context.iteration,
                [tc.name for tc in context.tool_calls],
            ))

        async def after_iteration(self, context: AgentHookContext) -> None:
            events.append((
                "after_iteration",
                context.iteration,
                context.final_content,
                list(context.tool_results),
                list(context.tool_events),
                context.stop_reason,
            ))

        def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
            events.append(("finalize_content", context.iteration, content))
            return content.upper() if content else content

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=3,
        hook=RecordingHook(),
    ))

    assert result.final_content == "DONE"
    assert events == [
        ("before_iteration", 0),
        ("before_execute_tools", 0, ["list_dir"]),
        (
            "after_iteration",
            0,
            None,
            ["tool result"],
            [{"name": "list_dir", "status": "ok", "detail": "tool result"}],
            None,
        ),
        ("before_iteration", 1),
        ("finalize_content", 1, "done"),
        ("after_iteration", 1, "DONE", [], [], "completed"),
    ]


@pytest.mark.asyncio
async def test_runner_streaming_hook_receives_deltas_and_end_signal():
    from nanobot.agent.hook import AgentHook, AgentHookContext
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    streamed: list[str] = []
    endings: list[bool] = []

    async def chat_stream_with_retry(*, on_content_delta, **kwargs):
        await on_content_delta("he")
        await on_content_delta("llo")
        return LLMResponse(content="hello", tool_calls=[], usage={})

    provider.chat_stream_with_retry = chat_stream_with_retry
    provider.chat_with_retry = AsyncMock()
    tools = MagicMock()
    tools.get_definitions.return_value = []

    class StreamingHook(AgentHook):
        def wants_streaming(self) -> bool:
            return True

        async def on_stream(self, context: AgentHookContext, delta: str) -> None:
            streamed.append(delta)

        async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
            endings.append(resuming)

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=1,
        hook=StreamingHook(),
    ))

    assert result.final_content == "hello"
    assert streamed == ["he", "llo"]
    assert endings == [False]
    provider.chat_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_disables_parallel_execution_for_serial_only_tools():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner
    from nanobot.agent.tools.base import Tool
    from nanobot.agent.tools.registry import ToolRegistry

    state = {"active": 0, "max_active": 0}

    class _BaseTestTool(Tool):
        def __init__(self, name: str, *, supports_parallel_calls: bool) -> None:
            self._name = name
            self._supports_parallel_calls = supports_parallel_calls

        @property
        def name(self) -> str:
            return self._name

        @property
        def description(self) -> str:
            return self._name

        @property
        def parameters(self) -> dict:
            return {"type": "object", "properties": {}}

        @property
        def supports_parallel_calls(self) -> bool:
            return self._supports_parallel_calls

        async def execute(self, **kwargs):
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            await asyncio.sleep(0.01)
            state["active"] -= 1
            return self._name

    tools = ToolRegistry()
    tools.register(_BaseTestTool("serial_preview", supports_parallel_calls=False))
    tools.register(_BaseTestTool("regular_inspect", supports_parallel_calls=True))

    runner = AgentRunner(MagicMock())

    async def _fake_run_tool(_spec, _tool_call):
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return "ok", {"name": "noop", "status": "ok", "detail": "ok"}, None

    runner._run_tool = _fake_run_tool  # type: ignore[method-assign]

    await runner._execute_tools(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=1,
        concurrent_tools=True,
    ), [
        ToolCallRequest(id="call_1", name="serial_preview", arguments={}),
        ToolCallRequest(id="call_2", name="regular_inspect", arguments={}),
    ])

    assert state["max_active"] == 1


@pytest.mark.asyncio
async def test_runner_streaming_max_iterations_emits_final_end_signal():
    from nanobot.agent.hook import AgentHook, AgentHookContext
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="tool result")

    endings: list[bool] = []

    class StreamingHook(AgentHook):
        def wants_streaming(self) -> bool:
            return True

        async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
            endings.append(resuming)

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=1,
        hook=StreamingHook(),
    ))

    assert result.stop_reason == "max_iterations"
    assert endings == [True, False]


@pytest.mark.asyncio
async def test_runner_returns_max_iterations_fallback():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="still working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="tool result")

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=2,
    ))

    assert result.stop_reason == "max_iterations"
    assert result.final_content == (
        "I reached the maximum number of tool call iterations (2) "
        "without completing the task. You can try breaking the task into smaller steps."
    )


@pytest.mark.asyncio
async def test_runner_returns_structured_tool_error():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(side_effect=RuntimeError("boom"))

    runner = AgentRunner(provider)

    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=2,
        fail_on_tool_error=True,
    ))

    assert result.stop_reason == "tool_error"
    assert result.error == "Error: RuntimeError: boom"
    assert result.tool_events == [
        {"name": "list_dir", "status": "error", "detail": "boom"}
    ]


@pytest.mark.asyncio
async def test_runner_marks_nonzero_exitcode_json_result_as_tool_error():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    call_count = {"n": 0}

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="working",
                tool_calls=[ToolCallRequest(id="call_1", name="agent_browser", arguments={})],
            )
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value=json.dumps({
        "command": ["npx", "--yes", "agent-browser", "--help"],
        "exitCode": 1,
        "stderr": "npm error code E404\nnpm error 404 Not Found",
    }))

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=2,
    ))

    assert result.final_content == "done"
    assert result.tool_events == [
        {
            "name": "agent_browser",
            "status": "error",
            "detail": "exitCode 1: npm error code E404",
        }
    ]


@pytest.mark.asyncio
async def test_loop_max_iterations_message_stays_stable(tmp_path):
    loop = _make_loop(tmp_path)
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
    ))
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="ok")
    loop.max_iterations = 2

    final_content, _, _ = await loop._run_agent_loop([])

    assert final_content == (
        "I reached the maximum number of tool call iterations (2) "
        "without completing the task. You can try breaking the task into smaller steps."
    )


@pytest.mark.asyncio
async def test_loop_stream_filter_handles_think_only_prefix_without_crashing(tmp_path):
    loop = _make_loop(tmp_path)
    deltas: list[str] = []
    endings: list[bool] = []

    async def chat_stream_with_retry(*, on_content_delta, **kwargs):
        await on_content_delta("<think>hidden")
        await on_content_delta("</think>Hello")
        return LLMResponse(content="<think>hidden</think>Hello", tool_calls=[], usage={})

    loop.provider.chat_stream_with_retry = chat_stream_with_retry

    async def on_stream(delta: str) -> None:
        deltas.append(delta)

    async def on_stream_end(*, resuming: bool = False) -> None:
        endings.append(resuming)

    final_content, _, _ = await loop._run_agent_loop(
        [],
        on_stream=on_stream,
        on_stream_end=on_stream_end,
    )

    assert final_content == "Hello"
    # Deltas are now forwarded raw (think-tag stripping happens at the renderer level)
    assert deltas == ["<think>hidden", "</think>Hello"]
    assert endings == [False]


@pytest.mark.asyncio
async def test_loop_streaming_suppresses_intermediate_tool_preamble(tmp_path):
    loop = _make_loop(tmp_path)
    deltas: list[str] = []
    endings: list[bool] = []
    call_count = {"n": 0}

    async def chat_stream_with_retry(*, on_content_delta, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            await on_content_delta("Checking the repo now.")
            return LLMResponse(
                content="Checking the repo now.",
                tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
                usage={},
            )

        await on_content_delta("All set.")
        return LLMResponse(content="All set.", tool_calls=[], usage={})

    loop.provider.chat_stream_with_retry = chat_stream_with_retry
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="ok")

    async def on_stream(delta: str) -> None:
        deltas.append(delta)

    async def on_stream_end(*, resuming: bool = False) -> None:
        endings.append(resuming)

    final_content, tools_used, _ = await loop._run_agent_loop(
        [],
        on_stream=on_stream,
        on_stream_end=on_stream_end,
    )

    assert final_content == "All set."
    assert tools_used == ["list_dir"]
    # Deltas are now forwarded live for every iteration (not just the final one)
    assert deltas == ["Checking the repo now.", "All set."]
    assert endings == [True, False]
