"""Regression coverage for bounded prompts, consolidation, and session links."""

from __future__ import annotations

import json

import pytest

from nanobot.agent.hook import AgentHook
from nanobot.agent.memory import MemoryConsolidator, MemoryStore
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse
from nanobot.session.manager import SessionManager
from nanobot.utils.prompt_budget import (
    PromptBudget,
    PromptBudgetExceeded,
    measure_prompt,
    reduce_messages_to_budget,
)


class _PromptSpy:
    """Deterministic provider counter used to distinguish oversized prompts."""

    def __init__(self) -> None:
        self.chat_calls = 0

    def estimate_prompt_tokens(self, messages, _tools, _model):
        if any("oversized" in str(message.get("content", "")) for message in messages):
            return 10_000, "test-counter"
        return max(1, len(messages)), "test-counter"

    async def chat_with_retry(self, **_kwargs):
        self.chat_calls += 1
        return LLMResponse(content="unexpected transport", tool_calls=[])


class _DeterministicCounter:
    """Stable message counter independent of tokenizer/model availability."""

    @staticmethod
    def estimate_prompt_tokens(messages, tools, _model):
        total = len(messages) * 5 + len(json.dumps(tools or [], sort_keys=True)) // 10
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                total += max(1, (len(content) + 9) // 10)
            elif content is not None:
                total += max(1, len(json.dumps(content, sort_keys=True)) // 10)
            for call in message.get("tool_calls") or []:
                total += 7 + len(str(call.get("id", "")))
            if message.get("tool_call_id"):
                total += 3 + len(str(message["tool_call_id"]))
        return total, "deterministic-counter"


class _ProviderSpy(LLMProvider):
    """Provider implementation whose transport calls are observable."""

    def __init__(self) -> None:
        super().__init__()
        self.generation = GenerationSettings(
            max_tokens=32,
            context_window_tokens=4_096,
        )
        self.transport_calls: list[str] = []

    def estimate_prompt_tokens(self, messages, tools, model):
        return _DeterministicCounter.estimate_prompt_tokens(messages, tools, model)

    async def chat(self, *args, **kwargs) -> LLMResponse:
        self.transport_calls.append("chat")
        return LLMResponse(content="fitted", tool_calls=[])

    async def chat_stream(self, *args, on_content_delta=None, **kwargs) -> LLMResponse:
        self.transport_calls.append("stream")
        if on_content_delta is not None:
            await on_content_delta("fitted")
        return LLMResponse(content="fitted", tool_calls=[])

    def get_default_model(self) -> str:
        return "test-model"


class _RunnerTransportSpy:
    """Runner-only provider spy; its retry methods represent transport."""

    def __init__(self) -> None:
        self.transport_calls: list[str] = []

    def estimate_prompt_tokens(self, messages, tools, model):
        return _DeterministicCounter.estimate_prompt_tokens(messages, tools, model)

    async def chat_with_retry(self, **kwargs) -> LLMResponse:
        self.transport_calls.append("chat")
        return LLMResponse(content="fitted", tool_calls=[])

    async def chat_stream_with_retry(self, **kwargs) -> LLMResponse:
        self.transport_calls.append("stream")
        callback = kwargs.get("on_content_delta")
        if callback is not None:
            await callback("fitted")
        return LLMResponse(content="fitted", tool_calls=[])


class _StreamingHook(AgentHook):
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def wants_streaming(self) -> bool:
        return self.enabled


def _tool_turn(index: int) -> list[dict[str, object]]:
    call_a = f"call-{index}-a"
    call_b = f"call-{index}-b"
    return [
        {"role": "user", "content": f"turn-{index:04d} question"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": call_a, "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": call_b, "type": "function", "function": {"name": "b", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": call_a, "name": "a", "content": f"result-{index}-a"},
        {"role": "tool", "tool_call_id": call_b, "name": "b", "content": f"result-{index}-b"},
    ]


@pytest.mark.asyncio
async def test_over_budget_first_turn_is_reference_only_and_advances_offset(tmp_path) -> None:
    """A small but irreducibly large turn must never reach the LLM transport."""

    provider = _PromptSpy()
    sessions = SessionManager(tmp_path)
    consolidator = MemoryConsolidator(
        workspace=tmp_path,
        provider=provider,
        model="test-model",
        sessions=sessions,
        context_window_tokens=1_000,
        build_messages=lambda **_kwargs: [],
        get_tool_definitions=lambda: [],
        max_completion_tokens=0,
    )
    consolidator._SAFETY_BUFFER = 0

    session = sessions.get_or_create("cli:oversized")
    session.messages = [
        {"role": "user", "content": "oversized " + "x" * 5_000},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "keep"},
        {"role": "assistant", "content": "later"},
    ]

    def estimate(current_session):
        return (
            (1_000, "test-counter")
            if current_session.last_consolidated == 0
            else (0, "test-counter")
        )

    consolidator.estimate_session_prompt_tokens = estimate  # type: ignore[method-assign]
    original_consolidate = consolidator.store.consolidate
    seen_reference_flags: list[bool] = []

    async def spy_consolidate(*args, **kwargs):
        seen_reference_flags.append(bool(kwargs.get("reference_only")))
        return await original_consolidate(*args, **kwargs)

    consolidator.store.consolidate = spy_consolidate  # type: ignore[method-assign]

    await consolidator.maybe_consolidate_by_tokens(session)

    assert seen_reference_flags == [True]
    assert provider.chat_calls == 0
    assert session.last_consolidated == 2
    assert consolidator.store.pending_receipt_file.exists() is False


def test_generated_history_reducer_keeps_newest_legal_turns(tmp_path) -> None:
    """A 7,000-message history reduces deterministically without tool orphans."""

    del tmp_path  # The reducer is request-local and must not write a workspace file.
    counter = _DeterministicCounter()
    messages: list[dict[str, object]] = [{"role": "system", "content": "system"}]
    for index in range(1_750):
        messages.extend(_tool_turn(index))

    budget = PromptBudget(total_tokens=1_000, safety_buffer=0)
    reduced = reduce_messages_to_budget(
        messages,
        counter,
        "test-model",
        [],
        budget,
    )
    measurement = measure_prompt(counter, "test-model", reduced, [], budget)

    assert len(messages) == 7_001
    assert measurement.fits
    assert measurement.prompt_tokens <= budget.prompt_limit
    assert reduced[0] == messages[0]
    assert any(message.get("content") == "turn-1749 question" for message in reduced)
    assert any(message.get("content") == "turn-1748 question" for message in reduced)
    assert not any(message.get("content") == "turn-0001 question" for message in reduced)

    declared = {
        str(call["id"])
        for message in reduced
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    }
    returned = {
        str(message["tool_call_id"])
        for message in reduced
        if message.get("role") == "tool"
    }
    assert returned == declared


def test_consolidation_selection_uses_exact_budget_and_complete_turns(tmp_path) -> None:
    """Selection may fill the auxiliary budget, but never split a turn."""

    class _TurnBudgetProvider(_PromptSpy):
        def estimate_prompt_tokens(self, messages, _tools, _model):
            transcript = str(messages[-1].get("content", ""))
            return transcript.count("turn-") * 10, "turn-counter"

    provider = _TurnBudgetProvider()
    sessions = SessionManager(tmp_path)
    consolidator = MemoryConsolidator(
        workspace=tmp_path,
        provider=provider,
        model="test-model",
        sessions=sessions,
        context_window_tokens=1_000,
        build_messages=lambda **_kwargs: [],
        get_tool_definitions=lambda: [],
        max_completion_tokens=0,
    )
    consolidator._SAFETY_BUFFER = 0
    session = sessions.get_or_create("cli:exact-budget")
    for index in range(101):
        session.messages.extend([
            {"role": "user", "content": f"turn-{index}"},
            {"role": "assistant", "content": f"answer-{index}"},
        ])

    start, end, chunk = consolidator._select_consolidation_chunk(session) or (-1, -1, [])

    assert (start, end) == (0, 200)
    assert len(chunk) == consolidator._MAX_CONSOLIDATION_MESSAGES
    assert chunk[-1]["content"] == "answer-99"
    assert all(
        chunk[index]["role"] == "user"
        and chunk[index + 1]["role"] == "assistant"
        for index in range(0, len(chunk), 2)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_runner_rejects_irreducible_iteration_zero_before_transport(
    streaming: bool,
) -> None:
    """An irreducible first request fails locally for both runner paths."""

    provider = _RunnerTransportSpy()
    secret = "do-not-leak-" + "x" * 20_000
    with pytest.raises(PromptBudgetExceeded) as raised:
        await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": secret},
            ],
            tools=ToolRegistry(),
            model="test-model",
            max_iterations=1,
            max_tokens=0,
            hook=_StreamingHook(streaming),
            prompt_budget=PromptBudget(total_tokens=64, safety_buffer=0),
        ))

    assert provider.transport_calls == []
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value)
    assert not _ProviderSpy._is_transient_error(str(raised.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_runner_fitting_request_reaches_matching_transport(streaming: bool) -> None:
    provider = _RunnerTransportSpy()
    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "fit"}],
        tools=ToolRegistry(),
        model="test-model",
        max_iterations=1,
        max_tokens=0,
        hook=_StreamingHook(streaming),
        prompt_budget=PromptBudget(total_tokens=128, safety_buffer=0),
    ))

    assert result.final_content == "fitted"
    assert provider.transport_calls == ["stream" if streaming else "chat"]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_budget_boundary_has_zero_or_one_transport_call(streaming: bool) -> None:
    """Provider retry wrappers preflight oversized payloads before either path."""

    provider = _ProviderSpy()
    oversized = [{"role": "user", "content": "private-value-" + "x" * 50_000}]
    if streaming:
        with pytest.raises(PromptBudgetExceeded):
            await provider.chat_stream_with_retry(
                messages=oversized,
                model="test-model",
                max_tokens=32,
            )
    else:
        with pytest.raises(PromptBudgetExceeded):
            await provider.chat_with_retry(
                messages=oversized,
                model="test-model",
                max_tokens=32,
            )
    assert provider.transport_calls == []

    if streaming:
        result = await provider.chat_stream_with_retry(
            messages=[{"role": "user", "content": "fit"}],
            model="test-model",
            max_tokens=32,
        )
    else:
        result = await provider.chat_with_retry(
            messages=[{"role": "user", "content": "fit"}],
            model="test-model",
            max_tokens=32,
        )
    assert result.content == "fitted"
    assert provider.transport_calls == ["stream" if streaming else "chat"]


@pytest.mark.asyncio
async def test_receipt_recovery_is_idempotent_and_offset_monotonic(tmp_path) -> None:
    store = MemoryStore(tmp_path)
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("cli:receipt")
    session.messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    session.last_consolidated = 2
    sessions.save(session)

    receipt = store._make_receipt(
        messages=session.messages[:2],
        history_entry="[2026-01-01] recovered summary",
        memory_output="# Recovered\n",
        session_key=session.key,
        start_offset=0,
        end_offset=2,
        history_marker_id="receipt-marker",
    )
    store._write_receipt_unlocked(receipt)

    assert store.recover_pending_receipt(sessions)
    assert store.recover_pending_receipt(sessions)
    assert session.last_consolidated == 2
    assert store.history_file.read_text().count("nanobot-consolidation:receipt-marker") == 1
    assert store.memory_file.read_text() == "# Recovered\n"
    assert not store.pending_receipt_file.exists()

    # A stale receipt must never move an already advanced session backwards.
    stale = store._make_receipt(
        messages=session.messages[:1],
        history_entry="[2026-01-01] stale summary",
        memory_output="# Stale\n",
        session_key=session.key,
        start_offset=0,
        end_offset=1,
        history_marker_id="stale-marker",
    )
    store._write_receipt_unlocked(stale)
    assert store.recover_pending_receipt(sessions)
    assert session.last_consolidated == 2
    assert "stale summary" in store.history_file.read_text()


def test_scheduled_ring_is_bounded_and_legacy_history_is_collapsed(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("telegram:room")
    session.add_message("user", "ordinary question")
    session.add_message("assistant", "ordinary answer")
    for index in range(25):
        session.add_scheduled_run(
            job_id=f"job-{index}",
            run_id=f"run-{index}",
            instruction="instruction-" + "i" * 400,
            result="result-" + "r" * 400,
            detail_session_key=f"cron:job-{index}",
        )
    session.messages.extend([
        {"role": "user", "content": "[Scheduled task from cron:legacy]\nold instruction"},
        {"role": "assistant", "content": "old result"},
    ])
    manager.save(session)
    jsonl_before = manager._get_session_path(session.key).read_text()

    ring = session.recent_scheduled_runs()
    assert len(ring) == 20
    assert ring[0]["run_id"] == "run-5"
    assert len(ring[-1]["instruction"]) <= session.SCHEDULED_EXCERPT_CHARS
    assert len(ring[-1]["result"]) <= session.SCHEDULED_EXCERPT_CHARS

    model_history = session.get_model_history(max_messages=0)
    assert any(item.get("content") == "ordinary question" for item in model_history)
    assert not any(item.get("content") == "old result" for item in model_history)
    projections = [
        item for item in model_history
        if item.get("content", "").startswith("[Recent Scheduled Runs]")
    ]
    assert len(projections) == 1
    assert len(projections[0]["content"]) <= session.SCHEDULED_PROJECTION_CHARS

    assert manager._get_session_path(session.key).read_text() == jsonl_before


def test_session_managers_merge_append_and_scheduled_metadata(tmp_path) -> None:
    first_manager = SessionManager(tmp_path)
    second_manager = SessionManager(tmp_path)
    first = first_manager.get_or_create("cli:merge")
    second = second_manager.get_or_create("cli:merge")
    first.add_message("user", "base")
    first_manager.save(first)

    # Both managers now hold the same prefix, then append independently.
    second = second_manager.get_or_create("cli:merge")
    second.add_message("assistant", "from-second")
    second.add_scheduled_run(
        job_id="job-second",
        run_id="run-second",
        instruction="second instruction",
        result="second result",
        detail_session_key="cron:job-second",
    )
    second_manager.save(second)

    first.add_message("assistant", "from-first")
    first.add_scheduled_run(
        job_id="job-first",
        run_id="run-first",
        instruction="first instruction",
        result="first result",
        detail_session_key="cron:job-first",
    )
    first_manager.save(first)

    merged_manager = SessionManager(tmp_path)
    merged = merged_manager.get_or_create("cli:merge")
    assert [message["content"] for message in merged.messages] == [
        "base", "from-second", "from-first",
    ]
    assert {item["run_id"] for item in merged.recent_scheduled_runs()} == {
        "run-second", "run-first",
    }
