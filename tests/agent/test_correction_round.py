"""Focused regression checks for the second-round persistence hardening."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.agent.memory import MemoryConsolidator, MemoryStore
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule
from nanobot.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)
from nanobot.session.manager import SessionManager, SessionWriteConflict
from nanobot.utils.prompt_budget import (
    AsyncPortableFileLock,
    PromptBudget,
    PromptBudgetExceeded,
    reduce_messages_to_budget,
)


class _DirectProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.generation = GenerationSettings(max_tokens=0, context_window_tokens=2048)
        self.calls = 0

    def estimate_prompt_tokens(self, messages, _tools, _model):
        return sum(len(str(message.get("content", ""))) for message in messages), "test"

    async def chat(self, messages, **_kwargs):
        self.calls += 1
        return LLMResponse(content="ok")

    async def chat_stream(self, messages, on_content_delta=None, **_kwargs):
        self.calls += 1
        if on_content_delta:
            await on_content_delta("ok")
        return LLMResponse(content="ok")

    def get_default_model(self) -> str:
        return "test"


@pytest.mark.asyncio
async def test_async_lock_cancellation_does_not_leak(tmp_path: Path) -> None:
    path = tmp_path / "lock"
    owner = AsyncPortableFileLock(path)
    await owner.__aenter__()
    waiter = asyncio.create_task(AsyncPortableFileLock(path).__aenter__())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await owner.__aexit__(None, None, None)

    replacement = AsyncPortableFileLock(path)
    await asyncio.wait_for(replacement.__aenter__(), timeout=1)
    await replacement.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_direct_provider_transport_is_budget_guarded() -> None:
    provider = _DirectProvider()
    with pytest.raises(PromptBudgetExceeded):
        await provider.chat([{"role": "user", "content": "x" * 5000}])
    assert provider.calls == 0
    assert (await provider.chat([{"role": "user", "content": "fit"}])).content == "ok"


def test_reducer_keeps_late_system_and_removes_invalid_tool_group() -> None:
    class Counter:
        @staticmethod
        def estimate_prompt_tokens(messages, _tools, _model):
            return len(messages), "test"

    messages = [
        {"role": "system", "content": "first"},
        {"role": "user", "content": "question"},
        {"role": "system", "content": "late policy"},
        {
            "role": "assistant",
            "content": "visible",
            "tool_calls": [{"id": "one"}, {"id": "one"}],
        },
        {"role": "tool", "tool_call_id": "one", "content": "orphaned"},
    ]
    reduced = reduce_messages_to_budget(
        messages,
        Counter(),
        "test",
        [],
        PromptBudget(total_tokens=100, safety_buffer=0),
    )
    assert [item["content"] for item in reduced[:2]] == ["first", "late policy"]
    assert {item.get("content") for item in reduced} >= {"question", "visible"}
    assert not any(item.get("role") == "tool" for item in reduced)
    assert not any(item.get("tool_calls") for item in reduced)


def test_session_generation_rejects_stale_append_and_merges_identical_messages(tmp_path: Path) -> None:
    first_manager = SessionManager(tmp_path)
    first = first_manager.get_or_create("cli:revision")
    first.add_message("user", "same")
    first_manager.save(first)

    stale_manager = SessionManager(tmp_path)
    stale = stale_manager.get_or_create("cli:revision")
    first.add_message("user", "same")
    stale.add_message("user", "same")
    first_manager.save(first)
    stale_manager.save(stale)
    merged = SessionManager(tmp_path).get_or_create("cli:revision")
    assert [item["content"] for item in merged.messages] == ["same", "same", "same"]

    stale.clear(expected_revision=stale.revision)
    latest = first_manager.get_or_create("cli:revision")
    latest.add_message("assistant", "new generation")
    first_manager.save(latest)
    with pytest.raises(SessionWriteConflict):
        stale_manager.save(stale)
    assert "same" in [item["content"] for item in SessionManager(tmp_path).get_or_create("cli:revision").messages]


@pytest.mark.asyncio
async def test_memory_receipt_digest_conflict_preserves_new_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("cli:digest")
    session.messages = [{"role": "user", "content": "source"}]
    sessions.save(session)
    store.write_long_term("newer")
    receipt = store._make_receipt(
        messages=session.messages,
        history_entry="old summary",
        memory_output="obsolete",
        session_key=session.key,
        start_offset=0,
        end_offset=1,
        session_generation=session.generation,
    )
    store._write_receipt_unlocked(receipt)
    store.write_long_term("newest")

    assert store.recover_pending_receipt(sessions)
    assert store.read_long_term() == "newest"
    assert session.last_consolidated == 0
    assert not store.pending_receipt_file.exists()


def test_consolidation_cap_returns_fitting_prefix(tmp_path: Path) -> None:
    class Counter:
        def estimate_prompt_tokens(self, messages, _tools, _model):
            return len(messages[-1].get("content", "")), "test"

    sessions = SessionManager(tmp_path)
    consolidator = MemoryConsolidator(
        tmp_path,
        Counter(),
        "test",
        sessions,
        context_window_tokens=10_000,
        build_messages=lambda **_kwargs: [],
        get_tool_definitions=lambda: [],
        max_completion_tokens=0,
    )
    consolidator._SAFETY_BUFFER = 0
    session = sessions.get_or_create("cli:cap")
    session.messages = [
        {"role": "user", "content": "first"},
        *({"role": "assistant", "content": "x"} for _ in range(189)),
        {"role": "user", "content": "next"},
        *({"role": "assistant", "content": "x"} for _ in range(20)),
    ]
    start, end, _chunk = consolidator._select_consolidation_chunk(session) or (-1, -1, [])
    assert (start, end) == (0, 190)


@pytest.mark.asyncio
async def test_cron_completion_merges_runtime_state_without_resurrecting_removed_job(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    entered = asyncio.Event()
    release = asyncio.Event()

    async def on_job(_job):
        entered.set()
        await release.wait()
        return SimpleNamespace(status="completed", content="done")

    service = CronService(store_path, on_job=on_job)
    job = service.add_job("race", CronSchedule(kind="every", every_ms=1000), "run")
    execution = asyncio.create_task(service.run_job(job.id))
    await entered.wait()
    external = CronService(store_path)
    assert external.remove_job(job.id)
    external.add_job("added", CronSchedule(kind="every", every_ms=1000), "add")
    release.set()
    await execution

    payload = json.loads(store_path.read_text(encoding="utf-8"))
    assert [item["name"] for item in payload["jobs"]] == ["added"]


@pytest.mark.asyncio
async def test_memory_read_stays_live_while_consolidations_serialize(tmp_path: Path) -> None:
    """A normal memory snapshot must not wait on a provider-held transaction."""

    class _PausingProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.prompts: list[list[dict[str, object]]] = []
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def chat_with_retry(self, **kwargs: object) -> LLMResponse:
            call = self.calls
            self.calls += 1
            self.prompts.append(kwargs["messages"])  # type: ignore[arg-type]
            if call == 0:
                self.first_started.set()
                await self.release_first.wait()
            update = "M1" if call == 0 else "M1\nM2"
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(
                    id=f"save-{call}",
                    name="save_memory",
                    arguments={
                        "history_entry": f"summary-{call}",
                        "memory_update": update,
                    },
                )],
            )

    store = MemoryStore(tmp_path)
    store.write_long_term("M0")
    provider = _PausingProvider()
    first = asyncio.create_task(store.consolidate(
        [{"role": "user", "content": "first"}],
        provider,
        "test-model",
    ))
    await asyncio.wait_for(provider.first_started.wait(), timeout=1)

    second = asyncio.create_task(store.consolidate(
        [{"role": "user", "content": "second"}],
        provider,
        "test-model",
    ))
    await asyncio.sleep(0)
    snapshot = asyncio.create_task(asyncio.to_thread(store.read_long_term))
    try:
        observed = await asyncio.wait_for(asyncio.shield(snapshot), timeout=0.1)
    except asyncio.TimeoutError:
        provider.release_first.set()
        await asyncio.gather(first, second, snapshot)
        pytest.fail("normal memory reads waited on the provider-held transaction lock")

    provider.release_first.set()
    first_result, second_result, observed = await asyncio.gather(first, second, snapshot)

    assert first_result is True
    assert second_result is True
    assert provider.calls == 2
    assert "M1" in str(provider.prompts[1][-1]["content"])
    assert store.read_long_term() == "M1\nM2"
    assert observed in {"M0", "M1", "M1\nM2"}
    history = store.history_file.read_text(encoding="utf-8")
    assert "summary-0" in history
    assert "summary-1" in history


def test_scheduled_ring_preserves_approval_state_and_legacy_defaults(tmp_path: Path) -> None:
    from nanobot.agent.loop import AgentLoop

    manager = SessionManager(tmp_path)
    loop = AgentLoop.__new__(AgentLoop)
    loop.sessions = manager
    loop._mirror_task_session_to_visible_chat(
        session_key="cron:job-approval",
        channel="telegram",
        chat_id="approval",
        task_text="blocked instruction",
        response_text="approval needed",
        approval_granted=False,
        run_id="run-blocked",
        status="approval_required",
        stop_reason="approval_required",
    )
    loop._mirror_task_session_to_visible_chat(
        session_key="cron:job-approved",
        channel="telegram",
        chat_id="approval",
        task_text="approved instruction",
        response_text="completed result",
        approval_granted=True,
        run_id="run-approved",
        status="completed",
    )

    loaded = SessionManager(tmp_path).get_or_create("telegram:approval")
    ring = loaded.recent_scheduled_runs()
    assert ring[0]["approval_granted"] is False
    assert ring[0]["status"] == "approval_required"
    assert ring[0]["stop_reason"] == "approval_required"
    assert ring[1]["approval_granted"] is True
    projection = loaded.get_model_history()[0]["content"]
    assert "status=approval_required" in projection
    assert "approval_granted=false" in projection
    assert "status=completed" in projection
    assert "approval_granted=true" in projection

    legacy = SessionManager(tmp_path).get_or_create("telegram:legacy-approval")
    legacy.metadata[legacy.SCHEDULED_RING_KEY] = [{
        "job_id": "legacy-job",
        "run_id": "legacy-run",
        "detail_ref": {"session_key": "cron:legacy-job", "run_id": "legacy-run"},
        "instruction": "legacy instruction",
        "result": "legacy result",
        "status": "completed",
    }]
    SessionManager(tmp_path).save(legacy)
    legacy_loaded = SessionManager(tmp_path).get_or_create("telegram:legacy-approval")
    assert legacy_loaded.recent_scheduled_runs()[0].get("approval_granted", False) is False
    legacy_projection = legacy_loaded.get_model_history()[0]["content"]
    assert "approval_granted=false" in legacy_projection
