import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from nanobot.cron.service import CronService
from nanobot.cron.types import CronDestination, CronSchedule


def test_add_job_rejects_unknown_timezone(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    with pytest.raises(ValueError, match="unknown timezone 'America/Vancovuer'"):
        service.add_job(
            name="tz typo",
            schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="America/Vancovuer"),
            message="hello",
        )

    assert service.list_jobs(include_disabled=True) == []


def test_add_job_accepts_valid_timezone(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    job = service.add_job(
        name="tz ok",
        schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="America/Vancouver"),
        message="hello",
    )

    assert job.schedule.tz == "America/Vancouver"
    assert job.state.next_run_at_ms is not None


def test_additional_destinations_are_persisted_and_loaded(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    service = CronService(store_path)

    job = service.add_job(
        name="multi-target",
        schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="Asia/Kuala_Lumpur"),
        message="Send the report",
        deliver=True,
        channel="telegram",
        to="6344587670",
        additional_destinations=[
            CronDestination(channel="telegram", to="-1001234567890"),
        ],
    )

    raw = json.loads(store_path.read_text(encoding="utf-8"))
    assert raw["jobs"][0]["payload"]["additionalDestinations"] == [
        {"channel": "telegram", "to": "-1001234567890"},
    ]

    loaded = CronService(store_path).get_job(job.id)
    assert loaded is not None
    assert loaded.payload.delivery_destinations() == [
        CronDestination(channel="telegram", to="6344587670"),
        CronDestination(channel="telegram", to="-1001234567890"),
    ]


def test_legacy_job_without_additional_destinations_still_loads(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    service = CronService(store_path)
    job = service.add_job(
        name="legacy",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="hello",
        deliver=True,
        channel="telegram",
        to="6344587670",
    )
    raw = json.loads(store_path.read_text(encoding="utf-8"))
    raw["jobs"][0]["payload"].pop("additionalDestinations")
    store_path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = CronService(store_path).get_job(job.id)
    assert loaded is not None
    assert loaded.payload.delivery_destinations() == [
        CronDestination(channel="telegram", to="6344587670"),
    ]


@pytest.mark.asyncio
async def test_execute_job_records_run_history(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    service = CronService(store_path, on_job=lambda _: asyncio.sleep(0))
    job = service.add_job(
        name="hist",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="hello",
    )
    await service.run_job(job.id)

    loaded = service.get_job(job.id)
    assert loaded is not None
    assert len(loaded.state.run_history) == 1
    rec = loaded.state.run_history[0]
    assert rec.status == "ok"
    assert rec.duration_ms >= 0
    assert rec.error is None


@pytest.mark.asyncio
async def test_execute_job_records_canonical_run_id_when_callback_returns_result(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"

    async def on_job(_):
        return SimpleNamespace(run_id="turn-123", content="done", status="completed")

    service = CronService(store_path, on_job=on_job)
    job = service.add_job(
        name="linked",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="hello",
    )

    await service.run_job(job.id)

    loaded = service.get_job(job.id)
    assert loaded is not None
    assert loaded.state.run_history[0].run_id == "turn-123"
    raw = json.loads(store_path.read_text(encoding="utf-8"))
    assert raw["jobs"][0]["state"]["runHistory"][0]["runId"] == "turn-123"


def test_legacy_run_history_without_run_id_loads_as_none(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    store_path.parent.mkdir(parents=True)
    store_path.write_text(json.dumps({
        "version": 1,
        "jobs": [{
            "id": "legacy",
            "name": "legacy",
            "enabled": True,
            "schedule": {"kind": "every", "everyMs": 60_000},
            "payload": {"kind": "agent_turn", "message": "hello"},
            "state": {
                "runHistory": [{"runAtMs": 1, "status": "ok", "durationMs": 2}],
            },
        }],
    }), encoding="utf-8")

    loaded = CronService(store_path).get_job("legacy")
    assert loaded is not None
    assert loaded.state.run_history[0].run_id is None


@pytest.mark.asyncio
async def test_run_history_records_errors(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"

    async def fail(_):
        raise RuntimeError("boom")

    service = CronService(store_path, on_job=fail)
    job = service.add_job(
        name="fail",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="hello",
    )
    await service.run_job(job.id)

    loaded = service.get_job(job.id)
    assert len(loaded.state.run_history) == 1
    assert loaded.state.run_history[0].status == "error"
    assert loaded.state.run_history[0].error == "boom"


@pytest.mark.asyncio
async def test_run_history_trimmed_to_max(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    service = CronService(store_path, on_job=lambda _: asyncio.sleep(0))
    job = service.add_job(
        name="trim",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="hello",
    )
    for _ in range(25):
        await service.run_job(job.id)

    loaded = service.get_job(job.id)
    assert len(loaded.state.run_history) == CronService._MAX_RUN_HISTORY


@pytest.mark.asyncio
async def test_run_history_persisted_to_disk(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    service = CronService(store_path, on_job=lambda _: asyncio.sleep(0))
    job = service.add_job(
        name="persist",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="hello",
    )
    await service.run_job(job.id)

    raw = json.loads(store_path.read_text())
    history = raw["jobs"][0]["state"]["runHistory"]
    assert len(history) == 1
    assert history[0]["status"] == "ok"
    assert "runAtMs" in history[0]
    assert "durationMs" in history[0]

    fresh = CronService(store_path)
    loaded = fresh.get_job(job.id)
    assert len(loaded.state.run_history) == 1
    assert loaded.state.run_history[0].status == "ok"


@pytest.mark.asyncio
async def test_running_service_honors_external_disable(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    called: list[str] = []

    async def on_job(job) -> None:
        called.append(job.id)

    service = CronService(store_path, on_job=on_job)
    job = service.add_job(
        name="external-disable",
        schedule=CronSchedule(kind="every", every_ms=200),
        message="hello",
    )
    await service.start()
    try:
        # Wait slightly to ensure file mtime is definitively different
        await asyncio.sleep(0.05)
        external = CronService(store_path)
        updated = external.enable_job(job.id, enabled=False)
        assert updated is not None
        assert updated.enabled is False

        await asyncio.sleep(0.35)
        assert called == []
    finally:
        service.stop()


def test_start_reloads_recomputes_and_persists_as_one_locked_rmw(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    service = CronService(store_path)
    removed = service.add_job(
        name="remove-me",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="remove",
    )
    service.add_job(
        name="keep-me",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="keep",
    )

    recompute_entered = threading.Event()
    continue_recompute = threading.Event()
    original_recompute = service._recompute_next_runs

    def pause_before_recompute() -> None:
        recompute_entered.set()
        assert continue_recompute.wait(timeout=2)
        original_recompute()

    monkeypatch.setattr(service, "_recompute_next_runs", pause_before_recompute)
    monkeypatch.setattr(service, "_arm_timer", lambda: None)
    start_errors: list[BaseException] = []

    def run_start() -> None:
        try:
            asyncio.run(service.start())
        except BaseException as exc:  # pragma: no cover - surfaced below
            start_errors.append(exc)

    starter = threading.Thread(target=run_start)
    starter.start()
    assert recompute_entered.wait(timeout=1)

    external = CronService(store_path)
    mutation_started = threading.Event()
    mutation_done = threading.Event()

    def mutate_from_cli() -> None:
        mutation_started.set()
        external.remove_job(removed.id)
        external.add_job(
            name="added-by-cli",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="added",
        )
        mutation_done.set()

    mutator = threading.Thread(target=mutate_from_cli)
    mutator.start()
    assert mutation_started.wait(timeout=1)
    # The fixed startup RMW owns the lock here, so the CLI mutation waits for
    # startup persistence.  The unfixed implementation lets it finish first
    # and then overwrites it with the stale startup cache.
    mutation_done.wait(timeout=1)
    continue_recompute.set()
    starter.join(timeout=2)
    mutator.join(timeout=2)

    assert not start_errors
    assert not starter.is_alive()
    assert not mutator.is_alive()
    payload = json.loads(store_path.read_text(encoding="utf-8"))
    assert [item["name"] for item in payload["jobs"]] == ["keep-me", "added-by-cli"]
    assert all(item["state"]["nextRunAtMs"] is not None for item in payload["jobs"])
