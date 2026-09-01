"""Cron service for scheduling agent tasks."""

import asyncio
import json
import os
import tempfile
import time
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

from loguru import logger

from nanobot.cron.types import (
    CronDestination,
    CronJob,
    CronJobState,
    CronPayload,
    CronRunRecord,
    CronSchedule,
    CronStore,
)
from nanobot.utils.prompt_budget import portable_file_lock


def _now_ms() -> int:
    return int(time.time() * 1000)


def _execution_field(execution: Any, *names: str) -> Any:
    """Read a compatibility field from a cron callback result."""
    if isinstance(execution, Mapping):
        for name in names:
            if name in execution and execution[name] is not None:
                return execution[name]
        return None
    for name in names:
        value = getattr(execution, name, None)
        if value is not None:
            return value
    return None


def _execution_run_id(execution: Any) -> str | None:
    """Extract an optional canonical turn ID without changing callback APIs."""
    value = _execution_field(execution, "run_id", "runId")
    return str(value) if value else None


def _execution_status(execution: Any) -> str:
    """Normalize a canonical turn result to a stable lifecycle value."""
    status = _execution_field(execution, "status", "run_status", "runStatus")
    status = getattr(status, "value", status)
    if status is None:
        status = _execution_field(execution, "stop_reason", "stopReason")
    status = getattr(status, "value", status)
    return str(status or "completed").lower()


def _execution_error(execution: Any) -> str | None:
    """Normalize a canonical result's terminal failure, if any."""
    error = _execution_field(execution, "error")
    if error:
        return str(error)
    status = _execution_status(execution)
    if status in {"error", "tool_error", "policy_blocked", "cancelled", "max_iterations"}:
        return f"cron execution stopped with {status}"
    return None


def _compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    """Compute next run time in ms."""
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None

    if schedule.kind == "every":
        if not schedule.every_ms or schedule.every_ms <= 0:
            return None
        # Next interval from now
        return now_ms + schedule.every_ms

    if schedule.kind == "cron" and schedule.expr:
        try:
            from zoneinfo import ZoneInfo

            from croniter import croniter
            # Use caller-provided reference time for deterministic scheduling
            base_time = now_ms / 1000
            tz = ZoneInfo(schedule.tz) if schedule.tz else datetime.now().astimezone().tzinfo
            base_dt = datetime.fromtimestamp(base_time, tz=tz)
            cron = croniter(schedule.expr, base_dt)
            next_dt = cron.get_next(datetime)
            return int(next_dt.timestamp() * 1000)
        except Exception:
            return None

    return None


def _validate_schedule_for_add(schedule: CronSchedule) -> None:
    """Validate schedule fields that would otherwise create non-runnable jobs."""
    if schedule.tz and schedule.kind != "cron":
        raise ValueError("tz can only be used with cron schedules")

    if schedule.kind == "cron" and schedule.tz:
        try:
            from zoneinfo import ZoneInfo

            ZoneInfo(schedule.tz)
        except Exception:
            raise ValueError(f"unknown timezone '{schedule.tz}'") from None


class CronService:
    """Service for managing and executing scheduled jobs."""

    _MAX_RUN_HISTORY = 20

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, Any]] | None = None,
    ):
        self.store_path = store_path
        self.lock_path = Path(f"{store_path}.lock")
        self.on_job = on_job
        self._store: CronStore | None = None
        self._last_mtime: float = 0.0
        self._timer_task: asyncio.Task | None = None
        self._running = False

    def _load_store(self) -> CronStore:
        """Load jobs from disk. Reloads automatically if file was modified externally."""
        if self._store and self.store_path.exists():
            mtime = self.store_path.stat().st_mtime
            if mtime != self._last_mtime:
                logger.info("Cron: jobs.json modified externally, reloading")
                self._store = None
        if self._store:
            return self._store

        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                jobs = []
                for j in data.get("jobs", []):
                    jobs.append(CronJob(
                        id=j["id"],
                        name=j["name"],
                        enabled=j.get("enabled", True),
                        schedule=CronSchedule(
                            kind=j["schedule"]["kind"],
                            at_ms=j["schedule"].get("atMs"),
                            every_ms=j["schedule"].get("everyMs"),
                            expr=j["schedule"].get("expr"),
                            tz=j["schedule"].get("tz"),
                        ),
                        payload=CronPayload(
                            kind=j["payload"].get("kind", "agent_turn"),
                            message=j["payload"].get("message", ""),
                            deliver=j["payload"].get("deliver", False),
                            channel=j["payload"].get("channel"),
                            to=j["payload"].get("to"),
                            additional_destinations=[
                                CronDestination(
                                    channel=str(destination["channel"]),
                                    to=str(destination["to"]),
                                )
                                for destination in j["payload"].get(
                                    "additionalDestinations",
                                    j["payload"].get("additional_destinations", []),
                                ) or []
                                if isinstance(destination, dict)
                                and destination.get("channel")
                                and destination.get("to")
                            ],
                        ),
                        state=CronJobState(
                            next_run_at_ms=j.get("state", {}).get("nextRunAtMs"),
                            last_run_at_ms=j.get("state", {}).get("lastRunAtMs"),
                            last_status=j.get("state", {}).get("lastStatus"),
                            last_run_status=j.get("state", {}).get(
                                "lastRunStatus", j.get("state", {}).get("runStatus")
                            ),
                            last_error=j.get("state", {}).get("lastError"),
                            run_history=[
                                CronRunRecord(
                                    run_at_ms=r["runAtMs"],
                                    status=r["status"],
                                    duration_ms=r.get("durationMs", 0),
                                    error=r.get("error"),
                                    run_id=r.get("runId", r.get("run_id")),
                                    run_status=r.get("runStatus", r.get("run_status")),
                                    occurrence_id=r.get(
                                        "occurrenceId", r.get("occurrence_id")
                                    ),
                                )
                                for r in j.get("state", {}).get("runHistory", [])
                            ],
                        ),
                        created_at_ms=j.get("createdAtMs", 0),
                        updated_at_ms=j.get("updatedAtMs", 0),
                        delete_after_run=j.get("deleteAfterRun", False),
                    ))
                for job in jobs:
                    # Older stores and interrupted writers may contain the
                    # same occurrence more than once.  Keep the latest copy
                    # by stable occurrence ID while preserving legacy records
                    # that have no identifier.
                    deduped: list[CronRunRecord] = []
                    seen_occurrences: set[str] = set()
                    for record in reversed(job.state.run_history):
                        if record.occurrence_id and record.occurrence_id in seen_occurrences:
                            continue
                        if record.occurrence_id:
                            seen_occurrences.add(record.occurrence_id)
                        deduped.append(record)
                    job.state.run_history = list(reversed(deduped))[-self._MAX_RUN_HISTORY:]
                self._store = CronStore(jobs=jobs)
            except Exception as e:
                logger.warning("Failed to load cron store: {}", e)
                self._store = CronStore()
        else:
            self._store = CronStore()

        return self._store

    def _load_store_locked(self) -> CronStore:
        """Reload jobs from disk while the caller holds ``lock_path``."""
        self._store = None
        return self._load_store()

    def _save_store(self, *, _lock_held: bool = False) -> None:
        """Save jobs to disk, locking before serializing when needed."""
        if not _lock_held:
            with portable_file_lock(self.lock_path):
                self._save_store_locked()
            return
        self._save_store_locked()

    def _save_store_locked(self) -> None:
        """Serialize and atomically save jobs while ``lock_path`` is held."""
        if not self._store:
            return

        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": self._store.version,
            "jobs": [
                {
                    "id": j.id,
                    "name": j.name,
                    "enabled": j.enabled,
                    "schedule": {
                        "kind": j.schedule.kind,
                        "atMs": j.schedule.at_ms,
                        "everyMs": j.schedule.every_ms,
                        "expr": j.schedule.expr,
                        "tz": j.schedule.tz,
                    },
                    "payload": {
                        "kind": j.payload.kind,
                        "message": j.payload.message,
                        "deliver": j.payload.deliver,
                        "channel": j.payload.channel,
                        "to": j.payload.to,
                        "additionalDestinations": [
                            {"channel": destination.channel, "to": destination.to}
                            for destination in j.payload.additional_destinations
                        ],
                    },
                    "state": {
                        "nextRunAtMs": j.state.next_run_at_ms,
                        "lastRunAtMs": j.state.last_run_at_ms,
                        "lastStatus": j.state.last_status,
                        "lastRunStatus": j.state.last_run_status,
                        "lastError": j.state.last_error,
                        "runHistory": [
                            {
                                "runAtMs": r.run_at_ms,
                                "status": r.status,
                                "durationMs": r.duration_ms,
                                "error": r.error,
                                **({"runId": r.run_id} if r.run_id is not None else {}),
                                **({"runStatus": r.run_status} if r.run_status is not None else {}),
                                **({"occurrenceId": r.occurrence_id} if r.occurrence_id is not None else {}),
                            }
                            for r in j.state.run_history
                        ],
                    },
                    "createdAtMs": j.created_at_ms,
                    "updatedAtMs": j.updated_at_ms,
                    "deleteAfterRun": j.delete_after_run,
                }
                for j in self._store.jobs
            ]
        }

        payload = json.dumps(data, indent=2, ensure_ascii=False)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.store_path.name}.",
            suffix=".tmp",
            dir=self.store_path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.store_path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
        self._last_mtime = self.store_path.stat().st_mtime

    async def start(self) -> None:
        """Start the cron service."""
        self._running = True
        # Startup is one short locked read-modify-write.  Job execution is
        # never performed under this lock; only reload, schedule recompute,
        # and the atomic persistence belong to the transaction.
        with portable_file_lock(self.lock_path):
            self._load_store_locked()
            self._recompute_next_runs()
            self._save_store(_lock_held=True)
        self._arm_timer()
        logger.info("Cron service started with {} jobs", len(self._store.jobs if self._store else []))

    def stop(self) -> None:
        """Stop the cron service."""
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

    def _recompute_next_runs(self) -> None:
        """Recompute next run times for all enabled jobs."""
        if not self._store:
            return
        now = _now_ms()
        for job in self._store.jobs:
            if job.enabled:
                job.state.next_run_at_ms = _compute_next_run(job.schedule, now)

    def _get_next_wake_ms(self) -> int | None:
        """Get the earliest next run time across all jobs."""
        if not self._store:
            return None
        times = [j.state.next_run_at_ms for j in self._store.jobs
                 if j.enabled and j.state.next_run_at_ms]
        return min(times) if times else None

    def _arm_timer(self) -> None:
        """Schedule the next timer tick."""
        if self._timer_task:
            self._timer_task.cancel()

        next_wake = self._get_next_wake_ms()
        if not next_wake or not self._running:
            return

        delay_ms = max(0, next_wake - _now_ms())
        delay_s = delay_ms / 1000

        async def tick():
            await asyncio.sleep(delay_s)
            if self._running:
                await self._on_timer()

        self._timer_task = asyncio.create_task(tick())

    async def _on_timer(self) -> None:
        """Handle timer tick - run due jobs."""
        self._load_store()
        if not self._store:
            return

        now = _now_ms()
        due_jobs = [
            j for j in self._store.jobs
            if j.enabled and j.state.next_run_at_ms and now >= j.state.next_run_at_ms
        ]

        for job in due_jobs:
            await self._execute_job(job)

        self._arm_timer()

    def _persist_execution_state(self, job: CronJob, occurrence_id: str) -> None:
        """Merge one completed occurrence into the latest jobs.json.

        The LLM callback runs outside the store lock.  At completion, reload
        the file and patch only this job's runtime occurrence state, so a CLI
        disable/add/remove made during execution remains authoritative.
        """
        with portable_file_lock(self.lock_path):
            latest = CronService(self.store_path)._load_store()
            current = next((item for item in latest.jobs if item.id == job.id), None)
            if current is None:
                self._store = latest
                return

            existing = {
                record.occurrence_id
                for record in current.state.run_history
                if record.occurrence_id
            }
            if occurrence_id not in existing:
                current.state.last_run_at_ms = job.state.last_run_at_ms
                current.state.last_status = job.state.last_status
                current.state.last_run_status = job.state.last_run_status
                current.state.last_error = job.state.last_error
                current.state.run_history.extend(
                    record
                    for record in job.state.run_history
                    if record.occurrence_id == occurrence_id
                )
                current.state.run_history = current.state.run_history[-self._MAX_RUN_HISTORY:]
                current.updated_at_ms = max(current.updated_at_ms, job.updated_at_ms)

            # Recompute only runtime scheduling state from the latest schedule
            # and enabled flag.  Never copy stale payload/schedule/enabled data
            # from the object that was executing while the CLI was mutating.
            if current.enabled:
                current.state.next_run_at_ms = _compute_next_run(
                    current.schedule, _now_ms()
                )
            else:
                current.state.next_run_at_ms = None

            if current.schedule.kind == "at" and current.delete_after_run and current.enabled:
                latest.jobs = [item for item in latest.jobs if item.id != current.id]

            self._store = latest
            self._save_store(_lock_held=True)

    async def _execute_job(self, job: CronJob) -> None:
        """Execute a single job."""
        start_ms = _now_ms()
        occurrence_id = str(start_ms)
        if any(record.occurrence_id == occurrence_id for record in job.state.run_history):
            # Millisecond clocks can repeat during a tight manual-run loop;
            # retain the start timestamp as the stable prefix while keeping
            # each actual occurrence independently deduplicable.
            occurrence_id = f"{occurrence_id}-{uuid.uuid4().hex[:8]}"
        job._occurrence_id = occurrence_id  # type: ignore[attr-defined]
        # Make the occurrence visible to the callback so scheduled links use
        # the actual execution start rather than a previous run's timestamp.
        job.state.last_run_at_ms = start_ms
        logger.info("Cron: executing job '{}' ({})", job.name, job.id)
        execution_run_id: str | None = None
        execution_status = "completed"

        try:
            if self.on_job:
                execution = await self.on_job(job)
                execution_run_id = _execution_run_id(execution)
                execution_status = _execution_status(execution)
                execution_error = _execution_error(execution)
                # Structured canonical terminal statuses are recorded as-is;
                # only an otherwise-successful legacy callback that exposes
                # an error should take the exception path.
                if execution_error is not None and execution_status == "completed":
                    raise RuntimeError(execution_error)

            if execution_status == "approval_required":
                job.state.last_status = "skipped"
                job.state.last_error = "approval required"
                logger.info("Cron: job '{}' requires approval", job.name)
            elif execution_status == "completed":
                job.state.last_status = "ok"
                job.state.last_error = None
                logger.info("Cron: job '{}' completed", job.name)
            else:
                job.state.last_status = "error"
                job.state.last_error = _execution_error(execution) if self.on_job else (
                    f"cron execution stopped with {execution_status}"
                )
                logger.error("Cron: job '{}' stopped with {}", job.name, execution_status)

        except Exception as e:
            execution_status = "error"
            job.state.last_status = "error"
            job.state.last_error = str(e)
            logger.error("Cron: job '{}' failed: {}", job.name, e)

        end_ms = _now_ms()
        job.updated_at_ms = end_ms
        job.state.last_run_status = execution_status

        job.state.run_history.append(CronRunRecord(
            run_at_ms=start_ms,
            status=job.state.last_status,
            duration_ms=end_ms - start_ms,
            error=job.state.last_error,
            run_id=execution_run_id,
            run_status=execution_status,
            occurrence_id=occurrence_id,
        ))
        job.state.run_history = job.state.run_history[-self._MAX_RUN_HISTORY:]

        # Handle one-shot jobs
        if job.schedule.kind == "at":
            if job.delete_after_run:
                self._store.jobs = [j for j in self._store.jobs if j.id != job.id]
            else:
                job.enabled = False
                job.state.next_run_at_ms = None
        else:
            # Compute next run
            job.state.next_run_at_ms = _compute_next_run(job.schedule, _now_ms())

        self._persist_execution_state(job, occurrence_id)

    # ========== Public API ==========

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        """List all jobs."""
        store = self._load_store()
        jobs = store.jobs if include_disabled else [j for j in store.jobs if j.enabled]
        return sorted(jobs, key=lambda j: j.state.next_run_at_ms or float('inf'))

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        delete_after_run: bool = False,
        additional_destinations: list[CronDestination] | None = None,
    ) -> CronJob:
        """Add a new job."""
        _validate_schedule_for_add(schedule)
        now = _now_ms()

        job = CronJob(
            id=str(uuid.uuid4())[:8],
            name=name,
            enabled=True,
            schedule=schedule,
            payload=CronPayload(
                kind="agent_turn",
                message=message,
                deliver=deliver,
                channel=channel,
                to=to,
                additional_destinations=list(additional_destinations or []),
            ),
            state=CronJobState(next_run_at_ms=_compute_next_run(schedule, now)),
            created_at_ms=now,
            updated_at_ms=now,
            delete_after_run=delete_after_run,
        )

        with portable_file_lock(self.lock_path):
            store = CronService(self.store_path)._load_store()
            store.jobs.append(job)
            self._store = store
            self._save_store(_lock_held=True)
        self._arm_timer()

        logger.info("Cron: added job '{}' ({})", name, job.id)
        return job

    def remove_job(self, job_id: str) -> bool:
        """Remove a job by ID."""
        with portable_file_lock(self.lock_path):
            store = CronService(self.store_path)._load_store()
            before = len(store.jobs)
            store.jobs = [j for j in store.jobs if j.id != job_id]
            removed = len(store.jobs) < before

            if removed:
                self._store = store
                self._save_store(_lock_held=True)

        if removed:
            self._arm_timer()
            logger.info("Cron: removed job {}", job_id)

        return removed

    def enable_job(self, job_id: str, enabled: bool = True) -> CronJob | None:
        """Enable or disable a job."""
        with portable_file_lock(self.lock_path):
            store = CronService(self.store_path)._load_store()
            for job in store.jobs:
                if job.id == job_id:
                    job.enabled = enabled
                    job.updated_at_ms = _now_ms()
                    if enabled:
                        job.state.next_run_at_ms = _compute_next_run(job.schedule, _now_ms())
                    else:
                        job.state.next_run_at_ms = None
                    self._store = store
                    self._save_store(_lock_held=True)
                    self._arm_timer()
                    return job
        return None

    async def run_job(self, job_id: str, force: bool = False) -> bool:
        """Manually run a job."""
        store = self._load_store()
        for job in store.jobs:
            if job.id == job_id:
                if not force and not job.enabled:
                    return False
                await self._execute_job(job)
                self._arm_timer()
                return True
        return False

    def get_job(self, job_id: str) -> CronJob | None:
        """Get a job by ID."""
        store = self._load_store()
        return next((j for j in store.jobs if j.id == job_id), None)

    def status(self) -> dict:
        """Get service status."""
        store = self._load_store()
        return {
            "enabled": self._running,
            "jobs": len(store.jobs),
            "next_wake_at_ms": self._get_next_wake_ms(),
        }
