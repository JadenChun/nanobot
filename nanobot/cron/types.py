"""Cron types."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CronSchedule:
    """Schedule definition for a cron job."""
    kind: Literal["at", "every", "cron"]
    # For "at": timestamp in ms
    at_ms: int | None = None
    # For "every": interval in ms
    every_ms: int | None = None
    # For "cron": cron expression (e.g. "0 9 * * *")
    expr: str | None = None
    # Timezone for cron expressions
    tz: str | None = None


@dataclass(frozen=True)
class CronDestination:
    """One channel/chat destination for a cron result."""

    channel: str
    to: str


@dataclass
class CronPayload:
    """What to do when the job runs."""
    kind: Literal["system_event", "agent_turn"] = "agent_turn"
    message: str = ""
    # Deliver response to channel
    deliver: bool = False
    channel: str | None = None  # e.g. "whatsapp"
    to: str | None = None  # e.g. phone number
    # Additional result destinations. channel/to remains the execution context.
    additional_destinations: list[CronDestination] = field(default_factory=list)
    # Per-job overrides for the agent loop
    planning_mode: Literal["on", "off", "agent"] | None = None  # None = use global default
    skip_verification: bool = False

    def delivery_destinations(self) -> list[CronDestination]:
        """Return de-duplicated primary and additional result destinations."""
        if not self.deliver:
            return []

        destinations: list[CronDestination] = []
        if self.to:
            destinations.append(CronDestination(channel=self.channel or "cli", to=str(self.to)))
        destinations.extend(self.additional_destinations)

        unique: list[CronDestination] = []
        seen: set[tuple[str, str]] = set()
        for destination in destinations:
            key = (destination.channel, destination.to)
            if not destination.channel or not destination.to or key in seen:
                continue
            seen.add(key)
            unique.append(destination)
        return unique


@dataclass
class CronRunRecord:
    """A single execution record for a cron job."""
    run_at_ms: int
    status: Literal["ok", "error", "skipped"]
    duration_ms: int = 0
    error: str | None = None


@dataclass
class CronJobState:
    """Runtime state of a job."""
    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    last_status: Literal["ok", "error", "skipped"] | None = None
    last_error: str | None = None
    run_history: list[CronRunRecord] = field(default_factory=list)


@dataclass
class CronJob:
    """A scheduled job."""
    id: str
    name: str
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = field(default_factory=CronPayload)
    state: CronJobState = field(default_factory=CronJobState)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    delete_after_run: bool = False


@dataclass
class CronStore:
    """Persistent store for cron jobs."""
    version: int = 1
    jobs: list[CronJob] = field(default_factory=list)
