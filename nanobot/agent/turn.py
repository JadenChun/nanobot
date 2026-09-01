"""Typed contracts shared by the canonical per-turn execution path.

This module intentionally has no imports from the runner, tool registry, or
session manager.  Those modules can depend on these small value/state objects
without creating an import cycle while the legacy execution entrypoints are
being migrated.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias

__all__ = [
    "ApprovalGrant",
    "Callback",
    "DelegationBudget",
    "DeliveryState",
    "DeliveryTarget",
    "HistoryMode",
    "RunPolicyContext",
    "RunRecord",
    "RunStatus",
    "ScheduledTurnLink",
    "SessionRunRef",
    "ToolOutcome",
    "TraceMode",
    "TurnCallbacks",
    "TurnContext",
    "TurnRequest",
    "TurnResult",
    "TurnSource",
    "sanitize_metadata",
]


class TurnSource(StrEnum):
    """Origin of a turn request."""

    GATEWAY = "gateway"
    DIRECT = "direct"
    SDK = "sdk"
    API = "api"
    CRON = "cron"
    HEARTBEAT = "heartbeat"
    SYSTEM_COMPAT = "system_compat"


class HistoryMode(StrEnum):
    """Whether a turn uses the existing session history or starts clean."""

    SESSION = "session"
    FRESH = "fresh"


class TraceMode(StrEnum):
    """Amount of execution detail retained for a run."""

    NONE = "none"
    SANITIZED = "sanitized"


class RunStatus(StrEnum):
    """Lifecycle and terminal states for one run."""

    RUNNING = "running"
    COMPLETED = "completed"
    APPROVAL_REQUIRED = "approval_required"
    POLICY_BLOCKED = "policy_blocked"
    TOOL_ERROR = "tool_error"
    MAX_ITERATIONS = "max_iterations"
    CANCELLED = "cancelled"
    ERROR = "error"
    COMMAND = "command"


Callback: TypeAlias = Callable[..., Any] | Callable[..., Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class DeliveryTarget:
    """A user-visible destination for a completed turn."""

    channel: str = ""
    chat_id: str | None = None
    to: str | None = None
    # ``recipient`` is an ergonomic alias for SDK callers.  ``to`` remains
    # compatible with the existing channel/cron destination terminology.
    recipient: str | None = None
    thread_id: str | None = None
    message_id: str | None = None

    def __post_init__(self) -> None:
        if self.chat_id is None and self.to is not None:
            object.__setattr__(self, "chat_id", self.to)
        elif self.chat_id is None and self.recipient is not None:
            object.__setattr__(self, "chat_id", self.recipient)
        if self.to is None and self.chat_id is not None:
            object.__setattr__(self, "to", self.chat_id)
        elif self.to is None and self.recipient is not None:
            object.__setattr__(self, "to", self.recipient)
        elif self.recipient is None and self.to is not None:
            object.__setattr__(self, "recipient", self.to)
        if self.recipient is None and self.chat_id is not None:
            object.__setattr__(self, "recipient", self.chat_id)


@dataclass(frozen=True, slots=True)
class ScheduledTurnLink:
    """Reference to the schedule occurrence that initiated a turn."""

    job_id: str | None = None
    job_name: str = ""
    instruction: str = ""
    visible_session_key: str = ""
    additional_destinations: tuple[DeliveryTarget, ...] = ()
    occurrence_id: str | None = None
    scheduled_at: str | datetime | None = None
    # Some schedulers use ``schedule_id`` for the same stable identifier.
    schedule_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.scheduled_at, datetime):
            object.__setattr__(self, "scheduled_at", self.scheduled_at.isoformat())
        if not isinstance(self.additional_destinations, tuple):
            object.__setattr__(self, "additional_destinations", tuple(self.additional_destinations))
        if self.job_id is None and self.schedule_id is not None:
            object.__setattr__(self, "job_id", self.schedule_id)
        elif self.schedule_id is None and self.job_id is not None:
            object.__setattr__(self, "schedule_id", self.job_id)


@dataclass(frozen=True, slots=True)
class ApprovalGrant:
    """Approval evidence supplied when resuming a previously blocked turn."""

    approved: bool = False
    granted: bool | None = None
    grant_id: str | None = None
    resumed_run_id: str | None = None
    source: str = "none"
    scope: str | None = None
    granted_at: str | None = None
    expires_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.granted is None:
            object.__setattr__(self, "granted", self.approved)
        else:
            object.__setattr__(self, "approved", self.granted)


@dataclass(frozen=True, slots=True)
class TurnCallbacks:
    """Optional callbacks owned by the caller of a turn."""

    on_progress: Callback | None = None
    on_stream: Callback | None = None
    on_stream_end: Callback | None = None
    on_complete: Callback | None = None
    on_error: Callback | None = None


@dataclass(frozen=True, slots=True)
class TurnRequest:
    """Immutable input contract for one turn."""

    prompt: str = ""
    content: str = ""
    source: TurnSource = TurnSource.DIRECT
    session_key: str | None = None
    route: DeliveryTarget | None = None
    sender_id: str = "user"
    media: tuple[str, ...] = ()
    hooks: tuple[Any, ...] = ()
    history_mode: HistoryMode = HistoryMode.SESSION
    trace_mode: TraceMode = TraceMode.NONE
    delivery_target: DeliveryTarget | None = None
    scheduled_link: ScheduledTurnLink | None = None
    approval_grant: ApprovalGrant | None = None
    approval: ApprovalGrant = field(default_factory=ApprovalGrant)
    callbacks: TurnCallbacks = field(default_factory=TurnCallbacks)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    tool_names: tuple[str, ...] | None = None
    scheduled: ScheduledTurnLink | None = None
    # ``message`` keeps construction convenient for API callers that use the
    # channel-layer name while ``prompt`` remains the canonical field.
    message: str | None = None

    def __post_init__(self) -> None:
        if self.content and self.prompt and self.content != self.prompt:
            raise ValueError("prompt and content must match when both are supplied")
        canonical_content = self.content or self.prompt or self.message or ""
        object.__setattr__(self, "content", canonical_content)
        object.__setattr__(self, "prompt", canonical_content)
        if not isinstance(self.media, tuple):
            object.__setattr__(self, "media", tuple(self.media))
        if not isinstance(self.hooks, tuple):
            object.__setattr__(self, "hooks", tuple(self.hooks))
        if self.tool_names is not None and not isinstance(self.tool_names, tuple):
            object.__setattr__(self, "tool_names", tuple(self.tool_names))
        if self.route is None and self.delivery_target is not None:
            object.__setattr__(self, "route", self.delivery_target)
        elif self.delivery_target is None and self.route is not None:
            object.__setattr__(self, "delivery_target", self.route)
        if self.approval_grant is None and self.approval.granted:
            object.__setattr__(self, "approval_grant", self.approval)
        elif self.approval_grant is not None and not self.approval.granted:
            object.__setattr__(self, "approval", self.approval_grant)
        if self.scheduled is None and self.scheduled_link is not None:
            object.__setattr__(self, "scheduled", self.scheduled_link)
        elif self.scheduled_link is None and self.scheduled is not None:
            object.__setattr__(self, "scheduled_link", self.scheduled)
        if not isinstance(self.source, TurnSource):
            object.__setattr__(self, "source", TurnSource(str(self.source)))
        if not isinstance(self.history_mode, HistoryMode):
            object.__setattr__(self, "history_mode", HistoryMode(str(self.history_mode)))
        if not isinstance(self.trace_mode, TraceMode):
            object.__setattr__(self, "trace_mode", TraceMode(str(self.trace_mode)))
        if self.message is not None:
            if self.prompt and self.prompt != self.message:
                raise ValueError("prompt and message must match when both are supplied")
            object.__setattr__(self, "content", self.message)
            object.__setattr__(self, "prompt", self.message)
        object.__setattr__(self, "message", self.content)

    @property
    def delivery(self) -> DeliveryTarget | None:
        """Alias for the delivery target."""

        return self.delivery_target


@dataclass(slots=True)
class DelegationBudget:
    """Mutable per-run budget for optional foreground delegation calls."""

    max_calls: int = 6
    max_worker_corrections: int = 2
    calls_used: int = 0
    worker_corrections_used: int = 0
    # Canonical names used by the orchestrator contract.  The *_used fields
    # above remain descriptive aliases for callers that track corrections.
    max_worker_calls: int | None = None
    calls: int | None = None
    worker_calls: int | None = None

    def __post_init__(self) -> None:
        if self.max_worker_calls is None:
            self.max_worker_calls = self.max_worker_corrections
        else:
            self.max_worker_corrections = self.max_worker_calls
        if self.calls is None:
            self.calls = self.calls_used
        else:
            self.calls_used = self.calls
        if self.worker_calls is None:
            self.worker_calls = self.worker_corrections_used
        else:
            self.worker_corrections_used = self.worker_calls

    @property
    def calls_remaining(self) -> int:
        return max(0, self.max_calls - self.calls_used)

    @property
    def worker_corrections_remaining(self) -> int:
        return max(0, self.max_worker_corrections - self.worker_corrections_used)

    # Descriptive aliases for integrations that name these limits explicitly.
    @property
    def max_total_calls(self) -> int:
        return self.max_calls

    @property
    def corrections_used(self) -> int:
        return self.worker_corrections_used

    def consume_call(self) -> bool:
        """Consume one role call if budget remains."""

        if self.calls_remaining <= 0:
            return False
        self.calls_used += 1
        self.calls = self.calls_used
        return True

    def consume_worker_correction(self) -> bool:
        """Consume one worker correction allowance if budget remains."""

        if self.worker_corrections_remaining <= 0:
            return False
        self.worker_corrections_used += 1
        self.worker_calls = self.worker_corrections_used
        return True


@dataclass(slots=True)
class RunPolicyContext:
    """Mutable policy inputs and observations for a run."""

    approval_grant: ApprovalGrant | None = None
    workspace: Path | None = None
    approval_granted: bool | None = None
    source: TurnSource | None = None
    context_manager: Any | None = None
    allow_side_effects: bool = True
    allow_external: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.approval_granted is None:
            self.approval_granted = bool(self.approval_grant and self.approval_grant.granted)


@dataclass(slots=True)
class DeliveryState:
    """Mutable delivery bookkeeping for a turn."""

    primary: DeliveryTarget | None = None
    sent_messages: list[Any] = field(default_factory=list)
    target: DeliveryTarget | None = None
    delivered: bool = False
    progress_sent: bool = False
    streaming: bool = False
    message_id: str | None = None
    last_error: str | None = None

    def __post_init__(self) -> None:
        if self.primary is None:
            self.primary = self.target
        elif self.target is None:
            self.target = self.primary


@dataclass(slots=True)
class TurnContext:
    """Mutable execution context explicitly passed to context-aware tools."""

    request: TurnRequest | None = None
    run_id: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_key: str | None = None
    iteration: int = 0
    delegation_budget: DelegationBudget = field(default_factory=DelegationBudget)
    policy: RunPolicyContext = field(default_factory=RunPolicyContext)
    delivery: DeliveryState = field(default_factory=DeliveryState)
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    cancelled: bool = False
    lock_owner: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.request is not None:
            if not self.run_id and self.request.run_id:
                self.run_id = self.request.run_id
            if self.session_key is None:
                self.session_key = self.request.session_key
            if self.delivery.target is None:
                self.delivery.target = self.request.delivery_target
            if self.policy.approval_grant is None:
                self.policy.approval_grant = self.request.approval_grant
            if self.policy.source is None:
                self.policy.source = self.request.source
            if self.policy.approval_granted is None:
                self.policy.approval_granted = bool(
                    self.request.approval and self.request.approval.granted
                )

    @property
    def policy_context(self) -> RunPolicyContext:
        """Alias retained for code that spells out the context role."""

        return self.policy

    @property
    def source(self) -> TurnSource | None:
        return self.request.source if self.request is not None else None


@dataclass(frozen=True, slots=True)
class SessionRunRef:
    """Reference connecting a run record to detailed session messages."""

    session_key: str = ""
    run_id: str = ""

    @property
    def key(self) -> str:
        """Alias for the session key."""

        return self.session_key


_SENSITIVE_METADATA_PARTS = frozenset({
    "api_key",
    "apikey",
    "args",
    "argument",
    "arguments",
    "audio",
    "context",
    "content",
    "credential",
    "credentials",
    "detail",
    "image",
    "media",
    "password",
    "prompt",
    "raw",
    "request",
    "response",
    "reasoning",
    "secret",
    "token",
    "trace",
    "tool_call",
    "tool_calls",
})
_KEY_PART_RE = re.compile(r"[a-z0-9]+")


def _metadata_key_is_safe(key: str) -> bool:
    lowered = key.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    if lowered.startswith("_"):
        return False
    if normalized in _SENSITIVE_METADATA_PARTS:
        return False
    return not any(part in _SENSITIVE_METADATA_PARTS for part in _KEY_PART_RE.findall(lowered))


def _sanitize_metadata_value(value: Any, *, depth: int = 0) -> Any:
    """Return a bounded JSON-compatible value without sensitive/raw fields."""

    if depth > 4:
        return "[truncated]"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in list(value.items())[:64]:
            key = str(raw_key)
            if _metadata_key_is_safe(key):
                result[key] = _sanitize_metadata_value(raw_value, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize_metadata_value(item, depth=depth + 1) for item in list(value)[:64]]
    # Avoid repr() for arbitrary objects: it can contain credentials or raw
    # request details.  A type marker is sufficient for sanitized metadata.
    return f"<{type(value).__name__}>"


def sanitize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Sanitize user/runtime metadata before it is placed in a run record."""

    if not isinstance(metadata, Mapping):
        return {}
    sanitized = _sanitize_metadata_value(metadata)
    return sanitized if isinstance(sanitized, dict) else {}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_value(value: str | datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, datetime) else str(value)


@dataclass(slots=True)
class RunRecord:
    """Sanitized durable summary of one turn execution.

    Detailed prompts, tool arguments, reasoning, media, and raw trace remain
    in the session JSONL file.  A record only points to that detail via
    ``session_ref`` and contains bounded metadata suitable for indexing.
    """

    schema_version: int = 1
    run_id: str = ""
    status: RunStatus = RunStatus.RUNNING
    source: TurnSource = TurnSource.DIRECT
    detail_ref: SessionRunRef | None = None
    visible_session_key: str | None = None
    scheduled_job_id: str | None = None
    resumed_run_id: str | None = None
    tools_used: tuple[str, ...] = ()
    usage: Mapping[str, int] = field(default_factory=dict)
    error_type: str | None = None
    session_ref: SessionRunRef | None = None
    started_at: str | datetime = field(default_factory=_utc_timestamp)
    updated_at: str | datetime = field(default_factory=_utc_timestamp)
    completed_at: str | datetime | None = None
    stop_reason: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    delivery_target: DeliveryTarget | None = None
    scheduled_link: ScheduledTurnLink | None = None
    # Convenience constructor field; it is folded into ``session_ref`` and
    # never persisted as a duplicate reference.
    session_key: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            if self.detail_ref is not None:
                self.run_id = self.detail_ref.run_id
            else:
                raise ValueError("run record must include a run_id")
        if not isinstance(self.status, RunStatus):
            self.status = RunStatus(str(self.status))
        if not isinstance(self.source, TurnSource):
            self.source = TurnSource(str(self.source))
        if self.detail_ref is None and self.session_ref is not None:
            self.detail_ref = self.session_ref
        elif self.session_ref is None and self.detail_ref is not None:
            self.session_ref = self.detail_ref
        if isinstance(self.detail_ref, Mapping):
            self.detail_ref = SessionRunRef(
                session_key=str(self.detail_ref.get("session_key", "")),
                run_id=str(self.detail_ref.get("run_id", self.run_id)),
            )
        if isinstance(self.session_ref, Mapping):
            self.session_ref = SessionRunRef(
                session_key=str(self.session_ref.get("session_key", "")),
                run_id=str(self.session_ref.get("run_id", self.run_id)),
            )
        if self.detail_ref is None and self.session_ref is not None:
            self.detail_ref = self.session_ref
        elif self.session_ref is None and self.detail_ref is not None:
            self.session_ref = self.detail_ref
        if self.session_ref is None and self.session_key is not None:
            self.session_ref = SessionRunRef(session_key=self.session_key, run_id=self.run_id)
        elif self.session_ref is not None:
            self.session_key = self.session_ref.session_key
        if self.detail_ref is None and self.session_ref is not None:
            self.detail_ref = self.session_ref
        elif self.session_ref is None and self.detail_ref is not None:
            self.session_ref = self.detail_ref
        if isinstance(self.delivery_target, Mapping):
            self.delivery_target = DeliveryTarget(**dict(self.delivery_target))
        if isinstance(self.scheduled_link, Mapping):
            self.scheduled_link = ScheduledTurnLink(**dict(self.scheduled_link))
        if not isinstance(self.tools_used, tuple):
            self.tools_used = tuple(self.tools_used)
        self.usage = {
            str(key): int(value)
            for key, value in self.usage.items()
            if isinstance(value, (int, float))
        }
        if self.scheduled_job_id is None and self.scheduled_link is not None:
            self.scheduled_job_id = self.scheduled_link.job_id
        if self.error_type is None and self.error is not None:
            self.error_type = str(self.error)
        self.metadata = sanitize_metadata(self.metadata)

    @property
    def session(self) -> SessionRunRef | None:
        """Alias for the detail-session reference."""

        return self.session_ref

    @property
    def trace_ref(self) -> SessionRunRef | None:
        """Alias used by trace readers."""

        return self.session_ref

    def to_dict(self) -> dict[str, Any]:
        """Serialize only the intentionally small, sanitized record schema."""

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status.value,
            "source": self.source.value,
            "started_at": _timestamp_value(self.started_at),
            "updated_at": _timestamp_value(self.updated_at),
            "metadata": sanitize_metadata(self.metadata),
            "tools_used": list(self.tools_used),
            "usage": dict(self.usage),
        }
        detail_ref = self.detail_ref or self.session_ref
        if detail_ref is not None:
            payload["detail_ref"] = {
                "session_key": detail_ref.session_key,
                "run_id": detail_ref.run_id,
            }
        if self.visible_session_key is not None:
            payload["visible_session_key"] = self.visible_session_key
        if self.scheduled_job_id is not None:
            payload["scheduled_job_id"] = self.scheduled_job_id
        if self.resumed_run_id is not None:
            payload["resumed_run_id"] = self.resumed_run_id
        if self.completed_at is not None:
            payload["completed_at"] = _timestamp_value(self.completed_at)
        if self.stop_reason is not None:
            payload["stop_reason"] = str(self.stop_reason)
        if self.error_type is not None:
            payload["error_type"] = str(self.error_type)
        if self.delivery_target is not None:
            payload["delivery_target"] = {
                "channel": self.delivery_target.channel,
                "chat_id": self.delivery_target.chat_id,
                "thread_id": self.delivery_target.thread_id,
                "message_id": self.delivery_target.message_id,
            }
        if self.scheduled_link is not None:
            payload["scheduled_link"] = {
                "job_id": self.scheduled_link.job_id,
                "job_name": self.scheduled_link.job_name,
                "visible_session_key": self.scheduled_link.visible_session_key,
                "occurrence_id": self.scheduled_link.occurrence_id,
                "scheduled_at": _timestamp_value(self.scheduled_link.scheduled_at),
            }
        return payload

    as_dict = to_dict

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunRecord":
        """Load a record while ignoring unknown/non-schema fields."""

        if not isinstance(payload, Mapping) or not payload.get("run_id"):
            raise ValueError("run record must include a run_id")
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            run_id=str(payload["run_id"]),
            status=RunStatus(str(payload.get("status", RunStatus.RUNNING.value))),
            source=TurnSource(str(payload.get("source", TurnSource.DIRECT.value))),
            detail_ref=payload.get("detail_ref", payload.get("session_ref")),
            session_ref=payload.get("session_ref"),
            visible_session_key=payload.get("visible_session_key"),
            scheduled_job_id=payload.get("scheduled_job_id"),
            resumed_run_id=payload.get("resumed_run_id"),
            tools_used=tuple(payload.get("tools_used") or ()),
            usage=payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {},
            started_at=payload.get("started_at") or _utc_timestamp(),
            updated_at=payload.get("updated_at") or _utc_timestamp(),
            completed_at=payload.get("completed_at"),
            stop_reason=payload.get("stop_reason"),
            error=payload.get("error"),
            error_type=payload.get("error_type", payload.get("error")),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {},
            delivery_target=payload.get("delivery_target"),
            scheduled_link=payload.get("scheduled_link"),
        )


@dataclass(slots=True)
class TurnResult:
    """Value returned by a canonical turn execution."""

    run_id: str | None = None
    status: RunStatus = RunStatus.COMPLETED
    content: Any = None
    final_content: Any = None
    stop_reason: str | None = None
    error: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    tools_used: list[str] = field(default_factory=list)
    outbound: Any | None = None
    sent_messages: tuple[Any, ...] = ()
    record: RunRecord | None = None
    policy_metadata: Mapping[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, RunStatus):
            self.status = RunStatus(str(self.status))
        if self.content is None and self.final_content is not None:
            self.content = self.final_content
        elif self.final_content is None and self.content is not None:
            self.final_content = self.content
        if not isinstance(self.tools_used, list):
            self.tools_used = list(self.tools_used)
        if self.messages is not None and not isinstance(self.messages, list):
            self.messages = list(self.messages)

    def to_dict(self) -> dict[str, Any]:
        """Return the non-trace result fields for lightweight callers."""

        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "status": self.status.value,
            "content": self.content,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "usage": dict(self.usage),
            "tools_used": list(self.tools_used),
            "policy_metadata": sanitize_metadata(self.policy_metadata),
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """Normalized result from a tool execution.

    Legacy registry callers still receive ``content`` directly.  Context-aware
    calls receive this envelope so stop and policy information can travel with
    the result without changing existing tool implementations.
    """

    content: Any = None
    stop_reason: str | None = None
    policy_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def result(self) -> Any:
        """Alias for callers that call the payload a result."""

        return self.content
