"""Logical prompt-budget measurement and deterministic request reduction.

The configured ``maxTokens.input`` value is the total logical context budget,
including the completion reserve.  This module is deliberately independent of
the agent/session layers so provider boundaries, delegated roles, and memory
consolidation can all apply the same rule.

Reduction helpers return detached message dictionaries.  Persisted session
JSONL is therefore never changed merely because a request needed a smaller
view of history.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanobot.utils.helpers import estimate_prompt_tokens_chain

DEFAULT_PROMPT_SAFETY_BUFFER = 1024


class PortableFileLock:
    """Small stdlib-only advisory lock usable by CLI and gateway processes."""

    # ``AsyncPortableFileLock`` may release the lock from a different worker
    # thread than the one which acquired it.  A plain lock is deliberately
    # used here: unlike ``RLock``, its ownership is not tied to a thread.
    _thread_locks: dict[str, threading.Lock] = {}
    _thread_guard = threading.Lock()

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)
        self._fd: int | None = None
        with self._thread_guard:
            self._local_lock = self._thread_locks.setdefault(self.path, threading.Lock())

    def acquire(self) -> None:
        self._local_lock.acquire()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_EX)
        except BaseException:
            os.close(self._fd)
            self._fd = None
            self._local_lock.release()
            raise

    def release(self) -> None:
        fd = self._fd
        self._fd = None
        try:
            if fd is not None:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
        finally:
            self._local_lock.release()

    def __enter__(self) -> "PortableFileLock":
        self.acquire()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.release()


@contextmanager
def portable_file_lock(path: str | os.PathLike[str]):
    """Context-manager spelling for synchronous persistence paths."""
    lock = PortableFileLock(path)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()


class AsyncPortableFileLock:
    """Async wrapper that avoids blocking the event loop while waiting."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._lock = PortableFileLock(path)
        self._acquire_task: Any | None = None
        self._acquired = False

    async def __aenter__(self) -> "AsyncPortableFileLock":
        import asyncio

        # ``to_thread`` cancellation does not cancel the underlying blocking
        # acquire.  Shield it and arrange for a late successful acquire to be
        # released, otherwise cancelling a waiter can leak a process lock.
        task = asyncio.create_task(asyncio.to_thread(self._lock.acquire))
        self._acquire_task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            def _release_late(done: asyncio.Future) -> None:
                if done.cancelled() or done.exception() is not None:
                    return
                try:
                    self._lock.release()
                except Exception:
                    pass

            task.add_done_callback(_release_late)
            raise
        self._acquired = True
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        import asyncio

        if not self._acquired:
            return
        self._acquired = False
        await asyncio.to_thread(self._lock.release)


@dataclass(frozen=True, slots=True)
class PromptBudget:
    """Total logical context and the reserve held back for a completion."""

    total_tokens: int
    completion_reserve: int = 0
    safety_buffer: int = DEFAULT_PROMPT_SAFETY_BUFFER

    @property
    def prompt_limit(self) -> int:
        """Maximum logical input tokens allowed before provider transport."""
        return max(
            0,
            int(self.total_tokens)
            - max(0, int(self.completion_reserve))
            - max(0, int(self.safety_buffer)),
        )

    # Compatibility spellings used by older context-budget callers.
    @property
    def effective_prompt_limit(self) -> int:
        return self.prompt_limit

    @property
    def available_budget(self) -> int:
        return self.prompt_limit

    @property
    def context_window_tokens(self) -> int:
        return self.total_tokens

    @property
    def max_tokens(self) -> int:
        return self.total_tokens

    @classmethod
    def from_generation(
        cls,
        generation: Any,
        *,
        total_tokens: int | None = None,
        completion_reserve: int | None = None,
        safety_buffer: int = DEFAULT_PROMPT_SAFETY_BUFFER,
    ) -> "PromptBudget | None":
        """Build a budget from provider generation settings when configured."""
        total = total_tokens
        if total is None:
            total = getattr(generation, "context_window_tokens", 0)
        if not isinstance(total, (int, float)) or int(total) <= 0:
            return None
        reserve = completion_reserve
        if reserve is None:
            reserve = getattr(generation, "max_tokens", 0)
        return cls(
            total_tokens=int(total),
            completion_reserve=max(0, int(reserve or 0)),
            safety_buffer=safety_buffer,
        )


@dataclass(frozen=True, slots=True)
class PromptMeasurement:
    """A measured prompt and the budget against which it was checked."""

    prompt_tokens: int
    source: str
    limit: int | None = None
    total_tokens: int | None = None
    completion_reserve: int = 0
    safety_buffer: int = DEFAULT_PROMPT_SAFETY_BUFFER

    @property
    def fits(self) -> bool:
        return self.limit is None or self.prompt_tokens <= self.limit

    @property
    def over_by(self) -> int:
        if self.limit is None:
            return 0
        return max(0, self.prompt_tokens - self.limit)


class PromptBudgetExceeded(Exception):  # noqa: N818 - public contract name
    """Raised locally when a prompt cannot fit; never includes prompt content."""

    DEFAULT_MESSAGE = "Prompt exceeds the configured context budget."

    def __init__(self, measurement: PromptMeasurement | None = None) -> None:
        self.measurement = measurement
        # Keep exception text deterministic and content-free.  In particular,
        # do not include the offending user message or serialized tool schema.
        super().__init__(self.DEFAULT_MESSAGE)


def _fallback_prompt_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> int:
    """Conservative deterministic fallback if a tokenizer is unavailable."""
    try:
        payload = json.dumps(
            {"messages": messages, "tools": tools or []},
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        return max(1, (len(payload) + 3) // 4 + len(messages) * 4)
    except Exception:
        return max(1, sum(len(str(message)) for message in messages) // 4)


def measure_prompt(
    provider: Any,
    model: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    budget: PromptBudget | None = None,
) -> PromptMeasurement:
    """Measure a fully rendered message/tool payload without mutating it."""
    measured, source = estimate_prompt_tokens_chain(provider, model, messages, tools)
    if not isinstance(measured, int) or measured <= 0:
        measured = _fallback_prompt_tokens(messages, tools)
        source = "fallback"
    return PromptMeasurement(
        prompt_tokens=int(measured),
        source=str(source),
        limit=budget.prompt_limit if budget is not None else None,
        total_tokens=budget.total_tokens if budget is not None else None,
        completion_reserve=budget.completion_reserve if budget is not None else 0,
        safety_buffer=budget.safety_buffer if budget is not None else DEFAULT_PROMPT_SAFETY_BUFFER,
    )


# Descriptive aliases make the boundary helper easy to discover from tests and
# integrations while keeping one implementation.
measure_prompt_tokens = measure_prompt


def assert_prompt_fits(
    provider: Any,
    model: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    budget: PromptBudget | None = None,
) -> PromptMeasurement:
    """Measure and raise :class:`PromptBudgetExceeded` when over the limit."""
    measurement = measure_prompt(provider, model, messages, tools, budget)
    if budget is not None and not measurement.fits:
        raise PromptBudgetExceeded(measurement)
    return measurement


assert_prompt_budget = assert_prompt_fits


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content or "")


def _is_managed_message(message: dict[str, Any]) -> bool:
    """Identify injected context messages that may be removed as a unit."""
    if message.get("role") != "user":
        return False
    text = _content_text(message.get("content"))
    return text.startswith((
        "[Past Knowledge]",
        "[Recent Scheduled Runs]",
        "[Recent Scheduled Run]",
    ))


def _tool_call_ids(message: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict) and call.get("id"):
            ids.append(str(call["id"]))
    return ids


def _legalize_turn(turn: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove orphan/partial tool groups while retaining ordinary messages."""
    result: list[dict[str, Any]] = []
    index = 0
    while index < len(turn):
        message = turn[index]
        role = message.get("role")
        if role != "assistant" or not message.get("tool_calls"):
            if role != "tool":
                result.append(copy.deepcopy(message))
            index += 1
            continue

        call_ids = _tool_call_ids(message)
        # A provider payload with duplicate declarations is not a legal tool
        # group.  Keeping one copy would make the history appear complete
        # while silently changing the model's requested calls.
        if not call_ids or len(call_ids) != len(set(call_ids)):
            clean = copy.deepcopy(message)
            clean.pop("tool_calls", None)
            result.append(clean)
            index += 1
            continue

        # Tool results are expected directly after the assistant call.  Keep
        # the entire group only when every declared result is present; this
        # avoids sending a provider an assistant call with missing results.
        results: list[dict[str, Any]] = []
        cursor = index + 1
        while cursor < len(turn) and turn[cursor].get("role") == "tool":
            results.append(turn[cursor])
            cursor += 1
        returned_ids = [str(item.get("tool_call_id")) for item in results if item.get("tool_call_id")]
        # Keep exactly one result for every declared ID.  Undeclared surplus,
        # duplicate results, missing IDs, and missing declared results all
        # invalidate the complete group.
        if (
            len(results) == len(call_ids)
            and len(returned_ids) == len(results)
            and len(set(returned_ids)) == len(returned_ids)
            and set(returned_ids) == set(call_ids)
        ):
            result.append(copy.deepcopy(message))
            result.extend(copy.deepcopy(item) for item in results)
        else:
            # Preserve visible assistant text if the provider emitted any, but
            # drop the incomplete call and all its result messages.
            if message.get("content"):
                clean = copy.deepcopy(message)
                clean.pop("tool_calls", None)
                result.append(clean)
        index = cursor
    return result


def _turns(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Split system/managed prefix from ordinary user-led complete turns."""
    prefix: list[dict[str, Any]] = []
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    ordinary_user_seen = False

    for message in messages:
        role = message.get("role")
        if role == "system":
            # System instructions are always a non-removable prefix.  Some
            # persisted histories contain a later system message after a
            # user turn; dropping it would silently alter provider policy.
            prefix.append(message)
            continue
        if role == "user" and _is_managed_message(message) and not ordinary_user_seen:
            prefix.append(message)
            continue
        if role == "user" and not _is_managed_message(message):
            ordinary_user_seen = True
            if current is not None:
                turns.append(_legalize_turn(current))
            current = [message]
            continue
        if current is not None:
            current.append(message)
        elif role not in {"assistant", "tool"}:
            # Unknown metadata before the first user is not safe to send.
            prefix.append(message)

    if current is not None:
        turns.append(_legalize_turn(current))
    return [copy.deepcopy(message) for message in prefix], turns


def _drop_managed_prefix_to_fit(
    prefix: list[dict[str, Any]],
    current: list[dict[str, Any]],
    provider: Any,
    model: str | None,
    tools: list[dict[str, Any]] | None,
    budget: PromptBudget,
) -> list[dict[str, Any]]:
    """Drop largest/oldest injected context before declaring an irreducible error."""
    candidate = [*prefix, *current]
    if measure_prompt(provider, model, candidate, tools, budget).fits:
        return candidate

    # System instructions are not removable.  Managed user context is a
    # request-local convenience and is dropped oldest-first when necessary.
    retained = list(prefix)
    for index, message in enumerate(prefix):
        if not _is_managed_message(message):
            continue
        trial = retained[:]
        trial.pop(trial.index(message))
        if measure_prompt(provider, model, [*trial, *current], tools, budget).fits:
            return [*trial, *current]
        retained = trial

    candidate = [*retained, *current]
    if measure_prompt(provider, model, candidate, tools, budget).fits:
        return candidate
    raise PromptBudgetExceeded(measure_prompt(provider, model, candidate, tools, budget))


def _try_drop_managed_prefix_to_fit(
    prefix: list[dict[str, Any]],
    body: list[dict[str, Any]],
    provider: Any,
    model: str | None,
    tools: list[dict[str, Any]] | None,
    budget: PromptBudget,
) -> list[dict[str, Any]] | None:
    """Return a fitting prefix/body candidate, or ``None`` without raising.

    The public reducer uses :func:`_drop_managed_prefix_to_fit` for an
    irreducible prompt.  During current-turn reduction, however, we need to
    probe progressively smaller bodies first.  Keeping this probe separate
    makes it impossible to accidentally turn a temporarily-too-large body
    into a terminal budget error before its older tool groups have been
    considered.
    """
    candidate = [*prefix, *body]
    if measure_prompt(provider, model, candidate, tools, budget).fits:
        return candidate

    retained = list(prefix)
    for message in prefix:
        if not _is_managed_message(message):
            continue
        retained.remove(message)
        candidate = [*retained, *body]
        if measure_prompt(provider, model, candidate, tools, budget).fits:
            return candidate
    return None


def _turn_groups(turn: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Split a legal user-led turn into its mandatory user and atomic groups.

    A user-led turn can contain many assistant/tool iterations.  Treating the
    whole turn as one indivisible item means a single large tool result makes
    an otherwise recoverable request fail.  Each assistant/tool-call group is
    atomic here: the assistant call and every declared tool result stay
    together, while an incomplete group has already been reduced to visible
    assistant text by :func:`_legalize_turn`.
    """
    legal = _legalize_turn(turn)
    if not legal:
        return [], []

    # ``_turns`` starts every group at an ordinary user message.  Be defensive
    # for direct callers and keep any leading non-user item in the head so the
    # final legality check can still decide whether it is sendable.
    head = [legal[0]]
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in legal[1:]:
        if message.get("role") == "assistant" and current:
            groups.append(current)
            current = []
        current.append(message)
    if current:
        groups.append(current)
    return head, groups


def _fit_current_turn(
    prefix: list[dict[str, Any]],
    turn: list[dict[str, Any]],
    provider: Any,
    model: str | None,
    tools: list[dict[str, Any]] | None,
    budget: PromptBudget,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit the current user-led turn while preserving newest atomic groups.

    The current user message is mandatory.  If all of its assistant/tool
    iterations do not fit, older complete groups are dropped from that same
    turn until the newest fitting suffix remains.  A tool-call group is never
    split, so this path cannot create an orphan tool result or an assistant
    call without all of its declared results.
    """
    head, groups = _turn_groups(turn)
    if not head:
        candidate = _drop_managed_prefix_to_fit(prefix, [], provider, model, tools, budget)
        return candidate, []

    def split_candidate(
        candidate: list[dict[str, Any]], body: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return (candidate[: len(candidate) - len(body)] if body else candidate), body

    whole_body = [*head, *(item for group in groups for item in group)]
    whole = _try_drop_managed_prefix_to_fit(
        prefix, whole_body,
        provider, model, tools, budget,
    )
    if whole is not None:
        return split_candidate(whole, whole_body)

    # Always retain the current user.  This raises only when the user request
    # itself (plus non-removable system context) is irreducibly oversized.
    base = _try_drop_managed_prefix_to_fit(prefix, head, provider, model, tools, budget)
    if base is None:
        candidate = _drop_managed_prefix_to_fit(prefix, head, provider, model, tools, budget)
        return split_candidate(candidate, head)

    effective_prefix, _ = split_candidate(base, head)
    selected: list[list[dict[str, Any]]] = []
    for group in reversed(groups):
        trial_groups = [group, *selected]
        trial_body = [*head, *(item for candidate in trial_groups for item in candidate)]
        trial = _try_drop_managed_prefix_to_fit(
            effective_prefix, trial_body, provider, model, tools, budget,
        )
        if trial is None:
            # Groups are considered newest-first.  An older group cannot be
            # preferred over a newer one without violating suffix semantics;
            # leave it and every earlier group out once the suffix is full.
            break
        selected.insert(0, group)
        effective_prefix, _ = split_candidate(trial, trial_body)

    selected_body = [*head, *(item for group in selected for item in group)]
    return effective_prefix, selected_body


def reduce_messages_to_budget(
    messages: list[dict[str, Any]],
    provider: Any,
    model: str | None,
    tools: list[dict[str, Any]] | None,
    budget: PromptBudget,
    *,
    preserve_last_n_turns: int = 2,
) -> list[dict[str, Any]]:
    """Return a deterministic legal suffix that fits *budget*.

    The current user turn is mandatory.  Older complete user-led turns are
    greedily retained newest-first; ``preserve_last_n_turns`` is only a soft
    preference because the hard prompt bound always wins.
    """
    del preserve_last_n_turns  # retained as a compatibility/API hint
    prefix, turns = _turns(messages)
    if not turns:
        return _drop_managed_prefix_to_fit(prefix, [], provider, model, tools, budget)

    current = turns[-1]
    selected: list[list[dict[str, Any]]] = []
    effective_prefix, current_body = _fit_current_turn(
        prefix, current, provider, model, tools, budget,
    )
    candidate = [*effective_prefix, *current_body]

    for turn in reversed(turns[:-1]):
        trial_turns = [turn, *selected]
        trial = [
            *effective_prefix,
            *(item for group in trial_turns for item in group),
            *current_body,
        ]
        if measure_prompt(provider, model, trial, tools, budget).fits:
            selected.insert(0, turn)
            candidate = trial
        # Once an older turn does not fit, even earlier turns cannot be
        # preferred over it without violating newest-turn retention semantics.
        else:
            break

    # A final measurement catches provider counters that account for fields
    # differently after a reduction.  It is also the common invariant used by
    # AgentRunner and provider retry boundaries.
    measurement = measure_prompt(provider, model, candidate, tools, budget)
    if not measurement.fits:
        raise PromptBudgetExceeded(measurement)
    return candidate


# A short alias for callers that naturally describe this operation as trim.
trim_messages_to_budget = reduce_messages_to_budget
