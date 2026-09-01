"""Owned subprocess lifecycle helpers.

All agent-facing subprocesses run in their own process group/session.  A
timeout or parent-task cancellation therefore cannot strand descendants such
as npm launchers, browser processes, or device bridges.  Cleanup always waits
for the asyncio child process to be reaped before the caller continues.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass(slots=True)
class OwnedProcessResult:
    """Captured output from one owned process invocation."""

    process: asyncio.subprocess.Process
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False

    @property
    def returncode(self) -> int | None:
        return self.process.returncode


async def create_owned_process(
    command: str | Sequence[str],
    *,
    shell: bool = False,
    **kwargs: Any,
) -> asyncio.subprocess.Process:
    """Start *command* in a dedicated process group/session."""

    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP gives terminate_owned_process a group
        # boundary on Windows.  ``terminate`` remains the fallback for
        # launchers that do not honor CTRL_BREAK_EVENT.
        kwargs.setdefault(
            "creationflags",
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    else:
        kwargs.setdefault("start_new_session", True)

    if shell:
        if not isinstance(command, str):
            raise TypeError("shell commands must be strings")
        return await asyncio.create_subprocess_shell(command, **kwargs)
    if isinstance(command, str):
        command = [command]
    return await asyncio.create_subprocess_exec(*command, **kwargs)


def _signal_process_group(process: asyncio.subprocess.Process, sig: int) -> None:
    """Send a signal to an owned process group, with a process fallback."""

    if process.returncode is not None:
        return
    if os.name == "nt":
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
        return

    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (ProcessLookupError, PermissionError):
        # The group can disappear between checking returncode and signalling.
        # ``kill`` is still useful for a child that was not fully detached by
        # an unusual event-loop or test process implementation.
        try:
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except (ProcessLookupError, PermissionError):
            pass


async def terminate_owned_process(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float = 2.0,
) -> None:
    """Terminate an owned process group and await child reaping.

    The escalation is intentionally bounded: SIGTERM first, then SIGKILL if
    the process group does not exit promptly.  The final wait is still issued
    after SIGKILL so normal asyncio subprocess implementations reap the child.
    """

    if process.returncode is None:
        _signal_process_group(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        except asyncio.TimeoutError:
            _signal_process_group(process, signal.SIGKILL)
            try:
                await asyncio.wait_for(process.wait(), timeout=grace_seconds)
            except asyncio.TimeoutError:
                # A real POSIX child should have exited after SIGKILL.  Keep
                # this final wait bounded for hostile/fake process objects.
                logger.warning("Owned process {} did not reap after SIGKILL", process.pid)
    else:
        # Calling wait for an already-exited process is cheap and guarantees
        # that an asyncio child watcher has completed its reap bookkeeping.
        try:
            await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        except asyncio.TimeoutError:
            logger.warning("Owned process {} did not finish reaping", process.pid)


async def _finish_communication(
    communication: asyncio.Task[tuple[bytes, bytes]],
    *,
    timeout: float,
) -> tuple[bytes, bytes]:
    """Drain a communication task after its process has been terminated."""

    try:
        return await asyncio.wait_for(asyncio.shield(communication), timeout=timeout)
    except asyncio.TimeoutError:
        communication.cancel()
        await asyncio.gather(communication, return_exceptions=True)
        return b"", b""


async def run_owned_process(
    command: str | Sequence[str],
    *,
    timeout: float | None = None,
    shell: bool = False,
    reap_timeout: float = 2.0,
    **kwargs: Any,
) -> OwnedProcessResult:
    """Run one owned process and capture output.

    ``timed_out`` is returned for ordinary command timeouts so callers can
    format a tool-specific response.  Parent cancellation is different: the
    process is cleaned up and ``CancelledError`` is re-raised unchanged.
    """

    process = await create_owned_process(command, shell=shell, **kwargs)
    communication = asyncio.create_task(process.communicate())
    try:
        if timeout is None:
            stdout, stderr = await asyncio.shield(communication)
        else:
            stdout, stderr = await asyncio.wait_for(
                asyncio.shield(communication), timeout=timeout
            )
        return OwnedProcessResult(process=process, stdout=stdout, stderr=stderr)
    except asyncio.TimeoutError:
        await terminate_owned_process(process, grace_seconds=reap_timeout)
        stdout, stderr = await _finish_communication(communication, timeout=reap_timeout)
        return OwnedProcessResult(
            process=process,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
        )
    except asyncio.CancelledError:
        # The communication task is explicitly retained and joined, so no
        # fire-and-forget task survives parent cancellation.
        await terminate_owned_process(process, grace_seconds=reap_timeout)
        await _finish_communication(communication, timeout=reap_timeout)
        raise
    except Exception:
        # A pipe/transport failure is also a failed owned invocation; do not
        # leave its process group behind while the caller formats the error.
        await terminate_owned_process(process, grace_seconds=reap_timeout)
        await _finish_communication(communication, timeout=reap_timeout)
        raise


async def await_owned_cleanup(
    awaitable: Awaitable[Any],
    *,
    timeout: float = 5.0,
) -> Any:
    """Await a cleanup operation with bounded cancellation-safe semantics.

    A cancellation first interrupts the caller, then gives the owned cleanup
    task one bounded window to finish before re-raising.  The task is always
    joined or cancelled-and-joined, so no shielded task is left untracked.
    """

    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except asyncio.CancelledError:
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        finally:
            # Always preserve cancellation even when cleanup itself fails.
            raise
    except asyncio.TimeoutError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return None


__all__ = [
    "OwnedProcessResult",
    "await_owned_cleanup",
    "create_owned_process",
    "run_owned_process",
    "terminate_owned_process",
]
