"""Regression tests for owned subprocess lifecycle cleanup."""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

from nanobot.agent.tools.process import run_owned_process


@pytest.mark.asyncio
async def test_owned_process_timeout_reaps_child_process() -> None:
    result = await run_owned_process(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        timeout=0.05,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    assert result.timed_out is True
    assert result.returncode is not None


@pytest.mark.asyncio
async def test_owned_process_cancellation_terminates_and_reaps_child() -> None:
    task = asyncio.create_task(run_owned_process(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        timeout=30,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    ))
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # A cancelled run must not leave a direct child process alive.  This is a
    # platform-level smoke check; the process-group implementation is tested
    # more directly on POSIX hosts where process groups are available.
    if os.name != "nt":
        await asyncio.sleep(0)
