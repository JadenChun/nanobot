"""Tests for /stop task cancellation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_loop(tmp_path, *, exec_config=None, channels_config=None):
    """Create a minimal AgentLoop with mocked dependencies."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = tmp_path

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.ForegroundAgentManager"):
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=workspace,
            exec_config=exec_config,
            channels_config=channels_config,
        )
    return loop, bus


class TestHandleStop:
    @pytest.mark.asyncio
    async def test_stop_no_active_task(self, tmp_path):
        from nanobot.bus.events import InboundMessage
        from nanobot.command.builtin import cmd_stop
        from nanobot.command.router import CommandContext

        loop, bus = _make_loop(tmp_path)
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/stop", loop=loop)
        out = await cmd_stop(ctx)
        assert "No active task" in out.content

    @pytest.mark.asyncio
    async def test_stop_cancels_active_task(self, tmp_path):
        from nanobot.bus.events import InboundMessage
        from nanobot.command.builtin import cmd_stop
        from nanobot.command.router import CommandContext

        loop, bus = _make_loop(tmp_path)
        cancelled = asyncio.Event()

        async def slow_task():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(slow_task())
        await asyncio.sleep(0)
        loop._active_tasks["test:c1"] = [task]

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/stop", loop=loop)
        out = await cmd_stop(ctx)

        assert cancelled.is_set()
        assert "stopped" in out.content.lower()

    @pytest.mark.asyncio
    async def test_stop_cancels_multiple_tasks(self, tmp_path):
        from nanobot.bus.events import InboundMessage
        from nanobot.command.builtin import cmd_stop
        from nanobot.command.router import CommandContext

        loop, bus = _make_loop(tmp_path)
        events = [asyncio.Event(), asyncio.Event()]

        async def slow(idx):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                events[idx].set()
                raise

        tasks = [asyncio.create_task(slow(i)) for i in range(2)]
        await asyncio.sleep(0)
        loop._active_tasks["test:c1"] = tasks

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/stop", loop=loop)
        out = await cmd_stop(ctx)

        assert all(e.is_set() for e in events)
        assert "2 task" in out.content


class TestDispatch:
    def test_exec_tool_not_registered_when_disabled(self, tmp_path):
        from nanobot.config.schema import ExecToolConfig

        loop, _bus = _make_loop(tmp_path, exec_config=ExecToolConfig(enable=False))

        assert loop.tools.get("exec") is None

    @pytest.mark.asyncio
    async def test_dispatch_processes_and_publishes(self, tmp_path):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop(tmp_path)
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hello")
        loop._process_message = AsyncMock(
            return_value=OutboundMessage(channel="test", chat_id="c1", content="hi")
        )
        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert out.content == "hi"

    @pytest.mark.asyncio
    async def test_cli_dispatch_marks_only_the_final_response_as_turn_complete(self, tmp_path):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop(tmp_path)
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="marketing", content="hello")
        loop._process_message = AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="marketing", content="hi")
        )

        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

        assert out.content == "hi"
        assert out.metadata["_cli_turn_complete"] is True

    @pytest.mark.asyncio
    async def test_cli_dispatch_publishes_empty_completion_after_message_tool_delivery(self, tmp_path):
        from nanobot.bus.events import InboundMessage

        loop, bus = _make_loop(tmp_path)
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="marketing", content="hello")
        loop._process_message = AsyncMock(return_value=None)

        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

        assert out.content == ""
        assert out.metadata["_cli_turn_complete"] is True

    @pytest.mark.asyncio
    async def test_dispatch_streaming_preserves_message_metadata(self, tmp_path):
        from nanobot.bus.events import InboundMessage

        loop, bus = _make_loop(tmp_path)
        msg = InboundMessage(
            channel="matrix",
            sender_id="u1",
            chat_id="!room:matrix.org",
            content="hello",
            metadata={
                "_wants_stream": True,
                "thread_root_event_id": "$root1",
                "thread_reply_to_event_id": "$reply1",
            },
        )

        async def fake_process(_msg, *, on_stream=None, on_stream_end=None, **kwargs):
            assert on_stream is not None
            assert on_stream_end is not None
            await on_stream("hi")
            await on_stream_end(resuming=False)
            return None

        loop._process_message = fake_process

        await loop._dispatch(msg)
        first = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        second = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

        assert first.metadata["thread_root_event_id"] == "$root1"
        assert first.metadata["thread_reply_to_event_id"] == "$reply1"
        assert first.metadata["_stream_delta"] is True
        assert second.metadata["thread_root_event_id"] == "$root1"
        assert second.metadata["thread_reply_to_event_id"] == "$reply1"
        assert second.metadata["_stream_end"] is True

    @pytest.mark.asyncio
    async def test_dispatch_suppresses_resuming_stream_when_result_mode(self, tmp_path):
        from nanobot.bus.events import InboundMessage, OutboundMessage
        from nanobot.config.schema import ChannelsConfig

        loop, bus = _make_loop(tmp_path, channels_config=ChannelsConfig(task_update_mode="result"))
        msg = InboundMessage(
            channel="telegram",
            sender_id="u1",
            chat_id="123",
            content="hello",
            metadata={"_wants_stream": True, "message_id": 10},
        )

        async def fake_process(_msg, *, on_stream=None, on_stream_end=None, **kwargs):
            assert on_stream is not None
            assert on_stream_end is not None
            await on_stream("checking first")
            await on_stream_end(resuming=True)
            return OutboundMessage(
                channel="telegram",
                chat_id="123",
                content="final answer",
                metadata={"_streamed": True, "message_id": 10},
            )

        loop._process_message = fake_process

        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

        assert out.content == "final answer"
        assert out.metadata["message_id"] == 10
        assert "_stream_delta" not in out.metadata
        assert "_streamed" not in out.metadata

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bus.consume_outbound(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_dispatch_defers_terminal_stream_until_completion_in_result_mode(self, tmp_path):
        from nanobot.bus.events import InboundMessage, OutboundMessage
        from nanobot.config.schema import ChannelsConfig

        loop, bus = _make_loop(tmp_path, channels_config=ChannelsConfig(task_update_mode="result"))
        msg = InboundMessage(
            channel="matrix",
            sender_id="u1",
            chat_id="!room:matrix.org",
            content="hello",
            metadata={
                "_wants_stream": True,
                "thread_root_event_id": "$root1",
                "thread_reply_to_event_id": "$reply1",
            },
        )

        async def fake_process(_msg, *, on_stream=None, on_stream_end=None, **kwargs):
            assert on_stream is not None
            assert on_stream_end is not None
            # This is the initial action response. The real loop runs internal
            # verification after this callback and may then start a revision.
            await on_stream("failed response")
            await on_stream_end(resuming=False)
            assert bus.outbound_size == 0

            # A revision starts a fresh stream segment after verification.
            await on_stream_end(resuming=True)
            await on_stream("final answer")
            await on_stream_end(resuming=False)
            return OutboundMessage(
                channel="matrix",
                chat_id="!room:matrix.org",
                content="final answer",
                metadata={"_streamed": True, "message_id": 10},
            )

        loop._process_message = fake_process

        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

        assert out.content == "final answer"
        assert out.metadata == {"message_id": 10}

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bus.consume_outbound(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_dispatch_approval_response_visible_when_result_mode(self, tmp_path):
        from nanobot.bus.events import InboundMessage, OutboundMessage
        from nanobot.config.schema import ChannelsConfig

        loop, bus = _make_loop(tmp_path, channels_config=ChannelsConfig(task_update_mode="result"))
        msg = InboundMessage(
            channel="telegram",
            sender_id="u1",
            chat_id="123",
            content="delete files",
            metadata={"_wants_stream": True, "message_id": 10},
        )

        async def fake_process(_msg, *, on_stream=None, **kwargs):
            assert on_stream is not None
            await on_stream("this needs approval")
            return OutboundMessage(
                channel="telegram",
                chat_id="123",
                content="Reply `yes` to continue.",
                metadata={"message_id": 10},
            )

        loop._process_message = fake_process

        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

        assert out.content == "Reply `yes` to continue."
        assert out.metadata == {"message_id": 10}

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bus.consume_outbound(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_processing_lock_serializes(self, tmp_path):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop(tmp_path)
        order = []

        async def mock_process(m, **kwargs):
            order.append(f"start-{m.content}")
            await asyncio.sleep(0.05)
            order.append(f"end-{m.content}")
            return OutboundMessage(channel="test", chat_id="c1", content=m.content)

        loop._process_message = mock_process
        msg1 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="a")
        msg2 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="b")

        t1 = asyncio.create_task(loop._dispatch(msg1))
        t2 = asyncio.create_task(loop._dispatch(msg2))
        await asyncio.gather(t1, t2)
        assert order == ["start-a", "end-a", "start-b", "end-b"]
