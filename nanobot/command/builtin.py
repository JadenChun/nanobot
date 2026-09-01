"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import os
import sys

from nanobot import __version__
from nanobot.bus.events import OutboundMessage
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.utils.helpers import build_status_content


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel active processing for the session."""
    loop = ctx.loop
    msg = ctx.msg
    tasks = loop._active_tasks.pop(msg.session_key, [])
    cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    total = cancelled
    content = f"Stopped {total} task(s)." if total else "No active task to stop."
    if total:
        loop.record_task_cancellation(
            session_key=ctx.key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            active_tasks=cancelled,
        )
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process in-place via os.execv."""
    msg = ctx.msg

    async def _do_restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="Restarting...")


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    ctx_est = 0
    try:
        ctx_est, _ = loop.memory_consolidator.estimate_session_prompt_tokens(session)
    except Exception:
        pass
    if ctx_est <= 0:
        ctx_est = loop._last_usage.get("prompt_tokens", 0)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_status_content(
            version=__version__, model=loop.model,
            start_time=loop._start_time, last_usage=loop._last_usage,
            context_window_tokens=loop.context_window_tokens,
            session_msg_count=len(session.get_history(max_messages=0)),
            context_tokens_estimate=ctx_est,
        ),
        metadata={"render_as": "text"},
    )


async def cmd_quota(ctx: CommandContext) -> OutboundMessage:
    """Show the current ChatGPT/Codex OAuth quota windows."""
    from nanobot.providers.codex_auth import format_codex_usage, get_codex_usage

    try:
        payload = await asyncio.to_thread(get_codex_usage)
        content = format_codex_usage(payload)
    except RuntimeError as exc:
        content = f"Unable to check Codex quota: {exc}"
    except Exception:
        content = "Unable to check Codex quota due to an unexpected error."
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        metadata={"render_as": "text"},
    )


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Start a fresh session."""
    loop = ctx.loop
    from nanobot.session.manager import SessionWriteConflict

    snapshot: list[dict] = []
    for _attempt in range(3):
        session = (
            (ctx.session if _attempt == 0 and ctx.session is not None else None)
            or loop.sessions.get_or_create(ctx.key)
        )
        snapshot = [dict(message) for message in session.messages[session.last_consolidated:]]
        session.clear(expected_revision=session.revision)
        try:
            loop.sessions.save(session)
            loop.sessions.invalidate(session.key)
            break
        except SessionWriteConflict:
            loop.sessions.invalidate(ctx.key)
    else:
        # Preserve the content-free conflict contract after bounded retries.
        raise SessionWriteConflict()
    if snapshot:
        loop._schedule_background(loop.memory_consolidator.archive_messages(snapshot))
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="New session started.",
    )


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_help_text(),
        metadata={"render_as": "text"},
    )


def build_help_text() -> str:
    """Build canonical help text shared across channels."""
    lines = [
        "🐈 nanobot commands:",
        "/new — Start a new conversation",
        "/stop — Stop the current task",
        "/restart — Restart the bot",
        "/status — Show bot status",
        "/quota — Show Codex OAuth quota",
        "/help — Show available commands",
    ]
    return "\n".join(lines)


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.priority("/quota", cmd_quota)
    router.exact("/new", cmd_new)
    router.exact("/status", cmd_status)
    router.exact("/quota", cmd_quota)
    router.exact("/help", cmd_help)
