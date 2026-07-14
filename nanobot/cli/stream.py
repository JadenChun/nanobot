"""Streaming renderer for CLI output.

Uses Rich Live with auto_refresh=False for stable, flicker-free
markdown rendering during streaming. Ellipsis mode handles overflow.
"""

from __future__ import annotations

import sys
import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from nanobot import __logo__


def _make_console() -> Console:
    return Console(file=sys.stdout)


class ThinkingSpinner:
    """Spinner that shows 'nanobot is thinking...' with pause support."""

    def __init__(self, console: Console | None = None):
        c = console or _make_console()
        self._spinner = c.status("[dim]nanobot is thinking...[/dim]", spinner="dots")
        self._active = False

    def __enter__(self):
        self._spinner.start()
        self._active = True
        return self

    def __exit__(self, *exc):
        self._active = False
        self._spinner.stop()
        return False

    def pause(self):
        """Context manager: temporarily stop spinner for clean output."""
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            if self._spinner and self._active:
                self._spinner.stop()
            try:
                yield
            finally:
                if self._spinner and self._active:
                    self._spinner.start()

        return _ctx()


class StreamRenderer:
    """Rich Live streaming with markdown. auto_refresh=False avoids render races.

    Deltas arrive live from the agent loop; think tags are stripped at
    render time so partial think blocks are never shown to the user.

    Flow per round:
      spinner -> first visible delta -> header + Live renders ->
      on_end -> Live stops (content stays on screen)
    """

    def __init__(
        self,
        render_markdown: bool = True,
        show_spinner: bool = True,
        use_live: bool = True,
    ):
        self._md = render_markdown
        self._show_spinner = show_spinner
        self._use_live = use_live
        self._buf = ""
        self._live: Live | None = None
        self._t = 0.0
        self.streamed = False
        self._spinner: ThinkingSpinner | None = None
        self._header_printed = False
        self._start_spinner()

    def _render(self):
        from nanobot.utils.helpers import strip_think
        buf = strip_think(self._buf) if self._buf else ""
        return Markdown(buf) if self._md and buf else Text(buf or "")

    def _start_spinner(self) -> None:
        if self._show_spinner:
            self._spinner = ThinkingSpinner()
            self._spinner.__enter__()

    def _stop_spinner(self) -> None:
        if self._spinner:
            self._spinner.__exit__(None, None, None)
            self._spinner = None

    async def on_delta(self, delta: str) -> None:
        self.streamed = True
        self._buf += delta
        if not self._buf.strip():
            return

        if not self._header_printed:
            self._stop_spinner()
            c = _make_console()
            c.print()
            c.print(f"[cyan]{__logo__} nanobot[/cyan]")
            self._header_printed = True

        if not self._use_live:
            # Logs mode: write directly to stdout with an explicit flush so the
            # next loguru stderr line can't visually attach to the streamed text.
            sys.stdout.write(delta)
            sys.stdout.flush()
            return

        if self._live is None:
            c = _make_console()
            # transient=False: keep the rendered content on screen when Live
            # stops, avoiding the clear-then-reprint cycle that causes
            # duplicated output when the terminal doesn't perfectly process
            # the transient clearing escape sequences.
            self._live = Live(
                self._render(),
                console=c,
                auto_refresh=False,
                transient=False,
            )
            self._live.start()
            self._t = time.monotonic()
            return
        now = time.monotonic()
        if "\n" in delta or (now - self._t) > 0.05:
            self._live.update(self._render())
            self._live.refresh()
            self._t = now

    async def on_end(self, *, resuming: bool = False) -> None:
        if self._live:
            # With transient=False, stop() leaves the last rendered frame
            # on screen — no reprint needed, avoiding duplicated output.
            self._live.stop()
            self._live = None
        self._stop_spinner()
        printed_newline = False
        if self.streamed and (not self._use_live or self._header_printed):
            if not self._use_live:
                # Logs mode: flush a real newline to stdout immediately so any
                # background loguru stderr line lands on its own line instead of
                # sticking to the tail of the streamed response.
                sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                _make_console().print()
            printed_newline = True
        if resuming:
            _make_console().print("[dim](Continuing with tools...)[/dim]")
            self._buf = ""
            self._header_printed = False
            self._start_spinner()
        elif not printed_newline:
            _make_console().print()

    async def close(self) -> None:
        """Stop spinner/live without rendering a final streamed round."""
        if self._live:
            self._live.stop()
            self._live = None
        self._stop_spinner()
