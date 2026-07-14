import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from prompt_toolkit.formatted_text import HTML
from rich.text import Text

from nanobot.cli import commands
from nanobot.cli import stream as stream_mod


@pytest.fixture
def mock_prompt_session():
    """Mock the global prompt session."""
    mock_session = MagicMock()
    mock_session.prompt_async = AsyncMock()
    with patch("nanobot.cli.commands._PROMPT_SESSION", mock_session), \
         patch("nanobot.cli.commands.patch_stdout"):
        yield mock_session


@pytest.mark.asyncio
async def test_read_interactive_input_async_returns_input(mock_prompt_session):
    """Test that _read_interactive_input_async returns the user input from prompt_session."""
    mock_prompt_session.prompt_async.return_value = "hello world"

    result = await commands._read_interactive_input_async()
    
    assert result == "hello world"
    mock_prompt_session.prompt_async.assert_called_once()
    args, _ = mock_prompt_session.prompt_async.call_args
    assert isinstance(args[0], HTML)  # Verify HTML prompt is used


@pytest.mark.asyncio
async def test_read_interactive_input_async_handles_eof(mock_prompt_session):
    """Test that EOFError converts to KeyboardInterrupt."""
    mock_prompt_session.prompt_async.side_effect = EOFError()

    with pytest.raises(KeyboardInterrupt):
        await commands._read_interactive_input_async()


def test_init_prompt_session_creates_session():
    """Test that _init_prompt_session initializes the global session."""
    # Ensure global is None before test
    commands._PROMPT_SESSION = None
    
    with patch("nanobot.cli.commands.PromptSession") as MockSession, \
         patch("nanobot.cli.commands.FileHistory") as MockHistory, \
         patch("pathlib.Path.home") as mock_home:
        
        mock_home.return_value = MagicMock()
        
        commands._init_prompt_session()
        
        assert commands._PROMPT_SESSION is not None
        MockSession.assert_called_once()
        _, kwargs = MockSession.call_args
        assert kwargs["multiline"] is False
        assert kwargs["enable_open_in_editor"] is False


def test_thinking_spinner_pause_stops_and_restarts():
    """Pause should stop the active spinner and restart it afterward."""
    spinner = MagicMock()
    mock_console = MagicMock()
    mock_console.status.return_value = spinner

    thinking = stream_mod.ThinkingSpinner(console=mock_console)
    with thinking:
        with thinking.pause():
            pass

    assert spinner.method_calls == [
        call.start(),
        call.stop(),
        call.start(),
        call.stop(),
    ]


def test_print_cli_progress_line_pauses_spinner_before_printing():
    """CLI progress output should pause spinner to avoid garbled lines."""
    order: list[str] = []
    spinner = MagicMock()
    spinner.start.side_effect = lambda: order.append("start")
    spinner.stop.side_effect = lambda: order.append("stop")
    mock_console = MagicMock()
    mock_console.status.return_value = spinner

    with patch.object(commands.console, "print", side_effect=lambda *_args, **_kwargs: order.append("print")):
        thinking = stream_mod.ThinkingSpinner(console=mock_console)
        with thinking:
            commands._print_cli_progress_line("tool running", thinking)

    assert order == ["start", "stop", "print", "start", "stop"]


@pytest.mark.asyncio
async def test_print_interactive_progress_line_pauses_spinner_before_printing():
    """Interactive progress output should also pause spinner cleanly."""
    order: list[str] = []
    spinner = MagicMock()
    spinner.start.side_effect = lambda: order.append("start")
    spinner.stop.side_effect = lambda: order.append("stop")
    mock_console = MagicMock()
    mock_console.status.return_value = spinner

    async def fake_print(_text: str) -> None:
        order.append("print")

    with patch("nanobot.cli.commands._print_interactive_line", side_effect=fake_print):
        thinking = stream_mod.ThinkingSpinner(console=mock_console)
        with thinking:
            await commands._print_interactive_progress_line("tool running", thinking)

    assert order == ["start", "stop", "print", "start", "stop"]


def test_response_renderable_uses_text_for_explicit_plain_rendering():
    status = (
        "🐈 nanobot v0.1.4.post5\n"
        "🧠 Model: MiniMax-M2.7\n"
        "📊 Tokens: 20639 in / 29 out"
    )

    renderable = commands._response_renderable(
        status,
        render_markdown=True,
        metadata={"render_as": "text"},
    )

    assert renderable.__class__.__name__ == "Text"


def test_response_renderable_preserves_normal_markdown_rendering():
    renderable = commands._response_renderable("**bold**", render_markdown=True)

    assert renderable.__class__.__name__ == "Markdown"


def test_response_renderable_without_metadata_keeps_markdown_path():
    help_text = "🐈 nanobot commands:\n/status — Show bot status\n/help — Show available commands"

    renderable = commands._response_renderable(help_text, render_markdown=True)

    assert renderable.__class__.__name__ == "Markdown"


@pytest.mark.asyncio
async def test_stream_renderer_keeps_live_content_without_reprint():
    """Live content should stay on screen (transient=False) without reprinting."""
    mock_console = MagicMock()
    live_instances: list[object] = []

    class FakeLive:
        def __init__(self, renderable, *, console, auto_refresh, transient):
            self.renderable = renderable
            self.console = console
            self.auto_refresh = auto_refresh
            self.transient = transient
            self.started = False
            self.stopped = False
            self.updated: list[object] = []
            self.refreshed = 0
            live_instances.append(self)

        def start(self):
            self.started = True

        def update(self, renderable):
            self.updated.append(renderable)

        def refresh(self):
            self.refreshed += 1

        def stop(self):
            self.stopped = True

    with patch("nanobot.cli.stream._make_console", return_value=mock_console), \
         patch("nanobot.cli.stream.Live", FakeLive):
        renderer = stream_mod.StreamRenderer(
            render_markdown=False,
            show_spinner=False,
            use_live=True,
        )
        await renderer.on_delta("hello\nworld")
        await renderer.on_end()

    assert len(live_instances) == 1
    live = live_instances[0]
    assert live.started is True
    assert live.stopped is True
    # transient=False: content stays on screen, no clear-then-reprint cycle
    assert live.transient is False
    # The renderable passed to Live should contain the full content
    assert isinstance(live.renderable, Text)
    assert live.renderable.plain == "hello\nworld"
    # No redundant update/refresh on the first delta (start() already rendered)
    assert live.refreshed == 0
    # The full content should NOT be reprinted via console.print —
    # it's already visible from the Live display.
    rendered_texts = [
        args[0]
        for args, _ in mock_console.print.call_args_list
        if args and isinstance(args[0], Text)
    ]
    assert not any(text.plain == "hello\nworld" for text in rendered_texts)
