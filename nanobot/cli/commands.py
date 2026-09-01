"""CLI commands for nanobot."""

import asyncio
import os
import select
import shutil
import signal
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    if sys.stdout.encoding != "utf-8":
        os.environ["PYTHONIOENCODING"] = "utf-8"
        # Re-open stdout/stderr with UTF-8 encoding
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import typer
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from nanobot import __logo__, __version__
from nanobot.cli.stream import StreamRenderer, ThinkingSpinner
from nanobot.config.paths import (
    get_browser_test_artifacts_dir,
    get_workspace_path,
    is_default_workspace,
)
from nanobot.config.schema import Config
from nanobot.utils.helpers import sync_workspace_templates

app = typer.Typer(
    name="nanobot",
    context_settings={"help_option_names": ["-h", "--help"]},
    help=f"{__logo__} nanobot - Personal AI Assistant",
    no_args_is_help=True,
)

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}

# ---------------------------------------------------------------------------
# CLI input: prompt_toolkit for editing, paste, history, and display
# ---------------------------------------------------------------------------

_PROMPT_SESSION: PromptSession | None = None
_SAVED_TERM_ATTRS = None  # original termios settings, restored on exit


def _flush_pending_tty_input() -> None:
    """Drop unread keypresses typed while the model was generating output."""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    try:
        import termios
        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return


def _restore_terminal() -> None:
    """Restore terminal to its original state (echo, line buffering, etc.)."""
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_prompt_session() -> None:
    """Create the prompt_toolkit session with persistent file history."""
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS

    # Save terminal state so we can restore it on exit
    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    from nanobot.config.paths import get_cli_history_path

    history_file = get_cli_history_path()
    history_file.parent.mkdir(parents=True, exist_ok=True)

    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=False,
        multiline=False,   # Enter submits (single line mode)
    )


def _make_console() -> Console:
    return Console(file=sys.stdout)


def _render_interactive_ansi(render_fn) -> str:
    """Render Rich output to ANSI so prompt_toolkit can print it safely."""
    ansi_console = Console(
        force_terminal=True,
        color_system=console.color_system or "standard",
        width=console.width,
    )
    with ansi_console.capture() as capture:
        render_fn(ansi_console)
    return capture.get()


def _print_agent_response(
    response: str,
    render_markdown: bool,
    metadata: dict | None = None,
) -> None:
    """Render assistant response with consistent terminal styling."""
    console = _make_console()
    content = response or ""
    body = _response_renderable(content, render_markdown, metadata)
    console.print()
    console.print(f"[cyan]{__logo__} nanobot[/cyan]")
    console.print(body)
    console.print()


def _response_renderable(content: str, render_markdown: bool, metadata: dict | None = None):
    """Render plain-text command output without markdown collapsing newlines."""
    if not render_markdown:
        return Text(content)
    if (metadata or {}).get("render_as") == "text":
        return Text(content)
    return Markdown(content)


async def _print_interactive_line(text: str) -> None:
    """Print async interactive updates with prompt_toolkit-safe Rich styling."""
    def _write() -> None:
        ansi = _render_interactive_ansi(
            lambda c: c.print(f"  [dim]↳ {text}[/dim]")
        )
        print_formatted_text(ANSI(ansi), end="")

    await run_in_terminal(_write)


async def _print_interactive_response(
    response: str,
    render_markdown: bool,
    metadata: dict | None = None,
) -> None:
    """Print async interactive replies with prompt_toolkit-safe Rich styling."""
    def _write() -> None:
        content = response or ""
        ansi = _render_interactive_ansi(
            lambda c: (
                c.print(),
                c.print(f"[cyan]{__logo__} nanobot[/cyan]"),
                c.print(_response_renderable(content, render_markdown, metadata)),
                c.print(),
            )
        )
        print_formatted_text(ANSI(ansi), end="")

    await run_in_terminal(_write)


def _print_cli_progress_line(text: str, thinking: ThinkingSpinner | None) -> None:
    """Print a CLI progress line, pausing the spinner if needed."""
    with thinking.pause() if thinking else nullcontext():
        console.print(f"  [dim]↳ {text}[/dim]")


async def _print_interactive_progress_line(text: str, thinking: ThinkingSpinner | None) -> None:
    """Print an interactive progress line, pausing the spinner if needed."""
    with thinking.pause() if thinking else nullcontext():
        await _print_interactive_line(text)


def _is_exit_command(command: str) -> bool:
    """Return True when input should end interactive chat."""
    return command.lower() in EXIT_COMMANDS


async def _read_interactive_input_async() -> str:
    """Read user input using prompt_toolkit (handles paste, history, display).

    prompt_toolkit natively handles:
    - Multiline paste (bracketed paste mode)
    - History navigation (up/down arrows)
    - Clean display (no ghost characters or artifacts)
    """
    if _PROMPT_SESSION is None:
        raise RuntimeError("Call _init_prompt_session() first")
    try:
        with patch_stdout():
            return await _PROMPT_SESSION.prompt_async(
                HTML("<b fg='ansiblue'>You:</b> "),
            )
    except EOFError as exc:
        raise KeyboardInterrupt from exc



def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} nanobot v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """nanobot - Personal AI Assistant."""
    pass


# ============================================================================
# Onboard / Setup
# ============================================================================


@app.command()
def onboard(
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    wizard: bool = typer.Option(False, "--wizard", help="Use interactive wizard"),
):
    """Initialize nanobot configuration and workspace."""
    from nanobot.config.loader import get_config_path, load_config, save_config, set_config_path
    from nanobot.config.schema import Config

    if config:
        config_path = Path(config).expanduser().resolve()
        set_config_path(config_path)
        console.print(f"[dim]Using config: {config_path}[/dim]")
    else:
        config_path = get_config_path()

    def _apply_workspace_override(loaded: Config) -> Config:
        if workspace:
            loaded.agents.defaults.workspace = workspace
        return loaded

    # Create or update config
    if config_path.exists():
        if wizard:
            config = _apply_workspace_override(load_config(config_path))
        else:
            console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
            console.print("  [bold]y[/bold] = overwrite with defaults (existing values will be lost)")
            console.print("  [bold]N[/bold] = refresh config, keeping existing values and adding new fields")
            if typer.confirm("Overwrite?"):
                config = _apply_workspace_override(Config())
                save_config(config, config_path)
                console.print(f"[green]✓[/green] Config reset to defaults at {config_path}")
            else:
                config = _apply_workspace_override(load_config(config_path))
                save_config(config, config_path)
                console.print(f"[green]✓[/green] Config refreshed at {config_path} (existing values preserved)")
    else:
        config = _apply_workspace_override(Config())
        # In wizard mode, don't save yet - the wizard will handle saving if should_save=True
        if not wizard:
            save_config(config, config_path)
            console.print(f"[green]✓[/green] Created config at {config_path}")

    # Run interactive wizard if enabled
    if wizard:
        from nanobot.cli.onboard import run_onboard

        try:
            result = run_onboard(initial_config=config)
            if not result.should_save:
                console.print("[yellow]Configuration discarded. No changes were saved.[/yellow]")
                return

            config = result.config
            save_config(config, config_path)
            console.print(f"[green]✓[/green] Config saved at {config_path}")
        except Exception as e:
            console.print(f"[red]✗[/red] Error during configuration: {e}")
            console.print("[yellow]Please run 'nanobot onboard' again to complete setup.[/yellow]")
            raise typer.Exit(1)
    _onboard_plugins(config_path)

    # Create workspace, preferring the configured workspace path.
    workspace_path = get_workspace_path(config.workspace_path)
    if not workspace_path.exists():
        workspace_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created workspace at {workspace_path}")

    sync_workspace_templates(workspace_path)

    agent_cmd = 'nanobot agent -m "Hello!"'
    gateway_cmd = "nanobot gateway"
    if config:
        agent_cmd += f" --config {config_path}"
        gateway_cmd += f" --config {config_path}"

    console.print(f"\n{__logo__} nanobot is ready!")
    console.print("\nNext steps:")
    if wizard:
        console.print(f"  1. Chat: [cyan]{agent_cmd}[/cyan]")
        console.print(f"  2. Start gateway: [cyan]{gateway_cmd}[/cyan]")
    else:
        console.print(f"  1. Add your API key to [cyan]{config_path}[/cyan]")
        console.print("     Get one at: https://openrouter.ai/keys")
        console.print(f"  2. Chat: [cyan]{agent_cmd}[/cyan]")
    console.print("\n[dim]Want Telegram/WhatsApp? See: https://github.com/HKUDS/nanobot#-chat-apps[/dim]")


def _merge_missing_defaults(existing: Any, defaults: Any) -> Any:
    """Recursively fill in missing values from defaults without overwriting user config."""
    if not isinstance(existing, dict) or not isinstance(defaults, dict):
        return existing

    merged = dict(existing)
    for key, value in defaults.items():
        if key not in merged:
            merged[key] = value
        else:
            merged[key] = _merge_missing_defaults(merged[key], value)
    return merged


def _onboard_plugins(config_path: Path) -> None:
    """Inject default config for all discovered channels (built-in + plugins)."""
    import json

    from nanobot.channels.registry import discover_all

    all_channels = discover_all()
    if not all_channels:
        return

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    channels = data.setdefault("channels", {})
    for name, cls in all_channels.items():
        if name not in channels:
            channels[name] = cls.default_config()
        else:
            channels[name] = _merge_missing_defaults(channels[name], cls.default_config())

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _make_single_provider(config: Config, provider_name: str, model: str):
    """Create a single provider instance for a resolved provider/model pair."""
    from nanobot.providers.registry import find_by_name

    p, _ = config.get_provider_by_name(provider_name)
    keys = p.effective_keys if p else []
    primary_key = keys[0] if keys else None
    spec = find_by_name(provider_name)
    backend = spec.backend if spec else "openai_compat"

    if backend == "azure_openai":
        if not p or not primary_key or not p.api_base:
            console.print("[red]Error: Azure OpenAI requires api_key and api_base.[/red]")
            console.print("Set them in ~/.nanobot/config.json under providers.azure_openai section")
            console.print("Use the model field to specify the deployment name.")
            raise typer.Exit(1)
    elif backend == "openai_compat" and not model.startswith("bedrock/"):
        needs_key = not bool(primary_key)
        exempt = spec and (spec.is_oauth or spec.is_local or spec.is_direct)
        if needs_key and not exempt:
            console.print("[red]Error: No API key configured.[/red]")
            console.print("Set one in ~/.nanobot/config.json under providers section")
            raise typer.Exit(1)

    if backend == "openai_codex":
        from nanobot.providers.openai_codex_provider import OpenAICodexProvider

        return OpenAICodexProvider(
            default_model=model,
            workspace=str(config.workspace_path),
        )
    if backend == "azure_openai":
        from nanobot.providers.azure_openai_provider import AzureOpenAIProvider

        return AzureOpenAIProvider(
            api_key=primary_key,
            api_base=p.api_base,
            default_model=model,
        )
    if backend == "anthropic":
        from nanobot.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=primary_key,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )

    from nanobot.providers.openai_compat_provider import OpenAICompatProvider

    token_provider = None
    merged_headers = dict(p.extra_headers) if p and p.extra_headers else {}
    if spec and spec.name == "github_copilot":
        from nanobot.providers import copilot_auth

        token_provider = copilot_auth.get_copilot_bearer
        for k, v in copilot_auth.copilot_request_headers().items():
            merged_headers.setdefault(k, v)

    return OpenAICompatProvider(
        api_key=primary_key,
        api_keys=keys if len(keys) > 1 else None,
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=merged_headers or None,
        rate_limit=p.rate_limit if p else 0,
        timeout=p.timeout if p else 60.0,
        spec=spec,
        token_provider=token_provider,
    )


def _make_provider(config: Config):
    """Create the configured provider chain, including quota fallback."""
    from nanobot.providers.base import GenerationSettings

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    if not provider_name:
        console.print("[red]Error: No provider could be matched for the configured model.[/red]")
        raise typer.Exit(1)

    primary = _make_single_provider(config, provider_name, model)

    defaults = config.agents.defaults
    gen = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens.output,
        reasoning_effort=defaults.reasoning_effort,
        context_window_tokens=defaults.max_tokens.input,
    )

    fallback_entries = defaults.fallback
    if not fallback_entries:
        primary.generation = gen
        return primary

    from nanobot.providers.fallback_provider import FallbackProvider

    providers = [(primary, model)]
    for entry in fallback_entries:
        providers.append((_make_single_provider(config, entry.provider, entry.model), entry.model))

    fallback = FallbackProvider(providers)
    fallback.generation = gen
    return fallback


def _make_crawler_provider(config: Config):
    """Create the optional crawler provider independently from the main agent."""
    crawler = config.agents.crawler
    from nanobot.agent.tools.social_crawl import crawl_tools_enabled

    if not crawler.enabled or not crawl_tools_enabled():
        return None
    model = crawler.model or config.agents.defaults.model
    provider_name = crawler.provider or config.get_provider_name(model)
    if not provider_name:
        console.print("[red]Error: No provider could be matched for the crawler model.[/red]")
        raise typer.Exit(1)
    provider = _make_single_provider(config, provider_name, model)

    from nanobot.providers.base import GenerationSettings

    provider.generation = GenerationSettings(
        temperature=0.1,
        max_tokens=crawler.max_tokens.output,
        reasoning_effort=crawler.reasoning_effort,
        context_window_tokens=crawler.max_tokens.input,
    )
    return provider


def _load_runtime_config(config: str | None = None, workspace: str | None = None) -> Config:
    """Load config and optionally override the active workspace."""
    from nanobot.config.loader import load_config, set_config_path

    config_path = None
    if config:
        config_path = Path(config).expanduser().resolve()
        if not config_path.exists():
            console.print(f"[red]Error: Config file not found: {config_path}[/red]")
            raise typer.Exit(1)
        set_config_path(config_path)
        console.print(f"[dim]Using config: {config_path}[/dim]")

    loaded = load_config(config_path)
    _warn_deprecated_config_keys(config_path)
    if workspace:
        loaded.agents.defaults.workspace = workspace
    return loaded


def _warn_deprecated_config_keys(config_path: Path | None) -> None:
    """Hint users to remove obsolete keys from their config file."""
    import json

    from nanobot.config.loader import get_config_path

    path = config_path or get_config_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if "memoryWindow" in raw.get("agents", {}).get("defaults", {}):
        console.print(
            "[dim]Hint: `memoryWindow` in your config is no longer used "
            "and can be safely removed.[/dim]"
        )
    if "contextWindowTokens" in raw.get("agents", {}).get("defaults", {}):
        console.print(
            "[dim]Hint: `contextWindowTokens` has moved to `maxTokens.input`.[/dim]"
        )
    if "maxInputTokens" in raw.get("agents", {}).get("defaults", {}):
        console.print(
            "[dim]Hint: `maxInputTokens` has moved to `maxTokens.input`.[/dim]"
        )


def _migrate_cron_store(config: "Config") -> None:
    """One-time migration: move legacy global cron store into the workspace."""
    from nanobot.config.paths import get_cron_dir

    legacy_path = get_cron_dir() / "jobs.json"
    new_path = config.workspace_path / "cron" / "jobs.json"
    if legacy_path.is_file() and not new_path.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.move(str(legacy_path), str(new_path))


# ============================================================================
# OpenAI-Compatible API Server
# ============================================================================


@app.command()
def serve(
    port: int | None = typer.Option(None, "--port", "-p", help="API server port"),
    host: str | None = typer.Option(None, "--host", "-H", help="Bind address"),
    timeout: float | None = typer.Option(None, "--timeout", "-t", help="Per-request timeout (seconds)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show nanobot runtime logs"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Start the OpenAI-compatible API server (/v1/chat/completions)."""
    try:
        from aiohttp import web  # noqa: F401
    except ImportError:
        console.print("[red]aiohttp is required. Install with: pip install 'nanobot-ai[api]'[/red]")
        raise typer.Exit(1)

    from loguru import logger

    from nanobot.agent.loop import AgentLoop
    from nanobot.api.server import create_app
    from nanobot.bus.queue import MessageBus
    from nanobot.session.manager import SessionManager

    if verbose:
        logger.enable("nanobot")
    else:
        logger.disable("nanobot")

    runtime_config = _load_runtime_config(config, workspace)
    api_cfg = runtime_config.api
    host = host if host is not None else api_cfg.host
    port = port if port is not None else api_cfg.port
    timeout = timeout if timeout is not None else api_cfg.timeout
    sync_workspace_templates(runtime_config.workspace_path)
    bus = MessageBus()
    provider = _make_provider(runtime_config)
    crawler_provider = _make_crawler_provider(runtime_config)
    session_manager = SessionManager(runtime_config.workspace_path)
    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=runtime_config.workspace_path,
        model=runtime_config.agents.defaults.model,
        max_tokens=runtime_config.agents.defaults.max_tokens,
        max_iterations=runtime_config.agents.defaults.max_tool_iterations,
        crawler_agent_config=runtime_config.agents.crawler,
        crawler_provider=crawler_provider,
        web_search_config=runtime_config.tools.web.search,
        web_proxy=runtime_config.tools.web.proxy or None,
        exec_config=runtime_config.tools.exec,
        image_config=runtime_config.tools.image,
        restrict_to_workspace=runtime_config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=runtime_config.tools.mcp_servers,
        channels_config=runtime_config.channels,
        timezone=runtime_config.agents.defaults.timezone,
        context_paths=_resolve_context_paths(runtime_config),
        context_repos=runtime_config.agents.defaults.context_repos,
        tool_result_clearing_keep=runtime_config.agents.defaults.tool_result_clearing_keep,
        consolidation_trigger_ratio=runtime_config.agents.defaults.consolidation_trigger_ratio,
        consolidation_target_ratio=runtime_config.agents.defaults.consolidation_target_ratio,
    )

    model_name = runtime_config.agents.defaults.model
    console.print(f"{__logo__} Starting OpenAI-compatible API server")
    console.print(f"  [cyan]Endpoint[/cyan] : http://{host}:{port}/v1/chat/completions")
    console.print(f"  [cyan]Model[/cyan]    : {model_name}")
    console.print("  [cyan]Session[/cyan]  : api:default")
    console.print(f"  [cyan]Timeout[/cyan]  : {timeout}s")
    if host in {"0.0.0.0", "::"}:
        console.print(
            "[yellow]Warning:[/yellow] API is bound to all interfaces. "
            "Only do this behind a trusted network boundary, firewall, or reverse proxy."
        )
    console.print()

    api_app = create_app(agent_loop, model_name=model_name, request_timeout=timeout)

    async def on_startup(_app):
        await agent_loop._connect_mcp()

    async def on_cleanup(_app):
        await agent_loop.close_mcp()

    api_app.on_startup.append(on_startup)
    api_app.on_cleanup.append(on_cleanup)

    web.run_app(api_app, host=host, port=port, print=lambda msg: logger.info(msg))





def _resolve_context_paths(config: Config) -> list[Path] | None:
    """Resolve context_paths from config to a list of expanded Path objects."""
    raw = config.agents.defaults.context_paths
    if not raw:
        return None
    return [Path(p).expanduser().resolve() for p in raw if p]


def _link_cron_result_to_visible_chat(
    agent_loop: Any,
    *,
    job: Any,
    run_id: str | None,
    visible_session_key: str,
    instruction: str,
    response: str,
    status: str = "completed",
    approval_granted: bool = False,
    stop_reason: str | None = None,
    occurrence_id: str | None = None,
) -> None:
    """Record a bounded scheduled result reference for the visible chat.

    Cron execution keeps its complete, fresh-history trace in ``cron:<job>``.
    This separate link is intentionally bounded and tagged with the canonical
    run ID so normal chat history never receives another full scheduled turn.
    """
    if not run_id or not visible_session_key:
        return

    visible_session = agent_loop.sessions.get_or_create(visible_session_key)
    existing = visible_session.get_run_messages(run_id)
    if any(
        message.get("role") == "user" and message.get("content") == instruction
        for message in existing
    ) and any(
        message.get("role") == "assistant" and message.get("content") == response
        for message in existing
    ):
        return

    # Keep complete detail in the cron session.  The visible session receives
    # only the bounded ring metadata; legacy wrapper pairs remain on disk and
    # are filtered request-locally by ``Session.get_model_history``.
    visible_session.add_scheduled_run(
        job_id=str(job.id),
        run_id=run_id,
        instruction=instruction,
        result=response,
        detail_session_key=f"cron:{job.id}",
        status=status,
        approval_granted=approval_granted,
        stop_reason=stop_reason,
        occurrence_id=occurrence_id,
    )
    agent_loop.sessions.save(visible_session)


async def _execute_cron_job(agent_loop: Any, bus: Any, job: Any) -> Any:
    """Run one scheduled instruction through ``execute_turn`` and fan it out."""
    from nanobot.agent.tools.message import MessageTool
    from nanobot.agent.turn import (
        ApprovalGrant,
        DeliveryTarget,
        HistoryMode,
        ScheduledTurnLink,
        TurnCallbacks,
        TurnRequest,
        TurnSource,
    )
    from nanobot.cron.delivery import build_explicit_fanout_messages, build_result_messages

    delivery_destinations = job.payload.delivery_destinations()
    route_channel = job.payload.channel or "cli"
    route_chat = job.payload.to or "direct"
    visible_session_key = f"{route_channel}:{route_chat}"
    scheduled_link = ScheduledTurnLink(
        job_id=job.id,
        job_name=job.name,
        instruction=job.payload.message,
        visible_session_key=visible_session_key,
        additional_destinations=tuple(
            DeliveryTarget(channel=destination.channel, chat_id=destination.to)
            for destination in delivery_destinations[1:]
        ),
        occurrence_id=str(
            getattr(job, "_occurrence_id", job.state.last_run_at_ms or "")
        ),
    )

    async def _silent_progress(*_args: Any, **_kwargs: Any) -> None:
        return None

    request = TurnRequest(
        content=job.payload.message,
        source=TurnSource.CRON,
        session_key=f"cron:{job.id}",
        route=DeliveryTarget(channel=route_channel, chat_id=route_chat),
        approval_grant=ApprovalGrant(approved=True, source="cron"),
        history_mode=HistoryMode.FRESH,
        scheduled_link=scheduled_link,
        callbacks=TurnCallbacks(on_progress=_silent_progress),
        metadata={"cron_job_id": job.id},
    )
    try:
        result = await agent_loop.execute_turn(request)
    except Exception as exc:
        agent_loop.record_task_failure(
            session_key=f"cron:{job.id}",
            channel=route_channel,
            chat_id=route_chat,
            label=f"Scheduled task '{job.name}'",
            task=job.payload.message,
            error=str(exc),
        )
        raise

    response = getattr(result, "content", None)
    if response is None:
        outbound = getattr(result, "outbound", None)
        response = getattr(outbound, "content", None) if outbound else None
    response = str(response or "Task completed.")
    run_id = getattr(result, "run_id", None)
    raw_status = getattr(result, "status", None)
    status = getattr(raw_status, "value", raw_status)
    if status is None:
        status = getattr(result, "stop_reason", None) or "completed"
    status = str(status)
    stop_reason = getattr(result, "stop_reason", None)
    missing_ledger = object()
    raw_sent_messages = getattr(result, "sent_messages", missing_ledger)
    if raw_sent_messages is missing_ledger:
        # Compatibility for test/in-process loops that have not yet surfaced
        # the canonical delivery ledger.
        sent_messages = []
        message_tool = agent_loop.tools.get("message")
        if isinstance(message_tool, MessageTool):
            sent_messages = list(message_tool.sent_messages_in_turn)
    else:
        # An empty canonical ledger is authoritative for this run.  Never
        # borrow messages from the shared legacy MessageTool singleton.
        sent_messages = list(raw_sent_messages or ())

    _link_cron_result_to_visible_chat(
        agent_loop,
        job=job,
        run_id=str(run_id) if run_id else None,
        visible_session_key=visible_session_key,
        instruction=job.payload.message,
        response=response,
        status=status,
        approval_granted=bool(request.approval_grant and request.approval_grant.granted),
        stop_reason=str(stop_reason) if stop_reason else None,
        occurrence_id=(
            getattr(job, "_occurrence_id", None)
            or (str(job.state.last_run_at_ms) if job.state.last_run_at_ms else None)
        ),
    )
    explicit_fanout = build_explicit_fanout_messages(
        delivery_destinations,
        sent_messages,
    )
    for outbound in explicit_fanout:
        await bus.publish_outbound(outbound)

    if delivery_destinations and response:
        try:
            # An explicit attachment or auxiliary message must never replace
            # the completed text result of the scheduled task.
            for outbound in build_result_messages(
                response,
                delivery_destinations,
                [*sent_messages, *explicit_fanout],
            ):
                await bus.publish_outbound(outbound)
        except Exception as exc:
            agent_loop.record_task_failure(
                session_key=f"cron:{job.id}",
                channel=route_channel,
                chat_id=route_chat,
                label=f"Scheduled task delivery '{job.name}'",
                task=job.payload.message,
                error=str(exc),
            )
            raise
    return result


# ============================================================================
# Gateway / Server
# ============================================================================


@app.command()
def gateway(
    port: int | None = typer.Option(None, "--port", "-p", help="Gateway port"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    log_file: str | None = typer.Option(None, "--log-file", help="Write logs to this file (rotated at 20 MB, 7 days kept)"),
    no_cron: bool = typer.Option(False, "--no-cron", help="Disable cron jobs on this run"),
):
    """Start the nanobot gateway."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.channels.manager import ChannelManager
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronJob
    from nanobot.heartbeat.service import HeartbeatService
    from nanobot.session.manager import SessionManager

    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        # Also enable loguru debug output
        from loguru import logger as _logger
        _logger.remove()
        _logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )
        _logger.enable("nanobot")

    if log_file:
        from loguru import logger as _logger
        _logger.add(
            log_file,
            rotation="20 MB",
            retention="7 days",
            compression="gz",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            enqueue=True,
        )
        _logger.enable("nanobot")
        console.print(f"[dim]Logging to {log_file}[/dim]")

    config = _load_runtime_config(config, workspace)
    port = port if port is not None else config.gateway.port

    console.print(f"{__logo__} Starting nanobot gateway version {__version__} on port {port}...")
    sync_workspace_templates(config.workspace_path)
    bus = MessageBus()
    provider = _make_provider(config)
    crawler_provider = _make_crawler_provider(config)
    session_manager = SessionManager(config.workspace_path)

    # Preserve existing single-workspace installs, but keep custom workspaces clean.
    if is_default_workspace(config.workspace_path):
        _migrate_cron_store(config)

    # Create cron service with workspace-scoped store
    cron_store_path = config.workspace_path / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    # Create agent with cron service
    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        crawler_agent_config=config.agents.crawler,
        crawler_provider=crawler_provider,
        web_search_config=config.tools.web.search,
        web_proxy=config.tools.web.proxy or None,
        agent_browser_config=config.tools.agent_browser,
        agent_device_config=config.tools.agent_device,
        desktop_use_config=config.tools.desktop_use,
        exec_config=config.tools.exec,
        image_config=config.tools.image,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
        timezone=config.agents.defaults.timezone,
        context_paths=_resolve_context_paths(config),
        context_repos=config.agents.defaults.context_repos,
        tool_result_clearing_keep=config.agents.defaults.tool_result_clearing_keep,
        consolidation_trigger_ratio=config.agents.defaults.consolidation_trigger_ratio,
        consolidation_target_ratio=config.agents.defaults.consolidation_target_ratio,
    )

    # Set cron callback (needs agent)
    async def on_cron_job(job: CronJob) -> Any:
        """Execute a cron job through the agent."""
        return await _execute_cron_job(agent, bus, job)
    cron.on_job = on_cron_job

    # Create channel manager
    channels = ChannelManager(config, bus)

    def _pick_heartbeat_target() -> tuple[str, str]:
        """Pick a routable channel/chat target for heartbeat-triggered messages."""
        enabled = set(channels.enabled_channels)
        # Prefer the most recently updated non-internal session on an enabled channel.
        for item in session_manager.list_sessions():
            key = item.get("key") or ""
            if ":" not in key:
                continue
            channel, chat_id = key.split(":", 1)
            if channel in {"cli", "system"}:
                continue
            if channel in enabled and chat_id:
                return channel, chat_id
        # Fallback keeps prior behavior but remains explicit.
        return "cli", "direct"

    # Create heartbeat service
    async def on_heartbeat_execute(tasks: str) -> str:
        """Phase 2: execute heartbeat tasks through the full agent loop."""
        from nanobot.agent.turn import DeliveryTarget, TurnCallbacks, TurnRequest, TurnSource

        channel, chat_id = _pick_heartbeat_target()

        async def _silent(*_args, **_kwargs):
            pass

        result = await agent.execute_turn(TurnRequest(
            content=tasks,
            source=TurnSource.HEARTBEAT,
            session_key="heartbeat",
            route=DeliveryTarget(channel=channel, chat_id=chat_id),
            callbacks=TurnCallbacks(on_progress=_silent),
        ))

        # Keep a small tail of heartbeat history so the loop stays bounded
        # without losing all short-term context between runs.
        from nanobot.session.manager import SessionWriteConflict

        for _attempt in range(3):
            session = agent.sessions.get_or_create("heartbeat")
            session.retain_recent_legal_suffix(
                hb_cfg.keep_recent_messages,
                expected_revision=session.revision,
            )
            try:
                agent.sessions.save(session)
                break
            except SessionWriteConflict:
                agent.sessions.invalidate("heartbeat")
        else:
            raise SessionWriteConflict()

        return str(getattr(result, "content", None) or "")

    async def on_heartbeat_notify(response: str) -> None:
        """Deliver a heartbeat response to the user's channel."""
        from nanobot.bus.events import OutboundMessage
        channel, chat_id = _pick_heartbeat_target()
        if channel == "cli":
            return  # No external channel available to deliver to
        await bus.publish_outbound(OutboundMessage(channel=channel, chat_id=chat_id, content=response))

    hb_cfg = config.gateway.heartbeat
    heartbeat = HeartbeatService(
        workspace=config.workspace_path,
        provider=provider,
        model=agent.model,
        on_execute=on_heartbeat_execute,
        on_notify=on_heartbeat_notify,
        interval_s=hb_cfg.interval_s,
        enabled=hb_cfg.enabled,
        timezone=config.agents.defaults.timezone,
    )

    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")

    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")

    console.print(f"[green]✓[/green] Heartbeat: every {hb_cfg.interval_s}s")

    async def run():
        try:
            if not no_cron:
                await cron.start()
            await heartbeat.start()
            await asyncio.gather(
                agent.run(),
                channels.start_all(),
            )
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        except Exception:
            import traceback
            console.print("\n[red]Error: Gateway crashed unexpectedly[/red]")
            console.print(traceback.format_exc())
        finally:
            cron.stop()
            heartbeat.stop()
            agent.stop()
            agent.cancel_active_tasks()
            await agent.close_mcp()
            await channels.stop_all()

    asyncio.run(run())


@app.command("cron")
def cron_cmd(
    action: str = typer.Argument(..., help="Action: list, enable, disable"),
    job_id: str = typer.Option(None, "--job", "-j", help="Job ID (for enable/disable)"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
):
    """Manage cron jobs (list, enable, disable)."""
    from nanobot.cron.service import CronService

    cfg = _load_runtime_config(config, workspace)
    store_path = cfg.workspace_path / "cron" / "jobs.json"
    if not store_path.exists():
        console.print("[yellow]No cron jobs found.[/yellow]")
        raise typer.Exit()

    svc = CronService(store_path)

    if action == "list":
        jobs = svc.list_jobs(include_disabled=True)
        if not jobs:
            console.print("[yellow]No cron jobs.[/yellow]")
            return
        for j in jobs:
            status = "[green]enabled[/green]" if j.enabled else "[dim]disabled[/dim]"
            sched = j.schedule
            if sched.kind == "every":
                desc = f"every {sched.every_ms // 1000}s"
            elif sched.kind == "cron":
                desc = f"cron: {sched.expr}"
            elif sched.kind == "at":
                desc = f"at: {sched.at_ms}"
            else:
                desc = str(sched)
            console.print(f"  {j.id}  {status}  {j.name}  ({desc})")
    elif action == "enable":
        if not job_id:
            console.print("[red]--job is required for enable[/red]")
            raise typer.Exit(1)
        job = svc.enable_job(job_id, enabled=True)
        if job:
            console.print(f"[green]Enabled {job.name}[/green]")
        else:
            console.print(f"[red]Job {job_id} not found[/red]")
            raise typer.Exit(1)
    elif action == "disable":
        if not job_id:
            console.print("[red]--job is required for disable[/red]")
            raise typer.Exit(1)
        job = svc.enable_job(job_id, enabled=False)
        if job:
            console.print(f"[yellow]Disabled {job.name}[/yellow]")
        else:
            console.print(f"[red]Job {job_id} not found[/red]")
            raise typer.Exit(1)
    else:
        console.print(f"[red]Unknown action: {action}. Use list, enable, or disable.[/red]")
        raise typer.Exit(1)




# ============================================================================
# Agent Commands
# ============================================================================


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:direct", "--session", "-s", help="Session ID"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render assistant output as Markdown"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="Show nanobot runtime logs during chat"),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="Write runtime logs to this file (rotated at 20 MB, 7 days kept)",
    ),
):
    """Interact with the agent directly."""
    from loguru import logger

    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.cron.service import CronService

    config = _load_runtime_config(config, workspace)
    sync_workspace_templates(config.workspace_path)

    bus = MessageBus()
    provider = _make_provider(config)
    crawler_provider = _make_crawler_provider(config)

    # Preserve existing single-workspace installs, but keep custom workspaces clean.
    if is_default_workspace(config.workspace_path):
        _migrate_cron_store(config)

    # Create cron service with workspace-scoped store
    cron_store_path = config.workspace_path / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    if logs or log_file:
        logger.remove()
        if logs:
            # Let prompt_toolkit intercept runtime logs so they do not overwrite
            # the interactive input prompt.
            logger.add(
                lambda msg: sys.stderr.write(msg),
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            )
        if log_file:
            logger.add(
                log_file,
                rotation="20 MB",
                retention="7 days",
                compression="gz",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                enqueue=True,
            )
        logger.enable("nanobot")
        if logs:
            console.print("[dim]Runtime logs enabled (--logs). Use --no-logs to hide internal logs.[/dim]")
        if log_file:
            console.print(f"[dim]Logging to {log_file}[/dim]")
    else:
        logger.disable("nanobot")

    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        crawler_agent_config=config.agents.crawler,
        crawler_provider=crawler_provider,
        web_search_config=config.tools.web.search,
        web_proxy=config.tools.web.proxy or None,
        agent_browser_config=config.tools.agent_browser,
        agent_device_config=config.tools.agent_device,
        desktop_use_config=config.tools.desktop_use,
        exec_config=config.tools.exec,
        image_config=config.tools.image,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
        timezone=config.agents.defaults.timezone,
        context_paths=_resolve_context_paths(config),
        context_repos=config.agents.defaults.context_repos,
        tool_result_clearing_keep=config.agents.defaults.tool_result_clearing_keep,
        consolidation_trigger_ratio=config.agents.defaults.consolidation_trigger_ratio,
        consolidation_target_ratio=config.agents.defaults.consolidation_target_ratio,
    )

    # Shared reference for progress callbacks
    _thinking: ThinkingSpinner | None = None

    async def _cli_progress(content: str, *, tool_hint: bool = False) -> None:
        ch = agent_loop.channels_config
        if ch and tool_hint and not ch.send_tool_hints:
            return
        _print_cli_progress_line(content, _thinking)

    if message:
        # Single message mode — direct call, no bus needed
        async def run_once():
            from nanobot.agent.turn import DeliveryTarget, TurnCallbacks, TurnRequest, TurnSource

            if ":" in session_id:
                route_channel, route_chat = session_id.split(":", 1)
            else:
                route_channel, route_chat = "cli", session_id
            renderer = StreamRenderer(render_markdown=markdown, use_live=not logs)
            response = await agent_loop.execute_turn(TurnRequest(
                content=message,
                source=TurnSource.DIRECT,
                session_key=session_id,
                route=DeliveryTarget(channel=route_channel, chat_id=route_chat),
                callbacks=TurnCallbacks(
                    on_progress=_cli_progress,
                    on_stream=renderer.on_delta,
                    on_stream_end=renderer.on_end,
                ),
            ))
            outbound = getattr(response, "outbound", None)
            response_content = getattr(response, "content", None)
            if response_content is None and outbound is not None:
                response_content = getattr(outbound, "content", "")
            response_metadata = getattr(outbound, "metadata", {}) if outbound else {}
            if response_content is None:
                response_content = ""
            if not renderer.streamed:
                await renderer.close()
                _print_agent_response(
                    response_content,
                    render_markdown=markdown,
                    metadata=response_metadata,
                )
            await agent_loop.close_mcp()

        asyncio.run(run_once())
    else:
        # Interactive mode — route through bus like other channels
        from nanobot.bus.events import InboundMessage
        _init_prompt_session()
        console.print(f"{__logo__} Interactive mode (type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit)\n")

        if ":" in session_id:
            cli_channel, cli_chat_id = session_id.split(":", 1)
        else:
            cli_channel, cli_chat_id = "cli", session_id

        def _handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name
            _restore_terminal()
            console.print(f"\nReceived {sig_name}, goodbye!")
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        # SIGHUP is not available on Windows
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, _handle_signal)
        # Ignore SIGPIPE to prevent silent process termination when writing to closed pipes
        # SIGPIPE is not available on Windows
        if hasattr(signal, 'SIGPIPE'):
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)

        async def run_interactive():
            bus_task = asyncio.create_task(agent_loop.run())
            turn_done = asyncio.Event()
            turn_done.set()
            turn_response: list[tuple[str, dict]] = []
            renderer: StreamRenderer | None = None

            async def _consume_outbound():
                while True:
                    try:
                        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

                        if msg.metadata.get("_stream_delta"):
                            if renderer:
                                await renderer.on_delta(msg.content)
                            continue
                        if msg.metadata.get("_stream_end"):
                            if renderer:
                                await renderer.on_end(
                                    resuming=msg.metadata.get("_resuming", False),
                                )
                            # Signal the dispatcher that we've finished rendering
                            # this stream segment so it can safely emit subsequent
                            # log lines without racing our terminating newline.
                            ack = msg.metadata.get("_stream_render_ack")
                            if ack is not None:
                                try:
                                    ack.set()
                                except AttributeError:
                                    pass
                            continue
                        if msg.metadata.get("_streamed"):
                            if msg.metadata.get("_cli_turn_complete"):
                                turn_done.set()
                            continue

                        if msg.metadata.get("_progress"):
                            is_tool_hint = msg.metadata.get("_tool_hint", False)
                            ch = agent_loop.channels_config
                            if ch and is_tool_hint and not ch.send_tool_hints:
                                pass
                            else:
                                await _print_interactive_progress_line(msg.content, _thinking)
                            continue

                        if msg.metadata.get("_cli_turn_complete"):
                            if msg.content:
                                turn_response.append((msg.content, dict(msg.metadata or {})))
                            turn_done.set()
                        elif msg.content:
                            await _print_interactive_response(
                                msg.content,
                                render_markdown=markdown,
                                metadata=msg.metadata,
                            )

                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

            outbound_task = asyncio.create_task(_consume_outbound())

            try:
                while True:
                    try:
                        _flush_pending_tty_input()
                        user_input = await _read_interactive_input_async()
                        command = user_input.strip()
                        if not command:
                            continue

                        if _is_exit_command(command):
                            _restore_terminal()
                            console.print("\nGoodbye!")
                            break

                        turn_done.clear()
                        turn_response.clear()
                        renderer = StreamRenderer(render_markdown=markdown, use_live=not logs)

                        await bus.publish_inbound(InboundMessage(
                            channel=cli_channel,
                            sender_id="user",
                            chat_id=cli_chat_id,
                            content=user_input,
                            # Interactive PowerShell output is rendered once at turn
                            # completion. Rich Live cursor redraws are not reliable in
                            # the legacy Windows console host, and a MessageTool send
                            # may occur before the agent turn has actually completed.
                            metadata={"_wants_stream": False},
                        ))

                        await turn_done.wait()

                        if turn_response:
                            content, meta = turn_response[0]
                            if content and not meta.get("_streamed"):
                                if renderer:
                                    await renderer.close()
                                _print_agent_response(
                                    content, render_markdown=markdown, metadata=meta,
                                )
                        elif renderer and not renderer.streamed:
                            await renderer.close()
                    except KeyboardInterrupt:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
                    except EOFError:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
            finally:
                agent_loop.stop()
                outbound_task.cancel()
                await asyncio.gather(bus_task, outbound_task, return_exceptions=True)
                await agent_loop.close_mcp()

        asyncio.run(run_interactive())


# ============================================================================
# Channel Commands
# ============================================================================


channels_app = typer.Typer(help="Manage channels")
app.add_typer(channels_app, name="channels")


@channels_app.command("status")
def channels_status():
    """Show channel status."""
    from nanobot.channels.registry import discover_all
    from nanobot.config.loader import load_config

    config = load_config()

    table = Table(title="Channel Status")
    table.add_column("Channel", style="cyan")
    table.add_column("Enabled", style="green")

    for name, cls in sorted(discover_all().items()):
        section = getattr(config.channels, name, None)
        if section is None:
            enabled = False
        elif isinstance(section, dict):
            enabled = section.get("enabled", False)
        else:
            enabled = getattr(section, "enabled", False)
        table.add_row(
            cls.display_name,
            "[green]\u2713[/green]" if enabled else "[dim]\u2717[/dim]",
        )

    console.print(table)


def _get_bridge_dir() -> Path:
    """Get the bridge directory, setting it up if needed."""
    import shutil
    import subprocess

    # User's bridge location
    from nanobot.config.paths import get_bridge_install_dir

    user_bridge = get_bridge_install_dir()

    # Check if already built
    if (user_bridge / "dist" / "index.js").exists():
        return user_bridge

    # Check for npm
    npm_path = shutil.which("npm")
    if not npm_path:
        console.print("[red]npm not found. Please install Node.js >= 18.[/red]")
        raise typer.Exit(1)

    # Find source bridge: first check package data, then source dir
    pkg_bridge = Path(__file__).parent.parent / "bridge"  # nanobot/bridge (installed)
    src_bridge = Path(__file__).parent.parent.parent / "bridge"  # repo root/bridge (dev)

    source = None
    if (pkg_bridge / "package.json").exists():
        source = pkg_bridge
    elif (src_bridge / "package.json").exists():
        source = src_bridge

    if not source:
        console.print("[red]Bridge source not found.[/red]")
        console.print("Try reinstalling: pip install --force-reinstall nanobot")
        raise typer.Exit(1)

    console.print(f"{__logo__} Setting up bridge...")

    # Copy to user directory
    user_bridge.parent.mkdir(parents=True, exist_ok=True)
    if user_bridge.exists():
        shutil.rmtree(user_bridge)
    shutil.copytree(source, user_bridge, ignore=shutil.ignore_patterns("node_modules", "dist"))

    # Install and build
    try:
        console.print("  Installing dependencies...")
        subprocess.run([npm_path, "install"], cwd=user_bridge, check=True, capture_output=True)

        console.print("  Building...")
        subprocess.run([npm_path, "run", "build"], cwd=user_bridge, check=True, capture_output=True)

        console.print("[green]✓[/green] Bridge ready\n")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        if e.stderr:
            console.print(f"[dim]{e.stderr.decode()[:500]}[/dim]")
        raise typer.Exit(1)

    return user_bridge


@channels_app.command("login")
def channels_login(
    channel_name: str = typer.Argument(..., help="Channel name (e.g. weixin, whatsapp)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-authentication even if already logged in"),
):
    """Authenticate with a channel via QR code or other interactive login."""
    from nanobot.channels.registry import discover_all
    from nanobot.config.loader import load_config

    config = load_config()
    channel_cfg = getattr(config.channels, channel_name, None) or {}

    # Validate channel exists
    all_channels = discover_all()
    if channel_name not in all_channels:
        available = ", ".join(all_channels.keys())
        console.print(f"[red]Unknown channel: {channel_name}[/red]  Available: {available}")
        raise typer.Exit(1)

    console.print(f"{__logo__} {all_channels[channel_name].display_name} Login\n")

    channel_cls = all_channels[channel_name]
    channel = channel_cls(channel_cfg, bus=None)

    success = asyncio.run(channel.login(force=force))

    if not success:
        raise typer.Exit(1)


# ============================================================================
# Plugin Commands
# ============================================================================

plugins_app = typer.Typer(help="Manage channel plugins")
app.add_typer(plugins_app, name="plugins")


@plugins_app.command("list")
def plugins_list():
    """List all discovered channels (built-in and plugins)."""
    from nanobot.channels.registry import discover_all, discover_channel_names
    from nanobot.config.loader import load_config

    config = load_config()
    builtin_names = set(discover_channel_names())
    all_channels = discover_all()

    table = Table(title="Channel Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Source", style="magenta")
    table.add_column("Enabled", style="green")

    for name in sorted(all_channels):
        cls = all_channels[name]
        source = "builtin" if name in builtin_names else "plugin"
        section = getattr(config.channels, name, None)
        if section is None:
            enabled = False
        elif isinstance(section, dict):
            enabled = section.get("enabled", False)
        else:
            enabled = getattr(section, "enabled", False)
        table.add_row(
            cls.display_name,
            source,
            "[green]yes[/green]" if enabled else "[dim]no[/dim]",
        )

    console.print(table)


# ============================================================================
# Status Commands
# ============================================================================


@app.command()
def status():
    """Show nanobot status."""
    from nanobot.config.loader import get_config_path, load_config

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} nanobot Status\n")

    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    if config_path.exists():
        from nanobot.providers.registry import PROVIDERS

        console.print(f"Model: {config.agents.defaults.model}")

        # Check API keys from registry
        for spec in PROVIDERS:
            p = getattr(config.providers, spec.name, None)
            if p is None:
                continue
            if spec.is_oauth:
                console.print(f"{spec.label}: [green]✓ (OAuth)[/green]")
            elif spec.is_local:
                # Local deployments show api_base instead of api_key
                if p.api_base:
                    console.print(f"{spec.label}: [green]✓ {p.api_base}[/green]")
                else:
                    console.print(f"{spec.label}: [dim]not set[/dim]")
            else:
                has_key = bool(p.effective_keys)
                console.print(f"{spec.label}: {'[green]✓[/green]' if has_key else '[dim]not set[/dim]'}")


# ============================================================================
# Artifact Commands
# ============================================================================


artifacts_app = typer.Typer(help="Manage browser test artifacts")
app.add_typer(artifacts_app, name="artifacts")


def _get_browser_artifacts_path(config: str | None = None, workspace: str | None = None) -> Path:
    """Resolve the browser test artifacts directory for the active instance."""
    runtime_config = _load_runtime_config(config, workspace)
    return get_browser_test_artifacts_dir(runtime_config.workspace_path)


@artifacts_app.command("path")
def artifacts_path(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
):
    """Show the workspace path used for browser test artifacts."""
    artifacts_dir = _get_browser_artifacts_path(config, workspace)
    entries = [p for p in artifacts_dir.iterdir() if p.name != ".gitkeep"]

    console.print(f"Browser test artifacts: {artifacts_dir}")
    console.print(f"Entries: {len(entries)}")


@artifacts_app.command("clean")
def artifacts_clean(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Remove all saved browser test artifacts while preserving the root directory."""
    artifacts_dir = _get_browser_artifacts_path(config, workspace)
    targets = [p for p in artifacts_dir.iterdir() if p.name != ".gitkeep"]

    if not targets:
        console.print(f"[green]✓[/green] Browser test artifacts already empty: {artifacts_dir}")
        return

    if not yes:
        confirmed = typer.confirm(
            f"Delete {len(targets)} item(s) from {artifacts_dir}?",
            default=False,
        )
        if not confirmed:
            console.print("Cancelled.")
            raise typer.Exit(1)

    removed = 0
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        removed += 1

    console.print(f"[green]✓[/green] Removed {removed} item(s) from {artifacts_dir}")


# ============================================================================
# OAuth Login
# ============================================================================

provider_app = typer.Typer(help="Manage providers")
app.add_typer(provider_app, name="provider")


_LOGIN_HANDLERS: dict[str, callable] = {}


def _register_login(name: str):
    def decorator(fn):
        _LOGIN_HANDLERS[name] = fn
        return fn
    return decorator


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(..., help="OAuth provider (e.g. 'openai-codex', 'github-copilot')"),
):
    """Authenticate with an OAuth provider."""
    from nanobot.providers.registry import PROVIDERS

    key = provider.replace("-", "_")
    spec = next((s for s in PROVIDERS if s.name == key and s.is_oauth), None)
    if not spec:
        names = ", ".join(s.name.replace("_", "-") for s in PROVIDERS if s.is_oauth)
        console.print(f"[red]Unknown OAuth provider: {provider}[/red]  Supported: {names}")
        raise typer.Exit(1)

    handler = _LOGIN_HANDLERS.get(spec.name)
    if not handler:
        console.print(f"[red]Login not implemented for {spec.label}[/red]")
        raise typer.Exit(1)

    console.print(f"{__logo__} OAuth Login - {spec.label}\n")
    handler()


@_register_login("openai_codex")
def _login_openai_codex() -> None:
    from nanobot.providers.codex_auth import get_token as get_codex_token

    try:
        from oauth_cli_kit import login_oauth_interactive
        console.print("[cyan]Starting interactive OAuth login...[/cyan]\n")
        token = login_oauth_interactive(
            print_fn=lambda s: console.print(s),
            prompt_fn=lambda s: typer.prompt(s),
            originator="codex_cli_rs",
        )
        if not (token and token.access):
            console.print("[red]✗ Authentication failed[/red]")
            raise typer.Exit(1)
        console.print(f"[green]✓ Authenticated with OpenAI Codex[/green]  [dim]{token.account_id}[/dim]")
    except ImportError:
        console.print("[red]oauth_cli_kit not installed. Run: pip install oauth-cli-kit[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        try:
            token = get_codex_token()
        except Exception:
            console.print(f"[red]Authentication error: {exc}[/red]")
            console.print(
                "[yellow]Headless workaround:[/yellow] run `codex login` on your laptop, then copy "
                "`~/.codex/auth.json` (or `$CODEX_HOME/auth.json`) to this machine and retry."
            )
            raise typer.Exit(1)
        console.print(
            "[yellow]Interactive exchange failed, but imported existing Codex credentials successfully.[/yellow]"
        )
        console.print(f"[green]✓ Authenticated with OpenAI Codex[/green]  [dim]{token.account_id}[/dim]")


@_register_login("github_copilot")
def _login_github_copilot() -> None:
    from nanobot.providers import copilot_auth

    console.print("[cyan]Starting GitHub Copilot device flow...[/cyan]\n")

    def _show(user_code: str, verification_uri: str) -> None:
        console.print(f"  1. Open: [link]{verification_uri}[/link]")
        console.print(f"  2. Enter code: [bold yellow]{user_code}[/bold yellow]\n")
        console.print("[dim]Waiting for approval... (Ctrl-C to cancel)[/dim]\n")

    try:
        copilot_auth.device_login(on_user_code=_show)
        console.print("[green]✓ Authenticated with GitHub Copilot[/green]")
        console.print(
            f"[dim]Credentials saved to {copilot_auth._auth_path()}[/dim]"
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Login cancelled[/yellow]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Authentication error: {exc}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
