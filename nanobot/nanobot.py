"""High-level programmatic interface to nanobot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanobot.agent.hook import AgentHook
from nanobot.agent.loop import AgentLoop
from nanobot.agent.message_content import content_to_text
from nanobot.agent.turn import DeliveryTarget, TraceMode, TurnRequest, TurnSource
from nanobot.bus.queue import MessageBus


@dataclass(slots=True)
class RunResult:
    """Result of a single agent run."""

    content: str = ""
    tools_used: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    run_id: str | None = None
    stop_reason: str | None = None


def _sanitize_trace(messages: Any) -> list[dict[str, Any]]:
    """Return a detached, bounded SDK trace without internal/raw fields.

    The canonical loop retains the full provider exchange for session
    persistence.  SDK callers only receive this reduced view when explicitly
    opting in: system prompts, internal metadata, media payloads, and tool
    arguments are omitted while textual content remains useful for debugging.
    """
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        return []

    detached: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, Mapping):
            continue
        role = raw.get("role")
        if role == "system" or not role:
            continue

        entry: dict[str, Any] = {"role": str(role)}
        if "content" in raw:
            entry["content"] = content_to_text(raw.get("content"), max_chars=4_000)

        if role == "assistant" and isinstance(raw.get("tool_calls"), Sequence):
            calls: list[dict[str, str]] = []
            for raw_call in raw["tool_calls"]:
                if not isinstance(raw_call, Mapping):
                    continue
                function = raw_call.get("function")
                name = function.get("name") if isinstance(function, Mapping) else None
                call: dict[str, str] = {}
                if raw_call.get("id"):
                    call["id"] = str(raw_call["id"])
                if name:
                    call["name"] = str(name)
                if call:
                    calls.append(call)
            if calls:
                entry["tool_calls"] = calls
        elif role == "tool":
            if raw.get("tool_call_id"):
                entry["tool_call_id"] = str(raw["tool_call_id"])
            if raw.get("name"):
                entry["name"] = str(raw["name"])

        detached.append(entry)
    return detached


class Nanobot:
    """Programmatic facade for running the nanobot agent.

    Usage::

        bot = Nanobot.from_config()
        result = await bot.run("Summarize this repo", hooks=[MyHook()])
        print(result.content)
    """

    def __init__(self, loop: AgentLoop) -> None:
        self._loop = loop

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        workspace: str | Path | None = None,
    ) -> Nanobot:
        """Create a Nanobot instance from a config file.

        Args:
            config_path: Path to ``config.json``.  Defaults to
                ``~/.nanobot/config.json``.
            workspace: Override the workspace directory from config.
        """
        from nanobot.config.loader import load_config
        from nanobot.config.schema import Config

        resolved: Path | None = None
        if config_path is not None:
            resolved = Path(config_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Config not found: {resolved}")

        config: Config = load_config(resolved)
        if workspace is not None:
            config.agents.defaults.workspace = str(
                Path(workspace).expanduser().resolve()
            )

        provider = _make_provider(config)
        crawler_provider = _make_crawler_provider(config)
        bus = MessageBus()
        defaults = config.agents.defaults

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=config.workspace_path,
            model=defaults.model,
            max_tokens=defaults.max_tokens,
            max_iterations=defaults.max_tool_iterations,
            crawler_agent_config=config.agents.crawler,
            crawler_provider=crawler_provider,
            web_search_config=config.tools.web.search,
            web_proxy=config.tools.web.proxy or None,
            exec_config=config.tools.exec,
            image_config=config.tools.image,
            restrict_to_workspace=config.tools.restrict_to_workspace,
            mcp_servers=config.tools.mcp_servers,
            channels_config=config.channels,
            timezone=defaults.timezone,
            context_paths=[Path(p).expanduser().resolve() for p in defaults.context_paths] if defaults.context_paths else None,
            context_repos=defaults.context_repos,
            tool_result_clearing_keep=defaults.tool_result_clearing_keep,
            consolidation_trigger_ratio=defaults.consolidation_trigger_ratio,
            consolidation_target_ratio=defaults.consolidation_target_ratio,
        )
        return cls(loop)

    async def run(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
        hooks: list[AgentHook] | None = None,
        include_messages: bool = False,
    ) -> RunResult:
        """Run the agent once and return the result.

        Args:
            message: The user message to process.
            session_key: Session identifier for conversation isolation.
                Different keys get independent history.
            hooks: Optional lifecycle hooks for this run.
            include_messages: Include a detached, sanitized execution trace.
        """
        request = TurnRequest(
            content=message,
            source=TurnSource.SDK,
            session_key=session_key,
            route=DeliveryTarget(channel="sdk", chat_id=session_key),
            hooks=tuple(hooks or ()),
            trace_mode=(TraceMode.SANITIZED if include_messages else TraceMode.NONE),
        )
        response = await self._loop.execute_turn(request)

        outbound = getattr(response, "outbound", None) if response is not None else None
        raw_content = getattr(response, "content", None) if response is not None else None
        if raw_content is None and outbound is not None:
            raw_content = getattr(outbound, "content", None)
        if raw_content is None and isinstance(response, str):
            raw_content = response
        content = str(raw_content or "")

        raw_tools = getattr(response, "tools_used", ()) if response is not None else ()
        if isinstance(raw_tools, str):
            tools_used = [raw_tools]
        else:
            tools_used = list(raw_tools or ())

        run_id = getattr(response, "run_id", None) if response is not None else None
        run_id = str(run_id) if run_id else None
        stop_reason = getattr(response, "stop_reason", None) if response is not None else None
        stop_reason = getattr(stop_reason, "value", stop_reason)
        stop_reason = str(stop_reason) if stop_reason else None

        raw_messages = getattr(response, "messages", None) if response is not None else None
        if include_messages and raw_messages is None and run_id:
            run_store = getattr(self._loop, "run_store", None)
            if run_store is not None:
                raw_messages = run_store.load_trace(run_id)

        return RunResult(
            content=content,
            tools_used=tools_used,
            messages=_sanitize_trace(raw_messages) if include_messages else [],
            run_id=run_id,
            stop_reason=stop_reason,
        )


def _make_single_provider(config: Any, provider_name: str, model: str) -> Any:
    """Create a single LLM provider instance for the given provider name and model."""
    from nanobot.providers.registry import find_by_name

    p, _ = config.get_provider_by_name(provider_name)
    keys = p.effective_keys if p else []
    primary_key = keys[0] if keys else None
    spec = find_by_name(provider_name)
    backend = spec.backend if spec else "openai_compat"

    if backend == "azure_openai":
        if not p or not primary_key or not p.api_base:
            raise ValueError("Azure OpenAI requires api_key and api_base in config.")
    elif backend == "openai_compat" and not model.startswith("bedrock/"):
        needs_key = not bool(primary_key)
        exempt = spec and (spec.is_oauth or spec.is_local or spec.is_direct)
        if needs_key and not exempt:
            raise ValueError(f"No API key configured for provider '{provider_name}'.")

    if backend == "openai_codex":
        from nanobot.providers.openai_codex_provider import OpenAICodexProvider

        return OpenAICodexProvider(
            default_model=model,
            workspace=str(config.workspace_path),
        )
    elif backend == "azure_openai":
        from nanobot.providers.azure_openai_provider import AzureOpenAIProvider

        return AzureOpenAIProvider(
            api_key=primary_key, api_base=p.api_base, default_model=model
        )
    elif backend == "anthropic":
        from nanobot.providers.anthropic_provider import AnthropicProvider

        api_base = p.api_base if p else None
        if not api_base and spec and spec.default_api_base:
            api_base = spec.default_api_base

        return AnthropicProvider(
            api_key=primary_key,
            api_base=api_base,
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )
    else:
        from nanobot.providers.openai_compat_provider import OpenAICompatProvider

        api_base = p.api_base if p else None
        if not api_base and spec:
            if spec.is_gateway or spec.is_local:
                api_base = spec.default_api_base or None
            elif spec.default_api_base:
                api_base = spec.default_api_base

        token_provider = None
        merged_headers = dict(p.extra_headers) if p and p.extra_headers else {}
        if spec and spec.name == "github_copilot":
            from nanobot.providers import copilot_auth

            token_provider = copilot_auth.get_copilot_bearer
            for k, v in copilot_auth.copilot_request_headers().items():
                merged_headers.setdefault(k, v)

        # Determine keep-alive interval for local providers
        _is_local_endpoint = (
            (spec and spec.is_local)
            or (api_base and ("127.0.0.1" in api_base or "localhost" in api_base))
        )
        keep_alive = 90.0 if _is_local_endpoint else 0.0

        return OpenAICompatProvider(
            api_key=primary_key,
            api_keys=keys if len(keys) > 1 else None,
            api_base=api_base,
            default_model=model,
            extra_headers=merged_headers or None,
            rate_limit=p.rate_limit if p else 0,
            timeout=p.timeout if p else 60.0,
            spec=spec,
            token_provider=token_provider,
            keep_alive_interval=keep_alive,
        )


def _make_provider(config: Any) -> Any:
    """Create the LLM provider from config (extracted from CLI)."""
    from nanobot.providers.base import GenerationSettings

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    if not provider_name:
        raise ValueError("No provider could be matched for the configured model.")

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

    providers: list[tuple[Any, str]] = [(primary, model)]
    for entry in fallback_entries:
        fb_provider = _make_single_provider(config, entry.provider, entry.model)
        providers.append((fb_provider, entry.model))

    fallback = FallbackProvider(providers)
    fallback.generation = gen
    return fallback


def _make_crawler_provider(config: Any) -> Any | None:
    """Create the optional crawler provider independently from the main agent."""
    crawler = config.agents.crawler
    from nanobot.agent.tools.social_crawl import crawl_tools_enabled

    if not crawler.enabled or not crawl_tools_enabled():
        return None
    model = crawler.model or config.agents.defaults.model
    provider_name = crawler.provider or config.get_provider_name(model)
    if not provider_name:
        raise ValueError("No provider could be matched for the configured crawler model.")
    provider = _make_single_provider(config, provider_name, model)

    from nanobot.providers.base import GenerationSettings

    provider.generation = GenerationSettings(
        temperature=0.1,
        max_tokens=crawler.max_tokens.output,
        reasoning_effort=crawler.reasoning_effort,
        context_window_tokens=crawler.max_tokens.input,
    )
    return provider
