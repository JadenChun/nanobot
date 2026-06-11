"""High-level programmatic interface to nanobot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanobot.agent.hook import AgentHook
from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus


@dataclass(slots=True)
class RunResult:
    """Result of a single agent run."""

    content: str
    tools_used: list[str]
    messages: list[dict[str, Any]]


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
        bus = MessageBus()
        defaults = config.agents.defaults

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=config.workspace_path,
            model=defaults.model,
            max_tokens=defaults.max_tokens,
            max_iterations=defaults.max_tool_iterations,
            planner_max_iterations=defaults.planner_max_iterations,
            planner_explore_subagent_max_iterations=defaults.planner_explore_subagent_max_iterations,
            planner_max_parallel_explore_agents=defaults.planner_max_parallel_explore_agents,
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
            planning_mode=defaults.planning_mode,
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
    ) -> RunResult:
        """Run the agent once and return the result.

        Args:
            message: The user message to process.
            session_key: Session identifier for conversation isolation.
                Different keys get independent history.
            hooks: Optional lifecycle hooks for this run.
        """
        prev = self._loop._extra_hooks
        if hooks is not None:
            self._loop._extra_hooks = list(hooks)
        try:
            response = await self._loop.process_direct(
                message, session_key=session_key,
            )
        finally:
            self._loop._extra_hooks = prev

        content = (response.content if response else None) or ""
        return RunResult(content=content, tools_used=[], messages=[])


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
