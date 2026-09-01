"""Configuration schema using Pydantic."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings


class Base(BaseModel):
    """Base model that accepts both camelCase and snake_case keys."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

class ChannelsConfig(Base):
    """Configuration for chat channels.

    Built-in and plugin channel configs are stored as extra fields (dicts).
    Each channel parses its own config in __init__.
    Per-channel "streaming": true enables streaming output (requires send_delta impl).
    """

    model_config = ConfigDict(extra="allow")

    task_update_mode: Literal["result", "verbose"] = "verbose"
    send_tool_hints: bool = False  # stream tool-call hints (e.g. read_file("…"))
    send_max_retries: int = Field(default=3, ge=0, le=10)  # Max delivery attempts (initial send included)


class MaxTokensConfig(Base):
    """Token limits configuration."""

    input: int = 120000
    output: int = 4096


class FallbackEntry(Base):
    """One entry in the provider fallback chain."""

    provider: str  # Provider name, e.g. "gemini"
    model: str  # Model to use with this provider, e.g. "gemini-2.5-flash"


class ContextRepoConfig(Base):
    """Optional advanced overrides for an attached context repository."""

    path: str
    read_only: bool = False
    auto_sync: bool = True
    credential_profile: str | None = None
    manifest: str = "nanobot.context.json"
    target_repo_paths: dict[str, str] = Field(default_factory=dict)


class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.nanobot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    provider: str = (
        "auto"  # Provider name (e.g. "anthropic", "openrouter") or "auto" for auto-detection
    )
    fallback: list[FallbackEntry] = Field(default_factory=list)  # Provider fallback chain for quota exhaustion
    max_tokens: MaxTokensConfig = Field(default_factory=MaxTokensConfig)
    temperature: float = 0.1
    max_tool_iterations: int = 200
    reasoning_effort: str | None = None  # low / medium / high - enables LLM thinking mode
    timezone: str = "UTC"  # IANA timezone, e.g. "Asia/Shanghai", "America/New_York"
    context_paths: list[str] = Field(default_factory=list)  # Paths to local context repos (e.g. git repos with memory, skills, bootstrap files)
    context_repos: list[str | ContextRepoConfig] = Field(default_factory=list)  # Managed context repos; string paths auto-detect nanobot.context.json.
    tool_result_clearing_keep: int = 3  # Number of recent tool results to keep; older ones are cleared to save context
    consolidation_trigger_ratio: float = 0.5  # Trigger memory consolidation when prompt exceeds this ratio of budget
    consolidation_target_ratio: float = 0.3  # Target ratio of budget after consolidation

    @property
    def context_window_tokens(self) -> int:
        """Compatibility alias for legacy callers that still read context_window_tokens."""
        return self.max_tokens.input

    @context_window_tokens.setter
    def context_window_tokens(self, value: int) -> None:
        """Compatibility alias for legacy callers that still set context_window_tokens."""
        self.max_tokens.input = value


class CrawlerAgentConfig(Base):
    """Dedicated low-cost model used by the foreground crawler role."""

    enabled: bool = False
    provider: str | None = None
    model: str | None = None
    max_tokens: MaxTokensConfig = Field(
        default_factory=lambda: MaxTokensConfig(input=20000, output=2000)
    )
    max_tool_iterations: int = Field(default=20, ge=1, le=100)
    reasoning_effort: str | None = "low"


class AgentsConfig(Base):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)
    crawler: CrawlerAgentConfig = Field(default_factory=CrawlerAgentConfig)


class ProviderConfig(Base):
    """LLM provider configuration."""

    api_key: str = ""
    api_keys: list[str] = Field(default_factory=list)  # Multiple keys for failover/rotation
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None  # Custom headers (e.g. APP-Code for AiHubMix)
    rate_limit: int = 0  # Max requests/minute (0 = unlimited), useful for strict free tiers.
    timeout: float = 60.0  # Per-attempt LLM request/stream idle timeout in seconds.

    @property
    def effective_keys(self) -> list[str]:
        """Return configured keys with api_keys taking precedence over api_key."""
        if self.api_keys:
            return [k for k in self.api_keys if k]
        return [self.api_key] if self.api_key else []


class ProvidersConfig(Base):
    """Configuration for LLM providers."""

    custom: ProviderConfig = Field(default_factory=ProviderConfig)  # Any OpenAI-compatible endpoint
    azure_openai: ProviderConfig = Field(default_factory=ProviderConfig)  # Azure OpenAI (model = deployment name)
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    opencode_go: ProviderConfig = Field(default_factory=ProviderConfig)  # OpenCode Go subscription
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)
    vllm: ProviderConfig = Field(default_factory=ProviderConfig)
    ollama: ProviderConfig = Field(default_factory=ProviderConfig)  # Ollama local models
    ovms: ProviderConfig = Field(default_factory=ProviderConfig)  # OpenVINO Model Server (OVMS)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)
    moonshot: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax: ProviderConfig = Field(default_factory=ProviderConfig)
    mistral: ProviderConfig = Field(default_factory=ProviderConfig)
    stepfun: ProviderConfig = Field(default_factory=ProviderConfig)  # Step Fun (阶跃星辰)
    aihubmix: ProviderConfig = Field(default_factory=ProviderConfig)  # AiHubMix API gateway
    siliconflow: ProviderConfig = Field(default_factory=ProviderConfig)  # SiliconFlow (硅基流动)
    volcengine: ProviderConfig = Field(default_factory=ProviderConfig)  # VolcEngine (火山引擎)
    volcengine_coding_plan: ProviderConfig = Field(default_factory=ProviderConfig)  # VolcEngine Coding Plan
    byteplus: ProviderConfig = Field(default_factory=ProviderConfig)  # BytePlus (VolcEngine international)
    byteplus_coding_plan: ProviderConfig = Field(default_factory=ProviderConfig)  # BytePlus Coding Plan
    openai_codex: ProviderConfig = Field(default_factory=ProviderConfig, exclude=True)  # OpenAI Codex (OAuth)
    github_copilot: ProviderConfig = Field(default_factory=ProviderConfig, exclude=True)  # Github Copilot (OAuth)


class HeartbeatConfig(Base):
    """Heartbeat service configuration."""

    enabled: bool = True
    interval_s: int = 30 * 60  # 30 minutes
    keep_recent_messages: int = 8


class ApiConfig(Base):
    """OpenAI-compatible API server configuration."""

    host: str = "127.0.0.1"  # Safer default: local-only bind.
    port: int = 8900
    timeout: float = 120.0  # Per-request timeout in seconds.


class GatewayConfig(Base):
    """Gateway/server configuration."""

    host: str = "0.0.0.0"
    port: int = 18790
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)


class WebSearchConfig(Base):
    """Web search tool configuration."""

    provider: str = "brave"  # brave, tavily, duckduckgo, searxng, jina
    api_key: str = ""
    base_url: str = ""  # SearXNG base URL
    max_results: int = 5


class WebToolsConfig(Base):
    """Web tools configuration."""

    proxy: str | None = (
        None  # HTTP/SOCKS5 proxy URL, e.g. "http://127.0.0.1:7890" or "socks5://127.0.0.1:1080"
    )
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class AgentBrowserConfig(Base):
    """Agent Browser tool configuration."""

    enabled: bool = True
    package: str = "agent-browser"
    timeout: int = 180
    max_output_chars: int = 12000


class AgentDeviceConfig(Base):
    """Agent Device tool configuration."""

    enabled: bool = True
    package: str = "agent-device"
    timeout: int = 180
    max_output_chars: int = 12000


class DesktopUseConfig(Base):
    """Desktop-use tool configuration (pure-Python, stateless).

    Stateless desktop driver: no daemon, no AX tree. Each call is a
    one-shot OS-level event (CGEvent on macOS). The model works from
    screenshots + pixel coordinates, like Anthropic's ``computer-use-demo``
    and OpenAI's Operator agent.

    Disabled by default. macOS requires the PyObjC frameworks
    (``pip install 'nanobot-ai[desktop]'``) plus Accessibility and Screen
    Recording permissions in System Settings > Privacy & Security.
    Windows / Linux backends are not yet implemented.
    """

    enabled: bool = False
    screenshot_delay: float = 0.5  # seconds to wait before capturing (let UI settle)
    typing_delay_ms: int = 12  # per-character delay during `type`
    scaling_enabled: bool = True  # downscale coords to XGA/WXGA/FWXGA
    max_output_chars: int = 12000
    humanize_typing: bool = True  # jittered human-like inter-key delays during `type`


class ExecToolConfig(Base):
    """Shell exec tool configuration."""

    enable: bool = True
    timeout: int = 60
    path_append: str = ""

class MCPServerConfig(Base):
    """MCP server connection configuration (stdio or HTTP)."""

    type: Literal["stdio", "sse", "streamableHttp"] | None = None  # auto-detected if omitted
    command: str = ""  # Stdio: command to run (e.g. "npx")
    args: list[str] = Field(default_factory=list)  # Stdio: command arguments
    env: dict[str, str] = Field(default_factory=dict)  # Stdio: extra env vars
    url: str = ""  # HTTP/SSE: endpoint URL
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP/SSE: custom headers
    tool_timeout: int = 30  # seconds before a tool call is cancelled
    enabled_tools: list[str] = Field(default_factory=lambda: ["*"])  # Only register these tools; accepts raw MCP names or wrapped mcp_<server>_<tool> names; ["*"] = all tools; [] = no tools

class ImageConfig(Base):
    """Image generation configuration."""
    provider: str = "codex_cli"
    model: str = "bytedance-seed/seedream-4.5"
    api_key: str = ""
    api_base: str = "https://openrouter.ai/api/v1"
    codex_command: str = "codex"
    codex_model: str = "gpt-5.4-mini"
    codex_timeout: int = 300
    codex_home: str = ""


class ToolsConfig(Base):
    """Tools configuration."""

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    agent_browser: AgentBrowserConfig = Field(default_factory=AgentBrowserConfig)
    agent_device: AgentDeviceConfig = Field(default_factory=AgentDeviceConfig)
    desktop_use: DesktopUseConfig = Field(default_factory=DesktopUseConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    restrict_to_workspace: bool = False  # If true, restrict all tool access to workspace directory
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class Config(BaseSettings):
    """Root configuration for nanobot."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @staticmethod
    def _migrate_config(data: dict) -> dict:
        """Migrate old config formats to current."""
        channels = data.get("channels", {})
        if (
            isinstance(channels, dict)
            and "taskUpdateMode" not in channels
            and "task_update_mode" not in channels
            and "sendProgress" in channels
        ):
            channels["taskUpdateMode"] = "verbose" if channels.pop("sendProgress") else "result"

        # Migrate legacy token fields into maxTokens { input, output }.
        agents = data.get("agents", {})
        defaults = agents.get("defaults", {})
        mt = defaults.get("maxTokens")
        legacy_input = defaults.get("maxInputTokens")
        if legacy_input is None:
            legacy_input = defaults.get("contextWindowTokens")

        # Migrate legacy context_path into context_paths
        if "context_path" in defaults:
            context_path = defaults.pop("context_path")
            if context_path and isinstance(context_path, str):
                defaults.setdefault("context_paths", []).append(context_path)

        if isinstance(mt, int):
            defaults["maxTokens"] = {
                "input": legacy_input if isinstance(legacy_input, int) else 120000,
                "output": mt,
            }
        elif isinstance(mt, dict):
            if "input" not in mt and isinstance(legacy_input, int):
                mt["input"] = legacy_input
            mt.setdefault("input", 120000)
            mt.setdefault("output", 4096)
        elif isinstance(legacy_input, int):
            defaults["maxTokens"] = {"input": legacy_input, "output": 4096}

        defaults.pop("maxInputTokens", None)
        defaults.pop("contextWindowTokens", None)

        # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
        tools = data.get("tools", {})
        exec_cfg = tools.get("exec", {})
        if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
            tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")
        return data

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    def _match_provider(
        self, model: str | None = None
    ) -> tuple["ProviderConfig | None", str | None]:
        """Match provider config and its registry name. Returns (config, spec_name)."""
        from nanobot.providers.registry import PROVIDERS, find_by_name

        forced = self.agents.defaults.provider
        if forced != "auto":
            spec = find_by_name(forced)
            if spec:
                p = getattr(self.providers, spec.name, None)
                return (p, spec.name) if p else (None, None)
            return None, None

        model_lower = (model or self.agents.defaults.model).lower()
        model_normalized = model_lower.replace("-", "_")
        model_prefix = model_lower.split("/", 1)[0] if "/" in model_lower else ""
        normalized_prefix = model_prefix.replace("-", "_")

        def _kw_matches(kw: str) -> bool:
            kw = kw.lower()
            return kw in model_lower or kw.replace("-", "_") in model_normalized

        # Explicit provider prefix wins — prevents `github-copilot/...codex` matching openai_codex.
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and model_prefix and normalized_prefix == spec.name:
                if spec.is_oauth or spec.is_local or bool(p.effective_keys):
                    return p, spec.name

        # Match by keyword (order follows PROVIDERS registry)
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and any(_kw_matches(kw) for kw in spec.keywords):
                if spec.is_oauth or spec.is_local or bool(p.effective_keys):
                    return p, spec.name

        # Fallback: configured local providers can route models without
        # provider-specific keywords (for example plain "llama3.2" on Ollama).
        # Prefer providers whose detect_by_base_keyword matches the configured api_base
        # (e.g. Ollama's "11434" in "http://localhost:11434") over plain registry order.
        local_fallback: tuple[ProviderConfig, str] | None = None
        for spec in PROVIDERS:
            if not spec.is_local:
                continue
            p = getattr(self.providers, spec.name, None)
            if not (p and p.api_base):
                continue
            if spec.detect_by_base_keyword and spec.detect_by_base_keyword in p.api_base:
                return p, spec.name
            if local_fallback is None:
                local_fallback = (p, spec.name)
        if local_fallback:
            return local_fallback

        # Fallback: gateways first, then others (follows registry order)
        # OAuth providers are NOT valid fallbacks — they require explicit model selection
        for spec in PROVIDERS:
            if spec.is_oauth:
                continue
            p = getattr(self.providers, spec.name, None)
            if p and bool(p.effective_keys):
                return p, spec.name
        return None, None

    def get_provider(self, model: str | None = None) -> ProviderConfig | None:
        """Get matched provider config (api_key, api_base, extra_headers). Falls back to first available."""
        p, _ = self._match_provider(model)
        return p

    def get_provider_name(self, model: str | None = None) -> str | None:
        """Get the registry name of the matched provider (e.g. "deepseek", "openrouter")."""
        _, name = self._match_provider(model)
        return name

    def get_api_key(self, model: str | None = None) -> str | None:
        """Get API key for the given model. Falls back to first available key."""
        p = self.get_provider(model)
        if not p:
            return None
        keys = p.effective_keys
        return keys[0] if keys else None

    def get_api_base(self, model: str | None = None) -> str | None:
        """Get API base URL for the given model. Applies default URLs for gateway/local providers."""
        from nanobot.providers.registry import find_by_name

        p, name = self._match_provider(model)
        if p and p.api_base:
            return p.api_base
        # Only gateways get a default api_base here. Standard providers
        # resolve their base URL from the registry in the provider constructor.
        if name:
            spec = find_by_name(name)
            if spec and (spec.is_gateway or spec.is_local) and spec.default_api_base:
                return spec.default_api_base
        return None

    def get_provider_by_name(self, provider_name: str) -> tuple["ProviderConfig | None", str | None]:
        """Look up a specific provider by registry name."""
        from nanobot.providers.registry import find_by_name

        spec = find_by_name(provider_name)
        if spec:
            p = getattr(self.providers, spec.name, None)
            return (p, spec.name) if p else (None, None)
        return None, None

    model_config = ConfigDict(env_prefix="NANOBOT_", env_nested_delimiter="__")
