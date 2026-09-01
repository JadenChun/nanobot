"""Tests for OpenAICompatProvider spec-driven behavior.

Validates that:
- OpenRouter (no strip) keeps model names intact.
- AiHubMix (strip_model_prefix=True) strips provider prefixes.
- Standard providers pass model names through as-is.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import find_by_name


def _fake_chat_response(content: str = "ok") -> SimpleNamespace:
    """Build a minimal OpenAI chat completion response."""
    message = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
    )
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage)


def _fake_tool_call_response() -> SimpleNamespace:
    """Build a minimal chat response that includes Gemini-style extra_content."""
    function = SimpleNamespace(
        name="exec",
        arguments='{"cmd":"ls"}',
        provider_specific_fields={"inner": "value"},
    )
    tool_call = SimpleNamespace(
        id="call_123",
        index=0,
        type="function",
        function=function,
        extra_content={"google": {"thought_signature": "signed-token"}},
    )
    message = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
        reasoning_content=None,
    )
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_openrouter_spec_is_gateway() -> None:
    spec = find_by_name("openrouter")
    assert spec is not None
    assert spec.is_gateway is True
    assert spec.default_api_base == "https://openrouter.ai/api/v1"


def test_openrouter_sets_default_attribution_headers() -> None:
    spec = find_by_name("openrouter")
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as MockClient:
        provider = OpenAICompatProvider(
            api_key="sk-or-test-key",
            api_base="https://openrouter.ai/api/v1",
            default_model="anthropic/claude-sonnet-4-5",
            spec=spec,
        )
        provider._build_client()

    headers = MockClient.call_args.kwargs["default_headers"]
    assert headers["HTTP-Referer"] == "https://github.com/HKUDS/nanobot"
    assert headers["X-OpenRouter-Title"] == "nanobot"
    assert headers["X-OpenRouter-Categories"] == "cli-agent,personal-agent"
    assert "x-session-affinity" in headers


def test_openrouter_user_headers_override_default_attribution() -> None:
    spec = find_by_name("openrouter")
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as MockClient:
        provider = OpenAICompatProvider(
            api_key="sk-or-test-key",
            api_base="https://openrouter.ai/api/v1",
            default_model="anthropic/claude-sonnet-4-5",
            extra_headers={
                "HTTP-Referer": "https://nanobot.ai",
                "X-OpenRouter-Title": "Nanobot Pro",
                "X-Custom-App": "enabled",
            },
            spec=spec,
        )
        provider._build_client()

    headers = MockClient.call_args.kwargs["default_headers"]
    assert headers["HTTP-Referer"] == "https://nanobot.ai"
    assert headers["X-OpenRouter-Title"] == "Nanobot Pro"
    assert headers["X-OpenRouter-Categories"] == "cli-agent,personal-agent"
    assert headers["X-Custom-App"] == "enabled"


@pytest.mark.asyncio
async def test_openrouter_keeps_model_name_intact() -> None:
    """OpenRouter gateway keeps the full model name (gateway does its own routing)."""
    mock_create = AsyncMock(return_value=_fake_chat_response())
    spec = find_by_name("openrouter")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as MockClient:
        client_instance = MockClient.return_value
        client_instance.chat.completions.create = mock_create

        provider = OpenAICompatProvider(
            api_key="sk-or-test-key",
            api_base="https://openrouter.ai/api/v1",
            default_model="anthropic/claude-sonnet-4-5",
            spec=spec,
        )
        await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="anthropic/claude-sonnet-4-5",
        )

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "anthropic/claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_aihubmix_strips_model_prefix() -> None:
    """AiHubMix strips the provider prefix (strip_model_prefix=True)."""
    mock_create = AsyncMock(return_value=_fake_chat_response())
    spec = find_by_name("aihubmix")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as MockClient:
        client_instance = MockClient.return_value
        client_instance.chat.completions.create = mock_create

        provider = OpenAICompatProvider(
            api_key="sk-aihub-test-key",
            api_base="https://aihubmix.com/v1",
            default_model="claude-sonnet-4-5",
            spec=spec,
        )
        await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="anthropic/claude-sonnet-4-5",
        )

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_opencode_go_strips_model_prefix() -> None:
    """OpenCode Go accepts opencode-go/<model-id> locally but wants the raw model ID upstream."""
    mock_create = AsyncMock(return_value=_fake_chat_response())
    spec = find_by_name("opencode-go")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as MockClient:
        client_instance = MockClient.return_value
        client_instance.chat.completions.create = mock_create

        provider = OpenAICompatProvider(
            api_key="opencode-test-key",
            api_base="https://opencode.ai/zen/go/v1",
            default_model="opencode-go/kimi-k2.6",
            spec=spec,
        )
        await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="opencode-go/kimi-k2.6",
        )

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "kimi-k2.6"


@pytest.mark.asyncio
async def test_opencode_deepseek_fills_missing_tool_reasoning_history() -> None:
    """A missing gateway field must not break the next DeepSeek tool continuation."""
    mock_create = AsyncMock(return_value=_fake_chat_response())
    spec = find_by_name("opencode-go")
    messages = [
        {"role": "user", "content": "research"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "result"},
    ]

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as MockClient:
        MockClient.return_value.chat.completions.create = mock_create
        provider = OpenAICompatProvider(
            api_key="opencode-test-key",
            api_base="https://opencode.ai/zen/go/v1",
            default_model="deepseek-v4-flash",
            spec=spec,
        )
        await provider.chat(messages=messages, model="deepseek-v4-flash")

    assistant = mock_create.call_args.kwargs["messages"][1]
    assert assistant["reasoning_content"] == ""
    assert assistant["content"] == ""
    assert "reasoning_content" not in messages[1]


@pytest.mark.asyncio
async def test_non_deepseek_does_not_invent_tool_reasoning_history() -> None:
    mock_create = AsyncMock(return_value=_fake_chat_response())
    spec = find_by_name("opencode-go")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as MockClient:
        MockClient.return_value.chat.completions.create = mock_create
        provider = OpenAICompatProvider(
            api_key="opencode-test-key",
            api_base="https://opencode.ai/zen/go/v1",
            default_model="kimi-k2.6",
            spec=spec,
        )
        await provider.chat(
            messages=[{
                "role": "assistant",
                "content": "calling",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }],
            }],
            model="kimi-k2.6",
        )

    assistant = mock_create.call_args.kwargs["messages"][0]
    assert "reasoning_content" not in assistant


@pytest.mark.asyncio
async def test_opencode_go_minimax_returns_clear_error() -> None:
    spec = find_by_name("opencode-go")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = OpenAICompatProvider(
            api_key="opencode-test-key",
            api_base="https://opencode.ai/zen/go/v1",
            default_model="opencode-go/minimax-m2.7",
            spec=spec,
        )
        result = await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="opencode-go/minimax-m2.7",
        )

    assert result.finish_reason == "error"
    assert "MiniMax models use the Anthropic-style /messages endpoint" in (result.content or "")


@pytest.mark.asyncio
async def test_standard_provider_passes_model_through() -> None:
    """Standard provider (e.g. deepseek) passes model name through as-is."""
    mock_create = AsyncMock(return_value=_fake_chat_response())
    spec = find_by_name("deepseek")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as MockClient:
        client_instance = MockClient.return_value
        client_instance.chat.completions.create = mock_create

        provider = OpenAICompatProvider(
            api_key="sk-deepseek-test-key",
            default_model="deepseek-chat",
            spec=spec,
        )
        await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="deepseek-chat",
        )

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "deepseek-chat"


@pytest.mark.asyncio
async def test_openai_compat_preserves_extra_content_on_tool_calls() -> None:
    """Gemini extra_content (thought signatures) must survive parse→serialize round-trip."""
    mock_create = AsyncMock(return_value=_fake_tool_call_response())
    spec = find_by_name("gemini")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as MockClient:
        client_instance = MockClient.return_value
        client_instance.chat.completions.create = mock_create

        provider = OpenAICompatProvider(
            api_key="test-key",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-3.1-pro-preview",
            spec=spec,
        )
        result = await provider.chat(
            messages=[{"role": "user", "content": "run exec"}],
            model="google/gemini-3.1-pro-preview",
        )

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.extra_content == {"google": {"thought_signature": "signed-token"}}
    assert tool_call.function_provider_specific_fields == {"inner": "value"}

    serialized = tool_call.to_openai_tool_call()
    assert serialized["extra_content"] == {"google": {"thought_signature": "signed-token"}}
    assert serialized["function"]["provider_specific_fields"] == {"inner": "value"}


def test_openai_model_passthrough() -> None:
    """OpenAI models pass through unchanged."""
    spec = find_by_name("openai")
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = OpenAICompatProvider(
            api_key="sk-test-key",
            default_model="gpt-4o",
            spec=spec,
        )
    assert provider.get_default_model() == "gpt-4o"


@pytest.mark.asyncio
async def test_gemini_rotates_to_next_key_on_rate_limit() -> None:
    class _RateLimitError(Exception):
        status_code = 429

    spec = find_by_name("gemini")
    first = SimpleNamespace()
    first.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=_RateLimitError("quota exceeded")),
    ))
    second = SimpleNamespace()
    second.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=_fake_chat_response("ok from backup key")),
    ))

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=[first, second]) as MockClient:
        provider = OpenAICompatProvider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        result = await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="google/gemini-2.5-pro",
        )

    assert result.content == "ok from backup key"
    assert MockClient.call_args_list[0].kwargs["api_key"] == "gem-key-1"
    assert MockClient.call_args_list[1].kwargs["api_key"] == "gem-key-2"


@pytest.mark.asyncio
async def test_gemini_rotation_updates_active_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RateLimitError(Exception):
        status_code = 429

    spec = find_by_name("gemini")
    first = SimpleNamespace()
    first.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=_RateLimitError("quota exceeded")),
    ))
    second = SimpleNamespace()
    second.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=_fake_chat_response("ok from backup key")),
    ))

    monkeypatch.setenv("GEMINI_API_KEY", "stale-key")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=[first, second]):
        provider = OpenAICompatProvider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        result = await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="google/gemini-2.5-pro",
        )

    assert result.content == "ok from backup key"
    assert os.environ["GEMINI_API_KEY"] == "gem-key-2"


@pytest.mark.asyncio
async def test_gemini_does_not_rotate_on_non_rate_limit_error() -> None:
    class _AuthError(Exception):
        status_code = 401

    spec = find_by_name("gemini")
    client = SimpleNamespace()
    client.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=_AuthError("unauthorized")),
    ))

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", return_value=client) as MockClient:
        provider = OpenAICompatProvider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        result = await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="google/gemini-2.5-pro",
        )

    assert result.finish_reason == "error"
    assert MockClient.call_count == 1


@pytest.mark.asyncio
async def test_gemini_reports_when_all_keys_are_exhausted() -> None:
    class _RateLimitError(Exception):
        status_code = 429

    spec = find_by_name("gemini")
    first = SimpleNamespace()
    first.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=_RateLimitError("quota exceeded on key 1")),
    ))
    second = SimpleNamespace()
    second.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=_RateLimitError("quota exceeded on key 2")),
    ))

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=[first, second]):
        provider = OpenAICompatProvider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        result = await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="google/gemini-2.5-pro",
        )

    assert result.finish_reason == "error"
    assert "after trying 2 keys" in (result.content or "")


@pytest.mark.asyncio
async def test_gemini_concurrent_requests_keep_key_rotation_request_local() -> None:
    class _RateLimitError(Exception):
        status_code = 429

    spec = find_by_name("gemini")
    started_on_primary = 0
    started_on_backup = False
    release_primary = asyncio.Event()

    def _client_factory(*, api_key, **kwargs):
        async def _create(**req_kwargs):
            nonlocal started_on_primary, started_on_backup
            if api_key == "gem-key-1":
                started_on_primary += 1
                if started_on_primary >= 2 or started_on_backup:
                    release_primary.set()
                await release_primary.wait()
                raise _RateLimitError("quota exceeded on key 1")
            started_on_backup = True
            release_primary.set()
            return _fake_chat_response("ok from backup key")

        client = SimpleNamespace()
        client.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))
        return client

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=_client_factory):
        provider = OpenAICompatProvider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        first, second = await asyncio.gather(
            provider.chat(
                messages=[{"role": "user", "content": "hello a"}],
                model="google/gemini-2.5-pro",
            ),
            provider.chat(
                messages=[{"role": "user", "content": "hello b"}],
                model="google/gemini-2.5-pro",
            ),
        )

    assert first.finish_reason == "stop"
    assert second.finish_reason == "stop"
    assert first.content == "ok from backup key"
    assert second.content == "ok from backup key"


@pytest.mark.asyncio
async def test_gemini_prefers_last_successful_key_on_follow_up_request() -> None:
    class _RateLimitError(Exception):
        status_code = 429

    spec = find_by_name("gemini")
    call_keys: list[str] = []

    def _client_factory(*, api_key, **kwargs):
        async def _create(**req_kwargs):
            call_keys.append(api_key)
            if call_keys == ["gem-key-1"]:
                raise _RateLimitError("quota exceeded on key 1")
            return _fake_chat_response(f"ok from {api_key}")

        client = SimpleNamespace()
        client.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))
        return client

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=_client_factory):
        provider = OpenAICompatProvider(
            api_keys=["gem-key-1", "gem-key-2", "gem-key-3"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        first = await provider.chat(
            messages=[{"role": "user", "content": "hello first"}],
            model="google/gemini-2.5-pro",
        )
        second = await provider.chat(
            messages=[{"role": "user", "content": "hello second"}],
            model="google/gemini-2.5-pro",
        )

    assert first.content == "ok from gem-key-2"
    assert second.content == "ok from gem-key-2"
    assert call_keys == ["gem-key-1", "gem-key-2", "gem-key-2"]


@pytest.mark.asyncio
async def test_gemini_stream_timeout_rotates_to_next_key() -> None:
    class _TimedOutStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.Event().wait()

    class _TextStream:
        def __init__(self, parts: list[str]) -> None:
            self._parts = iter(parts)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._parts)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    spec = find_by_name("gemini")

    first = SimpleNamespace()
    first.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=_TimedOutStream()),
    ))
    second = SimpleNamespace()
    second.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=_TextStream(["ok after stream timeout"])),
    ))

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=[first, second]):
        provider = OpenAICompatProvider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            timeout=5.0,
            spec=spec,
        )
        provider._request_timeout = 0.01
        result = await provider.chat_stream(
            messages=[{"role": "user", "content": "hello"}],
            model="google/gemini-2.5-pro",
        )

    assert result.content == "ok after stream timeout"


@pytest.mark.asyncio
async def test_gemini_rotates_on_service_unavailable_503() -> None:
    class _ServiceUnavailableError(Exception):
        status_code = 503

    spec = find_by_name("gemini")
    first = SimpleNamespace()
    first.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=_ServiceUnavailableError("high demand")),
    ))
    second = SimpleNamespace()
    second.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=_fake_chat_response("ok after 503 rotate")),
    ))

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=[first, second]):
        provider = OpenAICompatProvider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        result = await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="google/gemini-2.5-pro",
        )

    assert result.content == "ok after 503 rotate"


@pytest.mark.asyncio
async def test_single_key_rate_limit_emits_canonical_exhaustion_message() -> None:
    """A single-key pool hitting 429 must still emit the canonical exhaustion
    message so FallbackProvider can detect quota exhaustion and fall back."""

    class _RateLimitError(Exception):
        status_code = 429

    spec = find_by_name("gemini")
    client = SimpleNamespace()
    client.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=_RateLimitError("429 Too Many Requests")),
    ))

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", return_value=client):
        provider = OpenAICompatProvider(
            api_keys=["only-key"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        result = await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="google/gemini-2.5-pro",
        )

    assert result.finish_reason == "error"
    assert "All configured API keys were rate-limited" in (result.content or "")
    # The canonical message is what FallbackProvider._is_quota_exhaustion matches on.
    from nanobot.providers.base import LLMProvider
    assert LLMProvider._is_quota_exhaustion(result.content) is True


def test_local_timeout_not_classified_as_rate_limit() -> None:
    """A bare TimeoutError is no longer classified as a rate-limit error.
    It is a transient network issue that chat_stream_with_retry handles
    with its own retry logic. Key rotation cooldown still applies for
    multi-key scenarios via the separate timeout path in chat_stream."""
    spec = find_by_name("gemini")
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = OpenAICompatProvider(
            api_keys=["k1", "k2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )

    err = TimeoutError("")
    assert OpenAICompatProvider._is_rate_limit_error(err) is False
    assert provider._cooldown_seconds_for_error(err) == OpenAICompatProvider._OVERLOAD_KEY_COOLDOWN_S


@pytest.mark.asyncio
async def test_chat_stream_does_not_rotate_after_delivering_content() -> None:
    """Once content has been streamed to the user, a mid-stream failure must
    not rotate to another key and concatenate a second response."""

    class _RateLimitError(Exception):
        status_code = 429

    class _PartialThenFailStream:
        """Yields one chunk with content then raises a rate-limit error."""

        def __init__(self) -> None:
            self._yielded = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._yielded:
                self._yielded = True
                delta = SimpleNamespace(content="partial-hello")
                choice = SimpleNamespace(delta=delta, finish_reason=None, index=0)
                return SimpleNamespace(choices=[choice], usage=None)
            raise _RateLimitError("429 mid-stream")

    class _BackupStream:
        def __init__(self) -> None:
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            delta = SimpleNamespace(content="should-not-see-this")
            choice = SimpleNamespace(delta=delta, finish_reason="stop", index=0)
            usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            return SimpleNamespace(choices=[choice], usage=usage)

    spec = find_by_name("gemini")
    first = SimpleNamespace()
    first.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=_PartialThenFailStream()),
    ))
    second = SimpleNamespace()
    second_create = AsyncMock(return_value=_BackupStream())
    second.chat = SimpleNamespace(completions=SimpleNamespace(create=second_create))

    deltas: list[str] = []

    async def _on_delta(text: str) -> None:
        deltas.append(text)

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=[first, second]):
        provider = OpenAICompatProvider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        result = await provider.chat_stream(
            messages=[{"role": "user", "content": "hello"}],
            model="google/gemini-2.5-pro",
            on_content_delta=_on_delta,
        )

    # User received only the first key's partial content — no second key content.
    assert deltas == ["partial-hello"]
    assert second_create.await_count == 0
    # The response should reflect the error; it must NOT contain the backup content.
    assert "should-not-see-this" not in (result.content or "")
