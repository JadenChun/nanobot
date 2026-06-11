from __future__ import annotations

import pytest

from nanobot.providers.openai_compat_provider import (
    OpenAICompatProvider,
    _is_reasoning_model,
)
from nanobot.providers.registry import find_by_name


@pytest.mark.parametrize(
    "model",
    [
        "o1",
        "o1-mini",
        "o1-preview",
        "o3",
        "o3-mini",
        "o4-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.2",
        "gpt-5.4",
        "gpt-5.3-codex",
        "github-copilot/gpt-5.4",
        "openai/o3-mini",
    ],
)
def test_reasoning_model_detected(model: str) -> None:
    assert _is_reasoning_model(model) is True


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4",
        "claude-sonnet-4.6",
        "claude-opus-4.7",
        "gemini-3-flash-preview",
        "olmo-7b",  # false-positive guard for "o" prefix
        "",
    ],
)
def test_classic_model_not_reasoning(model: str) -> None:
    assert _is_reasoning_model(model) is False


def test_build_kwargs_uses_max_completion_tokens_for_reasoning_model() -> None:
    spec = find_by_name("github_copilot")
    provider = OpenAICompatProvider(api_key="x", default_model="gpt-5.4", spec=spec)
    kwargs = provider._build_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model="github-copilot/gpt-5.4",
        max_tokens=2048,
        temperature=0.3,
        reasoning_effort=None,
        tool_choice=None,
    )
    assert kwargs["model"] == "gpt-5.4"  # strip_model_prefix
    assert kwargs["max_completion_tokens"] == 2048
    assert "max_tokens" not in kwargs
    # Reasoning models reject custom temperature
    assert "temperature" not in kwargs


def test_build_kwargs_keeps_max_tokens_for_classic_model() -> None:
    spec = find_by_name("github_copilot")
    provider = OpenAICompatProvider(api_key="x", default_model="gpt-4o", spec=spec)
    kwargs = provider._build_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model="github-copilot/gpt-4o",
        max_tokens=1024,
        temperature=0.3,
        reasoning_effort=None,
        tool_choice=None,
    )
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["max_tokens"] == 1024
    assert "max_completion_tokens" not in kwargs
    assert kwargs["temperature"] == 0.3
