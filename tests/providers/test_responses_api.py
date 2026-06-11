"""Tests for the Responses API code path (used by GitHub Copilot reasoning models)."""

from __future__ import annotations

import json

from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import find_by_name


# ---------------------------------------------------------------------------
# Static decision: when should we route through /v1/responses?
# ---------------------------------------------------------------------------


def _make_copilot_provider() -> OpenAICompatProvider:
    spec = find_by_name("github_copilot")
    assert spec is not None
    return OpenAICompatProvider(
        api_key="x",
        default_model="gpt-5.4",
        api_base=spec.default_api_base,
        spec=spec,
    )


def _make_openai_provider() -> OpenAICompatProvider:
    spec = find_by_name("openai")
    assert spec is not None
    return OpenAICompatProvider(
        api_key="x",
        default_model="gpt-4o",
        api_base=spec.default_api_base,
        spec=spec,
    )


def test_copilot_routes_reasoning_models_through_responses_api():
    p = _make_copilot_provider()
    assert p._should_use_responses_api("gpt-5.4") is True
    assert p._should_use_responses_api("github-copilot/gpt-5.4") is True
    assert p._should_use_responses_api("o3-mini") is True


def test_copilot_keeps_classic_models_on_chat_completions():
    p = _make_copilot_provider()
    assert p._should_use_responses_api("gpt-4o") is False
    assert p._should_use_responses_api("claude-sonnet-4.6") is False


def test_non_optin_provider_never_uses_responses_api():
    # OpenAI direct also serves reasoning models on chat/completions fine
    # (no tool-call restriction), so we don't force them onto /responses.
    p = _make_openai_provider()
    assert p._should_use_responses_api("gpt-5.4") is False
    assert p._should_use_responses_api("o3-mini") is False


# ---------------------------------------------------------------------------
# Message → input translation
# ---------------------------------------------------------------------------


def test_messages_to_responses_input_collapses_system_into_instructions():
    messages = [
        {"role": "system", "content": "You are nanobot."},
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hi"},
    ]
    items, instructions = OpenAICompatProvider._messages_to_responses_input(messages)
    assert instructions == "You are nanobot.\n\nBe concise."
    assert items == [{"role": "user", "content": "hi"}]


def test_messages_to_responses_input_translates_tool_roundtrip():
    messages = [
        {"role": "user", "content": "use the tool"},
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"q": "x"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "result-data"},
    ]
    items, instructions = OpenAICompatProvider._messages_to_responses_input(messages)
    assert instructions is None
    assert items[0] == {"role": "user", "content": "use the tool"}
    # assistant text becomes an output_text content item
    assert items[1] == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "calling"}],
    }
    # tool call becomes a typed function_call item
    assert items[2] == {
        "type": "function_call",
        "call_id": "call_abc",
        "name": "lookup",
        "arguments": '{"q": "x"}',
    }
    # tool result becomes function_call_output
    assert items[3] == {
        "type": "function_call_output",
        "call_id": "call_abc",
        "output": "result-data",
    }


def test_messages_to_responses_input_serializes_dict_arguments():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": {"a": 1, "b": [2, 3]}},
                }
            ],
        }
    ]
    items, _ = OpenAICompatProvider._messages_to_responses_input(messages)
    assert items[0]["type"] == "function_call"
    assert json.loads(items[0]["arguments"]) == {"a": 1, "b": [2, 3]}


# ---------------------------------------------------------------------------
# Tools translation
# ---------------------------------------------------------------------------


def test_tools_to_responses_flattens_function_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "look stuff up",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    out = OpenAICompatProvider._tools_to_responses(tools)
    assert out == [
        {
            "type": "function",
            "name": "lookup",
            "description": "look stuff up",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_tools_to_responses_handles_empty():
    assert OpenAICompatProvider._tools_to_responses(None) is None
    assert OpenAICompatProvider._tools_to_responses([]) is None


def test_tool_choice_to_responses_named_function():
    tc = {"type": "function", "function": {"name": "lookup"}}
    assert OpenAICompatProvider._tool_choice_to_responses(tc) == {
        "type": "function",
        "name": "lookup",
    }
    assert OpenAICompatProvider._tool_choice_to_responses("auto") == "auto"
    assert OpenAICompatProvider._tool_choice_to_responses(None) is None


# ---------------------------------------------------------------------------
# Full kwargs build
# ---------------------------------------------------------------------------


def test_build_responses_kwargs_includes_reasoning_effort_and_strips_prefix():
    p = _make_copilot_provider()
    kwargs = p._build_responses_kwargs(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
        tools=[{
            "type": "function",
            "function": {"name": "f", "description": "d", "parameters": {}},
        }],
        model="github-copilot/gpt-5.4",
        max_tokens=2048,
        reasoning_effort="high",
        tool_choice="auto",
    )
    assert kwargs["model"] == "gpt-5.4"  # strip_model_prefix=True
    assert kwargs["max_output_tokens"] == 2048
    assert kwargs["reasoning"] == {"effort": "high"}
    assert kwargs["instructions"] == "sys"
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"] == [
        {"type": "function", "name": "f", "description": "d", "parameters": {}}
    ]
    assert "max_tokens" not in kwargs
    assert "max_completion_tokens" not in kwargs
    assert "temperature" not in kwargs
    assert kwargs["input"] == [{"role": "user", "content": "hi"}]


def test_build_responses_kwargs_omits_reasoning_when_not_set():
    p = _make_copilot_provider()
    kwargs = p._build_responses_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model="gpt-5.4",
        max_tokens=1000,
        reasoning_effort=None,
        tool_choice=None,
    )
    assert "reasoning" not in kwargs
    assert "tools" not in kwargs


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parse_responses_extracts_text_and_usage():
    p = _make_copilot_provider()
    response = {
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thought process"}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello world"}],
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    parsed = p._parse_responses(response)
    assert parsed.content == "hello world"
    assert parsed.reasoning_content == "thought process"
    assert parsed.finish_reason == "stop"
    assert parsed.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert parsed.tool_calls == []


def test_parse_responses_extracts_function_call():
    p = _make_copilot_provider()
    response = {
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_xyz",
                "name": "lookup",
                "arguments": '{"q": "weather"}',
            }
        ],
    }
    parsed = p._parse_responses(response)
    assert parsed.finish_reason == "tool_calls"
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.id == "call_xyz"
    assert tc.name == "lookup"
    assert tc.arguments == {"q": "weather"}


def test_parse_responses_handles_empty_arguments():
    p = _make_copilot_provider()
    response = {
        "status": "completed",
        "output": [
            {"type": "function_call", "call_id": "c1", "name": "noop", "arguments": ""}
        ],
    }
    parsed = p._parse_responses(response)
    assert parsed.tool_calls[0].arguments == {}


def test_parse_responses_marks_length_on_incomplete():
    p = _make_copilot_provider()
    response = {
        "status": "incomplete",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "partial"}]}
        ],
    }
    parsed = p._parse_responses(response)
    assert parsed.finish_reason == "length"
    assert parsed.content == "partial"
