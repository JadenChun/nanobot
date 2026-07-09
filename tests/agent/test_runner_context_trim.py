"""Tests for mid-loop context trimming in AgentRunner."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nanobot.agent.runner import AgentRunSpec, AgentRunner
from nanobot.agent.tools.registry import ToolRegistry


class _DummyTool:
    name = "dummy"

    def to_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "dummy",
                "description": "dummy tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }


def _make_messages(n_turns: int) -> list[dict]:
    """Build a message list with n_turns assistant+tool pairs."""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Do something."},
    ]
    for i in range(n_turns):
        msgs.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"call_{i}", "function": {"name": "dummy", "arguments": "{}"}}],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "name": "dummy",
            "content": f"Result {i}" + "x" * 200,  # ~50 tokens each
        })
    return msgs


def _make_spec(messages: list[dict], max_input_tokens: int | None = None) -> AgentRunSpec:
    registry = ToolRegistry()
    return AgentRunSpec(
        initial_messages=messages,
        tools=registry,
        model="test-model",
        max_iterations=1,
        max_input_tokens=max_input_tokens,
    )


@pytest.fixture
def runner():
    provider = MagicMock()
    # Return a very high token count to force trimming.
    provider.estimate_tokens.return_value = 10000
    return AgentRunner(provider)


def test_no_trim_when_under_budget(runner):
    msgs = _make_messages(3)
    original_len = len(msgs)
    runner._trim_context_to_budget(msgs, spec=_make_spec(msgs, max_input_tokens=50000))
    # Provider says 10000 tokens, budget is 50000 → no trimming.
    assert len(msgs) == original_len


def test_trim_drops_oldest_turns(runner):
    msgs = _make_messages(5)
    original_len = len(msgs)  # 2 (system+user) + 5*2 (assistant+tool) = 12
    assert original_len == 12

    # Set budget below estimated tokens (10000) to force trimming.
    # Each trim iteration drops 2 messages; we need multiple iterations.
    # Use a call_count-based approach: provider returns 10000 until we've
    # dropped enough, then returns a low value.
    call_count = 0

    def fake_estimate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # First call: 10000 (over budget). After dropping 1 turn: 5000 (under).
        if call_count <= 1:
            return 10000, None
        return 5000, None

    runner.provider.estimate_tokens.side_effect = lambda *a, **kw: None  # reset
    from unittest.mock import patch
    with patch("nanobot.agent.runner.estimate_prompt_tokens_chain", side_effect=fake_estimate):
        runner._trim_context_to_budget(msgs, spec=_make_spec(msgs, max_input_tokens=8000))

    # Should have dropped 1 turn (2 messages: assistant + tool).
    assert len(msgs) == original_len - 2


def test_trim_preserves_system_and_user_messages(runner):
    msgs = _make_messages(5)

    def always_over(*args, **kwargs):
        return 10000, None

    from unittest.mock import patch
    with patch("nanobot.agent.runner.estimate_prompt_tokens_chain", side_effect=always_over):
        runner._trim_context_to_budget(msgs, spec=_make_spec(msgs, max_input_tokens=1000))

    # System and user messages must survive.
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    # At least one turn should remain (turns[:-1] is dropped, last turn kept).
    assert len(msgs) >= 4  # system + user + 1 assistant + 1 tool


def test_trim_skips_when_no_max_input_tokens(runner):
    msgs = _make_messages(5)
    original_len = len(msgs)
    runner._trim_context_to_budget(msgs, spec=_make_spec(msgs, max_input_tokens=None))
    assert len(msgs) == original_len


def test_trim_skips_when_zero_max_input_tokens(runner):
    msgs = _make_messages(5)
    original_len = len(msgs)
    runner._trim_context_to_budget(msgs, spec=_make_spec(msgs, max_input_tokens=0))
    assert len(msgs) == original_len


def test_trim_handles_user_message_between_turns(runner):
    """A user message between turns should not be dropped."""
    msgs = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "First request."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c0", "function": {"name": "dummy", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c0", "name": "dummy", "content": "Result 0"},
        {"role": "user", "content": "Follow-up question."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "dummy", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "dummy", "content": "Result 1"},
    ]
    original_len = len(msgs)

    call_count = 0

    def fake_estimate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            return 10000, None
        return 5000, None

    from unittest.mock import patch
    with patch("nanobot.agent.runner.estimate_prompt_tokens_chain", side_effect=fake_estimate):
        runner._trim_context_to_budget(msgs, spec=_make_spec(msgs, max_input_tokens=8000))

    # The first turn (assistant + tool at indices 2,3) should be dropped.
    # The user message at index 4 should remain.
    assert len(msgs) == original_len - 2
    assert any(m.get("role") == "user" and "Follow-up" in m.get("content", "") for m in msgs)
