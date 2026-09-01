"""Tests for the Nanobot programmatic facade."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.turn import TurnRequest, TurnResult, TurnSource
from nanobot.nanobot import Nanobot, RunResult


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    data = {
        "providers": {"openrouter": {"apiKey": "sk-test-key"}},
        "agents": {"defaults": {"model": "openai/gpt-4.1"}},
    }
    if overrides:
        data.update(overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data))
    return config_path


def test_from_config_missing_file():
    with pytest.raises(FileNotFoundError):
        Nanobot.from_config("/nonexistent/config.json")


def test_from_config_creates_instance(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    assert bot._loop is not None
    assert bot._loop.workspace == tmp_path


def test_from_config_default_path():
    from nanobot.config.schema import Config

    with patch("nanobot.config.loader.load_config") as mock_load, \
         patch("nanobot.nanobot._make_provider") as mock_prov:
        mock_load.return_value = Config()
        mock_prov.return_value = MagicMock()
        mock_prov.return_value.get_default_model.return_value = "test"
        mock_prov.return_value.generation.max_tokens = 4096
        Nanobot.from_config()
        mock_load.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_run_returns_result(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    mock_response = TurnResult(
        run_id="run-1",
        content="Hello back!",
        final_content="Hello back!",
        tools_used=["read_file"],
        stop_reason="completed",
    )
    bot._loop.execute_turn = AsyncMock(return_value=mock_response)

    result = await bot.run("hi")

    assert isinstance(result, RunResult)
    assert result.content == "Hello back!"
    assert result.tools_used == ["read_file"]
    assert result.run_id == "run-1"
    assert result.stop_reason == "completed"
    assert result.messages == []
    bot._loop.execute_turn.assert_awaited_once()
    request = bot._loop.execute_turn.await_args.args[0]
    assert isinstance(request, TurnRequest)
    assert request.content == "hi"
    assert request.source is TurnSource.SDK
    assert request.session_key == "sdk:default"
    assert request.hooks == ()


@pytest.mark.asyncio
async def test_run_with_hooks(tmp_path):
    from nanobot.agent.hook import AgentHook, AgentHookContext
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    class TestHook(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            pass

    mock_response = TurnResult(content="done", final_content="done", run_id="run-hooks")
    bot._loop.execute_turn = AsyncMock(return_value=mock_response)

    result = await bot.run("hi", hooks=[TestHook()])

    assert result.content == "done"
    request = bot._loop.execute_turn.await_args.args[0]
    assert request.hooks and isinstance(request.hooks[0], TestHook)


@pytest.mark.asyncio
async def test_run_hooks_restored_on_error(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    from nanobot.agent.hook import AgentHook

    bot._loop.execute_turn = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        await bot.run("hi", hooks=[AgentHook()])

    assert not hasattr(bot._loop, "_sdk_hooks")


@pytest.mark.asyncio
async def test_run_none_response(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    bot._loop.execute_turn = AsyncMock(return_value=None)

    result = await bot.run("hi")
    assert result.content == ""


def test_workspace_override(tmp_path):
    config_path = _write_config(tmp_path)
    custom_ws = tmp_path / "custom_workspace"
    custom_ws.mkdir()

    bot = Nanobot.from_config(config_path, workspace=custom_ws)
    assert bot._loop.workspace == custom_ws


@pytest.mark.asyncio
async def test_run_custom_session_key(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    mock_response = TurnResult(content="ok", final_content="ok", run_id="run-alice")
    bot._loop.execute_turn = AsyncMock(return_value=mock_response)

    await bot.run("hi", session_key="user-alice")
    bot._loop.execute_turn.assert_awaited_once()
    request = bot._loop.execute_turn.await_args.args[0]
    assert request.session_key == "user-alice"


@pytest.mark.asyncio
async def test_run_hooks_are_isolated_for_concurrent_calls(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    seen: list[tuple[str, tuple[object, ...]]] = []

    async def execute(request: TurnRequest) -> TurnResult:
        seen.append((request.content, request.hooks))
        await asyncio.sleep(0)
        return TurnResult(content=request.content, final_content=request.content)

    bot._loop.execute_turn = execute
    first_hook = object()
    second_hook = object()

    await asyncio.gather(
        bot.run("first", hooks=[first_hook]),
        bot.run("second", hooks=[second_hook]),
    )

    assert ("first", (first_hook,)) in seen
    assert ("second", (second_hook,)) in seen


@pytest.mark.asyncio
async def test_run_messages_are_opt_in_and_sanitized_detached_trace(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    raw_messages = [
        {"role": "system", "content": "internal system prompt"},
        {
            "role": "assistant",
            "content": "I inspected the file",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"secret.txt"}'},
            }],
            "_run_id": "run-1",
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "read_file",
            "content": "x" * 20_000,
        },
    ]
    bot._loop.execute_turn = AsyncMock(return_value=TurnResult(
        run_id="run-1",
        content="done",
        final_content="done",
        messages=raw_messages,
    ))

    default = await bot.run("inspect")
    opted_in = await bot.run("inspect", include_messages=True)

    assert default.messages == []
    assert opted_in.messages is not raw_messages
    assert all(message.get("role") != "system" for message in opted_in.messages)
    assert all("_run_id" not in message for message in opted_in.messages)
    assert all("arguments" not in repr(message) for message in opted_in.messages)
    assert len(next(message for message in opted_in.messages if message["role"] == "tool")["content"]) <= 4_003
    assert raw_messages[1]["tool_calls"][0]["function"]["arguments"]


def test_import_from_top_level():
    from nanobot import Nanobot as TopLevelNanobot
    from nanobot import RunResult as TopLevelRunResult
    assert TopLevelNanobot is Nanobot
    assert TopLevelRunResult is RunResult
