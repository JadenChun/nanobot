from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import _PendingApproval, _PlanDecision, _VerificationResult
from nanobot.agent.policy import RiskyActionPolicy, ToolPolicy, ToolPolicyDecision
from nanobot.agent.runner import AgentRunResult, AgentRunSpec, AgentRunner
from nanobot.agent.subagent import _ExploreLoopGuard
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ChannelsConfig
from nanobot.context_repo import ContextRepoManager
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_loop(tmp_path, *, planning_mode: str = "agent", planner_max_iterations: int = 20):
    from nanobot.agent.loop import AgentLoop

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        planning_mode=planning_mode,
        planner_max_iterations=planner_max_iterations,
    )
    return loop, provider


async def _drain_outbound(bus: MessageBus) -> list:
    messages = []
    while True:
        try:
            messages.append(await asyncio.wait_for(bus.consume_outbound(), timeout=0.05))
        except asyncio.TimeoutError:
            return messages


def test_parse_plan_decision_accepts_richer_handoff(tmp_path):
    loop, _provider = _make_loop(tmp_path)
    payload = {
        "decision": "execute",
        "action_summary": "Trim the gap cleanly",
        "review_goal": "Verify the gap is covered without extending past audio",
        "references": [
            {
                "finding": "Video ends before audio tail, leaving a visual gap.",
                "references": ["timeline: 15.8s", "asset: grocery_reminder_source.mp4"],
            }
        ],
    }

    parsed = loop._parse_plan_decision(
        f"---PLAN---\n{json.dumps(payload)}\n---END---"
    )

    assert parsed.decision == "execute"
    assert parsed.action_summary == "Trim the gap cleanly"
    assert parsed.review_goal == "Verify the gap is covered without extending past audio"
    assert parsed.references == [
        {
            "finding": "Video ends before audio tail, leaving a visual gap.",
            "references": ["timeline: 15.8s", "asset: grocery_reminder_source.mp4"],
        }
    ]


@pytest.mark.asyncio
async def test_run_internal_planner_uses_separate_cap_and_preserves_tool_results(tmp_path):
    loop, _provider = _make_loop(tmp_path, planner_max_iterations=23)
    loop._run_agent = AsyncMock(return_value=AgentRunResult(  # type: ignore[method-assign]
        final_content="""---PLAN---
{"decision":"execute","action_summary":"Do the fix","review_goal":"Confirm the result","references":[{"finding":"Checked foo.py","references":["file: /tmp/foo.py"]}]}
---END---""",
        messages=[],
    ))

    await loop._run_internal_planner(
        [{"role": "user", "content": "fix the gap"}],
        channel="cli",
        chat_id="direct",
    )

    loop._run_agent.assert_awaited_once()  # type: ignore[attr-defined]
    kwargs = loop._run_agent.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["max_iterations"] == 23
    assert kwargs["preserve_tool_results"] is True


def test_build_planner_tools_includes_foreground_explore(tmp_path):
    loop, _provider = _make_loop(tmp_path)

    tools = loop._build_planner_tools()

    assert tools.has("explore")


def test_explore_loop_guard_stops_after_three_stale_turns():
    guard = _ExploreLoopGuard()
    repeated = [SimpleNamespace(name="web_search", arguments={"query": "same query"})]

    guard.observe(repeated)
    assert guard.stop_requested is False

    guard.observe(repeated)
    assert guard.stop_requested is False

    guard.observe(repeated)
    assert guard.stop_requested is False

    guard.observe(repeated)
    assert guard.stop_requested is True
    assert guard.stop_reason == "no_new_references"


def test_recent_legal_messages_strips_orphan_tool_prefix(tmp_path):
    loop, _provider = _make_loop(tmp_path)
    messages = [
        {"role": "user", "content": "older"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                {"id": "call_b", "type": "function", "function": {"name": "y", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "name": "x", "content": "ok"},
        {"role": "tool", "tool_call_id": "call_b", "name": "y", "content": "ok"},
        {"role": "user", "content": "latest"},
        {"role": "assistant", "content": "done"},
    ]

    trimmed = loop._recent_legal_messages(messages, max_messages=4)

    assert trimmed == [
        {"role": "user", "content": "latest"},
        {"role": "assistant", "content": "done"},
    ]


def test_summarize_history_for_verifier_ends_with_user_message(tmp_path):
    """Verifier must produce a request that ends with a user role.

    Some strict providers (Copilot's Claude proxy) reject requests ending
    with an assistant message ("This model does not support assistant
    message prefill."). The summarizer collapses prior turns into a single
    user message so the verifier always satisfies this constraint.
    """
    loop, _provider = _make_loop(tmp_path)
    messages = [
        {"role": "user", "content": "please change the active app"},
        {
            "role": "assistant",
            "content": "I'll take a screenshot first.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "desktop_use", "arguments": '{"action":"screenshot"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "desktop_use",
            "content": [
                {"type": "text", "text": "Screenshot saved."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
        {"role": "assistant", "content": "Done!"},
    ]

    summarized = loop._summarize_history_for_verifier(messages)

    assert len(summarized) == 1
    assert summarized[0]["role"] == "user"
    # The summarized message must contain the narrative.
    content = summarized[0]["content"]
    if isinstance(content, list):
        text = next((b["text"] for b in content if b.get("type") == "text"), "")
        # The most recent screenshot is preserved as an image_url block.
        assert any(b.get("type") == "image_url" for b in content)
    else:
        text = content
    assert "USER" in text
    assert "ASSISTANT" in text
    assert "desktop_use" in text
    assert "Done!" in text


def test_summarize_history_for_verifier_handles_empty_messages(tmp_path):
    loop, _provider = _make_loop(tmp_path)
    assert loop._summarize_history_for_verifier([]) == []


def test_summarize_history_for_verifier_caps_image_blocks(tmp_path):
    loop, _provider = _make_loop(tmp_path)
    messages = [
        {
            "role": "tool",
            "tool_call_id": f"c{i}",
            "name": "desktop_use",
            "content": [
                {"type": "text", "text": f"shot {i}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{i}"}},
            ],
        }
        for i in range(5)
    ]
    summarized = loop._summarize_history_for_verifier(messages, max_image_blocks=2)
    assert len(summarized) == 1
    content = summarized[0]["content"]
    assert isinstance(content, list)
    image_blocks = [b for b in content if b.get("type") == "image_url"]
    assert len(image_blocks) == 2
    # The two retained images should be the most recent.
    assert image_blocks[0]["image_url"]["url"].endswith(",3")
    assert image_blocks[1]["image_url"]["url"].endswith(",4")


@pytest.mark.asyncio
async def test_runner_tool_policy_can_stop_before_execution():
    class StopPolicy(ToolPolicy):
        async def evaluate(self, *, messages, tool_calls):
            return ToolPolicyDecision(
                action="respond",
                stop_reason="approval_required",
                response="Need approval first.",
                metadata={"summary": "dangerous action"},
            )

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="exec", arguments={"command": "git push"})],
        usage={},
    ))

    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="should not run")

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=2,
        tool_policy=StopPolicy(),
    ))

    assert result.stop_reason == "approval_required"
    assert result.final_content == (
        "Need approval first.\n\n"
        "Proposed approach before approval:\nworking"
    )
    assert result.policy_metadata == {"summary": "dangerous action"}
    tools.execute.assert_not_awaited()


def test_risky_action_policy_detects_large_overwrite(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("\n".join(f"line {i}" for i in range(300)), encoding="utf-8")

    policy = RiskyActionPolicy(workspace=tmp_path)
    reason = policy._risky_write_reason(
        "app.py",
        "\n".join(f"new {i}" for i in range(320)),
    )

    assert "overwrite a large portion" in reason


def test_risky_action_policy_skips_approval_for_writable_managed_target_repo_path(tmp_path, monkeypatch):
    context_repo = tmp_path / "pota"
    target_repo = tmp_path / "website"
    target_file = target_repo / "content" / "blog" / "draft.md"
    context_repo.mkdir()
    target_file.parent.mkdir(parents=True)
    target_file.write_text("\n".join(f"line {i}" for i in range(300)), encoding="utf-8")
    monkeypatch.setenv("POTA_WEBSITE_REPO", str(target_repo))
    (context_repo / "nanobot.context.json").write_text(json.dumps({
        "name": "pota",
        "targetRepos": {
            "pota_website": {
                "type": "website",
                "pathEnv": "POTA_WEBSITE_REPO",
                "read": ["**"],
                "write": ["content/blog/**"],
                "proposalRequired": ["src/**"],
            }
        },
    }), encoding="utf-8")
    manager = ContextRepoManager.from_config(context_repos=[str(context_repo)])

    policy = RiskyActionPolicy(workspace=tmp_path, context_manager=manager)
    reason = policy._risky_write_reason(
        str(target_file),
        "\n".join(f"new {i}" for i in range(320)),
    )

    assert reason is None


def test_risky_action_policy_skips_large_edit_approval_for_writable_managed_target_repo_path(tmp_path, monkeypatch):
    context_repo = tmp_path / "pota"
    target_repo = tmp_path / "website"
    target_file = target_repo / "content" / "blog" / "draft.md"
    context_repo.mkdir()
    target_file.parent.mkdir(parents=True)
    original = "\n".join(f"line {i}" for i in range(300))
    target_file.write_text(original, encoding="utf-8")
    monkeypatch.setenv("POTA_WEBSITE_REPO", str(target_repo))
    (context_repo / "nanobot.context.json").write_text(json.dumps({
        "name": "pota",
        "targetRepos": {
            "pota_website": {
                "type": "website",
                "pathEnv": "POTA_WEBSITE_REPO",
                "read": ["**"],
                "write": ["content/blog/**"],
                "proposalRequired": ["src/**"],
            }
        },
    }), encoding="utf-8")
    manager = ContextRepoManager.from_config(context_repos=[str(context_repo)])

    policy = RiskyActionPolicy(workspace=tmp_path, context_manager=manager)
    reason = policy._risky_edit_reason(
        str(target_file),
        original,
        "\n".join(f"new line {i}" for i in range(320)),
        False,
    )

    assert reason is None


def test_risky_action_policy_skips_git_push_approval_for_autopush_context_repo(tmp_path):
    context_repo = tmp_path / "pota"
    context_repo.mkdir()
    (context_repo / "nanobot.context.json").write_text(json.dumps({
        "name": "pota",
        "sync": {
            "autoPush": True,
        },
    }), encoding="utf-8")
    manager = ContextRepoManager.from_config(context_repos=[str(context_repo)])

    policy = RiskyActionPolicy(workspace=tmp_path, context_manager=manager)
    reason = policy._risky_exec_reason("git push", str(context_repo))

    assert reason is None

    reason_with_git_c = policy._risky_exec_reason(f'git -C "{context_repo}" push')

    assert reason_with_git_c is None


def test_risky_action_policy_keeps_git_push_approval_for_target_repo(tmp_path, monkeypatch):
    context_repo = tmp_path / "pota"
    target_repo = tmp_path / "website"
    context_repo.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("POTA_WEBSITE_REPO", str(target_repo))
    (context_repo / "nanobot.context.json").write_text(json.dumps({
        "name": "pota",
        "sync": {
            "autoPush": True,
        },
        "targetRepos": {
            "pota_website": {
                "pathEnv": "POTA_WEBSITE_REPO",
                "read": ["**"],
                "write": ["content/blog/**"],
            }
        },
    }), encoding="utf-8")
    manager = ContextRepoManager.from_config(context_repos=[str(context_repo)])

    policy = RiskyActionPolicy(workspace=tmp_path, context_manager=manager)
    reason = policy._risky_exec_reason("git push", str(target_repo))

    assert reason == "push commits to a remote repository"


def test_risky_action_policy_skips_git_reset_approval_for_autopush_context_repo(tmp_path):
    context_repo = tmp_path / "pota"
    context_repo.mkdir()
    (context_repo / "nanobot.context.json").write_text(json.dumps({
        "name": "pota",
        "sync": {
            "autoPush": True,
        },
    }), encoding="utf-8")
    manager = ContextRepoManager.from_config(context_repos=[str(context_repo)])

    policy = RiskyActionPolicy(workspace=tmp_path, context_manager=manager)
    reason = policy._risky_exec_reason("git reset --hard", str(context_repo))

    assert reason is None


def test_risky_action_policy_keeps_branch_delete_approval_in_context_repo(tmp_path):
    context_repo = tmp_path / "pota"
    context_repo.mkdir()
    (context_repo / "nanobot.context.json").write_text(json.dumps({
        "name": "pota",
        "sync": {
            "autoPush": True,
        },
    }), encoding="utf-8")
    manager = ContextRepoManager.from_config(context_repos=[str(context_repo)])

    policy = RiskyActionPolicy(workspace=tmp_path, context_manager=manager)

    assert policy._risky_exec_reason("git branch -D old-branch", str(context_repo)) == "delete a git branch"
    assert policy._risky_exec_reason("git push origin --delete old-branch", str(context_repo)) == "delete a git branch"


@pytest.mark.asyncio
async def test_process_direct_requires_approval_then_resumes(tmp_path):
    loop, provider = _make_loop(tmp_path, planning_mode="off")

    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="working",
            tool_calls=[ToolCallRequest(id="call_1", name="exec", arguments={"command": "git push origin main"})],
            usage={},
        ),
        LLMResponse(content="Push complete.", tool_calls=[], usage={}),
    ])

    first = await loop.process_direct("push the current branch")
    assert first is not None
    assert "Reply `yes` to continue" in first.content

    second = await loop.process_direct("yes")
    assert second is not None
    assert second.content == "Push complete."


@pytest.mark.asyncio
async def test_process_direct_approval_prompt_stays_visible_when_streaming(tmp_path):
    loop, provider = _make_loop(tmp_path, planning_mode="off")

    provider.chat_stream_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="I’ll update the file after approval.",
            tool_calls=[ToolCallRequest(id="call_1", name="edit_file", arguments={
                "path": "script.md",
                "old_text": "old",
                "new_text": "\n".join(f"new line {i}" for i in range(300)),
                "replace_all": False,
            })],
            usage={},
        ),
    ])

    progress_updates: list[str] = []
    stream_updates: list[str] = []

    response = await loop.process_direct(
        "update the script",
        on_progress=progress_updates.append,
        on_stream=stream_updates.append,
        on_stream_end=AsyncMock(),
    )

    assert response is not None
    assert response.metadata.get("_streamed") is not True
    assert stream_updates == []
    assert "Reply `yes` to continue" in response.content
    assert "Proposed approach before approval" in response.content


@pytest.mark.asyncio
async def test_run_main_task_retries_after_verifier_failure(tmp_path):
    loop, _provider = _make_loop(tmp_path, planning_mode="agent")

    initial_result = AgentRunResult(
        final_content="Initial answer",
        messages=[{"role": "assistant", "content": "Initial answer"}],
        tools_used=["edit_file"],
    )
    revised_result = AgentRunResult(
        final_content="Revised answer",
        messages=[{"role": "assistant", "content": "Revised answer"}],
        tools_used=["edit_file"],
    )

    loop._run_agent = AsyncMock(side_effect=[initial_result, revised_result])  # type: ignore[method-assign]
    loop._run_internal_verifier = AsyncMock(side_effect=[
        _VerificationResult(verdict="FAIL", issues=["Missing test"], feedback="Add the missing test."),
        _VerificationResult(verdict="PASS", issues=[], feedback=""),
    ])  # type: ignore[method-assign]

    planned = _PlanDecision(
        decision="execute",
        action_summary="Fix the bug safely",
        review_goal="Verify the bug is fixed and tests cover it",
        references=[
            {
                "finding": "The failing test points at the broken branch behavior.",
                "references": ["file: /tmp/test_branch.py", "function: test_branch_breakage"],
            }
        ],
    )

    result = await loop._run_main_task(
        [{"role": "user", "content": "fix the bug"}],
        task_text="fix the bug",
        channel="cli",
        chat_id="direct",
        message_id=None,
        approval_granted=False,
        planned=planned,
    )

    assert result.final_content == "Revised answer"
    assert loop._run_agent.await_count == 2  # type: ignore[attr-defined]
    assert loop._run_internal_verifier.await_count == 2  # type: ignore[attr-defined]
    loop._run_internal_verifier.assert_any_await(  # type: ignore[attr-defined]
        [{"role": "assistant", "content": "Initial answer"}],
        goal=(
            "Verify the bug is fixed and tests cover it\n\n"
            "Planner References:\n"
            "1. The failing test points at the broken branch behavior.\n"
            "   - file: /tmp/test_branch.py\n"
            "   - function: test_branch_breakage"
        ),
        channel="cli",
        chat_id="direct",
    )


@pytest.mark.asyncio
async def test_run_main_task_verifies_without_plan_object(tmp_path):
    loop, _provider = _make_loop(tmp_path, planning_mode="agent")

    initial_result = AgentRunResult(
        final_content="Done",
        messages=[{"role": "assistant", "content": "Done"}],
        tools_used=["edit_file"],
    )

    loop._run_agent = AsyncMock(return_value=initial_result)  # type: ignore[method-assign]
    loop._run_internal_verifier = AsyncMock(return_value=  # type: ignore[method-assign]
        _VerificationResult(verdict="PASS", issues=[], feedback="")
    )

    result = await loop._run_main_task(
        [{"role": "user", "content": "ok"}],
        task_text="ok",
        channel="cli",
        chat_id="direct",
        message_id=None,
        approval_granted=False,
        planned=None,
    )

    assert result.final_content == "Done"
    loop._run_internal_verifier.assert_awaited_once_with(  # type: ignore[attr-defined]
        [{"role": "assistant", "content": "Done"}],
        goal="ok",
        channel="cli",
        chat_id="direct",
    )


@pytest.mark.asyncio
async def test_process_direct_injects_richer_planner_handoff_into_action(tmp_path):
    loop, _provider = _make_loop(tmp_path, planning_mode="agent")

    loop._run_internal_planner = AsyncMock(return_value=  # type: ignore[method-assign]
        _PlanDecision(
            decision="execute",
            action_summary="Create a still frame hold to cover the gap",
            review_goal="Check the hold fills the visual gap without overshooting audio",
            references=[
                {
                    "finding": "Video ends before narration, leaving a gap.",
                    "references": ["timeline: 15.8s", "asset: grocery_reminder_source.mp4"],
                }
            ],
        )
    )
    loop._run_main_task = AsyncMock(return_value=AgentRunResult(  # type: ignore[method-assign]
        final_content="Done",
        messages=[{"role": "assistant", "content": "Done"}],
    ))

    response = await loop.process_direct("fix the end gap in the video")

    assert response is not None
    assert response.content == "Done"
    initial_messages = loop._run_main_task.await_args.args[0]  # type: ignore[attr-defined]
    handoff_messages = [
        message for message in initial_messages
        if message.get("role") == "system" and "Internal planner handoff." in message.get("content", "")
    ]
    assert len(handoff_messages) == 1
    assert "Action Summary:" in handoff_messages[0]["content"]
    assert "Review Goal:" in handoff_messages[0]["content"]
    assert "References:" in handoff_messages[0]["content"]


@pytest.mark.asyncio
async def test_plan_result_emits_one_user_facing_plan_message_and_does_not_persist_it(tmp_path):
    loop, _provider = _make_loop(tmp_path, planning_mode="agent")
    loop.channels_config = ChannelsConfig(task_update_mode="plan_result")
    loop._should_plan_with_mode = MagicMock(return_value=True)  # type: ignore[method-assign]

    loop._run_internal_planner = AsyncMock(return_value=  # type: ignore[method-assign]
        _PlanDecision(
            decision="execute",
            action_summary="Check the current draft count and verify the pipeline totals.",
            review_goal="Confirm the final totals are correct.",
            references=[],
        )
    )
    loop._run_main_task = AsyncMock(return_value=AgentRunResult(  # type: ignore[method-assign]
        final_content="There are 3 drafted articles now.",
        messages=[{"role": "assistant", "content": "There are 3 drafted articles now."}],
    ))

    response = await loop.process_direct(
        "how much draft article do we have now?",
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
    )

    outbound = await _drain_outbound(loop.bus)

    assert response is not None
    assert response.content == "There are 3 drafted articles now."
    assert [msg.content for msg in outbound] == [
        "Plan: Check the current draft count and verify the pipeline totals."
    ]
    assert outbound[0].metadata.get("_intermediate") is True

    session = loop.sessions.get_or_create("telegram:123")
    assert all(
        message.get("content") != "Plan: Check the current draft count and verify the pipeline totals."
        for message in session.messages
    )


@pytest.mark.asyncio
async def test_plan_result_skips_plan_message_when_planning_is_skipped(tmp_path):
    loop, _provider = _make_loop(tmp_path, planning_mode="off")
    loop.channels_config = ChannelsConfig(task_update_mode="plan_result")
    loop._run_main_task = AsyncMock(return_value=AgentRunResult(  # type: ignore[method-assign]
        final_content="Done",
        messages=[{"role": "assistant", "content": "Done"}],
    ))

    response = await loop.process_direct(
        "hello",
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
    )

    outbound = await _drain_outbound(loop.bus)

    assert response is not None
    assert response.content == "Done"
    assert outbound == []


@pytest.mark.asyncio
async def test_plan_result_does_not_emit_second_plan_message_after_approval_resume(tmp_path):
    loop, _provider = _make_loop(tmp_path, planning_mode="agent")
    loop.channels_config = ChannelsConfig(task_update_mode="plan_result")
    loop._should_plan_with_mode = MagicMock(return_value=True)  # type: ignore[method-assign]
    loop._pending_approvals["telegram:123"] = _PendingApproval(
        summary="push commits to a remote repository",
        created_at=0.0,
        session_key="telegram:123",
    )

    loop._run_internal_planner = AsyncMock(return_value=  # type: ignore[method-assign]
        _PlanDecision(
            decision="execute",
            action_summary="Push the cleanup commit and confirm the remote state.",
            review_goal="Confirm the remote no longer contains the draft batch.",
            references=[],
        )
    )
    loop._run_main_task = AsyncMock(return_value=AgentRunResult(  # type: ignore[method-assign]
        final_content="Push complete.",
        messages=[{"role": "assistant", "content": "Push complete."}],
    ))

    response = await loop.process_direct(
        "yes",
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
    )

    outbound = await _drain_outbound(loop.bus)

    assert response is not None
    assert response.content == "Push complete."
    assert outbound == []


@pytest.mark.asyncio
async def test_approval_reply_to_chat_resumes_cron_session(tmp_path):
    loop, _provider = _make_loop(tmp_path, planning_mode="off")

    cron_session = loop.sessions.get_or_create("cron:job-1")
    cron_session.messages.append({
        "role": "user",
        "content": "[Scheduled Task] Timer finished.\n\nScheduled instruction: publish the report",
    })
    loop.sessions.save(cron_session)

    loop._store_pending_approval(
        session_key="cron:job-1",
        channel="telegram",
        chat_id="123",
        summary="publish the report",
    )
    loop._run_main_task = AsyncMock(return_value=AgentRunResult(  # type: ignore[method-assign]
        final_content="Report published.",
        messages=[
            {"role": "user", "content": "yes"},
            {"role": "assistant", "content": "Report published."},
        ],
    ))

    response = await loop.process_direct(
        "yes",
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
    )

    assert response is not None
    assert response.content == "Report published."
    assert "cron:job-1" not in loop._pending_approvals
    assert "telegram:123" not in loop._pending_approvals
    call = loop._run_main_task.await_args  # type: ignore[attr-defined]
    assert call.kwargs["session_key"] == "cron:job-1"
    assert call.kwargs["approval_granted"] is True
    initial_messages = call.args[0]
    assert any(
        "Scheduled instruction: publish the report" in str(message.get("content"))
        for message in initial_messages
    )
    visible_session = loop.sessions.get_or_create("telegram:123")
    assert any(
        "[Scheduled task approval from cron:job-1]" in str(message.get("content"))
        for message in visible_session.messages
    )
    assert visible_session.messages[-1]["content"] == "Report published."


@pytest.mark.asyncio
async def test_cron_response_is_mirrored_to_visible_chat_context(tmp_path):
    loop, _provider = _make_loop(tmp_path, planning_mode="off")

    loop._run_main_task = AsyncMock(return_value=AgentRunResult(  # type: ignore[method-assign]
        final_content="The report is ready.",
        messages=[{"role": "assistant", "content": "The report is ready."}],
    ))

    response = await loop.process_direct(
        "[Scheduled Task] Timer finished.\n\nScheduled instruction: prepare the report",
        session_key="cron:job-2",
        channel="telegram",
        chat_id="123",
    )

    assert response is not None
    assert response.content == "The report is ready."
    visible_session = loop.sessions.get_or_create("telegram:123")
    assert visible_session.messages[-2]["role"] == "user"
    assert "[Scheduled task from cron:job-2]" in visible_session.messages[-2]["content"]
    assert "Scheduled instruction: prepare the report" in visible_session.messages[-2]["content"]
    assert visible_session.messages[-1] == {
        "role": "assistant",
        "content": "The report is ready.",
        "timestamp": visible_session.messages[-1]["timestamp"],
    }


@pytest.mark.asyncio
async def test_subagent_completion_from_cron_is_mirrored_to_visible_chat(tmp_path):
    loop, _provider = _make_loop(tmp_path, planning_mode="off")

    await loop._record_subagent_completion({
        "task_id": "sub-1",
        "label": "Prepare report",
        "task": "prepare the report",
        "result": "The report is ready.",
        "origin": {"channel": "telegram", "chat_id": "123"},
        "status": "ok",
        "notify": False,
        "session_key": "cron:job-2",
        "review_meta": None,
    })

    cron_session = loop.sessions.get_or_create("cron:job-2")
    visible_session = loop.sessions.get_or_create("telegram:123")

    assert "[Background subagent completed]" in cron_session.messages[-1]["content"]
    assert "Notify user: no" in cron_session.messages[-1]["content"]
    assert "Result:\nThe report is ready." in cron_session.messages[-1]["content"]
    assert visible_session.messages[-1]["content"] == cron_session.messages[-1]["content"]


def test_cancellation_ledger_records_stopped_background_work(tmp_path):
    loop, _provider = _make_loop(tmp_path, planning_mode="off")

    loop.record_task_cancellation(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        active_tasks=1,
        subagent_tasks=2,
    )

    session = loop.sessions.get_or_create("telegram:123")
    assert "[Background task cancelled]" in session.messages[-1]["content"]
    assert "Stopped tasks: 3" in session.messages[-1]["content"]
    assert "Background subagent tasks: 2" in session.messages[-1]["content"]


def test_cron_failure_ledger_is_mirrored_to_visible_chat(tmp_path):
    loop, _provider = _make_loop(tmp_path, planning_mode="off")

    loop.record_task_failure(
        session_key="cron:job-3",
        channel="telegram",
        chat_id="123",
        label="Scheduled task 'Daily report'",
        task="prepare the report",
        error="network timeout",
    )

    cron_session = loop.sessions.get_or_create("cron:job-3")
    visible_session = loop.sessions.get_or_create("telegram:123")
    assert "[Background task failed]" in cron_session.messages[-1]["content"]
    assert "Label: Scheduled task 'Daily report'" in cron_session.messages[-1]["content"]
    assert "Error:\nnetwork timeout" in cron_session.messages[-1]["content"]
    assert visible_session.messages[-1]["content"] == cron_session.messages[-1]["content"]
