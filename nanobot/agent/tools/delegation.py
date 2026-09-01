"""Foreground role tools exposed to the main orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.turn import TurnContext

if TYPE_CHECKING:
    from nanobot.agent.delegation import ForegroundAgentManager


class _ForegroundTool(Tool):
    def __init__(self, manager: "ForegroundAgentManager") -> None:
        self._manager = manager

    @property
    def supports_parallel_calls(self) -> bool:
        return False


class PlanTaskTool(_ForegroundTool):
    name = "plan_task"
    description = (
        "Ask a foreground read-only planner for an implementation-ready plan when sequencing, "
        "dependencies, or ambiguity make direct work unreliable. Waits for the plan. Do not use "
        "for simple or already-clear tasks."
    )
    parameters = {
        "type": "object",
        "properties": {
            "objective": {"type": "string", "minLength": 3},
            "context": {
                "type": "string",
                "description": "Plain-text evidence already gathered; keep raw values and references.",
            },
        },
        "required": ["objective"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        objective: str,
        context: str = "",
        turn_context: TurnContext | None = None,
        **_: Any,
    ) -> str:
        return await self._manager.run_plan(
            objective=objective,
            context=context,
            turn_context=turn_context,
        )

    async def execute_with_context(self, turn_context: TurnContext, **kwargs: Any) -> Any:
        return await self.execute(turn_context=turn_context, **kwargs)


class DelegateTaskTool(_ForegroundTool):
    name = "delegate_task"
    description = (
        "Run one bounded implementation or artifact task in a foreground worker and wait for its "
        "completion report. Provide a self-contained Markdown contract and explicit writable "
        "workspace paths. Use direct tools when isolated worker context adds no value."
    )
    parameters = {
        "type": "object",
        "properties": {
            "contract": {
                "type": "string",
                "description": "Markdown task contract with objective, evidence, acceptance criteria, non-goals, and validation.",
                "minLength": 20,
            },
            "write_scope": {
                "type": "array",
                "description": "Workspace-relative files or directory prefixes ending in a slash.",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
            },
        },
        "required": ["contract", "write_scope"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        contract: str,
        write_scope: list[str],
        turn_context: TurnContext | None = None,
        **_: Any,
    ) -> str:
        return await self._manager.run_worker(
            contract=contract,
            write_scope=write_scope,
            turn_context=turn_context,
        )

    async def execute_with_context(self, context: TurnContext, **kwargs: Any) -> Any:
        return await self.execute(turn_context=context, **kwargs)


class ReviewWorkTool(_ForegroundTool):
    name = "review_work"
    description = (
        "Run a foreground independent read-only review and wait for PASS, CORRECT, or REJECT "
        "evidence. Use for important deliverables, risky changes, uncertain results, or explicit "
        "quality requests; do not review every routine task."
    )
    parameters = {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "minLength": 3},
            "acceptance_criteria": {"type": "string", "minLength": 3},
            "evidence": {"type": "string"},
            "relevant_paths": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["goal", "acceptance_criteria"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        goal: str,
        acceptance_criteria: str,
        evidence: str = "",
        relevant_paths: list[str] | None = None,
        turn_context: TurnContext | None = None,
        **_: Any,
    ) -> str:
        return await self._manager.run_review(
            goal=goal,
            acceptance_criteria=acceptance_criteria,
            evidence=evidence,
            relevant_paths=relevant_paths,
            turn_context=turn_context,
        )

    async def execute_with_context(self, context: TurnContext, **kwargs: Any) -> Any:
        return await self.execute(turn_context=context, **kwargs)


__all__ = ["DelegateTaskTool", "PlanTaskTool", "ReviewWorkTool"]
