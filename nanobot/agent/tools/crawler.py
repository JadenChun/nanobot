"""Foreground crawler delegation tool."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.delegation import ForegroundAgentManager


class CrawlResearchTool(Tool):
    """Run the configured lower-cost foreground crawler role and await its findings."""

    def __init__(self, manager: "ForegroundAgentManager") -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "crawl_research"

    @property
    def description(self) -> str:
        return (
            "Run a bounded public-web browser research task with the dedicated lower-cost "
            "crawler role and wait for its findings. Provide the objective, known public "
            "URLs, evidence fields, and limits. It can inspect rendered HTML and use "
            "constrained read-only browser actions. It cannot write files, use shell commands, "
            "enter credentials, solve CAPTCHAs, perform social actions, or control VPNs. An "
            "operator-prepared authenticated profile may expose content that profile is authorized "
            "to read."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Bounded crawl objective, URLs, evidence fields, and limits.",
                },
            },
            "required": ["task"],
            "additionalProperties": False,
        }

    async def execute(self, task: str, **_: Any) -> str:
        return await self._manager.run_crawler(task=task)


__all__ = ["CrawlResearchTool"]
