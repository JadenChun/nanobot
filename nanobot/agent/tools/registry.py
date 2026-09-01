"""Tool registry for dynamic tool management."""

from collections.abc import Iterator
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.turn import ToolOutcome


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    def iter_tools(self) -> Iterator[Tool]:
        """Iterate over registered tool instances."""
        return iter(self._tools.values())

    async def execute(
        self,
        name: str,
        params: dict[str, Any],
        context: Any | None = None,
        *,
        execution_context: Any | None = None,
    ) -> Any:
        """Execute a tool by name with given parameters.

        Existing callers receive the same bare result as before.  Passing an
        explicit ``context`` (or its ``execution_context`` alias) opts into a
        :class:`~nanobot.agent.turn.ToolOutcome` envelope, allowing the later
        canonical turn runner to observe stop and policy metadata without a
        blanket return-type migration.
        """
        hint = "\n\n[Analyze the error above and try a different approach.]"
        if context is None:
            context = execution_context
        contextual = context is not None

        def finish(outcome: ToolOutcome) -> Any:
            return outcome if contextual else outcome.content

        def error(message: str) -> Any:
            return finish(ToolOutcome(content=message, stop_reason="tool_error"))

        tool = self._tools.get(name)
        if not tool:
            return error(f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}")

        try:
            # Attempt to cast parameters to match schema types
            params = tool.cast_params(params)

            # Validate parameters
            errors = tool.validate_params(params)
            if errors:
                return error(
                    f"Error: Invalid parameters for tool '{name}': "
                    + "; ".join(errors)
                    + hint
                )
            if contextual:
                result = await tool.execute_with_context(context, **params)
            else:
                result = await tool.execute(**params)

            outcome = result if isinstance(result, ToolOutcome) else ToolOutcome(content=result)
            if isinstance(outcome.content, str) and outcome.content.startswith("Error"):
                content = outcome.content + hint
                outcome = ToolOutcome(
                    content=content,
                    stop_reason=outcome.stop_reason or "tool_error",
                    policy_metadata=outcome.policy_metadata,
                )
            return finish(outcome)
        except Exception as e:
            return error(f"Error executing {name}: {str(e)}" + hint)

    async def execute_with_context(
        self,
        name: str,
        params: dict[str, Any],
        context: Any,
    ) -> ToolOutcome:
        """Explicit contextual execution path returning a :class:`ToolOutcome`.

        This additive helper makes the opt-in behavior discoverable while the
        historical ``execute(name, params)`` method continues returning bare
        values for existing callers.
        """
        result = await self.execute(name, params, context=context)
        if isinstance(result, ToolOutcome):
            return result
        # ``context`` is non-None above, so this is defensive for unusual
        # registry overrides that still return a bare value.
        return ToolOutcome(content=result)

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def filtered(self, names: list[str]) -> "ToolRegistry":
        """Return a new registry containing only the named tools."""
        reg = ToolRegistry()
        for name in names:
            tool = self._tools.get(name)
            if tool is not None:
                reg.register(tool)
        return reg
