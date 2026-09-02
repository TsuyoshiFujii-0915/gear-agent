from __future__ import annotations

from gear_agent.errors import gear_error
from gear_agent.tools.base import Tool


class ToolRegistry:
    """Resolves and runs tools by name."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, object]]:
        return [tool.schema() for tool in self._tools.values()]

    def run(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        tool = self._tools.get(name)
        if tool is None:
            raise gear_error(
                "tool_not_found",
                f"Unknown tool requested: {name}",
                "tool_registry",
                True,
                {"tool": name},
            )
        return tool.run(arguments)
