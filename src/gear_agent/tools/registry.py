from __future__ import annotations

from gear_agent.errors import gear_error
from gear_agent.tools.base import Tool
from gear_agent.tools.web_search import WebSearchTool
from gear_agent.tools.web_fetch import WebFetchTool


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

    def web_observation(self, name: str, result: dict[str, object] | None) -> dict[str, object]:
        """Projects web provider identity and usage without query or page text.

        Args:
            name: Dispatched tool name.
            result: Completed result, or None before execution or on failure.

        Returns:
            Web-only provider and usage metadata; other tools have no additions.
        """
        tool = self._tools.get(name)
        if not isinstance(tool, (WebSearchTool, WebFetchTool)):
            return {}
        usage = None if result is None else result['provider_metadata'][tool.provider_name]['usage']
        return {'provider': tool.provider_name, 'usage': usage}
