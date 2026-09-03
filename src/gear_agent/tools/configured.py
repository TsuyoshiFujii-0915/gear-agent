from __future__ import annotations

from pathlib import Path

from gear_agent.config import ToolConfig, WebFetchConfig, WebSearchConfig
from gear_agent.errors import gear_error
from gear_agent.tools.base import Tool
from gear_agent.tools.filesystem import FileReadTool, FileWriteTool
from gear_agent.tools.filesystem_search import GlobTool, GrepTool
from gear_agent.tools.patch import ApplyPatchTool
from gear_agent.tools.runtimes import ShellRuntime
from gear_agent.tools.shell import ShellTool
from gear_agent.tools.web_fetch import UrllibTavilyFetchTransport, WebFetchTool
from gear_agent.tools.web_search import UrllibTavilySearchTransport, WebSearchTool


def build_configured_tools(
    config: ToolConfig,
    web_search_config: WebSearchConfig | None,
    web_fetch_config: WebFetchConfig | None,
    workspace: Path,
    shell_runtime: ShellRuntime,
) -> list[Tool]:
    """Builds model-callable tools enabled in config order.

    Args:
        config: Parsed tool availability configuration.
        web_search_config: Parsed Tavily configuration when web search is enabled.
        web_fetch_config: Parsed Tavily configuration when web fetch is enabled.
        workspace: Workspace root for file and shell tools.
        shell_runtime: Runtime used by the shell tool when enabled.

    Returns:
        Enabled tool instances in deterministic schema order.
    """

    tools: list[Tool] = []
    if config.shell_tool:
        tools.append(ShellTool(workspace, shell_runtime))
    if config.file_read:
        tools.append(FileReadTool(workspace))
    if config.file_write:
        tools.append(FileWriteTool(workspace))
    if config.apply_patch:
        tools.append(ApplyPatchTool(workspace))
    if config.glob:
        tools.append(GlobTool(workspace))
    if config.grep:
        tools.append(GrepTool(workspace))
    if config.web_search:
        if web_search_config is None:
            raise gear_error(
                "tool_config_invalid",
                "web_search tool is enabled but web search config is missing.",
                "tool_config",
                True,
                {"tool": "web_search"},
            )
        tools.append(WebSearchTool(web_search_config, UrllibTavilySearchTransport()))
    if config.web_fetch:
        if web_fetch_config is None:
            raise gear_error(
                "tool_config_invalid",
                "web_fetch tool is enabled but web fetch config is missing.",
                "tool_config",
                True,
                {"tool": "web_fetch"},
            )
        tools.append(WebFetchTool(web_fetch_config, UrllibTavilyFetchTransport()))
    return tools
