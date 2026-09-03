from __future__ import annotations

from pathlib import Path

from gear_agent.tools.base import Tool
from gear_agent.tools.runtimes import ShellRuntime
from gear_agent.tools.validation import (
    required_int,
    required_string,
    resolve_workspace_path,
    tool_error,
)


class ShellTool(Tool):
    """Runs shell commands through an explicit runtime."""

    def __init__(self, workspace: Path, runtime: ShellRuntime) -> None:
        self._workspace = workspace.resolve()
        self._runtime = runtime

    @property
    def name(self) -> str:
        return "shell"

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": self.name,
            "description": (
                "Run a shell command inside the workspace sandbox. Use this for commands "
                "that inspect or modify project files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "workdir": {
                        "type": "string",
                        "description": (
                            "Workspace-relative working directory. Use '.' for the "
                            "workspace root. Absolute paths are rejected."
                        ),
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Maximum runtime in seconds before the command times out.",
                    },
                },
                "required": ["command", "workdir", "timeout_seconds"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        command = required_string(arguments, "command", self.name)
        raw_workdir = required_string(arguments, "workdir", self.name)
        timeout_seconds = required_int(arguments, "timeout_seconds", self.name)
        if command.strip() == "":
            raise tool_error("empty_command", "Command is empty.", self.name, {})
        workdir = resolve_workspace_path(self._workspace, raw_workdir, self.name)
        if not workdir.exists() or not workdir.is_dir():
            raise tool_error(
                "workdir_invalid",
                "Working directory does not exist.",
                self.name,
                {"workdir": raw_workdir},
            )
        return self._runtime.run(command, workdir, timeout_seconds)
