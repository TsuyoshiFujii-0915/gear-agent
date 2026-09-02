from __future__ import annotations

from pathlib import Path

from gear_agent.tools.base import Tool
from gear_agent.tools.validation import required_string, resolve_workspace_path, tool_error


class FileReadTool(Tool):
    """Reads text files inside a workspace."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()

    @property
    def name(self) -> str:
        return "file_read"

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": self.name,
            "description": "Read a UTF-8 text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace-relative file path to read. Absolute paths are rejected."
                        ),
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        raw_path = required_string(arguments, "path", self.name)
        path = resolve_workspace_path(self._workspace, raw_path, self.name)
        if not path.exists():
            raise tool_error("file_missing", "File does not exist.", self.name, {"path": raw_path})
        if not path.is_file():
            raise tool_error("not_a_file", "Path is not a file.", self.name, {"path": raw_path})
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise tool_error(
                "file_not_text",
                "File is not valid UTF-8 text.",
                self.name,
                {"path": raw_path},
            ) from exc
        except OSError as exc:
            raise tool_error(
                "file_read_failed",
                "Failed to read file.",
                self.name,
                {"path": raw_path, "reason": str(exc)},
            ) from exc
        return {"path": raw_path, "content": content}


class FileWriteTool(Tool):
    """Writes text files inside a workspace."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()

    @property
    def name(self) -> str:
        return "file_write"

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": self.name,
            "description": "Overwrite a UTF-8 text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace-relative file path to write. Absolute paths are rejected."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "UTF-8 text content to write.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        raw_path = required_string(arguments, "path", self.name)
        content = required_string(arguments, "content", self.name)
        path = resolve_workspace_path(self._workspace, raw_path, self.name)
        if not path.parent.exists():
            raise tool_error(
                "parent_missing",
                "Parent directory does not exist.",
                self.name,
                {"path": raw_path},
            )
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise tool_error(
                "file_write_failed",
                "Failed to write file.",
                self.name,
                {"path": raw_path, "reason": str(exc)},
            ) from exc
        return {"path": raw_path, "bytes_written": len(content.encode("utf-8"))}
