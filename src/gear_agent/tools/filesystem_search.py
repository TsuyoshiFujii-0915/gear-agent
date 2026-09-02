from __future__ import annotations

from pathlib import Path
import re

from gear_agent.tools.base import Tool
from gear_agent.tools.validation import (
    is_relative_to,
    required_string,
    resolve_workspace_path,
    tool_error,
)


class GlobTool(Tool):
    """Finds filesystem paths inside a workspace with a glob pattern."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()

    @property
    def name(self) -> str:
        return "glob"

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": self.name,
            "description": (
                "Find files and directories inside the workspace using a glob pattern."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Workspace-relative glob pattern. Absolute paths and parent "
                            "directory references are rejected."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matches to return.",
                    },
                },
                "required": ["pattern", "max_results"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        pattern = required_string(arguments, "pattern", self.name)
        max_results = _required_positive_int(arguments, "max_results", self.name)
        _validate_relative_pattern(pattern, self.name)
        matches = _glob_matches(self._workspace, pattern, max_results, self.name)
        return {
            "pattern": pattern,
            "matches": matches[:max_results],
            "truncated": len(matches) > max_results,
        }


class GrepTool(Tool):
    """Searches UTF-8 text files inside a workspace with a regular expression."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()

    @property
    def name(self) -> str:
        return "grep"

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": self.name,
            "description": (
                "Search UTF-8 text files inside the workspace using a regular expression."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace-relative file or directory path to search. Absolute "
                            "paths are rejected."
                        ),
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression to search for.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of line matches to return.",
                    },
                },
                "required": ["path", "pattern", "max_results"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        raw_path = required_string(arguments, "path", self.name)
        pattern = required_string(arguments, "pattern", self.name)
        max_results = _required_positive_int(arguments, "max_results", self.name)
        if pattern == "":
            raise tool_error("pattern_empty", "Regex pattern is empty.", self.name, {})
        regex = _compile_regex(pattern, self.name)
        path = resolve_workspace_path(self._workspace, raw_path, self.name)
        files = _searchable_files(path, raw_path, self.name)
        matches = _grep_matches(self._workspace, files, regex, max_results, self.name)
        return {
            "path": raw_path,
            "pattern": pattern,
            "matches": matches[:max_results],
            "truncated": len(matches) > max_results,
        }


def _required_positive_int(
    arguments: dict[str, object],
    key: str,
    tool_name: str,
) -> int:
    value = arguments.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise tool_error(
            "argument_invalid",
            f"Required positive integer argument is missing: {key}",
            tool_name,
            {"argument": key},
        )
    return value


def _validate_relative_pattern(pattern: str, tool_name: str) -> None:
    if pattern == "":
        raise tool_error("pattern_empty", "Glob pattern is empty.", tool_name, {})
    path = Path(pattern)
    if path.is_absolute() or ".." in path.parts:
        raise tool_error(
            "path_outside_workspace",
            "Glob pattern may only reference paths inside the workspace.",
            tool_name,
            {"pattern": pattern},
        )


def _glob_matches(
    workspace: Path,
    pattern: str,
    max_results: int,
    tool_name: str,
) -> list[dict[str, object]]:
    try:
        candidates = list(workspace.glob(pattern))
    except ValueError as exc:
        raise tool_error(
            "glob_pattern_invalid",
            "Glob pattern is invalid.",
            tool_name,
            {"pattern": pattern, "reason": str(exc)},
        ) from exc
    results: list[dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda item: _relative_path(workspace, item)):
        resolved = candidate.resolve()
        relative_path = _relative_path(workspace, candidate)
        _ensure_inside_workspace(workspace, resolved, relative_path, tool_name)
        results.append(
            {
                "path": relative_path,
                "type": _path_type(candidate, tool_name),
            }
        )
        if len(results) > max_results:
            break
    return results


def _path_type(path: Path, tool_name: str) -> str:
    if path.is_file():
        return "file"
    if path.is_dir():
        return "directory"
    raise tool_error(
        "path_type_unsupported",
        "Glob match is neither a file nor a directory.",
        tool_name,
        {"path": str(path)},
    )


def _compile_regex(pattern: str, tool_name: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise tool_error(
            "regex_invalid",
            "regex pattern is invalid.",
            tool_name,
            {"pattern": pattern, "reason": str(exc)},
        ) from exc


def _searchable_files(path: Path, raw_path: str, tool_name: str) -> list[Path]:
    if not path.exists():
        raise tool_error(
            "path_missing",
            "Search path does not exist.",
            tool_name,
            {"path": raw_path},
        )
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted([candidate for candidate in path.rglob("*") if candidate.is_file()])
    raise tool_error(
        "path_type_unsupported",
        "Search path is neither a file nor a directory.",
        tool_name,
        {"path": raw_path},
    )


def _grep_matches(
    workspace: Path,
    files: list[Path],
    regex: re.Pattern[str],
    max_results: int,
    tool_name: str,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for file_path in sorted(files, key=lambda item: _relative_path(workspace, item)):
        resolved = file_path.resolve()
        relative_path = _relative_path(workspace, file_path)
        _ensure_inside_workspace(workspace, resolved, relative_path, tool_name)
        text = _read_text_file(file_path, relative_path, tool_name)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if regex.search(line) is None:
                continue
            results.append({"path": relative_path, "line": line_number, "text": line})
            if len(results) > max_results:
                return results
    return results


def _read_text_file(path: Path, relative_path: str, tool_name: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise tool_error(
            "file_not_text",
            "Search file is not valid UTF-8 text.",
            tool_name,
            {"path": relative_path},
        ) from exc
    except OSError as exc:
        raise tool_error(
            "file_read_failed",
            "Failed to read search file.",
            tool_name,
            {"path": relative_path, "reason": str(exc)},
        ) from exc


def _ensure_inside_workspace(
    workspace: Path,
    path: Path,
    raw_path: str,
    tool_name: str,
) -> None:
    if not is_relative_to(path, workspace):
        raise tool_error(
            "path_outside_workspace",
            "Search result path is outside the workspace.",
            tool_name,
            {"path": raw_path},
        )


def _relative_path(workspace: Path, path: Path) -> str:
    return path.relative_to(workspace).as_posix()
