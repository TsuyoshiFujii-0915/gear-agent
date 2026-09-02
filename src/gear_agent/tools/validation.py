from __future__ import annotations

from pathlib import Path

from gear_agent.errors import GearError, gear_error


def required_string(arguments: dict[str, object], key: str, tool_name: str) -> str:
    """Reads a required string tool argument."""

    value = arguments.get(key)
    if not isinstance(value, str):
        raise tool_error(
            "argument_invalid",
            f"Required string argument is missing: {key}",
            tool_name,
            {"argument": key},
        )
    return value


def required_int(arguments: dict[str, object], key: str, tool_name: str) -> int:
    """Reads a required integer tool argument."""

    value = arguments.get(key)
    if not isinstance(value, int):
        raise tool_error(
            "argument_invalid",
            f"Required integer argument is missing: {key}",
            tool_name,
            {"argument": key},
        )
    return value


def resolve_workspace_path(workspace: Path, raw_path: str, tool_name: str) -> Path:
    """Resolves a tool path and rejects paths outside the workspace."""

    path = (workspace / raw_path).resolve()
    if not is_relative_to(path, workspace):
        raise tool_error(
            "path_outside_workspace",
            "Path is outside the workspace.",
            tool_name,
            {"path": raw_path},
        )
    return path


def is_relative_to(path: Path, parent: Path) -> bool:
    """Returns whether path is inside parent."""

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def tool_error(
    error_type: str,
    message: str,
    origin: str,
    details: dict[str, object],
) -> GearError:
    """Builds a recoverable tool error."""

    return gear_error(error_type, message, origin, True, details)
