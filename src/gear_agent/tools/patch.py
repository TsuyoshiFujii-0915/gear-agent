from __future__ import annotations

from pathlib import Path
import shlex
import subprocess

from gear_agent.tools.base import Tool
from gear_agent.tools.validation import required_string, resolve_workspace_path, tool_error


class ApplyPatchTool(Tool):
    """Applies unified patches inside a workspace."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()

    @property
    def name(self) -> str:
        return "apply_patch"

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": self.name,
            "description": "Apply a unified diff patch inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        patch = required_string(arguments, "patch", self.name)
        if patch.strip() == "":
            raise tool_error("patch_empty", "Patch is empty.", self.name, {})
        _validate_patch_targets(patch, self.name)
        command = ["patch", "-p0"]
        try:
            completed = subprocess.run(
                command,
                input=patch,
                cwd=self._workspace,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except FileNotFoundError as exc:
            raise tool_error(
                "patch_missing",
                "patch executable was not found.",
                self.name,
                {},
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise tool_error(
                "patch_timeout",
                "Patch application timed out.",
                self.name,
                {"stdout": exc.stdout or "", "stderr": exc.stderr or ""},
            ) from exc
        if completed.returncode != 0:
            raise tool_error(
                "patch_failed",
                "Patch application failed.",
                self.name,
                {"stdout": completed.stdout, "stderr": completed.stderr},
            )
        changed_files = _changed_files_from_patch_output(completed.stdout)
        scopes = {
            resolve_workspace_path(self._workspace, path, self.name)
            .parent.relative_to(self._workspace).as_posix()
            for path in changed_files
        }
        return {"changed_files": changed_files, "resolved_scope_paths": sorted(scopes)}


def _changed_files_from_patch_output(output: str) -> list[str]:
    changed_files: list[str] = []
    for line in output.splitlines():
        if line.startswith("patching file "):
            reported_path = line.removeprefix("patching file ")
            try:
                paths = shlex.split(reported_path, comments=False, posix=True)
            except ValueError as exc:
                raise tool_error(
                    "patch_output_invalid", "Cannot decode the patched file path.",
                    "apply_patch", {"path": reported_path, "reason": str(exc)},
                ) from exc
            if len(paths) != 1:
                raise tool_error(
                    "patch_output_invalid", "Expected exactly one patched file path.",
                    "apply_patch", {"path": reported_path},
                )
            changed_files.append(paths[0])
    return changed_files


def _validate_patch_targets(patch: str, tool_name: str) -> None:
    for line in patch.splitlines():
        if not (line.startswith("--- ") or line.startswith("+++ ")):
            continue
        target = line[4:].split("\t", 1)[0].strip()
        if target == "/dev/null":
            continue
        path = Path(target)
        if path.is_absolute() or ".." in path.parts:
            raise tool_error(
                "patch_target_outside_workspace",
                "Patch target is outside the workspace.",
                tool_name,
                {"target": target},
            )
