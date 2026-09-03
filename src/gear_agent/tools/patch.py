from __future__ import annotations

from pathlib import Path
import subprocess

from gear_agent.tools.base import Tool
from gear_agent.tools.validation import required_string, tool_error


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
        return {"changed_files": _changed_files_from_patch_output(completed.stdout)}


def _changed_files_from_patch_output(output: str) -> list[str]:
    changed_files: list[str] = []
    for line in output.splitlines():
        if line.startswith("patching file "):
            changed_files.append(line.removeprefix("patching file "))
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
