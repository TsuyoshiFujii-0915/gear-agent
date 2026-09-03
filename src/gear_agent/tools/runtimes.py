from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess

from gear_agent.errors import gear_error


class ShellRuntime(ABC):
    """Runtime used by ShellTool."""

    @abstractmethod
    def run(self, command: str, workdir: Path, timeout_seconds: int) -> dict[str, object]:
        """Runs a shell command.

        Args:
            command: Shell command.
            workdir: Working directory.
            timeout_seconds: Timeout in seconds.

        Returns:
            Structured command result.
        """


class DockerShellRuntime(ShellRuntime):
    """Runs commands in a Docker container."""

    def __init__(self, workspace: Path, image: str, network_enabled: bool) -> None:
        self._workspace = workspace.resolve()
        self._image = image
        self._network_enabled = network_enabled

    def run(self, command: str, workdir: Path, timeout_seconds: int) -> dict[str, object]:
        network_mode = "bridge" if self._network_enabled else "none"
        relative_workdir = workdir.resolve().relative_to(self._workspace)
        container_workdir = Path("/workspace") / relative_workdir
        docker_command = [
            "docker",
            "run",
            "--rm",
            "--network",
            network_mode,
            "-v",
            f"{self._workspace}:/workspace",
            "-w",
            str(container_workdir),
            self._image,
            "sh",
            "-lc",
            command,
        ]
        try:
            completed = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise gear_error(
                "docker_missing",
                "Docker executable was not found. Local shell fallback is not allowed.",
                "shell",
                True,
                {"executable": "docker"},
            ) from exc
        except subprocess.TimeoutExpired as exc:
            return {
                "exit_code": -1,
                "stdout": _timeout_output_to_text(exc.stdout),
                "stderr": _timeout_output_to_text(exc.stderr),
                "timed_out": True,
            }
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
        }


def _timeout_output_to_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    return output.decode("utf-8", errors="backslashreplace")
