from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import shutil
import subprocess

from gear_agent.errors import GearError


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """Read-only Git baseline; absent HEAD explicitly means no baseline diff."""

    available: bool
    head: str | None
    branch: str | None
    status: str | None
    reason: str | None

    def metadata(self) -> dict[str, Any]:
        """Returns manifest metadata, distinguishing dirty starts and unborn HEAD."""
        return {'available': self.available, 'head': self.head, 'branch': self.branch,
                'detached': self.branch is None if self.available else None,
                'dirty': bool(self.status) if self.available else None,
                'starting_status': self.status, 'unavailable_reason': self.reason,
                'diff_baseline': 'starting_head' if self.head is not None else None,
                'includes_preexisting_changes': bool(self.status) if self.available else None}


def git_command(workspace: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Runs read-only Git without optional index writes or external diff drivers.

    Args:
        workspace: Effective workspace directory.
        arguments: Explicit Git arguments.

    Returns:
        Captured result; callers handle only documented expected exit statuses.
    """
    environment = dict(os.environ)
    environment['GIT_OPTIONAL_LOCKS'] = '0'
    environment['LC_ALL'] = 'C'
    try:
        return subprocess.run(['git', '-C', str(workspace), *arguments], capture_output=True,
                              text=True, encoding='utf-8', errors='surrogateescape',
                              env=environment, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GearError('artifact_git_failed', 'Could not execute read-only Git inspection.',
                        'run_artifacts', True, {'command': arguments}) from exc


def require_git(workspace: Path, arguments: list[str]) -> str:
    """Requires a successful Git query; never hides unexpected Git failures."""
    result = git_command(workspace, arguments)
    if result.returncode != 0:
        raise GearError('artifact_git_failed', 'Read-only Git inspection failed.',
                        'run_artifacts', True, {'command': arguments, 'exit_code': result.returncode})
    return result.stdout


def snapshot_workspace(workspace: Path) -> WorkspaceSnapshot:
    """Captures baseline HEAD/branch/status or an explicit non-Git availability state."""
    if shutil.which('git') is None:
        return WorkspaceSnapshot(False, None, None, None, 'git_not_installed')
    probe = git_command(workspace, ['rev-parse', '--show-toplevel'])
    if probe.returncode != 0:
        if 'not a git repository' in probe.stderr:
            return WorkspaceSnapshot(False, None, None, None, 'not_a_git_repository')
        raise GearError('artifact_git_failed', 'Cannot inspect workspace Git repository.',
                        'run_artifacts', True, {'exit_code': probe.returncode})
    branch_query = git_command(workspace, ['symbolic-ref', '--quiet', '--short', 'HEAD'])
    if branch_query.returncode not in (0, 1):
        raise GearError('artifact_git_failed', 'Cannot inspect Git branch.', 'run_artifacts', True, {})
    branch = branch_query.stdout.strip() if branch_query.returncode == 0 else None
    head_query = git_command(workspace, ['rev-parse', '--verify', '--quiet', 'HEAD'])
    if head_query.returncode not in (0, 1):
        raise GearError('artifact_git_failed', 'Cannot inspect Git HEAD.', 'run_artifacts', True, {})
    head = head_query.stdout.strip() if head_query.returncode == 0 else None
    status = require_git(workspace, ['status', '--porcelain=v1', '--untracked-files=all'])
    return WorkspaceSnapshot(True, head, branch, status, None)
