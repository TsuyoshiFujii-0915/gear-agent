from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import subprocess
import unittest
from unittest.mock import patch

from tests import test_headless as support


class ArtifactGitTests(unittest.TestCase):
    setUp = support.HeadlessCliTests.setUp
    configure = support.HeadlessCliTests.configure
    invoke = support.HeadlessCliTests.invoke

    def git(self, *arguments: str) -> str:
        return subprocess.run(['git', '-C', str(self.workspace), *arguments], check=True,
                              capture_output=True, text=True).stdout

    def run_once(self) -> dict[str, Any]:
        with support.model_endpoint([support.json_response(support.message('done'))]) as (url, requests):
            self.configure(url, False, '')
            result = self.invoke(['run', '--prompt', 'task', '--run-dir', str(self.root / 'run')])
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads((self.root / 'run/run.json').read_text())

    def test_unborn_repository_records_no_head_diff(self) -> None:
        self.git('init')
        manifest = self.run_once()
        self.assertTrue(manifest['workspace_git']['available'])
        self.assertIsNone(manifest['workspace_git']['head'])
        self.assertIsNone(manifest['workspace_git']['diff_baseline'])
        self.assertFalse((self.root / 'run/workspace.diff').exists())

    def test_detached_head_and_staged_changes_are_preserved(self) -> None:
        self.git('init')
        self.git('config', 'user.name', 'Test')
        self.git('config', 'user.email', 'test@example.invalid')
        (self.workspace / 'file').write_text('before\n')
        self.git('add', 'file')
        self.git('commit', '-m', 'initial')
        head = self.git('rev-parse', 'HEAD').strip()
        self.git('checkout', '--detach', head)
        (self.workspace / 'file').write_text('staged\n')
        self.git('add', 'file')
        staged = self.git('diff', '--cached')
        manifest = self.run_once()
        self.assertTrue(manifest['workspace_git']['detached'])
        self.assertIsNone(manifest['workspace_git']['branch'])
        self.assertTrue(manifest['workspace_git']['dirty'])
        self.assertEqual(self.git('diff', '--cached'), staged)
        self.assertIn('+staged', (self.root / 'run/workspace.diff').read_text())

    def test_no_git_binary_is_explicitly_available_as_non_git_artifact(self) -> None:
        with patch.dict("os.environ", {"PATH": ""}):
            manifest = self.run_once()
        self.assertFalse(manifest['workspace_git']['available'])
        self.assertEqual(manifest['workspace_git']['unavailable_reason'], 'git_not_installed')
        self.assertIsNone(manifest['gear']['commit'])
