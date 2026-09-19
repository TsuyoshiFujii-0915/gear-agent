from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch
import itertools
import json
import subprocess
import unittest

from gear_agent.cli import run_cli
from tests.test_headless import HeadlessCliTests, json_response, message, model_endpoint, tool_call


class RunArtifactTests(unittest.TestCase):
    setUp = HeadlessCliTests.setUp
    configure = HeadlessCliTests.configure
    invoke = HeadlessCliTests.invoke
    history = HeadlessCliTests.history

    def artifacts(self, destination: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        manifest = json.loads((destination / 'run.json').read_text())
        metrics = json.loads((destination / 'metrics.json').read_text())
        events = [json.loads(line) for line in (destination / 'events.jsonl').read_text().splitlines()]
        return manifest, metrics, events

    def git(self, *arguments: str) -> str:
        return subprocess.run(['git', '-C', str(self.workspace), *arguments], check=True,
                              capture_output=True, text=True).stdout

    def init_git(self) -> None:
        self.git('init')
        self.git('config', 'user.name', 'Artifact Test')
        self.git('config', 'user.email', 'artifacts@example.invalid')
        (self.workspace / 'tracked.txt').write_text('baseline\n')
        self.git('add', 'tracked.txt')
        self.git('commit', '-m', 'baseline')

    def test_complete_default_directory_and_self_contained_snapshot(self) -> None:
        prompt = self.root / 'task.txt'
        prompt.write_text('タスク\nexact input\n')
        (self.workspace / 'AGENTS.md').write_text('Repository rule\n')
        with model_endpoint([json_response(message('完成\nexact answer\n'))]) as (url, requests):
            self.configure(url, False, '')
            result = self.invoke(['run', '--prompt-file', str(prompt)])
        self.assertEqual(result.returncode, 0, result.stderr)
        destination, = (self.workspace / '.gear/runs').iterdir()
        manifest, metrics, events = self.artifacts(destination)
        self.assertEqual(manifest['schema_version'], 1)
        self.assertEqual(manifest['status'], 'success')
        self.assertNotEqual(manifest['run_id'], manifest['session_id'])
        self.assertEqual(destination.name, manifest['run_id'])
        self.assertIn(manifest['session_id'], result.stderr)
        self.assertEqual(manifest['task']['text'], prompt.read_text())
        self.assertEqual(manifest['task']['source'], 'file')
        self.assertEqual(manifest['task']['path'], str(prompt))
        self.assertEqual(manifest['model']['protocol'], 'responses')
        self.assertEqual(manifest['adapter_kind'], 'ResponsesModelAdapter')
        self.assertTrue(manifest['capabilities']['function_calling'])
        self.assertEqual(manifest['enabled_tools'], ['file_read'])
        self.assertIn('version', manifest['gear'])
        self.assertIsNone(manifest['workspace_git']['head'])
        self.assertFalse(manifest['workspace_git']['available'])
        self.assertIsNone(manifest['workspace_git']['diff_baseline'])
        self.assertEqual(manifest['repository_instructions'][0]['files'], [
            {'path': 'AGENTS.md', 'scope': '.', 'sha256': sha256(b'Repository rule\n').hexdigest()}])
        self.assertTrue(manifest['started_at'].endswith('+00:00'))
        self.assertTrue(manifest['finished_at'].endswith('+00:00'))
        self.assertEqual((destination / 'final.txt').read_text(), '完成\nexact answer\n')
        self.assertEqual(metrics['outcome']['completed_iterations'], 1)
        self.assertTrue(metrics['outcome']['final_text_produced'])
        self.assertEqual(metrics['model']['request_count'], 1)
        self.assertIsNone(metrics['model']['input_tokens'])
        self.assertIsNone(metrics['model']['output_tokens'])
        self.assertIsNone(metrics['model']['total_tokens'])
        canonical = [e for e in events if 'schema_version' not in e]
        self.assertEqual(canonical, self.history(result))
        for session in self.sessions.iterdir():
            session.unlink()
        self.assertEqual(self.artifacts(destination)[2], events)
        self.assertFalse((destination / 'workspace.diff').exists())

    def test_tool_usage_tokens_durations_and_replay(self) -> None:
        (self.workspace / 'data.txt').write_text('data')
        first = tool_call('file_read', {'path': 'data.txt'})
        first['output'].insert(0, {'type': 'reasoning', 'summary': [], 'encrypted_content': 'opaque-state'})
        first['usage'] = {'input_tokens': 11, 'output_tokens': 4, 'total_tokens': 15}
        second = tool_call('file_read', {'path': '../outside'})
        second['usage'] = {'input_tokens': 7, 'total_tokens': 12}
        third = message('done')
        third['usage'] = {'input_tokens': 8, 'output_tokens': 2, 'total_tokens': 10}
        destination = self.root / 'run'
        with model_endpoint([json_response(first), json_response(second), json_response(third)]) as (url, requests):
            self.configure(url, False, '')
            with patch('gear_agent.observation.perf_counter', side_effect=itertools.count()), \
                    redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                code = run_cli(['--config', str(self.config), 'run', '--prompt', 'read',
                                '--run-dir', str(destination)], {'HOME': str(self.root)})
        self.assertEqual(code, 0)
        manifest, metrics, events = self.artifacts(destination)
        self.assertEqual(metrics['model']['request_count'], 3)
        self.assertEqual(metrics['model']['input_tokens'], 26)
        self.assertIsNone(metrics['model']['output_tokens'])
        self.assertEqual(metrics['model']['total_tokens'], 37)
        self.assertIsNone(metrics['model']['requests'][1]['usage']['output_tokens'])
        self.assertEqual([r['duration_seconds'] for r in metrics['model']['requests']], [1.0] * 3)
        self.assertEqual(metrics['tools']['call_count'], 2)
        self.assertEqual(metrics['tools']['successful'], 1)
        self.assertEqual(metrics['tools']['failed'], 1)
        self.assertEqual(metrics['tools']['by_name']['file_read']['duration_seconds'], 2.0)
        self.assertGreater(metrics['timing']['wall_seconds'], 5)
        self.assertEqual(metrics['reasoning_replay']['dropped_disabled_items'], 1)
        self.assertNotIn('opaque-state', json.dumps(manifest) + json.dumps(metrics))
        self.assertIn('opaque-state', json.dumps(events))

    def test_failed_stream_has_terminal_record_and_no_final_artifact(self) -> None:
        fixture = Path(__file__).parent / 'fixtures/responses/premature_eof.sse'
        destination = self.root / 'failed'
        with model_endpoint([(200, 'text/event-stream', fixture.read_bytes())]) as (url, requests):
            self.configure(url, True, '')
            result = self.invoke(['run', '--prompt', 'hello', '--run-dir', str(destination)])
        self.assertEqual(result.returncode, 3, result.stderr)
        manifest, metrics, events = self.artifacts(destination)
        self.assertEqual(manifest['status'], 'failure')
        self.assertEqual(metrics['outcome']['status'], 'failure')
        self.assertEqual(metrics['outcome']['completed_iterations'], 0)
        self.assertFalse(metrics['outcome']['final_text_produced'])
        self.assertIn('stream', metrics['outcome']['error']['type'])
        self.assertEqual(metrics['model']['request_count'], 1)
        self.assertGreaterEqual(metrics['model']['requests'][0]['duration_seconds'], 0)
        self.assertFalse((destination / 'final.txt').exists())
        self.assertIn('turn_error', [e['kind'] for e in events])
        self.assertNotIn('ModelTextDelta', (destination / 'events.jsonl').read_text())
        self.assertEqual(result.stdout, '')

    def test_compaction_requests_and_context_observations_are_included(self) -> None:
        (self.workspace / 'large.txt').write_text('x' * 12000)
        responses = [tool_call('file_read', {'path': 'large.txt'}), message('summary'), message('done')]
        for response in responses:
            response['usage'] = {'total_tokens': 10}
        budget = ('[context_budget]\nauto_compaction = true\ncontext_window_tokens = 100000\n'
                  'reserved_tokens = 1000\nmax_input_tokens = 6000')
        destination = self.root / 'compacted'
        with model_endpoint([json_response(r) for r in responses]) as (url, requests):
            self.configure(url, False, budget)
            result = self.invoke(['run', '--prompt', 'read large file', '--run-dir', str(destination)])
        self.assertEqual(result.returncode, 0, result.stderr)
        _, metrics, _ = self.artifacts(destination)
        self.assertEqual(metrics['model']['request_count'], 3)
        self.assertEqual(metrics['model']['total_tokens'], 30)
        self.assertEqual(metrics['outcome']['completed_iterations'], 2)
        self.assertEqual(metrics['context']['automatic_compactions'], 1)
        self.assertEqual(metrics['context']['manual_compactions'], 0)
        self.assertEqual(metrics['context']['budget_failures'], 0)
        self.assertGreater(metrics['context']['largest_estimated_request'], 6000)
        self.assertEqual(len(metrics['context']['post_compaction_estimates']), 1)

    def test_budget_failure_before_dispatch_has_metrics(self) -> None:
        budget = ('[context_budget]\nauto_compaction = true\ncontext_window_tokens = 1000\n'
                  'reserved_tokens = 100\nmax_input_tokens = 800')
        destination = self.root / 'budget'
        with model_endpoint([]) as (url, requests):
            self.configure(url, False, budget)
            result = self.invoke(['run', '--prompt', 'long task ' * 1000, '--run-dir', str(destination)])
        self.assertEqual(result.returncode, 3, result.stderr)
        _, metrics, _ = self.artifacts(destination)
        self.assertEqual(metrics['context']['budget_failures'], 1)
        self.assertEqual(metrics['model']['request_count'], 0)
        self.assertEqual(metrics['context']['automatic_compactions'], 0)

    def test_git_baseline_diff_and_dirty_state(self) -> None:
        self.init_git()
        head = self.git('rev-parse', 'HEAD').strip()
        for dirty in (False, True):
            with self.subTest(dirty=dirty):
                (self.workspace / 'tracked.txt').write_text('preexisting\n' if dirty else 'baseline\n')
                destination = self.root / ('dirty' if dirty else 'clean')
                response = tool_call('file_write', {'path': 'tracked.txt', 'content': 'agent change\n'})
                with model_endpoint([json_response(response), json_response(message('done'))]) as (url, requests):
                    self.configure(url, False, '')
                    self.config.write_text(self.config.read_text().replace('file_write = false', 'file_write = true'))
                    result = self.invoke(['run', '--prompt', 'edit', '--run-dir', str(destination)])
                self.assertEqual(result.returncode, 0, result.stderr)
                manifest, _, _ = self.artifacts(destination)
                git = manifest['workspace_git']
                self.assertEqual(git['head'], head)
                self.assertEqual(git['dirty'], dirty)
                self.assertEqual(git['diff_baseline'], 'starting_head')
                self.assertIn('+agent change', (destination / 'workspace.diff').read_text())
                self.assertIn('-baseline', (destination / 'workspace.diff').read_text())
                self.assertIn(' M tracked.txt', (destination / 'workspace-status.txt').read_text())
                self.assertEqual(self.git('rev-parse', 'HEAD').strip(), head)
                self.assertEqual(self.git('diff', '--cached'), '')

    def test_destination_collision_does_not_dispatch_or_overwrite(self) -> None:
        destination = self.root / 'existing'
        destination.mkdir()
        (destination / 'run.json').write_text('keep me')
        with model_endpoint([]) as (url, requests):
            self.configure(url, False, '')
            result = self.invoke(['run', '--prompt', 'task', '--run-dir', str(destination)])
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('artifact_destination_exists', result.stderr)
        self.assertEqual((destination / 'run.json').read_text(), 'keep me')
        self.assertEqual(requests, [])

    def test_secrets_redacted_in_all_artifacts_and_endpoint_identity(self) -> None:
        secrets = ['model-secret-123', 'search-secret-456', 'env-secret-789', 'header-secret-abc']
        response = tool_call('file_read', {'path': 'secrets.txt'})
        (self.workspace / 'secrets.txt').write_text('\n'.join(secrets) + '\nAuthorization: Bearer header-secret-abc')
        destination = self.root / 'private'
        with model_endpoint([json_response(response), json_response(message('done'))]) as (url, requests):
            self.configure(url + '?api_key=model-secret-123', False, '')
            config = self.config.read_text().replace('api_key_env = ""', 'api_key_env = "MODEL_KEY"')
            config = config.replace('web_search = false', 'web_search = true')
            self.config.write_text(config)
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                code = run_cli(['--config', str(self.config), 'run', '--prompt', 'read', '--run-dir', str(destination)],
                               {'HOME': str(self.root), 'MODEL_KEY': secrets[0], 'TAVILY_API_KEY': secrets[1],
                                'CUSTOM_SECRET': secrets[2]})
        self.assertEqual(code, 0)
        all_artifacts = ''.join(p.read_text() for p in destination.iterdir() if p.is_file())
        for secret in secrets:
            self.assertNotIn(secret, all_artifacts)
        self.assertIn('[REDACTED]', all_artifacts)
        self.assertIn(secrets[0], ''.join(p.read_text() for p in self.sessions.iterdir()))

    def test_artifact_io_failure_is_distinct_and_never_success(self) -> None:
        destination = self.root / 'artifact'
        destination.mkdir()
        blocker = destination / 'blocked'
        blocker.write_text('not a directory')
        with model_endpoint([]) as (url, requests):
            self.configure(url, False, '')
            result = self.invoke(['run', '--prompt', 'task', '--run-dir', str(blocker / 'run')])
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('artifact_write_failed', result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertEqual(requests, [])
