from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
import json
import unittest

from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.cli import run_cli
from gear_agent.config import load_config
from gear_agent.errors import GearError
from gear_agent.headless import TaskPrompt, run_task
from gear_agent.run_artifacts import RunArtifacts
from gear_agent.run_collector import EvalRunCollector
from gear_agent.runtime import build_agent_runtime
from tests import test_headless as support


class ArtifactEdgeTests(unittest.TestCase):
    setUp = support.HeadlessCliTests.setUp
    configure = support.HeadlessCliTests.configure
    invoke = support.HeadlessCliTests.invoke

    def read(self, name: str) -> dict[str, Any]:
        return json.loads((self.root / 'run' / name).read_text())

    def test_failed_completion_does_not_publish_success_or_final(self) -> None:
        with support.model_endpoint([support.json_response(support.message('done'))]) as (url, requests):
            self.configure(url, False, '')
            config = load_config(self.config, {})
            collector = EvalRunCollector()
            agent = build_agent_runtime(config, config.runtime, collector, collector)
            artifacts = RunArtifacts(agent, 'session', TaskPrompt('task', 'inline'),
                                     self.root / 'run', collector, {})
            (self.root / 'run' / 'metrics.json').mkdir()
            result = run_task(agent, 'session', TaskPrompt('task', 'inline'))
            with self.assertRaises(GearError) as raised:
                artifacts.finish(result, None)
        self.assertEqual(raised.exception.error_type, 'artifact_write_failed')
        self.assertEqual(self.read('run.json')['status'], 'running')
        self.assertFalse((self.root / 'run' / 'final.txt').exists())
        self.assertTrue((self.root / 'run' / 'events.jsonl').is_file())

    def test_fatal_tool_error_keeps_tool_timing_and_trace(self) -> None:
        response = support.tool_call('unknown_tool', {})
        with support.model_endpoint([support.json_response(response)]) as (url, requests):
            self.configure(url, False, '')
            result = self.invoke(['--max-iterations', '1', 'run', '--prompt', 'tool',
                                  '--run-dir', str(self.root / 'run')])
        self.assertEqual(result.returncode, 3, result.stderr)
        metrics = self.read('metrics.json')
        self.assertEqual(metrics['tools']['call_count'], 1)
        self.assertEqual(metrics['tools']['failed'], 1)
        self.assertGreaterEqual(metrics['tools']['calls'][0]['duration_seconds'], 0)

    def test_encrypted_replay_reused_only_in_trace(self) -> None:
        (self.workspace / 'data.txt').write_text('data')
        response = support.tool_call('file_read', {'path': 'data.txt'})
        response['output'].insert(0, {'type': 'reasoning', 'id': 'reason-1', 'summary': [],
                                     'encrypted_content': 'opaque-encrypted-state'})
        with support.model_endpoint([support.json_response(response), support.json_response(support.message('done'))]) as (url, requests):
            self.configure(url, False, '')
            self.config.write_text(self.config.read_text().replace('reasoning_replay = "none"',
                                                                  'reasoning_replay = "encrypted"'))
            result = self.invoke(['run', '--prompt', 'read', '--run-dir', str(self.root / 'run')])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.read('metrics.json')['reasoning_replay']['reused_encrypted_items'], 1)
        self.assertIn('opaque-encrypted-state', (self.root / 'run/events.jsonl').read_text())
        self.assertNotIn('opaque-encrypted-state', (self.root / 'run/run.json').read_text())
        self.assertNotIn('opaque-encrypted-state', (self.root / 'run/metrics.json').read_text())

    def test_unavailable_usage_is_not_derived_from_reported_categories(self) -> None:
        response = support.message('done')
        response['usage'] = {'input_tokens': 2, 'output_tokens': 3}
        with support.model_endpoint([support.json_response(response)]) as (url, requests):
            self.configure(url, False, '')
            result = self.invoke(['run', '--prompt', 'task', '--run-dir', str(self.root / 'run')])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(self.read('metrics.json')['model']['total_tokens'])

    def test_invalid_usage_fails_explicitly_with_measured_request(self) -> None:
        response = support.message('done')
        response['usage'] = {'input_tokens': -1}
        with support.model_endpoint([support.json_response(response)]) as (url, requests):
            self.configure(url, False, '')
            result = self.invoke(['run', '--prompt', 'task', '--run-dir', str(self.root / 'run')])
        self.assertEqual(result.returncode, 3, result.stderr)
        metrics = self.read('metrics.json')
        self.assertEqual(metrics['outcome']['error']['type'], 'response_usage_invalid')
        self.assertEqual(metrics['model']['request_count'], 1)
        self.assertEqual(len(metrics['model']['requests']), 1)

    def test_sensitive_final_text_fails_without_success_metadata(self) -> None:
        with support.model_endpoint([support.json_response(support.message('secret-model-key'))]) as (url, requests):
            self.configure(url, False, '')
            self.config.write_text(self.config.read_text().replace('api_key_env = ""', 'api_key_env = "MODEL_TEST_KEY"'))
            result = self.invoke(['run', '--prompt', 'task', '--run-dir', str(self.root / 'run')])
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertFalse((self.root / 'run/final.txt').exists())
        self.assertNotEqual(self.read('run.json')['status'], 'success')
        self.assertNotIn('secret-model-key', ''.join(p.read_text() for p in (self.root / 'run').iterdir()))

    def test_endpoint_identity_does_not_depend_on_rotated_credentials(self) -> None:
        identities = []
        for index, credential in enumerate(('credential-one', 'credential-two')):
            with support.model_endpoint([support.json_response(support.message('done'))]) as (url, requests):
                self.configure(url, False, '')
                config = load_config(self.config, {})
                privacy = ArtifactPrivacy(config, {'CUSTOM_SECRET': credential})
                identities.append(privacy.endpoint(f'https://user:pass@example.invalid/v1?api_key={credential}&api-version=v1'))
        self.assertEqual(identities[0], identities[1])
        self.assertNotIn('user:pass', identities[0])
        self.assertIn('api-version=v1', identities[0])

    def test_error_diagnostics_omit_raw_remote_authorization_and_environment(self) -> None:
        with support.model_endpoint([(500, 'application/json', b'{"Authorization":"Bearer remote-private-value"}')]) as (url, requests):
            self.configure(url, False, '')
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                code = run_cli(['--config', str(self.config), 'run', '--prompt', 'task',
                                '--run-dir', str(self.root / 'run')], {'HOME': str(self.root), 'SECRET': 'env-value'})
        self.assertEqual(code, 3)
        artifact_text = ''.join(p.read_text() for p in (self.root / 'run').iterdir())
        self.assertNotIn('remote-private-value', artifact_text)
        self.assertNotIn('env-value', artifact_text)
