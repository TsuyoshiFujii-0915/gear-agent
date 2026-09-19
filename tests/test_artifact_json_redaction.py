from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from typing import Any
import json
import unittest

from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.cli import run_cli
from gear_agent.config import load_config
from gear_agent.errors import GearError
from tests import test_headless as support


class ArtifactJsonRedactionTests(unittest.TestCase):
    setUp = support.HeadlessCliTests.setUp
    configure = support.HeadlessCliTests.configure

    def privacy(self, environment: dict[str, str]) -> ArtifactPrivacy:
        self.configure('http://localhost:1234/v1/responses', False, '')
        return ArtifactPrivacy(load_config(self.config, {}), environment)

    def test_structured_arguments_redact_nested_content_without_mutating_input(self) -> None:
        privacy = self.privacy({})
        content = json.dumps({'Authorization': 'Bearer header-private-ABC'})
        item = support.tool_call('file_write', {'path': 'settings.json', 'content': content})['output'][0]
        original = deepcopy(item)
        filtered = privacy.serialize(item)
        arguments = json.loads(filtered['arguments'])
        self.assertEqual(json.loads(arguments['content'])['Authorization'], '[REDACTED]')
        self.assertNotIn('header-private-ABC', json.dumps(filtered))
        self.assertEqual(item, original)

    def test_known_secret_survives_no_decoding_of_filtered_arguments(self) -> None:
        secret = 'demo"secret\\\\value\n日本語'
        privacy = self.privacy({'PRIVATE_SECRET': secret})
        for content in (secret, json.dumps({'content': secret}), json.dumps({'content': secret}, ensure_ascii=False)):
            with self.subTest(content=content):
                item = support.tool_call('file_write', {'path': 'settings.json', 'content': content})['output'][0]
                projected = privacy.serialize(item)
                decoded = json.loads(projected['arguments'])['content']
                if content == secret:
                    self.assertEqual(decoded, '[REDACTED]')
                else:
                    self.assertEqual(json.loads(decoded)['content'], '[REDACTED]')
                self.assertEqual(json.loads(item['arguments'])['content'], content)

    def test_protocol_arguments_must_be_valid_json_objects(self) -> None:
        privacy = self.privacy({})
        for arguments in ('{broken', '[]', 'null', '"text"', '{"content": NaN}', {'content': 'not encoded'}):
            with self.subTest(arguments=arguments):
                with self.assertRaises(GearError) as raised:
                    privacy.serialize({'type': 'function_call', 'arguments': arguments})
                self.assertEqual(raised.exception.error_type, 'artifact_json_invalid')
                self.assertEqual(raised.exception.origin, 'run_artifacts')
                self.assertNotIn('broken', str(raised.exception))

    def test_protocol_argument_size_and_depth_are_bounded(self) -> None:
        privacy = self.privacy({})
        cases = [json.dumps({'content': 'x' * (1024 * 1024)}),
                 '{"nested":' + '[' * 65 + '0' + ']' * 65 + '}']
        for arguments in cases:
            with self.subTest(size=len(arguments)):
                with self.assertRaises(GearError) as raised:
                    privacy.serialize({'type': 'function_call', 'arguments': arguments})
                self.assertEqual(raised.exception.error_type, 'artifact_json_limit_exceeded')
        safe = privacy.serialize({'type': 'function_call', 'arguments': '{"data": [[1, 2]]}'})
        self.assertEqual(json.loads(safe['arguments']), {'data': [[1, 2]]})

    def test_arbitrary_strings_and_argument_data_are_not_recursively_decoded(self) -> None:
        privacy = self.privacy({})
        plain = {'note': '{invalid json', 'arguments': 'not a protocol field'}
        self.assertEqual(privacy.serialize(plain), plain)
        data = {'type': 'function_call', 'arguments': 'user data, not another protocol call'}
        projected = privacy.serialize({'type': 'function_call', 'arguments': json.dumps(data)})
        self.assertEqual(json.loads(projected['arguments']), data)

    def test_exact_text_rejects_json_escaped_secret_but_preserves_safe_text(self) -> None:
        secret = 'demo"secret\\\\value'
        privacy = self.privacy({'PRIVATE_SECRET': secret})
        for source in ('task.text', 'final.txt'):
            for text in (secret, json.dumps({'content': secret})):
                with self.subTest(source=source, text=text):
                    with self.assertRaises(GearError) as raised:
                        privacy.exact_text(text, source)
                    self.assertEqual(raised.exception.error_type, 'artifact_sensitive_text')
            safe = '  { "text" : "ordinary \\n text" }\n'
            self.assertEqual(privacy.exact_text(safe, source), safe)

    def test_events_snapshot_redacts_every_copy_and_leaves_session_and_tool_inputs_intact(self) -> None:
        header = 'header-private-ABC'
        secret = 'demo"secret\\\\value\n日本語'
        content = json.dumps({'Authorization': f'Bearer {header}', 'note': secret})
        response = support.tool_call('file_write', {'path': 'settings.json', 'content': content})
        original = deepcopy(response)
        destination = self.root / 'run'
        stderr = StringIO()
        with support.model_endpoint([support.json_response(response), support.json_response(support.message('done'))]) as (url, requests):
            self.configure(url, False, '')
            self.config.write_text(self.config.read_text().replace('file_write = false', 'file_write = true'))
            with redirect_stdout(StringIO()), redirect_stderr(stderr):
                code = run_cli(['--config', str(self.config), 'run', '--prompt', 'write settings',
                                '--run-dir', str(destination)], {'HOME': str(self.root), 'PRIVATE_SECRET': secret})
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual((self.workspace / 'settings.json').read_text(), content)
        events = [json.loads(line) for line in (destination / 'events.jsonl').read_text().splitlines()]
        canonical = [e for e in events if e['kind'] == 'model_response'][0]['payload']['output'][0]
        decoded_call = [e for e in events if e['kind'] == 'tool_call'][0]['payload']['arguments']
        for arguments in (json.loads(canonical['arguments']), decoded_call):
            copied_content = json.loads(arguments['content'])
            self.assertEqual(copied_content['Authorization'], '[REDACTED]')
            self.assertEqual(copied_content['note'], '[REDACTED]')
        self.assertNotIn(header, (destination / 'events.jsonl').read_text())
        session_path, = self.sessions.glob('*.jsonl')
        history = [json.loads(line) for line in session_path.read_text().splitlines()]
        stored_response = [e for e in history if e['kind'] == 'model_response'][0]['payload']
        self.assertEqual(stored_response, original)
        stored_call = [e for e in history if e['kind'] == 'tool_call'][0]['payload']['arguments']
        self.assertEqual(json.loads(stored_call['content'])['note'], secret)
        continuation_call = next(item for item in requests[1]['input'] if item.get('type') == 'function_call')
        self.assertEqual(continuation_call['arguments'], original['output'][0]['arguments'])

    def test_invalid_protocol_json_never_enters_artifact_copy(self) -> None:
        response: dict[str, Any] = {'output': [{'type': 'function_call', 'call_id': 'c1', 'name': 'file_write',
                                               'arguments': '{"content": "sensitive-unfinished'}]}
        destination = self.root / 'run'
        stderr = StringIO()
        with support.model_endpoint([support.json_response(response)]) as (url, requests):
            self.configure(url, False, '')
            with redirect_stdout(StringIO()), redirect_stderr(stderr):
                code = run_cli(['--config', str(self.config), 'run', '--prompt', 'task',
                                '--run-dir', str(destination)], {'HOME': str(self.root)})
        self.assertEqual(code, 1, stderr.getvalue())
        self.assertIn('artifact_json_invalid', stderr.getvalue())
        self.assertEqual(json.loads((destination / 'run.json').read_text())['status'], 'running')
        self.assertFalse((destination / 'events.jsonl').exists())
        self.assertFalse((destination / 'final.txt').exists())
        self.assertNotIn('sensitive-unfinished', ''.join(path.read_text() for path in destination.iterdir()))
