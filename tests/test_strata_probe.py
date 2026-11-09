from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples.strata_probe import main
from tests.strata_fixtures import completed_response, function_item, text_item
from tests.test_headless import model_endpoint
from tests.test_strata_compatibility import wire_response


PROJECT = Path(__file__).resolve().parents[1]
SAMPLE = PROJECT / 'examples/strata/local.toml'
MARKER = '0e8d5ce0a5fc4a249083f594d652f817'


class StrataProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        parent = PROJECT / 'test_outputs'
        parent.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / 'config.toml'
        self.enterContext(patch.dict('os.environ', STRATA_API_KEY='fixture-bearer-secret'))
        self.enterContext(patch('secrets.token_hex', return_value=MARKER))

    def arguments(self, url: str, stream: bool, output: Path) -> list[str]:
        config = SAMPLE.read_text().replace('http://127.0.0.1:8080/v1/responses', url)
        self.config.write_text(config.replace('stream = true', f'stream = {str(stream).lower()}'))
        return ['--config', str(self.config), '--deployment', 'local', '--output', str(output)]

    def test_requires_explicit_live_opt_in_before_connecting(self) -> None:
        output = self.root / 'probe'
        with model_endpoint([]) as (url, requests):
            with self.assertRaises(SystemExit) as rejected:
                main(self.arguments(url, False, output))
        self.assertEqual(rejected.exception.code, 2)
        self.assertEqual(requests, [])
        self.assertFalse(output.exists())

    def test_tool_and_resume_send_one_matching_result_in_each_transport(self) -> None:
        for stream in (True, False):
            with self.subTest(stream=stream):
                output = self.root / str(stream)
                responses = [
                    completed_response([function_item('read', 'file_read', {'path': 'marker.txt'})]),
                    completed_response([text_item(MARKER)]),
                    completed_response([text_item(MARKER)]),
                ]
                with model_endpoint([wire_response(r, stream) for r in responses]) as (url, requests):
                    self.assertEqual(main(['--live', *self.arguments(url, stream, output)]), 0)
                self.assertEqual(len(requests), 3)
                self.assertNotIn(MARKER, json.dumps(requests[0]))
                for request in requests[1:]:
                    calls = [i for i in request['input'] if i.get('type') == 'function_call']
                    results = [i for i in request['input'] if i.get('type') == 'function_call_output']
                    self.assertEqual([i['call_id'] for i in calls], ['read'])
                    self.assertEqual([i['call_id'] for i in results], ['read'])
                    self.assertEqual(json.loads(results[0]['output'])['content'], MARKER)
                self.assertEqual([i for i in requests[-1]['input'] if i.get('role') == 'user'],
                                 [requests[0]['input'][0], requests[-1]['input'][-1]])
                report = (output / 'result.json').read_text()
                self.assertNotIn(url, report)
                self.assertNotIn('fixture-bearer-secret', report)
                self.assertEqual(json.loads(report)['status'], 'passed')
                self.assertEqual(json.loads(report)['metrics']['model']['total_tokens'], 480)

    def test_matching_answer_without_a_file_read_fails_before_resume(self) -> None:
        output = self.root / 'guess'
        with model_endpoint([wire_response(completed_response([text_item(MARKER)]), False)]) as (url, requests):
            self.assertEqual(main(['--live', *self.arguments(url, False, output)]), 1)
        self.assertEqual(len(requests), 1)
        result = json.loads((output / 'result.json').read_text())
        self.assertEqual((result['status'], result['stage']), ('failed', 'tool_round_trip'))

    def test_incomplete_or_disconnected_followup_preserves_tool_result_without_retry(self) -> None:
        for stream in (False, True):
            with self.subTest(stream=stream):
                output = self.root / str(stream)
                read = completed_response([function_item('read', 'file_read', {'path': 'marker.txt'})])
                if stream:
                    failure = (200, 'text/event-stream', b': heartbeat\n\n')
                else:
                    failure = wire_response({**completed_response([text_item(MARKER)]),
                                             'status': 'incomplete'}, False)
                with model_endpoint([wire_response(read, stream), failure]) as (url, requests):
                    self.assertEqual(main(['--live', *self.arguments(url, stream, output)]), 1)
                self.assertEqual(len(requests), 2)
                events = [json.loads(line) for line in (output / 'sessions/probe.jsonl').read_text().splitlines()]
                self.assertEqual([e['kind'] for e in events].count('tool_result'), 1)
                self.assertEqual(events[-1]['kind'], 'turn_error')
                result = json.loads((output / 'result.json').read_text())
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['error']['type'],
                                 'response_stream_terminated' if stream else 'response_incomplete')
