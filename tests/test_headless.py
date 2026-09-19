from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Iterator
import json
import os
import subprocess
import sys
import tempfile
import unittest

from gear_agent.config import DEFAULT_CONFIG_TEXT


@contextmanager
def model_endpoint(
    responses: list[tuple[int, str, bytes]],
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Serves scripted external model responses and records actual requests."""
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers['Content-Length'])
            requests.append(json.loads(self.rfile.read(length)))
            index = len(requests) - 1
            if index >= len(responses):
                self.send_error(500, 'Unexpected extra request')
                return
            status, content_type, body = responses[index]
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1/responses', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def message(text: str) -> dict[str, Any]:
    return {'output': [{'type': 'message', 'role': 'assistant',
                        'content': [{'type': 'output_text', 'text': text}]}]}


def json_response(value: dict[str, Any]) -> tuple[int, str, bytes]:
    return 200, 'application/json', json.dumps(value).encode()


def tool_call(name: str, arguments: dict[str, object]) -> dict[str, Any]:
    return {'output': [{'type': 'function_call', 'call_id': 'call-1',
                        'name': name, 'arguments': json.dumps(arguments)}]}


class HeadlessCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()
        self.sessions = self.root / 'sessions'
        self.config = self.root / 'config.toml'
        self.source = Path(__file__).resolve().parents[1] / 'src'

    def configure(self, url: str, stream: bool, budget: str) -> None:
        config = DEFAULT_CONFIG_TEXT.replace('http://localhost:1234/v1/responses', url)
        config = config.replace('workdir = "."', f'workdir = {json.dumps(str(self.workspace))}')
        config = config.replace('session_dir = ".gear/sessions"',
                                f'session_dir = {json.dumps(str(self.sessions))}')
        for tool in ('shell_tool', 'file_write', 'apply_patch', 'glob', 'grep'):
            config = config.replace(f'{tool} = true', f'{tool} = false')
        config = config.replace('stream = false', f'stream = {str(stream).lower()}')
        config = config.replace('[context_budget]\nauto_compaction = false', budget)
        self.config.write_text(config, encoding='utf-8')

    def invoke(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(self.source)
        environment['MODEL_TEST_KEY'] = 'secret-model-key'
        environment['HOME'] = str(self.root)
        return subprocess.run(
            [sys.executable, '-m', 'gear_agent.cli', '--config', str(self.config), *arguments],
            cwd=self.root, env=environment, capture_output=True, text=True, timeout=20,
        )

    def history(self, result: subprocess.CompletedProcess[str]) -> list[dict[str, Any]]:
        session_line = next(line for line in result.stderr.splitlines()
                            if line.startswith('session_id='))
        session_id = session_line.removeprefix('session_id=')
        return [json.loads(line) for line in
                (self.sessions / f'{session_id}.jsonl').read_text().splitlines()]

    def test_inline_success_persists_canonical_history_and_fresh_sessions(self) -> None:
        with model_endpoint([json_response(message('完了')), json_response(message('完了'))]) as (url, requests):
            self.configure(url, False, '')
            first = self.invoke(['run', '--prompt', 'タスク'])
            second = self.invoke(['run', '--prompt', 'タスク'])
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout, '完了\n')
        self.assertNotEqual(first.stderr, second.stderr)
        events = self.history(first)
        self.assertEqual([e['kind'] for e in events],
                         ['user_input', 'model_response', 'assistant_message'])
        self.assertEqual(events[0]['payload']['text'], 'タスク')
        self.assertEqual(events[-1]['payload']['text'], '完了')
        self.assertEqual(requests[0]['input'], [{'role': 'user', 'content': 'タスク'}])

    def test_utf8_prompt_file_preserves_input(self) -> None:
        prompt = self.root / 'task.md'
        prompt.write_text('修正してください\n二行目\n', encoding='utf-8')
        with model_endpoint([json_response(message('done'))]) as (url, requests):
            self.configure(url, False, '')
            result = self.invoke(['run', '--prompt-file', str(prompt)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(requests[0]['input'][0]['content'], prompt.read_text())

    def test_missing_or_conflicting_prompts_are_usage_errors(self) -> None:
        for arguments in (['run'], ['run', '--prompt', 'x', '--prompt-file', 'task.md']):
            with self.subTest(arguments=arguments):
                result = self.invoke(arguments)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, '')
                self.assertIn('usage:', result.stderr)

    def test_invalid_prompt_files_and_empty_prompts_are_runtime_errors(self) -> None:
        bad = self.root / 'bad.md'
        bad.write_bytes(b'\xff')
        empty = self.root / 'empty.md'
        empty.write_text(' \n')
        for arguments in (['--prompt-file', 'missing.md'], ['--prompt-file', str(bad)],
                          ['--prompt-file', str(empty)], ['--prompt', '  ']):
            with self.subTest(arguments=arguments):
                result = self.invoke(['run', *arguments])
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, '')
                self.assertIn('prompt', result.stderr)
                self.assertNotIn('Traceback', result.stderr)

    def test_real_file_tool_and_hierarchical_instructions_use_effective_workspace(self) -> None:
        target = self.root / 'target'
        (target / 'pkg').mkdir(parents=True)
        (target / 'AGENTS.md').write_text('ROOT-RULE', encoding='utf-8')
        (target / 'pkg' / 'AGENTS.md').write_text('CHILD-RULE', encoding='utf-8')
        (target / 'pkg' / 'data.txt').write_text('file contents', encoding='utf-8')
        responses = [json_response(tool_call('file_read', {'path': 'pkg/data.txt'})),
                     json_response(message('read done'))]
        with model_endpoint(responses) as (url, requests):
            self.configure(url, False, '')
            result = self.invoke(['--workdir', str(target), '--session-dir', str(self.sessions),
                                  '--network', 'enabled', '--max-iterations', '2',
                                  '--model-timeout-seconds', '3', 'run', '--prompt', 'read'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'read done\n')
        self.assertEqual([tool['name'] for tool in requests[0]['tools']], ['file_read'])
        self.assertIn('ROOT-RULE', requests[0]['instructions'])
        self.assertIn('CHILD-RULE', requests[1]['instructions'])
        self.assertIn('file contents', json.dumps(requests[1]['input']))
        self.assertIn('tool_result', [e['kind'] for e in self.history(result)])

    def test_streaming_outputs_only_canonical_answer_once_without_loading_textual(self) -> None:
        (self.root / 'textual.py').write_text('raise RuntimeError("Textual must not load")\n')
        fixture = Path(__file__).parent / 'fixtures/responses/text.sse'
        body = fixture.read_bytes() + b'\n'
        with model_endpoint([(200, 'text/event-stream', body)]) as (url, requests):
            self.configure(url, True, '')
            result = self.invoke(['run', '--prompt', 'hello'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(requests[0]['stream'])
        answer = self.history(result)[-1]['payload']['text']
        self.assertEqual(result.stdout, answer + '\n')
        self.assertEqual(len(result.stderr.splitlines()), 1)

    def test_stream_failure_keeps_partial_text_off_stdout_without_retry(self) -> None:
        fixture = Path(__file__).parent / 'fixtures/responses/premature_eof.sse'
        with model_endpoint([(200, 'text/event-stream', fixture.read_bytes())]) as (url, requests):
            self.configure(url, True, '')
            result = self.invoke(['run', '--prompt', 'hello'])
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertEqual(len(requests), 1)
        self.assertNotIn('assistant_message', [e['kind'] for e in self.history(result)])
        self.assertIn('stream', result.stderr)

    def test_model_failure_redacts_credentials_and_remote_error_details(self) -> None:
        with model_endpoint([(401, 'application/json', b'{"error":"secret-model-key remote-secret"}')]) as (url, requests):
            self.configure(url + '?token=secret-model-key', False, '')
            self.config.write_text(self.config.read_text().replace('api_key_env = ""',
                                                                  'api_key_env = "MODEL_TEST_KEY"'))
            result = self.invoke(['run', '--prompt', 'hello'])
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertIn('http_status_error', result.stderr)
        self.assertIn('model_client', result.stderr)
        self.assertNotIn('secret-model-key', result.stderr)
        self.assertNotIn('remote-secret', result.stderr)
        self.assertEqual(len(requests), 1)

    def test_iteration_limit_and_missing_final_text_fail(self) -> None:
        for response, error in ((tool_call('file_read', {'path': '../outside'}), 'iteration_limit_reached'),
                                ({'output': []}, 'final_text_missing')):
            with self.subTest(error=error), model_endpoint([json_response(response)]) as (url, requests):
                self.configure(url, False, '')
                result = self.invoke(['--max-iterations', '1', 'run', '--prompt', 'hello'])
                self.assertEqual(result.returncode, 3, result.stderr)
                self.assertEqual(result.stdout, '')
                self.assertIn(error, result.stderr)
                self.assertEqual(len(requests), 1)

    def test_automatic_compaction_preserves_current_task_and_history(self) -> None:
        (self.workspace / 'large.txt').write_text('x' * 12000)
        responses = [json_response(tool_call('file_read', {'path': 'large.txt'})),
                     json_response(message('short checkpoint')), json_response(message('done'))]
        budget = ('[context_budget]\nauto_compaction = true\ncontext_window_tokens = 100000\n'
                  'reserved_tokens = 1000\nmax_input_tokens = 6000')
        with model_endpoint(responses) as (url, requests):
            self.configure(url, False, budget)
            result = self.invoke(['run', '--prompt', 'read large file'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'done\n')
        self.assertEqual(len(requests), 3)
        self.assertIn('short checkpoint', json.dumps(requests[2]['input']))
        self.assertEqual(requests[2]['input'][-1]['content'], 'read large file')
        checkpoints = [e for e in self.history(result) if e['kind'] == 'compaction_summary']
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]['payload']['trigger'], 'automatic')

    def test_context_budget_failure_prevents_dispatch(self) -> None:
        budget = ('[context_budget]\nauto_compaction = true\ncontext_window_tokens = 1000\n'
                  'reserved_tokens = 100\nmax_input_tokens = 800')
        with model_endpoint([]) as (url, requests):
            self.configure(url, False, budget)
            result = self.invoke(['run', '--prompt', 'huge prompt ' * 1000])
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertIn('context_budget_exceeded', result.stderr)
        self.assertEqual(requests, [])

    def test_config_and_filesystem_errors_are_explicit_runtime_failures(self) -> None:
        self.configure('http://127.0.0.1:1/v1/responses', False, '')
        bad_session_dir = self.root / 'blocked'
        bad_session_dir.write_text('file')
        cases = [(['--config', str(self.root / 'missing.toml')], 'config_read_failed'),
                 (['--workdir', str(self.root / 'missing')], 'workspace'),
                 (['--session-dir', str(bad_session_dir)], 'runtime')]
        for options, expected in cases:
            with self.subTest(options=options):
                result = self.invoke([*options, 'run', '--prompt', 'hello'])
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.stdout, '')
                self.assertIn(expected, result.stderr)
                self.assertNotIn('Traceback', result.stderr)

    def test_nonpositive_runtime_overrides_fail_before_model_dispatch(self) -> None:
        with model_endpoint([]) as (url, requests):
            self.configure(url, False, '')
            for option in ('--max-iterations', '--model-timeout-seconds'):
                result = self.invoke([option, '0', 'run', '--prompt', 'hello'])
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.stdout, '')
            self.assertEqual(requests, [])


if __name__ == '__main__':
    unittest.main()
