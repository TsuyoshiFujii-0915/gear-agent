from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch
import json
import os
import subprocess
import sys

import pytest

from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.cli import run_cli
from gear_agent.config import AppConfig, DEFAULT_CONFIG_TEXT, load_config
from gear_agent.context_budget import ContextBudgetConfig
from gear_agent.errors import GearError
from gear_agent.headless import TaskPrompt, run_task
from gear_agent.runtime import build_agent_runtime
from gear_agent.tui import collect_chat_lines
from tests.test_headless import json_response, message, model_endpoint, tool_call


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    text = DEFAULT_CONFIG_TEXT.replace('api_key_env = ""', 'api_key_env = "MODEL_TEST_KEY"')
    text = text.replace('workdir = "."', f'workdir = {json.dumps(str(tmp_path))}')
    text = text.replace('session_dir = ".gear/sessions"',
                        f'session_dir = {json.dumps(str(tmp_path / "sessions"))}')
    path = tmp_path / 'config.toml'
    path.write_text(text, encoding='utf-8')
    return path


def settings(path: Path, url: str) -> AppConfig:
    config = load_config(path, {'MODEL_TEST_KEY': 'model-secret'})
    return replace(config, model=replace(config.model, url=url))


@pytest.mark.parametrize('failure', ['http', 'stream', 'iterations', 'budget'])
def test_cli_persists_one_safe_terminal_error(config_path: Path, failure: str) -> None:
    stream = (b'data: {"type":"response.created","response":{"id":"r","output":[]}}\n\n'
              b'data: {"type":"response.output_text.delta","output_index":0,'
              b'"content_index":0,"delta":"partial response"}\n\n')
    responses = {
        'http': [(401, 'application/json', b'{"error":"model-secret remote-body"}')],
        'stream': [(200, 'text/event-stream', stream)],
        'iterations': [json_response(tool_call('file_read', {'path': 'missing'}))],
        'budget': [],
    }
    error_types = {'http': 'http_status_error', 'stream': 'response_stream_terminated',
                   'iterations': 'iteration_limit_reached', 'budget': 'context_budget_exceeded'}
    with model_endpoint(responses[failure]) as (url, requests):
        config = config_path.read_text().replace('http://localhost:1234/v1/responses',
                                                url + '?token=model-secret')
        if failure == 'stream':
            config = config.replace('stream = false', 'stream = true')
        if failure == 'budget':
            config = config.replace('auto_compaction = false',
                                    'auto_compaction = true\ncontext_window_tokens = 1000\n'
                                    'reserved_tokens = 100\nmax_input_tokens = 800')
        config_path.write_text(config)
        prompt = 'task ' * 1000 if failure == 'budget' else 'task'
        result = subprocess.run(
            [sys.executable, '-m', 'gear_agent.cli', '--config', str(config_path),
             '--max-iterations', '1', 'run', '--prompt', prompt],
            env={**os.environ, 'MODEL_TEST_KEY': 'model-secret'},
            capture_output=True, text=True, timeout=20,
        )
    assert result.returncode == 3, result.stderr
    assert result.stdout == ''
    assert len(requests) == (0 if failure == 'budget' else 1)
    session_line, diagnostic_line = result.stderr.splitlines()
    session_id = session_line.removeprefix('session_id=')
    events = [json.loads(line) for line in
              (config_path.parent / 'sessions' / f'{session_id}.jsonl').read_text().splitlines()]
    failures = [event for event in events if event['kind'] == 'turn_error']
    assert len(failures) == 1
    payload = failures[0]['payload']
    diagnostic = json.loads(diagnostic_line)['error']
    assert diagnostic['type'] == error_types[failure]
    assert payload['error'] == diagnostic
    assert payload['text'] == f"{diagnostic['origin']}: {diagnostic['message']}"
    for value in ('model-secret', 'remote-body', url, 'partial response'):
        assert value not in json.dumps(payload)
        assert value not in result.stderr
    assert not any(event['kind'] == 'assistant_message' for event in events)
    assert collect_chat_lines(events)[-1].text == payload['text']


def test_python_runner_records_failure_for_resume(config_path: Path) -> None:
    with model_endpoint([(401, 'application/json', b'remote-body model-secret')]) as (url, requests):
        config = settings(config_path, url)
        agent = build_agent_runtime(config, config.runtime, SilentAgentLoopEventSink())
        with pytest.raises(GearError) as caught:
            run_task(agent, 'python-run', TaskPrompt('task', 'inline'))
    assert caught.value.error_type == 'http_status_error'
    assert len(requests) == 1
    events = agent.store.load('python-run')
    assert [event['kind'] for event in events] == ['user_input', 'turn_error']
    assert events[-1]['payload']['error']['type'] == caught.value.error_type
    assert 'remote-body' not in json.dumps(events[-1])
    assert 'model-secret' not in json.dumps(events[-1])


def test_corrected_tool_error_is_not_a_terminal_failure(config_path: Path) -> None:
    responses = [json_response(tool_call('file_read', {'path': 'missing'})),
                 json_response(message('Explained the missing file.'))]
    with model_endpoint(responses) as (url, requests):
        config = settings(config_path, url)
        agent = build_agent_runtime(config, config.runtime, SilentAgentLoopEventSink())
        result = run_task(agent, 'recovered', TaskPrompt('task', 'inline'))
    assert result.turn.final_text == 'Explained the missing file.'
    assert len(requests) == 2
    events = agent.store.load('recovered')
    assert any(event['kind'] == 'tool_result' and 'error' in event['payload']['result']
               for event in events)
    assert not any(event['kind'] == 'turn_error' for event in events)


def test_known_credentials_are_redacted_in_persisted_message(config_path: Path) -> None:
    url = 'https://user:password@example.test/v1/responses?token=model-secret'
    config = settings(config_path, url)
    agent = build_agent_runtime(config, config.runtime, SilentAgentLoopEventSink())
    original = GearError('http_request_failed', f'Failed with model-secret at {url}',
                         'model_client', True, {'body': 'remote-body'})
    with patch('gear_agent.model.transport.HttpxHttpTransport.post_json', side_effect=original):
        with pytest.raises(GearError) as caught:
            run_task(agent, 'redacted', TaskPrompt('task', 'inline'))
    assert caught.value is original
    event = agent.store.load('redacted')[-1]
    assert event['kind'] == 'turn_error'
    assert '[REDACTED]' in event['payload']['text']
    for value in ('model-secret', 'password', url, 'remote-body'):
        assert value not in json.dumps(event)


def test_session_write_failure_preserves_original_error_and_reports_cause(config_path: Path) -> None:
    config = settings(config_path, 'http://example.test/v1/responses')
    agent = build_agent_runtime(config, config.runtime, SilentAgentLoopEventSink())
    original = GearError('http_status_error', 'Model endpoint returned HTTP 401.',
                         'model_client', True, {'body': 'remote-body'})

    def fail_request(*args: object, **kwargs: object) -> dict[str, Any]:
        config.runtime.session_dir.rename(config_path.parent / 'saved-sessions')
        config.runtime.session_dir.write_text('Blocked session directory')
        raise original

    with patch('gear_agent.model.transport.HttpxHttpTransport.post_json', side_effect=fail_request):
        with pytest.raises(GearError) as caught:
            run_task(agent, 'failed-write', TaskPrompt('task', 'inline'))
    assert caught.value is original
    assert isinstance(caught.value.__cause__, GearError)
    assert caught.value.__cause__.error_type == 'turn_error_write_failed'
    assert 'remote-body' not in str(caught.value.__cause__)
    assert config.runtime.session_dir.is_file()


def test_cli_keeps_task_exit_code_when_failure_record_cannot_be_saved(
    config_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    original = GearError('http_status_error', 'Model endpoint returned HTTP 401.',
                         'model_client', True, {'body': 'remote-body'})

    def fail_request(*args: object, **kwargs: object) -> dict[str, Any]:
        sessions = config_path.parent / 'sessions'
        sessions.rename(config_path.parent / 'saved-sessions')
        sessions.write_text('Blocked session directory')
        raise original

    with patch('gear_agent.model.transport.HttpxHttpTransport.post_json', side_effect=fail_request):
        exit_code = run_cli(['--config', str(config_path), 'run', '--prompt', 'task'],
                            {'MODEL_TEST_KEY': 'model-secret'})
    output = capsys.readouterr()
    assert exit_code == 3
    assert output.out == ''
    diagnostic = json.loads(output.err.splitlines()[-1])
    assert diagnostic['error']['type'] == 'http_status_error'
    assert diagnostic['persistence_error']['type'] == 'turn_error_write_failed'
    assert 'remote-body' not in output.err
