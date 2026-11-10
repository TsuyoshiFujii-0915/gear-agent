from collections.abc import Iterator
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from gear_agent.agent.events import AgentLoopEvent
from gear_agent.config import AppConfig, DEFAULT_CONFIG_TEXT, load_config
from gear_agent.errors import GearError, gear_error
from gear_agent.headless import TaskPrompt
from gear_agent.model.transport import HttpxHttpTransport, SseEvent
from gear_agent.run_artifacts import run_with_artifacts
from gear_agent.run_collector import EvalRunCollector
from gear_agent.runtime import build_agent_runtime
from gear_agent.store.sessions import JsonlSessionDiscovery
from tests.test_headless import json_response, message, model_endpoint, tool_call


class Events:
    def __init__(self) -> None:
        self.received: list[AgentLoopEvent] = []

    def publish(self, event: AgentLoopEvent) -> None:
        self.received.append(event)


def configuration(root: Path, url: str, stream: bool) -> AppConfig:
    path = root / 'config.toml'
    path.write_text(DEFAULT_CONFIG_TEXT)
    config = load_config(path, {})
    return replace(config, model=replace(config.model, url=url, stream=stream, api_key='secret-key'),
                   runtime=replace(config.runtime, workdir=root, session_dir=root / 'sessions'))


def records(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (root / 'sessions/s.requests.ndjson').read_text().splitlines()]


def test_interactive_resume_and_manual_summary_share_only_safe_observations(tmp_path: Path) -> None:
    sink = Events()
    with model_endpoint([json_response(message('private answer'))] * 3) as (url, requests):
        config = configuration(tmp_path, url + '?api_key=secret-key', False)
        runtime = build_agent_runtime(config, config.runtime, sink)
        runtime.loop.run_turn('s', 'private prompt', 1, 5)
        resumed = build_agent_runtime(config, config.runtime, sink)
        resumed.compaction.compact('s', resumed.store, 5)
        resumed.loop.run_turn('s', 'follow up', 1, 5)
    rows = records(tmp_path)
    assert [r['purpose'] for r in rows] == ['agent', 'manual_summary', 'agent']
    assert len({r['request_id'] for r in rows}) == len(requests) == 3
    assert len({r['turn_id'] for r in rows}) == 3
    assert rows[1]['iteration'] is rows[1]['turn_id'] is None
    assert all(r['client']['first_content']['reason'] == 'non_streaming' for r in rows)
    assert all(r['session_id'] == 's' and r['schema_version'] == 1 for r in rows)
    assert [e.request_id for e in sink.received if type(e).__name__ == 'RequestStarted'] == [r['request_id'] for r in rows]
    assert [e.request.request_id for e in sink.received if type(e).__name__ == 'RequestFinished'] == [r['request_id'] for r in rows]
    assert JsonlSessionDiscovery(config.runtime.session_dir).latest_session_id() == 's'
    for value in ('secret-key', 'private prompt', 'private answer', url):
        assert value not in json.dumps(rows)
    for row in rows:
        assert row['request_id'] not in json.dumps(requests) + json.dumps(resumed.store.load('s'))


@pytest.mark.parametrize(('content', 'answer', 'first', 'second'), [
    ('reasoning', True, 'response.reasoning_text.delta', 'response.output_text.delta'),
    ('tool', False, 'response.function_call_arguments.delta', None),
    ('terminal', True, 'response.completed', 'response.completed'),
])
def test_client_arrivals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str,
                         answer: bool, first: str, second: str | None) -> None:
    clock = [10.]
    config = configuration(tmp_path, 'http://example.test', True)
    body = message('private') if answer else tool_call('missing_tool', {})
    body['status'] = 'completed'
    events = [(11, {'type': 'response.created', 'response': {}}),
              (12, {'type': 'response.output_text.delta', 'output_index': 0, 'content_index': 0, 'delta': ''}),
              (13, {'type': 'response.output_item.added', 'output_index': 0,
                    'item': {'type': 'function_call', 'name': 'private', 'id': 'call'}})]
    if content == 'reasoning':
        events.extend([(14, {'type': first, 'output_index': 0, 'content_index': 0, 'delta': 'secret reasoning'}),
                       (16, {'type': second, 'output_index': 1, 'content_index': 0, 'delta': 'private'})])
    if content == 'tool':
        events.append((14, {'type': first, 'output_index': 0, 'item_id': 'call', 'delta': '{}'}))
    events.append((19, {'type': 'response.completed', 'response': body}))

    def stream(self: HttpxHttpTransport, *args: object) -> Iterator[SseEvent]:
        for timestamp, event in events:
            clock[0] = timestamp
            yield SseEvent('message', json.dumps(event))
        clock[0] = 20.

    monkeypatch.setattr(HttpxHttpTransport, 'post_sse', stream)
    monkeypatch.setattr('gear_agent.observation.perf_counter', lambda: clock[0])
    runtime = build_agent_runtime(config, config.runtime, Events())
    if answer:
        runtime.loop.run_turn('s', 'task', 1, 5, 5)
    else:
        with pytest.raises(GearError):
            runtime.loop.run_turn('s', 'task', 1, 5, 5)
    row, = records(tmp_path)
    client = row['client']
    assert row['status'] == 'success'
    assert client['total']['value'] == 10.
    assert (client['first_content']['basis'], client['first_content']['value']) == (first, 9 if content == 'terminal' else 4)
    assert (client['first_answer']['basis'], client['first_answer']['value']) == (second, {'terminal': 9, 'reasoning': 6, 'tool': None}[content])
    if not answer:
        assert client['first_answer']['reason'] == 'not_applicable'


@pytest.mark.parametrize(('ending', 'status'), [('failed', 'provider_failure'), ('incomplete', 'incomplete'),
                                                ('eof', 'eof'), ('timeout', 'timeout'), ('cancel', 'cancelled')])
def test_failed_request_after_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      ending: str, status: str) -> None:
    config = configuration(tmp_path, 'http://example.test', True)
    calls: list[object] = []

    def stream(self: HttpxHttpTransport, *args: object) -> Iterator[SseEvent]:
        calls.append(args)
        if len(calls) == 1:
            body = {**message('done'), 'status': 'completed', 'usage': {'input_tokens': 100}}
            yield SseEvent('message', json.dumps({'type': 'response.completed', 'response': body}))
        elif ending == 'timeout':
            raise gear_error('http_stream_idle_timeout', 'private remote message', 'model_client', True, {})
        elif ending == 'cancel':
            raise KeyboardInterrupt()
        elif ending != 'eof':
            body = {**message('partial'), 'status': ending, 'usage': {'input_tokens': 7}}
            yield SseEvent('message', json.dumps({'type': 'response.' + ending, 'response': body}))

    monkeypatch.setattr(HttpxHttpTransport, 'post_sse', stream)
    runtime = build_agent_runtime(config, config.runtime, Events())
    runtime.loop.run_turn('s', 'one', 1, 5, 5)
    with pytest.raises((GearError, KeyboardInterrupt)):
        runtime.loop.run_turn('s', 'two', 1, 5, 5)
    one, two = records(tmp_path)
    assert len(calls) == 2 and one['request_id'] != two['request_id'] and one['turn_id'] != two['turn_id']
    assert two['status'] == status
    if ending in ('failed', 'incomplete'):
        assert two['statistics']['input_tokens']['value'] == 7
    else:
        assert two['statistics'] is None and two['statistics_reason'] == 'interrupted_before_report'
    assert 'partial' not in json.dumps(runtime.store.load('s'))
    assert 'private remote message' not in json.dumps(two)


def test_headless_cache_coverage_and_shared_request_ids(tmp_path: Path) -> None:
    (tmp_path / 'data').write_text('file')
    call = tool_call('file_read', {'path': 'data'})
    bodies = [{**call, 'usage': {'input_tokens': 4, 'input_tokens_details': {'cached_tokens': 4}}},
              {**call, 'usage': {'input_tokens': 36, 'input_tokens_details': {'cached_tokens': 0}}},
              message('done')]
    with model_endpoint([json_response(b) for b in bodies]) as (url, requests):
        config = configuration(tmp_path, url, False)
        collector = EvalRunCollector()
        runtime = build_agent_runtime(config, config.runtime, collector, collector)
        run_with_artifacts(runtime, 's', TaskPrompt('read', 'inline'), tmp_path / 'run', collector, {})
    model = json.loads((tmp_path / 'run/metrics.json').read_text())['model']
    rows = records(tmp_path)
    assert len(rows) == len(requests) == model['request_count'] == 3
    assert [r['request_id'] for r in model['requests']] == [r['request_id'] for r in rows]
    assert model['cache_by_purpose']['agent'] == {'ratio': 0.1, 'included': 2, 'requests': 3,
            'cached_tokens': 4, 'input_tokens': 40, 'excluded': {'not_reported': 1}}
    assert model['input_tokens'] is None
    assert all(r['turn_id'] == rows[0]['turn_id'] for r in rows)
    assert [r['iteration'] for r in rows] == [1, 2, 3]


def test_sidecar_write_failure_is_safe_and_does_not_repeat_work(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with model_endpoint([json_response(message('done'))]) as (url, requests):
        config = configuration(tmp_path, url, False)
        runtime = build_agent_runtime(config, config.runtime, Events())
        target = tmp_path / 'secret-path'
        target.write_text('keep')
        (config.runtime.session_dir / 's.requests.ndjson').symlink_to(target)
        assert runtime.loop.run_turn('s', 'task', 1, 5).final_text == 'done'
    assert len(requests) == 1 and target.read_text() == 'keep'
    assert 'request_log_write_failed' in caplog.text and 'secret-path' not in caplog.text


def test_resuming_legacy_history_does_not_backfill_requests(tmp_path: Path) -> None:
    with model_endpoint([json_response(message('done'))]) as (url, requests):
        config = configuration(tmp_path, url, False)
        runtime = build_agent_runtime(config, config.runtime, Events())
        runtime.store.append('s', 'user_input', {'text': 'legacy task'})
        runtime.store.append('s', 'assistant_message', {'text': 'legacy answer'})
        old = runtime.store.load('s')
        runtime.loop.run_turn('s', 'new task', 1, 5)
    assert runtime.store.load('s')[:2] == old
    assert 'legacy answer' in json.dumps(requests[0]['input'])
    assert len(records(tmp_path)) == 1
