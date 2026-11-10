import json
from pathlib import Path
from typing import Any

import pytest

from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.headless import TaskPrompt
from gear_agent.run_artifacts import run_with_artifacts
from gear_agent.run_collector import EvalRunCollector
from gear_agent.runtime import build_agent_runtime
from tests.test_headless import json_response, message, model_endpoint
from tests.test_request_telemetry import configuration


@pytest.mark.parametrize(('usage', 'included', 'excluded'), [
    ({'input_tokens': 0, 'input_tokens_details': {'cached_tokens': 0}}, 1, {}),
    ({'input_tokens': 4, 'input_tokens_details': {'cached_tokens': 5}}, 0, {'exceeds_input_tokens': 1}),
])
def test_cache_coverage_uses_valid_operand_pairs(tmp_path: Path, usage: dict[str, Any], included: int,
                                                excluded: dict[str, int]) -> None:
    with model_endpoint([json_response({**message('done'), 'usage': usage})]) as (url, requests):
        config = configuration(tmp_path, url, False)
        collector = EvalRunCollector()
        runtime = build_agent_runtime(config, config.runtime, collector, collector)
        run_with_artifacts(runtime, 's', TaskPrompt('read', 'inline'), tmp_path / 'run', collector, {})
    aggregate = json.loads((tmp_path / 'run/metrics.json').read_text())['model']['cache_by_purpose']['agent']
    assert aggregate['included'] == included and aggregate['excluded'] == excluded
    assert aggregate['ratio'] is None
    assert aggregate['input_tokens'] == aggregate['cached_tokens'] == (0 if included else None)


def test_duplicate_notifications_cannot_count_one_dispatch_twice(tmp_path: Path) -> None:
    with model_endpoint([json_response(message('done'))]) as (url, requests):
        config = configuration(tmp_path, url, False)
        collector = EvalRunCollector()
        collector.start()
        runtime = build_agent_runtime(config, config.runtime, collector, collector)
        runtime.loop.run_turn('s', 'task', 1, 5)
    for event in tuple(collector.events):
        if event['kind'] in ('model_request_started', 'model_request_finished'):
            collector.record(event['kind'], event['payload'])
    model = collector.metrics('success', None, True, runtime.store.load('s'), ArtifactPrivacy(config, {}))['model']
    assert model['request_count'] == len(model['requests']) == len(requests) == 1


def test_stream_preflight_error_has_no_http_request_or_lifecycle(tmp_path: Path) -> None:
    from gear_agent.errors import GearError

    with model_endpoint([]) as (url, requests):
        config = configuration(tmp_path, url, True)
        collector = EvalRunCollector()
        runtime = build_agent_runtime(config, config.runtime, collector, collector)
        with pytest.raises(GearError, match='idle timeout'):
            runtime.loop.run_turn('s', 'task', 1, 5, None)
        assert not requests
        assert not [event for event in collector.events if event['kind'].startswith('model_request_')]
        assert not (config.runtime.session_dir / 's.requests.ndjson').exists()


def test_disconnect_after_terminal_report_retains_statistics_without_committing_answer(tmp_path: Path,
                                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    from collections.abc import Iterator
    from gear_agent.errors import GearError, gear_error
    from gear_agent.model.transport import HttpxHttpTransport, SseEvent
    from tests.test_request_telemetry import Events, records

    def interrupted(self: HttpxHttpTransport, *args: object) -> Iterator[SseEvent]:
        body = {**message('private'), 'status': 'completed', 'usage': {'input_tokens': 9}}
        yield SseEvent('message', json.dumps({'type': 'response.completed', 'response': body}))
        raise gear_error('http_stream_failed', 'stream interrupted', 'model_client', True, {})

    monkeypatch.setattr(HttpxHttpTransport, 'post_sse', interrupted)
    config = configuration(tmp_path, 'http://example.test', True)
    runtime = build_agent_runtime(config, config.runtime, Events())
    with pytest.raises(GearError):
        runtime.loop.run_turn('s', 'task', 1, 5, 5)
    row, = records(tmp_path)
    assert row['status'] == 'transport_error' and row['statistics']['input_tokens']['value'] == 9
    assert [event['kind'] for event in runtime.store.load('s')] == ['user_input']


def test_failed_json_finish_preserves_safe_error_identity(tmp_path: Path) -> None:
    from gear_agent.errors import GearError
    from tests.test_request_telemetry import Events, records

    response = {'status': 'failed', 'error': {'message': 'private remote message'}}
    with model_endpoint([json_response(response)]) as (url, requests):
        config = configuration(tmp_path, url, False)
        runtime = build_agent_runtime(config, config.runtime, Events())
        with pytest.raises(GearError):
            runtime.loop.run_turn('s', 'task', 1, 5)
    row, = records(tmp_path)
    assert row['status'] == 'provider_failure'
    assert row['error']['type'] == 'response_failed' and row['error']['origin'] == 'model_client'
    assert 'private remote message' not in json.dumps(row)


def test_incomplete_sidecar_tail_does_not_corrupt_resumed_append(tmp_path: Path) -> None:
    from tests.test_request_telemetry import Events

    with model_endpoint([json_response(message('done'))]) as (url, requests):
        config = configuration(tmp_path, url, False)
        events = Events()
        runtime = build_agent_runtime(config, config.runtime, events)
        path = config.runtime.session_dir / 's.requests.ndjson'
        path.write_bytes(b'{"partial":')
        assert runtime.loop.run_turn('s', 'continue', 1, 5).final_text == 'done'
    assert path.read_bytes() == b'{"partial":'
    assert any(type(event).__name__ == 'RequestLogWriteFailed' for event in events.received)


def test_rejected_core_usage_keeps_legacy_totals_unknown(tmp_path: Path) -> None:
    from gear_agent.errors import GearError

    body = {**message('received'), 'usage': {'input_tokens': -1, 'output_tokens': 3, 'total_tokens': 8}}
    with model_endpoint([json_response(body)]) as (url, requests):
        config = configuration(tmp_path, url, False)
        collector = EvalRunCollector()
        runtime = build_agent_runtime(config, config.runtime, collector, collector)
        with pytest.raises(GearError, match='nonnegative'):
            run_with_artifacts(runtime, 's', TaskPrompt('task', 'inline'), tmp_path / 'run', collector, {})
    model = json.loads((tmp_path / 'run/metrics.json').read_text())['model']
    row, = model['requests']
    assert row['usage'] is None and model['output_tokens'] is model['total_tokens'] is None
    assert row['statistics']['input_tokens']['availability'] == 'invalid'
    assert row['statistics']['output_tokens']['value'] == 3
