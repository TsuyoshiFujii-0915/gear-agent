from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import pytest

from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.loop import AgentLoop
from gear_agent.errors import GearError
from gear_agent.model.client import ModelClient
from gear_agent.model.events import SilentModelProgressEventSink
from gear_agent.model.responses_adapter import ResponsesModelAdapter, ResponsesModelResponse
from gear_agent.model.transport import HttpxHttpTransport
from gear_agent.repository import RepositoryContext
from gear_agent.store.memory import MemoryContextStore
from gear_agent.tools.filesystem import FileWriteTool
from tests.test_headless import json_response, message, model_endpoint, tool_call
from tests.test_response_completion import ResponseTransport, model_config


def receive(body: dict[str, Any], events: list[dict[str, Any] | str]) -> ResponsesModelResponse:
    transport = ResponseTransport(body, events)
    adapter = ResponsesModelAdapter(ModelClient(transport), model_config(bool(events)))
    return adapter.create_response('input', [], 'instructions', 30, 5, SilentModelProgressEventSink())


@pytest.mark.parametrize(('usage', 'expected'), [
    pytest.param({'input_tokens': 12000, 'output_tokens': 50, 'total_tokens': 12050,
                  'input_tokens_details': {'cached_tokens': 4000, 'cache_write_tokens': 7000},
                  'output_tokens_details': {'reasoning_tokens': 20}},
                 (12000, 50, 12050, 4000, 7000, 20), id='openai'),
    pytest.param({'input_tokens': 17, 'output_tokens': 3, 'total_tokens': 20,
                  'output_tokens_details': {'reasoning_tokens': 0}},
                 (17, 3, 20, None, None, 0), id='azure'),
    pytest.param({'input_tokens': 60, 'output_tokens': 10, 'total_tokens': 70,
                  'input_tokens_details': {'cached_tokens': 40}},
                 (60, 10, 70, 40, None, None), id='lm-studio'),
    pytest.param({'input_tokens': 8, 'output_tokens': 2, 'total_tokens': 10,
                  'input_tokens_details': {'cached_tokens': 3, 'cache_write_tokens': 0},
                  'output_tokens_details': {'reasoning_tokens': 1}},
                 (8, 2, 10, 3, 0, 1), id='maya-v1.1.56'),
    pytest.param({'private_key': 'SECRET'}, (None,) * 6, id='unknown'),
])
def test_standard_usage(usage: dict[str, Any], expected: tuple[int | None, ...]) -> None:
    body = {**message('SECRET'), 'usage': usage, 'stats': {'tokens_per_second': 999}}
    result = receive(body, [])
    s = result.statistics
    assert tuple(m.value for m in (s.input_tokens, s.output_tokens, s.total_tokens,
                                   s.cached_tokens, s.cache_write_tokens, s.reasoning_tokens)) == expected
    if expected[0] is not None:
        assert s.total_tokens.origin == 'server_reported'
        assert s.input_tokens.sources[0].schema == 'responses_usage'
    if s.cached_tokens.value is not None:
        assert s.cache_hit_ratio.value == s.cached_tokens.value / s.input_tokens.value
    assert s.prefill_rate.availability == s.decode_rate.availability == 'missing'
    assert s.prefill_tokens.value is None
    assert 'SECRET' not in json.dumps(asdict(s))


def test_derived_total_does_not_change_legacy_usage_or_history() -> None:
    body = {**message('done'), 'usage': {'input_tokens': 12, 'output_tokens': 5,
                                         'output_tokens_details': {'reasoning_tokens': 3}}}
    result = receive(body, [])
    total = result.statistics.total_tokens
    assert (total.value, total.unit, total.origin) == (17, 'tokens', 'derived')
    assert total.formula == 'input_tokens + output_tokens'
    assert [(f.path, f.value) for f in total.sources] == [
        ('usage.input_tokens', 12), ('usage.output_tokens', 5)]
    assert result.usage.total_tokens is None
    assert result.persisted_payload == body
    assert receive({**body, 'usage': {'output_tokens': 5}}, []).statistics.total_tokens.value is None


@pytest.mark.parametrize(('inputs', 'cached', 'ratio', 'state'), [
    (4, 0, 0.0, 'available'), (4, 4, 1.0, 'available'), (4, 2, 0.5, 'available'),
    (None, 4, None, 'missing'), (0, 0, None, 'not_applicable'), (3, 4, None, 'invalid'),
])
def test_cache_fraction(inputs: int | None, cached: int, ratio: float | None, state: str) -> None:
    usage = {'input_tokens': inputs, 'input_tokens_details': {'cached_tokens': cached}}
    stats = receive({'usage': usage}, []).statistics
    assert (stats.cache_hit_ratio.value, stats.cache_hit_ratio.availability) == (ratio, state)
    if state == 'invalid':
        assert stats.cached_tokens.value is None
        assert stats.cache_hit_ratio.reason == 'exceeds_input_tokens'
    else:
        assert stats.cached_tokens.value == cached
    if state == 'available':
        assert stats.cache_hit_ratio.origin == 'derived'
        assert [s.value for s in stats.cache_hit_ratio.sources] == [cached, inputs]


TIMINGS = {'prompt_n': 20, 'prompt_ms': 250, 'prompt_per_second': 81,
           'predicted_n': 7, 'predicted_ms': 500, 'predicted_per_second': 12,
           'private_extension': 'SECRET'}


@pytest.mark.parametrize('location', ['json', 'event', 'nested'])
def test_strata_timing_locations(location: str) -> None:
    body = {**message('done'), 'status': 'completed', 'usage': {'output_tokens': 8}}
    event = {'type': 'response.completed', 'response': body}
    if location == 'event':
        event['timings'] = TIMINGS
        prefix = 'event.timings'
    else:
        body['timings'] = TIMINGS
        prefix = 'timings' if location == 'json' else 'event.response.timings'
    result = receive(body, [] if location == 'json' else [event])
    stats = result.statistics
    assert result.persisted_payload == body
    assert (stats.prefill_rate.value, stats.decode_rate.value) == (81, 12)
    assert (stats.prefill_tokens.value, stats.decode_tokens.value, stats.output_tokens.value) == (20, 7, 8)
    assert (stats.prefill_seconds.value, stats.decode_seconds.value) == (0.25, 0.5)
    source = stats.prefill_seconds.sources[0]
    assert (source.path, source.value, source.unit) == (prefix + '.prompt_ms', 250, 'ms')
    assert source.schema == 'strata_qwen_timings'
    assert (stats.prefill_seconds.unit, stats.decode_rate.unit) == ('s', 'tokens/s')
    assert stats.decode_rate.origin == 'server_reported'
    assert 'SECRET' not in json.dumps(asdict(stats))


def test_sse_prefers_whole_outer_timings_and_ignores_earlier_snapshots() -> None:
    body = {**message('done'), 'status': 'completed', 'timings': TIMINGS}
    events = [{'type': 'response.created', 'response': {'usage': {'input_tokens': 999}}},
              {'type': 'response.completed', 'response': body, 'timings': {'prompt_n': 9}}]
    stats = receive({}, events).statistics
    assert stats.prefill_tokens.value == 9
    assert stats.prefill_seconds.value is None
    assert stats.input_tokens.value is None


def test_only_engine_counts_and_durations_can_supply_missing_throughput() -> None:
    stats = receive({'usage': {'input_tokens': 999, 'output_tokens': 999},
                     'timings': {'prompt_n': 10, 'prompt_ms': 500, 'predicted_n': 6, 'predicted_ms': 2000}},
                    []).statistics
    assert (stats.prefill_rate.value, stats.decode_rate.value) == (20, 3)
    assert stats.decode_rate.origin == 'derived'
    assert [f.value for f in stats.decode_rate.sources] == [6, 2000]
    assert stats.decode_rate.formula == 'tokens / (ms / 1000)'
    stats = receive({'usage': {'input_tokens': 999, 'output_tokens': 999},
                     'timings': {'prompt_ms': 500, 'predicted_ms': 500}}, []).statistics
    assert stats.prefill_rate.value is stats.decode_rate.value is None


@pytest.mark.parametrize('bad', [True, -1, 'SECRET', float('nan'), float('inf'), [], {}])
def test_invalid_optional_values_are_safe_and_do_not_prevent_success(bad: object) -> None:
    result = receive({**message('ok'), 'usage': {'input_tokens_details': {'cached_tokens': bad}},
                      'timings': {'predicted_per_second': bad, 'prompt_ms': bad,
                                  'predicted_n': 20, 'predicted_ms': 1000}}, [])
    assert result.text == 'ok'
    for metric in (result.statistics.decode_rate, result.statistics.prefill_seconds,
                   result.statistics.cached_tokens):
        assert metric.availability == 'invalid'
        assert metric.value is None
        assert metric.reason is not None
    assert 'SECRET' not in json.dumps(asdict(result.statistics), allow_nan=False)


@pytest.mark.parametrize('timings', [
    {'prompt_n': 0, 'prompt_ms': 10, 'prompt_per_second': 900},
    {'prompt_n': 5, 'prompt_ms': 0},
])
def test_no_speed_for_empty_work_or_zero_duration(timings: dict[str, int]) -> None:
    rate = receive({'timings': timings}, []).statistics.prefill_rate
    assert (rate.value, rate.availability) == (None, 'not_applicable')


def test_unrepresentable_rate_is_invalid_without_losing_counts() -> None:
    result = receive({**message('ok'), 'timings': {'prompt_n': 10**400, 'prompt_ms': 1}}, [])
    assert result.text == 'ok'
    assert result.statistics.prefill_tokens.value == 10**400
    assert result.statistics.prefill_rate.availability == 'invalid'


def test_core_usage_keeps_its_existing_validation() -> None:
    result = receive({'usage': {'total_tokens': False, 'input_tokens': 2, 'output_tokens': 1}}, [])
    with pytest.raises(GearError) as error:
        _ = result.usage
    assert error.value.error_type == 'response_usage_invalid'
    assert result.statistics.total_tokens.availability == 'invalid'


@pytest.mark.parametrize('status', ['incomplete', 'failed'])
@pytest.mark.parametrize('stream', [False, True])
def test_unsuccessful_response_exposes_only_safe_statistics(status: str, stream: bool) -> None:
    body = {**message('SECRET'), 'status': status, 'usage': {'output_tokens': 3},
            'error': {'message': 'SECRET'}, 'id': 'SECRET'}
    event = {'type': 'response.' + status, 'response': body, 'timings': TIMINGS}
    with pytest.raises(GearError) as exc:
        receive({**body, 'timings': TIMINGS}, [event] if stream else [])
    assert exc.value.error_type == 'response_' + status
    assert exc.value.statistics.output_tokens.value == 3
    assert exc.value.statistics.decode_rate.value == 12
    assert exc.value.details == {}
    assert 'SECRET' not in json.dumps(asdict(exc.value))


def test_bad_optional_metadata_allows_a_complete_tool_round_trip(tmp_path: Path) -> None:
    call = tool_call('file_write', {'path': 'result.txt', 'content': 'saved'})
    call.update({'timings': 'malformed', 'usage': {'input_tokens_details': False}})
    with model_endpoint([json_response(call), json_response(message('done'))]) as (url, requests):
        model = replace(model_config(False), url=url)
        adapter = ResponsesModelAdapter(ModelClient(HttpxHttpTransport()), model)
        loop = AgentLoop(adapter, [FileWriteTool(tmp_path)], MemoryContextStore(),
                         SilentAgentLoopEventSink(), RepositoryContext(tmp_path))
        result = loop.run_turn('session', 'save', 2, 30)
    assert result.final_text == 'done'
    assert (tmp_path / 'result.txt').read_text() == 'saved'
    assert len(requests) == 2
    assert any(item.get('type') == 'function_call_output' for item in requests[1]['input'])
