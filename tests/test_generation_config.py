from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path
from typing import Any
import json

import pytest

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.loop import AgentLoop
from gear_agent.cli import run_cli
from gear_agent.compaction_config import CompactionConfig, JevConfig, DEFAULT_JEV_POLICY
from gear_agent.config import AppConfig, DEFAULT_CONFIG_TEXT, ModelConfig, ReasoningReplayMode, load_config
from gear_agent.context_budget import ByteTokenEstimator, ContextBudgetConfig, ContextBudgetManager, ContextRequest
from gear_agent.errors import GearError
from gear_agent.headless import TaskPrompt, describe_run
from gear_agent.model.client import ModelClient
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.model.transport import HttpxHttpTransport
from gear_agent.repository import RepositoryContext
from gear_agent.runtime import build_agent_runtime
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.filesystem import FileReadTool, FileWriteTool
from tests.test_context_budget import BudgetSink
from tests.test_headless import json_response, message, model_endpoint, tool_call


def configured(root: Path, settings: str) -> AppConfig:
    path = root / 'config.toml'
    text = DEFAULT_CONFIG_TEXT.replace('[model]', f'[model]\n{settings}')
    path.write_text(text, encoding='utf-8')
    config = load_config(path, {})
    return replace(config, runtime=replace(config.runtime, workdir=root, session_dir=root / 'sessions'))


def wire_response(response: dict[str, Any], stream: bool) -> tuple[int, str, bytes]:
    if not stream:
        return json_response(response)
    event = {'type': f'response.{response["status"]}', 'response': response}
    return 200, 'text/event-stream', f'data: {json.dumps(event)}\n\n'.encode()


@pytest.mark.parametrize('effort', ['none', 'minimal', 'low', 'medium', 'high', 'xhigh'])
def test_loads_supported_effort_and_positive_cap(tmp_path: Path, effort: str) -> None:
    config = configured(tmp_path, f'reasoning_effort = "{effort}"\nmax_output_tokens = 8192')
    assert config.model.reasoning_effort == effort
    assert config.model.max_output_tokens == 8192
    assert config.model.reasoning_replay is ReasoningReplayMode.NONE


@pytest.mark.parametrize(('key', 'value'), [
    ('reasoning_effort', value) for value in ('', ' ', 'automatic', 'HIGH', 1, True, [], {})
] + [
    ('max_output_tokens', value) for value in (True, False, '8192', 8192.0, 0, -1, [], {})
])
def test_invalid_direct_model_config_fails_early(key: str, value: object) -> None:
    model = ModelConfig('https://example.test/responses', 'local', None, ReasoningReplayMode.NONE)
    with pytest.raises(GearError) as caught:
        replace(model, **{key: value})
    assert caught.value.error_type == 'config_value_invalid'
    assert caught.value.origin == 'config'
    assert caught.value.details['key'] == key
    assert caught.value.details['table'] == 'model'


@pytest.mark.parametrize(('key', 'literal'), [
    ('reasoning_effort', literal) for literal in ('""', '" "', '"automatic"', 'true', '42', '[]', '{}')
] + [
    ('max_output_tokens', literal) for literal in ('true', 'false', '"8192"', '8192.0', '0', '-1', '[]', '{}')
])
def test_invalid_toml_values_fail_early(tmp_path: Path, key: str, literal: str) -> None:
    with pytest.raises(GearError) as caught:
        configured(tmp_path, f'{key} = {literal}')
    assert caught.value.error_type == 'config_value_invalid'
    assert caught.value.details['key'] == key


@pytest.mark.parametrize('direct', [False, True])
def test_auto_budget_rejects_cap_larger_than_reserve(tmp_path: Path, direct: bool) -> None:
    if direct:
        config = configured(tmp_path, 'max_output_tokens = 8192')
        with pytest.raises(GearError) as caught:
            replace(config, context_budget=ContextBudgetConfig(True, 100_000, 8191, None))
    else:
        path = tmp_path / 'config.toml'
        text = DEFAULT_CONFIG_TEXT.replace('[model]', '[model]\nmax_output_tokens = 8192')
        text = text.replace('auto_compaction = false',
                            'auto_compaction = true\ncontext_window_tokens = 100000\nreserved_tokens = 8191')
        path.write_text(text, encoding='utf-8')
        with pytest.raises(GearError) as caught:
            load_config(path, {})
    assert caught.value.error_type == 'config_value_invalid'
    assert caught.value.details['key'] == 'reserved_tokens'
    assert 'max_output_tokens' in caught.value.message


@pytest.mark.parametrize(('automatic', 'reserve', 'cap'), [
    (True, 8192, 8192), (True, 9000, 8192), (True, 0, None), (False, 0, 8192),
])
def test_reserve_boundary_and_unknown_cap_are_explicit(
    tmp_path: Path, automatic: bool, reserve: int, cap: int | None,
) -> None:
    config = configured(tmp_path, '')
    model = replace(config.model, max_output_tokens=cap)
    budget = ContextBudgetConfig(automatic, 100_000, reserve, None)
    result = replace(config, model=model, context_budget=budget)
    assert result.context_budget == budget
    assert result.model.max_output_tokens == cap


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize(('effort', 'cap'), [(None, None), ('none', None), (None, 512), ('medium', 512)])
def test_json_and_sse_send_only_explicit_generation_fields(
    stream: bool, effort: str | None, cap: int | None,
) -> None:
    response = {**message('done'), 'status': 'completed'}
    history = [{'role': 'user', 'content': 'preserve history'}]
    original = deepcopy(history)
    schemas = [{'type': 'function', 'name': 'read', 'parameters': {'type': 'object'}}]
    with model_endpoint([wire_response(response, stream)]) as (url, requests):
        model = ModelConfig(url, 'actual-model-id', 'credential-sentinel', ReasoningReplayMode.ENCRYPTED,
                            stream, reasoning_effort=effort, max_output_tokens=cap)
        assert ModelClient(HttpxHttpTransport()).create_response(model, history, schemas, 'rules', 5, 5) == response
    expected = {'model': model.model, 'input': original, 'instructions': 'rules', 'stream': stream,
                'include': ['reasoning.encrypted_content'], 'tools': schemas, 'tool_choice': 'auto'}
    if effort is not None:
        expected['reasoning'] = {'effort': effort}
    if cap is not None:
        expected['max_output_tokens'] = cap
    assert requests == [expected]
    assert history == original


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('trigger', ['manual', 'automatic', 'jev_fallback'])
@pytest.mark.parametrize('status', ['completed', 'incomplete'])
def test_summaries_inherit_settings_and_never_checkpoint_incomplete(
    tmp_path: Path, stream: bool, trigger: str, status: str,
) -> None:
    config = configured(tmp_path, 'reasoning_effort = "high"\nmax_output_tokens = 512')
    store = JsonlContextStore(tmp_path / 'history')
    store.append('s', 'user_input', {'text': 'old context ' * 600})
    original = store.load('s')
    summary = {**message('short checkpoint'), 'status': status}
    if status == 'incomplete':
        summary['incomplete_details'] = {'reason': 'max_output_tokens'}
    responses = [wire_response(summary, stream), wire_response({**message('done'), 'status': 'completed'}, stream)]
    sink = BudgetSink()
    budget = ContextBudgetConfig(True, 100_000, 1000, 3000)
    with model_endpoint(responses) as (url, requests):
        adapter = ResponsesModelAdapter(ModelClient(HttpxHttpTransport()), replace(config.model, url=url, stream=stream))
        compaction = config.compaction
        if trigger == 'jev_fallback':
            compaction = CompactionConfig('jev', 'summary', JevConfig('jev-1.13.0', 'jev-secret', 10, DEFAULT_JEV_POLICY))
        loop = AgentLoop(adapter, [], store, sink, RepositoryContext(tmp_path),
                         context_budget=budget, compaction_config=compaction)

        def execute() -> str:
            if trigger == 'manual':
                return CompactionService(adapter).compact('s', store, 5, 5)
            return loop.run_turn('s', 'continue', 1, 5, 5).final_text

        if status == 'incomplete':
            with pytest.raises(GearError) as caught:
                execute()
            assert caught.value.error_type == 'response_incomplete'
        else:
            assert execute() == ('short checkpoint' if trigger == 'manual' else 'done')
    assert store.load('s')[:len(original)] == original
    checkpoints = [e for e in store.load('s') if e['kind'] in ('compaction_summary', 'compaction_selective')]
    assert len(checkpoints) == (1 if status == 'completed' else 0)
    assert len(requests) == (2 if status == 'completed' and trigger != 'manual' else 1)
    for request in requests:
        assert request['reasoning'] == {'effort': 'high'}
        assert request['max_output_tokens'] == 512
    assert isinstance(requests[0]['input'], str)
    assert 'tools' not in requests[0]
    if trigger != 'manual':
        for event in sink.budgets():
            if event.phase == 'before':
                continue
            payload = requests[0] if event.phase == 'compaction' else requests[1]
            base = ContextBudgetManager(budget, ByteTokenEstimator()).evaluate_compaction(
                ContextRequest(payload['input'], [], payload['instructions'], payload['instructions'], 0))
            extra = {key: payload[key] for key in ('reasoning', 'max_output_tokens')}
            assert event.diagnostic.request_overhead == base.request_overhead + ByteTokenEstimator().estimate(extra)
            assert event.diagnostic.reserved_headroom == 1000


def test_embedded_loop_rejects_insufficient_reserve_before_dispatch(tmp_path: Path) -> None:
    config = configured(tmp_path, 'max_output_tokens = 512')
    adapter = ResponsesModelAdapter(ModelClient(HttpxHttpTransport()), config.model)
    with pytest.raises(GearError) as caught:
        AgentLoop(adapter, [], JsonlContextStore(tmp_path / 'history'), BudgetSink(), RepositoryContext(tmp_path),
                  context_budget=ContextBudgetConfig(True, 100_000, 511, None))
    assert caught.value.error_type == 'config_value_invalid'


@pytest.mark.parametrize('stream', [False, True])
def test_tool_round_trip_preserves_replay_and_accounts_for_generation_fields(tmp_path: Path, stream: bool) -> None:
    config = configured(tmp_path, 'reasoning_effort = "medium"\nmax_output_tokens = 512')
    (tmp_path / 'data.txt').write_text('file contents', encoding='utf-8')
    first = {**tool_call('file_read', {'path': 'data.txt'}), 'status': 'completed'}
    first['output'].insert(0, {'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'opaque-state'})
    final = {**message('done'), 'status': 'completed'}
    budget = ContextBudgetConfig(True, 100_000, 1000, None)
    sink = BudgetSink()
    with model_endpoint([wire_response(first, stream), wire_response(final, stream)]) as (url, requests):
        adapter = ResponsesModelAdapter(ModelClient(HttpxHttpTransport()), replace(
            config.model, url=url, stream=stream, reasoning_replay=ReasoningReplayMode.ENCRYPTED))
        store = JsonlContextStore(tmp_path / 'history')
        loop = AgentLoop(adapter, [FileReadTool(tmp_path)], store, sink, RepositoryContext(tmp_path), context_budget=budget)
        assert loop.run_turn('s', 'read data', 2, 5, 5).final_text == 'done'
    assert len(requests) == 2
    assert requests[1]['input'][0] == {'role': 'user', 'content': 'read data'}
    assert 'opaque-state' in json.dumps(requests[1]['input'])
    assert 'file contents' in json.dumps(requests[1]['input'])
    for payload, event in zip(requests, sink.budgets(), strict=True):
        assert payload['reasoning'] == {'effort': 'medium'}
        assert payload['max_output_tokens'] == 512
        base = ContextBudgetManager(budget, ByteTokenEstimator()).evaluate(ContextRequest(
            payload['input'], payload['tools'], payload['instructions'], payload['instructions'], 0))
        extra = {key: payload[key] for key in ('reasoning', 'max_output_tokens')}
        assert event.diagnostic.total_estimated_request_tokens == base.total_estimated_request_tokens + ByteTokenEstimator().estimate(extra)
        assert event.diagnostic.reserved_headroom == 1000


@pytest.mark.parametrize('stream', [False, True])
def test_capped_incomplete_tool_call_does_not_write(tmp_path: Path, stream: bool) -> None:
    config = configured(tmp_path, 'max_output_tokens = 512')
    target = tmp_path / 'target.txt'
    target.write_text('original', encoding='utf-8')
    partial = {**tool_call('file_write', {'path': 'target.txt', 'content': 'changed'}),
               'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'}}
    with model_endpoint([wire_response(partial, stream)]) as (url, requests):
        adapter = ResponsesModelAdapter(ModelClient(HttpxHttpTransport()), replace(config.model, url=url, stream=stream))
        store = JsonlContextStore(tmp_path / 'history')
        loop = AgentLoop(adapter, [FileWriteTool(tmp_path)], store, SilentAgentLoopEventSink(), RepositoryContext(tmp_path))
        with pytest.raises(GearError, match='incomplete'):
            loop.run_turn('s', 'write target', 2, 5, 5)
    assert len(requests) == 1
    assert target.read_text() == 'original'
    assert [event['kind'] for event in store.load('s')] == ['user_input']


@pytest.mark.parametrize('settings', ['', 'reasoning_effort = "medium"\nmax_output_tokens = 512'])
def test_effective_config_and_run_json_record_explicit_settings_without_credentials(tmp_path: Path, settings: str) -> None:
    config = configured(tmp_path, settings)
    config = replace(config, model=replace(config.model, api_key='credential-sentinel'))
    agent = build_agent_runtime(config, config.runtime, SilentAgentLoopEventSink())
    spec = describe_run(agent, 's', TaskPrompt('task', 'inline'))
    assert spec.model['reasoning_effort'] == ('medium' if settings else None)
    assert spec.model['max_output_tokens'] == (512 if settings else None)
    assert 'credential-sentinel' not in json.dumps(asdict(spec))
    destination = tmp_path / 'run'
    with model_endpoint([json_response({**message('done'), 'status': 'completed'})]) as (url, requests):
        path = tmp_path / 'config.toml'
        text = path.read_text().replace('http://localhost:1234/v1/responses', url)
        text = text.replace('api_key_env = ""', 'api_key_env = "MODEL_TEST_KEY"')
        text = text.replace('workdir = "."', f'workdir = {json.dumps(str(tmp_path))}')
        text = text.replace('session_dir = ".gear/sessions"', f'session_dir = {json.dumps(str(tmp_path / "sessions"))}')
        path.write_text(text, encoding='utf-8')
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = run_cli(['--config', str(path), 'run', '--prompt', 'task', '--run-dir', str(destination)],
                           {'HOME': str(tmp_path), 'MODEL_TEST_KEY': 'credential-sentinel'})
    assert code == 0
    manifest_text = (destination / 'run.json').read_text()
    model = json.loads(manifest_text)['model']
    assert model['reasoning_effort'] == spec.model['reasoning_effort']
    assert model['max_output_tokens'] == spec.model['max_output_tokens']
    assert 'credential-sentinel' not in manifest_text
    if settings:
        assert requests[0]['reasoning']['effort'] == model['reasoning_effort']
        assert requests[0]['max_output_tokens'] == model['max_output_tokens']
    else:
        assert 'reasoning' not in requests[0]
        assert 'max_output_tokens' not in requests[0]


@pytest.mark.parametrize('stream', [False, True])
def test_unsupported_generation_settings_return_safe_actionable_error_without_retry(tmp_path: Path, stream: bool) -> None:
    config = configured(tmp_path, 'reasoning_effort = "xhigh"\nmax_output_tokens = 512')
    body = b'{"error":{"message":"unsupported reasoning credential-sentinel private-prompt"}}'
    with model_endpoint([(400, 'application/json', body)]) as (url, requests):
        model = replace(config.model, url=f'{url}?token=url-secret', stream=stream, api_key='credential-sentinel')
        with pytest.raises(GearError) as caught:
            ModelClient(HttpxHttpTransport()).create_response(model, 'private-prompt', [], '', 5, 5)
    assert len(requests) == 1
    assert requests[0]['reasoning'] == {'effort': 'xhigh'}
    assert requests[0]['max_output_tokens'] == 512
    assert caught.value.error_type == 'http_status_error'
    assert caught.value.details['status'] == 400
    assert 'model.reasoning_effort' in caught.value.message
    assert 'model.max_output_tokens' in caught.value.message
    serialized = json.dumps(asdict(caught.value))
    for secret in ('credential-sentinel', 'url-secret', 'private-prompt', url):
        assert secret not in serialized
