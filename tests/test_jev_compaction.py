from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.history import build_model_history, select_effective_events
from gear_agent.agent.jev import JevAnswers, JevClient, JevCompactionStrategy
from gear_agent.agent.loop import AgentLoop
from gear_agent.compaction_config import CompactionConfig, JevConfig, JevPolicy
from gear_agent.config import DEFAULT_CONFIG_TEXT, ModelConfig, ReasoningReplayMode, load_config
from gear_agent.context_budget import ContextBudgetConfig
from gear_agent.errors import GearError
from gear_agent.model.client import ModelClient
from gear_agent.model.replay import model_response_event_payload, reasoning_replay_policy
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.repository import RepositoryContext
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.store.memory import MemoryContextStore
from tests.test_agent_loop import SequencedTransport
from tests.test_context_budget import BudgetSink, response


MODEL = ModelConfig('https://example.test/responses', 'test-model', None, ReasoningReplayMode.ENCRYPTED)
POLICY = JevPolicy(0.2, 1, 160, 25_000, 30_000, 32)
JEV = JevConfig('jev-1.13.0', 'private-jev-key', 10, POLICY)


def event(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {'kind': kind, 'payload': payload}


def history() -> list[dict[str, Any]]:
    calls = [{'type': 'function_call', 'call_id': name, 'name': tool, 'arguments': '{"path":"src/example.py"}'}
             for name, tool in [('keep', 'file_read'), ('truncate', 'grep'), ('drop', 'glob'),
                                ('write', 'file_write'), ('shell', 'shell'), ('patch', 'apply_patch'),
                                ('error', 'file_read')]]
    output = [{'type': 'reasoning', 'summary': [], 'encrypted_content': 'opaque-secret'}, *calls,
              {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'checking'}]}]
    events = [event('user_input', {'text': 'Preserve the public API while fixing the bug.'}),
              event('model_response', model_response_event_payload({'output': output}, reasoning_replay_policy(MODEL)))]
    for call in calls:
        events.append(event('tool_call', {'call_id': call['call_id'], 'name': call['name'], 'arguments': {}}))
        result = {'error': {'message': 'recoverable failure'}} if call['call_id'] == 'error' else {'content': 'RESULT-BODY-' * 600}
        events.append(event('tool_result', {'call_id': call['call_id'], 'name': call['name'], 'result': result}))
    events.extend([event('assistant_message', {'text': 'checking'}),
                   event('user_input', {'text': 'recent task'}),
                   event('model_response', {'output': [{'type': 'function_call', 'call_id': 'recent',
                                                        'name': 'file_read', 'arguments': '{}'}]}),
                   event('tool_result', {'call_id': 'recent', 'result': {'content': 'recent body'}}),
                   event('assistant_message', {'text': 'recent done'}),
                   event('user_input', {'text': 'current task'})])
    return events


class JevAPI:
    def __init__(self, scores: dict[str, tuple[float, float]]) -> None:
        self.scores = scores
        self.requests: list[tuple[str, dict[str, str]]] = []

    def evaluate(self, state: str, questions: dict[str, str]) -> JevAnswers:
        self.requests.append((state, questions))
        answers: dict[str, float] = {}
        for key in questions:
            call_id, part = key.rsplit(':', 1)
            pair = self.scores[call_id]
            answers[key] = pair[0 if part == 'call' else 1]
        return JevAnswers(answers, {'input_tokens': 42, 'output_tokens': 0})


def strategy(api: JevAPI, policy: JevPolicy) -> JevCompactionStrategy:
    return JevCompactionStrategy(replace(JEV, policy=policy), api)


def call_ids(items: list[object]) -> list[str]:
    return [item['call_id'] for item in items if isinstance(item, dict) and item.get('type') == 'function_call']


def test_semantic_pairs_multicall_decisions_and_protection() -> None:
    source = history()
    original = copy.deepcopy(source)
    api = JevAPI({'keep': (0.0, 1.0), 'truncate': (1.0, 0.0), 'drop': (0.0, 0.0)})
    candidate = strategy(api, POLICY).compact(source, 30, None)
    selected = select_effective_events([event(candidate.kind, candidate.payload)])
    items = build_model_history(selected, reasoning_replay_policy(MODEL)).items
    assert call_ids(items) == ['keep', 'truncate', 'write', 'shell', 'patch', 'error', 'recent']
    results = {item['call_id']: item['output'] for item in items if isinstance(item, dict) and item.get('type') == 'function_call_output'}
    assert 'RESULT-BODY-' * 50 in results['keep']
    assert len(results['truncate']) < 400
    assert 'truncated' in results['truncate']
    assert 'recoverable failure' in results['error']
    assert sum(item.get('type') == 'message' for item in items if isinstance(item, dict)) == 1
    assert source == original
    assert candidate.metrics['eligible_interactions'] == 3
    assert candidate.metrics['pinned_interactions'] == 5
    assert candidate.metrics['dropped_pairs'] == 1
    assert candidate.metrics['truncated_results'] == 1
    state = api.requests[0][0]
    assert 'current task' in state and 'Preserve the public API' in state
    assert 'RESULT-BODY-' not in state and 'opaque-secret' not in state
    assert 'private-jev-key' not in json.dumps(candidate.payload)
    assert 'private-jev-key' not in json.dumps(candidate.metrics)


def test_checkpoint_resume_repeat_and_scope(tmp_path: Path) -> None:
    store = JsonlContextStore(tmp_path)
    for item in history():
        store.append('s', item['kind'], item['payload'])
    original_bytes = (tmp_path / 's.jsonl').read_bytes()
    api = JevAPI({'keep': (0.0, 1.0), 'truncate': (1.0, 0.0), 'drop': (0.0, 0.0)})
    first = strategy(api, POLICY).compact(store.load('s'), 30, None)
    store.append('s', first.kind, first.payload)
    assert (tmp_path / 's.jsonl').read_bytes().startswith(original_bytes)
    resumed = JsonlContextStore(tmp_path)
    items = build_model_history(resumed.load('s'), reasoning_replay_policy(MODEL)).items
    assert 'drop' not in call_ids(items)
    assert 'opaque-secret' in json.dumps(items)
    other = replace(MODEL, model='other-model')
    assert 'opaque-secret' not in json.dumps(build_model_history(resumed.load('s'), reasoning_replay_policy(other)).items)
    second = strategy(api, POLICY).compact(resumed.load('s'), 30, None)
    resumed.append('s', second.kind, second.payload)
    assert 'drop' not in call_ids(build_model_history(resumed.load('s'), reasoning_replay_policy(MODEL)).items)
    transport = SequencedTransport([response('summary')])
    CompactionService(ResponsesModelAdapter(ModelClient(transport), MODEL)).compact('s', resumed, 30)
    assert '"call_id": "drop"' not in transport.payloads[0]['input']
    assert 'opaque-secret' not in transport.payloads[0]['input']
    assert len(select_effective_events(resumed.load('s'))) == 1


def test_current_chain_and_terminal_history_are_pinned() -> None:
    source = history()
    source.extend([event('model_response', {'output': [{'type': 'function_call', 'call_id': 'unfinished', 'name': 'file_read', 'arguments': '{}'}]}),
                   event('turn_error', {'message': 'terminal failure'})])
    api = JevAPI({'keep': (1, 1), 'truncate': (1, 1), 'drop': (1, 1)})
    candidate = strategy(api, POLICY).compact(source, 30, None)
    selected = select_effective_events([event(candidate.kind, candidate.payload)])
    assert selected[-2:] == source[-2:]
    assert 'unfinished:call' not in api.requests[0][1]


def test_batches_and_independent_limits() -> None:
    api = JevAPI({'keep': (1, 1), 'truncate': (1, 1), 'drop': (1, 1)})
    candidate = strategy(api, replace(POLICY, batch_size=1)).compact(history(), 30, None)
    assert len(api.requests) == 3
    assert len({state for state, _ in api.requests}) == 1
    assert candidate.metrics['jev_requests'] == 3
    for changed in [replace(POLICY, max_state_tokens=10), replace(POLICY, max_request_tokens=10)]:
        with pytest.raises(GearError):
            strategy(api, changed).compact(history(), 30, None)


@pytest.mark.parametrize('scores', [{}, {'keep:call': float('nan')}, {'keep:call': -1}, {'keep:call': True}])
def test_invalid_answers_leave_source_unchanged(scores: dict[str, float]) -> None:
    source = history()
    original = copy.deepcopy(source)
    with patch.object(JevClient, 'evaluate', return_value=JevAnswers(scores, {})):
        with pytest.raises(GearError):
            JevCompactionStrategy(JEV, JevClient(JEV)).compact(source, 30, None)
    assert source == original


def test_invalid_checkpoint_rejected() -> None:
    with pytest.raises(GearError):
        build_model_history([event('compaction_selective', {'schema': 'unknown', 'events': []})], reasoning_replay_policy(MODEL))
    with pytest.raises(GearError):
        build_model_history([event('compaction_selective', {'schema': 'gear-agent.selective.v1', 'events': [event('tool_result', {'call_id': 'orphan', 'result': {}})]})], reasoning_replay_policy(MODEL))


class Observer:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any]]] = []

    def record(self, kind: str, payload: dict[str, Any]) -> None:
        self.records.append((kind, copy.deepcopy(payload)))


def seed(store: MemoryContextStore) -> None:
    source = history()
    for item in source[:-1]:
        if item['kind'] == 'model_response':
            payload = item['payload']
            raw = payload.get('response', payload)
            raw['output'] = [x for x in raw['output'] if x.get('call_id') not in {'write', 'shell', 'patch', 'error'}]
        if item['kind'] in {'tool_call', 'tool_result'} and item['payload']['call_id'] in {'write', 'shell', 'patch', 'error'}:
            continue
        store.append('s', item['kind'], item['payload'])


def make_loop(root: Path, store: MemoryContextStore, transport: SequencedTransport, observer: Observer, fallback: str) -> AgentLoop:
    return AgentLoop(ResponsesModelAdapter(ModelClient(transport), MODEL), [], store, BudgetSink(),
                     RepositoryContext(root), context_budget=ContextBudgetConfig(True, 200_000, 1000, 4000),
                     observer=observer, compaction_config=CompactionConfig('jev', fallback, JEV))


def test_automatic_commit_rechecks_budget_and_user_once(tmp_path: Path) -> None:
    store = MemoryContextStore()
    seed(store)
    transport = SequencedTransport([response('done')])
    observer = Observer()
    api = JevAPI({'keep': (0, 0), 'truncate': (0, 0), 'drop': (0, 0)})
    with patch.object(JevClient, 'evaluate', side_effect=api.evaluate):
        result = make_loop(tmp_path, store, transport, observer, 'none').run_turn('s', 'current task', 1, 30)
    assert result.final_text == 'done'
    assert len(transport.payloads) == 1
    items = transport.payloads[0]['input']
    assert sum(item.get('content') == 'current task' for item in items) == 1
    assert items[-1] == {'role': 'user', 'content': 'current task'}
    assert len([e for e in store.load('s') if e['kind'] == 'compaction_selective']) == 1
    metrics = [p for k, p in observer.records if k == 'compaction_strategy'][-1]
    assert metrics['estimated_input_after'] < metrics['estimated_input_before']
    assert metrics['reduction_ratio'] > 0
    assert 'RESULT-BODY' not in json.dumps(metrics)


@pytest.mark.parametrize('failure', ['timeout', 'invalid', 'insufficient'])
@pytest.mark.parametrize('fallback', ['none', 'summary'])
def test_automatic_failures_are_atomic_with_explicit_fallback(tmp_path: Path, failure: str, fallback: str) -> None:
    store = MemoryContextStore()
    seed(store)
    original = copy.deepcopy(store.load('s'))
    transport = SequencedTransport([response('small summary'), response('done')])
    observer = Observer()
    api = JevAPI({'keep': (1, 1), 'truncate': (1, 1), 'drop': (1, 1)})
    effect = TimeoutError('private-jev-key remote body') if failure == 'timeout' else api.evaluate
    value = JevAnswers({}, {})
    kwargs: dict[str, Any] = {'return_value': value} if failure == 'invalid' else {'side_effect': effect}
    with patch.object(JevClient, 'evaluate', **kwargs):
        loop = make_loop(tmp_path, store, transport, observer, fallback)
        if fallback == 'none':
            with pytest.raises(GearError):
                loop.run_turn('s', 'current task', 1, 30)
            assert not transport.payloads
        else:
            assert loop.run_turn('s', 'current task', 1, 30).final_text == 'done'
            assert 'RESULT-BODY-' in transport.payloads[0]['input']
            assert sum(x.get('content') == 'current task' for x in transport.payloads[1]['input']) == 1
    assert store.load('s')[:len(original)] == original
    assert not any(e['kind'] == 'compaction_selective' for e in store.load('s'))
    metrics = [p for k, p in observer.records if k == 'compaction_strategy'][-1]
    assert metrics['fallback_outcome'] == ('succeeded' if fallback == 'summary' else 'disabled')
    assert metrics['fallback_reason']
    assert 'private-jev-key' not in json.dumps(observer.records)


def test_configuration_opt_in_and_validation(tmp_path: Path) -> None:
    path = tmp_path / 'config.toml'
    path.write_text(DEFAULT_CONFIG_TEXT)
    assert load_config(path, {}).compaction.strategy == 'summary'
    section = '\n[compaction]\nstrategy="jev"\nfallback="summary"\n[compaction.jev]\nmodel="jev-1.13.0"\napi_key_env="CUSTOM_AUTH"\n'
    path.write_text(DEFAULT_CONFIG_TEXT + section)
    config = load_config(path, {'CUSTOM_AUTH': 'private-jev-key'})
    assert config.compaction.jev.model == 'jev-1.13.0'
    assert 'private-jev-key' not in repr(config.compaction)
    from gear_agent.artifact_privacy import ArtifactPrivacy
    assert ArtifactPrivacy(config, {}).text('private-jev-key') == '[REDACTED]'
    for invalid in [section.replace('jev-1.13.0', 'jev-latest'), section.replace('fallback="summary"', 'fallback="magic"'),
                    section + 'keep_threshold=1.5\n', section + 'preserve_recent_turns=-1\n', section + 'batch_size=true\n']:
        path.write_text(DEFAULT_CONFIG_TEXT + invalid)
        with pytest.raises(GearError):
            load_config(path, {'CUSTOM_AUTH': 'private-jev-key'})
    path.write_text(DEFAULT_CONFIG_TEXT + section)
    with pytest.raises(GearError):
        load_config(path, {})


def test_run_metrics_include_selective_checkpoints_and_observations(tmp_path: Path) -> None:
    from gear_agent.artifact_privacy import ArtifactPrivacy
    from gear_agent.run_collector import EvalRunCollector
    path = tmp_path / 'config.toml'
    path.write_text(DEFAULT_CONFIG_TEXT)
    collector = EvalRunCollector()
    collector.start()
    collector.record('compaction_strategy', {'strategy': 'jev', 'dropped_pairs': 3, 'jev_requests': 1})
    metrics = collector.metrics('succeeded', None, True, [event('compaction_selective', {'trigger': 'automatic'})],
                                ArtifactPrivacy(load_config(path, {}), {}))
    assert metrics['context']['automatic_compactions'] == 1
    assert metrics['context']['strategies'][0]['dropped_pairs'] == 3
    assert metrics['model']['request_count'] == 0


def test_sdk_pins_model_validates_and_hides_provider_error() -> None:
    from types import SimpleNamespace
    from typesafe_sdk import TypeSafeError
    reply = SimpleNamespace(model=JEV.model, answers={'x': object()}, nouls={'x': SimpleNamespace(noul=0.3)},
                            usage=SimpleNamespace(input_tokens=20, output_tokens=0))
    with patch('gear_agent.agent.jev.TypeSafeClient') as factory:
        remote = factory.return_value.__enter__.return_value
        remote.system_one.return_value = reply
        answer = JevClient(JEV).evaluate('task', {'x': 'keep?'})
        assert answer.scores == {'x': 0.3}
        assert factory.call_args.kwargs['model'] == 'jev-1.13.0'
        assert factory.call_args.kwargs['retry'].max_retries == 0
        assert remote.system_one.call_args.kwargs['model'] == 'jev-1.13.0'
        remote.system_one.side_effect = TypeSafeError('private-jev-key remote contents')
        with pytest.raises(GearError) as caught:
            JevClient(JEV).evaluate('task', {'x': 'keep?'})
        assert 'private-jev-key' not in str(caught.value)
        assert 'remote contents' not in str(caught.value)


def test_active_chain_is_preserved_verbatim_and_resume_matches_dispatch(tmp_path: Path) -> None:
    from gear_agent.tools.filesystem import FileReadTool
    from gear_agent.context_budget import ByteTokenEstimator, ContextBudgetManager, ContextRequest
    from gear_agent.agent.loop import AGENT_INSTRUCTIONS
    store = MemoryContextStore()
    source = history()[:-1]
    for item in source:
        if item['kind'] == 'tool_result':
            item['payload']['name'] = item['payload'].get('name', 'file_read')
            item['payload']['result']['resolved_scope_paths'] = ['.']
        store.append('s', item['kind'], item['payload'])
    tool = FileReadTool(tmp_path)
    (tmp_path / 'large.txt').write_text('current output\n' * 1000)
    request_call = {'type': 'function_call', 'call_id': 'active', 'name': 'file_read',
                    'arguments': json.dumps({'path': 'large.txt'})}
    transport = SequencedTransport([{'output': [
        {'type': 'reasoning', 'summary': [], 'encrypted_content': 'active-opaque'}, request_call]}, response('done')])
    adapter = ResponsesModelAdapter(ModelClient(transport), MODEL)
    initial = adapter.prepare_history(store.load('s'), 'current task')
    probe = ContextRequest(initial.items, [tool.schema()], AGENT_INSTRUCTIONS, AGENT_INSTRUCTIONS, len(initial.items) - 1)
    unlimited = ContextBudgetManager(ContextBudgetConfig(False, None, 0, None), ByteTokenEstimator())
    limit = unlimited.evaluate(probe).total_estimated_request_tokens + 500
    api = JevAPI({'keep': (0, 0), 'truncate': (0, 0), 'drop': (0, 0)})
    loop = AgentLoop(adapter, [tool], store, BudgetSink(), RepositoryContext(tmp_path),
                     context_budget=ContextBudgetConfig(True, 200_000, 1000, limit),
                     compaction_config=CompactionConfig('jev', 'none', JEV))
    with patch.object(JevClient, 'evaluate', side_effect=api.evaluate):
        assert loop.run_turn('s', 'current task', 2, 30).final_text == 'done'
    sent = transport.payloads[1]['input']
    output = next(item for item in sent if item.get('type') == 'function_call_output' and item['call_id'] == 'active')
    assert 'current output\n' * 1000 == json.loads(output['output'])['content']
    assert 'active-opaque' in json.dumps(sent)
    assert 'active-opaque' not in api.requests[0][0]
    checkpoint_index = next(i for i, item in enumerate(store.load('s')) if item['kind'] == 'compaction_selective')
    resumed = build_model_history(store.load('s')[:checkpoint_index + 1], reasoning_replay_policy(MODEL)).items
    assert resumed == sent


def test_failed_later_batch_and_duplicate_pairs_never_change_history() -> None:
    source = history()
    original = copy.deepcopy(source)
    api = JevAPI({'keep': (0, 0), 'truncate': (0, 0), 'drop': (0, 0)})
    first = api.evaluate
    calls = 0

    def fail_second(state: str, questions: dict[str, str]) -> JevAnswers:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError('remote data')
        return first(state, questions)

    with patch.object(JevClient, 'evaluate', side_effect=fail_second):
        with pytest.raises(GearError) as caught:
            config = replace(JEV, policy=replace(POLICY, batch_size=1))
            JevCompactionStrategy(config, JevClient(config)).compact(source, 30, None)
    assert calls == 2
    assert caught.value.details['metrics']['jev_requests'] == 2
    assert source == original
    duplicate = copy.deepcopy(source)
    duplicate.insert(-1, next(e for e in duplicate if e['kind'] == 'tool_result'))
    with pytest.raises(GearError):
        strategy(api, POLICY).compact(duplicate, 30, None)


def test_jev_credential_is_registered_for_headless_diagnostics(tmp_path: Path) -> None:
    from gear_agent.headless import diagnostic_secrets
    path = tmp_path / 'config.toml'
    path.write_text(DEFAULT_CONFIG_TEXT)
    config = replace(load_config(path, {}), compaction=CompactionConfig('jev', 'none', JEV))
    assert 'private-jev-key' in diagnostic_secrets(config)


def test_default_summary_reports_strategy_metrics(tmp_path: Path) -> None:
    store = MemoryContextStore()
    store.append('s', 'user_input', {'text': 'old context ' * 600})
    transport = SequencedTransport([response('summary'), response('done')])
    observer = Observer()
    loop = AgentLoop(ResponsesModelAdapter(ModelClient(transport), MODEL), [], store, BudgetSink(),
                     RepositoryContext(tmp_path), context_budget=ContextBudgetConfig(True, 100_000, 1000, 3000),
                     observer=observer)
    loop.run_turn('s', 'current', 1, 30)
    metrics = [p for k, p in observer.records if k == 'compaction_strategy']
    assert len(metrics) == 1
    assert metrics[0]['strategy'] == 'summary'
    assert metrics[0]['estimated_input_after'] < metrics[0]['estimated_input_before']


def test_run_spec_records_policy_without_credentials(tmp_path: Path) -> None:
    from dataclasses import asdict
    from gear_agent.headless import TaskPrompt, describe_run
    from gear_agent.runtime import build_agent_runtime
    path = tmp_path / 'config.toml'
    path.write_text(DEFAULT_CONFIG_TEXT)
    config = load_config(path, {})
    config = replace(config, compaction=CompactionConfig('jev', 'summary', JEV),
                     runtime=replace(config.runtime, workdir=tmp_path, session_dir=tmp_path / 'sessions'))
    agent = build_agent_runtime(config, config.runtime, BudgetSink())
    spec = asdict(describe_run(agent, 's', TaskPrompt('task', 'inline')))
    assert spec['compaction']['strategy'] == 'jev'
    assert spec['compaction']['jev']['policy']['keep_threshold'] == 0.2
    assert 'private-jev-key' not in json.dumps(spec)


def test_finalization_retry_survives_selective_checkpoint(tmp_path: Path) -> None:
    from gear_agent.context_budget import ByteTokenEstimator, ContextBudgetManager, ContextRequest
    from gear_agent.agent.loop import AGENT_INSTRUCTIONS, FINALIZATION_RETRY_INSTRUCTION
    store = MemoryContextStore()
    for item in history()[:-1]:
        if item['kind'] == 'tool_result':
            item['payload']['name'] = item['payload'].get('name', 'file_read')
            item['payload']['result']['resolved_scope_paths'] = ['.']
        store.append('s', item['kind'], item['payload'])
    transport = SequencedTransport([{'output': [
        {'type': 'reasoning', 'summary': [], 'encrypted_content': 'retry-opaque' * 300}]}, response('done')])
    adapter = ResponsesModelAdapter(ModelClient(transport), MODEL)
    initial = adapter.prepare_history(store.load('s'), 'current task')
    probe = ContextRequest(initial.items, [], AGENT_INSTRUCTIONS, AGENT_INSTRUCTIONS, len(initial.items) - 1)
    manager = ContextBudgetManager(ContextBudgetConfig(False, None, 0, None), ByteTokenEstimator())
    limit = manager.evaluate(probe).total_estimated_request_tokens + 500
    api = JevAPI({'keep': (0, 0), 'truncate': (0, 0), 'drop': (0, 0)})
    loop = AgentLoop(adapter, [], store, BudgetSink(), RepositoryContext(tmp_path),
                     context_budget=ContextBudgetConfig(True, 200_000, 1000, limit),
                     compaction_config=CompactionConfig('jev', 'none', JEV))
    with patch.object(JevClient, 'evaluate', side_effect=api.evaluate):
        loop.run_turn('s', 'current task', 2, 30)
    sent = transport.payloads[-1]['input']
    assert sent[-1] == {'role': 'user', 'content': FINALIZATION_RETRY_INSTRUCTION}
    checkpoint_index = next(i for i, item in enumerate(store.load('s')) if item['kind'] == 'compaction_selective')
    assert build_model_history(store.load('s')[:checkpoint_index + 1], reasoning_replay_policy(MODEL)).items == sent


def test_selective_checkpoint_rejects_empty_embedded_summary() -> None:
    checkpoint = event('compaction_selective', {'schema': 'gear-agent.selective.v1', 'events': [
        event('compaction_summary', {'text': '   '})]})
    with pytest.raises(GearError):
        build_model_history([checkpoint], reasoning_replay_policy(MODEL))


def test_result_in_current_turn_protects_its_historical_call() -> None:
    source = history()
    moved = next(e for e in source if e['kind'] == 'tool_result' and e['payload']['call_id'] == 'drop')
    source.remove(moved)
    source.append(moved)
    api = JevAPI({'keep': (1, 1), 'truncate': (1, 1)})
    candidate = strategy(api, POLICY).compact(source, 30, None)
    assert 'drop' in call_ids(build_model_history([event(candidate.kind, candidate.payload)], reasoning_replay_policy(MODEL)).items)
    assert all('drop:call' not in questions for _, questions in api.requests)


def test_fallback_over_capacity_reports_failure_without_dispatch(tmp_path: Path) -> None:
    store = MemoryContextStore()
    store.append('s', 'user_input', {'text': 'too much text ' * 2000})
    transport = SequencedTransport([])
    observer = Observer()
    loop = AgentLoop(ResponsesModelAdapter(ModelClient(transport), MODEL), [], store, BudgetSink(),
                     RepositoryContext(tmp_path), context_budget=ContextBudgetConfig(True, 5000, 1000, 3000),
                     observer=observer, compaction_config=CompactionConfig('jev', 'summary', JEV))
    with patch.object(JevClient, 'evaluate') as remote:
        with pytest.raises(GearError) as caught:
            loop.run_turn('s', 'current task', 1, 30)
    assert caught.value.error_type == 'context_budget_exceeded'
    assert caught.value.details['phase'] == 'compaction'
    remote.assert_not_called()
    assert not transport.payloads
    assert not any(e['kind'].startswith('compaction_') for e in store.load('s'))
    metrics = [p for k, p in observer.records if k == 'compaction_strategy'][-1]
    assert metrics['fallback_reason'] == 'jev_no_eligible_interactions'
    assert metrics['fallback_outcome'] == 'failed'
