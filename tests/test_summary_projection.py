from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.history import build_model_history
from gear_agent.agent.jev import JevClient
from gear_agent.agent.loop import AgentLoop
from gear_agent.compaction_config import CompactionConfig
from gear_agent.context_budget import ContextBudgetConfig
from gear_agent.model.client import ModelClient
from gear_agent.model.replay import reasoning_replay_policy
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.repository import RepositoryContext
from gear_agent.store.memory import MemoryContextStore
from gear_agent.tools.filesystem import FileReadTool
from tests.test_agent_loop import SequencedTransport
from tests.test_context_budget import BudgetSink, response
from tests.test_jev_compaction import JEV, MODEL, JevAPI, Observer, event


class SnapshotTransport(SequencedTransport):
    def post_json(
        self, url: str, headers: dict[str, str], payload: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        return super().post_json(url, headers, copy.deepcopy(payload), timeout_seconds)


def test_manual_summary_omits_replay_copy_but_preserves_checkpoint() -> None:
    result = {'content': 'unique result body ' * 1000, 'resolved_scope_paths': ['.']}
    checkpoint = event('compaction_selective', {'schema': 'gear-agent.selective.v1', 'events': [
        event('user_input', {'text': 'read a file'}),
        event('model_response', {'output': [{'type': 'function_call', 'call_id': 'read',
                                           'name': 'file_read', 'arguments': '{"path":"file.txt"}'}]}),
        event('tool_result', {'call_id': 'read', 'name': 'file_read', 'result': result,
                              'model_visible_output': json.dumps(result, ensure_ascii=False)}),
    ]})
    store = MemoryContextStore()
    store.append('s', checkpoint['kind'], checkpoint['payload'])
    original = copy.deepcopy(store.load('s'))
    transport = SequencedTransport([response('summary')])
    adapter = ResponsesModelAdapter(ModelClient(transport), MODEL)
    before = build_model_history(original, reasoning_replay_policy(MODEL)).items

    CompactionService(adapter).compact('s', store, 30)

    prompt = transport.payloads[0]['input']
    assert prompt.count(result['content']) == 1
    assert 'model_visible_output' not in prompt
    assert store.load('s')[:1] == original
    assert build_model_history(store.load('s')[:1], reasoning_replay_policy(MODEL)).items == before


def test_large_selective_checkpoint_can_fall_back_to_summary_in_same_turn(tmp_path: Path) -> None:
    tool = FileReadTool(tmp_path)
    store = MemoryContextStore()
    store.append('s', 'user_input', {'text': 'read historical files'})
    calls = []
    for index in range(5):
        name = f'old-{index}'
        (tmp_path / name).write_text('historical' * 1000, encoding='utf-8')
        calls.append({'type': 'function_call', 'call_id': name, 'name': 'file_read',
                      'arguments': json.dumps({'path': name})})
    store.append('s', 'model_response', {'output': calls})
    for call in calls:
        store.append('s', 'tool_result', {'call_id': call['call_id'], 'name': 'file_read',
                                         'result': tool.run(json.loads(call['arguments']))})
    store.append('s', 'model_response', response('historical work complete'))
    store.append('s', 'assistant_message', {'text': 'historical work complete'})
    store.append('s', 'user_input', {'text': 'recent request'})
    store.append('s', 'model_response', response('recent work complete'))
    store.append('s', 'assistant_message', {'text': 'recent work complete'})
    first_content, second_content = 'A' * 45_000, 'B' * 20_000
    (tmp_path / 'first.txt').write_text(first_content, encoding='utf-8')
    (tmp_path / 'second.txt').write_text(second_content, encoding='utf-8')
    transport = SnapshotTransport([
        {'output': [{'type': 'function_call', 'call_id': 'first', 'name': 'file_read',
                     'arguments': '{"path":"first.txt"}'}]},
        {'output': [{'type': 'function_call', 'call_id': 'second', 'name': 'file_read',
                     'arguments': '{"path":"second.txt"}'}]},
        response('continuation summary'), response('done'),
    ])
    adapter = ResponsesModelAdapter(ModelClient(transport), MODEL)
    observer = Observer()
    sink = BudgetSink()
    api = JevAPI({f'old-{index}': (0, 0) for index in range(5)})
    loop = AgentLoop(adapter, [tool], store, sink, RepositoryContext(tmp_path),
                     context_budget=ContextBudgetConfig(True, 128_000, 16_000, 70_000),
                     observer=observer, compaction_config=CompactionConfig('jev', 'summary', JEV))

    with patch.object(JevClient, 'evaluate', side_effect=api.evaluate):
        result = loop.run_turn('s', 'read first.txt and second.txt', 3, 30)

    assert result.final_text == 'done'
    assert len(api.requests) == 1
    assert len(transport.payloads) == 4
    summary_prompt = transport.payloads[2]['input']
    assert summary_prompt.count(first_content) == 1
    assert summary_prompt.count(second_content) == 1
    assert 'model_visible_output' not in summary_prompt
    checkpoints = [item for item in store.load('s') if item['kind'].startswith('compaction_')]
    assert [item['kind'] for item in checkpoints] == ['compaction_selective', 'compaction_summary']
    selected = checkpoints[0]
    stored_result = next(item['payload'] for item in selected['payload']['events']
                         if item['kind'] == 'tool_result' and item['payload']['call_id'] == 'first')
    assert first_content in stored_result['model_visible_output']
    assert build_model_history([selected], reasoning_replay_policy(MODEL)).items == transport.payloads[1]['input']
    attempts = [payload for kind, payload in observer.records if kind == 'compaction_strategy']
    assert [attempt['fallback_outcome'] for attempt in attempts] == ['not_needed', 'succeeded']
    summary_budgets = [item.diagnostic for item in sink.budgets() if item.phase == 'compaction']
    assert len(summary_budgets) == 1
    assert summary_budgets[0].total_estimated_request_tokens <= 112_000
