from pathlib import Path
import json
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import pytest

from examples.compaction_comparison import CURRENT, FIXTURES, TURNS, checkpoint_input, interaction_changes, ObservationLog, run
from gear_agent.agent.compaction_strategy import CompactionCandidate
from gear_agent.agent.jev import JevClient
from gear_agent.config import DEFAULT_CONFIG_TEXT, ModelConfig, ReasoningReplayMode
from gear_agent.model.factory import build_model_adapter
from gear_agent.model.transport import HttpxHttpTransport
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.filesystem import FileReadTool
from tests.test_context_budget import response
from tests.test_jev_compaction import JevAPI


def test_summary_and_selective_continue_with_one_current_message() -> None:
    adapter = build_model_adapter(ModelConfig('http://localhost/v1/responses', 'test', None,
                                             ReasoningReplayMode.NONE))
    current = 'Continue fixing the retry limit.'
    summary = CompactionCandidate('compaction_summary', {'text': 'Inspected retry.py'}, {})
    selective = CompactionCandidate('compaction_selective', {
        'schema': 'gear-agent.selective.v1',
        'events': [{'kind': 'user_input', 'payload': {'text': current}}],
    }, {})
    for candidate in (summary, selective):
        items = checkpoint_input(adapter, candidate, current)
        assert sum(item.get('content') == current for item in items) == 1
    assert 'Inspected retry.py' in checkpoint_input(adapter, summary, current)[0]['content']


def test_comparison_distinguishes_preserved_shortened_and_removed_outputs() -> None:
    before: list[dict[str, Any]] = []
    for call_id in ('keep', 'shorten', 'drop'):
        before.extend([
            {'type': 'function_call', 'call_id': call_id, 'name': 'file_read',
             'arguments': '{"path":"' + call_id + '.txt"}'},
            {'type': 'function_call_output', 'call_id': call_id, 'output': 'full result'},
        ])
    after = [dict(item) for item in before[:4]]
    after[3]['output'] = 'short'
    changes = interaction_changes(before, after)
    assert [(row['call_id'], row['action']) for row in changes] == [
        ('keep', 'kept'), ('shorten', 'changed'), ('drop', 'removed'),
    ]
    assert changes[0]['arguments'] == '{"path":"keep.txt"}'
    assert changes[1]['after_bytes'] == 5
    assert changes[2]['after_bytes'] == 0


def test_tool_failure_stops_experiment_before_recovery(tmp_path: Path) -> None:
    observer = ObservationLog(tmp_path / 'observations.jsonl')
    with pytest.raises(RuntimeError, match='tool_finished'):
        observer.record('tool_finished', {'successful': False, 'error': None})
    assert '"successful": false' in (tmp_path / 'observations.jsonl').read_text()


def test_new_comparison_records_and_applies_current_default_policy(tmp_path: Path) -> None:
    path = tmp_path / 'config.toml'
    path.write_text(DEFAULT_CONFIG_TEXT, encoding='utf-8')
    source = tmp_path / 'source'
    workspace = source / 'workspace'
    workspace.mkdir(parents=True)
    for name, content in FIXTURES.items():
        (workspace / name).write_text(content, encoding='utf-8')
    (source / 'scenario.json').write_text(json.dumps({'current': CURRENT}), encoding='utf-8')
    store = JsonlContextStore(source / 'sessions')
    for prompt, names in zip(TURNS, [
        ['docs-build.log', 'docs-release.md', 'retry.py', 'retry-contract.md'], ['retry-ticket.md'],
    ], strict=True):
        store.append('source', 'user_input', {'text': prompt})
        calls = [{'type': 'function_call', 'call_id': name, 'name': 'file_read',
                  'arguments': json.dumps({'path': name})} for name in names]
        store.append('source', 'model_response', {'output': calls})
        for name in names:
            store.append('source', 'tool_call', {'call_id': name, 'name': 'file_read', 'arguments': {'path': name}})
            store.append('source', 'tool_result', {'call_id': name, 'name': 'file_read',
                                                  'result': FileReadTool(workspace).run({'path': name})})
        store.append('source', 'assistant_message', {'text': 'Investigation complete.'})
    store.append('source', 'user_input', {'text': CURRENT})
    original = (source / 'sessions/source.jsonl').read_bytes()
    api = JevAPI({'docs-build.log': (0.25, 0.25), 'docs-release.md': (0.25, 0.25),
                  'retry.py': (0.7, 0.3), 'retry-contract.md': (0.7, 0.2)})
    output_root = Path(__file__).resolve().parents[1] / 'test_outputs'
    output_root.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output_root) as directory:
        output = Path(directory) / 'comparison'
        with patch.dict('os.environ', {'TYPESAFE_API_KEY': 'private-jev-key'}), \
                patch.object(HttpxHttpTransport, 'post_json', return_value=response('Investigation summary.')), \
                patch.object(JevClient, 'evaluate', side_effect=api.evaluate):
            run(path, output, 'jev-1.13.0', 10, source)
        results = json.loads((output / 'results.json').read_text())
        assert results['policy']['keep_threshold'] == 0.3
        assert {row['call_id']: row['action'] for row in results['jev']['interactions']} == {
            'docs-build.log': 'removed', 'docs-release.md': 'removed', 'retry.py': 'kept',
            'retry-contract.md': 'changed', 'retry-ticket.md': 'kept',
        }
        assert json.loads((output / 'status.json').read_text())['status'] == 'complete'
    assert (source / 'sessions/source.jsonl').read_bytes() == original
