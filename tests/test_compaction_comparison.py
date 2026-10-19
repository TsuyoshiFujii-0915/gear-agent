from pathlib import Path
from typing import Any

import pytest

from examples.compaction_comparison import checkpoint_input, interaction_changes, ObservationLog
from gear_agent.agent.compaction_strategy import CompactionCandidate
from gear_agent.config import ModelConfig, ReasoningReplayMode
from gear_agent.model.factory import build_model_adapter


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
