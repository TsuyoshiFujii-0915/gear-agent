import copy
from dataclasses import replace
from pathlib import Path

from examples.jev_threshold_sweep import run_case
from gear_agent.agent.history import build_model_history
from gear_agent.model.factory import build_model_adapter
from tests.test_jev_compaction import JEV, MODEL, JevAPI, history


def test_each_threshold_uses_original_history_and_preserves_recent_turn(tmp_path: Path) -> None:
    events = history()
    original = copy.deepcopy(events)
    adapter = build_model_adapter(MODEL)
    scores = {'keep': (0.88, 0.51), 'truncate': (0.82, 0.47), 'drop': (0.21, 0.26)}
    rows = []
    for threshold in (0.2, 0.5, 0.7):
        output = tmp_path / str(threshold)
        output.mkdir()
        config = replace(JEV, policy=replace(JEV.policy, keep_threshold=threshold))
        rows.append(run_case(events, adapter, config, JevAPI(scores), output, [], '', 'current task'))
    assert [row['metrics']['dropped_pairs'] for row in rows] == [0, 1, 1]
    assert [row['metrics']['truncated_results'] for row in rows] == [0, 1, 2]
    assert all(row['metrics']['pinned_interactions'] == 5 for row in rows)
    assert events == original
    assert rows[0]['measurements']['input_items'] == len(build_model_history(events, adapter.replay_policy).items)
    for row in rows:
        recent = next(item for item in row['interactions'] if item['call_id'] == 'recent')
        assert recent['action'] == 'kept'
