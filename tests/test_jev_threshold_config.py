from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.history import build_model_history, select_effective_events
from gear_agent.agent.jev import JevCompactionStrategy
from gear_agent.compaction_config import SUMMARY_COMPACTION
from gear_agent.config import AppConfig, DEFAULT_CONFIG_TEXT, load_config
from gear_agent.errors import GearError
from gear_agent.headless import TaskPrompt, describe_run
from gear_agent.model.replay import reasoning_replay_policy
from gear_agent.runtime import build_agent_runtime
from tests.test_jev_compaction import JevAPI, MODEL, call_ids, event, history


JEV_SECTION = '''
[compaction]
strategy = "jev"
[compaction.jev]
model = "jev-1.13.0"
api_key_env = "TEST_JEV_KEY"
'''


def configured_jev(path: Path, threshold_setting: str) -> AppConfig:
    path.write_text(DEFAULT_CONFIG_TEXT + JEV_SECTION + threshold_setting, encoding='utf-8')
    return load_config(path, {'TEST_JEV_KEY': 'private-jev-key'})


@pytest.mark.parametrize(('setting', 'threshold', 'actions'), [
    ('', 0.3, ('keep', 'truncate', 'drop')),
    ('keep_threshold = 0.2\n', 0.2, ('keep', 'keep', 'keep')),
    ('keep_threshold = 0.3\n', 0.3, ('keep', 'truncate', 'drop')),
    ('keep_threshold = 0.35\n', 0.35, ('drop', 'truncate', 'drop')),
    ('keep_threshold = 0.7\n', 0.7, ('drop', 'truncate', 'drop')),
    ('keep_threshold = 1.0\n', 1.0, ('drop', 'drop', 'drop')),
])
def test_configured_threshold_controls_selection_and_run_metadata(
    tmp_path: Path, setting: str, threshold: float, actions: tuple[str, str, str],
) -> None:
    path = tmp_path / 'config.toml'
    config = configured_jev(path, setting)
    original_config = path.read_bytes()
    jev = config.compaction.jev
    assert jev is not None
    assert asdict(jev.policy) == {
        'keep_threshold': threshold, 'preserve_recent_turns': 1, 'truncate_chars': 160,
        'max_state_tokens': 25_000, 'max_request_tokens': 30_000, 'batch_size': 32,
    }
    assert jev.model == 'jev-1.13.0'
    assert jev.timeout_seconds == 10
    assert config.compaction.fallback == 'none'
    assert not config.context_budget.auto_compaction

    source = history()
    original = json.dumps(source)
    api = JevAPI({'keep': (0.0, 0.3), 'truncate': (0.7, 0.2), 'drop': (0.29, 0.25)})
    candidate = JevCompactionStrategy(jev, api).compact(source, 30, None)
    checkpoint = event(candidate.kind, candidate.payload)
    selected = select_effective_events([checkpoint])
    items = build_model_history([checkpoint], reasoning_replay_policy(MODEL)).items
    results = {item['call_id']: item['output'] for item in items
               if item.get('type') == 'function_call_output'}
    for call_id, action in zip(('keep', 'truncate', 'drop'), actions, strict=True):
        if action == 'drop':
            assert call_id not in call_ids(items)
            assert call_id not in results
        else:
            assert call_id in call_ids(items)
            result = json.loads(results[call_id])
            if action == 'keep':
                assert result['content'] == 'RESULT-BODY-' * 600
            else:
                assert result['truncated'] is True
                assert len(result['preview']) == 160
    assert candidate.metrics['dropped_pairs'] == actions.count('drop')
    assert candidate.metrics['truncated_results'] == actions.count('truncate')

    protected = {'write', 'shell', 'patch', 'error', 'recent'}
    assert protected <= set(call_ids(items))
    for original_event in source:
        if original_event['kind'] == 'tool_result' and original_event['payload']['call_id'] in protected:
            retained = next(item for item in selected if item['kind'] == 'tool_result'
                            and item['payload']['call_id'] == original_event['payload']['call_id'])
            assert retained['payload']['result'] == original_event['payload']['result']
    assert next(item for item in items if item.get('type') == 'reasoning')['encrypted_content'] == 'opaque-secret'
    assert len(api.requests) == 1
    assert set(api.requests[0][1]) == {
        'keep:call', 'keep:result', 'truncate:call', 'truncate:result', 'drop:call', 'drop:result',
    }
    assert json.dumps(source) == original

    runtime = replace(config.runtime, workdir=tmp_path, session_dir=tmp_path / 'sessions')
    agent = build_agent_runtime(config, runtime, SilentAgentLoopEventSink())
    spec = asdict(describe_run(agent, 'session', TaskPrompt('task', 'inline')))
    assert spec['compaction']['jev']['policy']['keep_threshold'] == threshold
    assert spec['compaction']['fallback'] == 'none'
    assert 'private-jev-key' not in json.dumps(spec)
    assert path.read_bytes() == original_config


@pytest.mark.parametrize('literal', [
    '0', '-0.1', '1.01', 'true', 'false', '"0.3"', 'nan', '+inf', '-inf',
])
def test_invalid_threshold_is_rejected_when_loading_config(tmp_path: Path, literal: str) -> None:
    with pytest.raises(GearError) as caught:
        configured_jev(tmp_path / 'config.toml', f'keep_threshold = {literal}\n')
    assert caught.value.error_type == 'config_value_invalid'
    assert caught.value.origin == 'config'
    assert caught.value.details['key'] == 'keep_threshold'
    assert 'compaction.keep_threshold' in str(caught.value)
    assert 'private-jev-key' not in str(caught.value)


@pytest.mark.parametrize('section', ['', '\n[compaction]\nstrategy = "summary"\n'])
def test_legacy_and_explicit_summary_do_not_enable_jev(tmp_path: Path, section: str) -> None:
    path = tmp_path / 'config.toml'
    path.write_text(DEFAULT_CONFIG_TEXT + section, encoding='utf-8')
    config = load_config(path, {})
    assert config.compaction == SUMMARY_COMPACTION
    assert not config.context_budget.auto_compaction
