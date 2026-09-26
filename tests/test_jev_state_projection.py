from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from gear_agent.agent.history import build_model_history
from gear_agent.context_budget import ByteTokenEstimator
from gear_agent.errors import GearError
from gear_agent.model.replay import read_model_response_event, reasoning_replay_policy
from tests.test_jev_compaction import JevAPI, MODEL, POLICY, event, history, strategy


def test_long_final_answer_reaches_jev_once_without_changing_replay() -> None:
    source = history()
    final_text = 'LONG-FINAL-' + 'x' * 10_000
    for item in source:
        if item['kind'] == 'model_response':
            raw = read_model_response_event(item['payload']).response
            for output in raw['output']:
                if output['type'] == 'message':
                    output['content'][0]['text'] = final_text
        elif item['kind'] == 'assistant_message' and item['payload']['text'] == 'checking':
            item['payload']['text'] = final_text
    original = copy.deepcopy(source)
    api = JevAPI({'keep': (1, 1), 'truncate': (1, 1), 'drop': (1, 1)})

    candidate = strategy(api, POLICY).compact(source, 30, None)

    assert len(api.requests) == 1
    state = api.requests[0][0]
    assert state.count(final_text) == 1
    assert ByteTokenEstimator().estimate(state) <= POLICY.max_state_tokens
    assert source == original
    assert build_model_history([event(candidate.kind, candidate.payload)], reasoning_replay_policy(MODEL)).items == build_model_history(source, reasoning_replay_policy(MODEL)).items


def test_only_corresponding_mirrors_are_removed_from_jev_state() -> None:
    source = history()
    repeated_response: dict[str, Any] = {'output': [
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'alpha'}]},
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'beta'}]},
    ]}
    source[-1:-1] = [
        event('user_input', {'text': 'first multi-message answer'}),
        event('model_response', repeated_response),
        event('assistant_message', {'text': 'alphabeta'}),
        event('user_input', {'text': 'independent legacy answer'}),
        event('assistant_message', {'text': 'alphabeta'}),
        event('assistant_message', {'text': 'alphabeta'}),
        event('user_input', {'text': 'independent repeated response'}),
        event('model_response', copy.deepcopy(repeated_response)),
        event('assistant_message', {'text': 'alphabeta'}),
    ]
    api = JevAPI({key: (1, 1) for key in ('keep', 'truncate', 'drop', 'recent')})

    strategy(api, POLICY).compact(source, 30, None)

    messages = [item['text'] for item in json.loads(api.requests[0][0])
                if item['kind'] in ('assistant', 'assistant_message')]
    assert messages == ['checking', 'recent done', 'alpha', 'beta', 'alphabeta', 'alphabeta', 'alpha', 'beta']


def test_mismatched_mirror_fails_before_sending_jev_state() -> None:
    source = history()
    mirror = next(item for item in source if item['kind'] == 'assistant_message')
    mirror['payload']['text'] = 'does not match the response'
    api = JevAPI({'keep': (1, 1), 'truncate': (1, 1), 'drop': (1, 1)})

    with pytest.raises(GearError) as caught:
        strategy(api, POLICY).compact(source, 30, None)

    assert caught.value.error_type == 'history_shape_invalid'
    assert not api.requests
