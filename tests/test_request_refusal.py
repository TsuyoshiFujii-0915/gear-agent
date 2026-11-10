import json
from pathlib import Path
from typing import Any

import pytest

from gear_agent.errors import GearError
from gear_agent.runtime import build_agent_runtime
from tests.test_headless import model_endpoint
from tests.test_request_telemetry import Events, configuration, records


@pytest.mark.parametrize('delta', [True, False])
def test_refusal_is_generated_content_without_output_text(tmp_path: Path, delta: bool) -> None:
    events: list[dict[str, Any]] = []
    if delta:
        events.append({'type': 'response.refusal.delta', 'delta': 'private refusal'})
    events.append({'type': 'response.completed', 'response': {'status': 'completed', 'output': [
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'refusal', 'refusal': 'private refusal'}]},
    ]}})
    data = ''.join(f'data: {json.dumps(event)}\n\n' for event in events).encode()
    with model_endpoint([(200, 'text/event-stream', data)]) as (url, requests):
        config = configuration(tmp_path, url, True)
        runtime = build_agent_runtime(config, config.runtime, Events())
        with pytest.raises(GearError):
            runtime.loop.run_turn('s', 'task', 1, 5, 5)
    row, = records(tmp_path)
    assert row['client']['first_content']['basis'] == ('response.refusal.delta' if delta else 'response.completed')
    assert row['client']['first_answer']['reason'] == 'not_applicable'
    assert 'private refusal' not in json.dumps(row)
