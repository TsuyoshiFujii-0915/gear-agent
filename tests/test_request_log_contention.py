import json
import multiprocessing
from pathlib import Path

import pytest

from gear_agent.headless import TaskPrompt
from gear_agent.run_artifacts import run_with_artifacts
from gear_agent.run_collector import EvalRunCollector
from gear_agent.runtime import build_agent_runtime
from tests.test_headless import json_response, message, model_endpoint, tool_call
from tests.test_request_telemetry import configuration


def _run_tool_turn(root: Path, url: str) -> None:
    config = configuration(root, url, False)
    collector = EvalRunCollector()
    agent = build_agent_runtime(config, config.runtime, collector, collector)
    run_with_artifacts(agent, 's', TaskPrompt('read the file', 'inline'), root / 'run', collector, {})


def test_locked_request_log_allows_tool_and_final_answer_to_complete(tmp_path: Path) -> None:
    fcntl = pytest.importorskip('fcntl')
    sessions = tmp_path / 'sessions'
    sessions.mkdir()
    sidecar = sessions / 's.requests.ndjson'
    original = b'{"request_id":"earlier"}\n'
    sidecar.write_bytes(original)
    (tmp_path / 'data.txt').write_text('tool output')
    responses = [json_response(tool_call('file_read', {'path': 'data.txt'})), json_response(message('done'))]

    with model_endpoint(responses) as (url, requests), sidecar.open('r+b') as locked:
        fcntl.flock(locked, fcntl.LOCK_EX)
        worker = multiprocessing.get_context('spawn').Process(target=_run_tool_turn, args=(tmp_path, url))
        worker.start()
        try:
            worker.join(timeout=10)
            assert not worker.is_alive(), 'Turn did not complete while another process held the request log lock.'
            assert worker.exitcode == 0
        finally:
            if worker.is_alive():
                worker.kill()
                worker.join()
        assert len(requests) == 2
        assert sidecar.read_bytes() == original

    run = tmp_path / 'run'
    assert (run / 'final.txt').read_text() == 'done'
    history = [json.loads(line) for line in (sessions / 's.jsonl').read_text().splitlines()]
    assert [event['kind'] for event in history] == [
        'user_input', 'model_response', 'tool_call', 'tool_result', 'model_response', 'assistant_message',
    ]
    assert history[3]['payload']['result']['content'] == 'tool output'
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
    finished = [e['payload']['request_id'] for e in events if e['kind'] == 'model_request_finished']
    diagnostics = [e['payload']['request_id'] for e in events if e['kind'] == 'request_log_write_failed']
    assert len(set(finished)) == len(finished) == 2
    assert diagnostics == finished
