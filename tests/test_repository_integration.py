from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterator
import json

import pytest

from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.loop import AGENT_INSTRUCTIONS, AgentLoop
from gear_agent.config import ModelConfig, ReasoningReplayMode
from gear_agent.errors import GearError
from gear_agent.model.client import ModelClient
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.model.transport import HttpTransport, SseEvent
from gear_agent.repository import RepositoryContext
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.filesystem import FileReadTool
from tests.test_repository import write_instruction


class InstructionTransport(HttpTransport):
    """Simulates the external model, including its echoed request instructions."""

    def __init__(self, responses: list[dict[str, Any]], after_request: Callable[[], None]) -> None:
        self.responses = responses
        self.after_request = after_request
        self.payloads: list[dict[str, Any]] = []

    def post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
        self.payloads.append(deepcopy(payload))
        response = dict(self.responses.pop(0), instructions=payload['instructions'])
        self.after_request()
        return response

    def post_sse(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: float, idle_timeout_seconds: float) -> Iterator[SseEvent]:
        response = self.post_json(url, headers, payload, timeout_seconds)
        yield SseEvent(event='message', data=json.dumps({'type': 'response.completed', 'response': response}))


def no_change() -> None:
    pass


def final_response() -> dict[str, Any]:
    return {'id': 'resp_final', 'status': 'completed', 'output': [
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'done'}]},
    ]}


def read_response() -> dict[str, Any]:
    return {'id': 'resp_read', 'status': 'completed', 'output': [
        {'type': 'function_call', 'call_id': 'call_read', 'name': 'file_read', 'arguments': '{"path": "backend/service.py"}'},
    ]}


def make_loop(workspace: Path, store: JsonlContextStore, transport: HttpTransport, config: ModelConfig) -> AgentLoop:
    return AgentLoop(
        ResponsesModelAdapter(ModelClient(transport), config), [FileReadTool(workspace)],
        store, SilentAgentLoopEventSink(), RepositoryContext(workspace),
    )


def model_config(stream: bool, replay: ReasoningReplayMode) -> ModelConfig:
    return ModelConfig(url='https://example.test/v1/responses', model='test', api_key=None, reasoning_replay=replay, stream=stream)


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('replay', [ReasoningReplayMode.NONE, ReasoningReplayMode.ENCRYPTED])
def test_next_model_request_uses_changed_and_newly_scoped_instructions(tmp_path: Path, stream: bool, replay: ReasoningReplayMode) -> None:
    root = write_instruction(tmp_path, '.', 'ROOT_INSTRUCTION_OLD')
    write_instruction(tmp_path, 'backend', 'BACKEND_INSTRUCTION')
    (tmp_path / 'backend/service.py').write_text('code', encoding='utf-8')

    def change_instruction() -> None:
        root.write_text('ROOT_INSTRUCTION_NEW', encoding='utf-8')

    transport = InstructionTransport([read_response(), final_response()], change_instruction)
    store = JsonlContextStore(tmp_path / 'sessions')
    loop = make_loop(tmp_path, store, transport, model_config(stream, replay))
    assert loop.run_turn('session', 'inspect the code', 3, 30, 5).final_text == 'done'
    first, second = transport.payloads
    assert first['instructions'].startswith(AGENT_INSTRUCTIONS)
    assert 'ROOT_INSTRUCTION_OLD' in first['instructions']
    assert 'BACKEND_INSTRUCTION' not in first['instructions']
    assert 'ROOT_INSTRUCTION_NEW' in second['instructions']
    assert 'ROOT_INSTRUCTION_OLD' not in second['instructions']
    assert 'BACKEND_INSTRUCTION' in second['instructions']
    assert 'INSTRUCTION' not in json.dumps(second['input'])
    persisted = (tmp_path / 'sessions/session.jsonl').read_text(encoding='utf-8')
    assert 'INSTRUCTION' not in persisted
    assert [event['kind'] for event in store.load('session')] == [
        'user_input', 'model_response', 'tool_call', 'tool_result', 'model_response', 'assistant_message',
    ]


def test_later_turn_observes_creation_deletion_and_updates(tmp_path: Path) -> None:
    transport = InstructionTransport([final_response() for _ in range(4)], no_change)
    store = JsonlContextStore(tmp_path / 'sessions')
    loop = make_loop(tmp_path, store, transport, model_config(False, ReasoningReplayMode.NONE))
    loop.run_turn('session', 'hello', 1, 30)
    instruction = write_instruction(tmp_path, '.', 'created')
    loop.run_turn('session', 'continue', 1, 30)
    instruction.write_text('updated', encoding='utf-8')
    loop.run_turn('session', 'continue', 1, 30)
    instruction.unlink()
    loop.run_turn('session', 'continue', 1, 30)
    assert transport.payloads[0]['instructions'] == AGENT_INSTRUCTIONS
    assert 'created' in transport.payloads[1]['instructions']
    assert 'updated' in transport.payloads[2]['instructions']
    assert transport.payloads[3]['instructions'] == AGENT_INSTRUCTIONS


@pytest.mark.parametrize('compacted', [False, True])
def test_resume_reloads_scopes_and_current_contents_from_real_jsonl(tmp_path: Path, compacted: bool) -> None:
    write_instruction(tmp_path, '.', 'ROOT_OLD')
    nested = write_instruction(tmp_path, 'backend', 'BACKEND_OLD')
    (tmp_path / 'backend/service.py').write_text('code', encoding='utf-8')
    config = model_config(False, ReasoningReplayMode.NONE)
    store = JsonlContextStore(tmp_path / 'sessions')
    first = InstructionTransport([read_response(), final_response()], no_change)
    make_loop(tmp_path, store, first, config).run_turn('session', 'inspect', 3, 30)
    if compacted:
        store.append('session', 'compaction_summary', {'text': 'Inspected backend code.'})
    write_instruction(tmp_path, '.', 'ROOT_CURRENT')
    nested.write_text('BACKEND_CURRENT', encoding='utf-8')
    resumed = InstructionTransport([final_response()], no_change)
    resumed_store = JsonlContextStore(tmp_path / 'sessions')
    make_loop(tmp_path, resumed_store, resumed, config).run_turn('session', 'continue', 1, 30)
    instructions = resumed.payloads[0]['instructions']
    assert 'ROOT_CURRENT' in instructions and 'BACKEND_CURRENT' in instructions
    assert '_OLD' not in instructions
    assert '_CURRENT' not in json.dumps(resumed.payloads[0]['input'])


def test_legacy_session_without_repository_metadata_resumes(tmp_path: Path) -> None:
    store = JsonlContextStore(tmp_path / 'sessions')
    store.append('legacy', 'user_input', {'text': 'old request'})
    store.append('legacy', 'assistant_message', {'text': 'old answer'})
    write_instruction(tmp_path, '.', 'CURRENT_RULES')
    transport = InstructionTransport([final_response()], no_change)
    make_loop(tmp_path, store, transport, model_config(False, ReasoningReplayMode.NONE)).run_turn('legacy', 'continue', 1, 30)
    assert 'CURRENT_RULES' in transport.payloads[0]['instructions']
    assert transport.payloads[0]['input'][:2] == [
        {'role': 'user', 'content': 'old request'}, {'role': 'assistant', 'content': 'old answer'},
    ]


def test_invalid_applicable_instructions_prevent_model_request(tmp_path: Path) -> None:
    (tmp_path / 'AGENTS.md').write_bytes(b'\xff')
    transport = InstructionTransport([final_response()], no_change)
    store = JsonlContextStore(tmp_path / 'sessions')
    loop = make_loop(tmp_path, store, transport, model_config(False, ReasoningReplayMode.NONE))
    with pytest.raises(GearError) as error:
        loop.run_turn('session', 'hello', 1, 30)
    assert error.value.error_type == 'repository_instruction_invalid_utf8'
    assert transport.payloads == []


def test_invalid_nested_instructions_stop_followup_request(tmp_path: Path) -> None:
    write_instruction(tmp_path, 'backend', 'valid')
    (tmp_path / 'backend/AGENTS.md').write_bytes(b'\xff')
    (tmp_path / 'backend/service.py').write_text('code', encoding='utf-8')
    transport = InstructionTransport([read_response(), final_response()], no_change)
    store = JsonlContextStore(tmp_path / 'sessions')
    with pytest.raises(GearError) as error:
        make_loop(tmp_path, store, transport, model_config(False, ReasoningReplayMode.NONE)).run_turn('session', 'inspect', 3, 30)
    assert error.value.error_type == 'repository_instruction_invalid_utf8'
    assert len(transport.payloads) == 1
