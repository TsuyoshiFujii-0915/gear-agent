from pathlib import Path
from dataclasses import asdict, replace
import json
from typing import Any

import pytest

from gear_agent.config import ModelConfig, ReasoningReplayMode
from gear_agent.errors import GearError
from gear_agent.model.adapter import ModelAdapter
from gear_agent.model.client import ModelClient
from gear_agent.model.factory import build_model_adapter
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from tests.test_responses_stream import FixtureTransport, RecordingModelProgressSink, streaming_config
from gear_agent.model.events import ModelTextDelta, SilentModelProgressEventSink


@pytest.mark.parametrize('stream', [False, True])
def test_adapter_preserves_canonical_response_and_progress(stream: bool) -> None:
    config = replace(streaming_config(), stream=stream)
    terminal = json.loads((__import__('pathlib').Path(__file__).parent / 'fixtures/responses/text.sse').read_text().split('data: ')[-1].strip())['response']
    transport = FixtureTransport('text.sse', terminal)
    sink = RecordingModelProgressSink()
    adapter: ModelAdapter = ResponsesModelAdapter(ModelClient(transport, sink), config)
    response = adapter.create_response('hello', [], 'Follow instructions.', 30, 5, sink)
    assert response.persisted_payload == terminal
    assert response.text == 'こんにちは'
    assert response.function_calls == []
    assert response.replayed_output.items == terminal['output']
    assert [e.delta for e in sink.events if isinstance(e, ModelTextDelta)] == (['こんにちは'] if stream else [])


@pytest.mark.parametrize('fixture', ['premature_eof.sse', 'stream_error.sse'])
def test_adapter_propagates_stream_failures_without_retry(fixture: str) -> None:
    transport = FixtureTransport(fixture, {})
    adapter = ResponsesModelAdapter(ModelClient(transport), streaming_config())
    with pytest.raises(GearError):
        adapter.create_response(
            'hello', [], '', 30, 5, SilentModelProgressEventSink(),
        )
    assert transport.stream_calls == 1
    assert transport.json_calls == 0


def test_legacy_configuration_builds_serializable_capabilities() -> None:
    config = ModelConfig(url='http://localhost:1234/v1/responses', model='local', api_key=None, reasoning_replay=ReasoningReplayMode.NONE)
    adapter = build_model_adapter(config)
    assert isinstance(adapter, ResponsesModelAdapter)
    assert json.loads(json.dumps(asdict(adapter.capabilities))) == {
        'streaming': True,
        'opaque_reasoning_replay': True,
        'textual_compaction': True,
        'native_compaction': False,
        'function_calling': True,
    }
    assert adapter.capabilities == build_model_adapter(config).capabilities
    assert adapter.replay_policy.mode is ReasoningReplayMode.NONE


@pytest.mark.parametrize('output', [None, [7]])
def test_invalid_output_fails_in_model_layer(output: Any) -> None:
    transport = FixtureTransport('text.sse', {'output': output})
    adapter = ResponsesModelAdapter(ModelClient(transport), replace(streaming_config(), stream=False))
    response = adapter.create_response(
        'hello', [], '', 30, None, SilentModelProgressEventSink(),
    )
    with pytest.raises(GearError) as error:
        _ = response.replayed_output
    assert error.value.error_type == 'response_shape_invalid'
    assert error.value.origin == 'responses_adapter'


def test_loop_and_compaction_accept_provider_neutral_model_results() -> None:
    from unittest.mock import Mock

    from gear_agent.agent.compaction import CompactionService
    from gear_agent.agent.events import SilentAgentLoopEventSink
    from gear_agent.agent.loop import AgentLoop
    from gear_agent.model.adapter import ModelResponse
    from gear_agent.model.replay import (
        ModelReplayScope, ReasoningReplayPolicy, ReplayedOutput,
        empty_replay_diagnostic,
    )
    from gear_agent.model.types import ModelHistory
    from gear_agent.store.memory import MemoryContextStore

    # Mock only the external model boundary; execute the loop and store normally.
    adapter = Mock(spec=ModelAdapter)
    adapter.replay_policy = ReasoningReplayPolicy(
        ReasoningReplayMode.NONE, ModelReplayScope('test', 'test-endpoint', 'test-model'),
    )
    adapter.prepare_history.return_value = ModelHistory(['hello'], empty_replay_diagnostic())
    response = Mock(spec=ModelResponse)
    response.persisted_payload = {'provider_result': 'done'}
    response.replayed_output = ReplayedOutput([], empty_replay_diagnostic())
    response.function_calls = []
    response.text = 'done'
    adapter.create_response.return_value = response
    store = MemoryContextStore()

    result = AgentLoop(adapter, [], store, SilentAgentLoopEventSink()).run_turn('session', 'hello', 1, 30)
    assert result.final_text == 'done'
    assert store.load('session')[1]['payload'] == {'provider_result': 'done'}
    assert CompactionService(adapter).compact('session', store, 30) == 'done'
    assert store.load('session')[-1]['payload'] == {'text': 'done'}


def test_legacy_file_constructs_adapter(tmp_path: 'Path') -> None:
    from gear_agent.config import DEFAULT_CONFIG_TEXT, load_config

    config_path = tmp_path / 'gear.toml'
    config_path.write_text(
        DEFAULT_CONFIG_TEXT.replace('stream = false\n', '').replace(
            'model_stream_idle_timeout_seconds = 60\n', '',
        ).replace('reasoning_replay = "none"\n', ''),
        encoding='utf-8',
    )
    config = load_config(config_path, {})
    adapter = build_model_adapter(config.model)
    assert isinstance(adapter, ResponsesModelAdapter)
    assert adapter.replay_policy.mode is ReasoningReplayMode.NONE
    assert config.model.stream is False


def test_streaming_compaction_is_silent_and_preserves_summary() -> None:
    from gear_agent.agent.compaction import CompactionService
    from gear_agent.store.memory import MemoryContextStore

    sink = RecordingModelProgressSink()
    transport = FixtureTransport('text.sse', {})
    adapter = ResponsesModelAdapter(ModelClient(transport, sink), streaming_config())
    store = MemoryContextStore()
    store.append('session', 'user_input', {'text': 'hello'})
    assert CompactionService(adapter).compact('session', store, 30, 5) == 'こんにちは'
    assert store.load('session')[-1]['payload'] == {'text': 'こんにちは'}
    assert [e.delta for e in sink.events if isinstance(e, ModelTextDelta)] == []
