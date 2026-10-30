from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import json
import tempfile
import unittest

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.loop import AgentLoop
from gear_agent.config import ModelConfig, ReasoningReplayMode
from gear_agent.context_budget import ByteTokenEstimator
from gear_agent.errors import GearError
from gear_agent.model.client import ModelClient
from gear_agent.model.events import ModelFunctionCallArgumentsDelta, ModelOutputItemCompleted, ModelReasoningTextDelta
from gear_agent.model.replay import read_model_response_event
from gear_agent.model.responses_adapter import ResponsesModelAdapter, ResponsesModelResponse
from gear_agent.model.transport import HttpxHttpTransport
from gear_agent.repository import RepositoryContext
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.filesystem import FileReadTool, FileWriteTool
from gear_agent.tools.providers.tavily_search import TavilySearchProvider
from gear_agent.tools.web_search import WebSearchTool
from tests.strata_fixtures import (
    OPAQUE_REASONING, REASONING_TEXT, completed_response, reasoning_item,
    stream_events, text_item, tool_exchanges,
)
from tests.test_headless import json_response, model_endpoint
from tests.test_response_completion import ResponseTransport
from tests.test_responses_stream import RecordingModelProgressSink
from tests.test_web_search import FakeTavilySearchTransport, _config, _tavily_response


def wire_response(response: dict[str, Any], stream: bool) -> tuple[int, str, bytes]:
    if not stream:
        return json_response(response)
    body = ''.join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n'
                   for event in stream_events(response))
    return 200, 'text/event-stream', body.encode('utf-8')


def config(stream: bool, mode: ReasoningReplayMode) -> ModelConfig:
    return ModelConfig('http://example.test/v1/responses', 'qwen-fixture', None, mode, stream)


class StrataCompatibilityTests(unittest.TestCase):
    def test_json_and_sse_decode_identical_content_tools_usage_and_progress(self) -> None:
        for response in tool_exchanges():
            with self.subTest(output=response['output']):
                decoded: list[ResponsesModelResponse] = []
                for stream in (False, True):
                    sink = RecordingModelProgressSink()
                    transport = ResponseTransport(response, stream_events(response))
                    adapter = ResponsesModelAdapter(ModelClient(transport), config(stream, ReasoningReplayMode.NONE))
                    decoded.append(adapter.create_response('prompt', [], 'rules', 30, 5, sink))
                    if stream:
                        completed = [e for e in sink.events if isinstance(e, ModelOutputItemCompleted)]
                        self.assertEqual([e.item for e in completed], response['output'])
                        self.assertEqual([e.output_index for e in completed], list(range(len(response['output']))))
                        calls = [item for item in response['output'] if item['type'] == 'function_call']
                        for call in calls:
                            fragments = [e.delta for e in sink.events
                                         if isinstance(e, ModelFunctionCallArgumentsDelta) and e.item_id == call['id']]
                            self.assertEqual(len(fragments), 2)
                            self.assertEqual(''.join(fragments), call['arguments'])
                        if response['output'][0]['type'] == 'reasoning':
                            self.assertEqual(''.join(e.delta for e in sink.events if isinstance(e, ModelReasoningTextDelta)),
                                             REASONING_TEXT)
                left, right = decoded
                self.assertEqual(left.response, right.response)
                self.assertEqual(left.function_calls, right.function_calls)
                self.assertEqual(left.text, right.text)
                self.assertEqual(left.usage, right.usage)
                self.assertEqual((left.usage.input_tokens, left.usage.output_tokens, left.usage.total_tokens), (120, 40, 160))
                self.assertNotIn(REASONING_TEXT, left.text)

    def test_unreported_usage_stays_unknown_in_both_modes(self) -> None:
        for stream in (False, True):
            for usage_fields in ({}, {'usage': None}, {'usage': {'input_tokens': 120}}):
                with self.subTest(stream=stream, usage=usage_fields):
                    response = completed_response([text_item('done')])
                    del response['usage']
                    response.update(usage_fields)
                    transport = ResponseTransport(response, stream_events(response))
                    adapter = ResponsesModelAdapter(ModelClient(transport), config(stream, ReasoningReplayMode.NONE))
                    result = adapter.create_response('prompt', [], '', 30, 5, RecordingModelProgressSink())
                    self.assertIsNone(result.usage.output_tokens)
                    self.assertIsNone(result.usage.total_tokens)
                    self.assertEqual(result.usage.input_tokens, 120 if isinstance(usage_fields.get('usage'), dict) else None)

    def test_keepalives_and_fragment_assembly_do_not_duplicate_or_lose_items(self) -> None:
        response = tool_exchanges()[0]
        events = stream_events(response)
        for event in events:
            if event['type'] == 'response.function_call_arguments.done':
                event['type'] = 'fixture.omitted_arguments_done'
            if event['type'] == 'response.output_item.done' and event['item']['type'] == 'function_call':
                event['item']['arguments'] = ''
        events[-1]['response'].pop('output')
        transport = ResponseTransport(response, events)
        result = ModelClient(transport).create_response(config(True, ReasoningReplayMode.NONE), 'task', [], '', 30, 5)
        self.assertEqual(result, response)

    def test_strata_partial_items_and_keepalives_are_not_a_successful_terminal(self) -> None:
        response = tool_exchanges()[0]
        for ending, expected in (('incomplete', 'response_incomplete'), ('failed', 'response_failed'),
                                 ('eof', 'response_stream_terminated'), ('done', 'response_stream_terminated')):
            with self.subTest(ending=ending):
                events: list[dict[str, Any] | str] = stream_events(response)[:-1]
                if ending in ('incomplete', 'failed'):
                    terminal = {**response, 'status': ending}
                    events.append({'type': f'response.{ending}', 'response': terminal})
                if ending == 'done':
                    events.append('[DONE]')
                transport = ResponseTransport(response, events)
                with self.assertRaises(GearError) as caught:
                    ModelClient(transport).create_response(config(True, ReasoningReplayMode.NONE), 'task', [], '', 30, 5)
                self.assertEqual(caught.exception.error_type, expected)
                self.assertEqual(transport.modes, ['sse'])

    def test_tool_round_trips_and_jsonl_resume_preserve_plaintext_and_scoped_opaque_reasoning(self) -> None:
        for stream in (False, True):
            for mode in ReasoningReplayMode:
                with self.subTest(stream=stream, mode=mode), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    (root / 'input.txt').write_text('local input', encoding='utf-8')
                    (root / 'AGENTS.md').write_text('Keep fixture instructions.', encoding='utf-8')
                    search_transport = FakeTavilySearchTransport(_tavily_response())
                    tools = [FileReadTool(root), FileWriteTool(root),
                             WebSearchTool(TavilySearchProvider(_config(), search_transport))]
                    responses = [*tool_exchanges(), completed_response([text_item('Resumed.')])]
                    if mode is ReasoningReplayMode.NONE:
                        responses[0]['output'][0].pop('encrypted_content')
                    with model_endpoint([wire_response(r, stream) for r in responses]) as (url, requests):
                        model = replace(config(stream, mode), url=url)
                        store = JsonlContextStore(root / 'sessions')
                        adapter = ResponsesModelAdapter(ModelClient(HttpxHttpTransport()), model)
                        loop = AgentLoop(adapter, tools, store, SilentAgentLoopEventSink(), RepositoryContext(root))
                        result = loop.run_turn('session', 'start', 4, 30, 5)
                        self.assertEqual(result.final_text, 'Task complete.')
                        self.assertEqual(result.iterations, 3)
                        self.assertEqual((root / 'output.txt').read_text(), 'written')
                        self.assertEqual(len(search_transport.calls), 1)
                        audit = (root / 'sessions/session.jsonl').read_bytes()
                        resumed_store = JsonlContextStore(root / 'sessions')
                        resumed_adapter = ResponsesModelAdapter(ModelClient(HttpxHttpTransport()), model)
                        resumed = AgentLoop(resumed_adapter, tools, resumed_store, SilentAgentLoopEventSink(), RepositoryContext(root))
                        self.assertEqual(resumed.run_turn('session', 'continue', 1, 30, 5).final_text, 'Resumed.')
                    self.assertEqual(len(requests), 4)
                    for request in requests:
                        self.assertEqual(request['tool_choice'], 'auto')
                        self.assertEqual(request['tools'], [tool.schema() for tool in tools])
                        self.assertIn('Keep fixture instructions.', request['instructions'])
                        self.assertFalse({'previous_response_id', 'conversation', 'background'} & request.keys())
                        self.assertEqual('include' in request, mode is ReasoningReplayMode.ENCRYPTED)
                        if mode is ReasoningReplayMode.ENCRYPTED:
                            self.assertEqual(request['include'], ['reasoning.encrypted_content'])
                    self.assertEqual(requests[0]['input'], [{'role': 'user', 'content': 'start'}])
                    expected_ids = [['read'], ['read', 'write', 'search'], ['read', 'write', 'search']]
                    for request, call_ids in zip(requests[1:], expected_ids, strict=True):
                        items = request['input']
                        self.assertEqual([i['call_id'] for i in items if i.get('type') == 'function_call'], call_ids)
                        results = [i for i in items if i.get('type') == 'function_call_output']
                        self.assertEqual([i['call_id'] for i in results], call_ids)
                        self.assertEqual(json.loads(results[0]['output'])['content'], 'local input')
                        reasonings = [i for i in items if i.get('type') == 'reasoning']
                        self.assertEqual(len(reasonings), 1)
                        self.assertEqual(reasonings[0]['content'], reasoning_item()['content'])
                        self.assertEqual(reasonings[0]['summary'], [])
                        self.assertEqual(reasonings[0], responses[0]['output'][0])
                    self.assertEqual(requests[-1]['input'][-1], {'role': 'user', 'content': 'continue'})
                    self.assertTrue((root / 'sessions/session.jsonl').read_bytes().startswith(audit))
                    history = resumed_store.load('session')
                    saved = next(e['payload'] for e in history if e['kind'] == 'model_response')
                    parsed = read_model_response_event(saved)
                    self.assertEqual(parsed.source_scope is not None, mode is ReasoningReplayMode.ENCRYPTED)
                    self.assertEqual(parsed.response['output'][0], responses[0]['output'][0])

    def test_none_strips_only_opaque_and_size_counts_both_representations(self) -> None:
        response = completed_response([reasoning_item(), text_item('done')])
        prepared: dict[ReasoningReplayMode, list[object]] = {}
        for mode in ReasoningReplayMode:
            transport = ResponseTransport(response, [])
            adapter = ResponsesModelAdapter(ModelClient(transport), config(False, mode))
            result = adapter.create_response('prompt', [], '', 30, None, RecordingModelProgressSink())
            events = [{'kind': 'model_response', 'payload': result.persisted_payload}]
            history = adapter.prepare_history(events, 'next')
            prepared[mode] = history.items
            self.assertEqual(history.items[0]['content'], reasoning_item()['content'])
            self.assertEqual(len(history.items), 3)
            self.assertEqual(result.text, 'done')
            self.assertEqual('encrypted_content' in history.items[0], mode is ReasoningReplayMode.ENCRYPTED)
            if mode is ReasoningReplayMode.NONE:
                self.assertEqual(history.diagnostic.dropped_disabled_items, 1)
            else:
                self.assertEqual(history.diagnostic.reused_encrypted_items, 1)
                self.assertEqual(history.items[0]['encrypted_content'], OPAQUE_REASONING)
        estimator = ByteTokenEstimator()
        plain_size = estimator.estimate(prepared[ReasoningReplayMode.NONE])
        both_size = estimator.estimate(prepared[ReasoningReplayMode.ENCRYPTED])
        self.assertGreater(both_size, plain_size)
        for mode, items in prepared.items():
            serialized = json.dumps(items, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            self.assertEqual(estimator.estimate(items), (len(serialized) * 5 + 3) // 4)

    def test_summary_uses_string_input_and_preserves_plaintext_reasoning(self) -> None:
        for stream in (False, True):
            with self.subTest(stream=stream), tempfile.TemporaryDirectory() as temporary:
                response = completed_response([text_item('Summary complete.')])
                transport = ResponseTransport(response, stream_events(response))
                adapter = ResponsesModelAdapter(ModelClient(transport), config(stream, ReasoningReplayMode.NONE))
                store = JsonlContextStore(Path(temporary))
                store.append('session', 'model_response', completed_response([reasoning_item(), text_item('done')]))
                summary = CompactionService(adapter).compact('session', store, 30, 5)
                self.assertEqual(summary, 'Summary complete.')
                self.assertEqual(store.load('session')[-1]['payload']['text'], summary)
                request = transport.payloads[0]
                self.assertIsInstance(request['input'], str)
                self.assertIn(REASONING_TEXT, request['input'])
                self.assertNotIn(OPAQUE_REASONING, request['input'])
                self.assertFalse({'tools', 'tool_choice', 'previous_response_id', 'conversation', 'background'} & request.keys())
