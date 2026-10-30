from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator
import json
import tempfile
import unittest

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.loop import AgentLoop
from gear_agent.config import DEFAULT_CONFIG_TEXT, ModelConfig, ReasoningReplayMode, load_config
from gear_agent.context_budget import ContextBudgetConfig
from gear_agent.errors import GearError
from gear_agent.headless import TaskPrompt, run_task
from gear_agent.model.client import ModelClient
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.model.transport import HttpTransport, SseEvent
from gear_agent.repository import RepositoryContext
from gear_agent.runtime import build_agent_runtime
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.filesystem import FileWriteTool
from tests.test_headless import json_response, message, model_endpoint, tool_call


class ResponseTransport(HttpTransport):
    """Scripts the external model boundary without replacing agent or storage logic."""

    def __init__(self, response: dict[str, Any], events: list[dict[str, Any] | str]) -> None:
        self.response = response
        self.events = events
        self.payloads: list[dict[str, Any]] = []
        self.modes: list[str] = []

    def post_json(
        self, url: str, headers: dict[str, str], payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.payloads.append(deepcopy(payload))
        self.modes.append('json')
        return deepcopy(self.response)

    def post_sse(
        self, url: str, headers: dict[str, str], payload: dict[str, Any],
        timeout_seconds: float, idle_timeout_seconds: float,
    ) -> Iterator[SseEvent]:
        self.payloads.append(deepcopy(payload))
        self.modes.append('sse')
        for event in self.events:
            yield SseEvent('message', event if isinstance(event, str) else json.dumps(event))


def model_config(stream: bool) -> ModelConfig:
    return ModelConfig('https://example.test/v1/responses', 'fixture-model',
                       'credential-sentinel', ReasoningReplayMode.NONE, stream)


class ResponseCompletionTests(unittest.TestCase):
    def test_incomplete_empty_output_is_not_retried_as_a_missing_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            response = {'status': 'incomplete', 'output': []}
            transport = ResponseTransport(response, [])
            adapter = ResponsesModelAdapter(ModelClient(transport), model_config(False))
            store = JsonlContextStore(root / 'sessions')
            loop = AgentLoop(adapter, [], store, SilentAgentLoopEventSink(), RepositoryContext(root))
            with self.assertRaises(GearError) as caught:
                loop.run_turn('empty', 'task', 3, 30)
            self.assertEqual(caught.exception.error_type, 'response_incomplete')
            self.assertEqual(len(transport.payloads), 1)
            self.assertEqual([e['kind'] for e in store.load('empty')], ['user_input'])

    def test_assembled_sse_response_cannot_inherit_an_unchecked_error(self) -> None:
        events = [
            {'type': 'response.created', 'response': {'status': 'in_progress', 'error': {'code': 'server_error'}}},
            {'type': 'response.output_item.done', 'output_index': 0, 'item': message('partial')['output'][0]},
            {'type': 'response.completed', 'response': {'status': 'completed'}},
        ]
        transport = ResponseTransport({}, events)
        with self.assertRaises(GearError) as caught:
            ModelClient(transport).create_response(model_config(True), '', [], '', 30, 5)
        self.assertEqual(caught.exception.error_type, 'response_failed')

    def test_rejects_explicit_unsuccessful_status_in_json_and_completed_sse(self) -> None:
        cases = [
            ('incomplete', 'response_incomplete'), ('failed', 'response_failed'),
            ('cancelled', 'response_status_invalid'), ('queued', 'response_status_invalid'),
            ('in_progress', 'response_status_invalid'), ('unknown-secret', 'response_status_invalid'),
            (None, 'response_status_invalid'), (False, 'response_status_invalid'),
            (42, 'response_status_invalid'), ([], 'response_status_invalid'),
            ({'secret': 'private-value'}, 'response_status_invalid'),
        ]
        for stream in (False, True):
            for status, expected in cases:
                with self.subTest(stream=stream, status=status):
                    response = {**message('partial answer'), 'status': status}
                    transport = ResponseTransport(response, [{'type': 'response.completed', 'response': response}])
                    with self.assertRaises(GearError) as caught:
                        ModelClient(transport).create_response(model_config(stream), 'input-secret', [], '', 30, 5)
                    self.assertEqual(caught.exception.error_type, expected)
                    self.assertEqual(caught.exception.origin, 'responses_stream' if stream else 'model_client')
                    self.assertEqual(transport.modes, ['sse' if stream else 'json'])
                    for private in ('unknown-secret', 'private-value', 'partial answer', 'input-secret'):
                        self.assertNotIn(private, json.dumps(asdict(caught.exception)))

    def test_rejects_error_body_even_with_completed_status_or_output(self) -> None:
        for stream in (False, True):
            for error in ({'code': 'server_error', 'message': 'provider-secret'}, {}, '', False, []):
                for status in ({}, {'status': 'completed'}):
                    if stream and not status:
                        continue
                    with self.subTest(stream=stream, error=error, status=status):
                        response = {**message('partial answer'), **status, 'error': error}
                        transport = ResponseTransport(response, [{'type': 'response.completed', 'response': response}])
                        with self.assertRaises(GearError) as caught:
                            ModelClient(transport).create_response(model_config(stream), '', [], '', 30, 5)
                        self.assertEqual(caught.exception.error_type, 'response_failed')
                        self.assertNotIn('provider-secret', json.dumps(asdict(caught.exception)))
        transport = ResponseTransport({'error': {'message': 'provider-secret'}}, [])
        with self.assertRaises(GearError) as caught:
            ModelClient(transport).create_response(model_config(False), '', [], '', 30, None)
        self.assertEqual(caught.exception.error_type, 'response_failed')

    def test_rejects_conflicting_incomplete_details(self) -> None:
        for stream in (False, True):
            for details in ({'reason': 'max_output_tokens'}, {}, 'private-details', False):
                with self.subTest(stream=stream, details=details):
                    response = {**message('partial'), 'status': 'completed', 'incomplete_details': details}
                    transport = ResponseTransport(response, [{'type': 'response.completed', 'response': response}])
                    with self.assertRaises(GearError) as caught:
                        ModelClient(transport).create_response(model_config(stream), '', [], '', 30, 5)
                    self.assertEqual(caught.exception.error_type, 'response_status_invalid')

    def test_json_omitted_status_is_preserved_without_fabricating_completion(self) -> None:
        for extra in ({}, {'status': 'completed'}, {'error': None, 'incomplete_details': None}):
            with self.subTest(extra=extra):
                response = {**message('done'), **extra}
                transport = ResponseTransport(response, [])
                actual = ModelClient(transport).create_response(model_config(False), '', [], '', 30, None)
                self.assertEqual(actual, response)
                self.assertEqual('status' in actual, 'status' in extra)

    def test_sse_still_requires_completed_status_and_terminal_event(self) -> None:
        for events, expected in (
            ([{'type': 'response.completed', 'response': message('partial')}], 'response_stream_event_invalid'),
            ([], 'response_stream_terminated'),
            (['[DONE]'], 'response_stream_terminated'),
            ([{'type': 'response.output_item.done', 'output_index': 0,
               'item': message('partial')['output'][0]}], 'response_stream_terminated'),
        ):
            with self.subTest(events=events):
                transport = ResponseTransport({}, events)
                with self.assertRaises(GearError) as caught:
                    ModelClient(transport).create_response(model_config(True), '', [], '', 30, 5)
                self.assertEqual(caught.exception.error_type, expected)
                self.assertEqual(transport.modes, ['sse'])

    def test_failure_diagnostics_only_expose_known_codes_and_reasons(self) -> None:
        for stream in (False, True):
            for status in ('incomplete', 'failed'):
                for reason in ('max_output_tokens', 'content_filter', 'reason-secret', {'secret': 'reason-secret'}):
                    with self.subTest(stream=stream, status=status, reason=reason):
                        response = {
                            **message('body-secret'), 'status': status, 'id': 'id-secret',
                            'error': {'code': 'server_error', 'message': 'credential-sentinel Authorization: Bearer header-secret'},
                            'incomplete_details': {'reason': reason, 'input': 'input-secret'},
                        }
                        transport = ResponseTransport(response, [{'type': f'response.{status}', 'response': response}])
                        with self.assertRaises(GearError) as caught:
                            ModelClient(transport).create_response(model_config(stream), 'input-secret', [], '', 30, 5)
                        self.assertEqual(caught.exception.error_type, f'response_{status}')
                        self.assertEqual(caught.exception.details['code'], 'server_error')
                        if isinstance(reason, str) and reason in ('max_output_tokens', 'content_filter'):
                            self.assertEqual(caught.exception.details['reason'], reason)
                        serialized = json.dumps(asdict(caught.exception)) + str(caught.exception)
                        for private in ('body-secret', 'id-secret', 'credential-sentinel', 'header-secret',
                                        'input-secret', 'reason-secret', 'Authorization'):
                            self.assertNotIn(private, serialized)

    def test_failure_events_cannot_claim_success_and_stream_errors_are_safe(self) -> None:
        for kind, expected in (('response.failed', 'response_failed'),
                               ('response.incomplete', 'response_incomplete'), ('error', 'response_stream_error')):
            with self.subTest(kind=kind):
                event = {'type': kind, 'code': 'code-secret', 'message': 'message-secret',
                         'response': {**message('body-secret'), 'status': 'completed',
                                      'error': {'code': 'code-secret', 'message': 'message-secret'}}}
                transport = ResponseTransport({}, [event])
                with self.assertRaises(GearError) as caught:
                    ModelClient(transport).create_response(model_config(True), '', [], '', 30, 5)
                self.assertEqual(caught.exception.error_type, expected)
                for private in ('code-secret', 'message-secret', 'body-secret'):
                    self.assertNotIn(private, json.dumps(asdict(caught.exception)))


class ResponseCompletionIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = JsonlContextStore(self.root / 'sessions')

    def test_unsuccessful_tool_calls_never_write_or_trigger_finalization_retry(self) -> None:
        target = self.root / 'target.txt'
        for stream in (False, True):
            for status in ('incomplete', 'failed', 'cancelled', 'queued', 'in_progress'):
                for output in (message('partial answer'), tool_call('file_write', {'path': 'target.txt', 'content': 'changed'})):
                    with self.subTest(stream=stream, status=status, output=output):
                        target.write_text('original', encoding='utf-8')
                        response = {**output, 'status': status}
                        transport = ResponseTransport(response, [{'type': 'response.completed', 'response': response}])
                        adapter = ResponsesModelAdapter(ModelClient(transport), model_config(stream))
                        session = f'{stream}-{status}-{output["output"][0]["type"]}'
                        loop = AgentLoop(adapter, [FileWriteTool(self.root)], self.store,
                                         SilentAgentLoopEventSink(), RepositoryContext(self.root))
                        with self.assertRaises(GearError):
                            loop.run_turn(session, 'write', 3, 30, 5)
                        self.assertEqual(target.read_text(), 'original')
                        self.assertEqual(len(transport.payloads), 1)
                        self.assertEqual([e['kind'] for e in self.store.load(session)], ['user_input'])

    def test_incomplete_manual_and_automatic_summary_keep_original_history(self) -> None:
        for stream in (False, True):
            for trigger in ('manual', 'automatic'):
                with self.subTest(stream=stream, trigger=trigger):
                    session = f'{stream}-{trigger}'
                    self.store.append(session, 'user_input', {'text': 'old context ' * 600})
                    before = self.store.load(session)
                    response = {**message('partial summary'), 'status': 'incomplete',
                                'incomplete_details': {'reason': 'max_output_tokens'}}
                    transport = ResponseTransport(response, [{'type': 'response.incomplete', 'response': response}])
                    adapter = ResponsesModelAdapter(ModelClient(transport), model_config(stream))
                    with self.assertRaises(GearError) as caught:
                        if trigger == 'manual':
                            CompactionService(adapter).compact(session, self.store, 30, 5)
                        else:
                            loop = AgentLoop(adapter, [], self.store, SilentAgentLoopEventSink(),
                                             RepositoryContext(self.root),
                                             context_budget=ContextBudgetConfig(True, 100_000, 1_000, 3_000))
                            loop.run_turn(session, 'continue', 3, 30, 5)
                    self.assertEqual(caught.exception.error_type, 'response_incomplete')
                    self.assertEqual(self.store.load(session)[:len(before)], before)
                    self.assertFalse(any(e['kind'] in ('compaction_summary', 'model_response', 'assistant_message')
                                         for e in self.store.load(session)))
                    self.assertEqual(len(transport.payloads), 1)
                    self.assertIsInstance(transport.payloads[0]['input'], str)
                    self.assertNotIn('tools', transport.payloads[0])

    def test_headless_records_safe_failure_without_success_or_partial_checkpoint(self) -> None:
        config_path = self.root / 'config.toml'
        config_path.write_text(DEFAULT_CONFIG_TEXT, encoding='utf-8')
        config = load_config(config_path, {})
        for automatic in (False, True):
            with self.subTest(automatic=automatic):
                response = {**message('private-partial-answer'), 'status': 'incomplete',
                            'incomplete_details': {'reason': 'max_output_tokens', 'input': 'private-input'},
                            'error': {'message': 'credential-sentinel'}}
                with model_endpoint([json_response(response)]) as (url, requests):
                    configured = replace(
                        config, model=replace(model_config(False), url=url),
                        context_budget=ContextBudgetConfig(automatic, 100_000, 1_000, 3_000),
                        runtime=replace(config.runtime, workdir=self.root, session_dir=self.root / 'sessions'),
                    )
                    agent = build_agent_runtime(configured, configured.runtime, SilentAgentLoopEventSink())
                    session = f'headless-{automatic}'
                    if automatic:
                        agent.store.append(session, 'user_input', {'text': 'old context ' * 600})
                    with self.assertRaises(GearError) as caught:
                        run_task(agent, session, TaskPrompt('private-input', 'inline'))
                self.assertEqual(caught.exception.error_type, 'response_incomplete')
                self.assertEqual(len(requests), 1)
                events = agent.store.load(session)
                errors = [e for e in events if e['kind'] == 'turn_error']
                self.assertEqual(len(errors), 1)
                self.assertEqual(errors[0]['payload']['error']['type'], 'response_incomplete')
                self.assertFalse(any(e['kind'] in ('assistant_message', 'model_response', 'compaction_summary') for e in events))
                for private in ('private-input', 'private-partial-answer', 'credential-sentinel'):
                    self.assertNotIn(private, json.dumps(errors))
