from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import copy
import tempfile
import unittest

from gear_agent.agent.events import AgentLoopEvent, ContextBudgetEvaluated, ReasoningReplayEvaluated
from gear_agent.agent.loop import AgentLoop, FINALIZATION_RETRY_INSTRUCTION
from gear_agent.config import ModelConfig, ReasoningReplayMode
from gear_agent.context_budget import ContextBudgetConfig
from gear_agent.errors import GearError
from gear_agent.model.client import ModelClient
from gear_agent.model.replay import model_response_event_payload, reasoning_replay_policy
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.model.transport import HttpTransport
from gear_agent.repository import RepositoryContext
from gear_agent.store.memory import MemoryContextStore
from tests.test_agent_loop import EchoTool
from tests.test_context_budget import BudgetSink, response


class ReplayRecordingTransport(HttpTransport):
    """Records actual external requests and the events visible at dispatch."""

    def __init__(self, responses: list[dict[str, Any]], sink: BudgetSink) -> None:
        self.responses = responses
        self.sink = sink
        self.payloads: list[dict[str, Any]] = []
        self.events_at_dispatch: list[list[AgentLoopEvent]] = []

    def post_json(
        self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: int,
    ) -> dict[str, Any]:
        self.payloads.append(copy.deepcopy(payload))
        self.events_at_dispatch.append(list(self.sink.events))
        return self.responses.pop(0)


def reasoning(state: str) -> dict[str, Any]:
    return {'type': 'reasoning', 'summary': [], 'encrypted_content': state}


def replay_events(events: list[AgentLoopEvent]) -> list[ReasoningReplayEvaluated]:
    return [event for event in events if isinstance(event, ReasoningReplayEvaluated)]


class ContextBudgetReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = MemoryContextStore()
        self.sink = BudgetSink()
        self.model = ModelConfig('http://localhost/v1/responses', 'test-model', None, ReasoningReplayMode.ENCRYPTED)
        self.budget = ContextBudgetConfig(True, 100_000, 1_000, 3_000)

    def run_turn(self, transport: ReplayRecordingTransport, iterations: int) -> None:
        AgentLoop(
            ResponsesModelAdapter(ModelClient(transport), self.model), [EchoTool()],
            self.store, self.sink, RepositoryContext(self.root), context_budget=self.budget,
        ).run_turn('session', 'fix the bug', iterations, 30)

    def seed_history(self, state: str) -> None:
        self.store.append('session', 'user_input', {'text': 'previous request'})
        self.store.append('session', 'model_response', model_response_event_payload(
            {'output': [reasoning(state)]}, reasoning_replay_policy(self.model),
        ))

    def test_compacted_same_scope_history_does_not_report_discarded_reuse(self) -> None:
        self.seed_history('discarded-state' * 600)
        audit = copy.deepcopy(self.store.load('session'))
        transport = ReplayRecordingTransport([response('checkpoint'), response('done')], self.sink)

        self.run_turn(transport, 1)

        self.assertEqual(len(transport.payloads), 2)
        self.assertNotIn('discarded-state', str(transport.payloads))
        self.assertEqual(replay_events(transport.events_at_dispatch[0]), [])
        emitted = replay_events(self.sink.events)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].reused_encrypted_items, 0)
        after = next(event for event in self.sink.events
                     if isinstance(event, ContextBudgetEvaluated) and event.phase == 'after')
        self.assertLess(self.sink.events.index(after), self.sink.events.index(emitted[0]))
        self.assertEqual(self.store.load('session')[:len(audit)], audit)

    def test_under_budget_reports_original_history_diagnostic_at_dispatch(self) -> None:
        self.seed_history('retained-state')
        transport = ReplayRecordingTransport([response('done')], self.sink)

        self.run_turn(transport, 1)

        self.assertIn('retained-state', str(transport.payloads[0]['input']))
        emitted = replay_events(transport.events_at_dispatch[0])
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].reused_encrypted_items, 1)
        budget_event = next(event for event in self.sink.events if isinstance(event, ContextBudgetEvaluated))
        self.assertLess(self.sink.events.index(budget_event), self.sink.events.index(emitted[0]))

    def test_compacted_tool_continuation_discards_pending_reuse_diagnostic(self) -> None:
        transport = ReplayRecordingTransport([
            {'output': [reasoning('discarded-tool-state' * 600),
                        {'type': 'function_call', 'call_id': 'echo-1', 'name': 'echo',
                         'arguments': '{"text": "tool result"}'}]},
            response('checkpoint'), response('done'),
        ], self.sink)

        self.run_turn(transport, 2)

        self.assertEqual(len(transport.payloads), 3)
        self.assertNotIn('discarded-tool-state', str(transport.payloads))
        self.assertEqual([event.reused_encrypted_items for event in replay_events(self.sink.events)], [0, 0])
        self.assertEqual(len(replay_events(transport.events_at_dispatch[1])), 1)

    def test_compacted_finalization_retry_discards_pending_reuse_diagnostic(self) -> None:
        transport = ReplayRecordingTransport([
            {'output': [reasoning('discarded-retry-state' * 600)]},
            response('checkpoint'), response('done'),
        ], self.sink)

        self.run_turn(transport, 2)

        self.assertEqual([event.reused_encrypted_items for event in replay_events(self.sink.events)], [0, 0])
        self.assertNotIn('discarded-retry-state', str(transport.payloads))
        self.assertEqual(transport.payloads[-1]['input'][-1],
                         {'role': 'user', 'content': FINALIZATION_RETRY_INSTRUCTION})

    def test_uncompacted_continuation_reports_pending_reuse(self) -> None:
        transport = ReplayRecordingTransport([
            {'output': [reasoning('retained-tool-state'),
                        {'type': 'function_call', 'call_id': 'echo-1', 'name': 'echo',
                         'arguments': '{"text": "ok"}'}]},
            response('done'),
        ], self.sink)

        self.run_turn(transport, 2)

        self.assertIn('retained-tool-state', str(transport.payloads[1]['input']))
        self.assertEqual([event.reused_encrypted_items for event in replay_events(self.sink.events)], [0, 1])
        budget_events = [event for event in self.sink.events if isinstance(event, ContextBudgetEvaluated)]
        self.assertLess(self.sink.events.index(budget_events[-1]),
                        self.sink.events.index(replay_events(self.sink.events)[-1]))

    def test_budget_failure_never_reports_reuse_for_an_undispatched_request(self) -> None:
        for phase in ('compaction', 'after'):
            with self.subTest(phase=phase):
                self.store = MemoryContextStore()
                self.sink = BudgetSink()
                self.seed_history('discarded-state' * 600)
                if phase == 'compaction':
                    self.budget = ContextBudgetConfig(True, 4_000, 1_000, 3_000)
                    self.store.append('session', 'user_input', {'text': 'too large to summarize ' * 500})
                    responses: list[dict[str, Any]] = []
                else:
                    self.budget = ContextBudgetConfig(True, 100_000, 1_000, 3_000)
                    responses = [response('oversized checkpoint ' * 600)]
                transport = ReplayRecordingTransport(responses, self.sink)

                with self.assertRaises(GearError) as caught:
                    self.run_turn(transport, 1)

                self.assertEqual(caught.exception.details['phase'], phase)
                self.assertEqual(replay_events(self.sink.events), [])
                self.assertEqual(len(transport.payloads), 0 if phase == 'compaction' else 1)

    def test_disabled_budget_preserves_original_replay_notifications(self) -> None:
        self.budget = replace(self.budget, auto_compaction=False)
        self.seed_history('retained-large-state' * 600)
        transport = ReplayRecordingTransport([response('done')], self.sink)

        self.run_turn(transport, 1)

        self.assertIn('retained-large-state', str(transport.payloads[0]['input']))
        self.assertEqual([event.reused_encrypted_items for event in replay_events(self.sink.events)], [1])
        self.assertEqual(self.sink.budgets(), [])

    def test_summary_may_repeat_goal_while_current_request_remains_verbatim(self) -> None:
        self.seed_history('discarded-state' * 600)
        summary = 'Current goal: fix the bug. Investigation is complete.'
        transport = ReplayRecordingTransport([response(summary), response('done')], self.sink)

        self.run_turn(transport, 1)

        rebuilt = transport.payloads[-1]['input']
        self.assertIn(summary, rebuilt[0]['content'])
        self.assertEqual(rebuilt[-1], {'role': 'user', 'content': 'fix the bug'})
        self.assertEqual(str(rebuilt).count('fix the bug'), 2)
