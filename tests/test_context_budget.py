from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import copy
import json
import tempfile
import unittest

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.events import AgentLoopEvent, ContextBudgetEvaluated
from gear_agent.agent.loop import AgentLoop, AGENT_INSTRUCTIONS, FINALIZATION_RETRY_INSTRUCTION
from gear_agent.config import DEFAULT_CONFIG_TEXT, ModelConfig, ReasoningReplayMode, load_config
from gear_agent.context_budget import (
    ByteTokenEstimator, ContextBudgetConfig, ContextBudgetManager, ContextRequest,
    DISABLED_CONTEXT_BUDGET,
)
from gear_agent.errors import GearError
from gear_agent.model.client import ModelClient
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.repository import RepositoryContext
from gear_agent.store.base import ContextStore
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.store.memory import MemoryContextStore
from tests.test_agent_loop import EchoTool, LargeOutputTool, SequencedTransport


def response(text: str) -> dict[str, Any]:
    return {'output': [{'type': 'message', 'role': 'assistant',
                        'content': [{'type': 'output_text', 'text': text}]}]}


class BudgetSink:
    def __init__(self) -> None:
        self.events: list[AgentLoopEvent] = []

    def publish(self, event: AgentLoopEvent) -> None:
        self.events.append(event)

    def budgets(self) -> list[ContextBudgetEvaluated]:
        return [event for event in self.events if isinstance(event, ContextBudgetEvaluated)]


class ContextBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store: ContextStore = MemoryContextStore()
        self.sink = BudgetSink()
        self.config = ContextBudgetConfig(True, 100_000, 1_000, 3_000)
        self.model = ModelConfig('http://localhost:1234/v1/responses', 'unknown-model', None,
                                 ReasoningReplayMode.ENCRYPTED)

    def loop(self, transport: SequencedTransport) -> AgentLoop:
        return AgentLoop(
            ResponsesModelAdapter(ModelClient(transport), self.model),
            [EchoTool(), LargeOutputTool()], self.store, self.sink,
            RepositoryContext(self.root), context_budget=self.config,
        )

    def test_under_budget_dispatches_without_compaction(self) -> None:
        transport = SequencedTransport([response('done')])
        result = self.loop(transport).run_turn('session', 'hello', 1, 30)
        self.assertEqual(result.final_text, 'done')
        self.assertEqual(len(transport.payloads), 1)
        events = self.sink.budgets()
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].diagnostic.fits)
        self.assertFalse(events[0].auto_compaction_triggered)
        self.assertEqual(events[0].phase, 'before')
        self.assertFalse(any(e['kind'] == 'compaction_summary' for e in self.store.load('session')))

    def test_crossing_budget_compacts_once_rebuilds_and_remeasures(self) -> None:
        self.store.append('session', 'user_input', {'text': 'old context ' * 600})
        transport = SequencedTransport([response('short checkpoint'), response('done')])
        result = self.loop(transport).run_turn('session', 'current request', 1, 30)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(len(transport.payloads), 2)
        self.assertIn('old context', transport.payloads[0]['input'])
        rebuilt = transport.payloads[1]['input']
        self.assertIn('short checkpoint', json.dumps(rebuilt))
        self.assertNotIn('old context', json.dumps(rebuilt))
        self.assertEqual(rebuilt[-1], {'role': 'user', 'content': 'current request'})
        events = self.sink.budgets()
        self.assertEqual([e.phase for e in events], ['before', 'compaction', 'after'])
        self.assertTrue(all(e.auto_compaction_triggered for e in events))
        self.assertGreater(events[0].diagnostic.total_estimated_request_tokens, 3_000)
        self.assertLess(events[-1].diagnostic.total_estimated_request_tokens, 3_000)
        checkpoints = [e for e in self.store.load('session') if e['kind'] == 'compaction_summary']
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]['payload']['trigger'], 'automatic')

    def test_oversized_summary_fails_without_agent_dispatch_or_repeated_compaction(self) -> None:
        self.store.append('session', 'user_input', {'text': 'old ' * 2_000})
        transport = SequencedTransport([response('long summary ' * 1_000)])
        with self.assertRaises(GearError) as caught:
            self.loop(transport).run_turn('session', 'hello', 1, 30)
        self.assertEqual(caught.exception.error_type, 'context_budget_exceeded')
        self.assertEqual(caught.exception.origin, 'context_budget')
        self.assertEqual(caught.exception.details['phase'], 'after')
        self.assertEqual(len(transport.payloads), 1)
        self.assertTrue(self.sink.budgets()[-1].failed)

    def test_compaction_request_itself_must_fit_physical_input_capacity(self) -> None:
        self.config = ContextBudgetConfig(True, 6_000, 1_000, 3_000)
        self.store.append('session', 'user_input', {'text': 'enormous ' * 2_000})
        transport = SequencedTransport([])
        before = copy.deepcopy(self.store.load('session'))
        with self.assertRaises(GearError) as caught:
            self.loop(transport).run_turn('session', 'hello', 1, 30)
        self.assertEqual(caught.exception.details['phase'], 'compaction')
        self.assertEqual(transport.payloads, [])
        self.assertEqual(self.store.load('session')[:len(before)], before)
        self.assertFalse(any(e['kind'] == 'compaction_summary' for e in self.store.load('session')))

    def test_large_tool_output_triggers_compaction_before_next_iteration(self) -> None:
        transport = SequencedTransport([
            {'output': [{'type': 'function_call', 'call_id': 'large',
                         'name': 'large_output', 'arguments': '{}'}]},
            response('tool result summarized'), response('done'),
        ])
        result = self.loop(transport).run_turn('session', 'read', 2, 30)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(len(transport.payloads), 3)
        self.assertIn('xxxxxxxx', transport.payloads[1]['input'])
        self.assertNotIn('xxxxxxxx', json.dumps(transport.payloads[2]['input']))
        before = [e for e in self.sink.budgets() if e.phase == 'before']
        self.assertEqual([e.iteration for e in before], [1, 2])
        self.assertEqual([e.auto_compaction_triggered for e in before], [False, True])

    def test_repository_instructions_tools_current_input_and_headroom_count(self) -> None:
        (self.root / 'AGENTS.md').write_text('Always run tests. 日本語', encoding='utf-8')
        transport = SequencedTransport([response('done')])
        self.loop(transport).run_turn('session', 'new question 日本語', 1, 30)
        diagnostic = self.sink.budgets()[0].diagnostic
        self.assertGreater(diagnostic.instructions, 0)
        self.assertGreater(diagnostic.repository_context, 0)
        self.assertGreater(diagnostic.tool_schemas, 0)
        self.assertGreater(diagnostic.current_turn, 0)
        self.assertEqual(diagnostic.history, 0)
        self.assertEqual(diagnostic.reserved_headroom, 1_000)
        request = ContextRequest(transport.payloads[0]['input'], transport.payloads[0]['tools'],
                                 transport.payloads[0]['instructions'], AGENT_INSTRUCTIONS, 0)
        measured = ContextBudgetManager(self.config, ByteTokenEstimator()).evaluate(request)
        self.assertEqual(diagnostic, measured)
        json.dumps(asdict(diagnostic))
        self.assertNotIn('Always run tests', json.dumps(self.store.load('session')))

    def test_repository_rules_too_large_after_compaction_fail(self) -> None:
        (self.root / 'AGENTS.md').write_text('rules ' * 800, encoding='utf-8')
        transport = SequencedTransport([response('short')])
        with self.assertRaises(GearError) as caught:
            self.loop(transport).run_turn('session', 'hello', 1, 30)
        self.assertEqual(caught.exception.details['phase'], 'after')
        self.assertEqual(len(transport.payloads), 1)
        self.assertGreater(self.sink.budgets()[-1].diagnostic.repository_context, 3_000)

    def test_current_user_request_cannot_be_summarized_away_to_force_a_fit(self) -> None:
        transport = SequencedTransport([response('short')])
        with self.assertRaises(GearError) as caught:
            self.loop(transport).run_turn('session', 'new request ' * 1_000, 1, 30)
        self.assertEqual(caught.exception.details['phase'], 'after')
        self.assertEqual(len(transport.payloads), 1)
        self.assertGreater(self.sink.budgets()[-1].diagnostic.current_turn, 3_000)

    def test_finalization_instruction_survives_compaction_and_counts(self) -> None:
        transport = SequencedTransport([
            {'output': [{'type': 'reasoning', 'summary': [], 'encrypted_content': 'opaque' * 1_000}]},
            response('short checkpoint'), response('done'),
        ])
        self.loop(transport).run_turn('session', 'hello', 2, 30)
        self.assertEqual(transport.payloads[-1]['input'][-1],
                         {'role': 'user', 'content': FINALIZATION_RETRY_INSTRUCTION})
        self.assertNotIn('opaque', transport.payloads[1]['input'])
        self.assertNotIn('opaque', json.dumps(transport.payloads[-1]['input']))
        before = [e for e in self.sink.budgets() if e.phase == 'before']
        self.assertGreater(before[1].diagnostic.current_turn, before[0].diagnostic.current_turn)

    def test_checkpoint_boundaries_and_raw_reasoning_survive_jsonl_resume(self) -> None:
        self.store = JsonlContextStore(self.root / 'sessions')
        self.store.append('session', 'user_input', {'text': 'pre-checkpoint-secret'})
        self.store.append('session', 'model_response', {'output': [
            {'type': 'reasoning', 'summary': [], 'encrypted_content': 'old-secret'}]})
        self.store.append('session', 'compaction_summary', {'text': 'existing checkpoint'})
        self.store.append('session', 'user_input', {'text': 'recent work ' * 800})
        audit_path = self.root / 'sessions' / 'session.jsonl'
        audit_before = audit_path.read_bytes()
        transport = SequencedTransport([response('new checkpoint'), response('done')])
        self.store = JsonlContextStore(self.root / 'sessions')
        self.loop(transport).run_turn('session', 'resume', 1, 30)
        self.assertIn('existing checkpoint', transport.payloads[0]['input'])
        self.assertIn('recent work', transport.payloads[0]['input'])
        for payload in transport.payloads:
            self.assertNotIn('pre-checkpoint-secret', json.dumps(payload))
            self.assertNotIn('old-secret', json.dumps(payload))
        self.assertTrue(audit_path.read_bytes().startswith(audit_before))
        resumed = SequencedTransport([response('continued')])
        self.store = JsonlContextStore(self.root / 'sessions')
        self.loop(resumed).run_turn('session', 'next', 1, 30)
        self.assertIn('new checkpoint', json.dumps(resumed.payloads[0]['input']))
        self.assertNotIn('recent work', json.dumps(resumed.payloads[0]['input']))

    def test_disabled_preserves_dispatch_and_manual_compaction(self) -> None:
        self.config = ContextBudgetConfig(False, 2_000, 1_000, 500)
        self.store.append('session', 'user_input', {'text': 'long ' * 2_000})
        transport = SequencedTransport([response('done'), response('manual summary')])
        self.loop(transport).run_turn('session', 'hello', 1, 30)
        self.assertEqual(len(transport.payloads), 1)
        self.assertIn('long ', json.dumps(transport.payloads[0]['input']))
        service = CompactionService(ResponsesModelAdapter(ModelClient(transport), self.model))
        self.assertEqual(service.compact('session', self.store, 30), 'manual summary')
        self.assertEqual(self.sink.budgets(), [])

    def test_empty_summary_failure_keeps_audit_without_checkpoint(self) -> None:
        self.store.append('session', 'user_input', {'text': 'long ' * 2_000})
        transport = SequencedTransport([response(' ')])
        with self.assertRaises(GearError) as caught:
            self.loop(transport).run_turn('session', 'hello', 1, 30)
        self.assertEqual(caught.exception.error_type, 'compaction_summary_missing')
        self.assertFalse(any(e['kind'] == 'compaction_summary' for e in self.store.load('session')))


class EstimatorTests(unittest.TestCase):
    def test_utf8_estimator_applies_documented_byte_margin(self) -> None:
        value = {'content': '日本語\\\"\n', 'encrypted_content': 'opaque'}
        encoded = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        self.assertEqual(ByteTokenEstimator().estimate(value), (len(encoded) * 5 + 3) // 4)

    def test_capacity_reserves_headroom_and_accepts_exact_threshold(self) -> None:
        request = ContextRequest([{'role': 'user', 'content': 'hello'}], [], 'base', 'base', 0)
        probe = ContextBudgetManager(DISABLED_CONTEXT_BUDGET, ByteTokenEstimator()).evaluate(request)
        total = probe.total_estimated_request_tokens
        fitting = ContextBudgetManager(ContextBudgetConfig(True, total + 100, 100, None), ByteTokenEstimator())
        self.assertEqual(fitting.evaluate(request).input_limit_tokens, total)
        self.assertTrue(fitting.evaluate(request).fits)
        tight = ContextBudgetManager(ContextBudgetConfig(True, total + 100, 101, None), ByteTokenEstimator())
        self.assertFalse(tight.evaluate(request).fits)

    def test_major_component_totals_are_additive(self) -> None:
        request = ContextRequest([{'role': 'assistant', 'content': 'history'}, {'role': 'user', 'content': 'now'}],
                                 [EchoTool().schema()], 'base\nrepo', 'base', 1)
        diagnostic = ContextBudgetManager(DISABLED_CONTEXT_BUDGET, ByteTokenEstimator()).evaluate(request)
        self.assertEqual(diagnostic.total_estimated_request_tokens,
                         diagnostic.instructions + diagnostic.repository_context + diagnostic.history
                         + diagnostic.current_turn + diagnostic.tool_schemas + diagnostic.request_overhead)


class BudgetConfigTests(unittest.TestCase):
    def load(self, section: str) -> ContextBudgetConfig:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.toml'
            legacy = DEFAULT_CONFIG_TEXT.split('[context_budget]')[0]
            path.write_text(legacy + '\n' + section, encoding='utf-8')
            return load_config(path, {}).context_budget

    def test_legacy_config_has_explicit_disabled_policy(self) -> None:
        self.assertEqual(self.load(''), DISABLED_CONTEXT_BUDGET)

    def test_enabled_unknown_window_fails_at_configuration_load(self) -> None:
        with self.assertRaises(GearError) as caught:
            self.load('[context_budget]\nauto_compaction = true\nreserved_tokens = 1000\n')
        self.assertEqual(caught.exception.error_type, 'config_value_invalid')
        self.assertEqual(caught.exception.details['table'], 'context_budget')

    def test_explicit_policy_loads(self) -> None:
        config = self.load('[context_budget]\nauto_compaction = true\ncontext_window_tokens = 10000\n'
                           'reserved_tokens = 1000\nmax_input_tokens = 8000\n')
        self.assertEqual(config, ContextBudgetConfig(True, 10000, 1000, 8000))

    def test_enabled_requires_explicit_reserved_headroom(self) -> None:
        with self.assertRaises(GearError):
            self.load('[context_budget]\nauto_compaction = true\ncontext_window_tokens = 10000\n')

    def test_rejects_invalid_present_policy_values(self) -> None:
        sections = [
            'auto_compaction = "true"',
            'auto_compaction = true\ncontext_window_tokens = true\nreserved_tokens = 0',
            'auto_compaction = true\ncontext_window_tokens = 1000\nreserved_tokens = true',
            'auto_compaction = true\ncontext_window_tokens = 1000\nreserved_tokens = -1',
            'auto_compaction = true\ncontext_window_tokens = 1000\nreserved_tokens = 1000',
            'auto_compaction = true\ncontext_window_tokens = 1000\nreserved_tokens = 100\nmax_input_tokens = 901',
            'auto_compaction = true\ncontext_window_tokens = 1000\nreserved_tokens = 0\nmax_input_tokens = 0',
            'auto_compaction = false\ncontext_window_tokens = "unknown"',
            'auto_compaction = false\ntypo = 4',
        ]
        for section in sections:
            with self.subTest(section=section), self.assertRaises(GearError):
                self.load('[context_budget]\n' + section + '\n')
