from pathlib import Path
import tempfile
import unittest

from textual.widgets import Input, RichLog

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.loop import AgentLoop
from gear_agent.config import ModelConfig, ReasoningReplayMode, RuntimeConfig
from gear_agent.context_budget import ContextBudgetConfig
from gear_agent.model.client import ModelClient
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.repository import RepositoryContext
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tui_app import GearApp, TextualAgentLoopEventSink
from tests.test_agent_loop import SequencedTransport
from tests.test_context_budget import response


class ContextBudgetTuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_auto_compaction_progress_and_manual_command_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = JsonlContextStore(root / 'sessions')
            store.append('session', 'user_input', {'text': 'older context ' * 800})
            model = ModelConfig('http://localhost/v1/responses', 'test', None, ReasoningReplayMode.NONE)
            transport = SequencedTransport([response('auto summary'), response('finished'), response('manual summary')])
            adapter = ResponsesModelAdapter(ModelClient(transport), model)
            sink = TextualAgentLoopEventSink()
            loop = AgentLoop(adapter, [], store, sink, RepositoryContext(root),
                             context_budget=ContextBudgetConfig(True, 100_000, 1_000, 3_000))
            app = GearApp(model='test', session_id='session', workspace=root,
                          agent_loop=loop, compaction=CompactionService(adapter), store=store,
                          runtime=RuntimeConfig(root, root / 'sessions', False, 2, 30), model_config=model)
            sink.bind(app)
            async with app.run_test(size=(120, 30)) as pilot:
                entry = app.query_one(Input)
                entry.value = 'continue'
                await pilot.press('enter')
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertFalse(entry.disabled)
                self.assertIn('finished', '\n'.join(line.text for line in app.query_one(RichLog).lines))
                entry.value = '/compact'
                await pilot.press('enter')
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertFalse(entry.disabled)
            checkpoints = [event for event in store.load('session') if event['kind'] == 'compaction_summary']
            self.assertEqual([event['payload']['text'] for event in checkpoints], ['auto summary', 'manual summary'])
            self.assertFalse(any(event['kind'] == 'turn_error' for event in store.load('session')))
