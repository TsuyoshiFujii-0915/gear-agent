import tempfile
import unittest
from pathlib import Path
from typing import Any

from textual.widgets import Input

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.loop import AgentLoop
from gear_agent.config import ModelConfig, RuntimeConfig
from gear_agent.model.client import ModelClient
from gear_agent.model.transport import HttpTransport
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tui_app import GearApp


class EmptySummaryTransport(HttpTransport):
    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "   "}],
                }
            ]
        }


class GearAppTests(unittest.IsolatedAsyncioTestCase):
    async def test_compaction_error_is_persisted_and_input_reenabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = JsonlContextStore(workspace / "sessions")
            store.append("session-1", "user_input", {"text": "preserve context"})
            model_config = ModelConfig(
                url="http://localhost:1234/v1/responses",
                model="local-model-id",
                api_key=None,
            )
            client = ModelClient(EmptySummaryTransport())
            runtime = RuntimeConfig(
                workdir=workspace,
                session_dir=workspace / "sessions",
                network_enabled=False,
                max_iterations=4,
                model_timeout_seconds=30,
            )
            app = GearApp(
                model="local-model-id",
                session_id="session-1",
                workspace=workspace,
                agent_loop=AgentLoop(
                    client,
                    model_config,
                    [],
                    store,
                    SilentAgentLoopEventSink(),
                ),
                compaction=CompactionService(client),
                store=store,
                runtime=runtime,
                model_config=model_config,
            )

            async with app.run_test() as pilot:
                input_widget = app.query_one(Input)
                input_widget.value = "/compact"
                await pilot.press("enter")
                await app.workers.wait_for_complete()
                await pilot.pause()

                self.assertFalse(input_widget.disabled)

            events = store.load("session-1")
            self.assertEqual(events[-1]["kind"], "turn_error")
            self.assertIn(
                "Compaction response did not contain a summary",
                events[-1]["payload"]["text"],
            )


if __name__ == "__main__":
    unittest.main()
