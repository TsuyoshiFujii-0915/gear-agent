import html
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

from textual.widgets import Input, RichLog

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.loop import AgentLoop
from gear_agent.config import ModelConfig, ReasoningReplayMode, RuntimeConfig
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


class FinalAnswerTransport(HttpTransport):
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        self.payloads.append(payload)
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "resumed answer"}],
                }
            ],
            "usage": {"total_tokens": 12},
        }


class GearAppTests(unittest.IsolatedAsyncioTestCase):
    async def test_resumed_session_restores_history_usage_and_compacted_model_context(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            session_dir = workspace / "sessions"
            store = JsonlContextStore(session_dir)
            session_id = "12345678-1234-5678-1234-567812345678"
            store.append(session_id, "user_input", {"text": "old request"})
            store.append(session_id, "assistant_message", {"text": "old answer"})
            store.append(session_id, "compaction_summary", {"text": "saved summary"})
            store.append(session_id, "user_input", {"text": "post-summary request"})
            store.append(
                session_id,
                "model_response",
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "echo",
                            "arguments": '{"text": "ok"}',
                        }
                    ],
                    "usage": {"total_tokens": 1500},
                },
            )
            store.append(
                session_id,
                "tool_call",
                {
                    "iteration": 1,
                    "call_id": "call_1",
                    "name": "echo",
                    "arguments": {"text": "ok"},
                },
            )
            store.append(
                session_id,
                "tool_result",
                {
                    "iteration": 1,
                    "call_id": "call_1",
                    "name": "echo",
                    "result": {"text": "ok"},
                },
            )
            store.append(session_id, "assistant_message", {"text": "post-summary answer"})
            model_config = ModelConfig(
                url="http://localhost:1234/v1/responses",
                model="local-model-id",
                api_key=None,
                reasoning_replay=ReasoningReplayMode.NONE,
            )
            transport = FinalAnswerTransport()
            client = ModelClient(transport)
            runtime = RuntimeConfig(
                workdir=workspace,
                session_dir=session_dir,
                network_enabled=False,
                max_iterations=4,
                model_timeout_seconds=30,
            )
            app = GearApp(
                model="local-model-id",
                session_id=session_id,
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

            async with app.run_test(size=(120, 30)) as pilot:
                await pilot.pause()
                visible_history = "\n".join(
                    line.text for line in app.query_one("#chat", RichLog).lines
                )
                self.assertIn("old request", visible_history)
                self.assertIn("old answer", visible_history)
                self.assertIn("summary saved: saved summary", visible_history)
                self.assertIn("tool echo", visible_history)
                self.assertIn("post-summary answer", visible_history)
                screen_text = html.unescape(
                    "".join(re.findall(r"<text[^>]*>(.*?)</text>", app.export_screenshot()))
                )
                self.assertIn("session\u00a012345678", screen_text)
                self.assertIn("tokens1.5k", screen_text)

                input_widget = app.query_one(Input)
                input_widget.value = "continue"
                await pilot.press("enter")
                await app.workers.wait_for_complete()
                await pilot.pause()

            model_input = transport.payloads[0]["input"]
            self.assertNotIn("old request", str(model_input))
            self.assertIn("saved summary", str(model_input))
            self.assertIn("function_call_output", str(model_input))
            self.assertEqual(model_input[-1], {"role": "user", "content": "continue"})
            self.assertEqual(
                [path.name for path in session_dir.glob("*.jsonl")],
                [f"{session_id}.jsonl"],
            )
            self.assertEqual(store.load(session_id)[-1]["payload"]["text"], "resumed answer")

    async def test_compaction_error_is_persisted_and_input_reenabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = JsonlContextStore(workspace / "sessions")
            store.append("session-1", "user_input", {"text": "preserve context"})
            model_config = ModelConfig(
                url="http://localhost:1234/v1/responses",
                model="local-model-id",
                api_key=None,
                reasoning_replay=ReasoningReplayMode.NONE,
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
