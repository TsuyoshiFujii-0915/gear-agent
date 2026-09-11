from __future__ import annotations

import asyncio
import html
import json
import re
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, Iterator

from textual.widgets import Input, RichLog

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.events import (
    AgentLoopEvent,
    ModelReasoningSummaryDelta as AgentReasoningSummaryDelta,
    ModelRequestStarted,
    ModelTextDelta as AgentTextDelta,
    ToolUseFinished,
    ToolUseStarted,
)
from gear_agent.agent.loop import AgentLoop
from gear_agent.config import ModelConfig, ReasoningReplayMode, RuntimeConfig
from gear_agent.model.client import ModelClient
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.model.transport import HttpTransport, SseEvent
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.store.memory import MemoryContextStore
from gear_agent.tui_app import GearApp, TextualAgentLoopEventSink


def _sse_event(payload: dict[str, Any]) -> SseEvent:
    return SseEvent(event="message", data=json.dumps(payload))


def _completed_message(text: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


class EventSequenceTransport(HttpTransport):
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        raise AssertionError("Streaming test unexpectedly used JSON transport.")

    def post_sse(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
        idle_timeout_seconds: float,
    ) -> Iterator[SseEvent]:
        for event in self._events:
            yield _sse_event(event)


class PausedCanonicalTransport(HttpTransport):
    def __init__(self) -> None:
        self.delta_emitted = threading.Event()
        self.release_completion = threading.Event()

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        raise AssertionError("Streaming test unexpectedly used JSON transport.")

    def post_sse(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
        idle_timeout_seconds: float,
    ) -> Iterator[SseEvent]:
        yield _sse_event(
            {
                "type": "response.created",
                "response": {"id": "resp_1", "status": "in_progress", "output": []},
            }
        )
        yield _sse_event(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            }
        )
        yield _sse_event(
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "draft fragment",
            }
        )
        self.delta_emitted.set()
        if not self.release_completion.wait(timeout=2.0):
            raise AssertionError("Test did not release the model stream.")
        yield _sse_event(
            {
                "type": "response.output_text.done",
                "output_index": 0,
                "content_index": 0,
                "text": "canonical answer",
            }
        )
        message = _completed_message("canonical answer")
        yield _sse_event(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": message,
            }
        )
        yield _sse_event(
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [message],
                    "usage": {"total_tokens": 7},
                },
            }
        )


class FailingStreamTransport(HttpTransport):
    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        raise AssertionError("Streaming test unexpectedly used JSON transport.")

    def post_sse(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
        idle_timeout_seconds: float,
    ) -> Iterator[SseEvent]:
        yield _sse_event(
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "unsuccessful partial text",
            }
        )
        yield _sse_event(
            {
                "type": "error",
                "code": "server_error",
                "message": "stream failed",
            }
        )


class RecordingAgentEventSink:
    def __init__(self) -> None:
        self.events: list[AgentLoopEvent] = []

    def publish(self, event: AgentLoopEvent) -> None:
        self.events.append(event)


def _streaming_model_config() -> ModelConfig:
    return ModelConfig(
        url="https://example.test/v1/responses",
        model="test-model",
        api_key=None,
        reasoning_replay=ReasoningReplayMode.NONE,
        stream=True,
    )


def _screen_text(app: GearApp) -> str:
    return html.unescape(
        "".join(re.findall(r"<text[^>]*>(.*?)</text>", app.export_screenshot()))
    ).replace("\u00a0", " ")


def _create_app(
    workspace: Path,
    transport: HttpTransport,
) -> tuple[GearApp, TextualAgentLoopEventSink, JsonlContextStore]:
    store = JsonlContextStore(workspace / "sessions")
    model_config = _streaming_model_config()
    progress_sink = TextualAgentLoopEventSink()
    adapter = ResponsesModelAdapter(ModelClient(transport), model_config)
    app = GearApp(
        model=model_config.model,
        session_id="session-1",
        workspace=workspace,
        agent_loop=AgentLoop(
            adapter,
            [],
            store,
            progress_sink,
        ),
        compaction=CompactionService(adapter),
        store=store,
        runtime=RuntimeConfig(
            workdir=workspace,
            session_dir=workspace / "sessions",
            network_enabled=False,
            max_iterations=2,
            model_timeout_seconds=30,
            model_stream_idle_timeout_seconds=5,
        ),
        model_config=model_config,
    )
    progress_sink.bind(app)
    return app, progress_sink, store


class AgentModelProgressTests(unittest.TestCase):
    def test_agent_loop_publishes_only_public_displayable_model_deltas(self) -> None:
        reasoning_item = {
            "id": "reasoning_1",
            "type": "reasoning",
            "status": "completed",
            "summary": [{"type": "summary_text", "text": "public summary"}],
            "content": [{"type": "reasoning_text", "text": "private thought"}],
            "encrypted_content": "opaque-secret",
        }
        message = _completed_message("final answer")
        transport = EventSequenceTransport(
            [
                {
                    "type": "response.reasoning_summary_text.delta",
                    "output_index": 0,
                    "summary_index": 0,
                    "delta": "public summary",
                },
                {
                    "type": "response.reasoning_text.delta",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "private thought",
                },
                {
                    "type": "response.output_text.delta",
                    "output_index": 1,
                    "content_index": 0,
                    "delta": "final answer",
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": reasoning_item,
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 1,
                    "item": message,
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "output": [reasoning_item, message],
                    },
                },
            ]
        )
        sink = RecordingAgentEventSink()
        store = MemoryContextStore()

        AgentLoop(
            ResponsesModelAdapter(ModelClient(transport), _streaming_model_config()),
            [],
            store,
            sink,
        ).run_turn("session-1", "hello", 2, 30, 5)

        public_reasoning = [
            event for event in sink.events if isinstance(event, AgentReasoningSummaryDelta)
        ]
        text = [event for event in sink.events if isinstance(event, AgentTextDelta)]
        self.assertEqual(
            public_reasoning,
            [
                AgentReasoningSummaryDelta(
                    session_id="session-1",
                    iteration=1,
                    delta="public summary",
                )
            ],
        )
        self.assertEqual(
            text,
            [
                AgentTextDelta(
                    session_id="session-1",
                    iteration=1,
                    delta="final answer",
                )
            ],
        )
        self.assertNotIn("private thought", str(sink.events))
        self.assertNotIn("opaque-secret", str(sink.events))
        self.assertEqual(
            [event["kind"] for event in store.events],
            ["user_input", "model_response", "assistant_message"],
        )


class LiveModelProgressUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_and_reasoning_render_separately_without_reasoning_required(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app, sink, _ = _create_app(Path(directory), EventSequenceTransport([]))

            async with app.run_test(size=(100, 30)) as pilot:
                sink.publish(ModelRequestStarted("session-1", 1))
                sink.publish(AgentReasoningSummaryDelta("session-1", 1, "checking files"))
                sink.publish(AgentTextDelta("session-1", 1, "answer text"))
                await pilot.pause()

                screen_text = _screen_text(app)
                self.assertIn("gear thinking", screen_text)
                self.assertIn("checking files", screen_text)
                self.assertIn("gear", screen_text)
                self.assertIn("answer text", screen_text)

                sink.publish(ModelRequestStarted("session-1", 2))
                sink.publish(AgentTextDelta("session-1", 2, "second answer"))
                await pilot.pause()

                screen_text = _screen_text(app)
                self.assertNotIn("checking files", screen_text)
                self.assertIn("second answer", screen_text)

    async def test_tool_transition_and_next_iteration_reset_transient_buffers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app, sink, _ = _create_app(Path(directory), EventSequenceTransport([]))

            async with app.run_test(size=(100, 34)) as pilot:
                sink.publish(ModelRequestStarted("session-1", 1))
                sink.publish(AgentTextDelta("session-1", 1, "before tool"))
                sink.publish(
                    ToolUseStarted(
                        "session-1",
                        1,
                        "call_1",
                        "shell",
                        {"command": "pwd", "workdir": ".", "timeout_seconds": 30},
                    )
                )
                sink.publish(
                    ToolUseFinished(
                        "session-1",
                        1,
                        "call_1",
                        "shell",
                        {"exit_code": 0, "stdout": ".\n", "stderr": "", "timed_out": False},
                    )
                )
                sink.publish(ModelRequestStarted("session-1", 2))
                sink.publish(AgentTextDelta("session-1", 2, "after tool"))
                await pilot.pause()

                screen_text = _screen_text(app)
                self.assertNotIn("before tool", screen_text)
                self.assertIn("tool shell completed", screen_text)
                self.assertIn("after tool", screen_text)

    async def test_high_frequency_and_partial_markdown_use_one_live_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app, sink, _ = _create_app(Path(directory), EventSequenceTransport([]))

            async with app.run_test(size=(120, 30)) as pilot:
                sink.publish(ModelRequestStarted("session-1", 1))
                for fragment in ["x"] * 100:
                    sink.publish(AgentTextDelta("session-1", 1, fragment))
                sink.publish(AgentTextDelta("session-1", 1, " `unfinished [link]("))
                await pilot.pause()

                screen_text = _screen_text(app)
                self.assertIn("x" * 100, screen_text)
                self.assertIn("unfinished", screen_text)
                self.assertLess(len(app.query_one("#chat", RichLog).lines), 10)

    async def test_canonical_history_replaces_transient_text_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transport = PausedCanonicalTransport()
            app, _, store = _create_app(Path(directory), transport)

            async with app.run_test(size=(100, 30)) as pilot:
                input_widget = app.query_one(Input)
                input_widget.value = "start"
                await pilot.press("enter")
                emitted = await asyncio.to_thread(transport.delta_emitted.wait, 1.0)
                self.assertTrue(emitted)
                await pilot.pause()

                self.assertIn("draft fragment", _screen_text(app))
                self.assertTrue(input_widget.disabled)

                transport.release_completion.set()
                await app.workers.wait_for_complete()
                await pilot.pause()

                screen_text = _screen_text(app)
                self.assertNotIn("draft fragment", screen_text)
                self.assertEqual(screen_text.count("canonical answer"), 1)
                self.assertFalse(input_widget.disabled)

            events = store.load("session-1")
            self.assertEqual(events[-1]["kind"], "assistant_message")
            self.assertEqual(events[-1]["payload"], {"text": "canonical answer"})
            self.assertNotIn("draft fragment", str(events))

    async def test_stream_failure_clears_partial_text_and_reenables_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app, _, store = _create_app(Path(directory), FailingStreamTransport())

            async with app.run_test(size=(100, 30)) as pilot:
                input_widget = app.query_one(Input)
                input_widget.value = "start"
                await pilot.press("enter")
                await app.workers.wait_for_complete()
                await pilot.pause()

                screen_text = _screen_text(app)
                self.assertNotIn("unsuccessful partial text", screen_text)
                self.assertIn("Model endpoint emitted a stream error event", screen_text)
                self.assertFalse(input_widget.disabled)

            events = store.load("session-1")
            self.assertEqual(events[-1]["kind"], "turn_error")
            self.assertNotIn("unsuccessful partial text", str(events))


if __name__ == "__main__":
    unittest.main()
