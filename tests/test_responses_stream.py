from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Iterator

from gear_agent.config import ModelConfig, ReasoningReplayMode
from gear_agent.errors import GearError
from gear_agent.model.client import ModelClient
from gear_agent.model.events import (
    ModelFunctionCallArgumentsDelta,
    ModelOutputItemCompleted,
    ModelProgressEvent,
    ModelReasoningSummaryDelta,
    ModelReasoningTextDelta,
    ModelTextDelta,
)
from gear_agent.model.responses import extract_function_calls, extract_output_text
from gear_agent.model.transport import HttpTransport, SseEvent


FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "responses"


class FixtureTransport(HttpTransport):
    def __init__(self, fixture_name: str, json_response: dict[str, Any]) -> None:
        self.fixture_name = fixture_name
        self.json_response = json_response
        self.json_calls = 0
        self.stream_calls = 0
        self.stream_payloads: list[dict[str, Any]] = []

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.json_calls += 1
        return self.json_response

    def post_sse(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
        idle_timeout_seconds: float,
    ) -> Iterator[SseEvent]:
        self.stream_calls += 1
        self.stream_payloads.append(payload)
        fixture = (FIXTURE_DIRECTORY / self.fixture_name).read_text(encoding="utf-8")
        for block in fixture.strip().split("\n\n"):
            data_lines = [
                line.removeprefix("data: ")
                for line in block.splitlines()
                if line.startswith("data:")
            ]
            if len(data_lines) > 0:
                yield SseEvent(event="message", data="\n".join(data_lines))


class RecordingModelProgressSink:
    def __init__(self) -> None:
        self.events: list[ModelProgressEvent] = []

    def publish(self, event: ModelProgressEvent) -> None:
        self.events.append(event)


def streaming_config() -> ModelConfig:
    return ModelConfig(
        url="https://example.test/v1/responses",
        model="test-model",
        api_key=None,
        reasoning_replay=ReasoningReplayMode.NONE,
        stream=True,
    )


def create_streamed_response(
    fixture_name: str,
) -> tuple[dict[str, Any], FixtureTransport, RecordingModelProgressSink]:
    transport = FixtureTransport(fixture_name, {})
    sink = RecordingModelProgressSink()
    response = ModelClient(transport, sink).create_response(
        streaming_config(),
        "hello",
        [],
        "Follow instructions.",
        30.0,
        5.0,
    )
    return response, transport, sink


class ResponsesStreamTests(unittest.TestCase):
    def test_returns_terminal_text_response_with_usage_as_canonical(self) -> None:
        response, transport, sink = create_streamed_response("text.sse")

        self.assertEqual(extract_output_text(response), "こんにちは")
        self.assertEqual(response["usage"]["total_tokens"], 5)
        self.assertEqual(transport.stream_calls, 1)
        self.assertEqual(transport.json_calls, 0)
        self.assertIs(transport.stream_payloads[0]["stream"], True)
        text_events = [event for event in sink.events if isinstance(event, ModelTextDelta)]
        self.assertEqual([event.delta for event in text_events], ["こんにちは"])

    def test_emits_reasoning_and_completed_item_progress(self) -> None:
        response, _, sink = create_streamed_response("reasoning_text.sse")

        self.assertEqual(extract_output_text(response), "Done")
        reasoning_events = [
            event
            for event in sink.events
            if isinstance(event, ModelReasoningSummaryDelta)
        ]
        completed_events = [
            event for event in sink.events if isinstance(event, ModelOutputItemCompleted)
        ]
        self.assertEqual([event.delta for event in reasoning_events], ["Inspecting"])
        self.assertEqual([event.output_index for event in completed_events], [0, 1])

    def test_distinguishes_private_reasoning_text_from_reasoning_summary(self) -> None:
        response, _, sink = create_streamed_response("reasoning_raw_text.sse")

        reasoning_item = response["output"][0]
        self.assertEqual(reasoning_item["content"][0]["text"], "Private reasoning")
        private_events = [
            event
            for event in sink.events
            if isinstance(event, ModelReasoningTextDelta)
        ]
        summary_events = [
            event
            for event in sink.events
            if isinstance(event, ModelReasoningSummaryDelta)
        ]
        self.assertEqual([event.delta for event in private_events], ["Private reasoning"])
        self.assertEqual(summary_events, [])

    def test_assembles_fragmented_function_arguments_without_partial_json_parse(self) -> None:
        response, _, sink = create_streamed_response("fragmented_tool.sse")

        calls = extract_function_calls(response)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "shell")
        self.assertEqual(
            calls[0].arguments,
            {"command": "echo hi", "workdir": "."},
        )
        argument_events = [
            event
            for event in sink.events
            if isinstance(event, ModelFunctionCallArgumentsDelta)
        ]
        self.assertEqual(
            "".join(event.delta for event in argument_events),
            '{"command":"echo hi","workdir":"."}',
        )

    def test_assembles_interleaved_calls_by_identity_and_output_order(self) -> None:
        response, _, _ = create_streamed_response("interleaved_tools.sse")

        calls = extract_function_calls(response)

        self.assertEqual([call.call_id for call in calls], ["call_a", "call_b"])
        self.assertEqual([call.arguments for call in calls], [{"text": "A"}, {"text": "B"}])
        self.assertEqual(response["id"], "resp_tools")
        self.assertEqual(response["status"], "completed")

    def test_unknown_event_does_not_break_valid_stream(self) -> None:
        response, _, _ = create_streamed_response("unknown_event.sse")

        self.assertEqual(extract_output_text(response), "ok")

    def test_stream_error_after_partial_output_fails_without_non_stream_retry(self) -> None:
        transport = FixtureTransport("stream_error.sse", {"output": []})

        with self.assertRaises(GearError) as raised:
            ModelClient(transport, RecordingModelProgressSink()).create_response(
                streaming_config(),
                "hello",
                [],
                "Follow instructions.",
                30.0,
                5.0,
            )

        self.assertEqual(raised.exception.error_type, "response_stream_error")
        self.assertEqual(raised.exception.details["code"], "server_error")
        self.assertEqual(transport.stream_calls, 1)
        self.assertEqual(transport.json_calls, 0)

    def test_premature_eof_fails_explicitly(self) -> None:
        with self.assertRaises(GearError) as raised:
            create_streamed_response("premature_eof.sse")

        self.assertEqual(raised.exception.error_type, "response_stream_terminated")

    def test_failed_and_incomplete_terminal_events_fail_explicitly(self) -> None:
        for event_type, expected_error in [
            ("response.failed", "response_failed"),
            ("response.incomplete", "response_incomplete"),
        ]:
            with self.subTest(event_type=event_type):
                terminal = {
                    "type": event_type,
                    "response": {
                        "id": "resp_terminal",
                        "status": event_type.removeprefix("response."),
                        "error": {"code": "terminal", "message": "stopped"},
                    },
                }
                transport = FixtureTransport("text.sse", {})
                transport.post_sse = lambda *args: iter(
                    [SseEvent(event="message", data=json.dumps(terminal))]
                )
                with self.assertRaises(GearError) as raised:
                    ModelClient(transport, RecordingModelProgressSink()).create_response(
                        streaming_config(), "hello", [], "instructions", 30.0, 5.0
                    )
                self.assertEqual(raised.exception.error_type, expected_error)

    def test_rejects_invalid_event_json(self) -> None:
        transport = FixtureTransport("text.sse", {})
        transport.post_sse = lambda *args: iter(
            [SseEvent(event="message", data="not-json")]
        )

        with self.assertRaises(GearError) as raised:
            ModelClient(transport, RecordingModelProgressSink()).create_response(
                streaming_config(), "hello", [], "instructions", 30.0, 5.0
            )

        self.assertEqual(raised.exception.error_type, "response_stream_event_invalid")

    def test_requires_idle_timeout_for_streaming(self) -> None:
        with self.assertRaises(GearError) as raised:
            ModelClient(
                FixtureTransport("text.sse", {}),
                RecordingModelProgressSink(),
            ).create_response(
                streaming_config(), "hello", [], "instructions", 30.0, None
            )

        self.assertEqual(raised.exception.error_type, "stream_idle_timeout_missing")

    def test_non_stream_and_stream_return_equivalent_completed_responses(self) -> None:
        streamed_response, _, _ = create_streamed_response("text.sse")
        transport = FixtureTransport("text.sse", streamed_response)
        non_stream_config = ModelConfig(
            url="https://example.test/v1/responses",
            model="test-model",
            api_key=None,
            reasoning_replay=ReasoningReplayMode.NONE,
            stream=False,
        )

        non_stream_response = ModelClient(
            transport,
            RecordingModelProgressSink(),
        ).create_response(
            non_stream_config,
            "hello",
            [],
            "Follow instructions.",
            30.0,
            None,
        )

        self.assertEqual(non_stream_response, streamed_response)
        self.assertEqual(transport.json_calls, 1)
        self.assertEqual(transport.stream_calls, 0)


if __name__ == "__main__":
    unittest.main()
