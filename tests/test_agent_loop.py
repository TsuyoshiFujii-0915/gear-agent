import unittest
from typing import Any

from gear_agent.agent.events import (
    AgentLoopEvent,
    ModelRequestStarted,
    SilentAgentLoopEventSink,
    ToolUseFinished,
    ToolUseStarted,
)
from gear_agent.agent.history import FUNCTION_CALL_OUTPUT_HISTORY_MAX_CHARS
from gear_agent.agent.loop import AgentLoop
from gear_agent.config import ModelConfig
from gear_agent.errors import GearError, gear_error
from gear_agent.model.client import ModelClient
from gear_agent.model.transport import HttpTransport
from gear_agent.store.memory import MemoryContextStore
from gear_agent.tools.base import Tool


class SequencedTransport(HttpTransport):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, Any]] = []

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        self.payloads.append(payload)
        return self.responses.pop(0)


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": "echo",
            "description": "Echo text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        return {"text": arguments["text"]}


class LargeOutputTool(Tool):
    @property
    def name(self) -> str:
        return "large_output"

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": "large_output",
            "description": "Return large text.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        return {"text": "x" * (FUNCTION_CALL_OUTPUT_HISTORY_MAX_CHARS + 20)}


class RecoverableFailingTool(Tool):
    @property
    def name(self) -> str:
        return "recoverable_fail"

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": self.name,
            "description": "Fail recoverably.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        raise gear_error(
            "path_outside_workspace",
            "Path is outside the workspace.",
            self.name,
            True,
            {"path": "/testbed"},
        )


class UnrecoverableFailingTool(Tool):
    @property
    def name(self) -> str:
        return "unrecoverable_fail"

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": self.name,
            "description": "Fail unrecoverably.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        raise gear_error("fatal_tool_error", "Fatal tool error.", self.name, False, {})


class RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[AgentLoopEvent] = []

    def publish(self, event: AgentLoopEvent) -> None:
        self.events.append(event)


class AgentLoopTests(unittest.TestCase):
    def test_runs_tool_call_and_returns_final_text(self) -> None:
        transport = SequencedTransport(
            [
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "echo",
                            "arguments": '{"text": "ok"}',
                        }
                    ]
                },
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "done"}],
                        }
                    ]
                },
            ]
        )
        client = ModelClient(transport)
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )
        store = MemoryContextStore()
        event_sink = RecordingEventSink()
        loop = AgentLoop(client, config, [EchoTool()], store, event_sink)

        result = loop.run_turn("session-1", "hello", 4, 30)

        self.assertEqual(result.final_text, "done")
        self.assertEqual(len(transport.payloads), 2)
        self.assertIn("function_call_output", str(transport.payloads[1]["input"]))
        self.assertIn("Use workspace-relative paths", str(transport.payloads[0]["instructions"]))
        self.assertEqual(
            event_sink.events,
            [
                ModelRequestStarted(session_id="session-1", iteration=1),
                ToolUseStarted(
                    session_id="session-1",
                    iteration=1,
                    call_id="call_1",
                    name="echo",
                    arguments={"text": "ok"},
                ),
                ToolUseFinished(
                    session_id="session-1",
                    iteration=1,
                    call_id="call_1",
                    name="echo",
                    result={"text": "ok"},
                ),
                ModelRequestStarted(session_id="session-1", iteration=2),
            ],
        )

    def test_includes_previous_turn_history_in_next_model_request(self) -> None:
        transport = SequencedTransport(
            [
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "echo",
                            "arguments": '{"text": "ok"}',
                        }
                    ]
                },
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "done"}],
                        }
                    ]
                },
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "continued"}],
                        }
                    ]
                },
            ]
        )
        client = ModelClient(transport)
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )
        store = MemoryContextStore()
        event_sink = SilentAgentLoopEventSink()
        loop = AgentLoop(client, config, [EchoTool()], store, event_sink)

        loop.run_turn("session-1", "hello", 4, 30)
        result = loop.run_turn("session-1", "continue", 4, 30)

        self.assertEqual(result.final_text, "continued")
        self.assertEqual(len(transport.payloads), 3)
        next_turn_input = transport.payloads[2]["input"]
        self.assertEqual(
            next_turn_input,
            [
                {"role": "user", "content": "hello"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "echo",
                    "arguments": '{"text": "ok"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"text": "ok"}',
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done"}],
                },
                {"role": "user", "content": "continue"},
            ],
        )

    def test_compaction_summary_replaces_earlier_history_in_next_model_request(
        self,
    ) -> None:
        transport = SequencedTransport(
            [
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "continued"}],
                        }
                    ]
                }
            ]
        )
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )
        store = MemoryContextStore()
        store.append("session-1", "user_input", {"text": "old request"})
        store.append("session-1", "assistant_message", {"text": "old response"})
        store.append("session-1", "compaction_summary", {"text": "saved summary"})
        store.append("session-1", "user_input", {"text": "work after summary"})
        store.append("session-1", "assistant_message", {"text": "post-summary answer"})
        loop = AgentLoop(
            ModelClient(transport),
            config,
            [],
            store,
            SilentAgentLoopEventSink(),
        )

        result = loop.run_turn("session-1", "continue", 4, 30)

        self.assertEqual(result.final_text, "continued")
        self.assertEqual(
            transport.payloads[0]["input"],
            [
                {
                    "role": "user",
                    "content": (
                        "Earlier session context (compressed continuation, not a new user "
                        "request):\n\nsaved summary"
                    ),
                },
                {"role": "user", "content": "work after summary"},
                {"role": "assistant", "content": "post-summary answer"},
                {"role": "user", "content": "continue"},
            ],
        )

    def test_compaction_followed_immediately_by_new_user_turn(self) -> None:
        transport = SequencedTransport(
            [
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "continued"}],
                        }
                    ]
                }
            ]
        )
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )
        store = MemoryContextStore()
        store.append("session-1", "user_input", {"text": "old request"})
        store.append("session-1", "compaction_summary", {"text": "saved summary"})
        loop = AgentLoop(
            ModelClient(transport),
            config,
            [],
            store,
            SilentAgentLoopEventSink(),
        )

        loop.run_turn("session-1", "new request", 4, 30)

        self.assertEqual(
            transport.payloads[0]["input"],
            [
                {
                    "role": "user",
                    "content": (
                        "Earlier session context (compressed continuation, not a new user "
                        "request):\n\nsaved summary"
                    ),
                },
                {"role": "user", "content": "new request"},
            ],
        )

    def test_replays_and_truncates_tool_history_after_compaction(self) -> None:
        transport = SequencedTransport(
            [
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "continued"}],
                        }
                    ]
                }
            ]
        )
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )
        store = MemoryContextStore()
        store.append("session-1", "compaction_summary", {"text": "saved summary"})
        store.append(
            "session-1",
            "model_response",
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_after_summary",
                        "name": "large_output",
                        "arguments": "{}",
                    }
                ]
            },
        )
        store.append(
            "session-1",
            "tool_result",
            {
                "call_id": "call_after_summary",
                "iteration": 1,
                "name": "large_output",
                "result": {"text": "x" * (FUNCTION_CALL_OUTPUT_HISTORY_MAX_CHARS + 20)},
            },
        )
        loop = AgentLoop(
            ModelClient(transport),
            config,
            [],
            store,
            SilentAgentLoopEventSink(),
        )

        loop.run_turn("session-1", "continue", 4, 30)

        request_input = transport.payloads[0]["input"]
        self.assertEqual(request_input[1]["call_id"], "call_after_summary")
        self.assertEqual(request_input[2]["call_id"], "call_after_summary")
        self.assertIn('"truncated": true', request_input[2]["output"])
        self.assertNotIn(
            "x" * (FUNCTION_CALL_OUTPUT_HISTORY_MAX_CHARS + 20),
            request_input[2]["output"],
        )

    def test_rejects_orphan_tool_result_after_compaction(self) -> None:
        transport = SequencedTransport([])
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )
        store = MemoryContextStore()
        store.append(
            "session-1",
            "model_response",
            {"output": "malformed but before checkpoint"},
        )
        store.append("session-1", "compaction_summary", {"text": "saved summary"})
        store.append(
            "session-1",
            "tool_result",
            {
                "call_id": "orphan_call",
                "iteration": 1,
                "name": "echo",
                "result": {"text": "orphan"},
            },
        )
        loop = AgentLoop(
            ModelClient(transport),
            config,
            [],
            store,
            SilentAgentLoopEventSink(),
        )

        with self.assertRaises(GearError) as raised:
            loop.run_turn("session-1", "continue", 4, 30)

        self.assertIn(
            "Stored tool_result has no preceding function_call",
            str(raised.exception),
        )
        self.assertEqual(transport.payloads, [])

    def test_rejects_stored_empty_compaction_checkpoint(self) -> None:
        transport = SequencedTransport([])
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )
        store = MemoryContextStore()
        store.append("session-1", "user_input", {"text": "preserved request"})
        store.append("session-1", "compaction_summary", {"text": "   "})
        loop = AgentLoop(
            ModelClient(transport),
            config,
            [],
            store,
            SilentAgentLoopEventSink(),
        )

        with self.assertRaises(GearError) as raised:
            loop.run_turn("session-1", "continue", 4, 30)

        self.assertIn("compaction_summary.text must not be empty", str(raised.exception))
        self.assertEqual(transport.payloads, [])

    def test_truncates_stored_function_call_output_only_on_later_turns(self) -> None:
        transport = SequencedTransport(
            [
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_large",
                            "name": "large_output",
                            "arguments": "{}",
                        }
                    ]
                },
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "large done"}],
                        }
                    ]
                },
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "after large"}],
                        }
                    ]
                },
            ]
        )
        client = ModelClient(transport)
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )
        store = MemoryContextStore()
        event_sink = SilentAgentLoopEventSink()
        loop = AgentLoop(client, config, [LargeOutputTool()], store, event_sink)

        loop.run_turn("session-1", "produce large output", 4, 30)
        loop.run_turn("session-1", "continue", 4, 30)

        current_turn_output = transport.payloads[1]["input"][2]["output"]
        self.assertIn("x" * (FUNCTION_CALL_OUTPUT_HISTORY_MAX_CHARS + 20), current_turn_output)
        next_turn_output = transport.payloads[2]["input"][2]["output"]
        self.assertIn('"truncated": true', next_turn_output)
        self.assertIn('"original_json_chars"', next_turn_output)
        self.assertNotIn("x" * (FUNCTION_CALL_OUTPUT_HISTORY_MAX_CHARS + 20), next_turn_output)

    def test_recoverable_tool_error_is_returned_to_model(self) -> None:
        transport = SequencedTransport(
            [
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "recoverable_fail",
                            "arguments": "{}",
                        }
                    ]
                },
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "retried"}],
                        }
                    ]
                },
            ]
        )
        client = ModelClient(transport)
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )
        store = MemoryContextStore()
        event_sink = RecordingEventSink()
        loop = AgentLoop(client, config, [RecoverableFailingTool()], store, event_sink)

        result = loop.run_turn("session-1", "hello", 4, 30)

        self.assertEqual(result.final_text, "retried")
        self.assertEqual(len(transport.payloads), 2)
        second_input = transport.payloads[1]["input"]
        self.assertIn("function_call_output", str(second_input))
        self.assertIn("path_outside_workspace", str(second_input))
        self.assertIn("/testbed", str(second_input))
        self.assertEqual(len(event_sink.events), 4)
        self.assertEqual(
            event_sink.events[1],
            ToolUseStarted(
                session_id="session-1",
                iteration=1,
                call_id="call_1",
                name="recoverable_fail",
                arguments={},
            ),
        )
        self.assertEqual(
            event_sink.events[2],
            ToolUseFinished(
                session_id="session-1",
                iteration=1,
                call_id="call_1",
                name="recoverable_fail",
                result={
                    "error": {
                        "type": "path_outside_workspace",
                        "message": "Path is outside the workspace.",
                        "origin": "recoverable_fail",
                        "details": {"path": "/testbed"},
                    }
                },
            ),
        )
        self.assertEqual(
            event_sink.events[3],
            ModelRequestStarted(session_id="session-1", iteration=2),
        )

    def test_unrecoverable_tool_error_is_not_returned_to_model(self) -> None:
        transport = SequencedTransport(
            [
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "unrecoverable_fail",
                            "arguments": "{}",
                        }
                    ]
                }
            ]
        )
        client = ModelClient(transport)
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )
        store = MemoryContextStore()
        event_sink = SilentAgentLoopEventSink()
        loop = AgentLoop(client, config, [UnrecoverableFailingTool()], store, event_sink)

        with self.assertRaises(GearError):
            loop.run_turn("session-1", "hello", 4, 30)

        self.assertEqual(len(transport.payloads), 1)

    def test_retries_once_when_model_returns_reasoning_without_final_text(self) -> None:
        transport = SequencedTransport(
            [
                {
                    "output": [
                        {
                            "type": "reasoning",
                            "status": "completed",
                            "summary": [],
                            "content": [
                                {
                                    "type": "reasoning_text",
                                    "text": "I have enough information to answer.",
                                }
                            ],
                        }
                    ]
                },
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "final answer"}],
                        }
                    ]
                },
            ]
        )
        client = ModelClient(transport)
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )
        store = MemoryContextStore()
        event_sink = RecordingEventSink()
        loop = AgentLoop(client, config, [], store, event_sink)

        result = loop.run_turn("session-1", "hello", 4, 30)

        self.assertEqual(result.final_text, "final answer")
        self.assertEqual(result.iterations, 2)
        self.assertEqual(len(transport.payloads), 2)
        retry_input = transport.payloads[1]["input"]
        self.assertIn("reasoning", str(retry_input))
        self.assertIn("final answer as output_text", str(retry_input))
        stored_events = store.load("session-1")
        assistant_messages = [
            event for event in stored_events if event.get("kind") == "assistant_message"
        ]
        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(assistant_messages[0]["payload"], {"text": "final answer"})

    def test_fails_when_finalization_retry_also_has_no_final_text(self) -> None:
        transport = SequencedTransport(
            [
                {
                    "output": [
                        {
                            "type": "reasoning",
                            "status": "completed",
                            "summary": [],
                            "content": [
                                {"type": "reasoning_text", "text": "I should answer."}
                            ],
                        }
                    ]
                },
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "\n\n"}],
                        }
                    ]
                },
            ]
        )
        client = ModelClient(transport)
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )
        store = MemoryContextStore()
        event_sink = RecordingEventSink()
        loop = AgentLoop(client, config, [], store, event_sink)

        with self.assertRaises(GearError) as error:
            loop.run_turn("session-1", "hello", 4, 30)

        self.assertEqual(error.exception.origin, "agent_loop")
        self.assertEqual(error.exception.error_type, "final_text_missing")
        self.assertEqual(len(transport.payloads), 2)


if __name__ == "__main__":
    unittest.main()
