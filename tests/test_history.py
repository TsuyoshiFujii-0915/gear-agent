import unittest
from typing import Any

from gear_agent.agent.history import build_model_history
from gear_agent.errors import GearError


class ModelHistoryTests(unittest.TestCase):
    def test_replays_complete_output_without_duplicate_assistant_message(self) -> None:
        reasoning_item = {
            "type": "reasoning",
            "id": "reasoning_1",
            "status": "completed",
            "summary": [],
            "encrypted_content": "opaque-state",
        }
        message_item = {
            "type": "message",
            "id": "message_1",
            "role": "assistant",
            "status": "completed",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "done"}],
        }
        future_item = {
            "type": "provider_future_item",
            "id": "future_1",
            "provider_metadata": {"preserve": True},
        }
        events: list[dict[str, Any]] = [
            {"kind": "user_input", "payload": {"text": "hello"}},
            {
                "kind": "model_response",
                "payload": {"output": [reasoning_item, message_item, future_item]},
            },
            {"kind": "assistant_message", "payload": {"text": "done"}},
        ]

        history = build_model_history(events)

        self.assertEqual(
            history,
            [
                {"role": "user", "content": "hello"},
                reasoning_item,
                message_item,
                future_item,
            ],
        )

    def test_replays_multiple_function_calls_and_results_in_conversation_order(
        self,
    ) -> None:
        reasoning_item = {
            "type": "reasoning",
            "id": "reasoning_1",
            "summary": [],
        }
        first_call = {
            "type": "function_call",
            "call_id": "call_1",
            "name": "echo",
            "arguments": '{"text": "one"}',
        }
        second_call = {
            "type": "function_call",
            "call_id": "call_2",
            "name": "echo",
            "arguments": '{"text": "two"}',
        }
        message_item = {
            "type": "message",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "complete"}],
        }
        events: list[dict[str, Any]] = [
            {
                "kind": "model_response",
                "payload": {"output": [reasoning_item, first_call, second_call]},
            },
            {
                "kind": "tool_call",
                "payload": {"call_id": "call_1", "name": "echo"},
            },
            {
                "kind": "tool_result",
                "payload": {"call_id": "call_1", "result": {"text": "one"}},
            },
            {
                "kind": "tool_call",
                "payload": {"call_id": "call_2", "name": "echo"},
            },
            {
                "kind": "tool_result",
                "payload": {"call_id": "call_2", "result": {"text": "two"}},
            },
            {
                "kind": "model_response",
                "payload": {"output": [message_item]},
            },
            {"kind": "assistant_message", "payload": {"text": "complete"}},
        ]

        history = build_model_history(events)

        self.assertEqual(
            history,
            [
                reasoning_item,
                first_call,
                second_call,
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"text": "one"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_2",
                    "output": '{"text": "two"}',
                },
                message_item,
            ],
        )

    def test_replays_legacy_assistant_message_without_raw_message_item(self) -> None:
        events: list[dict[str, Any]] = [
            {"kind": "user_input", "payload": {"text": "legacy request"}},
            {"kind": "assistant_message", "payload": {"text": "legacy answer"}},
        ]

        history = build_model_history(events)

        self.assertEqual(
            history,
            [
                {"role": "user", "content": "legacy request"},
                {"role": "assistant", "content": "legacy answer"},
            ],
        )

    def test_replays_complete_output_after_latest_compaction_checkpoint(self) -> None:
        reasoning_item = {"type": "reasoning", "id": "reasoning_after_summary"}
        message_item = {
            "type": "message",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "new answer"}],
        }
        events: list[dict[str, Any]] = [
            {"kind": "model_response", "payload": {"output": "obsolete malformed"}},
            {"kind": "compaction_summary", "payload": {"text": "saved summary"}},
            {
                "kind": "model_response",
                "payload": {"output": [reasoning_item, message_item]},
            },
            {"kind": "assistant_message", "payload": {"text": "new answer"}},
        ]

        history = build_model_history(events)

        self.assertEqual(
            history,
            [
                {
                    "role": "user",
                    "content": (
                        "Earlier session context (compressed continuation, not a new user "
                        "request):\n\nsaved summary"
                    ),
                },
                reasoning_item,
                message_item,
            ],
        )

    def test_rejects_output_item_without_type(self) -> None:
        events: list[dict[str, Any]] = [
            {
                "kind": "model_response",
                "payload": {"output": [{"id": "missing_type"}]},
            }
        ]

        with self.assertRaises(GearError) as raised:
            build_model_history(events)

        self.assertEqual(raised.exception.error_type, "history_shape_invalid")
        self.assertEqual(raised.exception.origin, "model_response.output")
        self.assertIn("type must be a string", str(raised.exception))

    def test_rejects_assistant_message_that_differs_from_raw_output(self) -> None:
        events: list[dict[str, Any]] = [
            {
                "kind": "model_response",
                "payload": {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "raw answer"}],
                        }
                    ]
                },
            },
            {"kind": "assistant_message", "payload": {"text": "different answer"}},
        ]

        with self.assertRaises(GearError) as raised:
            build_model_history(events)

        self.assertEqual(raised.exception.error_type, "history_shape_invalid")
        self.assertEqual(raised.exception.origin, "assistant_message")
        self.assertIn("does not match", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
