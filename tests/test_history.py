import unittest
from typing import Any

from gear_agent.agent.history import build_model_history
from gear_agent.config import ReasoningReplayMode
from gear_agent.errors import GearError
from gear_agent.model.replay import ModelReplayScope, ReasoningReplayPolicy


CURRENT_SCOPE = ModelReplayScope(
    protocol="responses",
    endpoint_identity="sha256:ee0291cefbb5b6136483fb38ba9efe9264f9b685d5006c273e293a54b43a1883",
    model="gpt-5.5",
)


def _policy(mode: ReasoningReplayMode) -> ReasoningReplayPolicy:
    return ReasoningReplayPolicy(mode=mode, current_scope=CURRENT_SCOPE)


def _encrypted_response_payload(
    output: list[dict[str, Any]],
    scope: ModelReplayScope,
) -> dict[str, Any]:
    return {
        "response": {"output": output},
        "source": {
            "reasoning_replay": "encrypted",
            "replay_scope": {
                "protocol": scope.protocol,
                "endpoint_identity": scope.endpoint_identity,
                "model": scope.model,
            },
        },
    }


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
                "payload": _encrypted_response_payload(
                    [reasoning_item, message_item, future_item],
                    CURRENT_SCOPE,
                ),
            },
            {"kind": "assistant_message", "payload": {"text": "done"}},
        ]

        history = build_model_history(
            events,
            _policy(ReasoningReplayMode.ENCRYPTED),
        )

        self.assertEqual(
            history.items,
            [
                {"role": "user", "content": "hello"},
                reasoning_item,
                message_item,
                future_item,
            ],
        )
        self.assertEqual(history.diagnostic.reused_encrypted_items, 1)
        self.assertEqual(history.diagnostic.dropped_encrypted_items, 0)

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

        history = build_model_history(events, _policy(ReasoningReplayMode.NONE))

        self.assertEqual(
            history.items,
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

        history = build_model_history(events, _policy(ReasoningReplayMode.NONE))

        self.assertEqual(
            history.items,
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

        history = build_model_history(events, _policy(ReasoningReplayMode.NONE))

        self.assertEqual(
            history.items,
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
            build_model_history(events, _policy(ReasoningReplayMode.NONE))

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
            build_model_history(events, _policy(ReasoningReplayMode.NONE))

        self.assertEqual(raised.exception.error_type, "history_shape_invalid")
        self.assertEqual(raised.exception.origin, "assistant_message")
        self.assertIn("does not match", str(raised.exception))

    def test_drops_only_encrypted_content_when_model_scope_changes(self) -> None:
        reasoning_item = {
            "type": "reasoning",
            "id": "reasoning_1",
            "status": "completed",
            "summary": [{"type": "summary_text", "text": "portable summary"}],
            "encrypted_content": "opaque-state",
        }
        stored_scope = ModelReplayScope(
            protocol="responses",
            endpoint_identity=CURRENT_SCOPE.endpoint_identity,
            model="gpt-5.4",
        )
        events = [
            {
                "kind": "model_response",
                "payload": _encrypted_response_payload([reasoning_item], stored_scope),
            }
        ]

        history = build_model_history(
            events,
            _policy(ReasoningReplayMode.ENCRYPTED),
        )

        self.assertEqual(
            history.items,
            [
                {
                    "type": "reasoning",
                    "id": "reasoning_1",
                    "status": "completed",
                    "summary": [
                        {"type": "summary_text", "text": "portable summary"}
                    ],
                }
            ],
        )
        self.assertEqual(history.diagnostic.reused_encrypted_items, 0)
        self.assertEqual(history.diagnostic.dropped_incompatible_scope_items, 1)
        self.assertEqual(reasoning_item["encrypted_content"], "opaque-state")

    def test_drops_encrypted_content_when_endpoint_scope_changes(self) -> None:
        stored_scope = ModelReplayScope(
            protocol="responses",
            endpoint_identity="https://gateway.example/v1/responses",
            model=CURRENT_SCOPE.model,
        )
        events = [
            {
                "kind": "model_response",
                "payload": _encrypted_response_payload(
                    [
                        {
                            "type": "reasoning",
                            "summary": [],
                            "encrypted_content": "opaque-state",
                        }
                    ],
                    stored_scope,
                ),
            }
        ]

        history = build_model_history(
            events,
            _policy(ReasoningReplayMode.ENCRYPTED),
        )

        self.assertNotIn("encrypted_content", history.items[0])
        self.assertEqual(history.diagnostic.dropped_incompatible_scope_items, 1)

    def test_treats_legacy_response_without_scope_as_incompatible(self) -> None:
        events = [
            {
                "kind": "model_response",
                "payload": {
                    "output": [
                        {
                            "type": "reasoning",
                            "summary": [
                                {"type": "summary_text", "text": "legacy summary"}
                            ],
                            "encrypted_content": "legacy-opaque-state",
                        }
                    ]
                },
            }
        ]

        history = build_model_history(
            events,
            _policy(ReasoningReplayMode.ENCRYPTED),
        )

        self.assertNotIn("encrypted_content", history.items[0])
        self.assertEqual(
            history.items[0]["summary"],
            [{"type": "summary_text", "text": "legacy summary"}],
        )
        self.assertEqual(history.diagnostic.dropped_missing_scope_items, 1)

    def test_disabled_replay_drops_compatible_encrypted_content(self) -> None:
        events = [
            {
                "kind": "model_response",
                "payload": _encrypted_response_payload(
                    [
                        {
                            "type": "reasoning",
                            "summary": [],
                            "encrypted_content": "opaque-state",
                        }
                    ],
                    CURRENT_SCOPE,
                ),
            }
        ]

        history = build_model_history(events, _policy(ReasoningReplayMode.NONE))

        self.assertNotIn("encrypted_content", history.items[0])
        self.assertEqual(history.diagnostic.dropped_disabled_items, 1)

    def test_compaction_checkpoint_excludes_pre_checkpoint_opaque_state(self) -> None:
        events = [
            {
                "kind": "model_response",
                "payload": _encrypted_response_payload(
                    [
                        {
                            "type": "reasoning",
                            "encrypted_content": "pre-checkpoint-opaque",
                        }
                    ],
                    CURRENT_SCOPE,
                ),
            },
            {"kind": "compaction_summary", "payload": {"text": "saved summary"}},
            {"kind": "user_input", "payload": {"text": "new work"}},
        ]

        history = build_model_history(
            events,
            _policy(ReasoningReplayMode.ENCRYPTED),
        )

        self.assertNotIn("pre-checkpoint-opaque", str(history.items))
        self.assertEqual(history.diagnostic.reused_encrypted_items, 0)
        self.assertEqual(history.diagnostic.dropped_encrypted_items, 0)

    def test_rejects_malformed_model_response_envelope(self) -> None:
        events = [
            {
                "kind": "model_response",
                "payload": {"response": {"output": []}},
            }
        ]

        with self.assertRaises(GearError) as raised:
            build_model_history(
                events,
                _policy(ReasoningReplayMode.ENCRYPTED),
            )

        self.assertEqual(raised.exception.error_type, "history_shape_invalid")
        self.assertEqual(raised.exception.origin, "model_response.source")


if __name__ == "__main__":
    unittest.main()
