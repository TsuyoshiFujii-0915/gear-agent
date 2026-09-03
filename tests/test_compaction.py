import unittest
from typing import Any

from gear_agent.agent.compaction import CompactionService
from gear_agent.config import ModelConfig
from gear_agent.errors import GearError
from gear_agent.model.client import ModelClient
from gear_agent.model.transport import HttpTransport
from gear_agent.store.memory import MemoryContextStore


class CompactionTransport(HttpTransport):
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


class CompactionTests(unittest.TestCase):
    def test_compacts_existing_events_into_summary_event(self) -> None:
        transport = CompactionTransport(
            [
                {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "summary"}],
                    }
                ]
                }
            ]
        )
        store = MemoryContextStore()
        store.append("session-1", "user_input", {"text": "hello"})
        service = CompactionService(ModelClient(transport))
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )

        summary = service.compact("session-1", store, config, 30)

        self.assertEqual(summary, "summary")
        self.assertEqual(store.events[-1]["kind"], "compaction_summary")
        self.assertIn("hello", transport.payloads[0]["input"])

    def test_repeated_compaction_uses_effective_context_and_preserves_raw_events(
        self,
    ) -> None:
        transport = CompactionTransport(
            [
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "summary-one"}],
                        }
                    ]
                },
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "summary-two"}],
                        }
                    ]
                },
            ]
        )
        store = MemoryContextStore()
        store.append("session-1", "user_input", {"text": "pre-checkpoint-secret"})
        service = CompactionService(ModelClient(transport))
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )

        first_summary = service.compact("session-1", store, config, 30)
        store.append("session-1", "user_input", {"text": "post-checkpoint-work"})
        second_summary = service.compact("session-1", store, config, 30)

        self.assertEqual(first_summary, "summary-one")
        self.assertEqual(second_summary, "summary-two")
        self.assertIn("pre-checkpoint-secret", transport.payloads[0]["input"])
        self.assertNotIn("pre-checkpoint-secret", transport.payloads[1]["input"])
        self.assertIn("summary-one", transport.payloads[1]["input"])
        self.assertIn("post-checkpoint-work", transport.payloads[1]["input"])
        self.assertEqual(
            [event["kind"] for event in store.events],
            [
                "user_input",
                "compaction_summary",
                "user_input",
                "compaction_summary",
            ],
        )
        self.assertEqual(
            store.events[0]["payload"],
            {"text": "pre-checkpoint-secret"},
        )

    def test_compaction_failure_does_not_delete_existing_events(self) -> None:
        transport = CompactionTransport([{"output": "bad"}])
        store = MemoryContextStore()
        store.append("session-1", "user_input", {"text": "hello"})
        service = CompactionService(ModelClient(transport))
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )

        with self.assertRaises(GearError):
            service.compact("session-1", store, config, 30)

        self.assertEqual(len(store.events), 1)

    def test_empty_compaction_summary_fails_without_creating_checkpoint(self) -> None:
        transport = CompactionTransport(
            [
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "   "}],
                        }
                    ]
                }
            ]
        )
        store = MemoryContextStore()
        store.append("session-1", "user_input", {"text": "keep this context"})
        service = CompactionService(ModelClient(transport))
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )

        with self.assertRaises(GearError) as raised:
            service.compact("session-1", store, config, 30)

        self.assertIn("Compaction response did not contain a summary", str(raised.exception))
        self.assertEqual(
            [event["kind"] for event in store.events],
            ["user_input"],
        )

    def test_compaction_includes_only_post_checkpoint_turn_errors(self) -> None:
        transport = CompactionTransport(
            [
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "summary-two"}],
                        }
                    ]
                }
            ]
        )
        store = MemoryContextStore()
        store.append("session-1", "turn_error", {"message": "obsolete error"})
        store.append("session-1", "compaction_summary", {"text": "summary-one"})
        store.append("session-1", "turn_error", {"message": "current error"})
        service = CompactionService(ModelClient(transport))
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )

        service.compact("session-1", store, config, 30)

        self.assertNotIn("obsolete error", transport.payloads[0]["input"])
        self.assertIn("summary-one", transport.payloads[0]["input"])
        self.assertIn("current error", transport.payloads[0]["input"])


if __name__ == "__main__":
    unittest.main()
