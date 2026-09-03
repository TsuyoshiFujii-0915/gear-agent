import unittest
from typing import Any

from gear_agent.agent.compaction import CompactionService
from gear_agent.config import ModelConfig
from gear_agent.errors import GearError
from gear_agent.model.client import ModelClient
from gear_agent.model.transport import HttpTransport
from gear_agent.store.memory import MemoryContextStore


class CompactionTransport(HttpTransport):
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.payloads: list[dict[str, Any]] = []

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        self.payloads.append(payload)
        return self.response


class CompactionTests(unittest.TestCase):
    def test_compacts_existing_events_into_summary_event(self) -> None:
        transport = CompactionTransport(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "summary"}],
                    }
                ]
            }
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

    def test_compaction_failure_does_not_delete_existing_events(self) -> None:
        transport = CompactionTransport({"output": "bad"})
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


if __name__ == "__main__":
    unittest.main()
