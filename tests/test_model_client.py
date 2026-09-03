import unittest
from typing import Any

from gear_agent.config import ModelConfig
from gear_agent.model.client import ModelClient
from gear_agent.model.transport import HttpTransport


class RecordingTransport(HttpTransport):
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, str], dict[str, Any], int]] = []

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        self.calls.append((url, headers, payload, timeout_seconds))
        return self.response


class ModelClientTests(unittest.TestCase):
    def test_sends_non_stream_responses_request_with_authorization(self) -> None:
        transport = RecordingTransport({"output": []})
        client = ModelClient(transport)
        config = ModelConfig(
            url="https://api.openai.com/v1/responses",
            model="gpt-5.5",
            api_key="secret-key",
        )

        client.create_response(config, "hello", [], "Follow the instructions.", 30)

        url, headers, payload, timeout_seconds = transport.calls[0]
        self.assertEqual(url, "https://api.openai.com/v1/responses")
        self.assertEqual(headers["Authorization"], "Bearer secret-key")
        self.assertEqual(payload["model"], "gpt-5.5")
        self.assertEqual(payload["input"], "hello")
        self.assertEqual(payload["instructions"], "Follow the instructions.")
        self.assertIs(payload["stream"], False)
        self.assertEqual(timeout_seconds, 30)

    def test_omits_authorization_header_without_api_key(self) -> None:
        transport = RecordingTransport({"output": []})
        client = ModelClient(transport)
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )

        client.create_response(config, "hello", [], "Follow the instructions.", 30)

        _, headers, _, _ = transport.calls[0]
        self.assertNotIn("Authorization", headers)


if __name__ == "__main__":
    unittest.main()
