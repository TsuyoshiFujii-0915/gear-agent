import json
import unittest
from typing import Any

from gear_agent.config import WebSearchConfig
from gear_agent.errors import GearError, gear_error
from gear_agent.tools.web_search import TavilySearchTransport, WebSearchTool


class FakeTavilySearchTransport(TavilySearchTransport):
    def __init__(self, response: dict[str, Any] | GearError) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object], int]] = []

    def search(
        self,
        api_key: str,
        payload: dict[str, object],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        self.calls.append((api_key, payload, timeout_seconds))
        if isinstance(self.response, GearError):
            raise self.response
        return self.response


class WebSearchToolTests(unittest.TestCase):
    def test_schema_exposes_query_only(self) -> None:
        tool = WebSearchTool(_config(), FakeTavilySearchTransport(_tavily_response()))

        schema = tool.schema()

        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["name"], "web_search")
        self.assertTrue(schema["strict"])
        parameters = schema["parameters"]
        self.assertIsInstance(parameters, dict)
        self.assertEqual(parameters["required"], ["query"])
        self.assertFalse(parameters["additionalProperties"])

    def test_runs_tavily_search_and_returns_serializable_results(self) -> None:
        transport = FakeTavilySearchTransport(_tavily_response())
        tool = WebSearchTool(_config(), transport)

        result = tool.run({"query": "OpenAI Responses API latest docs"})

        self.assertEqual(
            transport.calls,
            [
                (
                    "tvly-secret",
                    {
                        "query": "OpenAI Responses API latest docs",
                        "search_depth": "basic",
                        "max_results": 5,
                        "include_answer": True,
                        "include_raw_content": False,
                        "include_usage": True,
                    },
                    20,
                )
            ],
        )
        self.assertEqual(result["query"], "OpenAI Responses API latest docs")
        self.assertEqual(result["answer"], "Use the Responses API.")
        self.assertEqual(
            result["results"],
            [
                {
                    "title": "Responses API",
                    "url": "https://developers.openai.com/api/docs",
                    "content": "Current Responses API docs.",
                    "score": 0.91,
                    "raw_content": None,
                }
            ],
        )
        self.assertEqual(result["credits"], 1)
        self.assertEqual(result["request_id"], "req-123")
        json.dumps(result)

    def test_rejects_missing_query_argument(self) -> None:
        tool = WebSearchTool(_config(), FakeTavilySearchTransport(_tavily_response()))

        with self.assertRaises(GearError) as error:
            tool.run({})

        self.assertEqual(error.exception.origin, "web_search")
        self.assertIn("query", error.exception.message)

    def test_rejects_invalid_tavily_response_shape(self) -> None:
        tool = WebSearchTool(_config(), FakeTavilySearchTransport({"results": "invalid"}))

        with self.assertRaises(GearError) as error:
            tool.run({"query": "OpenAI"})

        self.assertEqual(error.exception.origin, "web_search")
        self.assertEqual(error.exception.error_type, "tavily_response_invalid")

    def test_propagates_tavily_transport_error(self) -> None:
        transport_error = gear_error(
            "tavily_http_error",
            "Tavily search request failed.",
            "web_search",
            True,
            {"status": 401},
        )
        tool = WebSearchTool(_config(), FakeTavilySearchTransport(transport_error))

        with self.assertRaises(GearError) as error:
            tool.run({"query": "OpenAI"})

        self.assertEqual(error.exception, transport_error)


def _config() -> WebSearchConfig:
    return WebSearchConfig(
        api_key="tvly-secret",
        search_depth="basic",
        max_results=5,
        timeout_seconds=20,
        include_answer=True,
        include_raw_content=False,
    )


def _tavily_response() -> dict[str, Any]:
    return {
        "query": "OpenAI Responses API latest docs",
        "answer": "Use the Responses API.",
        "results": [
            {
                "title": "Responses API",
                "url": "https://developers.openai.com/api/docs",
                "content": "Current Responses API docs.",
                "score": 0.91,
                "raw_content": None,
            }
        ],
        "response_time": "1.23",
        "usage": {"credits": 1},
        "request_id": "req-123",
    }


if __name__ == "__main__":
    unittest.main()
