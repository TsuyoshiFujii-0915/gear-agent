import json
import unittest
from typing import Any

from gear_agent.config import WebFetchConfig
from gear_agent.errors import GearError, gear_error
from gear_agent.tools.web_fetch import TavilyFetchTransport, WebFetchTool


class FakeTavilyFetchTransport(TavilyFetchTransport):
    def __init__(self, response: dict[str, Any] | GearError) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object], int]] = []

    def fetch(
        self,
        api_key: str,
        payload: dict[str, object],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        self.calls.append((api_key, payload, timeout_seconds))
        if isinstance(self.response, GearError):
            raise self.response
        return self.response


class WebFetchToolTests(unittest.TestCase):
    def test_schema_exposes_url_only(self) -> None:
        tool = WebFetchTool(_config(), FakeTavilyFetchTransport(_tavily_response()))

        schema = tool.schema()

        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["name"], "web_fetch")
        self.assertTrue(schema["strict"])
        parameters = schema["parameters"]
        self.assertIsInstance(parameters, dict)
        self.assertEqual(parameters["required"], ["url"])
        self.assertFalse(parameters["additionalProperties"])

    def test_runs_tavily_extract_and_returns_serializable_content(self) -> None:
        transport = FakeTavilyFetchTransport(_tavily_response())
        tool = WebFetchTool(_config(), transport)

        result = tool.run({"url": "https://example.com/docs"})

        self.assertEqual(
            transport.calls,
            [
                (
                    "tvly-secret",
                    {
                        "urls": ["https://example.com/docs"],
                        "extract_depth": "basic",
                        "format": "markdown",
                        "include_images": False,
                        "include_favicon": True,
                        "timeout": 20,
                        "include_usage": True,
                    },
                    20,
                )
            ],
        )
        self.assertEqual(result["url"], "https://example.com/docs")
        self.assertEqual(result["title"], "Example Docs")
        self.assertEqual(result["content"], "# Example Docs\nCurrent content.")
        self.assertEqual(result["images"], [])
        self.assertEqual(result["favicon"], "https://example.com/favicon.ico")
        self.assertEqual(result["response_time"], 1.23)
        self.assertEqual(result["credits"], 1)
        self.assertEqual(result["request_id"], "req-123")
        json.dumps(result)

    def test_rejects_missing_url_argument(self) -> None:
        tool = WebFetchTool(_config(), FakeTavilyFetchTransport(_tavily_response()))

        with self.assertRaises(GearError) as error:
            tool.run({})

        self.assertEqual(error.exception.origin, "web_fetch")
        self.assertIn("url", error.exception.message)

    def test_rejects_failed_tavily_extraction(self) -> None:
        tool = WebFetchTool(
            _config(),
            FakeTavilyFetchTransport(
                {
                    "results": [],
                    "failed_results": [
                        {
                            "url": "https://example.com/docs",
                            "error": "Could not extract content.",
                        }
                    ],
                }
            ),
        )

        with self.assertRaises(GearError) as error:
            tool.run({"url": "https://example.com/docs"})

        self.assertEqual(error.exception.origin, "web_fetch")
        self.assertEqual(error.exception.error_type, "tavily_extract_failed")
        self.assertIn("Could not extract content.", error.exception.message)

    def test_rejects_invalid_tavily_response_shape(self) -> None:
        tool = WebFetchTool(_config(), FakeTavilyFetchTransport({"results": "invalid"}))

        with self.assertRaises(GearError) as error:
            tool.run({"url": "https://example.com/docs"})

        self.assertEqual(error.exception.origin, "web_fetch")
        self.assertEqual(error.exception.error_type, "tavily_extract_response_invalid")

    def test_rejects_empty_content(self) -> None:
        response = _tavily_response()
        response["results"][0]["raw_content"] = ""
        tool = WebFetchTool(_config(), FakeTavilyFetchTransport(response))

        with self.assertRaises(GearError) as error:
            tool.run({"url": "https://example.com/docs"})

        self.assertEqual(error.exception.origin, "web_fetch")
        self.assertEqual(error.exception.error_type, "tavily_extract_content_empty")

    def test_rejects_content_over_configured_limit(self) -> None:
        response = _tavily_response()
        response["results"][0]["raw_content"] = "abcdef"
        tool = WebFetchTool(
            WebFetchConfig(
                api_key="tvly-secret",
                extract_depth="basic",
                content_format="markdown",
                timeout_seconds=20,
                include_images=False,
                include_favicon=True,
                max_content_chars=5,
            ),
            FakeTavilyFetchTransport(response),
        )

        with self.assertRaises(GearError) as error:
            tool.run({"url": "https://example.com/docs"})

        self.assertEqual(error.exception.origin, "web_fetch")
        self.assertEqual(error.exception.error_type, "tavily_extract_content_too_large")

    def test_propagates_tavily_transport_error(self) -> None:
        transport_error = gear_error(
            "tavily_extract_http_error",
            "Tavily extract request failed.",
            "web_fetch",
            True,
            {"status": 401},
        )
        tool = WebFetchTool(_config(), FakeTavilyFetchTransport(transport_error))

        with self.assertRaises(GearError) as error:
            tool.run({"url": "https://example.com/docs"})

        self.assertEqual(error.exception, transport_error)


def _config() -> WebFetchConfig:
    return WebFetchConfig(
        api_key="tvly-secret",
        extract_depth="basic",
        content_format="markdown",
        timeout_seconds=20,
        include_images=False,
        include_favicon=True,
        max_content_chars=20_000,
    )


def _tavily_response() -> dict[str, Any]:
    return {
        "results": [
            {
                "url": "https://example.com/docs",
                "title": "Example Docs",
                "raw_content": "# Example Docs\nCurrent content.",
                "images": [],
                "favicon": "https://example.com/favicon.ico",
            }
        ],
        "failed_results": [],
        "response_time": 1.23,
        "usage": {"credits": 1},
        "request_id": "req-123",
    }


if __name__ == "__main__":
    unittest.main()
