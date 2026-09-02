from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json

from gear_agent.config import WebSearchConfig
from gear_agent.errors import GearError, gear_error
from gear_agent.tools.base import Tool


TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class TavilySearchTransport(ABC):
    """Transport for Tavily Search API requests."""

    @abstractmethod
    def search(
        self,
        api_key: str,
        payload: dict[str, object],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Runs a Tavily Search API request.

        Args:
            api_key: Tavily API key.
            payload: JSON request body.
            timeout_seconds: HTTP request timeout in seconds.

        Returns:
            Parsed Tavily JSON response.
        """


class UrllibTavilySearchTransport(TavilySearchTransport):
    """Standard-library Tavily Search API transport."""

    def search(
        self,
        api_key: str,
        payload: dict[str, object],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        request_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            TAVILY_SEARCH_URL,
            data=request_bytes,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise gear_error(
                "tavily_http_status_error",
                f"Tavily search returned HTTP {exc.code}.",
                "web_search",
                True,
                {"status": exc.code, "body": response_body},
            ) from exc
        except URLError as exc:
            raise gear_error(
                "tavily_http_request_failed",
                "Failed to reach Tavily search.",
                "web_search",
                True,
                {"reason": str(exc.reason)},
            ) from exc
        except TimeoutError as exc:
            raise gear_error(
                "tavily_http_timeout",
                "Tavily search request timed out.",
                "web_search",
                True,
                {"timeout_seconds": timeout_seconds},
            ) from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise gear_error(
                "tavily_json_parse_failed",
                "Tavily search returned invalid JSON.",
                "web_search",
                True,
                {"body": response_body},
            ) from exc
        if not isinstance(parsed, dict):
            raise gear_error(
                "tavily_response_invalid",
                "Tavily search returned JSON that is not an object.",
                "web_search",
                True,
                {},
            )
        return parsed


class WebSearchTool(Tool):
    """Model-callable web search tool backed by Tavily Search."""

    def __init__(
        self,
        config: WebSearchConfig,
        transport: TavilySearchTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    @property
    def name(self) -> str:
        return "web_search"

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": self.name,
            "description": "Search the web for current information using Tavily Search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to send to Tavily.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        query = _required_argument_string(arguments, "query", "web_search arguments")
        payload: dict[str, object] = {
            "query": query,
            "search_depth": self._config.search_depth,
            "max_results": self._config.max_results,
            "include_answer": self._config.include_answer,
            "include_raw_content": self._config.include_raw_content,
            "include_usage": True,
        }
        response = self._transport.search(
            self._config.api_key,
            payload,
            self._config.timeout_seconds,
        )
        return _normalize_tavily_response(response)


def _normalize_tavily_response(response: dict[str, Any]) -> dict[str, object]:
    query = _required_response_string(response, "query", "Tavily response")
    raw_results = response.get("results")
    if not isinstance(raw_results, list):
        raise _invalid_response("Tavily response.results must be a list.")
    results = [_normalize_tavily_result(item, index) for index, item in enumerate(raw_results)]
    return {
        "query": query,
        "answer": _optional_string(response, "answer", "Tavily response"),
        "results": results,
        "response_time": _optional_string_or_number(
            response,
            "response_time",
            "Tavily response",
        ),
        "credits": _optional_credits(response),
        "request_id": _optional_string(response, "request_id", "Tavily response"),
    }


def _normalize_tavily_result(item: object, index: int) -> dict[str, object]:
    if not isinstance(item, dict):
        raise _invalid_response(f"Tavily response.results[{index}] must be an object.")
    origin = f"Tavily response.results[{index}]"
    return {
        "title": _required_response_string(item, "title", origin),
        "url": _required_response_string(item, "url", origin),
        "content": _required_response_string(item, "content", origin),
        "score": _optional_number(item, "score", origin),
        "raw_content": _optional_string(item, "raw_content", origin),
    }


def _optional_credits(response: dict[str, Any]) -> object:
    usage = response.get("usage")
    if usage is None:
        return None
    if not isinstance(usage, dict):
        raise _invalid_response("Tavily response.usage must be an object when present.")
    credits = usage.get("credits")
    if credits is None:
        return None
    if not isinstance(credits, int):
        raise _invalid_response("Tavily response.usage.credits must be an integer.")
    return credits


def _required_argument_string(payload: dict[object, object], key: str, origin: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise gear_error(
            "tool_argument_invalid",
            f"Missing or invalid string value: {origin}.{key}",
            "web_search",
            True,
            {"origin": origin, "key": key},
        )
    return value


def _required_response_string(payload: dict[object, object], key: str, origin: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise _invalid_response(f"{origin}.{key} must be a string.")
    return value


def _optional_string(payload: dict[object, object], key: str, origin: str) -> object:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid_response(f"{origin}.{key} must be a string when present.")
    return value


def _optional_number(payload: dict[object, object], key: str, origin: str) -> object:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid_response(f"{origin}.{key} must be a number when present.")
    return value


def _optional_string_or_number(
    payload: dict[object, object],
    key: str,
    origin: str,
) -> object:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise _invalid_response(f"{origin}.{key} must be a string or number when present.")
    return value


def _invalid_response(message: str) -> GearError:
    return gear_error(
        "tavily_response_invalid",
        message,
        "web_search",
        True,
        {},
    )
