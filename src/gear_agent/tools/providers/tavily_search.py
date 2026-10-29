from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any

from gear_agent.config import WebSearchConfig
from gear_agent.errors import GearError, gear_error
from gear_agent.tools.providers.base import ProviderResult, WebUsage
from gear_agent.tools.providers.http import WebHttpClient, post_web_json
from gear_agent.tools.providers.tavily_common import reject_tavily_errors


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


class HttpxTavilySearchTransport(TavilySearchTransport):
    """Tavily transport using bounded HTTPX JSON requests.

    Args:
        http: Injected HTTP client for external API tests; None creates a scoped
            production client for each request.
    """

    def __init__(self, http: WebHttpClient | None) -> None:
        self._http = http

    def search(
        self, api_key: str, payload: dict[str, object], timeout_seconds: int,
    ) -> dict[str, Any]:
        return post_web_json(self._http, TAVILY_SEARCH_URL, api_key, payload,
                             timeout_seconds, "web_search", "tavily")


class TavilySearchProvider:
    """Normalizes Tavily Search behind the independent search boundary."""

    name = "tavily"
    display_name = "Tavily Search"

    def __init__(self, config: WebSearchConfig, transport: TavilySearchTransport) -> None:
        self._config = config
        self._transport = transport

    @property
    def max_results(self) -> int:
        return self._config.max_results

    def search(self, query: str) -> ProviderResult:
        payload: dict[str, object] = {
            "query": query,
            "search_depth": self._config.options.search_depth,
            "max_results": self._config.max_results,
            "include_answer": self._config.options.include_answer,
            "include_raw_content": self._config.options.include_raw_content,
            "include_usage": True,
        }
        response = self._transport.search(
            self._config.api_key,
            payload,
            self._config.timeout_seconds,
        )
        data = _normalize_tavily_response(response, query)
        credits = data["credits"]
        return ProviderResult(data, () if credits is None else (WebUsage("credits", credits),))


def _normalize_tavily_response(response: dict[str, Any], requested_query: str) -> dict[str, object]:
    reject_tavily_errors(response, "web_search", "tavily")
    query = (_required_response_string(response, "query", "Tavily response")
             if "query" in response else requested_query)
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
        "title": _optional_string(item, "title", origin),
        "url": _required_response_string(item, "url", origin),
        "content": _required_response_string(item, "content", origin),
        "score": _optional_number(item, "score", origin),
        "raw_content": _optional_string(item, "raw_content", origin),
    }


def _optional_credits(response: dict[str, Any]) -> int | None:
    usage = response.get("usage")
    if usage is None:
        return None
    if not isinstance(usage, dict):
        raise _invalid_response("Tavily response.usage must be an object when present.")
    credits = usage.get("credits")
    if credits is None:
        return None
    if isinstance(credits, bool) or not isinstance(credits, int) or credits < 0:
        raise _invalid_response("Tavily response.usage.credits must be an integer.")
    return credits


def _required_response_string(payload: dict[object, object], key: str, origin: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise _invalid_response(f"{origin}.{key} must be a string.")
    return value


def _optional_string(payload: dict[object, object], key: str, origin: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid_response(f"{origin}.{key} must be a string when present.")
    return value


def _optional_number(payload: dict[object, object], key: str, origin: str) -> int | float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
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
