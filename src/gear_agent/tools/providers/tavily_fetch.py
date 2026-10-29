from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any

from gear_agent.config import WebFetchConfig, TavilyFetchOptions
from gear_agent.errors import GearError, gear_error
from gear_agent.tools.providers.base import ProviderResult, WebUsage
from gear_agent.tools.providers.http import WebHttpClient, post_web_json
from gear_agent.tools.providers.tavily_common import reject_tavily_errors


TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


class TavilyFetchTransport(ABC):
    """Transport for Tavily Extract API requests."""

    @abstractmethod
    def fetch(
        self,
        api_key: str,
        payload: dict[str, object],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Runs a Tavily Extract API request.

        Args:
            api_key: Tavily API key.
            payload: JSON request body.
            timeout_seconds: HTTP request timeout in seconds.

        Returns:
            Parsed Tavily JSON response.
        """


class HttpxTavilyFetchTransport(TavilyFetchTransport):
    """Tavily transport using bounded HTTPX JSON requests.

    Args:
        http: Injected HTTP client for external API tests; None creates a scoped
            production client for each request.
    """

    def __init__(self, http: WebHttpClient | None) -> None:
        self._http = http

    def fetch(
        self, api_key: str, payload: dict[str, object], timeout_seconds: int,
    ) -> dict[str, Any]:
        return post_web_json(self._http, TAVILY_EXTRACT_URL, api_key, payload,
                             timeout_seconds, "web_fetch", "tavily_extract")


class TavilyFetchProvider:
    """Normalizes Tavily Extract behind the independent fetch boundary."""

    name = "tavily"
    display_name = "Tavily Extract"

    def __init__(self, config: WebFetchConfig, transport: TavilyFetchTransport) -> None:
        if not isinstance(config.options, TavilyFetchOptions):
            raise gear_error("tool_config_invalid", "Tavily fetch requires Tavily fetch options.",
                             "tool_config", True, {})
        self._config = config
        self._options = config.options
        self._transport = transport

    @property
    def max_content_chars(self) -> int:
        return self._config.max_content_chars

    def fetch(self, url: str) -> ProviderResult:
        payload: dict[str, object] = {
            "urls": [url],
            "extract_depth": self._options.extract_depth,
            "format": self._options.content_format,
            "include_images": self._options.include_images,
            "include_favicon": self._options.include_favicon,
            "timeout": self._config.timeout_seconds,
            "include_usage": True,
        }
        response = self._transport.fetch(
            self._config.api_key,
            payload,
            self._config.timeout_seconds,
        )
        data = _normalize_tavily_response(response, self._config.max_content_chars)
        data.update(requested_url=url, final_url=None)
        credits = data["credits"]
        return ProviderResult(data, () if credits is None else (WebUsage("credits", credits),))


def _normalize_tavily_response(
    response: dict[str, Any],
    max_content_chars: int,
) -> dict[str, object]:
    reject_tavily_errors(response, "web_fetch", "tavily_extract")
    raw_results = response.get("results")
    if not isinstance(raw_results, list):
        raise _invalid_response("Tavily extract response.results must be a list.")
    if len(raw_results) != 1:
        raise _invalid_response(
            "Tavily extract response must contain exactly one result."
        )
    item = raw_results[0]
    if not isinstance(item, dict):
        raise _invalid_response("Tavily extract response.results[0] must be an object.")
    return _normalize_tavily_result(item, max_content_chars, response)


def _normalize_tavily_result(
    item: dict[object, object],
    max_content_chars: int,
    response: dict[str, Any],
) -> dict[str, object]:
    origin = "Tavily extract response.results[0]"
    url = _required_response_string(item, "url", origin)
    content = _required_response_string(item, "raw_content", origin)
    if not content.strip():
        raise gear_error(
            "tavily_extract_content_empty",
            "Tavily extract returned empty content.",
            "web_fetch",
            True,
            {},
        )
    if len(content) > max_content_chars:
        raise gear_error(
            "tavily_extract_content_too_large",
            "Tavily extract content exceeds configured maximum.",
            "web_fetch",
            True,
            {
                "content_chars": len(content),
                "max_content_chars": max_content_chars,
            },
        )
    return {
        "url": url,
        "title": _optional_string(item, "title", origin),
        "content": content,
        "images": _optional_string_list(item, "images", origin),
        "favicon": _optional_string(item, "favicon", origin),
        "response_time": _optional_number(response, "response_time", "Tavily extract response"),
        "credits": _optional_credits(response),
        "request_id": _optional_string(response, "request_id", "Tavily extract response"),
    }


def _optional_credits(response: dict[str, Any]) -> int | float | None:
    usage = response.get("usage")
    if usage is None:
        return None
    if not isinstance(usage, dict):
        raise _invalid_response("Tavily extract response.usage must be an object when present.")
    credits = usage.get("credits")
    if credits is None:
        return None
    if isinstance(credits, bool) or not isinstance(credits, (int, float)) or not math.isfinite(credits) or credits < 0:
        raise _invalid_response("Tavily extract response.usage.credits must be a number.")
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


def _optional_string_list(payload: dict[object, object], key: str, origin: str) -> list[str]:
    value = payload.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise _invalid_response(f"{origin}.{key} must be a list when present.")
    strings: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise _invalid_response(f"{origin}.{key}[{index}] must be a string.")
        strings.append(item)
    return strings


def _optional_number(payload: dict[object, object], key: str, origin: str) -> int | float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _invalid_response(f"{origin}.{key} must be a number when present.")
    return value


def _invalid_response(message: str) -> GearError:
    return gear_error(
        "tavily_extract_response_invalid",
        message,
        "web_fetch",
        True,
        {},
    )
