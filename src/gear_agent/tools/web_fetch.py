from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import json

from gear_agent.config import WebFetchConfig
from gear_agent.errors import GearError, gear_error
from gear_agent.tools.base import Tool


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


class UrllibTavilyFetchTransport(TavilyFetchTransport):
    """Standard-library Tavily Extract API transport."""

    def fetch(
        self,
        api_key: str,
        payload: dict[str, object],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        request_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            TAVILY_EXTRACT_URL,
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
                "tavily_extract_http_status_error",
                f"Tavily extract returned HTTP {exc.code}.",
                "web_fetch",
                True,
                {"status": exc.code, "body": response_body},
            ) from exc
        except URLError as exc:
            raise gear_error(
                "tavily_extract_http_request_failed",
                "Failed to reach Tavily extract.",
                "web_fetch",
                True,
                {"reason": str(exc.reason)},
            ) from exc
        except TimeoutError as exc:
            raise gear_error(
                "tavily_extract_http_timeout",
                "Tavily extract request timed out.",
                "web_fetch",
                True,
                {"timeout_seconds": timeout_seconds},
            ) from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise gear_error(
                "tavily_extract_json_parse_failed",
                "Tavily extract returned invalid JSON.",
                "web_fetch",
                True,
                {"body": response_body},
            ) from exc
        if not isinstance(parsed, dict):
            raise gear_error(
                "tavily_extract_response_invalid",
                "Tavily extract returned JSON that is not an object.",
                "web_fetch",
                True,
                {},
            )
        return parsed


class WebFetchTool(Tool):
    """Model-callable web fetch tool backed by Tavily Extract."""

    def __init__(
        self,
        config: WebFetchConfig,
        transport: TavilyFetchTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    @property
    def name(self) -> str:
        return "web_fetch"

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": self.name,
            "description": "Fetch readable web page content from a URL using Tavily Extract.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "HTTP or HTTPS URL to fetch with Tavily Extract.",
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        url = _required_argument_url(arguments, "url", "web_fetch arguments")
        payload: dict[str, object] = {
            "urls": [url],
            "extract_depth": self._config.extract_depth,
            "format": self._config.content_format,
            "include_images": self._config.include_images,
            "include_favicon": self._config.include_favicon,
            "timeout": self._config.timeout_seconds,
            "include_usage": True,
        }
        response = self._transport.fetch(
            self._config.api_key,
            payload,
            self._config.timeout_seconds,
        )
        return _normalize_tavily_response(response, url, self._config.max_content_chars)


def _normalize_tavily_response(
    response: dict[str, Any],
    requested_url: str,
    max_content_chars: int,
) -> dict[str, object]:
    _raise_failed_result(response, requested_url)
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


def _raise_failed_result(response: dict[str, Any], requested_url: str) -> None:
    raw_failed_results = response.get("failed_results")
    if raw_failed_results is None:
        raw_failed_results = []
    if not isinstance(raw_failed_results, list):
        raise _invalid_response(
            "Tavily extract response.failed_results must be a list when present."
        )
    for index, item in enumerate(raw_failed_results):
        if not isinstance(item, dict):
            raise _invalid_response(
                f"Tavily extract response.failed_results[{index}] must be an object."
            )
        origin = f"Tavily extract response.failed_results[{index}]"
        failed_url = _required_response_string(item, "url", origin)
        message = _required_response_string(item, "error", origin)
        if failed_url == requested_url:
            raise gear_error(
                "tavily_extract_failed",
                f"Tavily extract failed for URL: {message}",
                "web_fetch",
                True,
                {"url": failed_url, "error": message},
            )


def _normalize_tavily_result(
    item: dict[object, object],
    max_content_chars: int,
    response: dict[str, Any],
) -> dict[str, object]:
    origin = "Tavily extract response.results[0]"
    url = _required_response_string(item, "url", origin)
    content = _required_response_string(item, "raw_content", origin)
    if content == "":
        raise gear_error(
            "tavily_extract_content_empty",
            "Tavily extract returned empty content.",
            "web_fetch",
            True,
            {"url": url},
        )
    if len(content) > max_content_chars:
        raise gear_error(
            "tavily_extract_content_too_large",
            "Tavily extract content exceeds configured maximum.",
            "web_fetch",
            True,
            {
                "url": url,
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


def _optional_credits(response: dict[str, Any]) -> object:
    usage = response.get("usage")
    if usage is None:
        return None
    if not isinstance(usage, dict):
        raise _invalid_response("Tavily extract response.usage must be an object when present.")
    credits = usage.get("credits")
    if credits is None:
        return None
    if isinstance(credits, bool) or not isinstance(credits, (int, float)):
        raise _invalid_response("Tavily extract response.usage.credits must be a number.")
    return credits


def _required_argument_url(payload: dict[object, object], key: str, origin: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise gear_error(
            "tool_argument_invalid",
            f"Missing or invalid string value: {origin}.{key}",
            "web_fetch",
            True,
            {"origin": origin, "key": key},
        )
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.netloc == "":
        raise gear_error(
            "tool_argument_invalid",
            f"Missing or invalid HTTP URL value: {origin}.{key}",
            "web_fetch",
            True,
            {"origin": origin, "key": key, "value": value},
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


def _optional_number(payload: dict[object, object], key: str, origin: str) -> object:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
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
