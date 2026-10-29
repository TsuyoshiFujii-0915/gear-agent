from __future__ import annotations

from typing import Any
from urllib.parse import quote, quote_plus
import math
import re

import httpx

from gear_agent.errors import GearError, gear_error
from gear_agent.config import ExaFetchConfig, ExaSearchConfig
from gear_agent.tools.providers.base import ProviderResult, WebCostEstimate, WebUsage
from gear_agent.tools.providers.http import WebHttpClient
from gear_agent.tools.providers.urls import validate_reported_url


class ExaTransport:
    """Uses the shared bounded HTTP policy with Exa authentication."""

    def post(
        self, url: str, headers: dict[str, str], payload: dict[str, object],
        timeout_seconds: int, max_response_bytes: int, origin: str,
    ) -> dict[str, Any]:
        """Posts to an adapter-owned endpoint through WebHttpClient.

        Args:
            url: Fixed Exa API endpoint.
            headers: Exa authentication headers.
            payload: Provider JSON request.
            timeout_seconds: HTTP waiting budget in seconds.
            max_response_bytes: Configured response byte limit.
            origin: Model-callable tool identity.

        Returns:
            Parsed provider response.
        """
        with httpx.Client(follow_redirects=False) as client:
            return WebHttpClient(client, max_response_bytes).post_json_with_headers(
                url, headers, payload, timeout_seconds, origin, "exa",
            )


def _estimated_usage(response: dict[str, Any]) -> tuple[WebUsage, ...]:
    cost = response.get("costDollars")
    if cost is None or "total" not in cost:
        return ()
    return (WebCostEstimate("USD", cost["total"]),)


class ExaSearchProvider:
    """Exa Search behind the existing query-only tool interface."""

    def __init__(self, config: ExaSearchConfig, transport: ExaTransport) -> None:
        self._config = config
        self._transport = transport

    name = "exa"
    display_name = "Exa Search"

    @property
    def max_results(self) -> int:
        return self._config.max_results

    def search(self, query: str) -> ProviderResult:
        config = self._config
        contents: dict[str, object] = {"highlights": True}
        if config.options.content == "text":
            contents = {"text": {"maxCharacters": config.max_content_chars}}
        response = self._transport.post(
            "https://api.exa.ai/search", {"x-api-key": config.api_key},
            {"query": query, "numResults": config.max_results, "type": config.options.search_type,
             "contents": contents, "includeDomains": list(config.options.include_domains),
             "excludeDomains": list(config.options.exclude_domains)},
            config.timeout_seconds, config.max_response_bytes, "web_search",
        )
        _reject_error(response, "web_search")
        rows = response.get("results")
        if not isinstance(rows, list) or len(rows) > config.max_results:
            raise _invalid("web_search", "results must be a list within max_results.")
        results = [_search_result(row, index, config.max_content_chars) for index, row in enumerate(rows)]
        result = {"provider": "exa", "query": query, "answer": None, "results": results,
                  "credits": None, "request_id": _string(response, "requestId", "web_search"),
                  "provider_metadata": {"exa": _metadata(response, "web_search")}}
        return ProviderResult(_redact_result(result, config.api_key), _estimated_usage(response))


class ExaFetchProvider:
    """Exa Contents behind the existing single-URL tool interface."""

    def __init__(self, config: ExaFetchConfig, transport: ExaTransport) -> None:
        self._config = config
        self._transport = transport

    name = "exa"
    display_name = "Exa Contents (extracted text; may be cached)"

    @property
    def max_content_chars(self) -> int:
        return self._config.max_content_chars

    def fetch(self, url: str) -> ProviderResult:
        config = self._config
        payload: dict[str, object] = {"urls": [url], "text": True,
                                      "livecrawlTimeout": config.options.livecrawl_timeout_ms}
        if config.options.max_age_hours is not None:
            payload["maxAgeHours"] = config.options.max_age_hours
        response = self._transport.post(
            "https://api.exa.ai/contents", {"x-api-key": config.api_key}, payload,
            config.timeout_seconds, config.max_response_bytes, "web_fetch",
        )
        _reject_error(response, "web_fetch")
        statuses = response.get("statuses")
        if not isinstance(statuses, list) or len(statuses) != 1 or not isinstance(statuses[0], dict):
            raise _invalid("web_fetch", "statuses must contain exactly one URL status.")
        status = statuses[0]
        if status.get("id") != url:
            raise _invalid("web_fetch", "statuses[0].id does not match the requested URL.")
        if status.get("status") != "success" or status.get("error") is not None:
            raise gear_error("exa_fetch_failed", "Exa Contents failed for the requested URL.", "web_fetch", True, {})
        rows = response.get("results")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise _invalid("web_fetch", "results must contain exactly one page.")
        row = rows[0]
        returned_url = validate_reported_url(row.get("url"), "web_fetch")
        identifier = _string(row, "id", "web_fetch")
        if identifier != url and returned_url != url:
            raise _invalid("web_fetch", "results[0] does not match the requested URL by id or url.")
        text = _string(row, "text", "web_fetch")
        if text is None or not text.strip():
            raise gear_error("exa_content_empty", "Exa Contents returned no page text.", "web_fetch", True, {})
        _content_limit(text, config.max_content_chars, "web_fetch")
        metadata = _metadata(response, "web_fetch")
        metadata.update({"id": identifier, "source": _string(status, "source", "web_fetch"),
                         "publishedDate": _string(row, "publishedDate", "web_fetch"),
                         "author": _string(row, "author", "web_fetch")})
        result = {"provider": "exa", "requested_url": url, "url": returned_url, "final_url": None,
                  "title": _string(row, "title", "web_fetch"), "content": text, "content_format": "text",
                  "credits": None, "request_id": _string(response, "requestId", "web_fetch"),
                  "provider_metadata": {"exa": metadata}}
        return ProviderResult(_redact_result(result, config.api_key), _estimated_usage(response))


def _search_result(value: object, index: int, max_chars: int) -> dict[str, object]:
    origin = "web_search"
    if not isinstance(value, dict):
        raise _invalid(origin, f"results[{index}] must be an object.")
    highlights = value.get("highlights")
    if highlights is not None and (not isinstance(highlights, list)
                                   or any(not isinstance(item, str) for item in highlights)):
        raise _invalid(origin, f"results[{index}].highlights must be a list of strings.")
    content = None if not highlights else "\n".join(highlights)
    if content is not None and not content.strip():
        content = None
    text = _string(value, "text", origin)
    summary = _string(value, "summary", origin)
    for part in (content, text, summary):
        if part is not None:
            _content_limit(part, max_chars, origin)
    metadata = {"id": _string(value, "id", origin), "highlights": highlights, "summary": summary,
                "publishedDate": _string(value, "publishedDate", origin), "author": _string(value, "author", origin)}
    return {"url": validate_reported_url(value.get("url"), origin), "title": _string(value, "title", origin),
            "content": content, "raw_content": text, "score": _number(value, "score", origin),
            "provider_metadata": {"exa": metadata}}


def _metadata(response: dict[str, Any], origin: str) -> dict[str, object]:
    cost = response.get("costDollars")
    if cost is not None:
        _cost_object(cost, origin, 0)
    metadata: dict[str, object] = {
        "estimated_cost_dollars": cost, "search_time_ms": _number(response, "searchTime", origin),
    }
    # The empty legacy resolvedSearchType value is retained but never drives behavior.
    if "resolvedSearchType" in response:
        metadata["resolvedSearchType"] = _string(response, "resolvedSearchType", origin)
    if "notices" in response:
        notices = response["notices"]
        if not isinstance(notices, list) or any(not isinstance(item, dict) for item in notices):
            raise _invalid(origin, "notices must be a list of objects.")
        metadata["notices"] = [
            {key: _string(notice, key, origin) for key in ("code", "path", "message")}
            for notice in notices
        ]
    return metadata


def _cost_object(value: object, origin: str, depth: int) -> None:
    if not isinstance(value, dict) or depth > 3:
        raise _invalid(origin, "costDollars must contain bounded numeric estimates.")
    for item in value.values():
        if isinstance(item, dict):
            _cost_object(item, origin, depth + 1)
        elif not _finite_number(item) or item < 0:
            raise _invalid(origin, "costDollars must contain nonnegative finite estimates.")


def _string(value: dict[str, Any], key: str, origin: str) -> str | None:
    item = value.get(key)
    if item is not None and not isinstance(item, str):
        raise _invalid(origin, f"{key} must be a string when reported.")
    return item


def _number(value: dict[str, Any], key: str, origin: str) -> int | float | None:
    item = value.get(key)
    if item is not None and not _finite_number(item):
        raise _invalid(origin, f"{key} must be a finite number when reported.")
    return item


def _finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _content_limit(text: str, maximum: int, origin: str) -> None:
    if len(text) > maximum:
        raise gear_error("exa_content_too_large", "Exa content exceeds max_content_chars.",
                         origin, True, {"content_chars": len(text), "max_content_chars": maximum})


def _reject_error(response: dict[str, Any], origin: str) -> None:
    if response.get("error") is not None:
        raise gear_error("exa_provider_error", "Exa reported an unsuccessful request.", origin, True, {})


def _invalid(origin: str, message: str) -> GearError:
    return gear_error("exa_response_invalid", f"Exa response: {message}", origin, True, {})


def _redact_result(value: Any, api_key: str) -> Any:
    """Removes provider credentials before results reach any downstream consumer.

    Args:
        value: Normalized JSON-compatible result.
        api_key: Resolved credential used for this provider request.

    Returns:
        A copy without the known credential or credential-named metadata values.
    """
    if isinstance(value, str):
        for secret in sorted({api_key, quote(api_key, safe=""), quote_plus(api_key)}, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_redact_result(item, api_key) for item in value]
    if isinstance(value, dict):
        return {_redact_result(key, api_key): (
            "[REDACTED]" if re.search(r"authorization|api.?key|secret|password|token|cookie", key, re.I)
            else _redact_result(item, api_key)) for key, item in value.items()}
    return value
