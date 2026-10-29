from __future__ import annotations

from typing import Any
import math

from gear_agent.errors import GearError, gear_error
from gear_agent.exa_config import ExaFetchConfig, ExaSearchConfig
from gear_agent.tools.base import Tool
from gear_agent.tools.web_http import WebJsonTransport
from gear_agent.tools.web_safety import public_web_url, redact_web_result


class ExaSearchTool(Tool):
    """Exa Search behind the existing query-only tool interface."""

    def __init__(self, config: ExaSearchConfig, transport: WebJsonTransport) -> None:
        self._config = config
        self._transport = transport

    @property
    def name(self) -> str:
        return "web_search"

    def schema(self) -> dict[str, object]:
        return _schema(self.name, "query", "Search the web using Exa Search.",
                       "Search query; domain restrictions use configured filters, not guaranteed site: syntax.")

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip() or set(arguments) != {"query"}:
            raise gear_error("tool_argument_invalid", "web_search requires only a nonempty query string.",
                             self.name, True, {})
        config = self._config
        contents: dict[str, object] = {"highlights": True}
        if config.exa.content == "text":
            contents = {"text": {"maxCharacters": config.max_content_chars}}
        response = self._transport.post(
            "https://api.exa.ai/search", {"x-api-key": config.api_key},
            {"query": query, "numResults": config.max_results, "type": config.exa.search_type,
             "contents": contents, "includeDomains": list(config.exa.include_domains),
             "excludeDomains": list(config.exa.exclude_domains)},
            config.timeout_seconds, config.max_response_bytes, self.name,
        )
        _reject_error(response, self.name)
        rows = response.get("results")
        if not isinstance(rows, list) or len(rows) > config.max_results:
            raise _invalid(self.name, "results must be a list within max_results.")
        results = [_search_result(row, index, config.max_content_chars) for index, row in enumerate(rows)]
        result = {"provider": "exa", "query": query, "answer": None, "results": results,
                  "credits": None, "request_id": _string(response, "requestId", self.name),
                  "provider_metadata": {"exa": _metadata(response, self.name)}}
        return redact_web_result(result, config.api_key)


class ExaFetchTool(Tool):
    """Exa Contents behind the existing single-URL tool interface."""

    def __init__(self, config: ExaFetchConfig, transport: WebJsonTransport) -> None:
        self._config = config
        self._transport = transport

    @property
    def name(self) -> str:
        return "web_fetch"

    def schema(self) -> dict[str, object]:
        return _schema(self.name, "url", "Fetch extracted page text using Exa Contents; content may be cached.",
                       "Public HTTP or HTTPS URL without credentials.")

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        if set(arguments) != {"url"}:
            raise gear_error("tool_argument_invalid", "web_fetch requires only a URL.", self.name, True, {})
        url = public_web_url(arguments.get("url"), self.name)
        config = self._config
        payload: dict[str, object] = {"urls": [url], "text": True,
                                      "livecrawlTimeout": config.exa.livecrawl_timeout_ms}
        if config.exa.max_age_hours is not None:
            payload["maxAgeHours"] = config.exa.max_age_hours
        response = self._transport.post(
            "https://api.exa.ai/contents", {"x-api-key": config.api_key}, payload,
            config.timeout_seconds, config.max_response_bytes, self.name,
        )
        _reject_error(response, self.name)
        statuses = response.get("statuses")
        if not isinstance(statuses, list) or len(statuses) != 1 or not isinstance(statuses[0], dict):
            raise _invalid(self.name, "statuses must contain exactly one URL status.")
        status = statuses[0]
        if status.get("id") != url:
            raise _invalid(self.name, "statuses[0].id does not match the requested URL.")
        if status.get("status") != "success" or status.get("error") is not None:
            raise gear_error("exa_fetch_failed", "Exa Contents failed for the requested URL.", self.name, True, {})
        rows = response.get("results")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise _invalid(self.name, "results must contain exactly one page.")
        row = rows[0]
        returned_url = public_web_url(row.get("url"), self.name)
        identifier = _string(row, "id", self.name)
        if identifier != url and returned_url != url:
            raise _invalid(self.name, "results[0] does not match the requested URL by id or url.")
        text = _string(row, "text", self.name)
        if text is None or not text.strip():
            raise gear_error("exa_content_empty", "Exa Contents returned no page text.", self.name, True, {})
        _content_limit(text, config.max_content_chars, self.name)
        metadata = _metadata(response, self.name)
        metadata.update({"id": identifier, "source": _string(status, "source", self.name),
                         "publishedDate": _string(row, "publishedDate", self.name),
                         "author": _string(row, "author", self.name)})
        result = {"provider": "exa", "requested_url": url, "url": returned_url,
                  "title": _string(row, "title", self.name), "content": text, "content_format": "text",
                  "credits": None, "request_id": _string(response, "requestId", self.name),
                  "provider_metadata": {"exa": metadata}}
        return redact_web_result(result, config.api_key)


def _schema(name: str, argument: str, description: str, argument_description: str) -> dict[str, object]:
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": {argument: {"type": "string", "description": argument_description}},
                           "required": [argument], "additionalProperties": False}, "strict": True}


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
    return {"url": public_web_url(value.get("url"), origin), "title": _string(value, "title", origin),
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
