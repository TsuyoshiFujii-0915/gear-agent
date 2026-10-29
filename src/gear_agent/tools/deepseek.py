from __future__ import annotations

import json
import math
import re
from urllib.parse import urlsplit

import httpx

from gear_agent.errors import GearError, gear_error


class DeepSeekTransport:
    """Bounded, single-attempt transport for the standalone DeepSeek APIs."""

    def __init__(self, transport: httpx.BaseTransport, max_response_bytes: int) -> None:
        _positive_integer(max_response_bytes, "max_response_bytes", "web_transport")
        self._transport = transport
        self._max_response_bytes = max_response_bytes

    def post(
        self,
        operation: str,
        api_key: str,
        payload: dict[str, object],
        timeout_seconds: int,
    ) -> dict[str, object]:
        """Posts JSON without redirects, retries, or upstream error disclosure.

        Args:
            operation: The search or fetch API operation.
            api_key: Dedicated web API credential.
            payload: Validated request body.
            timeout_seconds: Timeout for each HTTP operation.

        Returns:
            A validated JSON object, without an embedded error.

        Raises:
            GearError: If the request, response, or configured limit fails.
        """
        if operation not in {"search", "fetch"}:
            raise gear_error("deepseek_operation_invalid", "Unknown DeepSeek operation.",
                             "web_transport", False, {})
        origin = f"web_{operation}"
        _positive_integer(timeout_seconds, "timeout_seconds", origin)
        body = bytearray()
        try:
            with httpx.Client(transport=self._transport, follow_redirects=False,
                              timeout=timeout_seconds) as client:
                request = client.build_request(
                    "POST", f"https://api.deepseek.com/v1/tools/{operation}",
                    headers={"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json", "Accept-Encoding": "identity"},
                    json=payload,
                )
                response = client.send(request, stream=True)
                try:
                    if response.status_code != 200:
                        raise _http_error(response, origin, api_key)
                    encoding = response.headers.get("content-encoding")
                    if encoding is not None and encoding.lower() != "identity":
                        raise _invalid_response(origin, "Unexpected response content encoding.")
                    for chunk in response.iter_bytes(chunk_size=1024):
                        if len(body) + len(chunk) > self._max_response_bytes:
                            raise gear_error(
                                "deepseek_response_too_large", "DeepSeek response exceeds byte limit.",
                                origin, True, {"max_response_bytes": self._max_response_bytes},
                            )
                        body.extend(chunk)
                finally:
                    response.close()
        except httpx.TimeoutException:
            raise gear_error("deepseek_http_timeout", "DeepSeek request timed out.", origin,
                             True, {"timeout_seconds": timeout_seconds}) from None
        except httpx.HTTPError:
            raise gear_error("deepseek_http_request_failed", "Failed to reach DeepSeek.",
                             origin, True, {}) from None
        try:
            parsed = json.loads(body.decode("utf-8"), parse_constant=_reject_constant)
        except (ValueError, UnicodeError):
            raise _invalid_response(origin, "DeepSeek returned invalid UTF-8 JSON.") from None
        if not isinstance(parsed, dict):
            raise _invalid_response(origin, "DeepSeek response must be an object.")
        if "error" in parsed:
            raise _invalid_response(origin, "DeepSeek returned an error in a successful HTTP response.")
        return parsed


class DeepSeekSearchProvider:
    """Search using DeepSeek without conflating summaries, bodies, or scores."""

    def __init__(
        self, api_key: str, timeout_seconds: int, max_results: int,
        need_content: bool, transport: DeepSeekTransport,
    ) -> None:
        _positive_integer(max_results, "max_results", "web_search")
        if max_results > 50:
            raise gear_error("config_value_invalid", "DeepSeek max_results must be 1 to 50.",
                             "web_search", True, {"key": "max_results"})
        if not isinstance(need_content, bool):
            raise gear_error("config_value_invalid", "DeepSeek need_content must be boolean.",
                             "web_search", True, {"key": "need_content"})
        _positive_integer(timeout_seconds, "timeout_seconds", "web_search")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_results = max_results
        self._need_content = need_content
        self._transport = transport

    def search(self, query: str) -> dict[str, object]:
        """Searches an intact expression containing 1 to 128 Unicode characters.

        Args:
            query: Search expression, including any operators.

        Returns:
            Query and results with separate snippet, summary, and raw content.

        Raises:
            GearError: If input, transport, or response validation fails.
        """
        if (not isinstance(query, str) or not query.strip() or len(query) > 128
                or any(0xD800 <= ord(char) <= 0xDFFF for char in query)):
            raise gear_error("tool_argument_invalid",
                             "DeepSeek query must contain 1 to 128 Unicode characters and not be blank.",
                             "web_search", True, {"key": "query", "max_length": 128})
        response = self._transport.post("search", self._api_key, {
            "query": query, "count": self._max_results, "need_content": self._need_content,
        }, self._timeout_seconds)
        results = response.get("results")
        if not isinstance(results, list):
            raise _invalid_response("web_search", "DeepSeek response.results must be an array.")
        return {"query": query, "results": [_search_result(item, index)
                                             for index, item in enumerate(results)],
                "provider": "deepseek"}


class DeepSeekFetchProvider:
    """Fetch a single page as Markdown through DeepSeek."""

    def __init__(
        self, api_key: str, timeout_seconds: int, max_content_chars: int,
        transport: DeepSeekTransport,
    ) -> None:
        _positive_integer(max_content_chars, "max_content_chars", "web_fetch")
        _positive_integer(timeout_seconds, "timeout_seconds", "web_fetch")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_content_chars = max_content_chars
        self._transport = transport

    def fetch(self, url: str) -> dict[str, object]:
        """Fetches Markdown while retaining the requested URL.

        Args:
            url: HTTP(S) page URL without credentials, at most 8192 characters.

        Returns:
            Requested URL, title, Markdown content, and provider identity.

        Raises:
            GearError: If input, transport, content, or configured limits fail.
        """
        _validate_url(url)
        response = self._transport.post("fetch", self._api_key, {"url": url}, self._timeout_seconds)
        title = _string(response, "title", "response", "web_fetch")
        content = _string(response, "content", "response", "web_fetch")
        if not content.strip():
            raise gear_error("deepseek_content_empty", "DeepSeek returned empty page content.",
                             "web_fetch", True, {})
        if len(content) > self._max_content_chars:
            raise gear_error("deepseek_content_too_large", "DeepSeek page exceeds content limit.",
                             "web_fetch", True, {"max_content_chars": self._max_content_chars,
                                                 "content_chars": len(content)})
        return {"url": url, "title": title, "content": content,
                "content_format": "markdown", "provider": "deepseek"}


def _search_result(item: object, index: int) -> dict[str, object]:
    field = f"results[{index}]"
    if not isinstance(item, dict):
        raise _invalid_response("web_search", f"DeepSeek {field} must be an object.")
    title = _string(item, "title", field, "web_search")
    url = _string(item, "url", field, "web_search")
    if not url:
        raise _invalid_response("web_search", f"DeepSeek {field}.url must not be empty.")
    snippet = _string(item, "snippet", field, "web_search")
    summary = _optional_string(item, "summary", field)
    content = _optional_string(item, "content", field)
    metadata: dict[str, object] = {}
    for key in ("site_name", "site_icon", "date"):
        if key in item:
            metadata[key] = _optional_string(item, key, field)
    score = item.get("rerank_score")
    if score is not None:
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 3):
            raise _invalid_response("web_search", f"DeepSeek {field}.rerank_score must be 0 to 3.")
        metadata["rerank_score"] = score
    authority = item.get("authority")
    if authority is not None:
        if type(authority) is not int or not -1 <= authority <= 4:
            raise _invalid_response("web_search", f"DeepSeek {field}.authority must be -1 to 4.")
        metadata["authority"] = authority
    return {"title": title, "url": url, "content": snippet, "snippet": snippet,
            "summary": summary, "raw_content": None if content == "" else content,
            "score": None, "metadata": {"deepseek": metadata}}


def _string(data: dict[str, object], key: str, field: str, origin: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise _invalid_response(origin, f"DeepSeek {field}.{key} must be a string.")
    return value


def _optional_string(data: dict[str, object], key: str, field: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    return _string(data, key, field, "web_search")


def _positive_integer(value: int, key: str, origin: str) -> None:
    if type(value) is not int or value < 1:
        raise gear_error("config_value_invalid", f"DeepSeek {key} must be a positive integer.",
                         origin, True, {"key": key})


def _validate_url(url: str) -> None:
    valid = isinstance(url, str) and 1 <= len(url) <= 8192
    if valid:
        try:
            parsed = urlsplit(url)
            valid = (parsed.scheme in {"http", "https"} and bool(parsed.hostname)
                     and parsed.username is None and parsed.password is None
                     and not any(char.isspace() or ord(char) < 32 or ord(char) == 127
                                 or 0xD800 <= ord(char) <= 0xDFFF for char in url))
            parsed.port
        except ValueError:
            valid = False
    if not valid:
        raise gear_error("tool_argument_invalid",
                         "DeepSeek URL must be HTTP(S), have a host, contain no credentials, "
                         "and be at most 8192 characters.",
                         "web_fetch", True, {"key": "url", "max_length": 8192})


def _invalid_response(origin: str, message: str) -> GearError:
    return gear_error("deepseek_response_invalid", message, origin, True, {})


def _reject_constant(value: str) -> object:
    raise ValueError("Non-finite JSON number")


def _http_error(response: httpx.Response, origin: str, api_key: str) -> GearError:
    status = response.status_code
    categories = {400: "invalid_request", 401: "authentication", 402: "insufficient_balance",
                  403: "security_policy", 404: "content_unavailable", 408: "timeout",
                  413: "request_too_large", 429: "rate_limit", 504: "timeout"}
    if status in categories:
        category = categories[status]
    elif 300 <= status < 400:
        category = "redirect"
    elif status >= 500:
        category = "server"
    else:
        category = "unexpected_status"
    details: dict[str, object] = {"provider": "deepseek", "status": status, "category": category}
    request_id = response.headers.get("x-request-id")
    if (request_id is not None and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request_id)
            and api_key not in request_id):
        details["request_id"] = request_id
    message = f"DeepSeek returned HTTP {status} ({category})."
    if status == 401:
        message += " Check the configured DeepSeek API key before retrying."
    elif status == 402:
        message += " Check your DeepSeek account balance and top up before retrying."
    return gear_error("deepseek_http_status_error", message, origin, True, details)
