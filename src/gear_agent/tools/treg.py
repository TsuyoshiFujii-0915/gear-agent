from __future__ import annotations

import ipaddress
import json
from time import monotonic
from typing import Any
from urllib.parse import quote, quote_plus, urlsplit

import httpx

from gear_agent.errors import GearError, gear_error
from gear_agent.treg_config import TregFetchConfig, TregSearchConfig


BASE_URL = "https://treg.to"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class TregSearchProvider:
    """Search through the fixed TinyFish endpoint, without routing or retries."""

    def __init__(self, config: TregSearchConfig, transport: httpx.BaseTransport | None) -> None:
        self._config = config
        self._http = _TregHTTP(config.api_key, config.org, config.timeout_seconds, transport, "web_search")

    def search(self, query: str) -> dict[str, Any]:
        """Returns ranked snippets with a local limit and reported call metadata.

        Args:
            query: Nonempty search text supplied by the user or model.

        Returns:
            Normalized snippets and metadata; missing billing remains None.

        Raises:
            GearError: If input, catalog, HTTP, billing, or result validation fails.
        """
        if not isinstance(query, str) or not query.strip():
            raise _error("tool_argument_invalid", "web_search requires a nonempty query.", "web_search")
        response, metadata = self._http.call("tinyfish.web.search", "GET", {"query": query, "domain_type": "web"}, None)
        _check_errors(response, "web_search")
        results = _results(response, "web_search")
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(results):
            location = f"results[{index}]"
            position = item.get("position")
            if position is not None and (type(position) is not int or position < 1):
                raise _invalid(f"{location}.position must be a positive integer when reported.", "web_search")
            normalized.append({
                "title": _string(item, "title", location, "web_search"),
                "url": _string(item, "url", location, "web_search"),
                "content": _string(item, "snippet", location, "web_search"),
                "score": None,
                "raw_content": None,
                "provider_metadata": {"position": position, "content_kind": "snippet"},
            })
        return self._http.redact({
            "query": query, "answer": None, "results": normalized[:self._config.max_results],
            "provider_metadata": metadata | {"max_results": self._config.max_results, "limit_scope": "local"},
        })


class TregFetchProvider:
    """Fetch one URL as Markdown through the fixed TinyFish endpoint."""

    def __init__(self, config: TregFetchConfig, transport: httpx.BaseTransport | None) -> None:
        self._config = config
        self._http = _TregHTTP(config.api_key, config.org, config.timeout_seconds, transport, "web_fetch")

    def fetch(self, url: str) -> dict[str, Any]:
        """Returns the requested page's text, preserving known redirect metadata.

        Args:
            url: Public HTTP(S) URL without userinfo.

        Returns:
            Page content and reported metadata, including configured freshness.

        Raises:
            GearError: If input, catalog, transport, or page validation fails.
        """
        _validate_url(url)
        response, metadata = self._http.call("tinyfish.web.fetch", "POST", None, {
            "urls": [url], "format": "markdown", "ttl": self._config.ttl_seconds,
            "per_url_timeout_ms": self._config.per_url_timeout_ms,
        })
        if "errors" not in response:
            raise _invalid("Fetch response.errors is required.", "web_fetch")
        _check_errors(response, "web_fetch")
        results = _results(response, "web_fetch")
        if len(results) != 1 or results[0].get("url") != url:
            raise _invalid("Fetch response must contain exactly one result matching the requested URL.", "web_fetch")
        item = results[0]
        content = _string(item, "text", "results[0]", "web_fetch")
        if not content.strip():
            raise _error("treg_content_empty", "TinyFish returned empty page text.", "web_fetch")
        if len(content) > self._config.max_content_chars:
            raise _error("treg_content_too_large", "TinyFish page text exceeds max_content_chars.", "web_fetch")
        content_format = _string(item, "format", "results[0]", "web_fetch")
        if content_format != "markdown":
            raise _invalid("Fetch response.results[0].format must be markdown.", "web_fetch")
        return self._http.redact({
            "url": url, "title": _nullable_string(item, "title", "web_fetch"), "content": content,
            "provider_metadata": metadata | {
                "final_url": _nullable_string(item, "final_url", "web_fetch"),
                "format": content_format, "ttl_seconds": self._config.ttl_seconds,
                "per_url_timeout_ms": self._config.per_url_timeout_ms,
            },
        })


class _TregHTTP:
    def __init__(self, token: str, org: str | None, timeout_seconds: int,
                 transport: httpx.BaseTransport | None, origin: str) -> None:
        self._token = token
        self._org = org
        self._timeout = timeout_seconds
        self._transport = transport
        self._origin = origin

    def call(self, endpoint: str, method: str, params: dict[str, str] | None,
             body: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
        started = monotonic()
        with httpx.Client(transport=self._transport, timeout=self._timeout, follow_redirects=False,
                          headers={"Accept-Encoding": "identity"}) as client:
            catalog, _ = self._request(client, "GET", f"/catalog/endpoints/{endpoint}", {}, None, None)
            self._check_catalog(catalog, endpoint, method)
            headers = {"X-Treg-Token": self._token, "X-Treg-Route-Max-Cost": "0"}
            if self._org is not None:
                headers["X-Treg-Org"] = self._org
            response, reported = self._request(client, method, f"/call/{endpoint}", headers, params, body)
        metadata = self._metadata(reported, endpoint)
        metadata["latency_seconds"] = monotonic() - started
        return response, metadata

    def _request(self, client: httpx.Client, method: str, path: str, headers: dict[str, str],
                 params: dict[str, str] | None, body: dict[str, Any] | None) -> tuple[dict[str, Any], httpx.Headers]:
        try:
            request = client.build_request(method, BASE_URL + path, headers=headers, params=params, json=body)
            response = client.send(request, stream=True)
            try:
                if response.status_code != 200:
                    raise gear_error("treg_http_status_error", f"Treg {path} returned HTTP {response.status_code}.",
                                     self._origin, True, {"status": response.status_code, "endpoint": path})
                encoding = response.headers.get("Content-Encoding")
                if encoding is not None and encoding.lower() != "identity":
                    raise _invalid("Treg response must use identity content encoding.", self._origin)
                length = response.headers.get("Content-Length")
                if length is not None:
                    if not length.isascii() or not length.isdigit() or len(length) > 20:
                        raise _invalid("Treg Content-Length must be a nonnegative integer.", self._origin)
                    if int(length) > MAX_RESPONSE_BYTES:
                        raise _error("treg_response_too_large", f"Treg {path} response exceeds 8 MiB.", self._origin)
                data = bytearray()
                for chunk in response.iter_bytes(chunk_size=65536):
                    if len(data) + len(chunk) > MAX_RESPONSE_BYTES:
                        raise _error("treg_response_too_large", f"Treg {path} response exceeds 8 MiB.", self._origin)
                    data.extend(chunk)
                reported = response.headers
            finally:
                response.close()
        except httpx.TimeoutException:
            raise _error("treg_http_timeout", f"Treg {path} request timed out.", self._origin) from None
        except httpx.HTTPError:
            raise _error("treg_http_request_failed", f"Treg {path} HTTP request failed.", self._origin) from None
        try:
            parsed = json.loads(data)
        except (ValueError, UnicodeError):
            raise _error("treg_json_parse_failed", f"Treg {path} returned invalid JSON.", self._origin) from None
        if not isinstance(parsed, dict):
            raise _invalid(f"Treg {path} response must be an object.", self._origin)
        return parsed, reported

    def _check_catalog(self, catalog: dict[str, Any], endpoint: str, method: str) -> None:
        entry = catalog.get("endpoint")
        if not isinstance(entry, dict) or any(entry.get(key) != value for key, value in
                                              (("id", endpoint), ("provider", "tinyfish"), ("method", method))):
            raise _error("treg_catalog_mismatch", "Treg catalog does not match the fixed TinyFish endpoint.", self._origin)
        cost = entry.get("cost")
        if (not isinstance(cost, dict) or cost.get("type") != "free" or cost.get("currency") != "USD"
                or type(cost.get("value")) not in (int, float) or cost["value"] != 0
                or type(cost.get("usd")) not in (int, float) or cost["usd"] != 0):
            raise _error("treg_price_unverified", "Treg catalog no longer confirms a free USD price; call refused.", self._origin)
        if any(entry.get(key) is not None for key in ("overflow_via", "overflow_price_usd", "routed_children")):
            raise _error("treg_route_mismatch", "Treg catalog advertises routing or overflow; call refused.", self._origin)

    def _metadata(self, headers: httpx.Headers, endpoint: str) -> dict[str, Any]:
        charged: int | None = None
        raw_cost = headers.get("X-Treg-Cost-Micro")
        if raw_cost is not None:
            if not raw_cost.isascii() or not raw_cost.isdigit() or len(raw_cost) > 20:
                raise _invalid("Treg X-Treg-Cost-Micro must be a nonnegative integer.", self._origin)
            charged = int(raw_cost)
            if charged != 0:
                raise gear_error("treg_unexpected_charge", "Treg reported a charge for a free direct call.",
                                 self._origin, True, {"endpoint": endpoint, "charged_micro": charged})
        served_by = headers.get("X-Treg-Served-By")
        if served_by is not None and served_by != endpoint:
            raise _error("treg_route_mismatch", "Treg reported an unexpected upstream endpoint.", self._origin)
        if any(key in headers for key in ("X-Treg-Served-Via", "X-Treg-Routed-Tool", "X-Treg-Async")):
            raise _error("treg_route_mismatch", "Treg reported overflow, routing, or an asynchronous call.", self._origin)
        if "X-Treg-Upstream-Failed" in headers:
            raise _error("treg_upstream_failed", "Treg reported an upstream failure.", self._origin)
        return {"provider": "treg", "endpoint": endpoint, "served_by": served_by,
                "charged_micro": charged, "cost_unit": "micro_USD", "call_id": headers.get("X-Treg-Call-Id")}

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            for secret in (self._token, quote(self._token, safe=""), quote_plus(self._token)):
                if secret:
                    value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {key: self.redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        return value


def _validate_url(url: str) -> None:
    valid = isinstance(url, str) and bool(url) and not any(ord(char) < 33 or ord(char) == 127 for char in url)
    try:
        parsed = urlsplit(url) if valid else None
        valid = (parsed is not None and parsed.scheme in ("http", "https") and bool(parsed.hostname)
                 and parsed.username is None and parsed.password is None)
        if valid:
            host = parsed.hostname.rstrip(".").lower()
            valid = host != "localhost" and not host.endswith((".localhost", ".local", ".internal"))
            try:
                valid = valid and ipaddress.ip_address(host).is_global
            except ValueError:
                valid = valid and "." in host
            _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise _error("tool_argument_invalid", "web_fetch requires a public HTTP(S) URL without credentials or control characters.", "web_fetch")


def _results(response: dict[str, Any], origin: str) -> list[dict[str, Any]]:
    items = response.get("results")
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise _invalid("TinyFish response.results must be an array of objects.", origin)
    return items


def _check_errors(response: dict[str, Any], origin: str) -> None:
    if "errors" in response and (not isinstance(response["errors"], list) or response["errors"]):
        raise _error("treg_upstream_failed", "TinyFish response.errors reports failure or is malformed.", origin)
    if "error" in response:
        raise _error("treg_upstream_failed", "TinyFish response contains an error.", origin)
    if "_treg" in response:
        raise _error("treg_route_mismatch", "Unexpected routed response envelope from a direct endpoint.", origin)


def _string(item: dict[str, Any], key: str, location: str, origin: str) -> str:
    value = item.get(key)
    if not isinstance(value, str):
        raise _invalid(f"TinyFish {location}.{key} must be a string.", origin)
    return value


def _nullable_string(item: dict[str, Any], key: str, origin: str) -> str | None:
    value = item.get(key)
    if value is not None and not isinstance(value, str):
        raise _invalid(f"TinyFish results[0].{key} must be a string when reported.", origin)
    return value


def _invalid(message: str, origin: str) -> GearError:
    return _error("treg_response_invalid", message, origin)


def _error(category: str, message: str, origin: str) -> GearError:
    return gear_error(category, message, origin, True, {"provider": "treg"})
