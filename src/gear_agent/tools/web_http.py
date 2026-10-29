from __future__ import annotations

from typing import Any
import json

import httpx

from gear_agent.errors import gear_error


class WebJsonTransport:
    """Bounded JSON HTTP requests without redirects, retries or upstream diagnostics."""

    def post(
        self, url: str, headers: dict[str, str], payload: dict[str, object],
        timeout_seconds: int, max_response_bytes: int, origin: str,
    ) -> dict[str, Any]:
        """Posts JSON and rejects oversized or malformed responses.

        Args:
            url: Fixed provider endpoint, never a model-supplied URL.
            headers: Provider authentication headers.
            payload: Provider request body.
            timeout_seconds: HTTP operation timeout in seconds.
            max_response_bytes: Maximum accepted response body size.
            origin: Tool name for safe diagnostics.

        Returns:
            Parsed JSON object.

        Raises:
            GearError: For HTTP, size, encoding, network or JSON failures, with
                no upstream body, URL credentials or exception message attached.
        """
        try:
            with httpx.Client(timeout=timeout_seconds, follow_redirects=False) as client:
                request = client.build_request(
                    "POST", url, json=payload,
                    headers={**headers, "Accept-Encoding": "identity"},
                )
                response = client.send(request, stream=True)
                try:
                    if not 200 <= response.status_code < 300:
                        raise gear_error("web_http_status_error",
                                         f"Web provider returned HTTP {response.status_code}.",
                                         origin, True, {"status": response.status_code})
                    # Reject compression to bound memory before decompression as well.
                    if response.headers.get("content-encoding", "identity") != "identity":
                        raise gear_error("web_http_encoding_invalid", "Expected an uncompressed web response.",
                                         origin, True, {})
                    body = bytearray()
                    for chunk in response.iter_bytes(chunk_size=8192):
                        if len(body) + len(chunk) > max_response_bytes:
                            raise gear_error("web_http_response_too_large", "Web response exceeds the byte limit.",
                                             origin, True, {"max_response_bytes": max_response_bytes})
                        body.extend(chunk)
                finally:
                    response.close()
        except httpx.TimeoutException:
            raise gear_error("web_http_timeout", "Web provider request timed out.", origin, True,
                             {"timeout_seconds": timeout_seconds}) from None
        except httpx.HTTPError:
            raise gear_error("web_http_request_failed", "Web provider request failed.", origin, True, {}) from None
        try:
            parsed = json.loads(body.decode("utf-8"), parse_constant=_invalid_constant,
                                object_pairs_hook=_unique_object)
        except (ValueError, UnicodeError, RecursionError):
            raise gear_error("web_json_invalid", "Web provider returned invalid JSON.", origin, True, {}) from None
        if not isinstance(parsed, dict):
            raise gear_error("web_json_invalid", "Web provider response must be an object.", origin, True, {})
        return parsed


def _invalid_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number.")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key.")
        result[key] = value
    return result
