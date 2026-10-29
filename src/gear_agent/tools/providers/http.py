from __future__ import annotations

import json
from typing import Any

import httpx

from gear_agent.errors import GearError, gear_error
from gear_agent.config import MAX_WEB_RESPONSE_BYTES


class WebHttpClient:
    """Bounded JSON HTTP with safe errors and no redirects or retries."""

    def __init__(self, client: httpx.Client, max_response_bytes: int) -> None:
        if isinstance(max_response_bytes, bool) or max_response_bytes < 1:
            raise ValueError('WebHttpClient.max_response_bytes must be positive.')
        self._client = client
        self._limit = max_response_bytes

    def post_json(
        self, url: str, api_key: str, payload: dict[str, object], timeout_seconds: int,
        origin: str, error_prefix: str,
    ) -> dict[str, Any]:
        """Posts to a fixed API endpoint and rejects unsafe responses.

        Args:
            url: Adapter-owned API endpoint, never a model argument.
            api_key: Environment-resolved bearer credential.
            payload: Vendor JSON body.
            timeout_seconds: Explicit HTTP timeout.
            origin: Tool identity for recoverable errors.
            error_prefix: Adapter error namespace, preserving existing categories.

        Returns:
            Parsed JSON object within the response byte budget.

        Raises:
            GearError: On HTTP, size, encoding, network or JSON failure.
        """
        return self.post_json_with_headers(url, {'Authorization': f'Bearer {api_key}'}, payload,
                                           timeout_seconds, origin, error_prefix)

    def post_json_with_headers(
        self, url: str, headers: dict[str, str], payload: dict[str, object], timeout_seconds: int,
        origin: str, error_prefix: str,
    ) -> dict[str, Any]:
        """Posts with adapter-owned authentication through the shared HTTP policy.

        Args:
            url: Fixed provider API URL.
            headers: Adapter authentication headers, never model arguments.
            payload: Vendor JSON body.
            timeout_seconds: Explicit HTTP timeout.
            origin: Tool identity for recoverable errors.
            error_prefix: Adapter error namespace.

        Returns:
            Bounded JSON response object.
        """
        def failure(category: str, message: str, details: dict[str, object]) -> GearError:
            return gear_error(f'{error_prefix}_{category}', message, origin, True, details)

        try:
            request = self._client.build_request(
                'POST', url, json=payload,
                headers={**headers, 'Accept-Encoding': 'identity'},
                timeout=timeout_seconds,
            )
            response = self._client.send(request, stream=True, follow_redirects=False)
            try:
                if not 200 <= response.status_code < 300:
                    raise failure('http_status_error', f'Web provider returned HTTP {response.status_code}.',
                                  {'status': response.status_code})
                encoding = response.headers.get('Content-Encoding')
                if encoding is not None and encoding.lower() != 'identity':
                    raise failure('response_encoding_invalid', 'Web provider returned unsupported content encoding.', {})
                length = response.headers.get('Content-Length')
                if length is not None:
                    if not length.isascii() or not length.isdecimal():
                        raise failure('response_invalid', 'Web provider returned invalid Content-Length.', {})
                    if len(length) > 20 or int(length) > self._limit:
                        raise failure('response_too_large', 'Web provider response exceeds the byte limit.',
                                      {'max_response_bytes': self._limit})
                body = bytearray()
                for chunk in response.iter_bytes():
                    if len(body) + len(chunk) > self._limit:
                        raise failure('response_too_large', 'Web provider response exceeds the byte limit.',
                                      {'max_response_bytes': self._limit})
                    body.extend(chunk)
            finally:
                response.close()
        except httpx.TimeoutException:
            raise failure('http_timeout', 'Web provider request timed out.',
                          {'timeout_seconds': timeout_seconds}) from None
        except httpx.HTTPError:
            raise failure('http_request_failed', 'Web provider HTTP request failed.', {}) from None
        try:
            parsed = json.loads(body.decode('utf-8'), parse_constant=_reject_json_constant,
                                object_pairs_hook=_unique_object)
        except (ValueError, RecursionError):
            raise failure('json_parse_failed', 'Web provider returned invalid JSON.', {}) from None
        if not isinstance(parsed, dict):
            raise failure('response_invalid', 'Web provider response must be a JSON object.', {})
        return parsed


def _reject_json_constant(value: str) -> None:
    raise ValueError('Nonfinite JSON number is not supported.')


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key.')
        result[key] = value
    return result


def post_web_json(
    http: WebHttpClient | None, url: str, api_key: str, payload: dict[str, object],
    timeout_seconds: int, origin: str, error_prefix: str,
) -> dict[str, Any]:
    """Uses an injected HTTP client or a scoped production client.

    Args:
        http: Explicit dependency injection seam; None owns a per-call client.
        url: Fixed vendor API URL.
        api_key: Environment-resolved credential.
        payload: Request body.
        timeout_seconds: Request timeout.
        origin: Tool error origin.
        error_prefix: Vendor error namespace.

    Returns:
        Bounded, parsed JSON response.
    """
    if http is not None:
        return http.post_json(url, api_key, payload, timeout_seconds, origin, error_prefix)
    with httpx.Client(follow_redirects=False) as client:
        return WebHttpClient(client, MAX_WEB_RESPONSE_BYTES).post_json(
            url, api_key, payload, timeout_seconds, origin, error_prefix,
        )
