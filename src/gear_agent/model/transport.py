from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator
import json

import httpx
from httpx_sse import SSEError, connect_sse

from gear_agent.errors import GearError, gear_error


class HttpTransport(ABC):
    """HTTP transport used by ModelClient."""

    @abstractmethod
    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Sends a JSON POST request.

        Args:
            url: Complete endpoint URL.
            headers: HTTP headers.
            payload: JSON-serializable request body.
            timeout_seconds: Request timeout in seconds.

        Returns:
            Parsed JSON response.
        """

    def post_sse(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
        idle_timeout_seconds: float,
    ) -> Iterator[SseEvent]:
        """Sends a JSON POST request and yields parsed SSE messages.

        Args:
            url: Complete endpoint URL.
            headers: HTTP headers.
            payload: JSON-serializable request body.
            timeout_seconds: Connect, write, and pool timeout in seconds.
            idle_timeout_seconds: Maximum time between received stream bytes.

        Returns:
            Parsed SSE message iterator.

        Raises:
            NotImplementedError: If a transport only implements JSON requests.
        """

        raise NotImplementedError("This HTTP transport does not support SSE streaming.")


@dataclass(frozen=True)
class SseEvent:
    """SSE message normalized at the HTTP transport boundary."""

    event: str
    data: str


class HttpxHttpTransport(HttpTransport):
    """HTTPX transport supporting JSON and incremental SSE responses."""

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout_seconds)) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise _http_status_error(url, exc) from exc
        except httpx.TimeoutException as exc:
            raise gear_error(
                "http_timeout",
                "Model endpoint request timed out.",
                "model_client",
                True,
                {"url": url, "timeout_seconds": timeout_seconds},
            ) from exc
        except httpx.RequestError as exc:
            raise gear_error(
                "http_request_failed",
                "Failed to reach model endpoint.",
                "model_client",
                True,
                {"url": url, "reason": str(exc)},
            ) from exc

        try:
            parsed = response.json()
        except json.JSONDecodeError as exc:
            raise gear_error(
                "json_parse_failed",
                "Model endpoint returned invalid JSON.",
                "model_client",
                True,
                {"url": url, "body": response.text},
            ) from exc

        if not isinstance(parsed, dict):
            raise gear_error(
                "json_shape_invalid",
                "Model endpoint returned JSON that is not an object.",
                "model_client",
                True,
                {"url": url},
            )
        return parsed

    def post_sse(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
        idle_timeout_seconds: float,
    ) -> Iterator[SseEvent]:
        """Sends a JSON POST request and yields parsed SSE messages.

        Args:
            url: Complete endpoint URL.
            headers: HTTP headers.
            payload: JSON-serializable request body.
            timeout_seconds: Connect, write, and pool timeout in seconds.
            idle_timeout_seconds: Maximum time between received stream bytes.

        Yields:
            Parsed SSE messages.
        """

        timeout = httpx.Timeout(
            connect=timeout_seconds,
            read=idle_timeout_seconds,
            write=timeout_seconds,
            pool=timeout_seconds,
        )
        try:
            status_error: httpx.HTTPStatusError | None = None
            with httpx.Client(timeout=timeout) as client:
                with connect_sse(
                    client,
                    "POST",
                    url,
                    headers=headers,
                    json=payload,
                ) as event_source:
                    try:
                        event_source.response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        event_source.response.read()
                        status_error = exc
                    if status_error is None:
                        for event in event_source.iter_sse():
                            yield SseEvent(event=event.event, data=event.data)
            if status_error is not None:
                raise _http_status_error(url, status_error) from status_error
        except GearError:
            raise
        except httpx.ReadTimeout as exc:
            raise gear_error(
                "http_stream_idle_timeout",
                "Model response stream exceeded its idle timeout.",
                "model_client",
                True,
                {
                    "url": url,
                    "idle_timeout_seconds": idle_timeout_seconds,
                },
            ) from exc
        except httpx.TimeoutException as exc:
            raise gear_error(
                "http_timeout",
                "Model endpoint request timed out.",
                "model_client",
                True,
                {"url": url, "timeout_seconds": timeout_seconds},
            ) from exc
        except SSEError as exc:
            raise gear_error(
                "http_sse_invalid",
                "Model endpoint did not return a valid SSE response.",
                "model_client",
                True,
                {"url": url, "reason": str(exc)},
            ) from exc
        except httpx.RequestError as exc:
            raise gear_error(
                "http_stream_failed",
                "Model response stream failed after the request started.",
                "model_client",
                True,
                {"url": url, "reason": str(exc)},
            ) from exc


def _http_status_error(url: str, error: httpx.HTTPStatusError) -> Exception:
    response = error.response
    return gear_error(
        "http_status_error",
        f"Model endpoint returned HTTP {response.status_code}.",
        "model_client",
        True,
        {
            "url": url,
            "status": response.status_code,
            "body": response.text,
        },
    )
