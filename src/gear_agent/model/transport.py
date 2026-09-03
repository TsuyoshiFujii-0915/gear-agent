from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json

from gear_agent.errors import gear_error


class HttpTransport(ABC):
    """HTTP transport used by ModelClient."""

    @abstractmethod
    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: int,
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


class UrllibHttpTransport(HttpTransport):
    """Standard-library HTTP transport."""

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        request_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(url, data=request_bytes, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise gear_error(
                "http_status_error",
                f"Model endpoint returned HTTP {exc.code}.",
                "model_client",
                True,
                {"url": url, "status": exc.code, "body": response_body},
            ) from exc
        except URLError as exc:
            raise gear_error(
                "http_request_failed",
                "Failed to reach model endpoint.",
                "model_client",
                True,
                {"url": url, "reason": str(exc.reason)},
            ) from exc
        except TimeoutError as exc:
            raise gear_error(
                "http_timeout",
                "Model endpoint request timed out.",
                "model_client",
                True,
                {"url": url, "timeout_seconds": timeout_seconds},
            ) from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise gear_error(
                "json_parse_failed",
                "Model endpoint returned invalid JSON.",
                "model_client",
                True,
                {"url": url, "body": response_body},
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
