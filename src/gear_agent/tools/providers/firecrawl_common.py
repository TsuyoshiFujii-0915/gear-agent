from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from gear_agent.errors import GearError, gear_error
from gear_agent.tools.providers.base import WebUsage


def invalid_response(field: str, origin: str) -> GearError:
    """Identifies an invalid response field without copying its upstream value."""
    return gear_error('firecrawl_response_invalid', f'Invalid Firecrawl response field: {field}.',
                      origin, True, {'field': field})


def response_data(response: dict[str, Any], origin: str) -> dict[str, Any]:
    """Requires the raw REST v2 success envelope, never an unwrapped SDK result.

    Args:
        response: Parsed REST response.
        origin: Tool name for safe diagnostics.

    Returns:
        The validated data object.

    Raises:
        GearError: When Firecrawl reports failure or violates the v2 envelope.
    """
    if response.get('success') is False:
        raise gear_error('firecrawl_failed', 'Firecrawl reported an upstream failure.', origin, True, {})
    if response.get('success') is not True:
        raise invalid_response('success', origin)
    reject_error(response, 'error', origin)
    data = response.get('data')
    if not isinstance(data, dict):
        raise invalid_response('data', origin)
    reject_error(data, 'data.error', origin)
    return data


def reject_error(payload: dict[str, Any], field: str, origin: str) -> None:
    """Rejects reported errors without exposing arbitrary upstream messages."""
    error = payload.get('error')
    if error is None:
        return
    if not isinstance(error, str):
        raise invalid_response(field, origin)
    if error:
        raise gear_error('firecrawl_failed', 'Firecrawl reported an upstream failure.',
                         origin, True, {'field': field})


def optional_string(payload: dict[str, Any], key: str, field: str, origin: str) -> str | None:
    """Reads nullable string metadata without inventing an unavailable value."""
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise invalid_response(field, origin)
    return value


def optional_integer(payload: dict[str, Any], key: str, field: str, origin: str) -> int | None:
    """Reads nullable nonnegative integer metadata, excluding booleans."""
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise invalid_response(field, origin)
    return value


def reported_usage(payload: dict[str, Any], origin: str) -> tuple[WebUsage, ...]:
    """Records only reported creditsUsed, without inferring cost or zero usage."""
    credits = optional_integer(payload, 'creditsUsed', 'creditsUsed', origin)
    return () if credits is None else (WebUsage('credits', credits),)


def safe_warning(payload: dict[str, Any], origin: str) -> str | None:
    """Projects warnings to a fixed diagnostic that cannot echo credentials."""
    warning = optional_string(payload, 'warning', 'warning', origin)
    return 'Firecrawl reported a warning.' if warning else None


def response_url(value: object, field: str, origin: str) -> str:
    """Validates reported HTTP URL syntax without resolving or fetching it."""
    try:
        if not isinstance(value, str) or not value or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value) or '\\' in value:
            raise ValueError('Invalid URL')
        parsed = urlsplit(value)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username is not None or parsed.password is not None:
            raise ValueError('Invalid URL')
        if parsed.port is not None and parsed.port < 1:
            raise ValueError('Invalid port')
    except ValueError:
        raise invalid_response(field, origin) from None
    return value
