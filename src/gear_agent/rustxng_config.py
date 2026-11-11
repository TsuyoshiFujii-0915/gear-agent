from __future__ import annotations

from dataclasses import dataclass, field
from ipaddress import ip_address
from typing import Literal
from urllib.parse import unquote, urlsplit
import re
import unicodedata

from gear_agent.errors import gear_error


@dataclass(frozen=True)
class RustXNGSearchOptions:
    """Connection-only settings; the configured endpoint is diagnostic-private."""

    mode: Literal['external']
    base_url: str = field(repr=False)


@dataclass(frozen=True)
class RustXNGSearchConfig:
    """Keyless search configuration independent of key-required providers."""

    api_key: str | None = field(repr=False)
    max_results: int
    timeout_seconds: int
    provider: Literal['rustxng']
    options: RustXNGSearchOptions


def validate_rustxng_endpoint(value: str) -> str:
    """Validates an operator-controlled endpoint, including local/LAN hosts.

    Args:
        value: Explicit base URL with an optional reverse-proxy path prefix.

    Returns:
        Original validated URL.

    Raises:
        GearError: If the URL contains credentials or unsafe syntax.
    """
    try:
        decoded = unquote(value, errors='strict')
        if (not 0 < len(value) <= 2048 or any(c.isspace() or unicodedata.category(c).startswith('C')
                                             for c in decoded) or '\\' in decoded):
            raise ValueError('Invalid URL characters')
        parsed = urlsplit(value)
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or '?' in value or '#' in value or '%' in parsed.netloc
                or (parsed.port is not None and parsed.port < 1)):
            raise ValueError('Invalid endpoint')
        host = parsed.hostname.encode('idna').decode('ascii')
        try:
            ip_address(host)
        except ValueError:
            if any(re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?', label) is None
                   for label in host.rstrip('.').split('.')):
                raise ValueError('Invalid hostname') from None
    except (ValueError, UnicodeError):
        raise gear_error('config_value_invalid',
                         'web_search.rustxng.base_url requires an HTTP(S) URL without credentials, query or fragment.',
                         'config', True, {'origin': 'web_search.rustxng.base_url'}) from None
    return value
