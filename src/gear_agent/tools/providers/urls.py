from __future__ import annotations

from collections.abc import Callable
from ipaddress import IPv4Address, IPv6Address, ip_address
import re
import socket
from urllib.parse import urlsplit

from gear_agent.errors import gear_error


AddressResolver = Callable[[str], list[str]]


def resolve_public_host(host: str) -> list[str]:
    """Resolves addresses without opening a connection to the target page.

    Args:
        host: Validated hostname.

    Returns:
        All addresses returned by the system resolver.
    """
    return [str(item[4][0]) for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)]


def validate_public_url(value: str, resolver: AddressResolver) -> str:
    """Rejects credentialed, malformed and nonpublic URL targets before dispatch.

    Args:
        value: Original model-supplied URL.
        resolver: DNS boundary, replaceable for deterministic network tests.

    Returns:
        Original URL without normalization or inferred redirects.

    Raises:
        GearError: If URL validation or public DNS resolution fails.
    """
    try:
        host = _public_url_host(value)
        try:
            ip_address(host)
        except ValueError:
            addresses = resolver(host)
            if not addresses or any(not _is_public_address(ip_address(item)) for item in addresses):
                raise ValueError('Public addresses required')
    except (ValueError, UnicodeError, OSError):
        raise gear_error('tool_argument_invalid', 'web_fetch.url must be a public HTTP(S) URL without credentials; DNS must resolve to public addresses.',
                         'web_fetch', True, {'origin': 'web_fetch arguments', 'key': 'url'}) from None
    return value


def validate_reported_url(value: object, origin: str) -> str:
    """Validates provider-reported URL syntax without performing DNS lookups.

    Args:
        value: URL reported in a provider result.
        origin: Tool identity for response errors.

    Returns:
        Original URL, without inventing a redirect destination.

    Raises:
        GearError: If the response URL is malformed or explicitly nonpublic.
    """
    try:
        if not isinstance(value, str) or not 0 < len(value) <= 2048:
            raise ValueError('URL string required')
        _public_url_host(value)
    except (ValueError, UnicodeError):
        raise gear_error('web_response_url_invalid', 'Provider returned an invalid public HTTP(S) URL.',
                         origin, True, {}) from None
    return value


def _public_url_host(value: str) -> str:
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value) or '\\' in value:
        raise ValueError('Forbidden URL characters')
    parsed = urlsplit(value)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
        raise ValueError('HTTP URL required')
    if parsed.username is not None or parsed.password is not None:
        raise ValueError('Userinfo forbidden')
    if parsed.port is not None and parsed.port < 1:
        raise ValueError('Invalid port')
    host = parsed.hostname.rstrip('.').encode('idna').decode('ascii').lower()
    if '%' in host:
        raise ValueError('Encoded host forbidden')
    try:
        address = ip_address(host)
    except ValueError:
        labels = host.split('.')
        if (len(labels) < 2 or labels[-1].isdigit()
                or labels[-1] in {'localhost', 'local', 'internal', 'lan', 'home', 'invalid', 'test'}
                or any(re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', label) is None for label in labels)):
            raise ValueError('Public hostname required') from None
    else:
        if not _is_public_address(address):
            raise ValueError('Public IP address required')
    return host


def _is_public_address(address: IPv4Address | IPv6Address) -> bool:
    return address.is_global and not address.is_multicast and not address.is_reserved
