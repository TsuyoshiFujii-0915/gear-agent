from __future__ import annotations

from ipaddress import ip_address
from typing import Any
from urllib.parse import quote, quote_plus, urlsplit
import re

from gear_agent.errors import gear_error


def public_web_url(value: object, origin: str) -> str:
    """Validates public HTTP URL syntax without accessing the target locally.

    Args:
        value: Requested or provider-returned URL.
        origin: Tool name used in safe diagnostics.

    Returns:
        Original URL, unchanged.

    Raises:
        GearError: If the URL has credentials, invalid syntax or a nonpublic host.
    """
    valid = isinstance(value, str) and 0 < len(value) <= 2048
    if valid:
        valid = not any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
    if valid:
        try:
            parsed = urlsplit(value)
            host = parsed.hostname
            valid = (parsed.scheme in ("http", "https") and host is not None
                     and parsed.username is None and parsed.password is None
                     and "\\" not in value and parsed.port != 0)
            if valid:
                valid = _public_host(host)
        except ValueError:
            valid = False
    if not valid:
        raise gear_error("web_url_invalid", "Expected a public HTTP(S) URL without credentials.",
                         origin, True, {})
    return value


def _public_host(host: str) -> bool:
    host = host.rstrip(".").lower()
    try:
        address = ip_address(host)
    except ValueError:
        # Reject local/special-use DNS names and alternate numeric IPv4 forms.
        if "." not in host or host.endswith((".localhost", ".local", ".internal", ".test", ".invalid", ".onion")):
            return False
        if all(re.fullmatch(r"(?:0x[0-9a-f]+|[0-9]+)", part) for part in host.split(".")):
            return False
        try:
            ascii_host = host.encode("idna").decode("ascii")
        except UnicodeError:
            return False
        return len(ascii_host) <= 253 and all(
            re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
            for label in ascii_host.split(".")
        )
    return address.is_global and not address.is_multicast


def redact_web_result(value: Any, api_key: str) -> Any:
    """Removes provider credentials before results reach any downstream consumer.

    Args:
        value: Normalized JSON-compatible result.
        api_key: Resolved credential used for this provider request.

    Returns:
        A copy without the known credential or credential-named metadata values.
    """
    if isinstance(value, str):
        for secret in sorted({api_key, quote(api_key, safe=""), quote_plus(api_key)}, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [redact_web_result(item, api_key) for item in value]
    if isinstance(value, dict):
        return {redact_web_result(key, api_key): (
            "[REDACTED]" if re.search(r"authorization|api.?key|secret|password|token|cookie", key, re.I)
            else redact_web_result(item, api_key)) for key, item in value.items()}
    return value
