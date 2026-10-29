from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import re

from gear_agent.errors import GearError, gear_error


@dataclass(frozen=True)
class ExaSearchOptions:
    """Explicit Exa search semantics, independent of the main model."""

    search_type: Literal["auto"]
    content: Literal["highlights", "text"]
    include_domains: tuple[str, ...]
    exclude_domains: tuple[str, ...]


@dataclass(frozen=True)
class ExaFetchOptions:
    """Cache age in hours and server-side retrieval timeout in milliseconds."""

    max_age_hours: int | None
    livecrawl_timeout_ms: int


def load_exa_search_options(options: dict[str, object]) -> ExaSearchOptions:
    """Validates Exa search options before credentials are resolved.

    Args:
        options: Selected web_search.exa table.

    Returns:
        Validated provider-specific search options.
    """
    origin = "web_search"
    _keys(options, {"search_type", "content", "include_domains", "exclude_domains"}, f"{origin}.exa")
    # These defaults define the documented initial mode, not error recovery.
    search_type = options.get("search_type", "auto")
    if search_type != "auto":
        raise _invalid(f"{origin}.exa.search_type", "Only auto is supported.")
    content = options.get("content", "highlights")
    if content not in ("highlights", "text"):
        raise _invalid(f"{origin}.exa.content", "Expected highlights or text.")
    return ExaSearchOptions("auto", content, _domains(options, "include_domains"),
                            _domains(options, "exclude_domains"))


def load_exa_fetch_options(options: dict[str, object], timeout_seconds: int) -> ExaFetchOptions:
    """Validates Exa freshness and server timeout against the HTTP budget.

    Args:
        options: Selected web_fetch.exa table.
        timeout_seconds: HTTP timeout in seconds.

    Returns:
        Validated provider-specific retrieval options.
    """
    origin = "web_fetch.exa"
    _keys(options, {"max_age_hours", "livecrawl_timeout_ms"}, origin)
    age = None
    if "max_age_hours" in options:
        age = _integer(options, "max_age_hours", origin, -1, 720)
    live_timeout = _integer(options, "livecrawl_timeout_ms", origin, 1, 90000)
    if live_timeout >= timeout_seconds * 1000:
        raise _invalid(f"{origin}.livecrawl_timeout_ms",
                       "Must be less than timeout_seconds converted to milliseconds.")
    return ExaFetchOptions(age, live_timeout)


def _keys(table: dict[str, object], allowed: set[str], origin: str) -> None:
    for key in table:
        if key not in allowed:
            raise _invalid(f"{origin}.{key}", "Unknown option.")


def _integer(table: dict[str, object], key: str, origin: str, minimum: int, maximum: int) -> int:
    value = table.get(key)
    if type(value) is not int or not minimum <= value <= maximum:
        raise _invalid(f"{origin}.{key}", f"Expected an integer in [{minimum}, {maximum}].")
    return value


def _domains(options: dict[str, object], key: str) -> tuple[str, ...]:
    value = options.get(key)
    if not isinstance(value, list) or len(value) > 1200:
        raise _invalid(f"web_search.exa.{key}", "Expected a list of up to 1200 domain filters.")
    if any(not isinstance(item, str) or not item.strip() or any(c.isspace() for c in item)
           for item in value):
        raise _invalid(f"web_search.exa.{key}", "Expected nonempty domain filters without whitespace.")
    for item in value:
        host = item.split("/", 1)[0].removeprefix("*.")
        if (any(char in item for char in (":", "@", "?", "#", "\\"))
                or not host or len(host) > 253
                or any(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label) is None
                       for label in host.split("."))):
            raise _invalid(f"web_search.exa.{key}", "Expected hostnames, optional wildcard subdomains or domain paths.")
    return tuple(value)


def _invalid(origin: str, message: str) -> GearError:
    return gear_error("config_value_invalid", f"{origin}: {message}", "config", True, {"origin": origin})
