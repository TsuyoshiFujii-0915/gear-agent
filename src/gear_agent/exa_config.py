from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class ExaSearchConfig:
    """Bounded search configuration with a non-displayable credential."""

    api_key: str = field(repr=False)
    timeout_seconds: int
    max_results: int
    max_content_chars: int
    max_response_bytes: int
    exa: ExaSearchOptions
    provider: Literal["exa"] = field(default="exa", init=False)


@dataclass(frozen=True)
class ExaFetchConfig:
    """Bounded single-page retrieval configuration."""

    api_key: str = field(repr=False)
    timeout_seconds: int
    max_content_chars: int
    max_response_bytes: int
    exa: ExaFetchOptions
    provider: Literal["exa"] = field(default="exa", init=False)


def load_exa_search(table: dict[str, object], api_key: str) -> ExaSearchConfig:
    """Validates TOML search options before any network request.

    Args:
        table: Selected web_search table.
        api_key: Credential resolved from api_key_env.

    Returns:
        Validated search configuration.
    """
    origin = "web_search"
    _keys(table, {"provider", "api_key_env", "timeout_seconds", "max_results",
                  "max_content_chars", "max_response_bytes", "exa"}, origin)
    options = _options(table, origin)
    _keys(options, {"search_type", "content", "include_domains", "exclude_domains"}, f"{origin}.exa")
    # These two defaults define the documented initial search mode, not error recovery.
    search_type = options.get("search_type", "auto")
    if search_type != "auto":
        raise _invalid(f"{origin}.exa.search_type", "Only auto is supported.")
    content = options.get("content", "highlights")
    if content not in ("highlights", "text"):
        raise _invalid(f"{origin}.exa.content", "Expected highlights or text.")
    return ExaSearchConfig(
        api_key=api_key,
        timeout_seconds=_positive(table, "timeout_seconds", origin),
        max_results=_integer(table, "max_results", origin, 1, 100),
        max_content_chars=_positive(table, "max_content_chars", origin),
        max_response_bytes=_positive(table, "max_response_bytes", origin),
        exa=ExaSearchOptions("auto", content, _domains(options, "include_domains"),
                             _domains(options, "exclude_domains")),
    )


def load_exa_fetch(table: dict[str, object], api_key: str) -> ExaFetchConfig:
    """Validates TOML retrieval settings and the HTTP/server timeout budget.

    Args:
        table: Selected web_fetch table.
        api_key: Credential resolved from api_key_env.

    Returns:
        Validated single-URL retrieval configuration.
    """
    origin = "web_fetch"
    _keys(table, {"provider", "api_key_env", "timeout_seconds", "max_content_chars",
                  "max_response_bytes", "exa"}, origin)
    options = _options(table, origin)
    _keys(options, {"max_age_hours", "livecrawl_timeout_ms"}, f"{origin}.exa")
    age = None
    if "max_age_hours" in options:
        age = _integer(options, "max_age_hours", f"{origin}.exa", -1, 720)
    timeout = _positive(table, "timeout_seconds", origin)
    live_timeout = _integer(options, "livecrawl_timeout_ms", f"{origin}.exa", 1, 90000)
    if live_timeout >= timeout * 1000:
        raise _invalid(f"{origin}.exa.livecrawl_timeout_ms",
                       "Must be less than timeout_seconds converted to milliseconds.")
    return ExaFetchConfig(
        api_key=api_key, timeout_seconds=timeout,
        max_content_chars=_positive(table, "max_content_chars", origin),
        max_response_bytes=_positive(table, "max_response_bytes", origin),
        exa=ExaFetchOptions(age, live_timeout),
    )


def _options(table: dict[str, object], origin: str) -> dict[str, object]:
    value = table.get("exa")
    if not isinstance(value, dict):
        raise _invalid(f"{origin}.exa", "Expected an options table.")
    return value


def _keys(table: dict[str, object], allowed: set[str], origin: str) -> None:
    for key in table:
        if key not in allowed:
            raise _invalid(f"{origin}.{key}", "Unknown option.")


def _positive(table: dict[str, object], key: str, origin: str) -> int:
    value = table.get(key)
    if type(value) is not int or value < 1:
        raise _invalid(f"{origin}.{key}", "Expected a positive integer.")
    return value


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
