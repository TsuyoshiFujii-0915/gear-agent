from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TregSearchOptions:
    """Treg-specific search options; None selects no organization header."""

    org: str | None


@dataclass(frozen=True)
class TregFetchOptions:
    """Treg fetch options with explicit cache and upstream timeout units."""

    org: str | None
    ttl_seconds: int
    per_url_timeout_ms: int


@dataclass(frozen=True)
class TregSearchConfig:
    """Direct TinyFish search settings.

    Attributes:
        api_key: Treg token resolved from an explicitly selected environment key.
        org: Explicit identity-token organization, or None for a per-org token.
        timeout_seconds: HTTP timeout in seconds.
        max_results: Local limit on the ranked response, without pagination.
    """

    api_key: str = field(repr=False)
    org: str | None
    timeout_seconds: int
    max_results: int


@dataclass(frozen=True)
class TregFetchConfig:
    """Direct TinyFish Markdown fetch settings.

    Attributes:
        api_key: Treg token resolved from an explicitly selected environment key.
        org: Explicit identity-token organization, or None for a per-org token.
        timeout_seconds: HTTP timeout in seconds.
        max_content_chars: Maximum accepted page content length.
        ttl_seconds: TinyFish cache lifetime; zero requests a fresh fetch.
        per_url_timeout_ms: TinyFish per-URL timeout in milliseconds.
    """

    api_key: str = field(repr=False)
    org: str | None
    timeout_seconds: int
    max_content_chars: int
    ttl_seconds: int
    per_url_timeout_ms: int
