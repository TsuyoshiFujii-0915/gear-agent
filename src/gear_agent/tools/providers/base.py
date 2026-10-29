from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Literal, Protocol

from gear_agent.errors import gear_error


@dataclass(frozen=True)
class WebUsage:
    """A provider-reported quantity; units are never converted or summed."""

    unit: Literal['credits', 'USD', 'tokens']
    value: int | float

    def __post_init__(self) -> None:
        if (self.unit not in {'credits', 'USD', 'tokens'} or isinstance(self.value, bool)
                or not isinstance(self.value, (int, float)) or not math.isfinite(self.value)
                or self.value < 0):
            raise gear_error('web_usage_invalid', 'Provider usage requires a supported unit and finite nonnegative value.',
                             'web_provider', True, {})


@dataclass(frozen=True)
class WebCostEstimate(WebUsage):
    """Provider-reported USD estimate, distinct from billed usage."""

    unit: Literal['USD']
    kind: Literal['estimate'] = field(default='estimate', init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.unit != 'USD':
            raise gear_error('web_usage_invalid', 'Web cost estimates require USD.', 'web_provider', True, {})


@dataclass(frozen=True)
class ProviderResult:
    """Normalized model-visible data and separately typed reported usage."""

    data: dict[str, object]
    usage: tuple[WebUsage, ...]

    def serialize(self, provider: str) -> dict[str, object]:
        """Adds namespaced, unit-labelled metadata without changing result fields.

        Args:
            provider: Configured adapter identity.

        Returns:
            Canonical tool payload suitable for JSONL and model history.
        """
        metadata = self.data.get('provider_metadata', {})
        if not isinstance(metadata, dict) or not isinstance(metadata.get(provider, {}), dict):
            raise gear_error('web_metadata_invalid', 'Provider metadata must contain namespaced objects.',
                             'web_provider', True, {})
        return {**self.data, 'provider': provider,
                'provider_metadata': {**metadata, provider: {
                    **metadata.get(provider, {}), 'usage': [asdict(item) for item in self.usage],
                }}}


class WebSearchProvider(Protocol):
    """Search boundary; content is a snippet, never an inferred full page."""

    @property
    def name(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def max_results(self) -> int: ...

    def search(self, query: str) -> ProviderResult:
        """Returns query/results with nullable titles and vendor-specific scores."""
        ...


class WebFetchProvider(Protocol):
    """Extraction boundary; content must contain page text, not a summary."""

    @property
    def name(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def max_content_chars(self) -> int: ...

    def fetch(self, url: str) -> ProviderResult:
        """Returns url/title/content and only established URL provenance."""
        ...
