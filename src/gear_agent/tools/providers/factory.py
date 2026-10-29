from __future__ import annotations

from gear_agent.config import WebSearchConfig, WebFetchConfig, TavilySearchOptions, TavilyFetchOptions, ExaSearchConfig, ExaFetchConfig
from gear_agent.exa_config import ExaSearchOptions, ExaFetchOptions
from gear_agent.errors import gear_error
from gear_agent.tools.providers.base import WebSearchProvider, WebFetchProvider
from gear_agent.tools.providers.tavily_search import TavilySearchProvider, HttpxTavilySearchTransport
from gear_agent.tools.providers.tavily_fetch import TavilyFetchProvider, HttpxTavilyFetchTransport
from gear_agent.tools.providers.exa import ExaSearchProvider, ExaFetchProvider, ExaTransport


def build_search_provider(config: WebSearchConfig) -> WebSearchProvider:
    """Builds only the configured, implemented search provider.

    Args:
        config: Independently validated search settings.

    Returns:
        Search adapter with a scoped production HTTP transport.
    """
    if config.provider == 'tavily' and isinstance(config.options, TavilySearchOptions):
        return TavilySearchProvider(config, HttpxTavilySearchTransport(None))
    if isinstance(config, ExaSearchConfig) and config.provider == 'exa' and isinstance(config.options, ExaSearchOptions):
        return ExaSearchProvider(config, ExaTransport())
    raise gear_error('tool_config_invalid', 'Unsupported web_search provider or options.', 'tool_config', True, {})


def build_fetch_provider(config: WebFetchConfig) -> WebFetchProvider:
    """Builds only the configured, implemented extraction provider.

    Args:
        config: Independently validated fetch settings.

    Returns:
        Fetch adapter with a scoped production HTTP transport.
    """
    if config.provider == 'tavily' and isinstance(config.options, TavilyFetchOptions):
        return TavilyFetchProvider(config, HttpxTavilyFetchTransport(None))
    if isinstance(config, ExaFetchConfig) and config.provider == 'exa' and isinstance(config.options, ExaFetchOptions):
        return ExaFetchProvider(config, ExaTransport())
    raise gear_error('tool_config_invalid', 'Unsupported web_fetch provider or options.', 'tool_config', True, {})
