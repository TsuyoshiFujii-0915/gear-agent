from __future__ import annotations

from gear_agent.config import WebSearchConfig, WebFetchConfig, TavilySearchOptions, TavilyFetchOptions
from gear_agent.errors import gear_error
from gear_agent.treg_config import TregSearchOptions, TregFetchOptions
from gear_agent.tools.providers.treg import TregSearchAdapter, TregFetchAdapter
from gear_agent.tools.providers.base import WebSearchProvider, WebFetchProvider
from gear_agent.tools.providers.tavily_search import TavilySearchProvider, HttpxTavilySearchTransport
from gear_agent.tools.providers.tavily_fetch import TavilyFetchProvider, HttpxTavilyFetchTransport


def build_search_provider(config: WebSearchConfig) -> WebSearchProvider:
    """Builds only the configured, implemented search provider.

    Args:
        config: Independently validated search settings.

    Returns:
        Search adapter with a scoped production HTTP transport.
    """
    if config.provider == 'treg' and isinstance(config.options, TregSearchOptions):
        return TregSearchAdapter(config)
    if config.provider == 'tavily' and isinstance(config.options, TavilySearchOptions):
        return TavilySearchProvider(config, HttpxTavilySearchTransport(None))
    raise gear_error('tool_config_invalid', 'Unsupported web_search provider or options.', 'tool_config', True, {})


def build_fetch_provider(config: WebFetchConfig) -> WebFetchProvider:
    """Builds only the configured, implemented extraction provider.

    Args:
        config: Independently validated fetch settings.

    Returns:
        Fetch adapter with a scoped production HTTP transport.
    """
    if config.provider == 'treg' and isinstance(config.options, TregFetchOptions):
        return TregFetchAdapter(config)
    if config.provider == 'tavily' and isinstance(config.options, TavilyFetchOptions):
        return TavilyFetchProvider(config, HttpxTavilyFetchTransport(None))
    raise gear_error('tool_config_invalid', 'Unsupported web_fetch provider or options.', 'tool_config', True, {})
