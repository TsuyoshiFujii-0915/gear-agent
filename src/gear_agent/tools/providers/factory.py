from __future__ import annotations

import httpx

from gear_agent.config import (
    WebSearchConfig, WebFetchConfig, TavilySearchOptions, TavilyFetchOptions,
    DeepSeekSearchOptions, DeepSeekFetchOptions,
    FirecrawlSearchOptions, FirecrawlFetchOptions,
)
from gear_agent.errors import gear_error
from gear_agent.tools.deepseek import DeepSeekTransport

from gear_agent.treg_config import TregSearchOptions, TregFetchOptions
from gear_agent.tools.providers.treg import TregSearchAdapter, TregFetchAdapter
from gear_agent.tools.providers.base import WebSearchProvider, WebFetchProvider
from gear_agent.tools.providers.deepseek import DeepSeekSearchAdapter, DeepSeekFetchAdapter
from gear_agent.tools.providers.http import MAX_WEB_RESPONSE_BYTES
from gear_agent.tools.providers.tavily_search import TavilySearchProvider, HttpxTavilySearchTransport
from gear_agent.tools.providers.tavily_fetch import TavilyFetchProvider, HttpxTavilyFetchTransport
from gear_agent.tools.providers.firecrawl_search import FirecrawlSearchProvider
from gear_agent.tools.providers.firecrawl_fetch import FirecrawlFetchProvider


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
    if config.provider == 'firecrawl' and isinstance(config.options, FirecrawlSearchOptions):
        return FirecrawlSearchProvider(config, None)
    if config.provider == 'deepseek' and isinstance(config.options, DeepSeekSearchOptions):
        return DeepSeekSearchAdapter(config, DeepSeekTransport(httpx.HTTPTransport(retries=0), MAX_WEB_RESPONSE_BYTES))
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
    if config.provider == 'firecrawl' and isinstance(config.options, FirecrawlFetchOptions):
        return FirecrawlFetchProvider(config, None)
    if config.provider == 'deepseek' and isinstance(config.options, DeepSeekFetchOptions):
        return DeepSeekFetchAdapter(config, DeepSeekTransport(httpx.HTTPTransport(retries=0), MAX_WEB_RESPONSE_BYTES))
    raise gear_error('tool_config_invalid', 'Unsupported web_fetch provider or options.', 'tool_config', True, {})
