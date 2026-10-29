from __future__ import annotations

from gear_agent.config import FirecrawlSearchOptions, WebSearchConfig
from gear_agent.errors import gear_error
from gear_agent.tools.providers.base import ProviderResult
from gear_agent.tools.providers.firecrawl_common import (
    invalid_response, optional_integer, optional_string, reported_usage,
    response_data, response_url, safe_warning,
)
from gear_agent.tools.providers.http import WebHttpClient, post_web_json


FIRECRAWL_SEARCH_URL = 'https://api.firecrawl.dev/v2/search'


class FirecrawlSearchProvider:
    """Cloud v2 web search without bundled page scraping or tool discovery.

    Args:
        config: Validated search configuration.
        http: External HTTP seam; None creates a scoped production client.
    """

    name = 'firecrawl'
    display_name = 'Firecrawl Search'

    def __init__(self, config: WebSearchConfig, http: WebHttpClient | None) -> None:
        if config.provider != self.name or not isinstance(config.options, FirecrawlSearchOptions):
            raise gear_error('tool_config_invalid', 'Firecrawl search requires Firecrawl options.', 'tool_config', True, {})
        self._config = config
        self._options = config.options
        self._http = http

    @property
    def max_results(self) -> int:
        return self._config.max_results

    def search(self, query: str) -> ProviderResult:
        """Returns snippets or explicitly labelled highlights from data.web.

        Args:
            query: Original user query, at most 500 characters.

        Returns:
            Common search payload and provider-reported credit usage.
        """
        if len(query) > 500:
            raise gear_error('tool_argument_invalid', 'Firecrawl web_search.query must not exceed 500 characters.',
                             'web_search', True, {'key': 'query', 'max_chars': 500})
        response = post_web_json(
            self._http, FIRECRAWL_SEARCH_URL, self._config.api_key,
            {'query': query, 'limit': self.max_results, 'sources': ['web'],
             'timeout': (self._config.timeout_seconds - 1) * 1000, 'highlights': self._options.highlights},
            self._config.timeout_seconds, 'web_search', 'firecrawl_search',
        )
        data = response_data(response, 'web_search')
        raw_results = data.get('web')
        if not isinstance(raw_results, list):
            raise invalid_response('data.web', 'web_search')
        kind = 'highlights' if self._options.highlights else 'snippet'
        results = [_normalize_result(item, index, kind) for index, item in enumerate(raw_results)]
        usage = reported_usage(response, 'web_search')
        return ProviderResult({
            'query': query, 'answer': None, 'results': results, 'response_time': None,
            'credits': usage[0].value if usage else None,
            'request_id': optional_string(response, 'id', 'id', 'web_search'),
            'firecrawl': {'warning': safe_warning(response, 'web_search')},
        }, usage)


def _normalize_result(item: object, index: int, kind: str) -> dict[str, object]:
    field = f'data.web[{index}]'
    origin = 'web_search'
    if not isinstance(item, dict):
        raise invalid_response(field, origin)
    description = item.get('description')
    if not isinstance(description, str):
        raise invalid_response(f'{field}.description', origin)
    return {
        'title': optional_string(item, 'title', f'{field}.title', origin),
        'url': response_url(item.get('url'), f'{field}.url', origin),
        'content': description, 'score': None, 'raw_content': None,
        'firecrawl': {
            'position': optional_integer(item, 'position', f'{field}.position', origin),
            'content_kind': kind,
        },
    }
