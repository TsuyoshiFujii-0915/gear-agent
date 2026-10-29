from __future__ import annotations

from gear_agent.errors import gear_error
from gear_agent.tools.base import Tool
from gear_agent.tools.providers.base import WebSearchProvider
from gear_agent.tools.providers.tavily_search import TavilySearchTransport


class WebSearchTool(Tool):
    """Model-callable search independent of provider credentials and wire format."""

    def __init__(self, provider: WebSearchProvider) -> None:
        self._provider = provider

    @property
    def name(self) -> str:
        return 'web_search'

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def schema(self) -> dict[str, object]:
        return {
            'type': 'function', 'name': self.name,
            'description': f'Search the web for current information using {self._provider.display_name}.',
            'parameters': {
                'type': 'object',
                'properties': {'query': {'type': 'string', 'description': f'Search query to send to {self._provider.name.title()}.'}},
                'required': ['query'], 'additionalProperties': False,
            },
            'strict': True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        query = arguments.get('query')
        if set(arguments) != {'query'} or not isinstance(query, str) or not query.strip():
            raise gear_error('tool_argument_invalid', 'web_search arguments require only a nonempty query string.',
                             self.name, True, {'key': 'query'})
        result = self._provider.search(query)
        results = result.data.get('results')
        if (not isinstance(result.data.get('query'), str) or not isinstance(results, list)
                or len(results) > self._provider.max_results):
            raise gear_error('web_search_response_invalid', 'Search response requires query and results within max_results.',
                             self.name, True, {})
        for item in results:
            if (not isinstance(item, dict) or not isinstance(item.get('url'), str)
                    or 'content' not in item
                    or (item['content'] is not None and not isinstance(item['content'], str))
                    or (item.get('title') is not None and not isinstance(item['title'], str))):
                raise gear_error('web_search_response_invalid', 'Search result requires url and nullable content/title.',
                                 self.name, True, {})
        return result.serialize(self.provider_name)
