from __future__ import annotations

from gear_agent.errors import gear_error
from gear_agent.tools.base import Tool
from gear_agent.tools.providers.base import WebFetchProvider
from gear_agent.tools.providers.tavily_fetch import TavilyFetchTransport
from gear_agent.tools.providers.urls import AddressResolver, validate_public_url


class WebFetchTool(Tool):
    """Model-callable page extraction with shared public URL and content checks."""

    def __init__(self, provider: WebFetchProvider, resolver: AddressResolver) -> None:
        self._provider = provider
        self._resolver = resolver

    @property
    def name(self) -> str:
        return 'web_fetch'

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def schema(self) -> dict[str, object]:
        return {
            'type': 'function', 'name': self.name,
            'description': f'Fetch readable web page content from a URL using {self._provider.display_name}.',
            'parameters': {
                'type': 'object',
                'properties': {'url': {'type': 'string', 'description': f'HTTP or HTTPS URL to fetch with {self._provider.display_name}.'}},
                'required': ['url'], 'additionalProperties': False,
            },
            'strict': True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        url = arguments.get('url')
        if set(arguments) != {'url'} or not isinstance(url, str) or not url:
            raise gear_error('tool_argument_invalid', 'web_fetch arguments require only a nonempty url string.',
                             self.name, True, {'key': 'url'})
        validate_public_url(url, self._resolver)
        result = self._provider.fetch(url)
        content = result.data.get('content')
        if (not isinstance(content, str) or not isinstance(result.data.get('url'), str)
                or (result.data.get('title') is not None and not isinstance(result.data['title'], str))):
            raise gear_error('web_fetch_response_invalid', 'Fetch response requires url/content and a nullable title.',
                             self.name, True, {})
        if not content.strip():
            raise gear_error('web_fetch_content_empty', 'Web provider returned empty content.', self.name, True, {})
        if len(content) > self._provider.max_content_chars:
            raise gear_error('web_fetch_content_too_large', 'Web provider content exceeds configured maximum.',
                             self.name, True, {'content_chars': len(content), 'max_content_chars': self._provider.max_content_chars})
        return result.serialize(self.provider_name)
