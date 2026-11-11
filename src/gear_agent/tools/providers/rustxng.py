from __future__ import annotations

import html
import json
import re
import unicodedata

import httpx

from gear_agent.config import MAX_WEB_RESPONSE_BYTES
from gear_agent.errors import GearError, gear_error
from gear_agent.rustxng_config import RustXNGSearchConfig
from gear_agent.tools.providers.base import ProviderResult
from gear_agent.tools.providers.http import WebHttpClient
from gear_agent.tools.providers.urls import validate_reported_url


MAX_SNIPPET_CHARS = 4000
MAX_TITLE_CHARS = 500
MAX_RESULT_BYTES = 65536
MAX_QUERY_BYTES = 16384
MAX_ENGINE_METADATA = 32
_FAILURE_REASONS = {
    'Timeout': 'timeout', 'Access denied': 'access_denied', 'CAPTCHA': 'captcha',
    'Too many requests': 'rate_limited', 'HTTP error': 'http_error',
    'Parsing error': 'parsing_error', 'Unexpected error': 'unexpected_error',
}


class RustXNGSearchProvider:
    """Search-only connection to an operator-managed external RustXNG server."""

    name = 'rustxng'
    display_name = 'RustXNG'

    def __init__(self, config: RustXNGSearchConfig) -> None:
        self._config = config

    @property
    def max_results(self) -> int:
        return self._config.max_results

    def search(self, query: str) -> ProviderResult:
        """Submits a form search and bounds the model-visible snippet response.

        Args:
            query: Model-supplied search text, never an endpoint or fetch request.

        Returns:
            Ordered snippets with explicit degradation/truncation and no usage.

        Raises:
            GearError: On unsafe, malformed, oversized or unavailable responses.
        """
        try:
            query.encode('utf-8')
        except UnicodeEncodeError:
            raise gear_error('tool_argument_invalid', 'RustXNG query must be valid UTF-8 text.',
                             'web_search', True, {'key': 'query'}) from None
        if len(json.dumps(query, ensure_ascii=True)) > MAX_QUERY_BYTES:
            raise gear_error('tool_argument_invalid', 'RustXNG query exceeds the serialized size limit.',
                             'web_search', True, {'max_query_bytes': MAX_QUERY_BYTES})
        headers = {}
        if self._config.api_key is not None:
            headers['Authorization'] = f'Bearer {self._config.api_key}'
        with httpx.Client(follow_redirects=False) as client:
            raw = WebHttpClient(client, MAX_WEB_RESPONSE_BYTES).post_form(
                self._config.options.base_url.rstrip('/') + '/search', headers,
                {'q': query, 'format': 'json', 'max_results': str(self.max_results), 'include_raw_content': '0'},
                self._config.timeout_seconds, 'web_search', 'rustxng',
            )
        return _normalize(raw, self.max_results)


def _normalize(raw: dict[str, object], max_results: int) -> ProviderResult:
    query, rows, failures = raw.get('query'), raw.get('results'), raw.get('unresponsive_engines')
    if not isinstance(query, str) or not isinstance(rows, list) or not isinstance(failures, list):
        raise _invalid('query/results/unresponsive_engines')
    query = _without_controls(query)
    if len(json.dumps(query, ensure_ascii=True)) > MAX_QUERY_BYTES:
        raise _invalid('query')
    if raw.get('error') is not None or raw.get('redirect_url') is not None:
        raise _invalid('error/redirect_url')
    failure_metadata: list[dict[str, object]] = []
    for failure in failures:
        if (not isinstance(failure, list) or len(failure) != 2
                or not isinstance(failure[1], str)):
            raise _invalid('unresponsive_engines')
        engine = _engine_name(failure[0])
        reason = failure[1].removeprefix('Suspended: ')
        # Unknown upstream messages are deliberately classified, never echoed.
        category = _FAILURE_REASONS[reason] if reason in _FAILURE_REASONS else 'unknown'
        if len(failure_metadata) < MAX_ENGINE_METADATA:
            failure_metadata.append({'engine': engine, 'reason': category,
                                     'suspended': failure[1].startswith('Suspended: ')})
    if not rows and failures:
        raise gear_error('rustxng_search_unavailable',
                         'RustXNG returned no results and reported engine failures; search availability is uncertain.',
                         'web_search', True, {'failed_engine_count': len(failures)})
    results: list[dict[str, object]] = []
    engines: list[str] = []
    metadata: dict[str, object] = {
        'degraded': bool(failures), 'unresponsive_engines': failure_metadata,
        'engines': engines, 'truncated': len(failures) > MAX_ENGINE_METADATA or len(rows) > max_results,
    }
    result = ProviderResult({'query': query, 'results': results, 'provider_metadata': {'rustxng': metadata}}, ())
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise _invalid('results[]')
        url = validate_reported_url(row.get('url'), 'web_search')
        if any(unicodedata.category(char).startswith('C') for char in url):
            raise _invalid('results[].url')
        title, content = row.get('title'), row.get('content')
        if ((title is not None and not isinstance(title, str)) or 'content' not in row
                or (content is not None and not isinstance(content, str))):
            raise _invalid('results[].title/content')
        row_engines: list[str] = []
        if 'engine' in row:
            row_engines.append(_engine_name(row['engine']))
        if 'engines' in row:
            if not isinstance(row['engines'], list):
                raise _invalid('results[].engines')
            row_engines.extend(_engine_name(name) for name in row['engines'])
        if index >= max_results:
            continue
        normalized_title = _plain_text(title) if title is not None else None
        normalized_content = _plain_text(content) if content is not None else None
        if ((normalized_title is not None and len(normalized_title) > MAX_TITLE_CHARS)
                or (normalized_content is not None and len(normalized_content) > MAX_SNIPPET_CHARS)):
            metadata['truncated'] = True
        results.append({'url': url,
                        'title': normalized_title[:MAX_TITLE_CHARS] if normalized_title is not None else None,
                        'content': normalized_content[:MAX_SNIPPET_CHARS] if normalized_content is not None else None})
        for engine in row_engines:
            if engine not in engines:
                if len(engines) < MAX_ENGINE_METADATA:
                    engines.append(engine)
                else:
                    metadata['truncated'] = True
    # Measure the actual tool serialization, including the provider and usage envelope.
    size = len(json.dumps(result.serialize('rustxng'), ensure_ascii=True))
    while size > MAX_RESULT_BYTES:
        if not results:
            raise _invalid('serialized_result')
        removed = results.pop()
        size -= len(json.dumps(removed, ensure_ascii=True)) + (2 if results else 0)
        metadata['truncated'] = True
    return result


def _plain_text(value: str) -> str:
    return html.escape(_without_controls(value), quote=False)


def _without_controls(value: str) -> str:
    return ''.join(char for char in value if char in '\n\t'
                   or not unicodedata.category(char).startswith('C'))


def _engine_name(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r'[A-Za-z0-9_. -]{1,64}', value) is None:
        raise _invalid('engine')
    return value


def _invalid(field: str) -> GearError:
    return gear_error('rustxng_response_invalid', 'RustXNG returned an invalid search response.',
                     'web_search', True, {'field': field})
