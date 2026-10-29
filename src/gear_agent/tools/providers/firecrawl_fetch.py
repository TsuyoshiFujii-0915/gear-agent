from __future__ import annotations

from typing import Any

from gear_agent.config import FirecrawlFetchOptions, WebFetchConfig
from gear_agent.errors import gear_error
from gear_agent.tools.providers.base import ProviderResult
from gear_agent.tools.providers.firecrawl_common import (
    invalid_response, optional_integer, optional_string, reject_error,
    reported_usage, response_data, response_url, safe_warning,
)
from gear_agent.tools.providers.http import WebHttpClient, post_web_json


FIRECRAWL_SCRAPE_URL = 'https://api.firecrawl.dev/v2/scrape'
PAGE_CONTENT_TYPES = frozenset({'text/html', 'application/xhtml+xml', 'text/plain', 'text/markdown'})


class FirecrawlFetchProvider:
    """Cloud v2 single-page Markdown extraction with explicit cost/TLS policy.

    Args:
        config: Validated fetch configuration.
        http: External HTTP seam; None creates a scoped production client.
    """

    name = 'firecrawl'
    display_name = 'Firecrawl Scrape'

    def __init__(self, config: WebFetchConfig, http: WebHttpClient | None) -> None:
        if config.provider != self.name or not isinstance(config.options, FirecrawlFetchOptions):
            raise gear_error('tool_config_invalid', 'Firecrawl fetch requires Firecrawl options.', 'tool_config', True, {})
        self._config = config
        self._options = config.options
        self._http = http

    @property
    def max_content_chars(self) -> int:
        return self._config.max_content_chars

    def fetch(self, url: str) -> ProviderResult:
        """Extracts Markdown, rejecting target errors and unsupported binary files.

        Args:
            url: Public URL validated by the model-callable tool.

        Returns:
            Common page payload, explicit URL provenance and reported credits.
        """
        options = self._options
        response = post_web_json(
            self._http, FIRECRAWL_SCRAPE_URL, self._config.api_key,
            {'url': url, 'formats': ['markdown'], 'onlyMainContent': options.only_main_content,
             'onlyCleanContent': False, 'skipTlsVerification': False, 'parsers': [],
             'proxy': options.proxy, 'maxAge': options.max_age_ms, 'storeInCache': options.store_in_cache,
             'waitFor': options.wait_for_ms, 'timeout': (self._config.timeout_seconds - 1) * 1000},
            self._config.timeout_seconds, 'web_fetch', 'firecrawl_scrape',
        )
        return _normalize_response(response, url)


def _normalize_response(response: dict[str, Any], requested_url: str) -> ProviderResult:
    origin = 'web_fetch'
    data = response_data(response, origin)
    metadata = data.get('metadata')
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise invalid_response('data.metadata', origin)
    reject_error(metadata, 'data.metadata.error', origin)
    status = optional_integer(metadata, 'statusCode', 'data.metadata.statusCode', origin)
    if status is not None and not 200 <= status < 300:
        raise gear_error('firecrawl_target_failed', 'Firecrawl reported a non-success target HTTP status.',
                         origin, True, {'status': status})
    content_type = optional_string(metadata, 'contentType', 'data.metadata.contentType', origin)
    if ((content_type is not None and content_type.split(';', 1)[0].strip().lower() not in PAGE_CONTENT_TYPES)
            or data.get('rawBase64') is not None):
        raise gear_error('firecrawl_content_type_unsupported', 'Firecrawl returned an unsupported file instead of page Markdown.',
                         origin, True, {})
    content = data.get('markdown')
    if not isinstance(content, str):
        raise invalid_response('data.markdown', origin)
    source = _optional_url(metadata, 'sourceURL')
    final = _optional_url(metadata, 'url')
    # When upstream omits URL metadata, the common URL explicitly refers to the input.
    url = final if final is not None else source if source is not None else requested_url
    usage = reported_usage(metadata, origin)
    top_warning = safe_warning(response, origin)
    page_warning = safe_warning(data, origin)
    return ProviderResult({
        'requested_url': requested_url, 'url': url, 'final_url': final,
        'title': _title(metadata), 'content': content,
        'credits': usage[0].value if usage else None, 'response_time': None,
        'request_id': optional_string(metadata, 'scrapeId', 'data.metadata.scrapeId', origin),
        'firecrawl': {'source_url': source, 'status_code': status, 'content_type': content_type,
                      'warning': page_warning if page_warning is not None else top_warning},
    }, usage)


def _optional_url(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    return response_url(value, f'data.metadata.{key}', 'web_fetch')


def _title(metadata: dict[str, Any]) -> str | None:
    value = metadata.get('title')
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return '\n'.join(value)
    raise invalid_response('data.metadata.title', 'web_fetch')
