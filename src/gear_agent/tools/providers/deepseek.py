from __future__ import annotations

from gear_agent.config import DeepSeekFetchOptions, DeepSeekSearchOptions, WebFetchConfig, WebSearchConfig
from gear_agent.errors import gear_error
from gear_agent.tools.deepseek import DeepSeekFetchProvider, DeepSeekSearchProvider, DeepSeekTransport
from gear_agent.tools.providers.base import ProviderResult
from gear_agent.tools.web_fetch import WebFetchTool
from gear_agent.tools.web_search import WebSearchTool


class DeepSeekSearchAdapter:
    """Adapts standalone DeepSeek search to the common web result contract."""

    name = "deepseek"
    display_name = "DeepSeek Search"

    def __init__(self, config: WebSearchConfig, transport: DeepSeekTransport) -> None:
        if config.provider != self.name or not isinstance(config.options, DeepSeekSearchOptions):
            raise gear_error("tool_config_invalid", "DeepSeek search requires DeepSeek options.",
                             "tool_config", True, {})
        self._config = config
        self._api = DeepSeekSearchProvider(config.api_key, config.timeout_seconds,
                                          config.max_results, config.options.need_content, transport)

    @property
    def max_results(self) -> int:
        return self._config.max_results

    def search(self, query: str) -> ProviderResult:
        """Returns normalized search data without inventing reported usage."""
        return ProviderResult(self._api.search(query), ())


class DeepSeekFetchAdapter:
    """Adapts standalone DeepSeek Markdown extraction to the common contract."""

    name = "deepseek"
    display_name = "DeepSeek Fetch"

    def __init__(self, config: WebFetchConfig, transport: DeepSeekTransport) -> None:
        if (config.provider != self.name or not isinstance(config.options, DeepSeekFetchOptions)
                or config.options.content_format != "markdown"):
            raise gear_error("tool_config_invalid", "DeepSeek fetch requires Markdown options.",
                             "tool_config", True, {})
        self._config = config
        self._api = DeepSeekFetchProvider(config.api_key, config.timeout_seconds,
                                         config.max_content_chars, transport)

    @property
    def max_content_chars(self) -> int:
        return self._config.max_content_chars

    def fetch(self, url: str) -> ProviderResult:
        """Returns requested URL provenance and no unreported usage."""
        data = self._api.fetch(url)
        data["requested_url"] = url
        data["final_url"] = None
        return ProviderResult(data, ())


class DeepSeekSearchTool(WebSearchTool):
    """Common search execution with DeepSeek's explicit argument constraints."""

    def schema(self) -> dict[str, object]:
        schema = super().schema()
        schema["parameters"]["properties"]["query"]["description"] = (
            "DeepSeek search query: 1 to 128 Unicode characters, not whitespace-only. "
            "Search operators are preserved; queries are never truncated or split."
        )
        return schema


class DeepSeekFetchTool(WebFetchTool):
    """Common public-URL validation with DeepSeek's Markdown-only schema."""

    def schema(self) -> dict[str, object]:
        schema = super().schema()
        schema["parameters"]["properties"]["url"]["description"] = (
            "Public HTTP or HTTPS URL with a host and no username/password, at most "
            "8192 characters. DeepSeek returns Markdown content."
        )
        return schema
