from __future__ import annotations

from typing import Any

from gear_agent.config import WebFetchConfig, WebSearchConfig
from gear_agent.errors import gear_error
from gear_agent.treg_config import TregFetchConfig, TregFetchOptions, TregSearchConfig, TregSearchOptions
from gear_agent.tools.providers.base import ProviderResult, WebUsage
from gear_agent.tools.treg import TregFetchProvider, TregSearchProvider


class TregSearchAdapter:
    """Connects direct TinyFish search to the shared Web tool contract."""

    name = "treg"
    display_name = "treg (TinyFish direct)"

    def __init__(self, config: WebSearchConfig) -> None:
        if config.provider != "treg" or not isinstance(config.options, TregSearchOptions):
            raise gear_error("tool_config_invalid", "Treg search requires treg search options.", "web_search", True, {})
        self.max_results = config.max_results
        self._provider = TregSearchProvider(TregSearchConfig(
            config.api_key, config.options.org, config.timeout_seconds, config.max_results,
        ), None)

    def search(self, query: str) -> ProviderResult:
        """Returns snippets and namespaced direct-call metadata.

        Args:
            query: Validated model search query.

        Returns:
            Normalized content with separately typed reported billing.
        """
        data = self._provider.search(query)
        for item in data["results"]:
            item["provider_metadata"] = {"treg": item["provider_metadata"]}
        return _provider_result(data)


class TregFetchAdapter:
    """Connects direct TinyFish extraction to the shared Web tool contract."""

    name = "treg"
    display_name = "treg (TinyFish direct)"

    def __init__(self, config: WebFetchConfig) -> None:
        if config.provider != "treg" or not isinstance(config.options, TregFetchOptions):
            raise gear_error("tool_config_invalid", "Treg fetch requires treg fetch options.", "web_fetch", True, {})
        self.max_content_chars = config.max_content_chars
        self._provider = TregFetchProvider(TregFetchConfig(
            config.api_key, config.options.org, config.timeout_seconds, config.max_content_chars,
            config.options.ttl_seconds, config.options.per_url_timeout_ms,
        ), None)

    def fetch(self, url: str) -> ProviderResult:
        """Returns page text and known URL provenance.

        Args:
            url: Validated model fetch URL.

        Returns:
            Normalized page content and separately typed reported billing.
        """
        data = self._provider.fetch(url)
        data["requested_url"] = url
        data["final_url"] = data["provider_metadata"]["final_url"]
        return _provider_result(data)


def _provider_result(data: dict[str, Any]) -> ProviderResult:
    metadata = data["provider_metadata"]
    charged = metadata["charged_micro"]
    usage = () if charged is None else (WebUsage("micro_USD", charged),)
    return ProviderResult(data | {"provider_metadata": {"treg": metadata}}, usage)
