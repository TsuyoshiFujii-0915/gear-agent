from __future__ import annotations

from pathlib import Path
import socket

import pytest

from gear_agent.config import WebFetchConfig, WebSearchConfig, load_config
from gear_agent.errors import GearError
from gear_agent.observation import execute_tool
from gear_agent.run_collector import EvalRunCollector
from gear_agent.tools.registry import ToolRegistry
from gear_agent.tools.web_search import WebSearchTool
from gear_agent.tools.web_fetch import WebFetchTool
from tests.test_exa import ExaAPI, FETCH, SEARCH, URL, api, config_text, configured, fetch_response


def test_exa_uses_common_tools_and_configuration(tmp_path: Path) -> None:
    config, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    assert isinstance(config.web_search, WebSearchConfig)
    assert isinstance(config.web_fetch, WebFetchConfig)
    assert isinstance(tools['web_search'], WebSearchTool)
    assert isinstance(tools['web_fetch'], WebFetchTool)
    assert tools['web_search'].provider_name == 'exa'
    assert tools['web_fetch'].provider_name == 'exa'


@pytest.mark.parametrize('name', ['web_search', 'web_fetch'])
def test_observations_preserve_estimates_without_page_or_query(tmp_path: Path, api: ExaAPI, name: str) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.response = fetch_response() if name == 'web_fetch' else {'results': [], 'costDollars': {'total': 0.001}}
    arguments = {'url': URL} if name == 'web_fetch' else {'query': 'private query'}
    observer = EvalRunCollector()
    registry = ToolRegistry(list(tools.values()))
    result = execute_tool(registry, name, arguments, 'web', observer)
    usage = [{'unit': 'USD', 'value': 0.001, 'kind': 'estimate'}]
    assert result['provider_metadata']['exa']['usage'] == usage
    assert result['provider_metadata']['exa']['estimated_cost_dollars'] == {'total': 0.001}
    assert observer.events[-1]['payload']['provider'] == 'exa'
    assert observer.events[-1]['payload']['usage'] == usage
    assert URL not in repr(observer.events)
    assert 'private query' not in repr(observer.events)
    assert 'Extracted page text' not in repr(observer.events)


def test_missing_estimate_never_becomes_zero_usage(tmp_path: Path, api: ExaAPI) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.response = {'results': [], 'costDollars': {'contents': {'text': 0.001}}}
    result = tools['web_search'].run({'query': 'query'})
    assert result['provider_metadata']['exa']['usage'] == []
    assert result['provider_metadata']['exa']['estimated_cost_dollars'] == {'contents': {'text': 0.001}}


@pytest.mark.parametrize('addresses', [['127.0.0.1'], ['93.184.216.34', '10.0.0.1'], []])
def test_exa_rejects_nonpublic_dns_before_submission(tmp_path: Path, api: ExaAPI,
                                                    monkeypatch: pytest.MonkeyPatch, addresses: list[str]) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))

    def resolve(host: str, port: object, *, type: int) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 0)) for address in addresses]

    monkeypatch.setattr(socket, 'getaddrinfo', resolve)
    with pytest.raises(GearError):
        tools['web_fetch'].run({'url': URL})
    assert not api.calls


def test_invalid_exa_settings_fail_before_secret_resolution(tmp_path: Path) -> None:
    path = tmp_path / 'config.toml'
    path.write_text(config_text(SEARCH.replace('max_results = 5', 'max_results = 101'), FETCH))
    with pytest.raises(GearError) as caught:
        load_config(path, {})
    assert caught.value.error_type == 'config_value_invalid'


def test_common_http_declared_size_limit_applies_to_exa(tmp_path: Path, api: ExaAPI,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))

    def handle(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        api.calls.append(request)
        return httpx.Response(200, headers={'content-length': '1000001'}, content=b'{}')

    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', handle)
    with pytest.raises(GearError):
        tools['web_search'].run({'query': 'query'})
    assert len(api.calls) == 1
