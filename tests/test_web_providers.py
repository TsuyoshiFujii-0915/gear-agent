from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import socket
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.history import build_model_history
from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.config import AppConfig, DEFAULT_CONFIG_TEXT, load_config
from gear_agent.errors import GearError, gear_error, safe_error_payload
from gear_agent.headless import TaskPrompt, describe_run, diagnostic_secrets
from gear_agent.model.client import ModelClient
from gear_agent.model.replay import reasoning_replay_policy
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.observation import execute_tool
from gear_agent.run_collector import EvalRunCollector
from gear_agent.runtime import build_agent_runtime
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.providers.base import ProviderResult, WebUsage
from gear_agent.tools.providers.factory import build_search_provider, build_fetch_provider
from gear_agent.tools.providers.http import WebHttpClient
from gear_agent.tools.providers.tavily_search import TavilySearchProvider, HttpxTavilySearchTransport
from gear_agent.tools.providers.tavily_fetch import TavilyFetchProvider, HttpxTavilyFetchTransport
from gear_agent.tools.registry import ToolRegistry
from gear_agent.tools.configured import build_configured_tools
from gear_agent.tools.runtimes import DockerShellRuntime
from gear_agent.tui import collect_chat_lines
from gear_agent.tools.web_search import WebSearchTool
from gear_agent.tools.web_fetch import WebFetchTool
from tests.test_web_search import _config as search_config, _tavily_response as search_response, FakeTavilySearchTransport
from tests.test_web_fetch import _config as fetch_config, _tavily_response as fetch_response, FakeTavilyFetchTransport, public_addresses
from tests.test_jev_compaction import MODEL, POLICY, JevAPI, event, strategy
from tests.test_agent_loop import SequencedTransport
from tests.test_context_budget import response


NESTED = '''[web_search]
provider = "tavily"
api_key_env = "SEARCH_KEY"
max_results = 5
timeout_seconds = 20
[web_search.tavily]
search_depth = "basic"
include_answer = true
include_raw_content = false

[web_fetch]
provider = "tavily"
api_key_env = "FETCH_KEY"
timeout_seconds = 20
max_content_chars = 20000
[web_fetch.tavily]
extract_depth = "basic"
content_format = "markdown"
include_images = false
include_favicon = true

'''


def config_text(web: str) -> str:
    start = DEFAULT_CONFIG_TEXT.index('[web_search]')
    end = DEFAULT_CONFIG_TEXT.index('[runtime]')
    return (DEFAULT_CONFIG_TEXT[:start] + web + DEFAULT_CONFIG_TEXT[end:]).replace('web_search = false', 'web_search = true').replace('web_fetch = false', 'web_fetch = true')


def load(tmp_path: Path, text: str, environment: dict[str, str]) -> AppConfig:
    path = tmp_path / 'config.toml'
    path.write_text(text)
    return load_config(path, environment)


def test_nested_and_legacy_configs_produce_identical_tavily_requests(tmp_path: Path) -> None:
    nested = load(tmp_path, config_text(NESTED), {'SEARCH_KEY': 'one', 'FETCH_KEY': 'two'})
    legacy = load(tmp_path, DEFAULT_CONFIG_TEXT.replace('web_search = false', 'web_search = true').replace('web_fetch = false', 'web_fetch = true'), {'TAVILY_API_KEY': 'one'})
    assert nested.web_search == legacy.web_search
    assert nested.web_fetch == replace(legacy.web_fetch, api_key='two')
    assert nested.web_search.provider == nested.web_fetch.provider == 'tavily'
    assert isinstance(build_search_provider(nested.web_search), TavilySearchProvider)
    assert isinstance(build_fetch_provider(nested.web_fetch), TavilyFetchProvider)


@pytest.mark.parametrize('old,new', [
    ('provider = "tavily"', 'provider = "exa"'),
    ('provider = "tavily"', 'provider = 1'),
    ('max_results = 5', 'max_results = true'),
    ('max_results = 5', 'max_results = 21'),
    ('timeout_seconds = 20', 'timeout_seconds = true'),
    ('[web_search.tavily]', '[web_search.tavily]\nunknown = 1'),
    ('max_results = 5', 'search_depth = "basic"\nmax_results = 5'),
    ('[web_search.tavily]', '[web_search.options]'),
    ('search_depth = "basic"', 'search_depth = "invalid"'),
    ('include_answer = true', 'include_answer = "true"'),
    ('content_format = "markdown"', 'content_format = "html"'),
    ('max_content_chars = 20000', 'max_content_chars = false'),
    ('max_content_chars = 20000', 'max_content_chars = 0'),
])
def test_invalid_provider_configuration_fails_explicitly(tmp_path: Path, old: str, new: str) -> None:
    with pytest.raises(GearError) as caught:
        load(tmp_path, config_text(NESTED.replace(old, new)), {'SEARCH_KEY': 'one', 'FETCH_KEY': 'two'})
    assert caught.value.origin == 'config'


def test_extract_timeout_outside_upstream_range_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(GearError):
        load(tmp_path, config_text(NESTED.replace('timeout_seconds = 20\nmax_content_chars', 'timeout_seconds = 61\nmax_content_chars')), {'SEARCH_KEY': 'one', 'FETCH_KEY': 'two'})


def test_disabled_tools_do_not_require_tables_keys_or_network(tmp_path: Path) -> None:
    text = config_text(NESTED).replace('web_fetch = true', 'web_fetch = false')
    config = load(tmp_path, text, {'SEARCH_KEY': 'one'})
    assert config.web_fetch is None
    assert config.web_search.api_key == 'one'
    with patch('httpx.Client.send', side_effect=AssertionError('Unexpected HTTP request')):
        config = load(tmp_path, DEFAULT_CONFIG_TEXT, {})
        agent = build_agent_runtime(config, replace(config.runtime, workdir=tmp_path, session_dir=tmp_path / 'sessions'), SilentAgentLoopEventSink())
        tools = build_configured_tools(config.tool, None, None, tmp_path, DockerShellRuntime(tmp_path, 'python:3.11-slim', False))
        assert not any(tool.name.startswith('web_') for tool in tools)


class SearchProvider:
    name = 'search-fixture'
    display_name = 'Fixture Search'
    max_results = 5

    def search(self, query: str) -> ProviderResult:
        return ProviderResult({'query': query, 'results': [{'title': None, 'url': 'https://example.com', 'content': 'snippet'}]}, (WebUsage('USD', 0.03),))


class FetchProvider:
    name = 'fetch-fixture'
    display_name = 'Fixture Fetch'
    max_content_chars = 20000

    def fetch(self, url: str) -> ProviderResult:
        return ProviderResult({'url': url, 'title': None, 'content': 'UNTRUSTED page body', 'requested_url': url, 'final_url': None}, ())


def test_search_and_fetch_use_independent_providers_with_original_model_arguments() -> None:
    registry = ToolRegistry([WebSearchTool(SearchProvider()), WebFetchTool(FetchProvider(), public_addresses)])
    schemas = registry.schemas()
    assert [s['name'] for s in schemas] == ['web_search', 'web_fetch']
    assert [s['parameters']['required'] for s in schemas] == [['query'], ['url']]
    assert 'Tavily' not in json.dumps(schemas)
    assert 'Fixture Search' in schemas[0]['description']
    search = registry.run('web_search', {'query': 'original query'})
    fetch = registry.run('web_fetch', {'url': 'https://example.com'})
    assert search['provider'] == 'search-fixture'
    assert search['query'] == 'original query'
    assert search['results'][0]['title'] is None
    assert search['provider_metadata']['search-fixture']['usage'] == [{'unit': 'USD', 'value': 0.03}]
    assert fetch['provider'] == 'fetch-fixture'
    assert fetch['final_url'] is None
    assert fetch['content'] == 'UNTRUSTED page body'


@pytest.mark.parametrize('url', ['http://localhost', 'https://localhost.', 'http://127.0.0.1', 'http://127.1', 'http://2130706433', 'http://0x7f000001', 'http://10.0.0.1', 'http://169.254.169.254', 'http://[::1]', 'http://[::ffff:127.0.0.1]', 'http://host.local', 'http://user:secret@example.com', 'http://example.com:99999', 'http://example.com\\@localhost', 'http://example.com\n', 'file:///tmp/test', 'https://[broken'])
def test_fetch_rejects_nonpublic_and_credentialed_urls(url: str) -> None:
    tool = WebFetchTool(FetchProvider(), public_addresses)
    with pytest.raises(GearError) as caught:
        tool.run({'url': url})
    assert caught.value.origin == 'web_fetch'
    assert url not in json.dumps(asdict(caught.value))


def private_addresses(host: str) -> list[str]:
    return ['93.184.215.14', '192.168.1.1']


def failed_resolution(host: str) -> list[str]:
    raise socket.gaierror('private resolver diagnostic')


@pytest.mark.parametrize('resolver', [private_addresses, failed_resolution])
def test_fetch_rejects_private_dns_and_resolution_failure(resolver: Any) -> None:
    with pytest.raises(GearError):
        WebFetchTool(FetchProvider(), resolver).run({'url': 'https://example.com'})


@pytest.mark.parametrize('payload', [{'query': 'x', 'provider': 'other'}, {'query': '  '}])
def test_search_rejects_invalid_arguments(payload: dict[str, object]) -> None:
    with pytest.raises(GearError):
        WebSearchTool(SearchProvider()).run(payload)


def test_tavily_missing_query_is_carried_from_input_and_zero_results_are_valid() -> None:
    result = WebSearchTool(TavilySearchProvider(search_config(), FakeTavilySearchTransport({'results': []}))).run({'query': 'original'})
    assert result['query'] == 'original'
    assert result['results'] == []
    assert result['credits'] is None
    assert result['provider_metadata']['tavily']['usage'] == []


@pytest.mark.parametrize('kind', ['search', 'fetch'])
@pytest.mark.parametrize('failure', ['error', 'partial', 'usage', 'result'])
def test_tavily_contract_errors_are_safe(kind: str, failure: str) -> None:
    data = search_response() if kind == 'search' else fetch_response()
    if failure == 'error':
        data['error'] = 'Bearer tvly-secret upstream-secret'
    elif failure == 'partial':
        data['failed_results'] = [{'url': 'https://different.example.com', 'error': 'tvly-secret upstream-secret'}]
    elif failure == 'usage':
        data['usage'] = {'credits': True}
    else:
        data['results'] = [None]
    tool = (WebSearchTool(TavilySearchProvider(search_config(), FakeTavilySearchTransport(data))) if kind == 'search' else
            WebFetchTool(TavilyFetchProvider(fetch_config(), FakeTavilyFetchTransport(data)), public_addresses))
    with pytest.raises(GearError) as caught:
        tool.run({'query': 'q'} if kind == 'search' else {'url': 'https://example.com/docs'})
    assert 'tvly-secret' not in json.dumps(asdict(caught.value))
    assert 'upstream-secret' not in json.dumps(asdict(caught.value))


def test_tavily_fetch_tracks_requested_url_without_inventing_redirect_destination() -> None:
    data = fetch_response()
    result = WebFetchTool(TavilyFetchProvider(fetch_config(), FakeTavilyFetchTransport(data)), public_addresses).run({'url': 'https://example.com/original'})
    assert result['requested_url'] == 'https://example.com/original'
    assert result['url'] == 'https://example.com/docs'
    assert result['final_url'] is None


class Chunks(httpx.SyncByteStream):
    def __iter__(self) -> Any:
        yield b'{' + b'x' * 32
        yield b'x' * 32
        raise AssertionError('Read after response limit')


@pytest.mark.parametrize('kind', ['search', 'fetch'])
@pytest.mark.parametrize('failure', ['status', 'redirect', 'json', 'unicode', 'shape', 'oversize', 'declared_size', 'compressed', 'timeout', 'network'])
def test_http_contract_rejects_unsafe_responses_without_leaking_or_retrying(kind: str, failure: str) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers['Authorization'] == 'Bearer secret'
        assert request.headers['Accept-Encoding'] == 'identity'
        if failure == 'status':
            return httpx.Response(401, text='secret upstream-secret')
        if failure == 'redirect':
            return httpx.Response(307, headers={'Location': 'https://attacker.example/'})
        if failure == 'json':
            return httpx.Response(200, text='secret upstream-secret')
        if failure == 'unicode':
            return httpx.Response(200, content=b'\xff')
        if failure == 'shape':
            return httpx.Response(200, json=[])
        if failure == 'oversize':
            return httpx.Response(200, stream=Chunks())
        if failure == 'declared_size':
            return httpx.Response(200, headers={'Content-Length': '99999'}, stream=Chunks())
        if failure == 'compressed':
            return httpx.Response(200, headers={'Content-Encoding': 'gzip'}, stream=Chunks())
        if failure == 'timeout':
            raise httpx.ReadTimeout('secret upstream-secret')
        raise httpx.ConnectError('secret upstream-secret')

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        http = WebHttpClient(client, 48)
        transport = HttpxTavilySearchTransport(http) if kind == 'search' else HttpxTavilyFetchTransport(http)
        with pytest.raises(GearError) as caught:
            if kind == 'search':
                transport.search('secret', {'query': 'q'}, 20)
            else:
                transport.fetch('secret', {'urls': ['https://example.com']}, 20)
    assert len(calls) == 1
    assert caught.value.origin == 'web_' + kind
    assert 'secret' not in json.dumps(asdict(caught.value))
    assert 'body' not in caught.value.details


def test_real_http_adapter_preserves_tavily_request_and_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == 'https://api.tavily.com/search'
        assert json.loads(request.content)['query'] == 'q'
        return httpx.Response(200, json=search_response())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        tool = WebSearchTool(TavilySearchProvider(search_config(), HttpxTavilySearchTransport(WebHttpClient(client, 8192))))
        result = tool.run({'query': 'q'})
    assert result['results'][0]['score'] == 0.91
    assert result['provider_metadata']['tavily']['usage'] == [{'unit': 'credits', 'value': 1}]


def test_provider_usage_observation_contains_no_query_or_body() -> None:
    collector = EvalRunCollector()
    registry = ToolRegistry([WebSearchTool(SearchProvider()), WebFetchTool(FetchProvider(), public_addresses)])
    execute_tool(registry, 'web_search', {'query': 'secret query'}, 'search', collector)
    execute_tool(registry, 'web_fetch', {'url': 'https://example.com/private-path'}, 'fetch', collector)
    with pytest.raises(GearError):
        execute_tool(registry, 'web_fetch', {'url': 'file:///tmp/secret'}, 'failure', collector)
    finished = [event['payload'] for event in collector.events if event['kind'] == 'tool_finished']
    assert [item['provider'] for item in finished] == ['search-fixture', 'fetch-fixture', 'fetch-fixture']
    assert finished[0]['usage'] == [{'unit': 'USD', 'value': 0.03}]
    assert finished[1]['usage'] == []
    assert finished[2]['successful'] is False
    assert all(item['duration_seconds'] >= 0 for item in finished)
    for value in ('secret query', 'UNTRUSTED', 'private-path', '/tmp/secret'):
        assert value not in json.dumps(collector.events)


def test_selected_provider_keys_are_hidden_in_repr_diagnostics_and_artifacts(tmp_path: Path) -> None:
    config = load(tmp_path, config_text(NESTED), {'SEARCH_KEY': 'private-search-key', 'FETCH_KEY': 'private-fetch-key'})
    runtime = replace(config.runtime, workdir=tmp_path, session_dir=tmp_path / 'sessions')
    agent = build_agent_runtime(config, runtime, SilentAgentLoopEventSink())
    spec = describe_run(agent, 's', TaskPrompt('task', 'inline'))
    assert spec.web_search['provider'] == spec.web_fetch['provider'] == 'tavily'
    diagnostic = safe_error_payload(gear_error('x', 'private-search-key private-fetch-key', 'test', True, {}), diagnostic_secrets(config))
    privacy = ArtifactPrivacy(config, {})
    encoded = repr(config) + json.dumps(asdict(spec)) + json.dumps(diagnostic) + json.dumps(privacy.serialize({'body': 'private-search-key private-fetch-key'}))
    assert 'private-search-key' not in encoded
    assert 'private-fetch-key' not in encoded


@pytest.mark.parametrize('kind', ['search', 'fetch'])
def test_new_results_survive_jsonl_model_jev_summary_and_resume(tmp_path: Path, kind: str) -> None:
    tool = WebSearchTool(SearchProvider()) if kind == 'search' else WebFetchTool(FetchProvider(), public_addresses)
    arguments = {'query': 'q'} if kind == 'search' else {'url': 'https://example.com'}
    result = tool.run(arguments)
    store = JsonlContextStore(tmp_path)
    events = [
        event('user_input', {'text': 'research'}),
        event('model_response', {'output': [{'type': 'function_call', 'call_id': 'web', 'name': tool.name, 'arguments': json.dumps(arguments)}]}),
        event('tool_result', {'name': tool.name, 'call_id': 'web', 'result': result}),
        event('assistant_message', {'text': 'done'}),
        event('user_input', {'text': 'recent task'}),
        event('assistant_message', {'text': 'recent task completed'}),
        event('user_input', {'text': 'new task'}),
    ]
    for item in events:
        store.append('s', item['kind'], item['payload'])
    items = build_model_history(store.load('s'), reasoning_replay_policy(MODEL)).items
    output = next(item for item in items if item.get('type') == 'function_call_output')
    assert json.loads(output['output']) == result
    api = JevAPI({'web': (1, 1)})
    checkpoint = strategy(api, POLICY).compact(store.load('s'), 30, None)
    assert checkpoint.metrics['eligible_interactions'] == 1
    store.append('s', checkpoint.kind, checkpoint.payload)
    resumed = JsonlContextStore(tmp_path)
    assert build_model_history(resumed.load('s'), reasoning_replay_policy(MODEL)).items == items
    transport = SequencedTransport([response('summary')])
    CompactionService(ResponsesModelAdapter(ModelClient(transport), MODEL)).compact('s', resumed, 30)
    assert result['provider'] in transport.payloads[0]['input']
    assert 'summary' in json.dumps(build_model_history(resumed.load('s'), reasoning_replay_policy(MODEL)).items)


def test_tui_displays_missing_search_title_without_inventing_a_title() -> None:
    result = WebSearchTool(SearchProvider()).run({'query': 'q'})
    lines = collect_chat_lines([event('tool_result', {'name': 'web_search', 'call_id': 's', 'result': result})])
    assert 'https://example.com' in lines[0].text
    assert 'snippet' in lines[0].text
    assert 'invalid' not in lines[0].text.lower()


class EmptyFetchProvider(FetchProvider):
    def fetch(self, url: str) -> ProviderResult:
        return ProviderResult({'url': url, 'title': None, 'content': '   '}, ())


class LargeFetchProvider(FetchProvider):
    def fetch(self, url: str) -> ProviderResult:
        return ProviderResult({'url': url, 'title': None, 'content': 'a' * 20001}, ())


@pytest.mark.parametrize('provider', [EmptyFetchProvider(), LargeFetchProvider()])
def test_fetch_contract_is_enforced_for_independent_providers(provider: FetchProvider) -> None:
    with pytest.raises(GearError):
        WebFetchTool(provider, public_addresses).run({'url': 'https://example.com'})
