from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.history import build_model_history
from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.config import AppConfig
from gear_agent.errors import GearError, gear_error, safe_error_payload
from gear_agent.headless import TaskPrompt, describe_run, diagnostic_secrets
from gear_agent.model.client import ModelClient
from gear_agent.model.replay import reasoning_replay_policy
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.observation import execute_tool
from gear_agent.run_collector import EvalRunCollector
from gear_agent.runtime import build_agent_runtime
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.providers.factory import build_search_provider, build_fetch_provider
from gear_agent.tools.providers.firecrawl_search import FirecrawlSearchProvider
from gear_agent.tools.providers.firecrawl_fetch import FirecrawlFetchProvider
from gear_agent.tools.providers.http import WebHttpClient
from gear_agent.tools.registry import ToolRegistry
from gear_agent.tools.web_search import WebSearchTool
from gear_agent.tools.web_fetch import WebFetchTool
from tests.test_web_providers import config_text, load, NESTED
from tests.test_web_fetch import public_addresses
from tests.test_jev_compaction import MODEL, POLICY, JevAPI, event, strategy
from tests.test_agent_loop import SequencedTransport
from tests.test_context_budget import response


SEARCH = '''[web_search]
provider = "firecrawl"
api_key_env = "FIRECRAWL_API_KEY"
max_results = 5
timeout_seconds = 20
'''
FETCH = '''[web_fetch]
provider = "firecrawl"
api_key_env = "FIRECRAWL_API_KEY"
timeout_seconds = 20
max_content_chars = 20000
'''
SECRET = 'fc-private-key'
URL = 'https://example.com/docs'


def settings(tmp_path: Path, web: str) -> AppConfig:
    return load(tmp_path, config_text(web), {'FIRECRAWL_API_KEY': SECRET})


def search_response() -> dict[str, Any]:
    return {'success': True, 'data': {'web': [
        {'url': URL, 'title': 'Docs', 'description': 'A short snippet', 'position': 1},
    ]}, 'creditsUsed': 2, 'id': 'search-request'}


def fetch_response() -> dict[str, Any]:
    return {'success': True, 'data': {'markdown': '# Docs\nPage text.', 'metadata': {
        'title': 'Docs', 'sourceURL': URL, 'url': 'https://example.com/final',
        'statusCode': 200, 'contentType': 'text/html; charset=utf-8',
        'creditsUsed': 1, 'scrapeId': 'scrape-request',
    }}}


def tools(config: AppConfig, http: WebHttpClient) -> tuple[WebSearchTool, WebFetchTool]:
    assert config.web_search is not None and config.web_fetch is not None
    return (WebSearchTool(FirecrawlSearchProvider(config.web_search, http)),
            WebFetchTool(FirecrawlFetchProvider(config.web_fetch, http), public_addresses))


def run_search(tmp_path: Path, body: object) -> dict[str, Any]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        search, _ = tools(settings(tmp_path, SEARCH + FETCH), WebHttpClient(client, 100000))
        return search.run({'query': 'query'})


def run_fetch(tmp_path: Path, body: object) -> dict[str, Any]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        _, fetch = tools(settings(tmp_path, SEARCH + FETCH), WebHttpClient(client, 100000))
        return fetch.run({'url': URL})


def test_cloud_v2_requests_and_results(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=search_response() if request.url.path == '/v2/search' else fetch_response())

    config = settings(tmp_path, SEARCH + FETCH)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        search, fetch = tools(config, WebHttpClient(client, 100000))
        searched = search.run({'query': 'query'})
        fetched = fetch.run({'url': URL})
    assert [str(r.url) for r in requests] == ['https://api.firecrawl.dev/v2/search', 'https://api.firecrawl.dev/v2/scrape']
    assert all(r.method == 'POST' and r.headers['Authorization'] == f'Bearer {SECRET}' for r in requests)
    assert all(r.headers['Accept-Encoding'] == 'identity' for r in requests)
    assert all(r.extensions['timeout']['read'] == 20 for r in requests)
    assert json.loads(requests[0].content) == {
        'query': 'query', 'limit': 5, 'sources': ['web'], 'timeout': 19000, 'highlights': False,
    }
    assert json.loads(requests[1].content) == {
        'url': URL, 'formats': ['markdown'], 'onlyMainContent': True,
        'onlyCleanContent': False, 'skipTlsVerification': False, 'parsers': [],
        'proxy': 'basic', 'maxAge': 172800000, 'storeInCache': False,
        'waitFor': 0, 'timeout': 19000,
    }
    assert searched['query'] == 'query'
    assert searched['answer'] is None
    assert searched['results'][0] == {
        'url': URL, 'title': 'Docs', 'content': 'A short snippet',
        'score': None, 'raw_content': None, 'firecrawl': {'position': 1, 'content_kind': 'snippet'},
    }
    assert searched['request_id'] == 'search-request'
    assert searched['provider_metadata']['firecrawl']['usage'] == [{'unit': 'credits', 'value': 2}]
    assert fetched['requested_url'] == URL
    assert fetched['url'] == fetched['final_url'] == 'https://example.com/final'
    assert fetched['title'] == 'Docs'
    assert fetched['content'] == '# Docs\nPage text.'
    assert fetched['firecrawl']['source_url'] == URL
    assert fetched['firecrawl']['status_code'] == 200
    assert fetched['request_id'] == 'scrape-request'
    assert fetched['provider_metadata']['firecrawl']['usage'] == [{'unit': 'credits', 'value': 1}]
    for tool, key in ((search, 'query'), (fetch, 'url')):
        assert tool.schema()['parameters']['required'] == [key]
        assert 'Firecrawl' in tool.schema()['description']


@pytest.mark.parametrize('firecrawl_kind', ['search', 'fetch'])
def test_independent_factory_selection_and_selected_keys(tmp_path: Path, firecrawl_kind: str) -> None:
    tavily_search, tavily_fetch = NESTED.split('[web_fetch]', 1)
    web = SEARCH + '[web_fetch]' + tavily_fetch if firecrawl_kind == 'search' else tavily_search + FETCH
    config = load(tmp_path, config_text(web), {
        'FIRECRAWL_API_KEY': SECRET, 'SEARCH_KEY': 'search-key', 'FETCH_KEY': 'fetch-key',
    })
    assert build_search_provider(config.web_search).name == ('firecrawl' if firecrawl_kind == 'search' else 'tavily')
    assert build_fetch_provider(config.web_fetch).name == ('firecrawl' if firecrawl_kind == 'fetch' else 'tavily')
    disabled = config_text(SEARCH + FETCH).replace(f'web_{firecrawl_kind} = true', f'web_{firecrawl_kind} = false')
    disabled = disabled.replace(f'[web_{firecrawl_kind}]\nprovider', f'[web_{firecrawl_kind}]\nunused = 1\nprovider')
    with patch('httpx.Client.send', side_effect=AssertionError('Unexpected network')):
        config = load(tmp_path, disabled, {'FIRECRAWL_API_KEY': SECRET})
    assert getattr(config, 'web_' + firecrawl_kind) is None


@pytest.mark.parametrize('old,new', [
    ('max_results = 5', 'max_results = 101'), ('max_results = 5', 'max_results = 0'),
    ('max_results = 5', 'max_results = true'), ('timeout_seconds = 20', 'timeout_seconds = 1'),
    ('timeout_seconds = 20', 'timeout_seconds = 302'), ('timeout_seconds = 20', 'timeout_seconds = 2.5'),
    ('max_results = 5', 'search_depth = "basic"\nmax_results = 5'),
    ('max_content_chars = 20000', 'only_main_content = true\nmax_content_chars = 20000'),
    ('[web_fetch]', '[web_search.firecrawl]\nhighlights = "true"\n[web_fetch]'),
    ('[web_fetch]', '[web_search.firecrawl]\nscrape_options = {}\n[web_fetch]'),
    ('[web_fetch]', '[web_search.tavily]\nsearch_depth = "basic"\n[web_fetch]'),
    ('max_content_chars = 20000', 'max_content_chars = 0'),
])
def test_invalid_config_is_rejected(tmp_path: Path, old: str, new: str) -> None:
    with pytest.raises(GearError) as caught:
        settings(tmp_path, (SEARCH + FETCH).replace(old, new))
    assert caught.value.origin == 'config'
    assert caught.value.error_type != 'config_secret_missing'


@pytest.mark.parametrize('option', [
    'max_age_ms = -1', 'max_age_ms = true', 'wait_for_ms = -1', 'wait_for_ms = 19000',
    'wait_for_ms = 60001', 'proxy = "stealth"', 'proxy = true',
    'only_main_content = "true"', 'store_in_cache = 0', 'skip_tls_verification = true',
    'parsers = ["pdf"]', 'only_clean_content = true',
])
def test_invalid_fetch_options_are_rejected(tmp_path: Path, option: str) -> None:
    with pytest.raises(GearError) as caught:
        settings(tmp_path, SEARCH + FETCH + '[web_fetch.firecrawl]\n' + option)
    assert caught.value.origin == 'config'


@pytest.mark.parametrize('proxy', ['basic', 'enhanced', 'auto'])
def test_explicit_options_and_units(tmp_path: Path, proxy: str) -> None:
    web = (SEARCH.replace('max_results = 5', 'max_results = 100') + '[web_search.firecrawl]\nhighlights = true\n'
           + FETCH + '[web_fetch.firecrawl]\nmax_age_ms = 0\nwait_for_ms = 1250\n'
           + f'proxy = "{proxy}"\nonly_main_content = false\nstore_in_cache = true\n')
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=search_response() if request.url.path.endswith('search') else fetch_response())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        search, fetch = tools(settings(tmp_path, web), WebHttpClient(client, 100000))
        result = search.run({'query': 'query'})
        fetch.run({'url': URL})
    assert payloads[0]['limit'] == 100 and payloads[0]['highlights'] is True
    assert result['results'][0]['firecrawl']['content_kind'] == 'highlights'
    assert payloads[1]['maxAge'] == 0 and payloads[1]['waitFor'] == 1250
    assert payloads[1]['onlyMainContent'] is False and payloads[1]['storeInCache'] is True
    assert payloads[1]['proxy'] == proxy and payloads[1]['timeout'] == 19000


def test_empty_search_and_absent_optional_fields(tmp_path: Path) -> None:
    result = run_search(tmp_path, {'success': True, 'data': {'web': []}})
    assert result['results'] == [] and result['credits'] is None
    assert result['provider_metadata']['firecrawl']['usage'] == []
    result = run_search(tmp_path, {'success': True, 'data': {'web': [{'url': URL, 'description': ''}]}})
    assert result['results'][0]['title'] is None
    assert result['results'][0]['firecrawl']['position'] is None


@pytest.mark.parametrize('body', [
    {}, {'success': False, 'error': SECRET}, {'success': 'true', 'data': {'web': []}},
    {'success': 1, 'data': {'web': []}}, {'success': True, 'data': []},
    {'success': True, 'data': {'web': None}}, {'success': True, 'data': {'web': [1]}},
    {'success': True, 'data': {'web': [{'url': URL, 'description': 123}]}},
    {'success': True, 'data': {'web': [{'description': 'snippet'}]}},
    {'success': True, 'data': {'web': []}, 'error': SECRET},
    {'success': True, 'data': {'web': []}, 'creditsUsed': True},
    {'success': True, 'data': {'web': []}, 'creditsUsed': -1},
    {'success': True, 'data': {'web': []}, 'creditsUsed': 1.5},
    {'success': True, 'data': {'web': []}, 'warning': {}},
])
def test_search_rejects_invalid_rest_envelopes_safely(tmp_path: Path, body: object) -> None:
    with pytest.raises(GearError) as caught:
        run_search(tmp_path, body)
    assert caught.value.origin == 'web_search'
    assert SECRET not in json.dumps(asdict(caught.value))


@pytest.mark.parametrize('body', [
    {'success': False, 'error': SECRET}, {'success': True, 'data': []},
    {'success': 'true', 'data': {'markdown': 'text'}}, {'success': True, 'data': {'summary': 'summary'}},
    {'success': True, 'data': {'markdown': ['text']}}, {'success': True, 'data': {'markdown': 'text', 'metadata': []}},
    {'success': True, 'data': {'markdown': 'text', 'error': SECRET}},
    {'success': True, 'data': {'markdown': 'text'}, 'error': SECRET},
])
def test_fetch_rejects_invalid_rest_envelopes_safely(tmp_path: Path, body: object) -> None:
    with pytest.raises(GearError) as caught:
        run_fetch(tmp_path, body)
    assert caught.value.origin == 'web_fetch'
    assert SECRET not in json.dumps(asdict(caught.value))


@pytest.mark.parametrize('metadata', [
    {'statusCode': 404}, {'statusCode': 500}, {'statusCode': 301}, {'statusCode': '200'},
    {'statusCode': True}, {'error': SECRET}, {'error': False}, {'creditsUsed': -1},
    {'creditsUsed': True}, {'title': ['ok', 1]}, {'sourceURL': 1}, {'url': ''},
    {'contentType': 'application/pdf'}, {'contentType': 'image/png'}, {'contentType': 'application/octet-stream'},
])
def test_fetch_rejects_target_failures_and_unsupported_files(tmp_path: Path, metadata: dict[str, object]) -> None:
    with pytest.raises(GearError) as caught:
        run_fetch(tmp_path, {'success': True, 'data': {'markdown': 'body', 'metadata': metadata}})
    assert caught.value.origin == 'web_fetch'
    assert SECRET not in json.dumps(asdict(caught.value))


@pytest.mark.parametrize('content', ['', ' \n', 'x' * 20001])
def test_fetch_shared_content_limits(tmp_path: Path, content: str) -> None:
    with pytest.raises(GearError) as caught:
        run_fetch(tmp_path, {'success': True, 'data': {'markdown': content}})
    assert caught.value.error_type in {'web_fetch_content_empty', 'web_fetch_content_too_large'}


def test_fetch_rejects_pdf_base64_even_without_content_type(tmp_path: Path) -> None:
    with pytest.raises(GearError):
        run_fetch(tmp_path, {'success': True, 'data': {'markdown': 'JVBERi0xLjQ=', 'rawBase64': 'JVBERi0xLjQ='}})


@pytest.mark.parametrize('metadata,expected_url,final_url', [
    ({}, URL, None), ({'sourceURL': URL}, URL, None),
    ({'sourceURL': URL, 'url': 'https://example.com/final'}, 'https://example.com/final', 'https://example.com/final'),
])
def test_optional_metadata_and_url_provenance(tmp_path: Path, metadata: dict[str, object], expected_url: str, final_url: str | None) -> None:
    result = run_fetch(tmp_path, {'success': True, 'data': {'markdown': 'text', 'metadata': metadata}})
    assert result['url'] == expected_url and result['final_url'] == final_url
    assert result['requested_url'] == URL and result['title'] is None
    assert result['credits'] is None and result['provider_metadata']['firecrawl']['usage'] == []


def test_titles_and_safe_warnings(tmp_path: Path) -> None:
    body = fetch_response()
    body['data']['metadata']['title'] = ['First', 'Second']
    body['data']['metadata']['unrecognized'] = SECRET
    body['data']['warning'] = f'Upstream warning contains {SECRET}'
    result = run_fetch(tmp_path, body)
    assert result['title'] == 'First\nSecond'
    assert result['firecrawl']['warning'] == 'Firecrawl reported a warning.'
    assert SECRET not in json.dumps(result)
    body = search_response()
    body['warning'] = SECRET
    result = run_search(tmp_path, body)
    assert result['firecrawl']['warning'] == 'Firecrawl reported a warning.'
    assert SECRET not in json.dumps(result)


class LargeChunks(httpx.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        yield b'x' * 80
        yield b'x' * 80


@pytest.mark.parametrize('kind', ['search', 'fetch'])
@pytest.mark.parametrize('failure', [401, 402, 403, 429, 500, 503, 307, 'timeout', 'network', 'json', 'size'])
def test_http_failures_are_safe_and_never_retried(tmp_path: Path, kind: str, failure: int | str) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if isinstance(failure, int):
            return httpx.Response(failure, headers={'Location': 'https://attacker.example'}, text=SECRET)
        if failure == 'timeout':
            raise httpx.ReadTimeout(SECRET)
        if failure == 'network':
            raise httpx.ConnectError(SECRET)
        if failure == 'size':
            return httpx.Response(200, stream=LargeChunks())
        return httpx.Response(200, text=SECRET)

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        search, fetch = tools(settings(tmp_path, SEARCH + FETCH), WebHttpClient(client, 128))
        with pytest.raises(GearError) as caught:
            if kind == 'search':
                search.run({'query': 'query'})
            else:
                fetch.run({'url': URL})
    assert len(calls) == 1
    assert caught.value.origin == 'web_' + kind
    assert SECRET not in json.dumps(asdict(caught.value))


@pytest.mark.parametrize('url', ['http://localhost', 'http://127.0.0.1', 'https://user:pass@example.com', 'file:///tmp/private'])
def test_unsafe_urls_never_reach_firecrawl(tmp_path: Path, url: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError('Unsafe URL was dispatched')

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        _, fetch = tools(settings(tmp_path, SEARCH + FETCH), WebHttpClient(client, 100000))
        with pytest.raises(GearError):
            fetch.run({'url': url})


def test_query_length_is_checked_before_dispatch(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError('Oversized query was dispatched')

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        search, _ = tools(settings(tmp_path, SEARCH + FETCH), WebHttpClient(client, 100000))
        with pytest.raises(GearError) as caught:
            search.run({'query': 'q' * 501})
    assert caught.value.error_type == 'tool_argument_invalid'


def test_privacy_factory_and_observations(tmp_path: Path) -> None:
    config = settings(tmp_path, SEARCH + FETCH)
    agent = build_agent_runtime(config, replace(config.runtime, workdir=tmp_path, session_dir=tmp_path / 'sessions'), SilentAgentLoopEventSink())
    spec = describe_run(agent, 's', TaskPrompt('task', 'inline'))
    assert spec.web_search['provider'] == spec.web_fetch['provider'] == 'firecrawl'
    privacy = ArtifactPrivacy(config, {})
    diagnostic = safe_error_payload(gear_error('x', SECRET, 'test', True, {}), diagnostic_secrets(config))
    assert SECRET not in repr(config) + json.dumps(asdict(spec)) + json.dumps(diagnostic) + json.dumps(privacy.serialize({'body': SECRET}))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fetch_response())

    collector = EvalRunCollector()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        registry = ToolRegistry(list(tools(config, WebHttpClient(client, 100000))))
        execute_tool(registry, 'web_fetch', {'url': URL}, 'call', collector)
    finished = collector.events[-1]['payload']
    assert finished['provider'] == 'firecrawl' and finished['successful'] is True
    assert finished['usage'] == [{'unit': 'credits', 'value': 1}]
    assert URL not in json.dumps(collector.events) and 'Page text.' not in json.dumps(collector.events)


@pytest.mark.parametrize('kind', ['search', 'fetch'])
def test_firecrawl_results_survive_history_compaction_and_resume(tmp_path: Path, kind: str) -> None:
    result = run_search(tmp_path, search_response()) if kind == 'search' else run_fetch(tmp_path, fetch_response())
    arguments = {'query': 'query'} if kind == 'search' else {'url': URL}
    name = 'web_' + kind
    store = JsonlContextStore(tmp_path / 'sessions')
    events = [
        event('user_input', {'text': 'research'}),
        event('model_response', {'output': [{'type': 'function_call', 'call_id': 'web', 'name': name, 'arguments': json.dumps(arguments)}]}),
        event('tool_result', {'name': name, 'call_id': 'web', 'result': result}),
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
    checkpoint = strategy(JevAPI({'web': (1, 1)}), POLICY).compact(store.load('s'), 30, None)
    assert checkpoint.metrics['eligible_interactions'] == 1
    store.append('s', checkpoint.kind, checkpoint.payload)
    resumed = JsonlContextStore(tmp_path / 'sessions')
    assert build_model_history(resumed.load('s'), reasoning_replay_policy(MODEL)).items == items
    transport = SequencedTransport([response('summary')])
    CompactionService(ResponsesModelAdapter(ModelClient(transport), MODEL)).compact('s', resumed, 30)
    assert 'firecrawl' in transport.payloads[0]['input']
    assert 'summary' in json.dumps(build_model_history(resumed.load('s'), reasoning_replay_policy(MODEL)).items)
