from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.loop import AgentLoop
from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.config import DEFAULT_CONFIG_TEXT, load_config
from gear_agent.errors import GearError
from gear_agent.model.client import ModelClient
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.repository import RepositoryContext
from gear_agent.run_collector import EvalRunCollector
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.providers.base import WebUsage
from gear_agent.tools.providers.http import WebHttpClient
from gear_agent.tools.providers.tavily_fetch import HttpxTavilyFetchTransport, TavilyFetchProvider
from gear_agent.tools.providers.tavily_search import HttpxTavilySearchTransport, TavilySearchProvider
from gear_agent.tools.web_fetch import WebFetchTool
from gear_agent.tools.web_search import WebSearchTool
from gear_agent.tui import collect_chat_lines
from tests.test_agent_loop import SequencedTransport
from tests.test_context_budget import response
from tests.test_jev_compaction import MODEL
from tests.test_web_fetch import _config as fetch_config, _tavily_response as fetch_response, public_addresses
from tests.test_web_search import _config as search_config, _tavily_response as search_response, FakeTavilySearchTransport
from tests.test_web_providers import FetchProvider


@pytest.mark.parametrize('url', ['http://224.0.0.1', 'https://[ff02::1]', 'https://[ff0e::1]'])
def test_multicast_is_not_a_public_fetch_target(url: str) -> None:
    with pytest.raises(GearError):
        WebFetchTool(FetchProvider(), public_addresses).run({'url': url})


def multicast_resolution(host: str) -> list[str]:
    return ['224.0.0.1']


def test_dns_multicast_is_not_a_public_fetch_target() -> None:
    with pytest.raises(GearError):
        WebFetchTool(FetchProvider(), multicast_resolution).run({'url': 'https://example.com'})


@pytest.mark.parametrize('value', [True, -1, float('inf'), float('nan')])
def test_usage_rejects_unusable_reported_quantities(value: Any) -> None:
    with pytest.raises(GearError):
        WebUsage('credits', value)


def test_search_never_truncates_results_over_the_configured_limit() -> None:
    data = search_response()
    data['results'] *= 6
    tool = WebSearchTool(TavilySearchProvider(search_config(), FakeTavilySearchTransport(data)))
    with pytest.raises(GearError):
        tool.run({'query': 'q'})


def test_fetch_http_adapter_preserves_request_shape_and_does_not_summarize() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == 'https://api.tavily.com/extract'
        assert json.loads(request.content) == {'urls': ['https://example.com/docs'], 'extract_depth': 'basic', 'format': 'markdown', 'include_images': False, 'include_favicon': True, 'timeout': 20, 'include_usage': True}
        return httpx.Response(200, json=fetch_response())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        tool = WebFetchTool(TavilyFetchProvider(fetch_config(), HttpxTavilyFetchTransport(WebHttpClient(client, 8192))), public_addresses)
        result = tool.run({'url': 'https://example.com/docs'})
    assert result['content'] == fetch_response()['results'][0]['raw_content']


@pytest.mark.parametrize('kind', ['search', 'fetch'])
def test_safe_http_errors_reach_model_tui_history_and_run_artifacts(tmp_path: Path, kind: str) -> None:
    config_path = tmp_path / 'config.toml'
    config_path.write_text(DEFAULT_CONFIG_TEXT)
    config = load_config(config_path, {})
    collector = EvalRunCollector()
    collector.start()
    store = JsonlContextStore(tmp_path / 'sessions')
    arguments = {'query': 'safe query'} if kind == 'search' else {'url': 'https://example.com/docs'}
    model_transport = SequencedTransport([{'output': [{'type': 'function_call', 'call_id': 'web', 'name': 'web_' + kind, 'arguments': json.dumps(arguments)}]}, response('The request failed safely.')])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text='tvly-secret upstream-secret Authorization: Bearer other-secret')

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        http = WebHttpClient(client, 8192)
        tool = (WebSearchTool(TavilySearchProvider(search_config(), HttpxTavilySearchTransport(http))) if kind == 'search' else
                WebFetchTool(TavilyFetchProvider(fetch_config(), HttpxTavilyFetchTransport(http)), public_addresses))
        loop = AgentLoop(ResponsesModelAdapter(ModelClient(model_transport), MODEL), [tool], store, SilentAgentLoopEventSink(), RepositoryContext(tmp_path), observer=collector)
        assert loop.run_turn('s', 'Look up information.', 3, 20).final_text == 'The request failed safely.'
    events = store.load('s')
    error = next(item['payload']['result']['error'] for item in events if item['kind'] == 'tool_result')
    assert 'HTTP 401' in error['message']
    assert 'HTTP 401' in json.dumps(model_transport.payloads[1]['input'])
    metrics = collector.metrics('completed', None, True, events, ArtifactPrivacy(config, {}))
    assert metrics['tools']['calls'][0]['provider'] == 'tavily'
    assert metrics['tools']['calls'][0]['successful'] is False
    surfaces = json.dumps(events) + json.dumps(model_transport.payloads) + json.dumps(metrics) + json.dumps(collector.events) + json.dumps([asdict(item) for item in collect_chat_lines(events)])
    for secret in ('tvly-secret', 'upstream-secret', 'other-secret'):
        assert secret not in surfaces
