from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Iterator
from urllib.parse import parse_qs
import json
import traceback

import httpx
import pytest

from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.config import AppConfig, DEFAULT_CONFIG_TEXT, MAX_WEB_RESPONSE_BYTES, load_config
from gear_agent.errors import GearError, gear_error, safe_error_payload
from gear_agent.headless import TaskPrompt, describe_run, diagnostic_secrets
from gear_agent.observation import execute_tool
from gear_agent.run_collector import EvalRunCollector
from gear_agent.runtime import build_agent_runtime
from gear_agent.tools.base import Tool
from gear_agent.tools.configured import build_configured_tools
from gear_agent.tools.registry import ToolRegistry
from gear_agent.tools.runtimes import DockerShellRuntime


QUERY = '日本語 & +\n検索'
SECRET = 'rustxng-private-token'


def config_text(base_url: str, credential: str) -> str:
    prefix, rest = DEFAULT_CONFIG_TEXT.split('[web_search]', 1)
    _, suffix = rest.split('[web_fetch]', 1)
    search = ('[web_search]\nprovider = "rustxng"\nmax_results = 5\ntimeout_seconds = 2\n'
              + credential + '\n[web_search.rustxng]\nmode = "external"\nbase_url = '
              + json.dumps(base_url) + '\n\n')
    return prefix.replace('web_search = false', 'web_search = true') + search + '[web_fetch]' + suffix


def configured(root: Path, text: str, environment: dict[str, str]) -> tuple[AppConfig, dict[str, Tool]]:
    path = root / 'config.toml'
    path.write_text(text, encoding='utf-8')
    config = load_config(path, environment)
    tools = build_configured_tools(config.tool, config.web_search, config.web_fetch, root,
                                   DockerShellRuntime(root, 'python:3.11-slim', False))
    return config, {tool.name: tool for tool in tools}


def response(items: list[dict[str, object]], failures: list[object]) -> bytes:
    return json.dumps({'query': QUERY, 'results': items, 'unresponsive_engines': failures}).encode()


def item(index: int) -> dict[str, object]:
    return {'url': f'https://example.com/{index}', 'title': None, 'content': f'snippet {index}',
            'engine': 'duckduckgo', 'engines': ['duckduckgo'], 'raw_content': 'MUST NOT REACH MODEL',
            'score': 42.5}


@dataclass
class RustAPI:
    base_url: str
    body: bytes
    status: int
    headers: dict[str, str]
    calls: list[tuple[str, dict[str, str], bytes]]


@pytest.fixture
def api() -> Iterator[RustAPI]:
    external = RustAPI('', response([item(0)], []), 200, {}, [])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers['Content-Length']))
            external.calls.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
            self.send_response(external.status)
            self.send_header('Content-Type', 'application/json')
            for key, value in external.headers.items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(external.body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    external.base_url = f'http://127.0.0.1:{server.server_port}/proxy/rustxng/'
    thread = Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    thread.start()
    try:
        yield external
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize('credential', ['', 'api_key_env = "RUSTXNG_TOKEN"'])
def test_production_search_form_encoding_auth_and_prefix(tmp_path: Path, api: RustAPI, credential: str) -> None:
    config, tools = configured(tmp_path, config_text(api.base_url, credential), {'RUSTXNG_TOKEN': SECRET})
    tool = tools['web_search']
    schema = tool.schema()['parameters']
    assert schema['required'] == ['query']
    assert set(schema['properties']) == {'query'}
    result = tool.run({'query': QUERY})
    assert len(api.calls) == 1
    path, headers, body = api.calls[0]
    assert path == '/proxy/rustxng/search'
    assert headers['content-type'] == 'application/x-www-form-urlencoded'
    assert parse_qs(body.decode()) == {'q': [QUERY], 'format': ['json'], 'max_results': ['5'],
                                      'include_raw_content': ['0']}
    assert headers.get('authorization') == (f'Bearer {SECRET}' if credential else None)
    assert result['query'] == QUERY
    assert result['provider'] == 'rustxng'
    assert result['results'] == [{'url': 'https://example.com/0', 'title': None, 'content': 'snippet 0'}]
    assert result['provider_metadata']['rustxng']['usage'] == []
    assert 'MUST NOT REACH MODEL' not in repr(result)
    assert 'score' not in repr(result)
    assert config.web_fetch is None
    assert not api.calls[1:]


def test_result_order_count_and_truncation(tmp_path: Path, api: RustAPI) -> None:
    api.body = response([item(i) for i in range(8)], [])
    _, tools = configured(tmp_path, config_text(api.base_url, ''), {})
    result = tools['web_search'].run({'query': QUERY})
    assert [row['url'] for row in result['results']] == [f'https://example.com/{i}' for i in range(5)]
    assert result['provider_metadata']['rustxng']['truncated'] is True


def test_snippets_titles_and_serialized_result_are_bounded(tmp_path: Path, api: RustAPI) -> None:
    rows = [{**item(i), 'content': '日本語' * 4000, 'title': 'title' * 200} for i in range(200)]
    api.body = response(rows, [])
    _, tools = configured(tmp_path, config_text(api.base_url, '').replace('max_results = 5', 'max_results = 200'), {})
    result = tools['web_search'].run({'query': QUERY})
    assert 0 < len(result['results']) < 200
    assert all(len(row['content']) <= 4000 and len(row['title']) <= 500 for row in result['results'])
    assert len(json.dumps(result, ensure_ascii=True).encode()) <= 65536
    assert result['provider_metadata']['rustxng']['truncated'] is True


def test_html_and_terminal_controls_are_inert(tmp_path: Path, api: RustAPI) -> None:
    api.body = response([{**item(0), 'title': '<script>x</script>\x1b[31m',
                          'content': '<b>text</b>\x00\x1b]8;;https://evil.example\x07'}], [])
    _, tools = configured(tmp_path, config_text(api.base_url, ''), {})
    result = tools['web_search'].run({'query': QUERY})
    encoded = repr(result['results'])
    assert '<script>' not in encoded and '<b>' not in encoded
    assert '\\x1b' not in encoded and '\\x00' not in encoded and '\\x07' not in encoded


@pytest.mark.parametrize('rows,failures,degraded', [([], [], False), ([item(0)], [], False),
    ([item(0)], [['bing', 'Timeout'], ['google', 'CAPTCHA'], ['brave', 'Too many requests']], True)])
def test_empty_and_degraded_searches(tmp_path: Path, api: RustAPI, rows: list[dict[str, object]],
                                     failures: list[object], degraded: bool) -> None:
    api.body = response(rows, failures)
    _, tools = configured(tmp_path, config_text(api.base_url, ''), {})
    result = tools['web_search'].run({'query': QUERY})
    metadata = result['provider_metadata']['rustxng']
    assert metadata['degraded'] is degraded
    assert len(metadata['unresponsive_engines']) == len(failures)
    assert len(result['results']) == len(rows)


def test_no_results_with_engine_failures_is_not_no_matches(tmp_path: Path, api: RustAPI) -> None:
    api.body = response([], [['bing', f'error {SECRET} {QUERY} {api.base_url}']])
    _, tools = configured(tmp_path, config_text(api.base_url, ''), {})
    with pytest.raises(GearError) as caught:
        tools['web_search'].run({'query': QUERY})
    assert caught.value.error_type == 'rustxng_search_unavailable'
    assert 'all engines' not in str(caught.value).lower()
    assert SECRET not in repr(caught.value) and api.base_url not in repr(caught.value)


@pytest.mark.parametrize('status', [301, 302, 307, 308, 401, 403, 429, 500, 503])
def test_http_errors_are_safe_without_redirect_or_retry(tmp_path: Path, api: RustAPI, status: int) -> None:
    api.status = status
    api.headers = {'Location': api.base_url + 'leak'}
    api.body = f'{SECRET} {QUERY} {api.base_url}'.encode()
    _, tools = configured(tmp_path, config_text(api.base_url, 'api_key_env = "RUSTXNG_TOKEN"'), {'RUSTXNG_TOKEN': SECRET})
    with pytest.raises(GearError) as caught:
        tools['web_search'].run({'query': QUERY})
    assert caught.value.error_type == 'rustxng_http_status_error'
    assert caught.value.details['status'] == status
    assert len(api.calls) == 1
    assert SECRET not in repr(caught.value) and api.base_url not in repr(caught.value)
    assert QUERY not in str(caught.value)


@pytest.mark.parametrize('body', [b'<html>login</html>', b'{', b'[]', b'null', b'{"results":NaN}',
    b'{"query":"q","query":"q","results":[]}', b'{}', b'{"query":"q","results":{}}',
    b'{"query":"q","results":[],"unresponsive_engines":null}',
    b'{"query":"q","results":[],"unresponsive_engines":["timeout"]}',
    b'{"query":"q","results":[],"unresponsive_engines":[["bing",1]]}',
    response([{'url': 'https://example.com', 'content': []}], []),
    response([{'url': 'https://example.com', 'title': 12, 'content': 'text'}], []),
    response([{**item(0), 'engines': 'bing'}], [])])
def test_invalid_responses_fail_explicitly(tmp_path: Path, api: RustAPI, body: bytes) -> None:
    api.body = body
    _, tools = configured(tmp_path, config_text(api.base_url, ''), {})
    with pytest.raises(GearError):
        tools['web_search'].run({'query': QUERY})


@pytest.mark.parametrize('url', ['javascript:alert(1)', 'data:text/html,test', 'file:///tmp/test',
    'https://name:pass@example.com', 'https://example.com/\npath', 'https://example.com/\x85path',
    'https://example.com\\evil', 'https://example.com:99999/', 'https://example.com/' + 'a' * 2048])
def test_unsafe_urls_are_never_emitted(tmp_path: Path, api: RustAPI, url: str) -> None:
    api.body = response([{**item(0), 'url': url}], [])
    _, tools = configured(tmp_path, config_text(api.base_url, ''), {})
    with pytest.raises(GearError):
        tools['web_search'].run({'query': QUERY})


@pytest.mark.parametrize('declared', [True, False])
def test_response_limit_is_enforced_while_reading(tmp_path: Path, api: RustAPI, declared: bool) -> None:
    api.body = b' ' * (MAX_WEB_RESPONSE_BYTES + 1)
    if declared:
        api.headers = {'Content-Length': str(len(api.body))}
    _, tools = configured(tmp_path, config_text(api.base_url, ''), {})
    with pytest.raises(GearError) as caught:
        tools['web_search'].run({'query': QUERY})
    assert caught.value.error_type == 'rustxng_response_too_large'


@pytest.mark.parametrize('header,value', [('Content-Encoding', 'gzip'), ('Content-Length', 'invalid')])
def test_response_headers_obey_shared_policy(tmp_path: Path, api: RustAPI, header: str, value: str) -> None:
    api.headers = {header: value}
    _, tools = configured(tmp_path, config_text(api.base_url, ''), {})
    with pytest.raises(GearError):
        tools['web_search'].run({'query': QUERY})


@pytest.mark.parametrize('error,category', [(httpx.ConnectError, 'http_request_failed'),
                                            (httpx.ReadTimeout, 'http_timeout')])
def test_transport_failures_hide_exception_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               error: type[httpx.RequestError], category: str) -> None:
    calls: list[httpx.Request] = []

    def handle(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert all(value == 2 for value in request.extensions['timeout'].values())
        raise error(f'{SECRET} {QUERY} http://private.lan:29483')

    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', handle)
    _, tools = configured(tmp_path, config_text('http://private.lan:29483', ''), {})
    with pytest.raises(GearError) as caught:
        tools['web_search'].run({'query': QUERY})
    assert caught.value.error_type == f'rustxng_{category}'
    diagnostic = ''.join(traceback.format_exception(caught.value))
    assert SECRET not in diagnostic and 'private.lan' not in diagnostic
    assert len(calls) == 1


@pytest.mark.parametrize('old,new', [('mode = "external"', ''), ('mode = "external"', 'mode = "managed"'),
    ('max_results = 5', 'max_results = 201'), ('max_results = 5', 'max_results = 0'),
    ('timeout_seconds = 2', 'timeout_seconds = 0'), ('timeout_seconds = 2', 'timeout_seconds = 301'),
    ('mode = "external"', 'mode = "external"\ninclude_raw_content = true'),
    ('mode = "external"', 'mode = "external"\nengines = ["bing"]')])
def test_invalid_config_is_rejected_before_credentials(tmp_path: Path, old: str, new: str) -> None:
    text = config_text('http://localhost:29483', 'api_key_env = "MISSING"').replace(old, new)
    with pytest.raises(GearError) as caught:
        configured(tmp_path, text, {})
    assert caught.value.error_type != 'config_env_missing'


@pytest.mark.parametrize('url', ['ftp://localhost', 'http://user:secret@localhost:29483',
    'http://localhost:29483?api_key=secret', 'http://localhost:29483#secret', 'http://localhost:bad',
    'http://localhost:0', 'http://localhost/\nsecret', 'http://localhost/%0asecret',
    'http://localhost\\secret', 'http://', 'http://local host'])
def test_unsafe_endpoints_fail_without_echoing_values(tmp_path: Path, url: str) -> None:
    with pytest.raises(GearError) as caught:
        configured(tmp_path, config_text(url, ''), {})
    assert 'secret' not in repr(caught.value)
    assert url not in str(caught.value)


@pytest.mark.parametrize('token', [None, '', ' ', '\nsecret', 'secret\rvalue', '日本語'])
def test_explicit_auth_requires_a_valid_token(tmp_path: Path, token: str | None) -> None:
    environment = {} if token is None else {'RUSTXNG_TOKEN': token}
    with pytest.raises(GearError):
        configured(tmp_path, config_text('http://localhost:29483', 'api_key_env = "RUSTXNG_TOKEN"'), environment)


def test_disabled_search_has_no_credentials_dependencies_or_requests(tmp_path: Path, api: RustAPI) -> None:
    text = config_text(api.base_url, 'api_key_env = "MISSING"').replace('web_search = true', 'web_search = false')
    text = text.replace('mode = "external"', 'mode = "managed"')
    config, tools = configured(tmp_path, text, {})
    assert config.web_search is None and 'web_search' not in tools
    assert not api.calls


def test_fetch_remains_independent_and_requires_its_own_key(tmp_path: Path, api: RustAPI) -> None:
    text = config_text(api.base_url, '').replace('web_fetch = false', 'web_fetch = true')
    with pytest.raises(GearError):
        configured(tmp_path, text, {})
    config, tools = configured(tmp_path, text, {'TAVILY_API_KEY': 'tavily-token'})
    assert config.web_fetch.provider == tools['web_fetch'].provider_name == 'tavily'
    with pytest.raises(GearError):
        tools['web_fetch'].run({'url': 'http://127.0.0.1/page'})
    assert not api.calls


def test_endpoints_and_secrets_are_hidden_in_metadata_and_diagnostics(tmp_path: Path, api: RustAPI) -> None:
    config, tools = configured(tmp_path, config_text(api.base_url, 'api_key_env = "RUSTXNG_TOKEN"'), {'RUSTXNG_TOKEN': SECRET})
    runtime = replace(config.runtime, workdir=tmp_path, session_dir=tmp_path / 'sessions')
    agent = build_agent_runtime(config, runtime, SilentAgentLoopEventSink())
    spec = describe_run(agent, 'session', TaskPrompt('task', 'inline'))
    diagnostic = safe_error_payload(gear_error('test', f'{SECRET} {api.base_url}', 'test', True, {}), diagnostic_secrets(config))
    privacy = ArtifactPrivacy(config, {})
    serialized = repr(config) + json.dumps(asdict(spec)) + json.dumps(diagnostic)
    serialized += json.dumps(privacy.serialize({'body': f'{SECRET} {api.base_url}'}))
    assert SECRET not in serialized and api.base_url not in serialized
    collector = EvalRunCollector()
    execute_tool(ToolRegistry([tools['web_search']]), 'web_search', {'query': QUERY}, 'search', collector)
    assert collector.events[-1]['payload']['usage'] == []
    assert QUERY not in repr(collector.events) and api.base_url not in repr(collector.events)
