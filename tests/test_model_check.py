"""Exercise the diagnostic CLI against local HTTP servers, without a model."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from typing import Any
import json
import socket
import subprocess
import sys

import pytest

from gear_agent.config import DEFAULT_CONFIG_TEXT
from tests.strata_fixtures import completed_response, stream_events, text_item


@dataclass
class Endpoint:
    base: str
    routes: dict[str, tuple[int, str, bytes]] = field(default_factory=dict)
    requests: list[tuple[str, str, str | None, bytes]] = field(default_factory=list)


@pytest.fixture
def endpoint() -> Iterator[Endpoint]:
    class Handler(BaseHTTPRequestHandler):
        def respond(self) -> None:
            body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
            site.requests.append((self.command, self.path, self.headers.get('Authorization'), body))
            status, mime, data = site.routes.get(self.path, (404, 'text/plain', b'secret-provider-error'))
            if self.command == 'HEAD' and self.path.endswith('/v1/responses'):
                status = 405
            self.send_response(status)
            self.send_header('Content-Type', mime)
            if status == 307:
                self.send_header('Location', site.base.replace('127.0.0.1', 'localhost') + '/redirected')
            self.end_headers()
            if self.command == 'HEAD':
                return
            try:
                if data == b'idle':
                    self.wfile.write(b': prefill\n\n')
                    self.wfile.flush()
                    sleep(3)
                elif data == b'heartbeats':
                    for _ in range(100):
                        self.wfile.write(b': prefill\n\n')
                        self.wfile.flush()
                        sleep(.1)
                else:
                    self.wfile.write(data)
            except (ConnectionError, OSError):
                pass

        do_GET = do_POST = do_HEAD = respond

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    site = Endpoint(f'http://127.0.0.1:{server.server_port}/private-tunnel')
    thread = Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
    thread.start()
    try:
        yield site
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def document(value: object) -> tuple[int, str, bytes]:
    return 200, 'application/json', json.dumps(value).encode()


def config(path: Path, url: str, stream: bool) -> Path:
    path.write_text(DEFAULT_CONFIG_TEXT.replace('http://localhost:1234/v1/responses', url)
                    .replace('web_search = false', 'web_search = true')
                    .replace('stream = false', f'stream = {str(stream).lower()}')
                    .replace('model_stream_idle_timeout_seconds = 60', 'model_stream_idle_timeout_seconds = 1')
                    .replace('reasoning_replay = "none"', 'max_output_tokens = 96\nreasoning_effort = "low"')
                    + 'context_window_tokens = 8192\nreserved_tokens = 128\n')
    return path


def check(path: Path, *args: str) -> tuple[int, dict[str, Any]]:
    import os
    result = subprocess.run(
        [sys.executable, '-m', 'gear_agent.cli', '--config', str(path), 'model', 'check', '--json', *args],
        capture_output=True, text=True, timeout=12,
        env={**os.environ, 'GEAR_TEST_KEY': 'secret-model-key', 'NO_PROXY': '*'},
    )
    for secret in ('private-tunnel', 'secret-model-key', 'secret-provider-error', 'private/template',
                   'private/model.gguf', 'secret-output', 'secret-query'):
        assert secret not in result.stdout + result.stderr
    assert result.stderr == ''
    report = json.loads(result.stdout)
    assert report['schema_version'] == 1
    return result.returncode, report


@pytest.mark.parametrize('profile', ['maya', 'strata'])
def test_profile_reads_metadata_without_inference(endpoint: Endpoint, tmp_path: Path, profile: str) -> None:
    fixture = Path(__file__).parent / 'fixtures' / 'model_check' / f'{profile}.json'
    for route, value in json.loads(fixture.read_text()).items():
        endpoint.routes['/private-tunnel/' + route] = document(value)
    path = config(tmp_path / 'gear.toml', endpoint.base + '/v1/responses', False)
    path.write_text(path.read_text().replace('api_key_env = ""', 'api_key_env = "GEAR_TEST_KEY"'))
    code, report = check(path, '--profile', profile)
    assert code == 0
    assert report['settings']['requested_model'] == 'local-model-id'
    assert report['settings']['reserved_tokens'] == 128
    assert report['server']['model_ids'][0] == 'fixture-model'
    assert report['server']['load_state'] == 'loaded'
    assert report['server']['context_window_tokens'] == 4096
    assert set(report['warnings']) >= {'configured_context_exceeds_runtime', 'verify_model_alias'}
    assert report['inference']['status'] == 'not_run'
    assert {r[1] for r in endpoint.requests if r[0] == 'GET'} == {
        '/private-tunnel/health', '/private-tunnel/v1/models', '/private-tunnel/props'}
    assert all(r[0] != 'POST' and r[2] == 'Bearer secret-model-key' for r in endpoint.requests)


@pytest.mark.parametrize(('health', 'models', 'state'), [
    ({'status': 'reloading', 'unloaded': True}, [], 'reloading'),
    ({'status': 'ok', 'unloaded': True}, [{'id': 'listed', 'status': {'value': 'unloaded'}}], 'unloaded'),
    ({}, [], 'no_model'),
])
def test_load_state(endpoint: Endpoint, tmp_path: Path, health: dict[str, Any],
                    models: list[dict[str, Any]], state: str) -> None:
    endpoint.routes['/private-tunnel/health'] = document(health)
    endpoint.routes['/private-tunnel/v1/models'] = document({'data': models})
    code, report = check(config(tmp_path / 'c', endpoint.base + '/v1/responses', False), '--profile', 'maya')
    assert code == 1
    assert report['server']['load_state'] == state
    assert report['server']['context_window_tokens'] is None


@pytest.mark.parametrize(('route', 'expected', 'code'), [
    ((404, 'text/plain', b''), 'unavailable', 0),
    ((401, 'text/plain', b'secret-provider-error'), 'http_error', 1),
    ((403, 'text/plain', b''), 'http_error', 1),
    ((307, 'text/plain', b''), 'http_error', 1),
    ((200, 'text/html', b'secret-provider-error'), 'unexpected_content_type', 1),
    ((200, 'application/json', b'{'), 'invalid_json', 1),
])
def test_metadata_outcomes(endpoint: Endpoint, tmp_path: Path, route: tuple[int, str, bytes],
                           expected: str, code: int) -> None:
    endpoint.routes['/private-tunnel/props'] = route
    actual, report = check(config(tmp_path / 'c', endpoint.base + '/v1/responses', False), '--profile', 'strata')
    assert actual == code
    assert report['metadata']['props']['status'] == expected
    assert report['metadata']['props']['http_status'] == route[0]
    if route[0] in (401, 403):
        assert report['metadata']['props']['authentication'] == 'denied'
    assert all('redirected' not in r[1] for r in endpoint.requests)
    assert report['inference']['status'] == 'not_run'


def test_generic_checks_exact_url_and_never_guesses_metadata(endpoint: Endpoint, tmp_path: Path) -> None:
    path = config(tmp_path / 'c', endpoint.base + '/wrong?token=secret-query', False)
    code, report = check(path)
    assert code == 1
    assert report['endpoint']['http_status'] == 404
    assert report['endpoint']['connection'] == 'reached'
    assert report['metadata'] == {}
    assert [(r[0], r[1], r[2]) for r in endpoint.requests] == [
        ('HEAD', '/private-tunnel/wrong?token=secret-query', None)]


@pytest.mark.parametrize('stream', [False, True])
def test_inference_uses_configured_wire_format_once(endpoint: Endpoint, tmp_path: Path, stream: bool) -> None:
    response = completed_response([text_item('secret-output')])
    route = document(response)
    if stream:
        route = (200, 'text/event-stream', ''.join(
            'data: ' + json.dumps(e) + '\n\n' for e in stream_events(response)).encode())
    endpoint.routes['/private-tunnel/v1/responses'] = route
    path = config(tmp_path / 'c', endpoint.base + '/v1/responses', stream)
    before = path.read_bytes()
    code, report = check(path, '--infer')
    assert code == 0 and path.read_bytes() == before
    inference = report['inference']
    assert inference['status'] == 'succeeded'
    assert inference['terminal_status'] == 'completed'
    assert inference['usage']['total_tokens'] == 160 and inference['elapsed_seconds'] > 0
    assert report['metadata'] == {}
    posts = [r for r in endpoint.requests if r[0] == 'POST']
    assert len(posts) == 1 and posts[0][2] is None
    payload = json.loads(posts[0][3])
    assert payload['stream'] is stream and payload['reasoning']['effort'] == 'low'
    assert 0 < payload['max_output_tokens'] == inference['max_output_tokens'] < 96
    assert set(payload) == {'model', 'instructions', 'input', 'reasoning', 'max_output_tokens', 'stream'}
    assert len(payload['input']) < 100


@pytest.mark.parametrize(('route', 'error'), [
    (document({'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'}}), 'response_incomplete'),
    ((200, 'text/event-stream', b': heartbeat\n\n'), 'response_stream_terminated'),
    ((200, 'text/event-stream', b'idle'), 'http_stream_idle_timeout'),
    ((200, 'text/event-stream', b'heartbeats'), 'deadline_exceeded'),
    ((401, 'text/html', b'secret-provider-error'), 'http_status_error'),
])
def test_inference_failure(endpoint: Endpoint, tmp_path: Path, route: tuple[int, str, bytes], error: str) -> None:
    endpoint.routes['/private-tunnel/v1/responses'] = route
    start = monotonic()
    code, report = check(config(tmp_path / 'c', endpoint.base + '/v1/responses', route[1] == 'text/event-stream'),
                         '--infer', '--deadline-seconds', '2')
    assert code == 1 and monotonic() - start < 6
    assert report['inference']['error'] == error
    if error == 'http_status_error':
        assert report['inference']['authentication'] == 'denied'
    assert len([r for r in endpoint.requests if r[0] == 'POST']) == 1


@pytest.mark.parametrize('failure', ['dns', 'connection_refused', 'tls'])
def test_network_failure(tmp_path: Path, endpoint: Endpoint, failure: str) -> None:
    if failure == 'dns':
        url = 'http://gear-model-check.invalid/v1/responses'
    elif failure == 'tls':
        url = endpoint.base.replace('http:', 'https:') + '/v1/responses'
    else:
        with socket.socket() as unused:
            unused.bind(('127.0.0.1', 0))
            port = unused.getsockname()[1]
        url = f'http://127.0.0.1:{port}/v1/responses'
    code, report = check(config(tmp_path / 'c', url, False))
    assert code == 1
    assert report['endpoint']['status'] == failure


def test_budget_error_is_reported_without_network(endpoint: Endpoint, tmp_path: Path) -> None:
    path = config(tmp_path / 'c', endpoint.base + '/v1/responses', False)
    path.write_text(path.read_text().replace('auto_compaction = false', 'auto_compaction = true')
                    .replace('reserved_tokens = 128', 'reserved_tokens = 4'))
    code, report = check(path, '--infer')
    assert code == 1 and 'reserved_tokens' in json.dumps(report['error'])
    assert endpoint.requests == []


def test_metadata_body_deadline_preserves_received_headers(endpoint: Endpoint, tmp_path: Path) -> None:
    endpoint.routes['/private-tunnel/health'] = (200, 'application/json', b'idle')
    code, report = check(config(tmp_path / 'c', endpoint.base + '/v1/responses', False),
                         '--profile', 'strata', '--deadline-seconds', '1')
    assert code == 1
    assert report['metadata']['health'] == {
        'status': 'deadline_exceeded', 'http_status': 200,
        'connection': 'reached', 'authentication': 'request_allowed',
    }
    assert report['endpoint']['http_status'] == 405
    assert report['inference']['status'] == 'not_run'
    assert [r[1] for r in endpoint.requests] == ['/private-tunnel/v1/responses', '/private-tunnel/health']
