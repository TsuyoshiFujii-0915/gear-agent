"""Regression cases for HTTP evidence, model aliases and diagnostic privacy."""

from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from tests.test_model_check import Endpoint, check, config, document, endpoint


def test_heartbeat_deadline_preserves_observed_http_status(endpoint: Endpoint, tmp_path: Path) -> None:
    endpoint.routes['/private-tunnel/v1/responses'] = (201, 'text/event-stream', b'heartbeats')
    code, report = check(config(tmp_path / 'c', endpoint.base + '/v1/responses', True),
                         '--infer', '--deadline-seconds', '2')
    assert code == 1
    assert report['endpoint']['http_status'] == 201
    assert report['endpoint']['connection'] == 'reached'
    assert report['inference']['max_output_tokens'] == 32
    assert report['inference']['error'] == 'deadline_exceeded'


def test_alias_does_not_hide_primary_model(endpoint: Endpoint, tmp_path: Path) -> None:
    endpoint.routes['/private-tunnel/v1/models'] = document({'data': [
        {'id': 'primary', 'status': {'value': 'loaded'}},
        {'id': 'local-model-id', 'alias_of': 'primary', 'status': {'value': 'loaded'}}]})
    code, report = check(config(tmp_path / 'c', endpoint.base + '/v1/responses', False), '--profile', 'strata')
    assert code == 0 and report['server']['reported_model'] == 'primary'
    assert 'verify_model_alias' in report['warnings']


def test_untrusted_config_key_is_never_echoed(endpoint: Endpoint, tmp_path: Path) -> None:
    path = config(tmp_path / 'c', endpoint.base + '/v1/responses', False)
    path.write_text(path.read_text() + '"secret-provider-error" = 1\n')
    code, report = check(path)
    assert code == 1 and report['error']['type'] == 'config_value_invalid'
    assert not endpoint.requests


def test_private_lan_endpoint_through_explicit_local_proxy(endpoint: Endpoint, tmp_path: Path) -> None:
    # HTTP proxy receives the absolute target without requiring a routable LAN interface in CI.
    url = 'http://192.168.10.22:8080/strata/v1/responses'
    path = config(tmp_path / 'c', url, False)
    result = subprocess.run([sys.executable, '-m', 'gear_agent.cli', '--config', str(path), 'model', 'check', '--json'],
                            env={**os.environ, 'HTTP_PROXY': endpoint.base.rsplit('/', 1)[0], 'NO_PROXY': '',
                                 'http_proxy': endpoint.base.rsplit('/', 1)[0], 'no_proxy': ''},
                            capture_output=True, text=True, timeout=8)
    assert result.returncode == 0 and json.loads(result.stdout)['endpoint']['http_status'] == 405
    assert [r[1] for r in endpoint.requests] == [url]


@pytest.mark.parametrize('suffix', ['/gateway', '/v1/responses?token=secret-query'])
def test_profile_skips_uninferable_metadata(endpoint: Endpoint, tmp_path: Path, suffix: str) -> None:
    _, report = check(config(tmp_path / 'c', endpoint.base + suffix, False), '--profile', 'maya')
    assert 'metadata_prefix_unknown' in report['warnings']
    assert [r[0] for r in endpoint.requests] == ['HEAD']
