from __future__ import annotations

import json
from pathlib import Path

import pytest

from gear_agent.errors import GearError
from tests.test_rustxng import QUERY, SECRET, RustAPI, api, config_text, configured, item, response


@pytest.mark.parametrize('query', ['\ud800', '\udfff', 'あ' * 4000])
def test_invalid_or_oversized_query_is_rejected_before_http(tmp_path: Path, api: RustAPI, query: str) -> None:
    _, tools = configured(tmp_path, config_text(api.base_url, ''), {})
    with pytest.raises(GearError) as caught:
        tools['web_search'].run({'query': query})
    assert caught.value.error_type == 'tool_argument_invalid'
    assert not api.calls


def test_degraded_metadata_is_bounded_and_does_not_echo_upstream_errors(tmp_path: Path, api: RustAPI) -> None:
    api.body = response([{**item(0), 'engines': [f'engine{i}' for i in range(100)]}],
                        [[f'engine{i}', f'{SECRET} {api.base_url} {QUERY}'] for i in range(100)])
    _, tools = configured(tmp_path, config_text(api.base_url, ''), {})
    result = tools['web_search'].run({'query': QUERY})
    metadata = result['provider_metadata']['rustxng']
    assert len(metadata['engines']) == len(metadata['unresponsive_engines']) == 32
    assert metadata['truncated'] is True
    assert metadata['degraded'] is True
    encoded = json.dumps(metadata)
    assert SECRET not in encoded and api.base_url not in encoded
    assert all(engine['reason'] == 'unknown' for engine in metadata['unresponsive_engines'])


@pytest.mark.parametrize('field,value', [('error', 'private server error'),
                                         ('redirect_url', 'https://example.com/redirect')])
def test_embedded_errors_and_redirects_are_not_search_success(tmp_path: Path, api: RustAPI,
                                                             field: str, value: str) -> None:
    raw = json.loads(response([item(0)], []))
    raw[field] = value
    api.body = json.dumps(raw).encode()
    _, tools = configured(tmp_path, config_text(api.base_url, ''), {})
    with pytest.raises(GearError) as caught:
        tools['web_search'].run({'query': QUERY})
    assert value not in repr(caught.value)
