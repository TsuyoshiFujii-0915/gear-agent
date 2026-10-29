from __future__ import annotations

from pathlib import Path

import pytest

from gear_agent.errors import GearError
from tests.test_exa import ExaAPI, FETCH, SEARCH, api, config_text, configured


@pytest.mark.parametrize('body', [b'{"results":[],"results":[]}',
                                  b'{"results":[],"costDollars":{"total":1,"total":0}}'])
def test_exa_keeps_rejecting_ambiguous_json(tmp_path: Path, api: ExaAPI, body: bytes) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.body = body
    with pytest.raises(GearError):
        tools['web_search'].run({'query': 'query'})


@pytest.mark.parametrize('selected', ['search', 'fetch'])
def test_exa_response_budget_cannot_exceed_shared_ceiling(tmp_path: Path, selected: str) -> None:
    search = SEARCH.replace('1000000', '8388609') if selected == 'search' else SEARCH
    fetch = FETCH.replace('1000000', '8388609') if selected == 'fetch' else FETCH
    with pytest.raises(GearError) as caught:
        configured(tmp_path, config_text(search, fetch))
    assert caught.value.error_type == 'config_value_invalid'
