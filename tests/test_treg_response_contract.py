from __future__ import annotations

import json

import httpx
import pytest

from gear_agent.errors import GearError
from gear_agent.tools.treg import TregSearchProvider
from tests.test_treg import SEARCH, catalog, search_response, transport_for


@pytest.mark.parametrize("body", [b'{"results": [], "invalid": NaN}', b"[" * 1500 + b"]" * 1500])
def test_nonstandard_or_excessively_nested_json_is_an_explicit_error(body: bytes) -> None:
    calls: list[httpx.Request] = []
    with pytest.raises(GearError) as raised:
        TregSearchProvider(SEARCH, transport_for(httpx.Response(200, content=body), calls)).search("query")
    assert raised.value.error_type == "treg_json_parse_failed"


def test_unreported_search_title_is_null_without_invention() -> None:
    calls: list[httpx.Request] = []
    data = search_response()
    del data["results"][0]["title"]
    result = TregSearchProvider(SEARCH, transport_for(httpx.Response(200, json=data), calls)).search("query")
    assert result["results"][0]["title"] is None
    json.dumps(result, allow_nan=False)


def test_catalog_changed_to_asynchronous_is_rejected_before_call() -> None:
    calls: list[httpx.Request] = []
    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        data = catalog("tinyfish.web.search", "GET")
        data["endpoint"]["async"] = True
        return httpx.Response(200, json=data)
    with pytest.raises(GearError):
        TregSearchProvider(SEARCH, httpx.MockTransport(handle)).search("query")
    assert len(calls) == 1
