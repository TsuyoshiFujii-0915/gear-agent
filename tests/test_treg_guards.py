from __future__ import annotations

from typing import Iterator

import httpx
import pytest

from gear_agent.errors import GearError
from gear_agent.tools.treg import TregFetchProvider, TregSearchProvider
from tests.test_treg import FETCH, SEARCH, TOKEN, catalog, fetch_response, search_response, transport_for


class OversizedStream(httpx.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        for _ in range(129):
            yield b"x" * 65536
        raise AssertionError("Must stop reading at the response limit")


@pytest.mark.parametrize("response", [
    httpx.Response(401, text=TOKEN), httpx.Response(200, json={}),
    httpx.Response(200, content=b"invalid"), httpx.ConnectTimeout(TOKEN),
])
def test_catalog_failure_stops_before_authenticated_call(response: httpx.Response | Exception) -> None:
    calls: list[httpx.Request] = []
    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if isinstance(response, Exception):
            raise response
        return response
    with pytest.raises(GearError) as caught:
        TregSearchProvider(SEARCH, httpx.MockTransport(handle)).search("query")
    assert len(calls) == 1
    assert "x-treg-token" not in calls[0].headers
    assert TOKEN not in repr(caught.value)


@pytest.mark.parametrize("headers", [{"Content-Encoding": "gzip"}, {"Content-Length": "99999999"}, {"Content-Length": "invalid"}])
def test_unsafe_transfer_is_rejected_before_reading(headers: dict[str, str]) -> None:
    class NeverRead(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            raise AssertionError("Unsafe response must not be read")
            yield b""
    calls: list[httpx.Request] = []
    with pytest.raises(GearError):
        TregFetchProvider(FETCH, transport_for(httpx.Response(200, headers=headers, stream=NeverRead()), calls)).fetch("https://example.com")
    assert calls[1].headers["accept-encoding"] == "identity"


def test_stream_limit_stops_reading_and_closes_response() -> None:
    calls: list[httpx.Request] = []
    response = httpx.Response(200, stream=OversizedStream())
    with pytest.raises(GearError) as caught:
        TregFetchProvider(FETCH, transport_for(response, calls)).fetch("https://example.com")
    assert caught.value.error_type == "treg_response_too_large"
    assert response.is_closed


def test_huge_cost_header_is_a_safe_protocol_error() -> None:
    calls: list[httpx.Request] = []
    with pytest.raises(GearError):
        TregSearchProvider(SEARCH, transport_for(httpx.Response(200, json=search_response(), headers={"X-Treg-Cost-Micro": "9" * 5000}), calls)).search("query")


@pytest.mark.parametrize("url", ["https://example.com\\@localhost", "https://example.com:99999"])
def test_invalid_url_syntax_does_not_reach_treg(url: str) -> None:
    calls: list[httpx.Request] = []
    with pytest.raises(GearError):
        TregFetchProvider(FETCH, transport_for(httpx.Response(200, json=fetch_response()), calls)).fetch(url)
    assert calls == []


def test_missing_fetch_errors_array_is_not_silently_accepted() -> None:
    calls: list[httpx.Request] = []
    data = fetch_response()
    del data["errors"]
    with pytest.raises(GearError):
        TregFetchProvider(FETCH, transport_for(httpx.Response(200, json=data), calls)).fetch("https://example.com")
