from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from gear_agent.errors import GearError
from tests.test_deepseek_web import API, search


class UnreadBody(httpx.SyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        raise AssertionError("Invalid response headers must fail before reading the body")
        yield b""

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("headers", [
    {"Content-Length": "4097"}, {"Content-Length": "invalid"},
    {"Content-Length": "-1"}, {"Content-Encoding": "gzip"},
])
def test_invalid_response_headers_rejected_before_read(headers: dict[str, str]) -> None:
    body = UnreadBody()
    api = API(httpx.Response(200, headers=headers, stream=body))
    with pytest.raises(GearError):
        search(api, 1, False).search("example")
    assert body.closed


def test_deeply_nested_json_returns_safe_structured_error() -> None:
    api = API(httpx.Response(200, content=b"[" * 1100 + b"]" * 1100))
    with pytest.raises(GearError) as caught:
        search(api, 1, False).search("example")
    assert caught.value.origin == "web_search"


def test_exact_response_byte_limit_is_accepted() -> None:
    payload = b'{"results": []}'
    api = API(httpx.Response(200, content=payload))
    assert api.transport(len(payload)).post("search", "key", {"query": "example"}, 20) == {"results": []}
