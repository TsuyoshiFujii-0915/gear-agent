from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest

from gear_agent.errors import GearError
from gear_agent.tools.deepseek import (
    DeepSeekFetchProvider,
    DeepSeekSearchProvider,
    DeepSeekTransport,
)


class API:
    def __init__(self, response: httpx.Response | Exception) -> None:
        self.response = response
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def transport(self, limit: int) -> DeepSeekTransport:
        return DeepSeekTransport(httpx.MockTransport(self.handle), limit)


def search(api: API, count: int, need_content: bool) -> DeepSeekSearchProvider:
    return DeepSeekSearchProvider("private-key", 20, count, need_content, api.transport(4096))


def fetch(api: API, limit: int) -> DeepSeekFetchProvider:
    return DeepSeekFetchProvider("private-key", 20, limit, api.transport(4096))


def result() -> dict[str, object]:
    return {
        "title": "Example", "url": "https://example.com/", "snippet": "Short excerpt",
        "summary": "Generated summary", "content": "# Full page", "rerank_score": 2.4,
        "site_name": "Example", "site_icon": None, "date": None, "authority": 3,
    }


def test_search_posts_directly_and_preserves_distinct_content() -> None:
    api = API(httpx.Response(200, json={"results": [result()]}))
    query = '日本語 😀 site:example.com -site:other.com "exact phrase" lang:ja after:2025 before:2027'
    output = search(api, 50, True).search(query)
    request, = api.requests
    assert str(request.url) == "https://api.deepseek.com/v1/tools/search"
    assert request.method == "POST"
    assert request.headers["Authorization"] == "Bearer private-key"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == {"query": query, "count": 50, "need_content": True}
    assert output["query"] == query
    item, = output["results"]
    assert item["content"] == "Short excerpt"
    assert item["snippet"] == "Short excerpt"
    assert item["summary"] == "Generated summary"
    assert item["raw_content"] == "# Full page"
    assert item["score"] is None
    assert item["metadata"]["deepseek"]["rerank_score"] == 2.4
    assert item["metadata"]["deepseek"]["authority"] == 3
    assert output.get("credits") is None
    assert output.get("usage") is None
    assert output.get("cost") is None
    json.dumps(output, allow_nan=False)


def test_search_without_optional_fields_does_not_invent_body_or_score() -> None:
    api = API(httpx.Response(200, json={"results": [{
        "title": "Example", "url": "https://example.com/", "snippet": "Excerpt",
    }]}))
    output = search(api, 1, False).search("example")
    item, = output["results"]
    assert item["raw_content"] is None
    assert item["summary"] is None
    assert item["score"] is None
    assert item["metadata"]["deepseek"].get("rerank_score") is None
    assert json.loads(api.requests[0].content)["need_content"] is False


def test_empty_search_is_success_and_empty_body_is_not_fetched() -> None:
    api = API(httpx.Response(200, json={"results": []}))
    assert search(api, 1, False).search("unmatched") == {
        "query": "unmatched", "results": [], "provider": "deepseek",
    }
    item = result()
    item["content"] = ""
    api = API(httpx.Response(200, json={"results": [item]}))
    assert search(api, 1, False).search("example")["results"][0]["raw_content"] is None


@pytest.mark.parametrize("query", ["日" * 128, "😀" * 128, "a" * 127 + "😀"])
def test_unicode_query_boundary_is_accepted(query: str) -> None:
    api = API(httpx.Response(200, json={"results": []}))
    search(api, 1, False).search(query)
    assert json.loads(api.requests[0].content)["query"] == query


@pytest.mark.parametrize("query", ["", " \t\n", "日" * 129, "😀" * 129, "a\ud800"])
def test_invalid_query_fails_before_network(query: str) -> None:
    api = API(httpx.Response(200, json={"results": []}))
    with pytest.raises(GearError, match="query"):
        search(api, 1, False).search(query)
    assert not api.requests


@pytest.mark.parametrize("count", [0, 51, True, 1.5])
def test_invalid_count_fails_before_network(count: object) -> None:
    api = API(httpx.Response(200, json={"results": []}))
    with pytest.raises(GearError):
        search(api, count, False).search("example")
    assert not api.requests


@pytest.mark.parametrize("payload", [
    {}, {"results": None}, {"results": {}}, {"results": [None]},
    {"results": [{"title": "Example"}]},
    {"results": [{**result(), "summary": 1}]},
    {"results": [{**result(), "content": []}]},
    {"results": [{**result(), "rerank_score": True}]},
    {"results": [{**result(), "rerank_score": 4}]},
    {"results": [{**result(), "rerank_score": -1}]},
    {"results": [{**result(), "site_name": []}]},
    {"results": [{**result(), "authority": 5}]},
])
def test_invalid_search_response_is_explicit(payload: dict[str, object]) -> None:
    api = API(httpx.Response(200, json=payload))
    with pytest.raises(GearError) as caught:
        search(api, 1, False).search("example")
    assert caught.value.error_type == "deepseek_response_invalid"
    assert caught.value.origin == "web_search"


@pytest.mark.parametrize("title", ["Example", ""])
def test_fetch_preserves_requested_url_and_markdown(title: str) -> None:
    api = API(httpx.Response(200, json={"title": title, "content": "# 日本語 😀"}))
    output = fetch(api, 100).fetch("https://example.com/docs?lang=ja")
    request, = api.requests
    assert str(request.url) == "https://api.deepseek.com/v1/tools/fetch"
    assert json.loads(request.content) == {"url": "https://example.com/docs?lang=ja"}
    assert output == {
        "url": "https://example.com/docs?lang=ja", "title": title,
        "content": "# 日本語 😀", "content_format": "markdown", "provider": "deepseek",
    }
    assert "final_url" not in output


@pytest.mark.parametrize("url", [
    "", "ftp://example.com/", "https:///missing", "https://", "https://user:pass@example.com/",
    "https://user@example.com/", "https://[invalid/", "https://example.com:bad/",
    "https://example.com/" + "x" * 8192, "https://exa mple.com/", "https://example.com/\n",
])
def test_invalid_url_fails_before_network(url: str) -> None:
    api = API(httpx.Response(200, json={"title": "", "content": "page"}))
    with pytest.raises(GearError, match="URL"):
        fetch(api, 100).fetch(url)
    assert not api.requests


def test_url_length_boundary_is_accepted() -> None:
    api = API(httpx.Response(200, json={"title": "", "content": "page"}))
    url = "https://example.com/" + "a" * (8192 - len("https://example.com/"))
    assert fetch(api, 4).fetch(url)["url"] == url


@pytest.mark.parametrize("payload,error", [
    ({"title": "", "content": ""}, "deepseek_content_empty"),
    ({"title": "", "content": " \n"}, "deepseek_content_empty"),
    ({"title": "", "content": "abcdef"}, "deepseek_content_too_large"),
    ({"title": "", "content": None}, "deepseek_response_invalid"),
    ({"content": "page"}, "deepseek_response_invalid"),
    ({"title": 1, "content": "page"}, "deepseek_response_invalid"),
])
def test_fetch_content_contract(payload: dict[str, object], error: str) -> None:
    api = API(httpx.Response(200, json=payload))
    with pytest.raises(GearError) as caught:
        fetch(api, 5).fetch("https://example.com/")
    assert caught.value.error_type == error
    assert caught.value.origin == "web_fetch"


@pytest.mark.parametrize("status,category", [
    (400, "invalid_request"), (401, "authentication"), (402, "insufficient_balance"),
    (403, "security_policy"), (404, "content_unavailable"), (408, "timeout"),
    (413, "request_too_large"), (429, "rate_limit"), (500, "server"), (504, "timeout"),
    (307, "redirect"),
])
@pytest.mark.parametrize("operation", ["search", "fetch"])
def test_http_errors_are_safe_and_never_retried(status: int, category: str, operation: str) -> None:
    api = API(httpx.Response(status, text="private-key raw secret body", headers={
        "x-request-id": "req-123", "location": "https://other.example/",
    }))
    with pytest.raises(GearError) as caught:
        if operation == "search":
            search(api, 1, False).search("example")
        else:
            fetch(api, 100).fetch("https://example.com/")
    error = caught.value
    assert error.details == {"provider": "deepseek", "status": status,
                             "category": category, "request_id": "req-123"}
    assert "private-key" not in repr(error)
    assert "raw secret body" not in repr(error)
    if status == 401:
        assert "key" in error.message.lower()
    if status == 402:
        assert "balance" in error.message.lower()
    assert len(api.requests) == 1


@pytest.mark.parametrize("body", [b"not-json private-key", b"[]", b"null", b'\xff',
                                  b'{"results": [], "error": {"message": "private-key"}}',
                                  b'{"results": [], "value": NaN}'])
def test_invalid_json_and_embedded_errors_do_not_leak(body: bytes) -> None:
    api = API(httpx.Response(200, content=body))
    with pytest.raises(GearError) as caught:
        search(api, 1, False).search("example")
    assert "private-key" not in repr(caught.value)


@pytest.mark.parametrize("exception,error_type", [
    (httpx.ReadTimeout("private-key"), "deepseek_http_timeout"),
    (httpx.ConnectError("private-key"), "deepseek_http_request_failed"),
])
def test_transport_exceptions_are_safe(exception: Exception, error_type: str) -> None:
    api = API(exception)
    with pytest.raises(GearError) as caught:
        fetch(api, 100).fetch("https://example.com/")
    assert caught.value.error_type == error_type
    assert "private-key" not in repr(caught.value)
    assert len(api.requests) == 1


class LargeBody(httpx.SyncByteStream):
    def __init__(self) -> None:
        self.chunks_read = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for _ in range(100):
            self.chunks_read += 1
            yield b"x" * 1024

    def close(self) -> None:
        self.closed = True


def test_response_byte_limit_stops_reading_and_closes_stream() -> None:
    stream = LargeBody()
    api = API(httpx.Response(200, stream=stream))
    with pytest.raises(GearError) as caught:
        search(api, 1, False).search("example")
    assert caught.value.error_type == "deepseek_response_too_large"
    assert stream.chunks_read < 100
    assert stream.closed


def test_request_id_cannot_echo_credentials() -> None:
    api = API(httpx.Response(401, headers={"x-request-id": "private-key"}))
    with pytest.raises(GearError) as caught:
        search(api, 1, False).search("example")
    assert "private-key" not in repr(caught.value)
