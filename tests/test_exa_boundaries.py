from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator
import json

import httpx
import pytest

from gear_agent.agent.loop import AgentLoop
from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.errors import GearError
from gear_agent.model.client import ModelClient
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.repository import RepositoryContext
from gear_agent.run_collector import EvalRunCollector
from gear_agent.store.jsonl import JsonlContextStore
from tests.test_agent_loop import SequencedTransport
from tests.test_context_budget import BudgetSink, response
from tests.test_exa import ExaAPI, FETCH, SEARCH, SECRET, URL, api, config_text, configured


@pytest.mark.parametrize("domain", ["", "https://example.com", "user@example.com", "example.com?query=1",
                                     "example.com#fragment", "example.com:1234", "*.example.com/\npath"])
def test_domain_filter_syntax_is_validated_before_network(tmp_path: Path, domain: str) -> None:
    search = SEARCH.replace("include_domains = []", "include_domains = " + json.dumps([domain]))
    with pytest.raises(GearError):
        configured(tmp_path, config_text(search, FETCH))


@pytest.mark.parametrize("highlights", [[], [""], ["", " "], ["\n"]])
def test_blank_highlights_are_explicitly_no_excerpt(tmp_path: Path, api: ExaAPI, highlights: list[str]) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.response = {"results": [{"url": URL, "highlights": highlights}]}
    result = tools["web_search"].run({"query": "query"})
    assert result["results"][0]["content"] is None
    assert result["results"][0]["provider_metadata"]["exa"]["highlights"] == highlights


def test_large_json_number_never_escapes_as_unclassified_exception(tmp_path: Path, api: ExaAPI) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.response = {"results": [{"url": URL, "score": 10 ** 400}]}
    with pytest.raises(GearError) as caught:
        tools["web_search"].run({"query": "query"})
    assert caught.value.origin == "web_search"


class OversizedStream(httpx.SyncByteStream):
    def __init__(self) -> None:
        self.reads = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for _ in range(10000):
            self.reads += 1
            yield b"x" * 8192

    def close(self) -> None:
        self.closed = True


def test_streaming_body_limit_stops_reading_and_closes_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    stream = OversizedStream()

    def handle(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    with pytest.raises(GearError):
        tools["web_search"].run({"query": "query"})
    assert stream.reads <= 124
    assert stream.closed


@pytest.mark.parametrize("status", [200, 401])
def test_loop_presentation_observations_and_jsonl_never_receive_key(tmp_path: Path, api: ExaAPI, status: int) -> None:
    config, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.status = status
    api.response = {"results": [{"url": URL, "title": SECRET, "highlights": [SECRET]}]}
    api.body = None if status == 200 else f"error {SECRET}".encode()
    transport = SequencedTransport([
        {"output": [{"type": "function_call", "call_id": "search", "name": "web_search",
                     "arguments": '{"query":"public query"}'}]}, response("done"),
    ])
    sink = BudgetSink()
    collector = EvalRunCollector()
    collector.start()
    store = JsonlContextStore(tmp_path / "sessions")
    loop = AgentLoop(ResponsesModelAdapter(ModelClient(transport), config.model),
                     list(tools.values()), store, sink, RepositoryContext(tmp_path), observer=collector)
    result = loop.run_turn("s", "search docs", 3, 30)
    assert result.final_text == "done"
    assert SECRET not in repr(sink.events)
    assert SECRET not in json.dumps(transport.payloads)
    assert SECRET not in json.dumps(collector.events)
    assert SECRET not in (tmp_path / "sessions" / "s.jsonl").read_text()
    metrics = collector.metrics("success", None, True, store.load("s"), ArtifactPrivacy(config, {}))
    assert metrics["tools"]["call_count"] == 1
    assert metrics["tools"]["failed"] == (0 if status == 200 else 1)
    assert SECRET not in json.dumps(metrics)
