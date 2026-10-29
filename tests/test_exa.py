from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
import json
import re
import traceback

import httpx
import pytest

from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.history import build_model_history
from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.config import AppConfig, DEFAULT_CONFIG_TEXT, load_config
from gear_agent.errors import GearError
from gear_agent.headless import TaskPrompt, describe_run, diagnostic_secrets
from gear_agent.model.replay import reasoning_replay_policy
from gear_agent.runtime import build_agent_runtime
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.base import Tool
from gear_agent.tools.configured import build_configured_tools
from gear_agent.tools.runtimes import DockerShellRuntime


SECRET = "exa-private-test-key"
URL = "https://example.com/docs"
SEARCH = '''provider = "exa"
api_key_env = "EXA_API_KEY"
max_results = 5
timeout_seconds = 20
max_content_chars = 20000
max_response_bytes = 1000000
[web_search.exa]
search_type = "auto"
content = "highlights"
include_domains = []
exclude_domains = []
'''
FETCH = '''provider = "exa"
api_key_env = "EXA_API_KEY"
timeout_seconds = 20
max_content_chars = 20000
max_response_bytes = 1000000
[web_fetch.exa]
livecrawl_timeout_ms = 10000
'''


def config_text(search: str, fetch: str) -> str:
    text = DEFAULT_CONFIG_TEXT.replace("web_search = false", "web_search = true")
    text = text.replace("web_fetch = false", "web_fetch = true")
    for name, body in (("web_search", search), ("web_fetch", fetch)):
        text = re.sub(rf"\[{name}\]\n.*?(?=\n\[)", f"[{name}]\n{body}\n", text, flags=re.S)
    return text


def configured(root: Path, text: str) -> tuple[AppConfig, dict[str, Tool]]:
    path = root / "config.toml"
    path.write_text(text, encoding="utf-8")
    config = load_config(path, {"EXA_API_KEY": SECRET, "TAVILY_API_KEY": "tavily-test-key"})
    tools = build_configured_tools(config.tool, config.web_search, config.web_fetch,
                                   root, DockerShellRuntime(root, "python:3.11-slim", False))
    return config, {tool.name: tool for tool in tools}


@dataclass
class ExaAPI:
    response: dict[str, Any] = field(default_factory=lambda: {"results": []})
    status: int = 200
    body: bytes | None = None
    failure: httpx.RequestError | None = None
    calls: list[httpx.Request] = field(default_factory=list)


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> ExaAPI:
    external = ExaAPI()

    def handle(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        external.calls.append(request)
        if external.failure is not None:
            raise external.failure
        body = external.body
        if body is None:
            body = json.dumps(external.response).encode()
        return httpx.Response(external.status, content=body,
                              headers={"location": "https://other.example/leak"})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    return external


def fetch_response() -> dict[str, Any]:
    return {"results": [{"id": URL, "url": URL, "text": "Extracted page text"}],
            "statuses": [{"id": URL, "status": "success", "source": "cached"}],
            "requestId": "fetch-123", "costDollars": {"total": 0.001}}


def test_search_payload_and_distinct_content_semantics(tmp_path: Path, api: ExaAPI) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.response = {"results": [{"url": URL, "id": "doc-1", "title": "Docs",
                                 "highlights": ["first", "second"], "text": "full text",
                                 "summary": "generated summary", "publishedDate": "2026-10-20",
                                 "author": "Writer", "score": 0.8}],
                    "requestId": "search-123", "searchTime": 25,
                    "costDollars": {"total": 0.005}, "resolvedSearchType": ""}
    result = tools["web_search"].run({"query": "original query"})
    request = api.calls[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.exa.ai/search"
    assert request.headers["x-api-key"] == SECRET
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {"query": "original query", "type": "auto", "numResults": 5,
                                            "contents": {"highlights": True},
                                            "includeDomains": [], "excludeDomains": []}
    assert request.extensions["timeout"]["read"] == 20
    assert result["query"] == "original query"
    assert result["request_id"] == "search-123"
    row = result["results"][0]
    assert row["content"] == "first\nsecond"
    assert row["raw_content"] == "full text"
    assert row["title"] == "Docs" and row["score"] == 0.8
    assert row["provider_metadata"]["exa"] == {
        "id": "doc-1", "highlights": ["first", "second"], "summary": "generated summary",
        "publishedDate": "2026-10-20", "author": "Writer"}
    assert result["provider_metadata"]["exa"]["estimated_cost_dollars"] == {"total": 0.005}
    assert result["provider_metadata"]["exa"]["search_time_ms"] == 25
    assert result["credits"] is None
    assert SECRET not in json.dumps(result)


@pytest.mark.parametrize("rows", [[], [{"url": URL}]])
def test_empty_search_and_missing_optional_fields(tmp_path: Path, api: ExaAPI, rows: list[dict[str, str]]) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.response = {"results": rows}
    result = tools["web_search"].run({"query": "query"})
    assert len(result["results"]) == len(rows)
    assert result["request_id"] is None
    assert result["provider_metadata"]["exa"]["estimated_cost_dollars"] is None
    for row in result["results"]:
        assert row["content"] is None and row["raw_content"] is None
        assert row["title"] is None and row["score"] is None


def test_explicit_full_text_and_domain_filters(tmp_path: Path, api: ExaAPI) -> None:
    search = SEARCH.replace('content = "highlights"', 'content = "text"')
    search = search.replace("include_domains = []", 'include_domains = ["example.com/docs"]')
    search = search.replace("exclude_domains = []", 'exclude_domains = ["other.example"]')
    _, tools = configured(tmp_path, config_text(search, FETCH))
    tools["web_search"].run({"query": "site:example.com unchanged"})
    payload = json.loads(api.calls[0].content)
    assert payload["query"] == "site:example.com unchanged"
    assert payload["contents"] == {"text": {"maxCharacters": 20000}}
    assert payload["includeDomains"] == ["example.com/docs"]
    assert payload["excludeDomains"] == ["other.example"]


@pytest.mark.parametrize("age", [None, -1, 0, 24, 720])
def test_fetch_body_freshness_url_mapping_and_timeout(tmp_path: Path, api: ExaAPI, age: int | None) -> None:
    fetch = FETCH if age is None else FETCH + f"max_age_hours = {age}\n"
    _, tools = configured(tmp_path, config_text(SEARCH, fetch))
    api.response = fetch_response()
    api.response["results"][0]["url"] = "https://example.com/new-docs"
    result = tools["web_fetch"].run({"url": URL})
    payload: dict[str, Any] = {"urls": [URL], "text": True, "livecrawlTimeout": 10000}
    if age is not None:
        payload["maxAgeHours"] = age
    assert str(api.calls[0].url) == "https://api.exa.ai/contents"
    assert api.calls[0].headers["x-api-key"] == SECRET
    assert json.loads(api.calls[0].content) == payload
    assert result["requested_url"] == URL
    assert result["url"] == "https://example.com/new-docs"
    assert result["content"] == "Extracted page text"
    assert result["content_format"] == "text"
    assert result["title"] is None
    assert result["provider_metadata"]["exa"]["id"] == URL
    assert result["provider_metadata"]["exa"]["source"] == "cached"


def test_unreported_source_is_not_invented(tmp_path: Path, api: ExaAPI) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.response = fetch_response()
    del api.response["statuses"][0]["source"]
    assert tools["web_fetch"].run({"url": URL})["provider_metadata"]["exa"]["source"] is None


@pytest.mark.parametrize("mutation", ["failed", "missing_status", "wrong_status", "wrong_result", "missing_result",
                                       "empty", "whitespace", "no_text", "huge", "duplicate", "top_error"])
def test_fetch_rejects_partial_or_invalid_success(tmp_path: Path, api: ExaAPI, mutation: str) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    body = fetch_response()
    if mutation == "failed":
        body["statuses"][0].update(status="error", error={"tag": SECRET})
    elif mutation == "missing_status":
        del body["statuses"]
    elif mutation == "wrong_status":
        body["statuses"][0]["id"] = "https://other.example"
    elif mutation == "wrong_result":
        body["results"][0].update(id="unrelated", url="https://other.example")
    elif mutation == "missing_result":
        body["results"] = []
    elif mutation == "empty":
        body["results"][0]["text"] = ""
    elif mutation == "whitespace":
        body["results"][0]["text"] = " \n "
    elif mutation == "no_text":
        del body["results"][0]["text"]
        body["results"][0]["summary"] = "not page text"
    elif mutation == "huge":
        body["results"][0]["text"] = "x" * 20001
    elif mutation == "duplicate":
        body["results"] *= 2
    elif mutation == "top_error":
        body["error"] = SECRET
    api.response = body
    with pytest.raises(GearError) as raised:
        tools["web_fetch"].run({"url": URL})
    assert raised.value.origin == "web_fetch"
    assert SECRET not in str(raised.value) + repr(raised.value.details)
    assert len(api.calls) == 1


@pytest.mark.parametrize("body", [{"results": "bad"}, {"results": [{}]}, {"results": [{"url": URL, "score": True}]},
                                    {"results": [{"url": URL, "highlights": "bad"}]},
                                    {"results": [{"url": URL, "text": "x" * 20001}]},
                                    {"results": [{"url": URL}] * 6},
                                    {"results": [], "error": SECRET}])
def test_search_rejects_invalid_results(tmp_path: Path, api: ExaAPI, body: dict[str, Any]) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.response = body
    with pytest.raises(GearError):
        tools["web_search"].run({"query": "query"})


@pytest.mark.parametrize("status", [301, 307, 400, 401, 402, 403, 429, 500, 503])
@pytest.mark.parametrize("name,arguments", [("web_search", {"query": "query"}), ("web_fetch", {"url": URL})])
def test_http_failure_is_safe_and_not_retried(tmp_path: Path, api: ExaAPI, status: int,
                                             name: str, arguments: dict[str, object]) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.status = status
    api.body = f"upstream confidential error {SECRET}".encode()
    with pytest.raises(GearError) as raised:
        tools[name].run(arguments)
    assert raised.value.origin == name
    assert raised.value.details["status"] == status
    assert "confidential" not in repr(raised.value)
    assert SECRET not in "".join(traceback.format_exception(raised.value))
    assert len(api.calls) == 1


@pytest.mark.parametrize("failure", [httpx.ReadTimeout(SECRET), httpx.ConnectError(SECRET)])
def test_transport_exception_is_safe(tmp_path: Path, api: ExaAPI, failure: httpx.RequestError) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.failure = failure
    with pytest.raises(GearError) as raised:
        tools["web_search"].run({"query": "query"})
    assert SECRET not in "".join(traceback.format_exception(raised.value))
    assert len(api.calls) == 1


@pytest.mark.parametrize("body", [b"not json", b"[]", b"\xff", b'{"results":[],"score":NaN}', b"x" * 1000001])
def test_invalid_or_oversized_http_body(tmp_path: Path, api: ExaAPI, body: bytes) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.body = body
    with pytest.raises(GearError):
        tools["web_search"].run({"query": "query"})


@pytest.mark.parametrize("url", ["file:///tmp/file", "http://localhost/x", "http://127.0.0.1/x",
                                  "http://169.254.169.254/", "http://[::1]/", "https://10.0.0.1/",
                                  "https://user:password@example.com/", "https://example.com:bad/",
                                  "https://example.com/\nheader", "https://2130706433/", "https://example.local/",
                                  "http://[::ffff:127.0.0.1]/"])
def test_fetch_rejects_nonpublic_or_credential_urls_before_http(tmp_path: Path, api: ExaAPI, url: str) -> None:
    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    with pytest.raises(GearError):
        tools["web_fetch"].run({"url": url})
    assert not api.calls


@pytest.mark.parametrize("old,new", [('search_type = "auto"', 'search_type = "deep"'),
                                     ('search_type = "auto"', 'search_type = "keyword"'),
                                     ('max_results = 5', 'max_results = 101'),
                                     ('max_results = 5', 'max_results = true'),
                                     ('content = "highlights"', 'content = "summary"'),
                                     ('include_domains = []', 'include_domains = "example.com"'),
                                     ('max_content_chars = 20000', 'max_content_chars = 0'),
                                     ('max_response_bytes = 1000000', 'max_response_bytes = -1'),
                                     ('provider = "exa"', 'provider = "unknown"')])
def test_invalid_search_config(tmp_path: Path, old: str, new: str) -> None:
    with pytest.raises(GearError) as raised:
        configured(tmp_path, config_text(SEARCH.replace(old, new), FETCH))
    assert raised.value.origin == "config"


@pytest.mark.parametrize("option", ["max_age_hours = -2", "max_age_hours = 721", "max_age_hours = true",
                                     "max_age_hours = 0.5", 'livecrawl = "always"', 'start_crawl_date = "2026-01-01"'])
def test_invalid_fetch_options(tmp_path: Path, option: str) -> None:
    with pytest.raises(GearError):
        configured(tmp_path, config_text(SEARCH, FETCH + option + "\n"))


@pytest.mark.parametrize("timeout", [0, 20000, 90001])
def test_live_timeout_must_fit_http_budget(tmp_path: Path, timeout: int) -> None:
    with pytest.raises(GearError):
        configured(tmp_path, config_text(SEARCH, FETCH.replace("10000", str(timeout))))


def test_defaults_and_provider_specific_option_conflicts(tmp_path: Path) -> None:
    search = SEARCH.replace('search_type = "auto"\n', '').replace('content = "highlights"\n', '')
    config, _ = configured(tmp_path, config_text(search, FETCH))
    assert config.web_search.exa.search_type == "auto"
    assert config.web_search.exa.content == "highlights"
    with pytest.raises(GearError):
        configured(tmp_path, config_text(SEARCH.replace('[web_search.exa]', 'search_depth = "basic"\n[web_search.exa]'), FETCH))


@pytest.mark.parametrize("selected", ["web_search", "web_fetch"])
def test_independent_selection_and_legacy_tavily(tmp_path: Path, selected: str) -> None:
    text = DEFAULT_CONFIG_TEXT.replace("web_search = false", "web_search = true").replace("web_fetch = false", "web_fetch = true")
    body = SEARCH if selected == "web_search" else FETCH
    text = re.sub(rf"\[{selected}\]\n.*?(?=\n\[)", f"[{selected}]\n{body}\n", text, flags=re.S)
    config, tools = configured(tmp_path, text)
    assert config.model.model == "local-model-id"
    assert "Exa" in tools[selected].schema()["description"]
    other = "web_fetch" if selected == "web_search" else "web_search"
    assert "Tavily" in tools[other].schema()["description"]
    assert tools[selected].schema()["parameters"]["required"] == (["query"] if selected == "web_search" else ["url"])


def test_disabled_exa_needs_no_key_and_missing_selected_key_fails(tmp_path: Path, api: ExaAPI) -> None:
    path = tmp_path / "config.toml"
    text = config_text(SEARCH, FETCH)
    path.write_text(text.replace("web_search = true", "web_search = false").replace("web_fetch = true", "web_fetch = false"))
    config = load_config(path, {})
    assert config.web_search is None and config.web_fetch is None
    assert not api.calls
    path.write_text(text)
    with pytest.raises(GearError) as raised:
        load_config(path, {})
    assert raised.value.error_type == "config_secret_missing"


def test_keys_cannot_reach_result_or_run_metadata(tmp_path: Path, api: ExaAPI) -> None:
    config, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.response = {"results": [{"url": URL, "title": SECRET, "highlights": [SECRET]}], "requestId": SECRET}
    result = tools["web_search"].run({"query": "query"})
    assert SECRET not in json.dumps(result)
    assert SECRET not in repr(config)
    assert SECRET in diagnostic_secrets(config)
    privacy = ArtifactPrivacy(config, {})
    assert SECRET not in json.dumps(privacy.serialize({"result": result}))
    runtime = replace(config.runtime, workdir=tmp_path, session_dir=tmp_path / "sessions")
    agent = build_agent_runtime(config, runtime, SilentAgentLoopEventSink())
    spec = asdict(describe_run(agent, "s", TaskPrompt("test", "inline")))
    assert spec["web_search"]["provider"] == "exa"
    assert spec["web_fetch"]["provider"] == "exa"
    assert SECRET not in json.dumps(spec)


def test_exa_results_round_trip_through_jsonl_jev_and_summary(tmp_path: Path, api: ExaAPI) -> None:
    from gear_agent.agent.compaction import CompactionService
    from gear_agent.model.client import ModelClient
    from gear_agent.model.responses_adapter import ResponsesModelAdapter
    from tests.test_agent_loop import SequencedTransport
    from tests.test_context_budget import response
    from tests.test_jev_compaction import JevAPI, MODEL, POLICY, strategy

    _, tools = configured(tmp_path, config_text(SEARCH, FETCH))
    api.response = fetch_response()
    result = tools["web_fetch"].run({"url": URL})
    store = JsonlContextStore(tmp_path / "sessions")
    store.append("s", "user_input", {"text": "read docs"})
    store.append("s", "model_response", {"output": [{"type": "function_call", "call_id": "fetch",
                                                     "name": "web_fetch", "arguments": json.dumps({"url": URL})}]})
    store.append("s", "tool_result", {"call_id": "fetch", "name": "web_fetch", "result": result})
    store.append("s", "assistant_message", {"text": "read"})
    store.append("s", "user_input", {"text": "recent"})
    store.append("s", "assistant_message", {"text": "done"})
    store.append("s", "user_input", {"text": "current"})
    before = build_model_history(store.load("s"), reasoning_replay_policy(MODEL)).items
    jev_api = JevAPI({"fetch": (1, 1)})
    candidate = strategy(jev_api, replace(POLICY, keep_threshold=0.3)).compact(store.load("s"), 30, None)
    store.append("s", candidate.kind, candidate.payload)
    resumed = JsonlContextStore(tmp_path / "sessions")
    assert build_model_history(resumed.load("s"), reasoning_replay_policy(MODEL)).items == before
    assert "fetch:call" in jev_api.requests[0][1]
    assert "Extracted page text" not in jev_api.requests[0][0]
    transport = SequencedTransport([response("summary")])
    CompactionService(ResponsesModelAdapter(ModelClient(transport), MODEL)).compact("s", resumed, 30)
    assert "Extracted page text" in transport.payloads[0]["input"]
    assert SECRET not in (tmp_path / "sessions" / "s.jsonl").read_text()
