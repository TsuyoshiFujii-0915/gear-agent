from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.history import build_model_history
from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.config import AppConfig, DEFAULT_CONFIG_TEXT, load_config
from gear_agent.errors import GearError, gear_error, safe_error_payload
from gear_agent.headless import TaskPrompt, describe_run, diagnostic_secrets
from gear_agent.model.replay import reasoning_replay_policy
from gear_agent.observation import execute_tool
from gear_agent.run_collector import EvalRunCollector
from gear_agent.runtime import build_agent_runtime
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.configured import build_configured_tools
from gear_agent.tools.registry import ToolRegistry
from gear_agent.tools.runtimes import DockerShellRuntime
from tests.test_treg import TOKEN, catalog, fetch_response, search_response


TREG_SEARCH = '''[web_search]
provider = "treg"
api_key_env = "TREG_TOKEN"
max_results = 1
timeout_seconds = 20
[web_search.treg]
org = "team"

'''
TREG_FETCH = '''[web_fetch]
provider = "treg"
api_key_env = "TREG_TOKEN"
max_content_chars = 100
timeout_seconds = 20
[web_fetch.treg]
org = "team"
ttl_seconds = 0
per_url_timeout_ms = 15000

'''
TAVILY_SEARCH = '''[web_search]
api_key_env = "TAVILY_API_KEY"
max_results = 5
timeout_seconds = 20
search_depth = "basic"
include_answer = true
include_raw_content = false

'''
TAVILY_FETCH = '''[web_fetch]
api_key_env = "TAVILY_API_KEY"
max_content_chars = 100
timeout_seconds = 20
extract_depth = "basic"
content_format = "markdown"
include_images = false
include_favicon = false

'''


def config_text(search: str, fetch: str) -> str:
    start = DEFAULT_CONFIG_TEXT.index("[web_search]")
    end = DEFAULT_CONFIG_TEXT.index("[runtime]")
    return (DEFAULT_CONFIG_TEXT[:start] + search + fetch + DEFAULT_CONFIG_TEXT[end:]).replace(
        "web_search = false", "web_search = true").replace("web_fetch = false", "web_fetch = true")


def load(tmp_path: Path, text: str, environment: dict[str, str]) -> AppConfig:
    path = tmp_path / "gear.toml"
    path.write_text(text, encoding="utf-8")
    return load_config(path, environment)


def registry(config: AppConfig, tmp_path: Path) -> ToolRegistry:
    return ToolRegistry(build_configured_tools(config.tool, config.web_search, config.web_fetch,
                                              tmp_path, DockerShellRuntime(tmp_path, "python:3.11-slim", False)))


def external_response(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
    endpoint = request.url.path.rsplit("/", 1)[1]
    if request.url.path.startswith("/catalog/"):
        return httpx.Response(200, json=catalog(endpoint, "GET" if endpoint.endswith("search") else "POST"))
    if request.url.host == "treg.to":
        assert request.headers["x-treg-token"] == TOKEN
        data = search_response() if endpoint.endswith("search") else fetch_response()
        return httpx.Response(200, json=data, headers={"X-Treg-Cost-Micro": "0", "X-Treg-Call-Id": "call-1"})
    assert request.url.host == "api.tavily.com"
    if endpoint == "search":
        return httpx.Response(200, json={"query": "q", "results": []})
    return httpx.Response(200, json={"results": [{"url": "https://example.com", "raw_content": "Tavily body"}], "failed_results": []})


@pytest.mark.parametrize("search,fetch,search_provider,fetch_provider", [
    (TREG_SEARCH, TREG_FETCH, "treg", "treg"),
    (TREG_SEARCH, TAVILY_FETCH, "treg", "tavily"),
    (TAVILY_SEARCH, TREG_FETCH, "tavily", "treg"),
])
def test_configured_tools_select_providers_independently(tmp_path: Path, search: str, fetch: str,
                                                       search_provider: str, fetch_provider: str) -> None:
    config = load(tmp_path, config_text(search, fetch), {"TREG_TOKEN": TOKEN, "TAVILY_API_KEY": "tvly-key"})
    tools = registry(config, tmp_path)
    schemas = {s["name"]: s for s in tools.schemas()}
    assert schemas["web_search"]["parameters"]["required"] == ["query"]
    assert schemas["web_fetch"]["parameters"]["required"] == ["url"]
    with patch.object(httpx.HTTPTransport, "handle_request", external_response), patch(
        "socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.215.14", 0))]
    ):
        search_result = tools.run("web_search", {"query": "q"})
        fetch_result = tools.run("web_fetch", {"url": "https://example.com"})
    assert search_result["provider"] == search_provider
    assert fetch_result["provider"] == fetch_provider
    if search_provider == "treg":
        meta = search_result["provider_metadata"]["treg"]
        assert meta["endpoint"] == "tinyfish.web.search"
        assert meta["charged_micro"] == 0
        assert meta["usage"] == [{"unit": "micro_USD", "value": 0}]
        assert len(search_result["results"]) == 1
    if fetch_provider == "treg":
        assert fetch_result["requested_url"] == "https://example.com"
        assert fetch_result["final_url"] == "https://example.com/new"
        assert fetch_result["provider_metadata"]["treg"]["ttl_seconds"] == 0


@pytest.mark.parametrize("old,new", [
    ('provider = "treg"', 'provider = "unknown"'),
    ('max_results = 1', 'max_results = true'), ('max_results = 1', 'max_results = 0'),
    ('timeout_seconds = 20', 'timeout_seconds = 0'),
    ('max_content_chars = 100', 'max_content_chars = 0'),
    ('ttl_seconds = 0', 'ttl_seconds = -1'), ('ttl_seconds = 0', 'ttl_seconds = true'),
    ('ttl_seconds = 0', 'ttl_seconds = 0.5'),
    ('per_url_timeout_ms = 15000', 'per_url_timeout_ms = 0'),
    ('per_url_timeout_ms = 15000', 'per_url_timeout_ms = false'),
    ('org = "team"', 'org = 12'), ('org = "team"', 'org = ""'),
    ('org = "team"', 'org = "bad\\nheader"'),
    ('[web_search.treg]', '[web_search.treg]\nunknown = true'),
    ('[web_search.treg]', 'search_depth = "basic"\n[web_search.treg]'),
    ('[web_fetch.treg]', '[web_fetch.treg]\ncontent_format = "html"'),
    ('provider = "treg"', 'provider = "treg"\nallow_paid = true'),
])
def test_invalid_treg_options_fail_at_config_load(tmp_path: Path, old: str, new: str) -> None:
    with pytest.raises(GearError) as raised:
        load(tmp_path, config_text(TREG_SEARCH, TREG_FETCH).replace(old, new), {"TREG_TOKEN": TOKEN})
    assert raised.value.origin == "config"
    assert TOKEN not in repr(raised.value)


@pytest.mark.parametrize("token", ["", "bad\nheader", "nonascii-秘密"])
def test_invalid_token_is_rejected_without_logging_value(tmp_path: Path, token: str) -> None:
    with pytest.raises(GearError) as raised:
        load(tmp_path, config_text(TREG_SEARCH, TREG_FETCH), {"TREG_TOKEN": token})
    assert raised.value.origin == "config"
    if token:
        assert token not in repr(raised.value)


def test_disabled_tools_do_not_require_treg_token_or_create_http_calls(tmp_path: Path) -> None:
    text = config_text(TREG_SEARCH, TREG_FETCH).replace("web_search = true", "web_search = false").replace("web_fetch = true", "web_fetch = false")
    with patch.object(httpx.HTTPTransport, "handle_request", side_effect=AssertionError("Unexpected HTTP")):
        config = load(tmp_path, text, {})
        tools = registry(config, tmp_path)
    assert config.web_search is None and config.web_fetch is None
    assert not any(s["name"].startswith("web_") for s in tools.schemas())


def test_per_org_token_and_cached_fetch_are_explicit_config(tmp_path: Path) -> None:
    config = load(tmp_path, config_text(TREG_SEARCH, TREG_FETCH).replace('org = "team"\n', '').replace('ttl_seconds = 0', 'ttl_seconds = 300'), {"TREG_TOKEN": TOKEN})
    assert config.web_search.options.org is None
    assert config.web_fetch.options.org is None
    assert config.web_fetch.options.ttl_seconds == 300


def test_results_and_observations_preserve_billing_without_leaking_token(tmp_path: Path) -> None:
    config = load(tmp_path, config_text(TREG_SEARCH, TREG_FETCH), {"TREG_TOKEN": TOKEN})
    collector = EvalRunCollector()
    tools = registry(config, tmp_path)
    with patch.object(httpx.HTTPTransport, "handle_request", external_response):
        result = execute_tool(tools, "web_search", {"query": "private query"}, "c", collector)
    finished = next(e["payload"] for e in collector.events if e["kind"] == "tool_finished")
    assert finished["provider"] == "treg"
    assert finished["usage"] == [{"unit": "micro_USD", "value": 0}]
    assert "private query" not in json.dumps(collector.events)
    store = JsonlContextStore(tmp_path / "sessions")
    store.append("s", "user_input", {"text": "research"})
    store.append("s", "model_response", {"output": [{"type": "function_call", "name": "web_search", "call_id": "c", "arguments": '{"query":"private query"}'}]})
    store.append("s", "tool_result", {"name": "web_search", "call_id": "c", "result": result})
    history = build_model_history(store.load("s"), reasoning_replay_policy(config.model)).items
    output = next(item for item in history if item.get("type") == "function_call_output")
    assert json.loads(output["output"]) == result
    runtime = replace(config.runtime, workdir=tmp_path, session_dir=tmp_path / "sessions")
    agent = build_agent_runtime(config, runtime, SilentAgentLoopEventSink())
    spec = describe_run(agent, "s", TaskPrompt("task", "inline"))
    assert spec.web_search["provider"] == "treg"
    diagnostic = safe_error_payload(gear_error("test", TOKEN, "test", True, {}), diagnostic_secrets(config))
    privacy = ArtifactPrivacy(config, {})
    encoded = repr(config) + json.dumps(asdict(spec)) + json.dumps(diagnostic) + json.dumps(collector.events)
    encoded += json.dumps(privacy.serialize({"content": TOKEN})) + (tmp_path / "sessions/s.jsonl").read_text()
    assert TOKEN not in encoded
