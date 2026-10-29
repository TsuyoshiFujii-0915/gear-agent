from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.history import build_model_history
from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.config import AppConfig, DEFAULT_CONFIG_TEXT, DEFAULT_DOCKER_IMAGE, load_config
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
from tests.test_jev_compaction import MODEL, POLICY, JevAPI, event, strategy


WEB = '''[web_search]
provider = "deepseek"
api_key_env = "SEARCH_KEY"
max_results = 50
timeout_seconds = 20
[web_search.deepseek]
need_content = false

[web_fetch]
provider = "deepseek"
api_key_env = "FETCH_KEY"
timeout_seconds = 20
max_content_chars = 20000
[web_fetch.deepseek]
content_format = "markdown"

'''


def configuration(tmp_path: Path, web: str, environment: dict[str, str]) -> AppConfig:
    text = DEFAULT_CONFIG_TEXT.replace("web_search = false", "web_search = true").replace(
        "web_fetch = false", "web_fetch = true")
    text = text[:text.index("[web_search]")] + web + text[text.index("[runtime]"):]
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return load_config(path, environment)


def config(tmp_path: Path) -> AppConfig:
    return configuration(tmp_path, WEB, {"SEARCH_KEY": "search-secret", "FETCH_KEY": "fetch-secret"})


def registry(settings: AppConfig, workspace: Path) -> ToolRegistry:
    return ToolRegistry(build_configured_tools(
        settings.tool, settings.web_search, settings.web_fetch, workspace,
        DockerShellRuntime(workspace, DEFAULT_DOCKER_IMAGE, settings.runtime.network_enabled),
    ))


def test_deepseek_config_is_independent_of_model_and_tavily(tmp_path: Path) -> None:
    settings = config(tmp_path)
    assert settings.model.model == "local-model-id"
    assert settings.web_search.provider == settings.web_fetch.provider == "deepseek"
    assert settings.web_search.api_key == "search-secret"
    assert settings.web_fetch.api_key == "fetch-secret"
    assert settings.web_search.max_results == 50
    assert settings.web_search.options.need_content is False
    assert settings.web_fetch.options.content_format == "markdown"


def test_need_content_defaults_to_false(tmp_path: Path) -> None:
    settings = configuration(tmp_path, WEB.replace("[web_search.deepseek]\nneed_content = false\n", ""),
                             {"SEARCH_KEY": "search-secret", "FETCH_KEY": "fetch-secret"})
    assert settings.web_search.options.need_content is False


@pytest.mark.parametrize("old,new", [
    ("max_results = 50", "max_results = 0"),
    ("max_results = 50", "max_results = 51"),
    ("max_results = 50", "max_results = true"),
    ("need_content = false", 'need_content = "false"'),
    ("need_content = false", "need_content = false\nunknown = true"),
    ("max_results = 50", 'search_depth = "basic"\nmax_results = 50'),
    ('content_format = "markdown"', 'content_format = "text"'),
    ('content_format = "markdown"', 'content_format = "markdown"\ninclude_images = true'),
    ('[web_search.deepseek]', '[web_search.tavily]'),
])
def test_invalid_deepseek_options_fail_during_configuration(tmp_path: Path, old: str, new: str) -> None:
    with pytest.raises(GearError) as caught:
        configuration(tmp_path, WEB.replace(old, new), {"SEARCH_KEY": "one", "FETCH_KEY": "two"})
    assert caught.value.origin == "config"


@pytest.mark.parametrize("disabled", ["web_search", "web_fetch"])
def test_disabled_deepseek_tool_does_not_require_its_key(tmp_path: Path, disabled: str) -> None:
    settings = config(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text(path.read_text().replace(f"{disabled} = true", f"{disabled} = false"))
    environment = {"FETCH_KEY": "fetch-secret"} if disabled == "web_search" else {"SEARCH_KEY": "search-secret"}
    settings = load_config(path, environment)
    assert getattr(settings, disabled) is None


@pytest.mark.parametrize("tavily_tool", ["web_search", "web_fetch"])
def test_search_and_fetch_can_mix_tavily_and_deepseek(tmp_path: Path, tavily_tool: str) -> None:
    start = DEFAULT_CONFIG_TEXT.index(f"[{tavily_tool}]")
    end = DEFAULT_CONFIG_TEXT.index("[web_fetch]" if tavily_tool == "web_search" else "[runtime]")
    legacy_table = DEFAULT_CONFIG_TEXT[start:end]
    if tavily_tool == "web_search":
        web = legacy_table + WEB[WEB.index("[web_fetch]"):]
    else:
        web = WEB[:WEB.index("[web_fetch]")] + legacy_table
    settings = configuration(tmp_path, web, {"TAVILY_API_KEY": "tavily-secret",
                                            "SEARCH_KEY": "search-secret", "FETCH_KEY": "fetch-secret"})
    assert getattr(settings, tavily_tool).provider == "tavily"
    assert settings.web_search.provider != settings.web_fetch.provider
    agent = build_agent_runtime(settings, replace(settings.runtime, workdir=tmp_path), SilentAgentLoopEventSink())
    schemas = {item["name"]: item for item in registry(settings, tmp_path).schemas()}
    assert "Tavily" in schemas[tavily_tool]["description"]
    other = "web_fetch" if tavily_tool == "web_search" else "web_search"
    assert "DeepSeek" in schemas[other]["description"]


def test_factory_schema_http_observation_and_secrets(tmp_path: Path) -> None:
    settings = config(tmp_path)
    agent = build_agent_runtime(settings, replace(settings.runtime, workdir=tmp_path,
                                                  session_dir=tmp_path / "sessions"), SilentAgentLoopEventSink())
    schemas = {item["name"]: item for item in registry(settings, tmp_path).schemas()}
    search_schema = schemas["web_search"]
    assert search_schema["parameters"]["required"] == ["query"]
    assert set(search_schema["parameters"]["properties"]) == {"query"}
    assert "128" in search_schema["parameters"]["properties"]["query"]["description"]
    assert "8192" in schemas["web_fetch"]["parameters"]["properties"]["url"]["description"]
    assert "Markdown" in json.dumps(schemas["web_fetch"])
    requests: list[httpx.Request] = []

    def handle(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("search"):
            assert request.headers["Authorization"] == "Bearer search-secret"
            return httpx.Response(200, json={"results": []})
        assert request.headers["Authorization"] == "Bearer fetch-secret"
        return httpx.Response(200, json={"title": "", "content": "untrusted page content"})

    collector = EvalRunCollector()
    with patch.object(httpx.HTTPTransport, "handle_request", handle), patch(
        "socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.215.14", 443))],
    ):
        search_result = execute_tool(registry(settings, tmp_path), "web_search", {"query": "private query"}, "s", collector)
        fetch_result = execute_tool(registry(settings, tmp_path), "web_fetch", {"url": "https://example.com/"}, "f", collector)
    assert search_result["query"] == "private query"
    assert fetch_result["content"] == "untrusted page content"
    assert len(requests) == 2
    observations = [item["payload"] for item in collector.events if item["kind"] == "tool_finished"]
    assert all(item["provider"] == "deepseek" and item["usage"] == [] and item["successful"] for item in observations)
    assert "private query" not in json.dumps(collector.events)
    assert "untrusted page content" not in json.dumps(collector.events)
    spec = describe_run(agent, "s", TaskPrompt("research", "inline"))
    assert spec.web_search["provider"] == spec.web_fetch["provider"] == "deepseek"
    error = safe_error_payload(gear_error("test", "search-secret fetch-secret", "test", True, {}), diagnostic_secrets(settings))
    privacy = ArtifactPrivacy(settings, {})
    encoded = repr(settings) + json.dumps(asdict(spec)) + json.dumps(error) + json.dumps(
        privacy.serialize({"body": "search-secret fetch-secret"}))
    assert "search-secret" not in encoded
    assert "fetch-secret" not in encoded


@pytest.mark.parametrize("tool_name", ["web_search", "web_fetch"])
def test_results_survive_jsonl_jev_and_resume(tmp_path: Path, tool_name: str) -> None:
    settings = config(tmp_path)
    agent = build_agent_runtime(settings, replace(settings.runtime, workdir=tmp_path), SilentAgentLoopEventSink())

    def handle(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        if tool_name == "web_search":
            return httpx.Response(200, json={"results": [{"title": "Example", "url": "https://example.com/",
                                                         "snippet": "excerpt", "summary": "summary", "content": ""}]})
        return httpx.Response(200, json={"title": "", "content": "# Example"})

    arguments = {"query": "example"} if tool_name == "web_search" else {"url": "https://example.com/"}
    with patch.object(httpx.HTTPTransport, "handle_request", handle), patch(
        "socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.215.14", 443))],
    ):
        output = registry(settings, tmp_path).run(tool_name, arguments)
    store = JsonlContextStore(tmp_path / "jsonl")
    events = [event("user_input", {"text": "research"}), event("model_response", {"output": [{
        "type": "function_call", "call_id": "web", "name": tool_name, "arguments": json.dumps(arguments)}]}),
        event("tool_result", {"name": tool_name, "call_id": "web", "result": output}),
        event("assistant_message", {"text": "done"}),
        event("user_input", {"text": "recent task"}),
        event("assistant_message", {"text": "recent task completed"}),
        event("user_input", {"text": "new task"})]
    for item in events:
        store.append("s", item["kind"], item["payload"])
    history = build_model_history(store.load("s"), reasoning_replay_policy(MODEL)).items
    saved_result = next(item for item in history if item.get("type") == "function_call_output")
    assert json.loads(saved_result["output"]) == output
    checkpoint = strategy(JevAPI({"web": (1, 1)}), POLICY).compact(store.load("s"), 30, None)
    assert checkpoint.metrics["eligible_interactions"] == 1
    store.append("s", checkpoint.kind, checkpoint.payload)
    resumed = JsonlContextStore(tmp_path / "jsonl")
    assert build_model_history(resumed.load("s"), reasoning_replay_policy(MODEL)).items == history
