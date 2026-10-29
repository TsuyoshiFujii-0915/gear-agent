from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.loop import AgentLoop
from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.model.client import ModelClient
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.repository import RepositoryContext
from gear_agent.run_collector import EvalRunCollector
from gear_agent.config import DEFAULT_DOCKER_IMAGE
from gear_agent.tools.configured import build_configured_tools
from gear_agent.tools.runtimes import DockerShellRuntime
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tui import collect_chat_lines
from tests.test_agent_loop import SequencedTransport
from tests.test_context_budget import response
from tests.test_deepseek_integration import config
from tests.test_jev_compaction import MODEL


@pytest.mark.parametrize("tool_name", ["web_search", "web_fetch"])
@pytest.mark.parametrize("status", [401, 402, 429])
def test_deepseek_errors_reach_model_and_ui_without_raw_secrets(
    tmp_path: Path, tool_name: str, status: int,
) -> None:
    settings = config(tmp_path)
    tools = build_configured_tools(
        settings.tool, settings.web_search, settings.web_fetch, tmp_path,
        DockerShellRuntime(tmp_path, DEFAULT_DOCKER_IMAGE, settings.runtime.network_enabled),
    )
    arguments = {"query": "example"} if tool_name == "web_search" else {"url": "https://example.com/"}
    model = SequencedTransport([{"output": [{"type": "function_call", "call_id": "web",
                                            "name": tool_name, "arguments": json.dumps(arguments)}]},
                                response("The web request needs attention.")])
    requests: list[httpx.Request] = []

    def handle(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, text="search-secret fetch-secret upstream-private-body")

    store = JsonlContextStore(tmp_path / "sessions")
    collector = EvalRunCollector()
    collector.start()
    loop = AgentLoop(ResponsesModelAdapter(ModelClient(model), MODEL), tools, store,
                     SilentAgentLoopEventSink(), RepositoryContext(tmp_path), observer=collector)
    with patch.object(httpx.HTTPTransport, "handle_request", handle), patch(
        "socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.215.14", 443))],
    ):
        assert loop.run_turn("s", "Research example.", 3, 20).final_text == "The web request needs attention."
    assert len(requests) == 1
    events = store.load("s")
    assert f"HTTP {status}" in json.dumps(model.payloads[1]["input"])
    metrics = collector.metrics("completed", None, True, events, ArtifactPrivacy(settings, {}))
    assert metrics["tools"]["calls"][0]["provider"] == "deepseek"
    assert metrics["tools"]["calls"][0]["successful"] is False
    surfaces = json.dumps(events) + json.dumps(model.payloads) + json.dumps(metrics) + json.dumps(
        [asdict(item) for item in collect_chat_lines(events)])
    for secret in ("search-secret", "fetch-secret", "upstream-private-body"):
        assert secret not in surfaces
    summary = SequencedTransport([response("Web request failed safely.")])
    CompactionService(ResponsesModelAdapter(ModelClient(summary), MODEL)).compact("s", store, 30)
    assert "deepseek" in summary.payloads[0]["input"]
