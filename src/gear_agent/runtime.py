from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gear_agent.agent.events import AgentLoopEventSink
from gear_agent.agent.loop import AgentLoop
from gear_agent.config import AppConfig, DEFAULT_DOCKER_IMAGE, RuntimeConfig
from gear_agent.model.adapter import ModelAdapter
from gear_agent.model.factory import build_model_adapter
from gear_agent.repository import RepositoryContext
from gear_agent.observation import RunObserver
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.configured import build_configured_tools
from gear_agent.tools.runtimes import DockerShellRuntime


@dataclass(frozen=True)
class AgentRuntime:
    """Effective configuration and services shared by interactive and batch execution."""

    config: AppConfig
    runtime: RuntimeConfig
    workspace: Path
    store: JsonlContextStore
    adapter: ModelAdapter
    loop: AgentLoop


def build_agent_runtime(
    config: AppConfig, runtime: RuntimeConfig, event_sink: AgentLoopEventSink,
    observer: RunObserver | None = None,
) -> AgentRuntime:
    """Constructs the production harness without creating presentation objects.

    Args:
        config: Loaded application configuration.
        runtime: Effective runtime after CLI overrides.
        event_sink: Presentation-specific progress consumer.
        observer: Existing TUI and embedding callers omit run-only observations.

    Returns:
        Configured agent, model adapter, tools, repository context and session store.
    """
    workspace = runtime.workdir.resolve()
    store = JsonlContextStore(runtime.session_dir)
    shell_runtime = DockerShellRuntime(workspace, DEFAULT_DOCKER_IMAGE, runtime.network_enabled)
    tools = build_configured_tools(
        config.tool, config.web_search, config.web_fetch, workspace, shell_runtime,
    )
    adapter = build_model_adapter(config.model)
    loop = AgentLoop(
        adapter, tools, store, event_sink, RepositoryContext(workspace),
        context_budget=config.context_budget, observer=observer,
    )
    return AgentRuntime(config, runtime, workspace, store, adapter, loop)
