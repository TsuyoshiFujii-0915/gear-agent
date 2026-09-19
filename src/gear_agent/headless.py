from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from gear_agent.agent.loop import TurnResult
from gear_agent.errors import GearError
from gear_agent.model.replay import endpoint_identity
from gear_agent.runtime import AgentRuntime


@dataclass(frozen=True)
class TaskPrompt:
    """Task text and its explicit source (inline or an absolute UTF-8 file path)."""

    text: str
    source: str


@dataclass(frozen=True)
class RunSpec:
    """Effective execution inputs, excluding credentials and raw endpoint URLs."""

    session_id: str
    prompt_source: str
    model: dict[str, object]
    adapter_kind: str
    capabilities: dict[str, object]
    workspace: str
    tools: dict[str, object]
    web_search: dict[str, object] | None
    web_fetch: dict[str, object] | None
    runtime: dict[str, object]
    context_budget: dict[str, object]


@dataclass(frozen=True)
class RunResult:
    """Successful task result and the effective inputs used to produce it."""

    spec: RunSpec
    turn: TurnResult


def read_task_prompt(inline: str | None, path: Path | None) -> TaskPrompt:
    """Loads exactly one prompt, preserving its original text.

    Args:
        inline: Inline task text, absent for file input.
        path: UTF-8 task file, absent for inline input.

    Returns:
        Nonempty prompt and its source.

    Raises:
        GearError: If selection, file access, encoding or prompt content is invalid.
    """
    if (inline is None) == (path is None):
        raise GearError('prompt_selection_invalid', 'Select exactly one prompt source.',
                        'headless', True, {})
    if path is not None:
        try:
            text = path.read_text(encoding='utf-8')
        except (OSError, UnicodeError) as exc:
            raise GearError('prompt_read_failed', f'Cannot read UTF-8 prompt file: {path}',
                            'headless', True, {'path': str(path)}) from exc
        prompt = TaskPrompt(text, str(path.resolve()))
    else:
        assert inline is not None
        prompt = TaskPrompt(inline, 'inline')
    if not prompt.text.strip():
        raise GearError('prompt_empty', 'Task prompt must contain non-whitespace text.',
                        'headless', True, {})
    return prompt


def describe_run(agent: AgentRuntime, session_id: str, prompt: TaskPrompt) -> RunSpec:
    """Projects effective settings into credential-free, JSON-serializable metadata.

    Args:
        agent: Shared production services and effective settings.
        session_id: Fresh session identifier.
        prompt: Selected task and source.

    Returns:
        Inputs for inspecting this run without serializing live service objects.
    """
    config = agent.config
    endpoint = urlsplit(config.model.url)
    public_endpoint = urlunsplit((endpoint.scheme, endpoint.netloc.rsplit('@', 1)[-1],
                                 endpoint.path, endpoint.query, ''))
    runtime = asdict(agent.runtime)
    runtime['workdir'] = str(agent.workspace)
    runtime['session_dir'] = str(agent.runtime.session_dir.resolve())
    web_search = None
    if config.web_search is not None:
        web_search = asdict(config.web_search)
        del web_search['api_key']
    web_fetch = None
    if config.web_fetch is not None:
        web_fetch = asdict(config.web_fetch)
        del web_fetch['api_key']
    return RunSpec(
        session_id=session_id,
        prompt_source=prompt.source,
        model={
            'model': config.model.model,
            'endpoint_identity': endpoint_identity(public_endpoint, config.model.api_key),
            'stream': config.model.stream,
            'reasoning_replay': config.model.reasoning_replay.value,
        },
        adapter_kind=type(agent.adapter).__name__,
        capabilities=asdict(agent.adapter.capabilities),
        workspace=str(agent.workspace),
        tools=asdict(config.tool),
        web_search=web_search,
        web_fetch=web_fetch,
        runtime=runtime,
        context_budget=asdict(config.context_budget),
    )


def run_task(agent: AgentRuntime, session_id: str, prompt: TaskPrompt) -> RunResult:
    """Executes one task using the normal agent loop and session semantics.

    Args:
        agent: Configured production harness with a headless event sink.
        session_id: Fresh session identifier published by the caller.
        prompt: Task text and source.

    Returns:
        Canonical completed answer with effective execution inputs.

    Raises:
        GearError: If the model, tools, repository context or budget fails.
        OSError: If local session persistence fails.
    """
    spec = describe_run(agent, session_id, prompt)
    runtime = agent.runtime
    turn = agent.loop.run_turn(
        session_id, prompt.text, runtime.max_iterations, runtime.model_timeout_seconds,
        runtime.model_stream_idle_timeout_seconds,
    )
    return RunResult(spec, turn)


def validate_headless_runtime(agent: AgentRuntime) -> None:
    """Rejects invalid effective runtime settings before starting a task.

    Args:
        agent: Services composed from configuration and CLI overrides.

    Raises:
        GearError: If the workspace, request endpoint or runtime limits are invalid.
    """
    if not agent.workspace.is_dir():
        raise GearError('workspace_invalid', f'Workspace is not a directory: {agent.workspace}',
                        'runtime', True, {})
    for name, value in (
        ('max_iterations', agent.runtime.max_iterations),
        ('model_timeout_seconds', agent.runtime.model_timeout_seconds),
    ):
        if value < 1:
            raise GearError('runtime_value_invalid', f'{name} must be at least 1.',
                            'runtime', True, {'key': name})
    try:
        endpoint = urlsplit(agent.config.model.url)
        valid_endpoint = endpoint.scheme in ('http', 'https') and bool(endpoint.hostname)
        endpoint.port
    except ValueError as exc:
        raise GearError('model_url_invalid', 'model.url is not a valid HTTP(S) endpoint.',
                        'config', True, {}) from exc
    if not valid_endpoint:
        raise GearError('model_url_invalid', 'model.url must be an absolute HTTP(S) endpoint.',
                        'config', True, {})
