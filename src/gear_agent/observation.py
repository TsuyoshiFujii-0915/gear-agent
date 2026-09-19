from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any, Protocol

from gear_agent.context_budget import ContextRequest
from gear_agent.errors import GearError
from gear_agent.model.adapter import ModelAdapter, ModelResponse
from gear_agent.model.events import ModelProgressEventSink
from gear_agent.tools.registry import ToolRegistry


class RunObserver(Protocol):
    """Receives run-only observations without coupling execution to persistence."""

    def record(self, kind: str, payload: dict[str, Any]) -> None:
        """Receives one complete, non-streaming observation."""
        ...


def error_identity(error: BaseException) -> dict[str, str]:
    """Projects failure identity without remote bodies or arbitrary details.

    Args:
        error: Original execution or persistence failure.

    Returns:
        Stable error identity and a message suitable for subsequent redaction.
    """
    if isinstance(error, GearError):
        return {'type': error.error_type, 'origin': error.origin, 'message': error.message}
    return {'type': type(error).__name__, 'origin': type(error).__module__,
            'message': 'Execution stopped with an unstructured exception.'}


def request_model(
    adapter: ModelAdapter, request: ContextRequest, timeout_seconds: int,
    stream_idle_timeout_seconds: int | None, progress_sink: ModelProgressEventSink,
    observer: RunObserver | None, purpose: str,
) -> ModelResponse:
    """Measures one dispatched request, including compaction and failed requests.

    Args:
        adapter: Configured model boundary.
        request: Exact effective request.
        timeout_seconds: Request timeout.
        stream_idle_timeout_seconds: Streaming idle timeout.
        progress_sink: Existing presentation sink.
        observer: Absent for callers not collecting run observations.
        purpose: Agent or compaction request identity.

    Returns:
        Original canonical response.
    """
    if observer is None:
        return adapter.create_response(request.input_value, request.tools, request.instructions,
                                       timeout_seconds, stream_idle_timeout_seconds, progress_sink)
    observer.record('model_request_started', {'purpose': purpose})
    started = perf_counter()
    try:
        response = adapter.create_response(request.input_value, request.tools, request.instructions,
                                           timeout_seconds, stream_idle_timeout_seconds, progress_sink)
    except BaseException as error:
        observer.record('model_request_finished', {
            'purpose': purpose, 'duration_seconds': perf_counter() - started,
            'usage': None, 'error': error_identity(error),
        })
        raise
    duration = perf_counter() - started
    observer.record('model_request_finished', {
        'purpose': purpose, 'duration_seconds': duration, 'usage': None, 'error': None,
    })
    return response


def record_model_usage(response: ModelResponse, observer: RunObserver | None) -> None:
    """Observes usage after the caller has persisted its canonical model response.

    Args:
        response: Received model result, already persisted when part of a turn.
        observer: Absent for callers not collecting run observations.

    Raises:
        GearError: If provider-reported usage is invalid; canonical history remains.
    """
    if observer is not None:
        observer.record('model_usage', {'usage': asdict(response.usage)})


def execute_tool(
    registry: ToolRegistry, name: str, arguments: dict[str, object], call_id: str,
    observer: RunObserver | None,
) -> dict[str, object]:
    """Measures tool execution before the loop applies recoverable-error policy.

    Args:
        registry: Existing tool dispatch registry.
        name: Tool name.
        arguments: Canonical parsed arguments.
        call_id: Model tool-call identifier.
        observer: Absent for callers not collecting run observations.

    Returns:
        Unchanged tool result.
    """
    if observer is None:
        return registry.run(name, arguments)
    observer.record('tool_started', {'name': name, 'call_id': call_id})
    started = perf_counter()
    try:
        result = registry.run(name, arguments)
    except BaseException as error:
        observer.record('tool_finished', {
            'name': name, 'call_id': call_id, 'duration_seconds': perf_counter() - started,
            'successful': False, 'error': error_identity(error),
        })
        raise
    observer.record('tool_finished', {
        'name': name, 'call_id': call_id, 'duration_seconds': perf_counter() - started,
        'successful': 'error' not in result and result.get('exit_code', 0) == 0
                      and not result.get('timed_out', False),
        'error': None,
    })
    return result
