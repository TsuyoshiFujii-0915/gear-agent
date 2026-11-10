from __future__ import annotations

from asyncio import CancelledError
from concurrent.futures import CancelledError as FutureCancelledError
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from gear_agent.context_budget import ContextRequest
from gear_agent.errors import GearError
from gear_agent.model.adapter import ModelAdapter, ModelResponse
from gear_agent.model.arrival import RequestTarget, TimedProgressSink
from gear_agent.model.events import ModelProgressEventSink
from gear_agent.model.responses import ResponseFailure
from gear_agent.request_events import (
    RequestError, RequestEventSink, RequestFinished, RequestScope, RequestStarted, RequestStatus,
)
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


_ERROR_STATUS: dict[str, RequestStatus] = {
    'response_incomplete': 'incomplete',
    'response_failed': 'provider_failure', 'response_stream_error': 'provider_failure',
    'http_status_error': 'provider_failure',
    'http_timeout': 'timeout', 'http_stream_idle_timeout': 'timeout',
    'response_stream_terminated': 'eof',
    'http_request_failed': 'transport_error', 'http_stream_failed': 'transport_error',
    'response_status_invalid': 'invalid_response', 'response_stream_event_invalid': 'invalid_response',
    'json_parse_failed': 'invalid_response', 'json_shape_invalid': 'invalid_response',
    'http_sse_invalid': 'invalid_response',
}


def _request_failure(error: BaseException) -> tuple[RequestStatus, RequestError]:
    if isinstance(error, (KeyboardInterrupt, CancelledError, FutureCancelledError)):
        return 'cancelled', RequestError('cancelled', 'model_request')
    if isinstance(error, GearError) and error.error_type in _ERROR_STATUS:
        status = _ERROR_STATUS[error.error_type]
        if error.error_type == 'response_status_invalid' and error.details.get('status') == 'cancelled':
            status = 'cancelled'
        origin = error.origin if error.origin in ('responses_stream', 'model_client') else 'model_request'
        return status, RequestError(error.error_type, origin)
    return 'execution_error', RequestError('execution_error', 'model_request')


def request_model(
    adapter: ModelAdapter, request: ContextRequest, timeout_seconds: int,
    stream_idle_timeout_seconds: int | None, progress_sink: ModelProgressEventSink,
    observer: RunObserver | None, lifecycle: RequestEventSink | None, scope: RequestScope,
) -> ModelResponse:
    """Measures a single adapter dispatch and publishes its terminal observation."""
    if lifecycle is None and observer is None:
        return adapter.create_response(request.input_value, request.tools, request.instructions,
                                       timeout_seconds, stream_idle_timeout_seconds, progress_sink)
    started: RequestStarted | None = None

    def dispatch(target: RequestTarget) -> None:
        nonlocal started
        event = RequestStarted(str(uuid4()), scope, target, datetime.now(timezone.utc).isoformat())
        if observer is not None:
            observer.record('model_request_started', event.payload())
        if lifecycle is not None:
            lifecycle.publish(event)
        started = event

    def publish_finished(finished: RequestFinished) -> None:
        if observer is not None:
            observer.record('model_request_finished', finished.payload())
        if lifecycle is not None:
            lifecycle.publish(finished)

    timing = TimedProgressSink(progress_sink, perf_counter, dispatch)
    try:
        response = adapter.create_response(request.input_value, request.tools, request.instructions,
                                           timeout_seconds, stream_idle_timeout_seconds, timing)
    except BaseException as error:
        if started is None:
            raise
        statistics = error.statistics if isinstance(error, ResponseFailure) else timing.terminal_statistics
        client = timing.finish(started.target.stream, 'interrupted_before_report' if statistics is None else 'not_observed')
        status, identity = _request_failure(error)
        publish_finished(RequestFinished(started, status, statistics, client, identity))
        raise
    if started is None:
        raise RuntimeError('Model adapter returned a response without reporting dispatch.')
    client = timing.finish(started.target.stream, 'not_applicable')
    publish_finished(RequestFinished(started, 'success', response.statistics, client, None))
    return response


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
    observer.record('tool_started', {'name': name, 'call_id': call_id, **registry.web_observation(name, None)})
    started = perf_counter()
    try:
        result = registry.run(name, arguments)
    except BaseException as error:
        observer.record('tool_finished', {
            'name': name, 'call_id': call_id, 'duration_seconds': perf_counter() - started,
            'successful': False, 'error': error_identity(error),
            **registry.web_observation(name, None),
        })
        raise
    observer.record('tool_finished', {
        'name': name, 'call_id': call_id, 'duration_seconds': perf_counter() - started,
        'successful': 'error' not in result and result.get('exit_code', 0) == 0
                      and not result.get('timed_out', False),
        'error': None, **registry.web_observation(name, result),
    })
    return result
