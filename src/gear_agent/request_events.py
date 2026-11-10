"""Content-free request lifecycle shared by presentation, sessions and runs."""

from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from gear_agent.model.arrival import ClientTiming, RequestTarget
from gear_agent.model.statistics import ResponseStatistics

Purpose = Literal['agent', 'manual_summary', 'automatic_summary', 'jev_summary_fallback']
RequestStatus = Literal['success', 'incomplete', 'provider_failure', 'transport_error', 'timeout',
                        'eof', 'cancelled', 'invalid_response', 'execution_error']


@dataclass(frozen=True)
class RequestScope:
    session_id: str
    turn_id: str | None
    iteration: int | None
    purpose: Purpose


@dataclass(frozen=True)
class RequestStarted:
    request_id: str
    scope: RequestScope
    target: RequestTarget
    started_at: str

    def payload(self) -> dict[str, Any]:
        return {'schema_version': 1, 'request_id': self.request_id, 'started_at': self.started_at,
                **asdict(self.scope), **asdict(self.target)}


@dataclass(frozen=True)
class RequestError:
    type: str
    origin: str


@dataclass(frozen=True)
class RequestFinished:
    request: RequestStarted
    status: RequestStatus
    statistics: ResponseStatistics | None
    client: ClientTiming
    error: RequestError | None

    def payload(self) -> dict[str, Any]:
        statistics = self.statistics
        usage = None
        if statistics is not None and self.status == 'success':
            # The v1 usage view contains only counts reported by the provider.
            core = {'input_tokens': statistics.input_tokens, 'output_tokens': statistics.output_tokens,
                    'total_tokens': statistics.total_tokens}
            if all(metric.availability != 'invalid' for metric in core.values()):
                usage = {name: metric.value if metric.origin == 'server_reported' else None
                         for name, metric in core.items()}
        return {**self.request.payload(), 'status': self.status, 'client': asdict(self.client),
                'duration_seconds': self.client.total.value, 'usage': usage,
                'error': asdict(self.error) if self.error is not None else None,
                'statistics': asdict(statistics) if statistics is not None else None,
                'statistics_reason': 'interrupted_before_report' if statistics is None else None}


@dataclass(frozen=True)
class RequestLogWriteFailed:
    request_id: str
    session_id: str


class RequestEventSink(Protocol):
    def publish(self, event: RequestStarted | RequestFinished) -> None: ...
