"""Monotonic client observations at the decoded SSE receive boundary."""

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from gear_agent.config import ModelConfig
from gear_agent.model.events import ModelProgressEvent, ModelProgressEventSink
from gear_agent.model.replay import endpoint_identity
from gear_agent.model.statistics import ResponseStatistics


@dataclass(frozen=True)
class RequestTarget:
    model_identity: str
    endpoint_identity: str
    stream: bool

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'RequestTarget':
        """Fingerprints the model and credential-free endpoint without a lookup."""
        url = urlsplit(config.url)
        public = urlunsplit((url.scheme, url.netloc.rsplit('@', 1)[-1], url.path, '', ''))
        return cls(sha256(config.model.encode()).hexdigest(), endpoint_identity(public, config.api_key), config.stream)


@dataclass(frozen=True)
class ClientTime:
    value: float | None
    basis: str | None
    reason: str | None
    unit: Literal['s'] = 's'
    origin: Literal['client_measured'] = 'client_measured'


@dataclass(frozen=True)
class ClientTiming:
    total: ClientTime
    first_content: ClientTime
    first_answer: ClientTime


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and len(value) > 0


def _parts(parts: object) -> tuple[bool, bool]:
    content = answer = False
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            kind = part.get('type')
            if kind == 'refusal':
                content |= _nonempty(part.get('refusal'))
            elif _nonempty(part.get('text')):
                answer |= kind == 'output_text'
                content |= kind in ('output_text', 'reasoning_text', 'summary_text')
    return content, answer


def _items(items: object) -> tuple[bool, bool]:
    content = answer = False
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            kind = item.get('type')
            if kind == 'function_call':
                content |= _nonempty(item.get('arguments'))
            elif kind in ('reasoning', 'message'):
                has_content, has_answer = _parts(item.get('content'))
                content |= has_content
                answer |= has_answer and kind == 'message'
                if kind == 'reasoning':
                    content |= _parts(item.get('summary'))[0]
    return content, answer


_DONE_FIELDS = {
    'response.output_text': 'text', 'response.reasoning_text': 'text',
    'response.reasoning_summary_text': 'text', 'response.function_call_arguments': 'arguments',
    'response.refusal': 'refusal',
}


def _arrival(kind: str, event: dict[str, Any]) -> tuple[bool, bool]:
    if kind in ('response.completed', 'response.failed', 'response.incomplete'):
        response = event.get('response')
        return _items(response.get('output')) if isinstance(response, dict) else (False, False)
    if kind in ('response.output_item.added', 'response.output_item.done'):
        return _items([event.get('item')])
    if kind in ('response.content_part.added', 'response.content_part.done',
                'response.reasoning_summary_part.added', 'response.reasoning_summary_part.done'):
        return _parts([event.get('part')])
    name, _, phase = kind.rpartition('.')
    if phase not in ('delta', 'done') or name not in _DONE_FIELDS:
        return False, False
    field = 'delta' if phase == 'delta' else _DONE_FIELDS[name]
    content = _nonempty(event.get(field))
    return content, content and name == 'response.output_text'


class TimedProgressSink:
    """Times first arrivals before forwarding progress to synchronous consumers."""

    def __init__(self, progress: ModelProgressEventSink, clock: Callable[[], float],
                 on_dispatch: Callable[[RequestTarget], None]) -> None:
        self._progress = progress
        self._clock = clock
        self._on_dispatch = on_dispatch
        self._start: float | None = None
        self._content: ClientTime | None = None
        self._answer: ClientTime | None = None
        self.terminal_statistics: ResponseStatistics | None = None

    def dispatch(self, target: RequestTarget) -> None:
        """Starts the clock after publishing the start and immediately before transport."""
        if self._start is not None:
            raise RuntimeError('A measured model request was dispatched twice.')
        self._on_dispatch(target)
        self._start = self._clock()

    def _elapsed(self) -> float:
        if self._start is None:
            raise RuntimeError('Model timing requires a dispatched request.')
        return self._clock() - self._start

    def observe(self, event: dict[str, Any]) -> None:
        kind = event.get('type')
        if self._answer is not None or not isinstance(kind, str):
            return
        content, answer = _arrival(kind, event)
        if answer or (content and self._content is None):
            arrival = ClientTime(self._elapsed(), kind, None)
            if self._content is None:
                self._content = arrival
            if answer:
                self._answer = arrival

    def publish(self, event: ModelProgressEvent) -> None:
        self._progress.publish(event)

    def finish(
        self, stream: bool,
        missing_stream_reason: Literal['not_applicable', 'not_observed', 'interrupted_before_report'],
    ) -> ClientTiming:
        elapsed = ClientTime(self._elapsed(), 'dispatch_to_return', None)
        reason = missing_stream_reason if stream else 'non_streaming'
        missing = ClientTime(None, None, reason)
        return ClientTiming(elapsed, self._content or missing, self._answer or missing)
