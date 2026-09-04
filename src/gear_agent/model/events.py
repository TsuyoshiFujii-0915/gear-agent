from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias


@dataclass(frozen=True)
class ModelTextDelta:
    """Provider-neutral output text progress."""

    delta: str


@dataclass(frozen=True)
class ModelReasoningDelta:
    """Provider-neutral reasoning text progress."""

    delta: str


@dataclass(frozen=True)
class ModelFunctionCallArgumentsDelta:
    """Provider-neutral function argument progress."""

    item_id: str
    output_index: int
    delta: str


@dataclass(frozen=True)
class ModelOutputItemCompleted:
    """Provider-neutral completed output item progress."""

    output_index: int
    item: dict[str, Any]


ModelProgressEvent: TypeAlias = (
    ModelTextDelta
    | ModelReasoningDelta
    | ModelFunctionCallArgumentsDelta
    | ModelOutputItemCompleted
)


class ModelProgressEventSink(Protocol):
    """Receives provider-neutral model progress events."""

    def publish(self, event: ModelProgressEvent) -> None:
        """Publishes model progress.

        Args:
            event: Provider-neutral progress event.
        """

        pass


class SilentModelProgressEventSink:
    """Model progress sink used when no upper layer consumes deltas."""

    def publish(self, event: ModelProgressEvent) -> None:
        """Ignores model progress.

        Args:
            event: Provider-neutral progress event.
        """

        pass
