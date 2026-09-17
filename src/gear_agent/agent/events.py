from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

from gear_agent.config import ReasoningReplayMode
from gear_agent.context_budget import ContextBudgetDiagnostic


@dataclass(frozen=True)
class ContextBudgetEvaluated:
    """Content-free accounting at a request or compaction decision boundary.

    Attributes:
        session_id: Session identifier.
        iteration: Agent iteration whose request is being prepared.
        phase: Whether this measures the original, summary, or rebuilt request.
        diagnostic: Component estimates and effective limit.
        auto_compaction_triggered: Whether this decision triggered compaction.
        failed: Whether the estimate prevents dispatch.
    """

    session_id: str
    iteration: int
    phase: Literal['before', 'compaction', 'after']
    diagnostic: ContextBudgetDiagnostic
    auto_compaction_triggered: bool
    failed: bool


@dataclass(frozen=True)
class ModelRequestStarted:
    """Event published before a model request starts.

    Attributes:
        session_id: Session identifier.
        iteration: Agent loop iteration number.
    """

    session_id: str
    iteration: int


@dataclass(frozen=True)
class ModelTextDelta:
    """Displayable assistant text received during a model request.

    Attributes:
        session_id: Session identifier.
        iteration: Agent loop iteration number.
        delta: Assistant text fragment in arrival order.
    """

    session_id: str
    iteration: int
    delta: str


@dataclass(frozen=True)
class ModelReasoningSummaryDelta:
    """Provider-exposed reasoning summary received during a model request.

    Attributes:
        session_id: Session identifier.
        iteration: Agent loop iteration number.
        delta: Public reasoning summary fragment in arrival order.
    """

    session_id: str
    iteration: int
    delta: str


@dataclass(frozen=True)
class ReasoningReplayEvaluated:
    """Structured diagnostic for active opaque reasoning history.

    Attributes:
        session_id: Session identifier.
        mode: Active encrypted reasoning replay mode.
        reused_encrypted_items: Compatible opaque items retained.
        dropped_disabled_items: Opaque items removed because replay is disabled.
        dropped_incompatible_scope_items: Opaque items removed on scope mismatch.
        dropped_missing_scope_items: Opaque items removed because metadata is absent.
    """

    session_id: str
    mode: ReasoningReplayMode
    reused_encrypted_items: int
    dropped_disabled_items: int
    dropped_incompatible_scope_items: int
    dropped_missing_scope_items: int


@dataclass(frozen=True)
class ToolUseStarted:
    """Event published before a tool starts running.

    Attributes:
        session_id: Session identifier.
        iteration: Agent loop iteration number.
        call_id: Responses API tool call identifier.
        name: Tool name.
        arguments: Parsed tool arguments.
    """

    session_id: str
    iteration: int
    call_id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ToolUseFinished:
    """Event published after a tool returns a result.

    Attributes:
        session_id: Session identifier.
        iteration: Agent loop iteration number.
        call_id: Responses API tool call identifier.
        name: Tool name.
        result: Tool result returned to the model.
    """

    session_id: str
    iteration: int
    call_id: str
    name: str
    result: dict[str, object]


AgentLoopEvent: TypeAlias = (
    ModelRequestStarted
    | ContextBudgetEvaluated
    | ModelTextDelta
    | ModelReasoningSummaryDelta
    | ReasoningReplayEvaluated
    | ToolUseStarted
    | ToolUseFinished
)


class AgentLoopEventSink(Protocol):
    """Receives explicit agent loop progress events."""

    def publish(self, event: AgentLoopEvent) -> None:
        """Publishes an agent loop progress event.

        Args:
            event: Agent loop event.
        """


class SilentAgentLoopEventSink:
    """Agent loop event sink that intentionally ignores events."""

    def publish(self, event: AgentLoopEvent) -> None:
        """Ignores an agent loop progress event.

        Args:
            event: Agent loop event.
        """
