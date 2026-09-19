from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from gear_agent.model.events import ModelProgressEventSink
from gear_agent.model.replay import ReasoningReplayPolicy, ReplayedOutput
from gear_agent.model.types import FunctionCall, ModelHistory, ModelUsage


@dataclass(frozen=True)
class ModelCapabilities:
    """Implemented operations, subject to remote model rejection.

    These describe adapter support, not which options a request enables.
    Serialize with dataclasses.asdict; no connection details are included.
    """

    streaming: bool
    opaque_reasoning_replay: bool
    textual_compaction: bool
    native_compaction: bool
    function_calling: bool


class ModelResponse(Protocol):
    """Completed result with canonical persistence and semantic accessors."""

    @property
    def usage(self) -> ModelUsage:
        """Returns provider-reported token categories; absent categories are null."""
        ...

    @property
    def persisted_payload(self) -> dict[str, Any]:
        """Returns the canonical session event payload."""
        ...

    @property
    def replayed_output(self) -> ReplayedOutput:
        """Returns complete continuation items with replay diagnostics."""
        ...

    @property
    def function_calls(self) -> list[FunctionCall]:
        """Returns validated tool calls in execution order."""
        ...

    @property
    def text(self) -> str:
        """Returns user-facing output text."""
        ...


class ModelAdapter(Protocol):
    """Configured model execution over manually managed canonical history."""

    @property
    def capabilities(self) -> ModelCapabilities:
        """Returns implemented operations without contacting the provider."""
        ...

    @property
    def replay_policy(self) -> ReasoningReplayPolicy:
        """Returns the active opaque-state replay policy."""
        ...

    def prepare_history(
        self, events: list[dict[str, Any]], user_text: str,
    ) -> ModelHistory:
        """Builds continuation history from stored events and user text."""
        ...

    def create_response(
        self, input_value: object, tools: list[dict[str, object]],
        instructions: str, timeout_seconds: float,
        stream_idle_timeout_seconds: float | None,
        progress_sink: ModelProgressEventSink,
    ) -> ModelResponse:
        """Executes one response, propagating model errors explicitly."""
        ...

    def tool_result_item(self, call_id: str, result: dict[str, object]) -> object:
        """Encodes a tool result for continuation."""
        ...

    def user_message_item(self, text: str) -> object:
        """Encodes an additional user instruction for continuation."""
        ...
