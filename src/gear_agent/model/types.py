from dataclasses import dataclass

from gear_agent.model.replay import ReasoningReplayDiagnostic


@dataclass(frozen=True)
class FunctionCall:
    """Provider-neutral tool invocation.

    Attributes:
        call_id: Tool call identifier used to correlate results.
        name: Tool name.
        arguments: Parsed JSON arguments.
    """

    call_id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ModelHistory:
    """Model-visible input and opaque reasoning replay diagnostics.

    Attributes:
        items: Canonical continuation items.
        diagnostic: Counts describing opaque state reuse and removal.
    """

    items: list[object]
    diagnostic: ReasoningReplayDiagnostic
