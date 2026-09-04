from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gear_agent.agent.events import (
    AgentLoopEventSink,
    ModelRequestStarted,
    ReasoningReplayEvaluated,
    ToolUseFinished,
    ToolUseStarted,
)
from gear_agent.agent.history import build_model_input
from gear_agent.config import ModelConfig, ReasoningReplayMode
from gear_agent.errors import GearError, gear_error
from gear_agent.model.client import ModelClient
from gear_agent.model.responses import (
    extract_function_calls,
    extract_output_text,
    function_call_output_item,
)
from gear_agent.model.replay import (
    ReasoningReplayDiagnostic,
    ReasoningReplayPolicy,
    ReplayedOutput,
    model_response_event_payload,
    reasoning_replay_policy,
    replay_output_items,
)
from gear_agent.store.base import ContextStore
from gear_agent.tools.base import Tool
from gear_agent.tools.registry import ToolRegistry


AGENT_INSTRUCTIONS = "\n".join(
    [
        "You are Gear Agent, a coding assistant operating inside one explicit workspace.",
        "Use workspace-relative paths for every tool argument that accepts a path.",
        "The workspace root is '.'. Use workdir='.' when running shell commands at the root.",
        "Absolute paths such as /testbed, /workspace, or host filesystem paths are invalid.",
        "When a tool returns an error, correct the tool arguments or explain the blocker.",
    ]
)

FINALIZATION_RETRY_INSTRUCTION = "\n".join(
    [
        "The previous response did not contain a user-facing final answer as output_text.",
        "If no tool call is needed, return the final answer for the user as output_text now.",
    ]
)


@dataclass(frozen=True)
class TurnResult:
    """Result of one user turn.

    Attributes:
        final_text: Final assistant text.
        iterations: Number of model calls made.
    """

    final_text: str
    iterations: int


class AgentLoop:
    """Coordinates model responses and tool execution."""

    def __init__(
        self,
        client: ModelClient,
        config: ModelConfig,
        tools: list[Tool],
        store: ContextStore,
        event_sink: AgentLoopEventSink,
    ) -> None:
        self._client = client
        self._config = config
        self._replay_policy = reasoning_replay_policy(config)
        self._registry = ToolRegistry(tools)
        self._store = store
        self._event_sink = event_sink

    def run_turn(
        self,
        session_id: str,
        user_text: str,
        max_iterations: int,
        timeout_seconds: int,
    ) -> TurnResult:
        """Runs one user turn until final text or explicit failure.

        Args:
            session_id: Session identifier.
            user_text: User message.
            max_iterations: Maximum model calls for this turn.
            timeout_seconds: HTTP timeout for each model call.

        Returns:
            Turn result.
        """

        if max_iterations < 1:
            raise gear_error(
                "iteration_limit_invalid",
                "max_iterations must be at least 1.",
                "agent_loop",
                True,
                {"max_iterations": max_iterations},
            )
        model_input = build_model_input(
            self._store.load(session_id),
            user_text,
            self._replay_policy,
        )
        input_items = model_input.items
        self._publish_replay_diagnostic(session_id, model_input.diagnostic)
        self._store.append(session_id, "user_input", {"text": user_text})
        tools = self._registry.schemas()
        finalization_retry_used = False
        pending_replay_diagnostic: ReasoningReplayDiagnostic | None = None

        for iteration in range(1, max_iterations + 1):
            if pending_replay_diagnostic is not None:
                self._publish_replay_diagnostic(
                    session_id,
                    pending_replay_diagnostic,
                )
                pending_replay_diagnostic = None
            self._event_sink.publish(
                ModelRequestStarted(session_id=session_id, iteration=iteration)
            )
            response = self._client.create_response(
                self._config,
                input_items,
                tools,
                AGENT_INSTRUCTIONS,
                timeout_seconds,
            )
            self._store.append(
                session_id,
                "model_response",
                model_response_event_payload(response, self._replay_policy),
            )
            replayed_output = _current_response_output(
                response,
                self._replay_policy,
            )
            output_items = replayed_output.items
            function_calls = extract_function_calls(response)
            if len(function_calls) == 0:
                final_text = extract_output_text(response)
                if _has_final_text(final_text):
                    self._store.append(session_id, "assistant_message", {"text": final_text})
                    return TurnResult(final_text=final_text, iterations=iteration)
                if not finalization_retry_used and iteration < max_iterations:
                    finalization_retry_used = True
                    input_items.extend(output_items)
                    input_items.append(
                        {
                            "role": "user",
                            "content": FINALIZATION_RETRY_INSTRUCTION,
                        }
                    )
                    pending_replay_diagnostic = _nonempty_replay_diagnostic(
                        replayed_output.diagnostic
                    )
                    continue
                raise gear_error(
                    "final_text_missing",
                    "Model returned neither a tool call nor a final output_text.",
                    "agent_loop",
                    True,
                    {"iteration": iteration, "retry_used": finalization_retry_used},
                )

            input_items.extend(output_items)
            pending_replay_diagnostic = _nonempty_replay_diagnostic(
                replayed_output.diagnostic
            )
            for function_call in function_calls:
                self._store.append(
                    session_id,
                    "tool_call",
                    {
                        "call_id": function_call.call_id,
                        "iteration": iteration,
                        "name": function_call.name,
                        "arguments": function_call.arguments,
                    },
                )
                self._event_sink.publish(
                    ToolUseStarted(
                        session_id=session_id,
                        iteration=iteration,
                        call_id=function_call.call_id,
                        name=function_call.name,
                        arguments=function_call.arguments,
                    )
                )
                try:
                    tool_result = self._registry.run(function_call.name, function_call.arguments)
                except GearError as exc:
                    if not exc.recoverable:
                        raise
                    tool_result = _recoverable_tool_error_result(exc)
                self._store.append(
                    session_id,
                    "tool_result",
                    {
                        "call_id": function_call.call_id,
                        "iteration": iteration,
                        "name": function_call.name,
                        "result": tool_result,
                    },
                )
                self._event_sink.publish(
                    ToolUseFinished(
                        session_id=session_id,
                        iteration=iteration,
                        call_id=function_call.call_id,
                        name=function_call.name,
                        result=tool_result,
                    )
                )
                input_items.append(function_call_output_item(function_call.call_id, tool_result))

        raise gear_error(
            "iteration_limit_reached",
            "Model did not produce a final answer before max_iterations.",
            "agent_loop",
            True,
            {"max_iterations": max_iterations},
        )

    def _publish_replay_diagnostic(
        self,
        session_id: str,
        diagnostic: ReasoningReplayDiagnostic,
    ) -> None:
        self._event_sink.publish(
            ReasoningReplayEvaluated(
                session_id=session_id,
                mode=self._replay_policy.mode,
                reused_encrypted_items=diagnostic.reused_encrypted_items,
                dropped_disabled_items=diagnostic.dropped_disabled_items,
                dropped_incompatible_scope_items=(
                    diagnostic.dropped_incompatible_scope_items
                ),
                dropped_missing_scope_items=diagnostic.dropped_missing_scope_items,
            )
        )


def _current_response_output(
    response: dict[str, Any],
    replay_policy: ReasoningReplayPolicy,
) -> ReplayedOutput:
    output_items = _output_items(response)
    source_scope = None
    if replay_policy.mode is ReasoningReplayMode.ENCRYPTED:
        source_scope = replay_policy.current_scope
    return replay_output_items(output_items, source_scope, replay_policy)


def _nonempty_replay_diagnostic(
    diagnostic: ReasoningReplayDiagnostic,
) -> ReasoningReplayDiagnostic | None:
    handled_items = (
        diagnostic.reused_encrypted_items + diagnostic.dropped_encrypted_items
    )
    if handled_items == 0:
        return None
    return diagnostic


def _output_items(response: dict[str, Any]) -> list[dict[str, Any]]:
    output = response.get("output")
    if not isinstance(output, list):
        raise gear_error(
            "response_shape_invalid",
            "Response output is not a list.",
            "agent_loop",
            True,
            {},
        )
    items: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            raise gear_error(
                "response_shape_invalid",
                "Response output item is not an object.",
                "agent_loop",
                True,
                {},
            )
        items.append(item)
    return items


def _has_final_text(text: str) -> bool:
    return text.strip() != ""


def _recoverable_tool_error_result(error: GearError) -> dict[str, object]:
    return {
        "error": {
            "type": error.error_type,
            "message": error.message,
            "origin": error.origin,
            "details": error.details,
        }
    }
