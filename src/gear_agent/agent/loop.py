from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gear_agent.agent.events import (
    AgentLoopEventSink,
    ModelRequestStarted,
    ToolUseFinished,
    ToolUseStarted,
)
from gear_agent.agent.history import build_model_input
from gear_agent.config import ModelConfig
from gear_agent.errors import GearError, gear_error
from gear_agent.model.client import ModelClient
from gear_agent.model.responses import (
    extract_function_calls,
    extract_output_text,
    function_call_output_item,
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
        input_items = build_model_input(self._store.load(session_id), user_text)
        self._store.append(session_id, "user_input", {"text": user_text})
        tools = self._registry.schemas()
        finalization_retry_used = False

        for iteration in range(1, max_iterations + 1):
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
            self._store.append(session_id, "model_response", response)
            function_calls = extract_function_calls(response)
            if len(function_calls) == 0:
                final_text = extract_output_text(response)
                if _has_final_text(final_text):
                    self._store.append(session_id, "assistant_message", {"text": final_text})
                    return TurnResult(final_text=final_text, iterations=iteration)
                if not finalization_retry_used and iteration < max_iterations:
                    finalization_retry_used = True
                    input_items.extend(_output_items(response))
                    input_items.append(
                        {
                            "role": "user",
                            "content": FINALIZATION_RETRY_INSTRUCTION,
                        }
                    )
                    continue
                raise gear_error(
                    "final_text_missing",
                    "Model returned neither a tool call nor a final output_text.",
                    "agent_loop",
                    True,
                    {"iteration": iteration, "retry_used": finalization_retry_used},
                )

            input_items.extend(_output_items(response))
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


def _output_items(response: dict[str, Any]) -> list[object]:
    output = response.get("output")
    if not isinstance(output, list):
        raise gear_error(
            "response_shape_invalid",
            "Response output is not a list.",
            "agent_loop",
            True,
            {},
        )
    return output


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
