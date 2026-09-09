from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gear_agent.agent.history import build_model_input
from gear_agent.model.types import FunctionCall, ModelHistory
from gear_agent.config import ModelConfig, ReasoningReplayMode
from gear_agent.errors import gear_error
from gear_agent.model.adapter import ModelCapabilities
from gear_agent.model.client import ModelClient
from gear_agent.model.replay import (
    ReasoningReplayPolicy, ReplayedOutput, model_response_event_payload,
    reasoning_replay_policy, replay_output_items,
)
from gear_agent.model.responses import (
    extract_function_calls, extract_output_text, function_call_output_item,
)


@dataclass(frozen=True)
class ResponsesModelResponse:
    """Canonical Responses result with model-owned, on-demand decoding."""

    response: dict[str, Any]
    policy: ReasoningReplayPolicy

    @property
    def persisted_payload(self) -> dict[str, Any]:
        """Returns the unchanged legacy payload or scoped replay envelope."""
        return model_response_event_payload(self.response, self.policy)

    @property
    def replayed_output(self) -> ReplayedOutput:
        """Validates and prepares complete output items for continuation."""
        output = self.response.get('output')
        if not isinstance(output, list):
            raise gear_error(
                "response_shape_invalid", "Response output is not a list.",
                "responses_adapter", True, {},
            )
        items: list[dict[str, Any]] = []
        for item in output:
            if not isinstance(item, dict):
                raise gear_error(
                    "response_shape_invalid", "Response output item is not an object.",
                    "responses_adapter", True, {},
                )
            items.append(item)
        source_scope = None
        if self.policy.mode is ReasoningReplayMode.ENCRYPTED:
            source_scope = self.policy.current_scope
        return replay_output_items(items, source_scope, self.policy)

    @property
    def function_calls(self) -> list[FunctionCall]:
        """Decodes tool calls without interpreting unrelated message text."""
        return extract_function_calls(self.response)

    @property
    def text(self) -> str:
        """Decodes user-facing text when requested by the caller."""
        return extract_output_text(self.response)


class ResponsesModelAdapter:
    """Executes the existing Responses path for one effective configuration."""

    def __init__(self, client: ModelClient, config: ModelConfig) -> None:
        """Binds model communication and replay identity.

        Args:
            client: Existing client, including any neutral progress sink.
            config: Effective model configuration.
        """
        self._client = client
        self._config = config
        self._replay_policy = reasoning_replay_policy(config)

    @property
    def capabilities(self) -> ModelCapabilities:
        """Returns adapter support, independent of remote model availability."""
        return ModelCapabilities(
            streaming=True,
            opaque_reasoning_replay=True,
            textual_compaction=True,
            native_compaction=False,
            function_calling=True,
        )

    @property
    def replay_policy(self) -> ReasoningReplayPolicy:
        """Returns the configured connection's replay policy."""
        return self._replay_policy

    def prepare_history(
        self, events: list[dict[str, Any]], user_text: str,
    ) -> ModelHistory:
        """Prepares stored history under the active replay policy.

        Args:
            events: Ordered session events.
            user_text: Current user message.

        Returns:
            Canonical input items and replay diagnostics.
        """
        return build_model_input(events, user_text, self.replay_policy)

    def create_response(
        self, input_value: object, tools: list[dict[str, object]],
        instructions: str, timeout_seconds: float,
        stream_idle_timeout_seconds: float | None,
    ) -> ResponsesModelResponse:
        """Creates a canonical response through the unchanged client.

        Args:
            input_value: Manually prepared history or compaction prompt.
            tools: Canonical function definitions.
            instructions: Model instructions.
            timeout_seconds: Request timeout.
            stream_idle_timeout_seconds: Required for streaming; otherwise unused.

        Returns:
            Completed response with semantic accessors.
        """
        response = self._client.create_response(
            self._config, input_value, tools, instructions,
            timeout_seconds, stream_idle_timeout_seconds,
        )
        return ResponsesModelResponse(response, self.replay_policy)

    def tool_result_item(self, call_id: str, result: dict[str, object]) -> object:
        """Encodes a tool result.

        Args:
            call_id: Model tool call identifier.
            result: Tool result payload.

        Returns:
            Canonical continuation item.
        """
        return function_call_output_item(call_id, result)

    def user_message_item(self, text: str) -> object:
        """Encodes an additional instruction.

        Args:
            text: User instruction text.

        Returns:
            Canonical continuation item.
        """
        return {'role': 'user', 'content': text}
