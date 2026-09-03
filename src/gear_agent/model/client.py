from __future__ import annotations

from typing import Any

from gear_agent.config import ModelConfig, ReasoningReplayMode
from gear_agent.model.transport import HttpTransport


class ModelClient:
    """Responses API-compatible non-stream model client."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def create_response(
        self,
        config: ModelConfig,
        input_value: object,
        tools: list[dict[str, object]],
        instructions: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Creates one non-stream response.

        Args:
            config: Model endpoint configuration.
            input_value: Responses API input value.
            tools: Function tool definitions.
            instructions: System-level instructions for the response.
            timeout_seconds: Request timeout in seconds.

        Returns:
            Parsed response object.
        """

        headers = {"Content-Type": "application/json"}
        if config.api_key is not None:
            headers["Authorization"] = f"Bearer {config.api_key}"

        payload: dict[str, Any] = {
            "model": config.model,
            "input": input_value,
            "instructions": instructions,
            "stream": False,
        }
        if config.reasoning_replay is ReasoningReplayMode.ENCRYPTED:
            payload["include"] = ["reasoning.encrypted_content"]
        if len(tools) > 0:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        return self._transport.post_json(
            config.url,
            headers,
            payload,
            timeout_seconds,
        )
