from __future__ import annotations

from typing import Any
import json

from gear_agent.config import ModelConfig, ReasoningReplayMode
from gear_agent.errors import gear_error
from gear_agent.model.events import (
    ModelProgressEventSink,
    SilentModelProgressEventSink,
)
from gear_agent.model.streaming import ResponsesStreamAssembler
from gear_agent.model.transport import HttpTransport


class ModelClient:
    """Responses API-compatible model client."""

    def __init__(
        self,
        transport: HttpTransport,
        progress_sink: ModelProgressEventSink | None = None,
    ) -> None:
        """Initializes a model client.

        The progress sink is optional because non-stream callers and the current
        TUI deliberately do not consume live model deltas.

        Args:
            transport: HTTP transport used for model requests.
            progress_sink: Consumer for provider-neutral streaming progress.
        """

        self._transport = transport
        if progress_sink is None:
            self._progress_sink: ModelProgressEventSink = SilentModelProgressEventSink()
        else:
            self._progress_sink = progress_sink

    def create_response(
        self,
        config: ModelConfig,
        input_value: object,
        tools: list[dict[str, object]],
        instructions: str,
        timeout_seconds: float,
        stream_idle_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Creates one completed response through JSON or SSE transport.

        Args:
            config: Model endpoint configuration.
            input_value: Responses API input value.
            tools: Function tool definitions.
            instructions: System-level instructions for the response.
            timeout_seconds: Request timeout in seconds.
            stream_idle_timeout_seconds: Stream read idle timeout, required in
                streaming mode and unused in non-stream mode.

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
            "stream": config.stream,
        }
        if config.reasoning_replay is ReasoningReplayMode.ENCRYPTED:
            payload["include"] = ["reasoning.encrypted_content"]
        if len(tools) > 0:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if not config.stream:
            return self._transport.post_json(
                config.url,
                headers,
                payload,
                timeout_seconds,
            )
        if stream_idle_timeout_seconds is None:
            raise gear_error(
                "stream_idle_timeout_missing",
                "Streaming requires an explicit model stream idle timeout.",
                "model_client",
                True,
                {"url": config.url},
            )

        assembler = ResponsesStreamAssembler(self._progress_sink)
        for sse_event in self._transport.post_sse(
            config.url,
            headers,
            payload,
            timeout_seconds,
            stream_idle_timeout_seconds,
        ):
            if sse_event.data == "[DONE]":
                break
            try:
                event = json.loads(sse_event.data)
            except json.JSONDecodeError as exc:
                raise gear_error(
                    "response_stream_event_invalid",
                    "Model response stream event contains invalid JSON.",
                    "model_client",
                    True,
                    {"sse_event": sse_event.event, "data": sse_event.data},
                ) from exc
            if not isinstance(event, dict):
                raise gear_error(
                    "response_stream_event_invalid",
                    "Model response stream event JSON is not an object.",
                    "model_client",
                    True,
                    {"sse_event": sse_event.event},
                )
            assembler.consume(event)
        return assembler.finish()
