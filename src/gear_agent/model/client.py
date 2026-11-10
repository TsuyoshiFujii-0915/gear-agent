from __future__ import annotations

from typing import Any
import json

from gear_agent.config import ModelConfig, ReasoningReplayMode
from gear_agent.errors import GearError, gear_error
from gear_agent.model.arrival import RequestTarget, TimedProgressSink
from gear_agent.model.events import (
    ModelProgressEventSink,
    SilentModelProgressEventSink,
)
from gear_agent.model.responses import ReceivedResponse, validate_response_completion
from gear_agent.model.statistics import json_statistics
from gear_agent.model.streaming import ResponsesStreamAssembler
from gear_agent.model.transport import HttpTransport


def response_generation_options(config: ModelConfig) -> dict[str, object]:
    """Projects explicit generation settings into Responses request fields.

    Args:
        config: Validated model settings shared by ordinary and summary requests.

    Returns:
        Only configured fields, with no nulls or inferred server defaults.
    """
    options: dict[str, object] = {}
    if config.reasoning_effort is not None:
        options['reasoning'] = {'effort': config.reasoning_effort}
    if config.max_output_tokens is not None:
        options['max_output_tokens'] = config.max_output_tokens
    return options


class ModelClient:
    """Responses API-compatible model client."""

    def __init__(
        self,
        transport: HttpTransport,
        progress_sink: ModelProgressEventSink | None = None,
    ) -> None:
        """Initializes a model client.

        The configured sink preserves the standalone client API for callers
        that consume model progress directly. AgentLoop supplies a contextual
        per-request sink instead.

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

        return self.create_response_with_statistics(
            config,
            input_value,
            tools,
            instructions,
            timeout_seconds,
            stream_idle_timeout_seconds,
            self._progress_sink,
        ).response

    def create_response_with_progress(
        self,
        config: ModelConfig,
        input_value: object,
        tools: list[dict[str, object]],
        instructions: str,
        timeout_seconds: float,
        stream_idle_timeout_seconds: float | None,
        progress_sink: ModelProgressEventSink,
    ) -> dict[str, Any]:
        """Creates one response with an explicit per-request progress sink.

        Args:
            config: Model endpoint configuration.
            input_value: Responses API input value.
            tools: Function tool definitions.
            instructions: System-level instructions for the response.
            timeout_seconds: Request timeout in seconds.
            stream_idle_timeout_seconds: Stream read idle timeout, required in
                streaming mode and unused in non-stream mode.
            progress_sink: Consumer for this request's provider-neutral progress.

        Returns:
            Parsed response object.
        """

        return self.create_response_with_statistics(
            config,
            input_value,
            tools,
            instructions,
            timeout_seconds,
            stream_idle_timeout_seconds,
            progress_sink,
        ).response

    def create_response_with_statistics(
        self,
        config: ModelConfig,
        input_value: object,
        tools: list[dict[str, object]],
        instructions: str,
        timeout_seconds: float,
        stream_idle_timeout_seconds: float | None,
        progress_sink: ModelProgressEventSink,
    ) -> ReceivedResponse:
        """Receives the canonical response and request-local observations together."""
        try:
            return self._send_response(
                config, input_value, tools, instructions, timeout_seconds,
                stream_idle_timeout_seconds, progress_sink,
            )
        except GearError as error:
            if (error.error_type == 'http_status_error' and error.details.get('status') == 400
                    and response_generation_options(config)):
                raise gear_error(
                    'http_status_error',
                    'Model endpoint returned HTTP 400. Check endpoint support and configured values for '
                    'model.reasoning_effort and model.max_output_tokens; change or omit unsupported settings explicitly. '
                    'The request was not retried.',
                    'model_client', True, {'status': 400},
                ) from None
            raise

    def _send_response(
        self,
        config: ModelConfig,
        input_value: object,
        tools: list[dict[str, object]],
        instructions: str,
        timeout_seconds: float,
        stream_idle_timeout_seconds: float | None,
        progress_sink: ModelProgressEventSink,
    ) -> ReceivedResponse:
        headers = {"Content-Type": "application/json"}
        if config.api_key is not None:
            headers["Authorization"] = f"Bearer {config.api_key}"

        payload: dict[str, Any] = {
            "model": config.model,
            "input": input_value,
            "instructions": instructions,
            "stream": config.stream,
        }
        payload.update(response_generation_options(config))
        if config.reasoning_replay is ReasoningReplayMode.ENCRYPTED:
            payload["include"] = ["reasoning.encrypted_content"]
        if len(tools) > 0:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if config.stream and stream_idle_timeout_seconds is None:
            raise gear_error(
                "stream_idle_timeout_missing",
                "Streaming requires an explicit model stream idle timeout.",
                "model_client",
                True,
                {"url": config.url},
            )

        if isinstance(progress_sink, TimedProgressSink):
            progress_sink.dispatch(RequestTarget.from_config(config))

        if not config.stream:
            response = self._transport.post_json(
                config.url,
                headers,
                payload,
                timeout_seconds,
            )
            statistics = json_statistics(response)
            validate_response_completion(response, 'model_client', statistics)
            return ReceivedResponse(response, statistics)
        assert stream_idle_timeout_seconds is not None
        assembler = ResponsesStreamAssembler(progress_sink)
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
            if isinstance(progress_sink, TimedProgressSink):
                progress_sink.observe(event)
            assembler.consume(event)
        return assembler.finish_with_statistics()
