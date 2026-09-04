from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from gear_agent.errors import gear_error
from gear_agent.model.events import (
    ModelFunctionCallArgumentsDelta,
    ModelOutputItemCompleted,
    ModelProgressEventSink,
    ModelReasoningDelta,
    ModelTextDelta,
)


@dataclass
class _ItemState:
    item: dict[str, Any]
    item_id: str | None
    argument_fragments: list[str] = field(default_factory=list)
    completed_arguments: str | None = None
    text_fragments: dict[int, list[str]] = field(default_factory=dict)
    reasoning_summary_fragments: dict[int, list[str]] = field(default_factory=dict)
    reasoning_text_fragments: dict[int, list[str]] = field(default_factory=dict)


class ResponsesStreamAssembler:
    """Assembles typed Responses stream events into one completed response."""

    def __init__(self, progress_sink: ModelProgressEventSink) -> None:
        self._progress_sink = progress_sink
        self._created_response: dict[str, Any] | None = None
        self._item_states: dict[int, _ItemState] = {}
        self._completed_items: dict[int, dict[str, Any]] = {}
        self._completed_response: dict[str, Any] | None = None
        self._terminal_received = False

    def consume(self, event: dict[str, Any]) -> None:
        """Consumes one decoded Responses event.

        Args:
            event: Decoded Responses event object.

        Raises:
            GearError: If a known event is malformed or terminal failure occurs.
        """

        if self._terminal_received:
            raise _event_error("Received an event after a terminal stream event.", event)
        event_type = event.get("type")
        if not isinstance(event_type, str):
            raise _event_error("Stream event type is not a string.", event)

        if event_type in {"response.created", "response.in_progress"}:
            self._consume_response_metadata(event)
            return
        if event_type == "response.output_item.added":
            self._consume_output_item_added(event)
            return
        if event_type == "response.output_item.done":
            self._consume_output_item_done(event)
            return
        if event_type == "response.output_text.delta":
            self._consume_text_delta(event)
            return
        if event_type == "response.output_text.done":
            self._consume_text_done(event)
            return
        if event_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            self._consume_reasoning_delta(event)
            return
        if event_type in {
            "response.reasoning_summary_text.done",
            "response.reasoning_text.done",
        }:
            self._consume_reasoning_done(event)
            return
        if event_type == "response.function_call_arguments.delta":
            self._consume_function_arguments_delta(event)
            return
        if event_type == "response.function_call_arguments.done":
            self._consume_function_arguments_done(event)
            return
        if event_type == "response.completed":
            self._consume_completed(event)
            return
        if event_type == "response.failed":
            self._raise_terminal_response_error("response_failed", event)
        if event_type == "response.incomplete":
            self._raise_terminal_response_error("response_incomplete", event)
        if event_type == "error":
            raise gear_error(
                "response_stream_error",
                "Model endpoint emitted a stream error event.",
                "responses_stream",
                True,
                {
                    "code": event.get("code"),
                    "message": event.get("message"),
                },
            )

    def finish(self) -> dict[str, Any]:
        """Returns the canonical response after successful termination.

        Returns:
            Completed Responses object.

        Raises:
            GearError: If EOF occurred before a completed terminal event.
        """

        if self._completed_response is None:
            raise gear_error(
                "response_stream_terminated",
                "Model response stream ended before a completed terminal event.",
                "responses_stream",
                True,
                {"completed_output_items": len(self._completed_items)},
            )
        return self._completed_response

    def _consume_response_metadata(self, event: dict[str, Any]) -> None:
        response = event.get("response")
        if not isinstance(response, dict):
            raise _event_error("Response lifecycle event lacks a response object.", event)
        self._created_response = deepcopy(response)

    def _consume_output_item_added(self, event: dict[str, Any]) -> None:
        output_index = _required_index(event, "output_index")
        item = _required_item(event)
        if output_index in self._item_states:
            raise _event_error("Output item index was added more than once.", event)
        item_id = item.get("id")
        if item_id is not None and not isinstance(item_id, str):
            raise _event_error("Output item id is not a string.", event)
        self._item_states[output_index] = _ItemState(deepcopy(item), item_id)

    def _consume_output_item_done(self, event: dict[str, Any]) -> None:
        output_index = _required_index(event, "output_index")
        item = _required_item(event)
        state = self._item_states.get(output_index)
        if state is not None:
            _validate_item_identity(state, item, event)
            item = _complete_item_from_fragments(item, state)
        completed_item = deepcopy(item)
        self._completed_items[output_index] = completed_item
        self._progress_sink.publish(
            ModelOutputItemCompleted(
                output_index=output_index,
                item=deepcopy(completed_item),
            )
        )

    def _consume_text_delta(self, event: dict[str, Any]) -> None:
        delta = _required_string(event, "delta")
        output_index = _required_index(event, "output_index")
        content_index = _required_index(event, "content_index")
        state = self._item_states.get(output_index)
        if state is not None:
            state.text_fragments.setdefault(content_index, []).append(delta)
        self._progress_sink.publish(ModelTextDelta(delta=delta))

    def _consume_text_done(self, event: dict[str, Any]) -> None:
        text = _required_string(event, "text")
        output_index = _required_index(event, "output_index")
        content_index = _required_index(event, "content_index")
        state = self._item_states.get(output_index)
        if state is not None:
            state.text_fragments[content_index] = [text]

    def _consume_reasoning_delta(self, event: dict[str, Any]) -> None:
        delta = _required_string(event, "delta")
        output_index = _required_index(event, "output_index")
        reasoning_index = _reasoning_index(event)
        state = self._item_states.get(output_index)
        if state is not None:
            fragments = _reasoning_fragments(state, event)
            fragments.setdefault(reasoning_index, []).append(delta)
        self._progress_sink.publish(ModelReasoningDelta(delta=delta))

    def _consume_reasoning_done(self, event: dict[str, Any]) -> None:
        text = _required_string(event, "text")
        output_index = _required_index(event, "output_index")
        reasoning_index = _reasoning_index(event)
        state = self._item_states.get(output_index)
        if state is not None:
            _reasoning_fragments(state, event)[reasoning_index] = [text]

    def _consume_function_arguments_delta(self, event: dict[str, Any]) -> None:
        output_index = _required_index(event, "output_index")
        item_id = _required_string(event, "item_id")
        delta = _required_string(event, "delta")
        state = self._required_item_state(output_index, event)
        if state.item_id != item_id:
            raise _event_error("Function argument delta item identity does not match.", event)
        state.argument_fragments.append(delta)
        self._progress_sink.publish(
            ModelFunctionCallArgumentsDelta(
                item_id=item_id,
                output_index=output_index,
                delta=delta,
            )
        )

    def _consume_function_arguments_done(self, event: dict[str, Any]) -> None:
        output_index = _required_index(event, "output_index")
        item_id = _required_string(event, "item_id")
        arguments = _required_string(event, "arguments")
        state = self._required_item_state(output_index, event)
        if state.item_id != item_id:
            raise _event_error("Completed function arguments item identity does not match.", event)
        state.completed_arguments = arguments

    def _consume_completed(self, event: dict[str, Any]) -> None:
        response = event.get("response")
        if not isinstance(response, dict):
            raise _event_error("Completed event lacks a response object.", event)
        terminal_response = deepcopy(response)
        status = terminal_response.get("status")
        if status == "failed":
            self._raise_terminal_response_error("response_failed", event)
        if status == "incomplete":
            self._raise_terminal_response_error("response_incomplete", event)
        if status != "completed":
            raise _event_error(
                "Completed event response status is not completed.",
                event,
            )
        output = terminal_response.get("output")
        if output is not None and not isinstance(output, list):
            raise _event_error("Completed response output is not a list.", event)
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    raise _event_error(
                        "Completed response output item is not an object.",
                        event,
                    )
        if isinstance(output, list) and _terminal_output_is_complete(
            output,
            self._completed_items,
            event,
        ):
            self._completed_response = terminal_response
        else:
            assembled = terminal_response
            if self._created_response is not None:
                assembled = deepcopy(self._created_response)
                assembled.update(terminal_response)
            assembled["output"] = [
                deepcopy(self._completed_items[index])
                for index in sorted(self._completed_items)
            ]
            self._completed_response = assembled
        self._terminal_received = True

    def _raise_terminal_response_error(
        self,
        error_type: str,
        event: dict[str, Any],
    ) -> None:
        response = event.get("response")
        if not isinstance(response, dict):
            raise _event_error("Terminal event lacks a response object.", event)
        raise gear_error(
            error_type,
            f"Model response terminated with status {response.get('status')}.",
            "responses_stream",
            True,
            {
                "response_id": response.get("id"),
                "status": response.get("status"),
                "error": response.get("error"),
                "incomplete_details": response.get("incomplete_details"),
            },
        )

    def _required_item_state(
        self,
        output_index: int,
        event: dict[str, Any],
    ) -> _ItemState:
        state = self._item_states.get(output_index)
        if state is None:
            raise _event_error("Stream delta refers to an unknown output item.", event)
        return state


def _complete_item_from_fragments(
    item: dict[str, Any],
    state: _ItemState,
) -> dict[str, Any]:
    completed = deepcopy(item)
    if completed.get("type") == "function_call":
        arguments = completed.get("arguments")
        if arguments is None or arguments == "":
            if state.completed_arguments is not None:
                completed["arguments"] = state.completed_arguments
            elif state.argument_fragments:
                completed["arguments"] = "".join(state.argument_fragments)
    content = completed.get("content")
    if completed.get("type") == "message" and (content is None or content == []):
        completed["content"] = [
            {
                "type": "output_text",
                "text": "".join(state.text_fragments[index]),
                "annotations": [],
            }
            for index in sorted(state.text_fragments)
        ]
    summary = completed.get("summary")
    if completed.get("type") == "reasoning" and (summary is None or summary == []):
        completed["summary"] = [
            {
                "type": "summary_text",
                "text": "".join(state.reasoning_summary_fragments[index]),
            }
            for index in sorted(state.reasoning_summary_fragments)
        ]
    reasoning_content = completed.get("content")
    if (
        completed.get("type") == "reasoning"
        and (reasoning_content is None or reasoning_content == [])
        and state.reasoning_text_fragments
    ):
        completed["content"] = [
            {
                "type": "reasoning_text",
                "text": "".join(state.reasoning_text_fragments[index]),
            }
            for index in sorted(state.reasoning_text_fragments)
        ]
    return completed


def _terminal_output_is_complete(
    output: list[object],
    completed_items: dict[int, dict[str, Any]],
    event: dict[str, Any],
) -> bool:
    if not completed_items:
        return True
    for output_index, completed_item in completed_items.items():
        if output_index >= len(output):
            return False
        terminal_item = output[output_index]
        if not isinstance(terminal_item, dict):
            raise _event_error("Completed response output item is not an object.", event)
        completed_id = completed_item.get("id")
        terminal_id = terminal_item.get("id")
        if isinstance(completed_id, str) and not isinstance(terminal_id, str):
            return False
        if isinstance(completed_id, str) and completed_id != terminal_id:
            raise _event_error(
                "Completed response output item identity does not match.",
                event,
            )
        completed_type = completed_item.get("type")
        terminal_type = terminal_item.get("type")
        if (
            isinstance(completed_type, str)
            and isinstance(terminal_type, str)
            and completed_type != terminal_type
        ):
            raise _event_error(
                "Completed response output item type does not match.",
                event,
            )
        if not _value_covers_completed_item(terminal_item, completed_item):
            return False
    return True


def _reasoning_fragments(
    state: _ItemState,
    event: dict[str, Any],
) -> dict[int, list[str]]:
    event_type = event.get("type")
    if event_type in {
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
    }:
        return state.reasoning_summary_fragments
    if event_type in {
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
    }:
        return state.reasoning_text_fragments
    raise _event_error("Unsupported reasoning event type.", event)


def _value_covers_completed_item(
    terminal_value: object,
    completed_value: object,
) -> bool:
    if _value_is_empty(completed_value):
        return True
    if isinstance(completed_value, dict):
        if not isinstance(terminal_value, dict):
            return False
        for key, nested_completed_value in completed_value.items():
            if key == "status" or nested_completed_value is None:
                continue
            if key not in terminal_value:
                if _value_is_empty(nested_completed_value):
                    continue
                return False
            if not _value_covers_completed_item(
                terminal_value[key],
                nested_completed_value,
            ):
                return False
        return True
    if isinstance(completed_value, list):
        if not isinstance(terminal_value, list):
            return False
        if len(terminal_value) < len(completed_value):
            return False
        return all(
            _value_covers_completed_item(terminal_item, completed_item)
            for terminal_item, completed_item in zip(
                terminal_value,
                completed_value,
                strict=False,
            )
        )
    if isinstance(completed_value, str):
        return isinstance(terminal_value, str) and terminal_value != ""
    return terminal_value is not None


def _value_is_empty(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _validate_item_identity(
    state: _ItemState,
    item: dict[str, Any],
    event: dict[str, Any],
) -> None:
    item_id = item.get("id")
    if item_id is not None and not isinstance(item_id, str):
        raise _event_error("Completed output item id is not a string.", event)
    if state.item_id is not None and item_id != state.item_id:
        raise _event_error("Completed output item identity does not match.", event)


def _reasoning_index(event: dict[str, Any]) -> int:
    if "summary_index" in event:
        return _required_index(event, "summary_index")
    if "content_index" in event:
        return _required_index(event, "content_index")
    raise _event_error("Reasoning event lacks an item-local index.", event)


def _required_index(event: dict[str, Any], key: str) -> int:
    value = event.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _event_error(f"Stream event {key} is not a non-negative integer.", event)
    return value


def _required_string(event: dict[str, Any], key: str) -> str:
    value = event.get(key)
    if not isinstance(value, str):
        raise _event_error(f"Stream event {key} is not a string.", event)
    return value


def _required_item(event: dict[str, Any]) -> dict[str, Any]:
    item = event.get("item")
    if not isinstance(item, dict):
        raise _event_error("Stream event item is not an object.", event)
    return item


def _event_error(message: str, event: dict[str, Any]) -> Exception:
    return gear_error(
        "response_stream_event_invalid",
        message,
        "responses_stream",
        True,
        {"event_type": event.get("type")},
    )
