from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

from gear_agent.errors import gear_error
from gear_agent.model.responses import function_call_output_item
from gear_agent.model.replay import (
    ReasoningReplayDiagnostic,
    ReasoningReplayPolicy,
    empty_replay_diagnostic,
    read_model_response_event,
    replay_output_items,
)


FUNCTION_CALL_OUTPUT_HISTORY_MAX_CHARS = 12000
COMPACTION_SUMMARY_PREFIX = (
    "Earlier session context (compressed continuation, not a new user request):"
)


@dataclass(frozen=True)
class ModelHistory:
    """Model-visible input and opaque reasoning replay diagnostics.

    Attributes:
        items: Responses-compatible input items.
        diagnostic: Counts describing opaque state reuse and removal.
    """

    items: list[object]
    diagnostic: ReasoningReplayDiagnostic


def build_model_input(
    events: list[dict[str, Any]],
    current_user_text: str,
    replay_policy: ReasoningReplayPolicy,
) -> ModelHistory:
    """Builds model-visible input from stored session events.

    Args:
        events: Stored session events ordered from oldest to newest.
        current_user_text: Current user message.
        replay_policy: Opaque reasoning replay policy for the active model.

    Returns:
        Responses API input items and replay diagnostics.

    Raises:
        GearError: If stored model-visible history has an invalid shape.
    """

    history = build_model_history(events, replay_policy)
    input_items = [*history.items, {"role": "user", "content": current_user_text}]
    return ModelHistory(items=input_items, diagnostic=history.diagnostic)


def build_model_history(
    events: list[dict[str, Any]],
    replay_policy: ReasoningReplayPolicy,
) -> ModelHistory:
    """Builds effective model-visible history from stored session events.

    Args:
        events: Stored session events ordered from oldest to newest.
        replay_policy: Opaque reasoning replay policy for the active model.

    Returns:
        Responses API input items and replay diagnostics.

    Raises:
        GearError: If effective model-visible history has an invalid shape.
    """

    input_items: list[object] = []
    replay_diagnostic = empty_replay_diagnostic()
    effective_events = select_effective_events(events)
    if (
        len(effective_events) > 0
        and _required_event_kind(effective_events[0]) == "compaction_summary"
    ):
        payload = _required_payload(effective_events[0], "compaction_summary")
        summary = _required_string(payload, "text", "compaction_summary")
        input_items.append(
            {
                "role": "user",
                "content": f"{COMPACTION_SUMMARY_PREFIX}\n\n{summary}",
            }
        )
        replay_events = effective_events[1:]
    else:
        replay_events = effective_events

    seen_call_ids: set[str] = set()
    preceding_response_assistant_text: str | None = None
    for event in replay_events:
        kind = _required_event_kind(event)
        if kind == "user_input":
            payload = _required_payload(event, kind)
            input_items.append(
                {
                    "role": "user",
                    "content": _required_string(payload, "text", "user_input"),
                }
            )
            preceding_response_assistant_text = None
            continue
        if kind == "assistant_message":
            payload = _required_payload(event, kind)
            assistant_text = _required_string(payload, "text", "assistant_message")
            if preceding_response_assistant_text is None:
                input_items.append(
                    {
                        "role": "assistant",
                        "content": assistant_text,
                    }
                )
            elif assistant_text != preceding_response_assistant_text:
                raise _shape_error(
                    (
                        "assistant_message.text does not match the preceding "
                        "model_response message output."
                    ),
                    "assistant_message",
                )
            preceding_response_assistant_text = None
            continue
        if kind == "model_response":
            payload = _required_payload(event, kind)
            stored_response = read_model_response_event(payload)
            output_items = _response_output_items(stored_response.response)
            replayed_output = replay_output_items(
                output_items,
                stored_response.source_scope,
                replay_policy,
            )
            output_items = replayed_output.items
            replay_diagnostic = replay_diagnostic.combine(
                replayed_output.diagnostic
            )
            preceding_response_assistant_text = _assistant_output_text(output_items)
            for item in output_items:
                item_type = _required_string(item, "type", "model_response.output")
                if item_type == "function_call":
                    _validate_function_call_item(item)
                    call_id = _required_string(
                        item,
                        "call_id",
                        "model_response.function_call",
                    )
                    seen_call_ids.add(call_id)
                input_items.append(item)
            continue
        if kind == "tool_result":
            payload = _required_payload(event, kind)
            call_id = _required_string(payload, "call_id", "tool_result")
            if call_id not in seen_call_ids:
                raise gear_error(
                    "history_shape_invalid",
                    "Stored tool_result has no preceding function_call.",
                    "history",
                    True,
                    {"call_id": call_id},
                )
            result = payload.get("result")
            if not isinstance(result, dict):
                raise _shape_error("tool_result.result must be an object.", "tool_result")
            input_items.append(_history_function_call_output_item(call_id, result))

    return ModelHistory(items=input_items, diagnostic=replay_diagnostic)


def select_effective_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Selects the latest compaction checkpoint and all subsequent events.

    Args:
        events: Stored session events ordered from oldest to newest.

    Returns:
        A new list containing the effective session event slice.

    Raises:
        GearError: If the checkpoint or later events have an invalid shape.
    """

    checkpoint = _latest_compaction_checkpoint(events)
    if checkpoint is None:
        return list(events)
    checkpoint_index, _ = checkpoint
    return events[checkpoint_index:]


def _latest_compaction_checkpoint(
    events: list[dict[str, Any]],
) -> tuple[int, str] | None:
    for index in range(len(events) - 1, -1, -1):
        event = events[index]
        kind = _required_event_kind(event)
        if kind != "compaction_summary":
            continue
        payload = _required_payload(event, kind)
        summary = _required_string(payload, "text", kind)
        if summary.strip() == "":
            raise _shape_error(
                "compaction_summary.text must not be empty.",
                "compaction_summary",
            )
        return index, summary
    return None


def _response_output_items(response: dict[str, Any]) -> list[dict[str, Any]]:
    output = response.get("output")
    if not isinstance(output, list):
        raise _shape_error("model_response.output must be a list.", "model_response")
    items: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            raise _shape_error(
                "model_response.output item must be an object.",
                "model_response.output",
            )
        _required_string(item, "type", "model_response.output")
        items.append(item)
    return items


def _assistant_output_text(items: list[dict[str, Any]]) -> str | None:
    message_found = False
    parts: list[str] = []
    for item in items:
        if item["type"] != "message":
            continue
        message_found = True
        content = item.get("content")
        if not isinstance(content, list):
            raise _shape_error(
                "model_response message content must be a list.",
                "model_response.message",
            )
        for content_item in content:
            if not isinstance(content_item, dict):
                raise _shape_error(
                    "model_response message content item must be an object.",
                    "model_response.message",
                )
            if content_item.get("type") == "output_text":
                parts.append(
                    _required_string(
                        content_item,
                        "text",
                        "model_response.message.output_text",
                    )
                )
    if not message_found:
        return None
    return "".join(parts)


def _validate_function_call_item(item: dict[str, Any]) -> None:
    _required_string(item, "call_id", "model_response.function_call")
    _required_string(item, "name", "model_response.function_call")
    _required_string(item, "arguments", "model_response.function_call")


def _history_function_call_output_item(
    call_id: str,
    output: dict[str, object],
) -> dict[str, str]:
    serialized_output = json.dumps(output, ensure_ascii=False)
    if len(serialized_output) <= FUNCTION_CALL_OUTPUT_HISTORY_MAX_CHARS:
        return function_call_output_item(call_id, output)
    truncated_output: dict[str, object] = {
        "truncated": True,
        "original_json_chars": len(serialized_output),
        "max_json_chars": FUNCTION_CALL_OUTPUT_HISTORY_MAX_CHARS,
        "json_prefix": serialized_output[:FUNCTION_CALL_OUTPUT_HISTORY_MAX_CHARS],
    }
    return function_call_output_item(call_id, truncated_output)


def _required_event_kind(event: dict[str, Any]) -> str:
    kind = event.get("kind")
    if not isinstance(kind, str):
        raise _shape_error("event.kind must be a string.", "event")
    return kind


def _required_payload(event: dict[str, Any], kind: str) -> dict[str, Any]:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise _shape_error(f"{kind}.payload must be an object.", kind)
    return payload


def _required_string(payload: dict[str, Any], key: str, origin: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise _shape_error(f"{origin}.{key} must be a string.", origin)
    return value


def _shape_error(message: str, origin: str) -> Exception:
    return gear_error(
        "history_shape_invalid",
        message,
        origin,
        True,
        {},
    )
