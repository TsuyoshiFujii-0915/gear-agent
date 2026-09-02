from __future__ import annotations

from typing import Any
import json

from gear_agent.errors import gear_error
from gear_agent.model.responses import function_call_output_item


FUNCTION_CALL_OUTPUT_HISTORY_MAX_CHARS = 12000


def build_model_input(events: list[dict[str, Any]], current_user_text: str) -> list[object]:
    """Builds model-visible input from stored session events.

    Args:
        events: Stored session events ordered from oldest to newest.
        current_user_text: Current user message.

    Returns:
        Responses API input items for the next model request.

    Raises:
        GearError: If stored model-visible history has an invalid shape.
    """

    input_items: list[object] = []
    seen_call_ids: set[str] = set()
    for event in events:
        kind = _required_event_kind(event)
        if kind == "user_input":
            payload = _required_payload(event, kind)
            input_items.append(
                {
                    "role": "user",
                    "content": _required_string(payload, "text", "user_input"),
                }
            )
            continue
        if kind == "assistant_message":
            payload = _required_payload(event, kind)
            input_items.append(
                {
                    "role": "assistant",
                    "content": _required_string(payload, "text", "assistant_message"),
                }
            )
            continue
        if kind == "model_response":
            payload = _required_payload(event, kind)
            for item in _function_call_items(payload):
                call_id = _required_string(item, "call_id", "model_response.function_call")
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

    input_items.append({"role": "user", "content": current_user_text})
    return input_items


def _function_call_items(response: dict[str, Any]) -> list[dict[str, object]]:
    output = response.get("output")
    if not isinstance(output, list):
        raise _shape_error("model_response.output must be a list.", "model_response")
    items: list[dict[str, object]] = []
    for item in output:
        if not isinstance(item, dict):
            raise _shape_error("model_response.output item must be an object.", "model_response")
        if item.get("type") == "function_call":
            _validate_function_call_item(item)
            items.append(item)
    return items


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
