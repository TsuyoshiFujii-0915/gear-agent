from __future__ import annotations

from typing import Any, Literal
import json

from gear_agent.errors import GearError, gear_error
from gear_agent.model.types import FunctionCall


def validate_response_completion(response: dict[str, Any], origin: str) -> None:
    """Rejects unsuccessful Responses objects before their output can be used.

    Missing status is deliberately accepted for legacy JSON-compatible servers;
    it is never replaced with a fabricated completed status. The SSE caller
    additionally requires an explicit completed terminal event.

    Args:
        response: Parsed or assembled response, not an intermediate snapshot.
        origin: Model component reporting the failure.

    Raises:
        GearError: If status or error metadata prevents successful completion.
    """
    status = response.get('status')
    if status == 'incomplete':
        raise response_failure_error('response_incomplete', response, origin)
    if status == 'failed':
        raise response_failure_error('response_failed', response, origin)
    if 'status' in response and status != 'completed':
        details: dict[str, object] = {'reason': 'invalid_status'}
        if isinstance(status, str) and status in ('cancelled', 'queued', 'in_progress'):
            details['status'] = status
        raise gear_error(
            'response_status_invalid', 'Model response status is not completed.',
            origin, True, details,
        )
    if response.get('error') is not None:
        raise response_failure_error('response_failed', response, origin)
    if response.get('incomplete_details') is not None:
        raise gear_error(
            'response_status_invalid', 'Model response has conflicting incomplete details.',
            origin, True, {'reason': 'conflicting_incomplete_details'},
        )


def response_failure_error(
    error_type: Literal['response_failed', 'response_incomplete', 'response_stream_error'],
    response: dict[str, Any], origin: str,
) -> GearError:
    """Builds a failure diagnostic without copying arbitrary provider content.

    Args:
        error_type: Category established by response status or SSE event type.
        response: Provider object from which only allowlisted values are read.
        origin: Model component reporting the failure.

    Returns:
        Structured error containing a fixed message and safe code/reason values.
    """
    messages = {
        'response_failed': 'Model response failed.',
        'response_incomplete': 'Model response is incomplete.',
        'response_stream_error': 'Model endpoint emitted a stream error event.',
    }
    details: dict[str, object] = {}
    error = response.get('error')
    if isinstance(error, dict):
        code = error.get('code')
        if isinstance(code, str) and code in (
            'server_error', 'rate_limit_exceeded', 'invalid_prompt',
            'vector_store_timeout', 'invalid_request_error', 'model_not_found',
        ):
            details['code'] = code
    incomplete = response.get('incomplete_details')
    if isinstance(incomplete, dict):
        reason = incomplete.get('reason')
        if isinstance(reason, str) and reason in ('max_output_tokens', 'content_filter'):
            details['reason'] = reason
    return gear_error(error_type, messages[error_type], origin, True, details)


def extract_output_text(response: dict[str, Any]) -> str:
    """Extracts assistant output text from a Responses API response.

    Args:
        response: Parsed response object.

    Returns:
        Concatenated output text.

    Raises:
        GearError: If the response output is malformed.
    """

    output = _required_output(response)
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            raise _shape_error("Output item is not an object.")
        if item.get("type") == "message":
            content = item.get("content")
            if not isinstance(content, list):
                raise _shape_error("Message content is not a list.")
            for content_item in content:
                if not isinstance(content_item, dict):
                    raise _shape_error("Message content item is not an object.")
                if content_item.get("type") == "output_text":
                    text = content_item.get("text")
                    if not isinstance(text, str):
                        raise _shape_error("output_text.text is not a string.")
                    parts.append(text)
    return "".join(parts)


def extract_function_calls(response: dict[str, Any]) -> list[FunctionCall]:
    """Extracts function calls from a Responses API response.

    Args:
        response: Parsed response object.

    Returns:
        Function calls in output order.

    Raises:
        GearError: If function call items are malformed.
    """

    output = _required_output(response)
    calls: list[FunctionCall] = []
    for item in output:
        if not isinstance(item, dict):
            raise _shape_error("Output item is not an object.")
        if item.get("type") != "function_call":
            continue
        call_id = item.get("call_id")
        name = item.get("name")
        raw_arguments = item.get("arguments")
        if not isinstance(call_id, str):
            raise _shape_error("function_call.call_id is not a string.")
        if not isinstance(name, str):
            raise _shape_error("function_call.name is not a string.")
        if not isinstance(raw_arguments, str):
            raise _shape_error("function_call.arguments is not a string.")
        try:
            parsed_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise gear_error(
                "response_shape_invalid",
                "function_call.arguments is not valid JSON.",
                "responses",
                True,
                {"tool": name},
            ) from exc
        if not isinstance(parsed_arguments, dict):
            raise _shape_error("function_call.arguments is not a JSON object.")
        calls.append(
            FunctionCall(
                call_id=call_id,
                name=name,
                arguments=parsed_arguments,
            )
        )
    return calls


def function_call_output_item(call_id: str, output: dict[str, object]) -> dict[str, str]:
    """Builds a Responses API function_call_output input item.

    Args:
        call_id: Tool call identifier.
        output: Tool result object.

    Returns:
        Responses API input item.
    """

    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(output, ensure_ascii=False),
    }


def _required_output(response: dict[str, Any]) -> list[object]:
    output = response.get("output")
    if not isinstance(output, list):
        raise _shape_error("Response output is not a list.")
    return output


def _shape_error(message: str) -> Exception:
    return gear_error(
        "response_shape_invalid",
        message,
        "responses",
        True,
        {},
    )
