from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

from gear_agent.errors import gear_error


@dataclass(frozen=True)
class FunctionCall:
    """Function call emitted by the Responses API.

    Attributes:
        call_id: Tool call identifier used for function_call_output.
        name: Tool name.
        arguments: Parsed JSON arguments.
    """

    call_id: str
    name: str
    arguments: dict[str, object]


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
