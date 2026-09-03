from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from gear_agent.agent.events import (
    AgentLoopEvent,
    ModelRequestStarted,
    ReasoningReplayEvaluated,
    ToolUseFinished,
    ToolUseStarted,
)
from gear_agent.model.replay import MODEL_RESPONSE_ENVELOPE_SCHEMA


_MAX_PREVIEW_LINES = 6
_MAX_PREVIEW_LINE_CHARS = 160


@dataclass(frozen=True)
class ChatLine:
    """Visible chat line in the Gear Agent TUI.

    Attributes:
        speaker: Display speaker label.
        text: Message text.
    """

    speaker: str
    text: str


def collect_chat_lines(events: list[dict[str, Any]]) -> list[ChatLine]:
    """Collects visible chat messages from stored session events.

    Args:
        events: Stored session events.

    Returns:
        Chat lines for user and assistant messages.

    Raises:
        ValueError: If a visible chat event has an invalid payload shape.
    """

    chat_lines: list[ChatLine] = []
    for event in events:
        kind = event.get("kind")
        if kind not in {
            "user_input",
            "assistant_message",
            "compaction_summary",
            "turn_error",
            "tool_call",
            "tool_result",
        }:
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"{kind} payload must be an object.")
        if kind == "user_input":
            text = _required_text_payload(kind, payload)
            chat_lines.append(ChatLine("you", text))
        if kind == "assistant_message":
            text = _required_text_payload(kind, payload)
            chat_lines.append(ChatLine("assistant", text))
        if kind == "compaction_summary":
            text = _required_text_payload(kind, payload)
            chat_lines.append(ChatLine("assistant", f"summary saved: {_single_line(text)}"))
        if kind == "turn_error":
            text = _required_text_payload(kind, payload)
            chat_lines.append(ChatLine("error", text))
        if kind == "tool_call":
            chat_lines.append(ChatLine("tool", _format_tool_payload(_format_tool_call_payload, payload)))
        if kind == "tool_result":
            chat_lines.append(
                ChatLine("tool", _format_tool_payload(_format_tool_result_payload, payload))
            )
    return chat_lines


def collect_token_usage(events: list[dict[str, Any]]) -> int | None:
    """Collects total token usage from model response events.

    Args:
        events: Stored session events.

    Returns:
        Total token count, or None when no response usage was reported.

    Raises:
        ValueError: If a response usage object exists but lacks integer total_tokens.
    """

    total = 0
    found_usage = False
    for event in events:
        if event.get("kind") != "model_response":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        response = _model_response_payload(payload)
        if "usage" not in response:
            continue
        usage = response["usage"]
        if usage is None:
            continue
        if not isinstance(usage, dict):
            raise ValueError("model_response usage must be an object when present.")
        total_tokens = usage.get("total_tokens")
        if not isinstance(total_tokens, int):
            raise ValueError(
                "model_response usage.total_tokens must be an integer when usage is present."
            )
        total += total_tokens
        found_usage = True
    if not found_usage:
        return None
    return total


def compact_path(path: Path) -> str:
    """Formats a path with home directory replaced by ~.

    Args:
        path: Filesystem path to format.

    Returns:
        Compact string representation of the path.
    """

    text = str(path)
    home = str(Path.home())
    if text == home:
        return "~"
    if text.startswith(f"{home}/"):
        return f"~/{text[len(home) + 1:]}"
    return text


def compact_session(session_id: str) -> str:
    """Truncates a session ID for display.

    Args:
        session_id: Full session identifier.

    Returns:
        Session ID truncated to 8 characters if longer than 12.
    """

    if len(session_id) <= 12:
        return session_id
    return session_id[:8]


def format_tokens(token_usage: int | None) -> str:
    """Formats a token count for display.

    Args:
        token_usage: Token count, or None when unavailable.

    Returns:
        Human-readable token count string.
    """

    if token_usage is None:
        return "unavailable"
    if token_usage < 1000:
        return str(token_usage)
    return f"{token_usage / 1000:.1f}k"


def format_progress_event(event: AgentLoopEvent) -> str:
    """Formats an agent loop progress event for TUI display.

    Args:
        event: Agent loop progress event.

    Returns:
        TUI display text.

    Raises:
        ValueError: If the event type is unsupported.
    """

    try:
        if isinstance(event, ModelRequestStarted):
            return _format_model_request_started(event)
        if isinstance(event, ReasoningReplayEvaluated):
            return _format_reasoning_replay_evaluated(event)
        if isinstance(event, ToolUseStarted):
            return _format_tool_use_started(event)
        if isinstance(event, ToolUseFinished):
            return _format_tool_use_finished(event)
        raise ValueError(f"Unsupported agent loop event: {type(event).__name__}")
    except ValueError as exc:
        return _format_tool_display_error(str(exc))


def _required_text_payload(kind: str, payload: dict[str, Any]) -> str:
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError(f"{kind} payload.text must be a string.")
    return text


def _single_line(text: str) -> str:
    return " ".join(text.split())


def _format_tool_call_payload(payload: dict[str, Any]) -> str:
    call_id = payload.get("call_id")
    iteration = payload.get("iteration")
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(call_id, str):
        raise ValueError("tool_call payload.call_id must be a string.")
    if not isinstance(name, str):
        raise ValueError("tool_call payload.name must be a string.")
    if not isinstance(arguments, dict):
        raise ValueError("tool_call payload.arguments must be an object.")
    return _format_tool_started(_format_iteration(iteration), name, arguments)


def _format_tool_payload(
    formatter: Callable[[dict[str, Any]], str],
    payload: dict[str, Any],
) -> str:
    try:
        return formatter(payload)
    except ValueError as exc:
        return _format_tool_display_error(str(exc))


def _format_tool_result_payload(payload: dict[str, Any]) -> str:
    call_id = payload.get("call_id")
    iteration = payload.get("iteration")
    name = payload.get("name")
    result = payload.get("result")
    if not isinstance(call_id, str):
        raise ValueError("tool_result payload.call_id must be a string.")
    if not isinstance(name, str):
        raise ValueError("tool_result payload.name must be a string.")
    if not isinstance(result, dict):
        raise ValueError("tool_result payload.result must be an object.")
    return _format_tool_finished(_format_iteration(iteration), name, result)


def _format_tool_use_started(event: ToolUseStarted) -> str:
    return _format_tool_started(str(event.iteration), event.name, event.arguments)


def _format_tool_use_finished(event: ToolUseFinished) -> str:
    return _format_tool_finished(str(event.iteration), event.name, event.result)


def _format_model_request_started(event: ModelRequestStarted) -> str:
    return f"loop {event.iteration} model request"


def _format_reasoning_replay_evaluated(
    event: ReasoningReplayEvaluated,
) -> str:
    lines = [f"reasoning replay {event.mode.value}"]
    if event.reused_encrypted_items > 0:
        lines.append(f"reused encrypted items {event.reused_encrypted_items}")
    if event.dropped_disabled_items > 0:
        lines.append(f"dropped disabled {event.dropped_disabled_items}")
    if event.dropped_incompatible_scope_items > 0:
        lines.append(
            f"dropped incompatible scope {event.dropped_incompatible_scope_items}"
        )
    if event.dropped_missing_scope_items > 0:
        lines.append(f"dropped missing scope {event.dropped_missing_scope_items}")
    return "\n".join(lines)


def _model_response_payload(payload: dict[str, Any]) -> dict[str, Any]:
    is_envelope = payload.get("schema") == MODEL_RESPONSE_ENVELOPE_SCHEMA or (
        "output" not in payload and ("response" in payload or "source" in payload)
    )
    if not is_envelope:
        return payload
    response = payload.get("response")
    source = payload.get("source")
    if not isinstance(response, dict):
        raise ValueError("model_response response must be an object.")
    if not isinstance(source, dict):
        raise ValueError("model_response source must be an object.")
    return response


def _format_tool_started(iteration: str, tool_name: str, arguments: dict[str, object]) -> str:
    if tool_name == "shell":
        return _format_shell_started(iteration, arguments)
    if tool_name == "file_read":
        return _format_file_read_started(iteration, arguments)
    if tool_name == "file_write":
        return _format_file_write_started(iteration, arguments)
    if tool_name == "apply_patch":
        return _format_apply_patch_started(iteration, arguments)
    if tool_name == "glob":
        return _format_glob_started(iteration, arguments)
    if tool_name == "grep":
        return _format_grep_started(iteration, arguments)
    if tool_name == "web_search":
        return _format_web_search_started(iteration, arguments)
    return _format_unsupported_tool_started(iteration, tool_name)


def _format_tool_finished(iteration: str, tool_name: str, result: dict[str, object]) -> str:
    error = result.get("error")
    if error is not None:
        if not isinstance(error, dict):
            raise ValueError("tool_result payload.result.error must be an object.")
        return _format_tool_error(iteration, tool_name, error)
    if tool_name == "shell":
        return _format_shell_finished(iteration, result)
    if tool_name == "file_read":
        return _format_file_read_finished(iteration, result)
    if tool_name == "file_write":
        return _format_file_write_finished(iteration, result)
    if tool_name == "apply_patch":
        return _format_apply_patch_finished(iteration, result)
    if tool_name == "glob":
        return _format_glob_finished(iteration, result)
    if tool_name == "grep":
        return _format_grep_finished(iteration, result)
    if tool_name == "web_search":
        return _format_web_search_finished(iteration, result)
    raise ValueError(f"Unsupported tool for TUI display: {tool_name}")


def _format_shell_started(iteration: str, arguments: dict[str, object]) -> str:
    command = _required_string(arguments, "command", "shell arguments")
    workdir = _required_string(arguments, "workdir", "shell arguments")
    timeout_seconds = _required_int(arguments, "timeout_seconds", "shell arguments")
    return "\n".join(
        [
            f"loop {iteration} tool shell started",
            f"command {command}",
            f"workdir {workdir}",
            f"timeout {timeout_seconds}s",
        ]
    )


def _format_shell_finished(iteration: str, result: dict[str, object]) -> str:
    exit_code = _required_int(result, "exit_code", "shell result")
    stdout = _required_string(result, "stdout", "shell result")
    stderr = _required_string(result, "stderr", "shell result")
    timed_out = _required_bool(result, "timed_out", "shell result")
    lines = [
        f"loop {iteration} tool shell completed",
        f"exit {exit_code}",
        f"timed_out {_format_yes_no(timed_out)}",
    ]
    lines.extend(_format_text_output("stdout", stdout))
    lines.extend(_format_text_output("stderr", stderr))
    return "\n".join(lines)


def _format_file_read_started(iteration: str, arguments: dict[str, object]) -> str:
    path = _required_string(arguments, "path", "file_read arguments")
    return "\n".join([f"loop {iteration} tool file_read started", f"path {path}"])


def _format_file_read_finished(iteration: str, result: dict[str, object]) -> str:
    path = _required_string(result, "path", "file_read result")
    content = _required_string(result, "content", "file_read result")
    lines = [
        f"loop {iteration} tool file_read completed",
        f"path {path}",
        f"content {len(content.encode('utf-8'))} bytes, {_line_count(content)} lines",
    ]
    lines.extend(_format_preview_block("preview", content))
    return "\n".join(lines)


def _format_file_write_started(iteration: str, arguments: dict[str, object]) -> str:
    path = _required_string(arguments, "path", "file_write arguments")
    content = _required_string(arguments, "content", "file_write arguments")
    return "\n".join(
        [
            f"loop {iteration} tool file_write started",
            f"path {path}",
            f"content {len(content.encode('utf-8'))} bytes, {_line_count(content)} lines",
        ]
    )


def _format_file_write_finished(iteration: str, result: dict[str, object]) -> str:
    path = _required_string(result, "path", "file_write result")
    bytes_written = _required_int(result, "bytes_written", "file_write result")
    return "\n".join(
        [
            f"loop {iteration} tool file_write completed",
            f"path {path}",
            f"bytes_written {bytes_written}",
        ]
    )


def _format_apply_patch_started(iteration: str, arguments: dict[str, object]) -> str:
    patch = _required_string(arguments, "patch", "apply_patch arguments")
    return "\n".join(
        [
            f"loop {iteration} tool apply_patch started",
            f"patch {len(patch.encode('utf-8'))} bytes, {_line_count(patch)} lines",
        ]
    )


def _format_apply_patch_finished(iteration: str, result: dict[str, object]) -> str:
    changed_files = _required_string_list(result, "changed_files", "apply_patch result")
    lines = [
        f"loop {iteration} tool apply_patch completed",
        f"changed_files {len(changed_files)}",
    ]
    if len(changed_files) == 0:
        lines.append("  none")
    else:
        lines.extend([f"  {path}" for path in changed_files])
    return "\n".join(lines)


def _format_glob_started(iteration: str, arguments: dict[str, object]) -> str:
    pattern = _required_string(arguments, "pattern", "glob arguments")
    max_results = _required_int(arguments, "max_results", "glob arguments")
    return "\n".join(
        [
            f"loop {iteration} tool glob started",
            f"pattern {pattern}",
            f"max_results {max_results}",
        ]
    )


def _format_glob_finished(iteration: str, result: dict[str, object]) -> str:
    pattern = _required_string(result, "pattern", "glob result")
    matches = _required_object_list(result, "matches", "glob result")
    truncated = _required_bool(result, "truncated", "glob result")
    lines = [
        f"loop {iteration} tool glob completed",
        f"pattern {pattern}",
        f"matches {len(matches)}",
        f"truncated {_format_yes_no(truncated)}",
    ]
    if len(matches) == 0:
        lines.append("  none")
    else:
        lines.extend(_format_glob_match_lines(matches))
    return "\n".join(lines)


def _format_glob_match_lines(matches: list[dict[object, object]]) -> list[str]:
    lines: list[str] = []
    for index, item in enumerate(matches, start=1):
        path = _required_string(item, "path", "glob result item")
        path_type = _required_string(item, "type", "glob result item")
        lines.append(f"  {index}. {path_type} {_preview_line(path)}")
    return lines


def _format_grep_started(iteration: str, arguments: dict[str, object]) -> str:
    path = _required_string(arguments, "path", "grep arguments")
    pattern = _required_string(arguments, "pattern", "grep arguments")
    max_results = _required_int(arguments, "max_results", "grep arguments")
    return "\n".join(
        [
            f"loop {iteration} tool grep started",
            f"path {path}",
            f"pattern {pattern}",
            f"max_results {max_results}",
        ]
    )


def _format_grep_finished(iteration: str, result: dict[str, object]) -> str:
    path = _required_string(result, "path", "grep result")
    pattern = _required_string(result, "pattern", "grep result")
    matches = _required_object_list(result, "matches", "grep result")
    truncated = _required_bool(result, "truncated", "grep result")
    lines = [
        f"loop {iteration} tool grep completed",
        f"path {path}",
        f"pattern {pattern}",
        f"matches {len(matches)}",
        f"truncated {_format_yes_no(truncated)}",
    ]
    if len(matches) == 0:
        lines.append("  none")
    else:
        lines.extend(_format_grep_match_lines(matches))
    return "\n".join(lines)


def _format_grep_match_lines(matches: list[dict[object, object]]) -> list[str]:
    lines: list[str] = []
    for index, item in enumerate(matches, start=1):
        path = _required_string(item, "path", "grep result item")
        line_number = _required_int(item, "line", "grep result item")
        text = _required_string(item, "text", "grep result item")
        lines.append(f"  {index}. {path}:{line_number} {_preview_line(text)}")
    return lines


def _format_web_search_started(iteration: str, arguments: dict[str, object]) -> str:
    query = _required_string(arguments, "query", "web_search arguments")
    return "\n".join([f"loop {iteration} tool web_search started", f"query {query}"])


def _format_web_search_finished(iteration: str, result: dict[str, object]) -> str:
    query = _required_string(result, "query", "web_search result")
    search_results = _required_object_list(result, "results", "web_search result")
    lines = [
        f"loop {iteration} tool web_search completed",
        f"query {query}",
        f"results {len(search_results)}",
        f"credits {_format_optional_int(result.get('credits'))}",
    ]
    answer = result.get("answer")
    if answer is not None:
        if not isinstance(answer, str):
            raise ValueError("web_search result.answer must be a string when present.")
        if answer != "":
            lines.append(f"answer {_preview_line(answer)}")
    if len(search_results) == 0:
        lines.append("  none")
    else:
        lines.extend(_format_web_search_result_lines(search_results))
    return "\n".join(lines)


def _format_web_search_result_lines(results: list[dict[object, object]]) -> list[str]:
    lines: list[str] = []
    for index, item in enumerate(results, start=1):
        title = _required_string(item, "title", "web_search result item")
        url = _required_string(item, "url", "web_search result item")
        content = _required_string(item, "content", "web_search result item")
        lines.append(f"  {index}. {_preview_line(title)}")
        lines.append(f"     {_preview_line(url)}")
        lines.append(f"     {_preview_line(content)}")
    return lines


def _format_unsupported_tool_started(iteration: str, tool_name: str) -> str:
    return "\n".join(
        [
            f"loop {iteration} tool {tool_name} started",
            "arguments unsupported tool",
        ]
    )


def _format_tool_error(iteration: str, tool_name: str, error: dict[object, object]) -> str:
    error_type = _required_string(error, "type", "tool error")
    message = _required_string(error, "message", "tool error")
    origin = _required_string(error, "origin", "tool error")
    details = error.get("details")
    if not isinstance(details, dict):
        raise ValueError("tool error details must be an object.")
    lines = [
        f"loop {iteration} tool {tool_name} failed {error_type}",
        f"message {message}",
        f"origin {origin}",
        "returned_to_model yes",
    ]
    lines.extend(_format_detail_lines(details))
    return "\n".join(lines)


def _format_tool_display_error(reason: str) -> str:
    return "\n".join(["tool display error", f"reason {reason}"])


def _format_iteration(value: object) -> str:
    if value is None:
        return "?"
    if not isinstance(value, int):
        raise ValueError("tool event payload.iteration must be an integer when present.")
    return str(value)


def _required_string(payload: dict[object, object], key: str, origin: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{origin}.{key} must be a string.")
    return value


def _required_int(payload: dict[object, object], key: str, origin: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{origin}.{key} must be an integer.")
    return value


def _required_bool(payload: dict[object, object], key: str, origin: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{origin}.{key} must be a boolean.")
    return value


def _required_string_list(payload: dict[object, object], key: str, origin: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{origin}.{key} must be a list.")
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{origin}.{key} must contain only strings.")
        strings.append(item)
    return strings


def _required_object_list(
    payload: dict[object, object],
    key: str,
    origin: str,
) -> list[dict[object, object]]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{origin}.{key} must be a list.")
    objects: list[dict[object, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{origin}.{key} must contain only objects.")
        objects.append(item)
    return objects


def _format_optional_int(value: object) -> str:
    if value is None:
        return "unavailable"
    if not isinstance(value, int):
        raise ValueError("web_search result.credits must be an integer when present.")
    return str(value)


def _format_yes_no(value: bool) -> str:
    if value:
        return "yes"
    return "no"


def _line_count(text: str) -> int:
    return len(text.splitlines())


def _format_text_output(label: str, text: str) -> list[str]:
    if text == "":
        return [f"{label} empty"]
    lines = text.splitlines()
    if len(lines) == 1:
        return [f"{label} {_preview_line(lines[0])}"]
    return [f"{label} {len(lines)} lines", *_indented_preview_lines(lines)]


def _format_preview_block(label: str, text: str) -> list[str]:
    lines = text.splitlines()
    if len(lines) == 0:
        return [f"{label} empty"]
    return [label, *_indented_preview_lines(lines)]


def _indented_preview_lines(lines: list[str]) -> list[str]:
    preview = [f"  {_preview_line(line)}" for line in lines[:_MAX_PREVIEW_LINES]]
    if len(lines) > _MAX_PREVIEW_LINES:
        preview.append("  ... [truncated]")
    return preview


def _preview_line(line: str) -> str:
    if len(line) <= _MAX_PREVIEW_LINE_CHARS:
        return line
    return f"{line[:_MAX_PREVIEW_LINE_CHARS]}... [truncated]"


def _format_detail_lines(details: dict[object, object]) -> list[str]:
    lines: list[str] = []
    for key, value in details.items():
        if not isinstance(key, str):
            raise ValueError("tool error details keys must be strings.")
        lines.append(f"detail {key} {_format_detail_value(value)}")
    return lines


def _format_detail_value(value: object) -> str:
    if isinstance(value, str):
        return _preview_line(value)
    if isinstance(value, bool):
        return _format_yes_no(value)
    if isinstance(value, int):
        return str(value)
    raise ValueError("tool error detail values must be strings, booleans, or integers.")
