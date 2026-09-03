import unittest
from typing import Any

from gear_agent.tui import ChatLine, collect_chat_lines, collect_token_usage, format_progress_event
from gear_agent.agent.events import (
    ModelRequestStarted,
    ReasoningReplayEvaluated,
    ToolUseFinished,
    ToolUseStarted,
)
from gear_agent.config import ReasoningReplayMode


class CollectTokenUsageTests(unittest.TestCase):
    def test_sums_total_tokens_across_model_responses(self) -> None:
        events: list[dict[str, Any]] = [
            {
                "kind": "model_response",
                "payload": {"usage": {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200}},
            },
            {
                "kind": "model_response",
                "payload": {"usage": {"input_tokens": 1500, "output_tokens": 300, "total_tokens": 1800}},
            },
        ]

        usage = collect_token_usage(events)

        self.assertEqual(usage, 3000)

    def test_returns_none_when_no_model_response_has_usage(self) -> None:
        events: list[dict[str, Any]] = [
            {"kind": "user_input", "payload": {"text": "hello"}},
        ]

        usage = collect_token_usage(events)

        self.assertIsNone(usage)

    def test_reads_token_usage_from_scoped_model_response(self) -> None:
        events: list[dict[str, Any]] = [
            {
                "kind": "model_response",
                "payload": {
                    "response": {"usage": {"total_tokens": 42}},
                    "source": {
                        "reasoning_replay": "encrypted",
                        "replay_scope": {
                            "protocol": "responses",
                            "endpoint_identity": "sha256:ee0291cefbb5b6136483fb38ba9efe9264f9b685d5006c273e293a54b43a1883",
                            "model": "gpt-5.5",
                        },
                    },
                },
            }
        ]

        usage = collect_token_usage(events)

        self.assertEqual(usage, 42)


class CollectChatLinesTests(unittest.TestCase):
    def test_maps_user_input_to_you_speaker(self) -> None:
        events: list[dict[str, Any]] = [
            {"kind": "user_input", "payload": {"text": "Refactor the CLI"}},
        ]

        lines = collect_chat_lines(events)

        self.assertEqual(lines, [ChatLine("you", "Refactor the CLI")])

    def test_maps_assistant_message_to_assistant_speaker(self) -> None:
        events: list[dict[str, Any]] = [
            {"kind": "assistant_message", "payload": {"text": "Sure, let me look."}},
        ]

        lines = collect_chat_lines(events)

        self.assertEqual(lines, [ChatLine("assistant", "Sure, let me look.")])

    def test_skips_events_that_are_not_chat(self) -> None:
        events: list[dict[str, Any]] = [
            {"kind": "model_response", "payload": {"usage": {"total_tokens": 100}}},
            {"kind": "user_input", "payload": {"text": "hello"}},
        ]

        lines = collect_chat_lines(events)

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].speaker, "you")

    def test_maps_tool_call_to_tool_speaker(self) -> None:
        events: list[dict[str, Any]] = [
            {
                "kind": "tool_call",
                "payload": {
                    "call_id": "call_1",
                    "name": "file_read",
                    "arguments": {"path": "src/gear_agent/agent/loop.py"},
                },
            },
        ]

        lines = collect_chat_lines(events)

        self.assertEqual(
            lines,
            [
                ChatLine(
                    "tool",
                    "loop ? tool file_read started\npath src/gear_agent/agent/loop.py",
                )
            ],
        )

    def test_maps_tool_error_result_to_tool_speaker(self) -> None:
        events: list[dict[str, Any]] = [
            {
                "kind": "tool_result",
                "payload": {
                    "call_id": "call_1",
                    "name": "file_read",
                    "result": {
                        "error": {
                            "type": "file_not_found",
                            "message": "File does not exist.",
                            "origin": "file_read",
                            "details": {"path": "missing.py"},
                        }
                    },
                },
            },
        ]

        lines = collect_chat_lines(events)

        self.assertEqual(
            lines,
            [
                ChatLine(
                    "tool",
                    (
                        "loop ? tool file_read failed file_not_found\n"
                        "message File does not exist.\n"
                        "origin file_read\n"
                        "returned_to_model yes\n"
                        "detail path missing.py"
                    ),
                )
            ],
        )

    def test_maps_malformed_tool_result_to_tool_display_error(self) -> None:
        events: list[dict[str, Any]] = [
            {
                "kind": "tool_result",
                "payload": {
                    "call_id": "call_1",
                    "name": "shell",
                    "result": {"exit_code": 0},
                },
            },
            {"kind": "assistant_message", "payload": {"text": "done"}},
        ]

        lines = collect_chat_lines(events)

        self.assertEqual(
            lines,
            [
                ChatLine(
                    "tool",
                    "tool display error\nreason shell result.stdout must be a string.",
                ),
                ChatLine("assistant", "done"),
            ],
        )


class FormatProgressEventTests(unittest.TestCase):
    def test_formats_tool_use_started(self) -> None:
        text = format_progress_event(
            ToolUseStarted(
                session_id="session-1",
                iteration=2,
                call_id="call_1",
                name="shell",
                arguments={"command": "pytest", "workdir": ".", "timeout_seconds": 30},
            )
        )

        self.assertEqual(
            text,
            "loop 2 tool shell started\ncommand pytest\nworkdir .\ntimeout 30s",
        )

    def test_formats_unsupported_tool_start_explicitly(self) -> None:
        text = format_progress_event(
            ToolUseStarted(
                session_id="session-1",
                iteration=2,
                call_id="call_1",
                name="missing_tool",
                arguments={"path": "x"},
            )
        )

        self.assertEqual(
            text,
            "loop 2 tool missing_tool started\narguments unsupported tool",
        )

    def test_formats_web_search_started(self) -> None:
        text = format_progress_event(
            ToolUseStarted(
                session_id="session-1",
                iteration=2,
                call_id="call_1",
                name="web_search",
                arguments={"query": "OpenAI Responses API"},
            )
        )

        self.assertEqual(
            text,
            "loop 2 tool web_search started\nquery OpenAI Responses API",
        )

    def test_formats_glob_started(self) -> None:
        text = format_progress_event(
            ToolUseStarted(
                session_id="session-1",
                iteration=2,
                call_id="call_1",
                name="glob",
                arguments={"pattern": "**/*.py", "max_results": 20},
            )
        )

        self.assertEqual(
            text,
            "loop 2 tool glob started\npattern **/*.py\nmax_results 20",
        )

    def test_formats_grep_started(self) -> None:
        text = format_progress_event(
            ToolUseStarted(
                session_id="session-1",
                iteration=2,
                call_id="call_1",
                name="grep",
                arguments={"path": "src", "pattern": "needle", "max_results": 20},
            )
        )

        self.assertEqual(
            text,
            "loop 2 tool grep started\npath src\npattern needle\nmax_results 20",
        )

    def test_formats_model_request_started(self) -> None:
        text = format_progress_event(ModelRequestStarted(session_id="session-1", iteration=2))

        self.assertEqual(text, "loop 2 model request")

    def test_formats_reasoning_replay_diagnostic_without_opaque_content(self) -> None:
        text = format_progress_event(
            ReasoningReplayEvaluated(
                session_id="session-1",
                mode=ReasoningReplayMode.ENCRYPTED,
                reused_encrypted_items=2,
                dropped_disabled_items=0,
                dropped_incompatible_scope_items=1,
                dropped_missing_scope_items=1,
            )
        )

        self.assertEqual(
            text,
            (
                "reasoning replay encrypted\n"
                "reused encrypted items 2\n"
                "dropped incompatible scope 1\n"
                "dropped missing scope 1"
            ),
        )
        self.assertNotIn("encrypted_content", text)

    def test_formats_tool_use_finished(self) -> None:
        text = format_progress_event(
            ToolUseFinished(
                session_id="session-1",
                iteration=2,
                call_id="call_1",
                name="shell",
                result={"exit_code": 0, "stdout": "ok\n", "stderr": "", "timed_out": False},
            )
        )

        self.assertEqual(
            text,
            (
                "loop 2 tool shell completed\n"
                "exit 0\n"
                "timed_out no\n"
                "stdout ok\n"
                "stderr empty"
            ),
        )

    def test_formats_file_read_result_with_content_summary(self) -> None:
        text = format_progress_event(
            ToolUseFinished(
                session_id="session-1",
                iteration=1,
                call_id="call_1",
                name="file_read",
                result={"path": "src/example.py", "content": "alpha\nbeta\n"},
            )
        )

        self.assertEqual(
            text,
            (
                "loop 1 tool file_read completed\n"
                "path src/example.py\n"
                "content 11 bytes, 2 lines\n"
                "preview\n"
                "  alpha\n"
                "  beta"
            ),
        )

    def test_formats_apply_patch_result_with_changed_files(self) -> None:
        text = format_progress_event(
            ToolUseFinished(
                session_id="session-1",
                iteration=1,
                call_id="call_1",
                name="apply_patch",
                result={"changed_files": ["src/gear_agent/tui.py", "tests/test_tui.py"]},
            )
        )

        self.assertEqual(
            text,
            (
                "loop 1 tool apply_patch completed\n"
                "changed_files 2\n"
                "  src/gear_agent/tui.py\n"
                "  tests/test_tui.py"
            ),
        )

    def test_formats_web_search_result_with_results_summary(self) -> None:
        text = format_progress_event(
            ToolUseFinished(
                session_id="session-1",
                iteration=1,
                call_id="call_1",
                name="web_search",
                result={
                    "query": "OpenAI Responses API",
                    "answer": "Use Responses for tool calls.",
                    "results": [
                        {
                            "title": "Responses API",
                            "url": "https://developers.openai.com/api/docs",
                            "content": "Current API docs.",
                            "score": 0.91,
                            "raw_content": None,
                        },
                        {
                            "title": "Function calling",
                            "url": "https://developers.openai.com/api/docs/guides/function-calling",
                            "content": "Use function tools.",
                            "score": 0.84,
                            "raw_content": None,
                        },
                    ],
                    "credits": 1,
                    "request_id": "req-123",
                },
            )
        )

        self.assertEqual(
            text,
            (
                "loop 1 tool web_search completed\n"
                "query OpenAI Responses API\n"
                "results 2\n"
                "credits 1\n"
                "answer Use Responses for tool calls.\n"
                "  1. Responses API\n"
                "     https://developers.openai.com/api/docs\n"
                "     Current API docs.\n"
                "  2. Function calling\n"
                "     https://developers.openai.com/api/docs/guides/function-calling\n"
                "     Use function tools."
            ),
        )

    def test_formats_glob_result_with_match_summary(self) -> None:
        text = format_progress_event(
            ToolUseFinished(
                session_id="session-1",
                iteration=1,
                call_id="call_1",
                name="glob",
                result={
                    "pattern": "**/*.py",
                    "matches": [
                        {"path": "src/gear_agent/config.py", "type": "file"},
                        {"path": "src/gear_agent/tools", "type": "directory"},
                    ],
                    "truncated": True,
                },
            )
        )

        self.assertEqual(
            text,
            (
                "loop 1 tool glob completed\n"
                "pattern **/*.py\n"
                "matches 2\n"
                "truncated yes\n"
                "  1. file src/gear_agent/config.py\n"
                "  2. directory src/gear_agent/tools"
            ),
        )

    def test_formats_grep_result_with_line_summary(self) -> None:
        text = format_progress_event(
            ToolUseFinished(
                session_id="session-1",
                iteration=1,
                call_id="call_1",
                name="grep",
                result={
                    "path": "src",
                    "pattern": "needle",
                    "matches": [
                        {
                            "path": "src/gear_agent/example.py",
                            "line": 12,
                            "text": "return 'needle'",
                        }
                    ],
                    "truncated": False,
                },
            )
        )

        self.assertEqual(
            text,
            (
                "loop 1 tool grep completed\n"
                "path src\n"
                "pattern needle\n"
                "matches 1\n"
                "truncated no\n"
                "  1. src/gear_agent/example.py:12 return 'needle'"
            ),
        )

    def test_truncates_large_tool_result_before_rendering(self) -> None:
        text = format_progress_event(
            ToolUseFinished(
                session_id="session-1",
                iteration=1,
                call_id="call_1",
                name="file_read",
                result={"path": "large.txt", "content": f"head{'x' * 3000}tail"},
            )
        )

        self.assertIn("head", text)
        self.assertIn("[truncated]", text)
        self.assertNotIn("tail", text)

    def test_formats_malformed_progress_event_as_display_error(self) -> None:
        text = format_progress_event(
            ToolUseFinished(
                session_id="session-1",
                iteration=1,
                call_id="call_1",
                name="shell",
                result={"exit_code": 0},
            )
        )

        self.assertEqual(
            text,
            "tool display error\nreason shell result.stdout must be a string.",
        )


if __name__ == "__main__":
    unittest.main()
