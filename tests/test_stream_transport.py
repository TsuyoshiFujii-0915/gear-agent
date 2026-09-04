from __future__ import annotations

import unittest

from gear_agent.errors import GearError
from gear_agent.model.transport import HttpxHttpTransport
from tests.sse_server import ServedResponse, serve_response


class StreamTransportTests(unittest.TestCase):
    def test_parses_crlf_comments_multiline_data_and_utf8_across_chunks(self) -> None:
        body = (
            ": keepalive\r\n"
            "event: response\r\n"
            "data: {\"type\":\r\n"
            "data: \"response.output_text.delta\",\"delta\":\"日\"}\r\n"
            "\r\n"
        ).encode("utf-8")
        response = ServedResponse(
            status=200,
            content_type="text/event-stream; charset=utf-8",
            chunks=[body[index : index + 1] for index in range(len(body))],
            delay_after_chunk_index=None,
            delay_seconds=0.0,
            declared_content_length=None,
        )

        with serve_response(response) as url:
            events = list(
                HttpxHttpTransport().post_sse(
                    url,
                    {"Content-Type": "application/json"},
                    {"stream": True},
                    2.0,
                    1.0,
                )
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event, "response")
        self.assertEqual(
            events[0].data,
            '{"type":\n"response.output_text.delta","delta":"日"}',
        )

    def test_reports_http_status_before_stream_iteration(self) -> None:
        body = b'{"error":{"message":"denied"}}'
        response = ServedResponse(
            status=401,
            content_type="application/json",
            chunks=[body],
            delay_after_chunk_index=None,
            delay_seconds=0.0,
            declared_content_length=len(body),
        )

        with serve_response(response) as url:
            with self.assertRaises(GearError) as raised:
                list(HttpxHttpTransport().post_sse(url, {}, {}, 2.0, 1.0))

        self.assertEqual(raised.exception.error_type, "http_status_error")
        self.assertEqual(raised.exception.details["status"], 401)
        self.assertIn("denied", str(raised.exception.details["body"]))

    def test_reports_stream_idle_timeout_separately(self) -> None:
        first_event = b'data: {"type":"response.created"}\n\n'
        response = ServedResponse(
            status=200,
            content_type="text/event-stream",
            chunks=[first_event, b'data: {"type":"response.completed"}\n\n'],
            delay_after_chunk_index=0,
            delay_seconds=0.2,
            declared_content_length=None,
        )

        with serve_response(response) as url:
            with self.assertRaises(GearError) as raised:
                list(HttpxHttpTransport().post_sse(url, {}, {}, 2.0, 0.05))

        self.assertEqual(raised.exception.error_type, "http_stream_idle_timeout")
        self.assertEqual(raised.exception.details["idle_timeout_seconds"], 0.05)

    def test_reports_connection_close_after_http_200(self) -> None:
        body = b'data: {"type":"response.created"}\n\n'
        response = ServedResponse(
            status=200,
            content_type="text/event-stream",
            chunks=[body],
            delay_after_chunk_index=None,
            delay_seconds=0.0,
            declared_content_length=len(body) + 100,
        )

        with serve_response(response) as url:
            with self.assertRaises(GearError) as raised:
                list(HttpxHttpTransport().post_sse(url, {}, {}, 2.0, 1.0))

        self.assertEqual(raised.exception.error_type, "http_stream_failed")


if __name__ == "__main__":
    unittest.main()
