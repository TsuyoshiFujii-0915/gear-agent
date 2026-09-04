from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from time import sleep
from typing import Iterator


@dataclass(frozen=True)
class ServedResponse:
    """Response emitted by the local SSE test endpoint."""

    status: int
    content_type: str
    chunks: list[bytes]
    delay_after_chunk_index: int | None
    delay_seconds: float
    declared_content_length: int | None


@contextmanager
def serve_response(response: ServedResponse) -> Iterator[str]:
    """Serves one deterministic HTTP response from a local endpoint.

    Args:
        response: HTTP response behavior.

    Yields:
        URL accepting POST requests.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(content_length)
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            if response.declared_content_length is not None:
                self.send_header(
                    "Content-Length",
                    str(response.declared_content_length),
                )
            self.end_headers()
            for index, chunk in enumerate(response.chunks):
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                if response.delay_after_chunk_index == index:
                    sleep(response.delay_seconds)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/v1/responses"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
