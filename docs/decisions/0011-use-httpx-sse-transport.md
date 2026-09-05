# Title

Use HTTPX and httpx-sse for model streaming transport

# Status

accepted

# Context

The model transport must consume Responses-compatible HTTP streams without
assuming that network read boundaries are SSE event boundaries. It must also
distinguish the existing request timeout from the maximum idle time between
stream bytes.

# Decision

Replace the urllib model transport with HTTPX and httpx-sse. The transport
normalizes parsed SSE messages into a small internal `SseEvent` value and does
not expose either dependency to the response assembler. For streams, the
existing model timeout applies to connect, write, and connection-pool
operations; a separate stream idle timeout is the HTTPX read timeout.

HTTP status validation happens before SSE iteration. HTTP, content-type,
timeout, and connection failures are converted to explicit model transport
errors. The transport does not reconnect or retry a partially consumed stream.

# Consequences

LF/CRLF framing, comments, multi-line data, network chunk boundaries, and UTF-8
incremental decoding are delegated to maintained libraries. Streaming adds two
runtime dependencies. A read timeout identifies a period with no received
stream bytes, not a total response duration.
