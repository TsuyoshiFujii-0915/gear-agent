# Title

Bound diagnostic network I/O in a supervised worker process

# Status

accepted

# Context

The synchronous Responses client validates JSON and SSE completion. Its timeouts
limit individual network operations. DNS resolution and a stream that keeps
receiving heartbeats can outlast the deadline for an explicit CLI check.

# Decision

Run the diagnostic network operations in one spawned process under a monotonic
deadline in the calling process. The worker sends allowlisted observations over
a local pipe, including the HTTP status when response headers arrive. On expiry
the caller terminates and joins the worker and reports the active check.

# Consequences

The report retains earlier observations when a later operation times out. The
diagnostic pays process startup overhead. The agent continues to use its normal
Responses client and per-operation timeout policy. The diagnostic cannot guarantee
cancellation of computation already accepted by the remote server.
