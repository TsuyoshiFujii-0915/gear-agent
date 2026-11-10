# Title

Share one request lifecycle between the session log, run artifacts and UI

# Status

Accepted

# Context

A single agent turn can dispatch tool continuations, an automatic summary and
a Jev fallback summary. A resumed session starts its iteration count again.
The terminal Responses metadata defined by ADRs 0036 and 0037 belongs to the
request that received it. ADRs 0014, 0021 and 0034 govern presentation, run
artifact completion and acceptance of model output.

# Decision

Generate a UUID for every agent turn and another for each model HTTP dispatch
after preflight. Publish a typed start and one finish with that request ID.
Carry the same turn ID into automatic summaries. A manual summary has a null
turn and iteration. The four purposes are `agent`, `manual_summary`,
`automatic_summary` and `jev_summary_fallback`. Jev scoring uses its existing
strategy observations.

Write each finish as one JSON line in
`<runtime.session_dir>/<session-id>.requests.ndjson`. Session discovery and
conversation replay read the canonical `*.jsonl` files. The sidecar payload and
the headless finish payload share the following schema.

| Fields | Meaning |
| --- | --- |
| `schema_version`, `request_id`, `started_at` | Request schema 1, generated UUID, UTC dispatch timestamp |
| `session_id`, `turn_id`, `iteration`, `purpose` | Session and execution scope |
| `model_identity`, `endpoint_identity`, `stream` | SHA-256 of the configured model name, endpoint fingerprint with userinfo/query/fragment removed and configured key excluded, requested streaming mode |
| `status` | `success`, `incomplete`, `provider_failure`, `transport_error`, `timeout`, `eof`, `cancelled`, `invalid_response` or `execution_error` |
| `statistics`, `statistics_reason` | Terminal numeric projection from ADRs 0036 and 0037, including units, sources, formulas, availability and reasons. A missing terminal report has null statistics and reason `interrupted_before_report` |
| `client.total`, `client.first_content`, `client.first_answer` | Each has `value`, `unit` (seconds), `origin` (`client_measured`), `basis` and nullable `reason` |
| `duration_seconds`, `usage`, `error` | Run compatibility fields. Duration is the client total. Usage contains the reported core counts for a successful response with valid core usage. An unsuccessful request carries a safe error category |

Use the monotonic clock immediately before the transport call and stop it when
the adapter returns or raises. The duration includes receiving the complete
response and excludes tool execution and sidecar I/O. For SSE, inspect decoded
events before forwarding progress to the synchronous presentation sink. Record
the first nonempty generated text, refusal, reasoning or function arguments, and the
first nonempty output text separately. The basis is the actual event type,
including done, item or terminal snapshots. These are client arrival times
subject to transport buffering and scheduling. The engine's prefill and decode
metrics come only from the server report.

JSON has `non_streaming` arrival times. An absent arrival on a successful SSE
response has reason `not_applicable` (including first answer for a tool-only
result). An unsuccessful request uses `not_observed` for absent arrivals when
a terminal report was received, or `interrupted_before_report` otherwise.
All these values are null. Terminal statistics survive a failed or incomplete
response and a subsequent transport interruption. Completion validation still
controls admission to conversation history and tool execution.

# Consequences

The TUI can keep the latest agent request in memory and identify finishes by
request ID. Headless metrics aggregate the run's own finish events. Consumers
joining those views to the sidecar deduplicate by request ID. Cache coverage
and weighted operand sums are grouped by purpose. All-request legacy token
totals remain unknown if any corresponding count is unknown. Engine speeds
are stored individually.

Readers can open an older run's existing metrics or replay its original
session directly. Absent request IDs, client measurements and server fields
remain unknown. Loading old history never backfills observations. A sidecar
write failure produces a content-free diagnostic with the request ID. Headless
artifact completion continues to follow ADR 0021. A process killed before
the finish publication can leave a request without a terminal record.
