# Title

Assemble Responses streams centrally into canonical responses

# Status

accepted

# Context

AgentLoop, history replay, storage, and tool extraction already consume one
completed Responses object per model call. Giving each upper layer its own
stream interpretation would duplicate tool-loop behavior and couple it to
provider wire events.

# Decision

Add a model-layer `ResponsesStreamAssembler`. A valid `response.completed`
response with complete output is canonical. When a compatible endpoint omits
terminal output detail, completed output-item events are assembled in output
index order and combined with available response metadata. Deltas are retained
only for progress publication and for completing an item whose final item event
legitimately omits accumulated text or function arguments.

Function-call argument fragments are accumulated by stable output index and
item identity without JSON parsing. The existing completed-response extraction
validates and parses the final argument string once.

`response.failed`, `response.incomplete`, stream `error`, and EOF before a
successful terminal event fail explicitly. Unknown event types are ignored so
that future additive events do not invalidate an otherwise valid stream. No
stream failure triggers a non-stream retry.

# Consequences

Streaming and non-streaming calls return the same response shape to AgentLoop.
The model layer can publish provider-neutral text, reasoning, argument, and
completed-item progress events without exposing raw SSE payloads. Unknown
events provide no progress signal until Gear Agent intentionally models them.
