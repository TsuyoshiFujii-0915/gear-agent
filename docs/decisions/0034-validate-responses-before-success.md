# Title

Validate Responses completion before exposing model success

# Status

accepted

# Context

The JSON client currently returns HTTP 200 objects without inspecting their
completion status. This permits partial answers, tool side effects and partial
summary checkpoints. The SSE assembler rejects unsuccessful terminals, but its
diagnostics copy arbitrary provider errors. Strata v0.2.7's Qwen Responses path
exercises both JSON and SSE and returns incomplete when output reaches its cap.

# Decision

Validate completed Responses objects in the model layer before returning them
to the semantic adapter. Share status/error validation between the JSON client
and SSE assembler. Explicit incomplete and failed states retain their existing
structured error categories; all other explicit non-completed or malformed
statuses fail. Reject non-null errors even in HTTP 200/completed responses, and
reject contradictory non-null incomplete details on otherwise successful ones.

Deliberately preserve status-omitted JSON compatibility without adding a status
field. SSE continues to require a response.completed event with completed status.
Failure events remain failures even if their response claims completed. Validate
the assembled result as well as the terminal snapshot. Only fixed allowlisted
provider codes/reasons may appear in failure diagnostics; do not copy messages,
response IDs, inputs, URLs or error bodies.

# Consequences

Ordinary turns and manual/automatic summary requests fail before persistence,
tool dispatch or missing-final-answer retry. Existing headless turn_error
recording remains responsible for failed turns. No provider-specific adapter,
polling, retries or protocol fallback is introduced. ADRs 0012 and 0015 continue
to govern assembly and adapter boundaries; this correctness change adds a
precondition to the canonical response they expose. Session/checkpoint formats,
Jev policy, configuration and reasoning replay behavior remain unchanged.
