# Title

Share explicit generation options across all Responses requests

# Status

accepted

# Context

Issue #37 adds optional reasoning effort and an output cap to the existing
Responses adapter. ADRs 0015, 0018, 0021 and 0034 require one configured adapter,
effective-request accounting, credential-free observations and completed results.
Applying settings only to ordinary answers would hide differences in summary
behavior, while silently reducing or removing them would change user intent.

# Decision

Bind optional `reasoning_effort` and `max_output_tokens` to ModelConfig, with
None preserving legacy omission. Validate the six supported effort strings and
positive integer caps for both TOML loading and direct construction. Build the
corresponding Responses fields once through a shared projection used by dispatch
and the adapter's context-budget accounting. All requests from that adapter,
including manual/automatic summaries and explicit Jev summary fallback, inherit
the same settings without overrides or protocol fallback.

With automatic context management enabled, require reserved_tokens to cover an
explicit output cap at application configuration and embedded-loop construction.
Measure the added JSON fields as input overhead in addition to the existing
framing allowance; generation reserve remains separate. Omitted caps remain
unknown. Effective run metadata records nullable configured values, not inferred
provider acceptance. A 400 with generation options returns a content-free error
with configuration guidance; it never retries after changing options.

# Consequences

Existing TOML, direct ModelConfig calls and omitted-option payloads retain their
behavior. The adapter exposes its explicit request options for budget accounting;
the same budget manager measures ordinary, rebuilt and summary requests. A short
cap can fail any summary. ADR 0034 rejects incomplete output before checkpoint
persistence or tool execution. No summary-only settings, model defaults, Jev
policy changes, deployment branches or new adapters are introduced.
