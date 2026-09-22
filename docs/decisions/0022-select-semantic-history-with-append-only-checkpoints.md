# Title

Select completed historical tool interactions with append-only checkpoints

# Status

accepted

# Context

Issue #25 adds experimental Jev selection alongside the summary compaction in
ADR-0018. A function call in a model response and a separate tool result form
one semantic interaction; dropping audit lines alone does not change replay.
ADRs 0010, 0015 and 0021 require scoped reasoning, adapter isolation, and safe
observations. Current-turn replay must preserve the complete reasoning chain.

# Decision

Use a small compaction strategy interface returning an in-memory checkpoint.
The summary implementation retains the existing summary request and checkpoint
behavior. Jev evaluates only completed historical read-only interactions paired
by call_id. Pin the current turn, the configured recent previous turns, errors,
incomplete/failed turns, and all tools outside the explicit read-only allowlist.
Keep normal user/assistant messages and scoped reasoning envelopes unchanged.

Persist selected effective events in a versioned compaction_selective snapshot.
Snapshots are flattened (never nested), and only the latest summary or selective
checkpoint plus subsequent events is effective. Keep original raw JSONL events.
The event snapshot retains original model-response scope metadata, so ordinary
replay applies the existing endpoint/model policy on every resume. Filter each
multi-call response at call granularity; also remove the matching result and
execution metadata. Truncation replaces only the result with a bounded marker.

Jev receives portable messages, bounded tool inputs, result sizes and status,
never result bodies or reasoning items. Use the official Python SDK with a
pinned model and no automatic retries. Bound state and batched questions
separately with the existing conservative estimator. Reject over-limit state
explicitly; do not silently discard task context to squeeze it into Jev.

Before appending a selective checkpoint, validate call/result ordering and
uniqueness, message preservation and protected interactions, then measure the
actual candidate request including untouched in-memory current-turn items.
The snapshot stores exact model_visible_output strings for active tool results
so resume does not apply the legacy historical truncation to them. Synthetic
finalization instructions are explicit continuation_instruction events in Jev
mode; they are portable input, not a new user-turn boundary. Compare the complete
checkpoint replay with the actual candidate request before committing.
The snapshot includes the persisted current user event; reconstruction must not
append that input a second time. Persist only if the candidate fits the budget.
A configured summary fallback uses the original effective events, at most once,
and records a content-free reason and outcome. Missing configuration retains
legacy summary behavior; manual /compact remains explicit textual summarization.

# Consequences

Retained items remain verbatim, but deletion and result truncation are lossy.
Snapshots cost disk space and preserve scoped opaque state locally, without
sending it to Jev. Prior snapshots cannot resurrect dropped calls. Unrecognized
or side-effecting tools, tool errors and incomplete turns reduce achievable
savings. Oversized Jev state, failed decisions, or insufficient savings can still
fail if fallback is disabled or the summary request exceeds its own capacity.
Jev observations are separate from main-model usage and never store provider
error bodies, prompts, responses or API credentials.
