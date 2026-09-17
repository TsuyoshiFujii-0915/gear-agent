# Title

Measure effective model context before each agent dispatch

# Status

accepted

# Context

Issue #17 requires deterministic automatic context management across providers,
iterations and resumed sessions. ADRs 0015 and 0017 define the semantic adapter
boundary and fresh repository instructions. Existing configuration does not
specify model capacity, and opaque history must retain ADR 0010's replay scope.

# Decision

Add a provider-neutral ContextBudgetManager with a TokenEstimator protocol.
Measure the actual input, instructions and tool schemas passed to the adapter,
partitioning that same input by its current turn boundary for diagnostics. Do
not estimate from a separate reconstruction of the session log. The initial
estimator charges one token per UTF-8 byte of compact JSON plus 25%, rounding
up per component, and an additional 256-token request framing allowance.
This is a conservative estimate, not exact tokenization or a guarantee for an
arbitrary tokenizer or opaque state encoding. No network token counting or
provider usage substitution occurs.

An absent context_budget table explicitly disables automatic policy to preserve
legacy behavior. Enabling it requires a configured context window and reserved
output/reasoning tokens. The regular input limit defaults to window minus
reserve; an explicit lower max_input_tokens can trigger earlier compaction.
No model capacity is inferred from names or adapter operation capabilities.

Before each agent iteration, publish a structured diagnostic. If over budget,
perform at most one textual compaction, rebuild input using the persisted
checkpoint and the exact current user request, restore any finalization retry
instruction, reload repository instructions and measure again. An oversized
rebuilt request raises a structured context_budget_exceeded error. Disabled
mode preserves the prior request and event behavior.

Compaction remains a separate service and reuses effective-event selection,
opaque-state sanitization and append-only checkpoint semantics. Automatic
checkpoints add trigger=automatic metadata. The automatic summary request is
also measured before dispatch, using the physical window minus reserved tokens
rather than the earlier regular trigger. If it cannot fit, fail explicitly
without truncation, recursive compaction or sending an oversized request.
Manual /compact retains its existing explicit behavior.

# Consequences

Diagnostics expose instructions, repository context, history (including any
checkpoint), current turn and synthetic input, schemas, framing, headroom and
total estimates with limits, phase and outcome. They contain counts, not content,
and are emitted per request decision, not per stream fragment. TUI rendering is
optional; the TUI accepts the events without changing the conversation display.

Raw JSONL history remains intact, pre-checkpoint opaque state cannot reappear,
and resume uses current configuration. Existing per-turn tool outputs remain
untruncated; stored-history truncation still applies only during history replay.
The budget measures the representation actually used on each path. Very large
jumps may exceed even the summary request capacity and require an explicit
configuration or input change. Users should configure an early trigger with
room for textual summarization overhead.
