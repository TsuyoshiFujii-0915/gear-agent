# Title

Scope Encrypted Reasoning Replay to an Exact Configured Connection

# Status

superseded by ADR-0010

# Context

Responses-compatible models can return opaque `encrypted_content` in reasoning
output items. Replaying the item can preserve reasoning continuity when Gear
Agent manually supplies conversation history, but the opaque state may be
invalid for a different protocol, endpoint, or model. Retrying after a provider
rejects incompatible encrypted state would detect the problem too late and
would make the effective history implicit.

Session events are append-only and may be resumed with a different current
configuration. Existing JSONL files also contain raw `model_response` payloads
without metadata that identifies the connection which produced them.

# Decision

Represent the source of replayable opaque state with a `ModelReplayScope`
containing:

- the exact protocol family, currently `responses`;
- a SHA-256 identity of the exact configured endpoint URL;
- the exact configured model identifier.

Opaque reasoning content is reusable only when encrypted replay is enabled and
all three stored fields exactly equal the active scope. Any field difference or
missing legacy metadata is incompatible. The endpoint is deliberately not
normalized: even a textual URL change creates a different fingerprint and
drops opaque state conservatively. The fingerprint prevents endpoint URLs from
being written in plaintext to replay metadata. Encrypted replay rejects
endpoint URLs containing user information, query parameters, or fragments so
the fingerprint is never derived from embedded credentials.

Agent-loop `model_response` events created with encrypted replay enabled are
persisted in an envelope that atomically associates the raw response with its
source scope. A fixed `gear-agent.model-response.v1` schema marker distinguishes
the envelope from legacy raw response payloads, which remain a supported read
format. When replay is disabled or a scope is not compatible, context
construction copies the reasoning item and removes only `encrypted_content`;
it preserves summaries and all other portable conversation items. It does not
rewrite stored session events.

# Consequences

Reasoning continuity survives later turns and resumed sessions only for the
same exact configured replay scope. Changing the protocol, any endpoint URL
character, or the model ID removes opaque state before the request instead of
waiting for a provider error. Structured progress diagnostics report reused
and dropped item counts without containing the encrypted payload.

Old sessions remain readable without migration, but their opaque reasoning
state cannot be trusted and is removed from active context. Compaction continues
to use the latest checkpoint boundary and strips opaque reasoning before
serializing effective events into its summary prompt. The raw append-only audit
history remains intact.
