# Title

Execute models through a configured semantic adapter

# Status

accepted

# Context

Issue #15 requires the loop and compaction to use an execution boundary while
preserving Responses requests, session payloads, replay, and failure behavior.
ADRs 0010, 0012, and 0013 remain applicable. ADR 0001's prohibition on adding
an abstraction applied to that layout-only change.

# Decision

Bind the effective ModelConfig to a ResponsesModelAdapter at construction.
Expose a ModelAdapter protocol for execution, history preparation, replay policy,
and continuation items. Expose a ModelResponse protocol for persisted payload,
text, tool calls, and replayable output. The existing canonical history format
remains the interchange representation; this is not a session format redesign.
Responses decoding stays behind the result interface and occurs at the same
points as before, including persisting a received response before interpreting
its output and skipping text decoding when tool calls exist.

Use one factory in CLI composition, sharing the configured adapter with
compaction. Keep ModelClient and its transport/assembler behavior unchanged.
Do not add a protocol selector or provider detection.

Capabilities are frozen dataclass fields describing operations implemented by
the adapter, not guaranteed remote model support or enabled request options.
Streaming, opaque replay, textual compaction, and function calling are supported;
native compaction is not. Runtime rejection propagates without retry/fallback.
Parallel calling is omitted because the configuration cannot establish support.
Dataclasses.asdict produces a JSON-serializable description without credentials.

# Consequences

Loop and compaction no longer construct protocol requests or decode responses.
Existing config and JSONL sessions require no migration. Neutral progress sinks
still attach to the model client behind the adapter; the TUI retains its existing
agent-event display. No new providers or speculative configuration are added.
