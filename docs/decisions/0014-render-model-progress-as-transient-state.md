# Title

Render model progress as transient TUI state

# Status

accepted

# Context

Streaming model requests publish frequent provider-neutral text and reasoning
events before the completed response is available. Session history must continue
to use the completed model response and assistant message as its source of truth.
Appending each fragment to the chat log would create duplicate messages, expose
partial Markdown to the Markdown renderer, and make restart behavior differ from
the live session.

# Decision

Translate only assistant text deltas and provider-exposed reasoning summary
deltas into session- and iteration-scoped agent events. Private reasoning text,
function argument fragments, and completed output items are not presentation
events. In particular, completed items are not forwarded because they may carry
opaque encrypted reasoning state.

Render public model progress in one dedicated transient Textual widget. Batch
queued agent events into one UI-loop update, render partial content as plain
text, and reset the widget at model-iteration and tool boundaries. Reasoning
summary progress remains visible only while its model request is active. On
success or failure, discard the transient widget and rebuild the chat from
persisted canonical events; completed assistant text then uses Markdown.

# Consequences

Long model requests show ordered incremental progress without generating one
widget or stored event per fragment. Public reasoning summaries are visually
distinct during execution but do not clutter resumed history. A failed stream
cannot make partial text appear to be a completed answer, and encrypted or
private reasoning never crosses the agent presentation boundary.
