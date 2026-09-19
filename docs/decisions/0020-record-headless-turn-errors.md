# Title

Record terminal headless failures in the existing session history

# Status

accepted

# Context

ADR 0019 reuses the production loop and JSONL store, but the TUI records terminal
errors outside AgentLoop. The headless boundary must also record those failures
so inspection and TUI resume explain why the task stopped. Error details may
contain remote bodies or credentials, and writing an error can itself fail.

# Decision

Catch terminal GearError at run_task's execution boundary and append exactly one
existing turn_error event. Keep payload.text compatible with the TUI renderer and
include a structured error projection containing type, origin and message.
Share that projection with CLI stderr; omit arbitrary details and redact resolved
API keys and the configured endpoint URL. Do not persist an error for recoverable
tool failures handled successfully inside AgentLoop.

Re-raise the original task exception after recording it. If the error record
cannot be written due to filesystem or encoding failure, preserve the original
exception as the raised error and explicitly chain a safe turn_error_write_failed
GearError. The CLI reports this secondary failure alongside the original error
and retains exit code 3. Do not retry the task or the failed append.

# Consequences

CLI and Python callers produce the same inspectable failure history without a new
session format. A storage failure cannot overwrite the task's identity, and is
observable through exception chaining or stderr. Existing task/model/tool payloads
and the TUI's own execution behavior remain unchanged.
