# Title

Run headless tasks through the shared production harness

# Status

accepted

# Context

Issue #18 requires non-interactive execution with the same model adapter, tools,
repository instructions, context budget and JSONL sessions as interactive runs.
The TUI currently owns startup composition but not the agent loop. ADRs 0015,
0017 and 0018 define the production execution policies to preserve.

# Decision

Use `gear run` with exactly one of `--prompt` or `--prompt-file`. Resolve config
and runtime using the existing CLI path and construct both frontends through
`build_agent_runtime`. Headless execution uses the existing silent event sink,
a fresh session ID and one normal `AgentLoop.run_turn`. Import Textual only on
the interactive path. Do not duplicate execution or introduce task-level retries.

Print `session_id=<UUID>` on stderr before execution. Print only the canonical
successful answer, followed by a newline, on stdout. Emit no progress by default.
Keep argparse's exit code 2 for invalid arguments; use 1 for setup, configuration
and local runtime I/O failures, and 3 for structured model/agent/tool/context
failures during the turn. Recoverable tool errors keep the existing model-feedback
semantics. Standard process interruption remains unchanged.

Diagnostics expose error type, origin and message, omit arbitrary error details
(which may contain response bodies or URLs), and redact resolved credentials.
Expose effective inputs through an in-memory RunSpec/RunResult; omit keys and raw
endpoint URLs. Reuse credential-redacted endpoint fingerprints, excluding URL
userinfo and fragments. Do not introduce benchmark files or a separate session
format. Session persistence continues to contain canonical task/model/tool text.

# Consequences

Headless tasks exercise streaming, repository context and automatic compaction
through the exact production services. Scripts receive a stable answer channel
and can use the session ID for normal inspection/resume. Runtime validation is
explicit before headless dispatch. Full benchmark artifacts and headless resume
remain outside this change.
