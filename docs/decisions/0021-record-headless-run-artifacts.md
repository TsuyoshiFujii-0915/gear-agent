# Title

Record headless evaluation artifacts through a neutral run observer

# Status

accepted

# Context

Issue #19 needs independent run identities, reproducibility metadata and metrics,
including failed requests and automatic summary requests. ADRs 0015, 0017, 0018
and 0019 retain authority over execution and canonical session semantics.

# Decision

Use an optional provider-neutral RunObserver at the loop and compaction execution
boundaries. Existing embedding and TUI callers omit it. Measure actual model and
tool calls with a monotonic clock, including failures. Obtain provider usage from
the model result interface. A headless collector consumes these observations and
existing content-free context/replay events; it never records streaming deltas.
Repository metadata describes the exact instruction snapshot used for dispatch.

Each CLI headless invocation reserves a new run directory, independent of session
storage. Write a versioned running manifest first and atomically replace it with
a terminal manifest only after trace, metrics, final answer and workspace records
are written. Existing destinations are errors. Interrupted/incomplete writes must
never look successful. Artifact failures are distinct setup/I/O errors (exit 1);
turn failures retain exit 3. Canonical session persistence remains independent.

Apply a shared serialization policy to artifacts: omit arbitrary error details,
redact configured credentials, secret-named environment values and credential
fields/headers, and sanitize endpoint identity before hashing. Exact task/final
text that would require redaction is rejected explicitly rather than silently
changed. Session snapshots retain canonical event envelopes and may retain opaque
provider encrypted reasoning state; convenience metadata never contains it.

Capture Git status before creating artifacts and after execution using read-only
Git commands with optional locks disabled. Diff against the recorded starting
HEAD, explicitly retaining dirty-start metadata. Unborn and non-Git workspaces
have no HEAD diff. Do not claim a filesystem delta or include untracked contents.

# Consequences

Headless artifacts can be analyzed after session deletion. Usage totals are null
when any request lacks that category; per-request reported values remain usable.
Summary requests count as model work but not as agent iterations. Filesystem
failure may leave an incomplete directory whose manifest is running or an
explicit artifact failure. Atomic manifest replacement is the completion marker,
not a multi-file transaction or a promise of recovery after disk failure.
