# Title

Persist physical repository scopes with completed tool results

# Status

accepted

# Context

PR #21 review identified that resolving historical raw paths on every request
can redirect a completed activity's scope when an alias is retargeted or deleted.
ADR 0016's physical-scope retention semantics therefore require a durable record
of the directory used at execution time, separate from fresh instruction content.

# Decision

Supersede ADR 0016's scope reconstruction policy. Retain its instruction discovery,
ordering, rendering, limits, boundary checks, freshness, and persistence policies.

The six scoped tools return an additive `resolved_scope_paths` list of physical,
workspace-relative directory strings with each completed result. File and shell
tools derive it from the already-resolved path used for execution. Search tools
capture scopes alongside matches and include only the returned matches. Patch
results capture changed-file scopes before returning to the agent loop. The
existing tool_result persistence records this metadata without a new event kind,
absolute host paths, credentials, or instruction text.

RepositoryContext treats a present list, including an empty list, as authoritative.
It validates paths lexically and reads instruction chains using its existing
workspace-anchored, no-follow directory traversal. It never re-resolves a recorded
physical scope through current symlinks. If a recorded physical directory itself
becomes a symlink, construction fails explicitly; deletion remains normal.
Instruction contents are still re-read on every request.

Absence of `resolved_scope_paths` explicitly identifies the legacy result schema.
Only those results use ADR 0016's raw-path resolution against the current workspace;
the historical physical target cannot be recovered from data never recorded.
A malformed present value is an error and must never select legacy behavior.
This compatibility rule allows old sessions to remain readable without pretending
their alias history is recoverable.

# Consequences

Later tools in the same model iteration, later requests, compaction, and resume
cannot redirect a completed activity by changing its original alias. Tools carry
filesystem metadata but do not construct prompts or discover instruction files.
Their existing result fields and agent/store interfaces remain intact. Legacy
results retain their documented limitation, while new results preserve scope
identity and independently refresh instruction content.
