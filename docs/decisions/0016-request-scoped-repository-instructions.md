# Title

Build repository instructions from the current workspace for each agent request

# Status

accepted

# Context

Issue #16 requires hierarchical AGENTS.md instructions without filesystem traversal
in the agent loop or prompt construction in tools/TUI. ADR 0015 places model
execution behind a configured adapter; repository context belongs before that
boundary. A startup snapshot becomes stale during editing and resume, and the
first request cannot know which files will be used.

# Decision

Use a dedicated RepositoryContext, explicitly constructed with the effective
workspace after CLI overrides. Before each agent model request it discovers,
reads, bounds, and renders current instruction files. Always include root
AGENTS.md. Add directory chains from completed tool activity in the entire
session log, including activity before compaction:

- file_read/file_write: the returned file path's containing directory;
- apply_patch: the containing directories of returned changed_files;
- glob: returned file/directory matches, using their reported type;
- grep: the containing directories of returned matching files;
- shell: the matching tool call's workdir, even for nonzero exit or timeout.

Tool results containing an error and calls without results contribute no scopes.
Unknown tools contribute no scopes. Do not infer paths from shell commands,
stdout, free text, glob patterns, or unreturned search matches. Required metadata
for recognized successful tools must be valid or produce a structured error.
Previously encountered scopes remain applicable for the session. Sibling rules
apply only to their own subtrees; deeper rules take precedence within their scope.

Resolve paths against the effective workspace, reject absolute/parent-traversal
and resolved outside paths, and use physical paths for internal aliases. Never
traverse parent directories above the workspace to find instructions. Open
instruction files through workspace-anchored directory descriptors without
following symlinks, and accept only regular files. AGENTS.md symlinks, including
internal and dangling links, are explicit errors. Missing files or deleted scoped
directories are normal; other filesystem and UTF-8 errors are structured GearError
values. Bound reads to 32 KiB per file and 128 KiB total raw UTF-8 bytes per
request; limits are inclusive and excess is an error, never truncation.

Deduplicate by physical scope and order by directory depth, then POSIX relative
path. Render XML-escaped content and path/scope metadata in separate
repository_instructions blocks following the unchanged Gear base instructions.
Re-read on every request, including retries and resumed sessions; retain no
content cache. No filesystem content is appended as an instruction chat message
or new session event. ResponsesModelResponse omits the provider's echoed
request-level instructions field from persisted payloads; output and replay
metadata retain their existing representation. Old session payloads remain
readable. Explicit tool reads/writes and model-generated text keep their existing
persistence semantics. Compaction summarizes history with its existing dedicated
instructions; it does not load repository rules as summarization instructions.

# Consequences

Root rules are available on the first request and deeper rules on subsequent
requests after observable tool activity. Direct edits issued before discovering
a path cannot retroactively follow newly loaded instructions. Arbitrary shell
file accesses are not inferred, and broad searches can include multiple sibling
scopes. Repeated reads add bounded I/O and the accumulated scope set can exceed
the total content limit explicitly. Existing sessions require no migration, and
resume/compaction cannot permanently freeze repository instructions.
