# Headless run artifacts, schema version 1

`gear run` writes a new evaluation record for each invocation. The default is
`<effective-workspace>/.gear/runs/<run-id>/`. Use an exact new directory when a
benchmark harness needs to choose the location:

```sh
uv run gear --workdir ../project run --prompt-file task.txt --run-dir ../results/case-001
```

An existing directory, file or symlink is rejected, even if empty. The directory
is reserved with mode 0700. `run_id` is a fresh UUID independent of the canonical
`session_id`. Canonical sessions continue to use `runtime.session_dir`; changing
the artifact destination does not relocate or change them. The TUI and direct
`run_task` embedding API retain their existing persistence behavior.

## Files and completion protocol

| File | Contents |
| --- | --- |
| `run.json` | Versioned manifest with effective inputs, identity and outcome |
| `metrics.json` | Versioned structured model/tool/context/timing measurements |
| `events.jsonl` | Independent session snapshot followed by run observer events |
| `final.txt` | Exact UTF-8 canonical successful answer, with no added newline |
| `workspace-status.txt` | Starting and ending porcelain Git status, or explicit unavailability |
| `workspace.diff` | Final tracked changes relative to starting HEAD; absent without a HEAD |

Both JSON documents have integer `schema_version: 1`. All JSON object keys are
sorted; ordered conversation, request and tool lists retain execution order.
Timestamps use UTC ISO-8601 with `+00:00`. Durations use `time.perf_counter()`;
timestamps are never subtracted to infer durations.

`run.json.status` is initially `running`, then `success` or `failure`. Each file
is written to an exclusive temporary file, flushed/fsynced and atomically
renamed. The terminal manifest is published **last**. Consumers must inspect its
status rather than infer completion from directory existence or `metrics.json`.
A crash, forced termination or artifact write failure may leave `running` and
incomplete files (including temporary files). This is not a multi-file filesystem
transaction. Artifact failures emit a structured `run_artifacts` error and exit
1; they never print the final answer or publish a successful terminal manifest.

A model, tool or context failure retains exit 3 and writes failure metrics, error
identity and the session/observation snapshot collected so far. Local runtime
I/O retains exit 1. Standard Python/process interruption is re-raised after
attempting a failure snapshot. Failed streams never become `final.txt`.

## Manifest

- `run_id`, `session_id`, `started_at`, `finished_at`, `status`, `error`: identity
  and terminal outcome. Error projection contains type/origin/message, not raw
  HTTP bodies, exception reprs or arbitrary error details.
- `gear`: installed package version and source checkout commit, each nullable
  when unavailable. Commit identifies HEAD, not uncommitted source changes.
- `task`: exact text, `source` (`inline` or `file`), and nullable absolute path.
  File paths preserve the supplied spelling, including symlink aliases.
- `adapter_kind`, `capabilities`, `model`: adapter class, operation capabilities,
  model ID, protocol, streaming/replay mode, sanitized endpoint and SHA-256 of
  that sanitized endpoint. This artifact fingerprint is not a replay credential.
  `reasoning_replay_scope` records protocol/model and the actual adapter scope
  fingerprint only when endpoint sanitization leaves the configured URL intact;
  otherwise its fingerprint is null and `endpoint_identity_omitted` is true.
- `tools`, `enabled_tools`, `web_search`, `web_fetch`: effective enablement and
  credential-free configuration. Enabled names are sorted. `runtime` includes
  limits, workspace, session path, shell network setting and shell Docker image.
  Network policy controls the Docker shell; configured model and web APIs still
  use their configured endpoints.
- `context_budget`: explicit context-capacity/compaction policy.
- `repository_instructions`: ordered dispatched agent requests with iteration
  and the applied files' workspace-relative path, scope and SHA-256 content hash.
  Instruction content is not added to convenience metadata.
- `workspace_git`: availability, reason when unavailable, starting HEAD and branch,
  detached state, starting/ending status, dirty state and diff baseline.

## Trace

Canonical session events retain their existing `created_at`, `session_id`,
`kind`, `payload` envelopes, subject to the redaction policy below. The original
session file can change or disappear without invalidating the snapshot.

Run observations have `schema_version: 1`, `run_id`, `session_id`, `created_at`,
`kind` and `payload`. Canonical and observer sequences are stored separately in
one file (canonical first); timestamps correlate the sequences. They are not a
replacement conversation format and are not replayed into model history.

| Observer kind | Payload |
| --- | --- |
| `model_request_started` | `purpose`: `agent` or `compaction` |
| `model_request_finished` | purpose, monotonic duration, unavailable usage placeholder, nullable error |
| `model_usage` | provider-reported usage for the preceding completed request, observed after canonical response persistence |
| `tool_started` | name, call_id |
| `tool_finished` | name, call_id, monotonic duration, successful, nullable error |
| `iteration_completed` | completed agent iteration number |
| `repository_instructions` | iteration, applied file metadata |
| `context_budget` | existing context diagnostic, phase, iteration, trigger/failure flags |
| `reasoning_replay` | existing replay mode and reused/dropped counts |

Execution is sequential. Requests correlate by ordered start/finish pairs;
`call_id` correlates tool observations with canonical events. Streaming text,
reasoning text and argument deltas are deliberately not recorded.

## Metrics

- `outcome`: status/error, completed iterations and final text availability.
  An iteration completes after tool results or a valid final answer are persisted,
  or a valid empty-response retry is prepared. A failed iteration does not count.
- `model`: actual dispatch count including failed and compaction requests;
  ordered requests with purpose/duration/error and provider-reported
  `input_tokens`, `output_tokens`, `total_tokens`. Missing fields are null; totals
  are null when any request lacks that category. The available per-request values
  are retained. Zero requests yields null token totals. Total tokens are never
  reconstructed from input/output counts. Invalid reported counts fail explicitly.
- `tools`: total calls, success/failure counts, per-name totals and ordered timings.
  Thrown failures, explicit error results, nonzero shell exit and shell timeout
  count as failure, including recoverable failures returned to the model.
- `reasoning_replay`: cumulative reused and dropped counts by reason plus total
  dropped. These are diagnostic counts, not unique encrypted-item identities.
- `context`: completed automatic/manual checkpoints, budget failures, all
  decision estimates, max/mean estimates and post-compaction estimates. Estimates
  include preflight/summary/post-compaction decisions and are not provider tokens.
  Disabled budgeting has empty estimate lists and null max/mean.
- `timing`: monotonic wall duration from artifact initialization through snapshot
  collection, aggregate request duration and aggregate tool duration. Final file
  writes after metric calculation are excluded. All durations are seconds.

## Workspace limits

Git inspection disables optional index locks and external diff/textconv helpers.
It never commits, resets, cleans, stages or changes branches. The diff is limited
to tracked paths in the effective workspace and compares against the **recorded
starting HEAD**, even if HEAD changes during execution. Starting and ending status
cover the repository. Staged and unstaged changes are included; untracked file
contents and binary patch contents are not captured. Default artifact/session
paths may appear as untracked files unless ignored by the repository.

A dirty-start diff can include pre-existing user edits. `dirty`,
`includes_preexisting_changes` and both statuses make this explicit: this is not
an exact delta from an arbitrary dirty filesystem. Unborn Git repositories have
status and branch metadata but no HEAD diff. Non-Git workspaces and unavailable
Git binaries record why Git data is unavailable and do not fail the task.
Unexpected Git errors are explicit artifact failures, not non-Git fallbacks.

## Privacy

The central serializer removes configured model/search/fetch credentials,
secret-named environment values, credential query values, URL userinfo/fragments,
recognized credential fields and Authorization headers. Environment values are
used only for redaction and are never included as configuration. Endpoint
fingerprints are calculated after sanitization, never from credential material.
Recognized secrets in task or final text cause an explicit artifact error instead
of silently altering text that promises exact preservation.

Protocol objects with `type: function_call` are projected by decoding their
`arguments` as a JSON object exactly once, redacting its fields, and serializing
it back into the argument string. Safe argument strings retain their original
whitespace and key order. Canonical session storage and actual model/tool inputs
are never changed. Already-decoded `tool_call.payload.arguments` and
`tool_result.payload.result` are data, not additional protocol objects. Strings
inside arguments (including file content) receive text redaction; they are not
recursively JSON-decoded. Known secrets are matched in raw/URL-encoded form and
one JSON string-escaping layer of each form (ASCII and UTF-8 spellings), covering
quotes, backslashes, control characters and non-ASCII characters. The outer
protocol string is decoded before these text rules are applied.

Each protocol argument string is limited to 1 MiB of UTF-8, and the projection
has a maximum value depth of 64, measured from the artifact object's root and
including decoded arguments. Invalid JSON, non-object arguments, duplicate
object keys, invalid UTF-8 and non-JSON numeric constants are explicit
`artifact_json_invalid` failures. Size/depth excess is an
`artifact_json_limit_exceeded` failure. Repeatedly encoded arbitrary strings are
not decoded. A rejected projection writes no unsafe event copy and cannot
publish a terminal success manifest; the CLI reports the artifact failure with
exit 1, even if the original task also failed. Its canonical session remains
available for debugging. Error messages do not echo rejected input.

Task and final text are checked with the same known-secret text rules. A match
raises `artifact_sensitive_text`; otherwise the original text is preserved
verbatim, without JSON normalization.

The session copy and Git text artifacts are redacted. Original canonical sessions
are not rewritten. `events.jsonl` may therefore differ from the original session
only through privacy filtering. Opaque provider `encrypted_content` in canonical
model events is retained under the existing session policy; it is not copied to
`run.json`, metrics or observer events. Treat the trace as sensitive conversation
data. This policy is not a content classifier for arbitrary confidential text or
unrecognized/encoded secrets; artifacts remain local.

## Verification

Run `uv run pytest -q` for the complete suite, including function-based tests.
Integration tests use a local scripted HTTP endpoint and real session/filesystem/
Git operations. Monotonic time is the only mocked value in timing assertions.
