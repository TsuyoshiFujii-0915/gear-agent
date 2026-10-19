# Compaction comparison

Run from the project root with a working Responses-compatible model in the Gear
configuration and `TYPESAFE_API_KEY` in the environment:

```sh
uv run python -m examples.compaction_comparison \
  --generate \
  --config .gear/config.toml \
  --output test_outputs/compaction-comparison/NEW-RUN \
  --jev-model jev-1.13.0 \
  --jev-timeout 10
```

The output must be a **new, Git-ignored directory inside this repository**.
`test_outputs/` is already ignored. Never commit the generated session, provider
responses, or local configuration. This command makes real main-model and Jev
API requests; it does not use fabricated model responses or scores.

The example creates an isolated fictional Python project, then runs two actual
`AgentLoop` turns with only `file_read` enabled. The first turn investigates an
old documentation build failure and a retry-limit bug. The second closes the
documentation task and reads the current retry ticket. A third, pending user
message requests the retry fix. Generation stops if a tool fails or the expected
fixture files were not read.

The exact same source events, including that pending message, go to the
production `SummaryCompactionStrategy` and `JevCompactionStrategy`, once each.
Jev uses PR #26's default policy: threshold 0.2, one recent turn protected,
160-character previews, and the default state/request/batch limits. No fallback
or provider retry is enabled. The main model uses the supplied configuration's
model, replay mode, streaming setting, and timeouts. Failure leaves partial
evidence and `status.json` with `status: stopped`; do not treat that as a result
or silently rerun with different settings. Setup failures exit before execution.

Open `report.md` in the output directory. It links to:

- `sessions/source.jsonl`: original, append-only session, with real model output
  and real tool results; `workspace/` contains the input files.
- `baseline-input.json`: model-visible input before compaction.
- `summary/summary.txt` and `summary/model-input.json`: full summary and replay.
- `jev/model-input.json`: selected replay, with original messages and retained
  tool results; `jev/jev-requests.json` contains actual state, questions, scores,
  and provider-reported usage. This deliberately detailed evidence is only for
  the isolated example, not production telemetry.
- `{summary,jev}/source.jsonl`: independent copies with their checkpoints
  appended; the original file is checked for byte equality after comparison.
- `results.json`: source SHA-256, code revision, settings, durations, sizes,
  strategy metrics, and each tool result's before/after disposition.
- `generation-observations.jsonl` and `summary/observations.jsonl`: actual
  provider usage and measured request durations, separate from size estimates.

The compared inputs include the current message exactly once and use identical
instructions/tool schemas. Byte sizes measure compact UTF-8 JSON. The reported
request estimate is Gear's conservative `ByteTokenEstimator`, including schemas,
instructions, and framing; it is **not actual tokenizer usage**. A removed tool
pair in the summary input means its content was replaced by prose, not necessarily
that its information was lost. Compare the summary text against the selected
results to judge what remains available verbatim.

This is a strategy-level observation on one history. It does not force automatic
budget triggering, assume a model context capacity, test an active tool chain,
or measure downstream answer quality. The preserved recent turn is not evidence
of a high Jev score. Timing includes different providers and is not a general
speed/cost benchmark. Model responses and scores may vary on subsequent runs.
Raw history is retained on disk; compaction reduces replay context, not log size.

To repeat the comparison on **exactly the same history**, replace `--generate`
with `--source-run test_outputs/compaction-comparison/PREVIOUS-RUN` and specify a
new `--output` directory. This copies the original JSONL byte for byte and reuses
the workspace files; it does not generate new turns or reuse previous scores.
The recorded `source_sha256` must match between runs. Generation observations
remain in the original run referenced by `source_run`.

Ensure the main model's **loaded context length**, not just its advertised model
maximum, can accommodate the textual summary request plus generated output and
reasoning. Summary serializes audit events into JSON; its input can be larger
than normal conversation replay. With LM Studio, `lms ps` shows the actual loaded
capacity. Changing that capacity requires reloading the target model with
`lms load MODEL --context-length TOKENS` after unloading it. The example does not
change server settings or infer a model capacity automatically.

## Jev threshold sweep

Use the same saved history for an independent live Jev call at each threshold:

```sh
uv run python -m examples.jev_threshold_sweep \
  --config .gear/config.toml \
  --source-run test_outputs/compaction-comparison/PREVIOUS-RUN \
  --output test_outputs/compaction-comparison/NEW-SWEEP \
  --thresholds 0.2 0.3 0.5 0.7 \
  --jev-model jev-1.13.0 --jev-timeout 10
```

This does not regenerate the conversation or repeat the main-model summary.
Every threshold starts from the original events. The threshold is applied locally
after scoring; it does not change the state or questions sent to Jev. Responses
may vary, so the report includes the actual scores for every run. No retries or
fallbacks are enabled, and the first failure stops the sweep. Each case saves its
checkpoint, model-visible input, per-tool disposition, usage, and timings in the
ignored output directory. Existing policy defaults remain unchanged.

`score >= keep_threshold` means keep. A higher threshold is more aggressive:
keep the full pair when the result score passes; otherwise retain the call with
a shorter result preview when only the call score passes; drop the pair when
neither passes. A preview is only substituted if smaller than the original.
Protected recent interactions remain regardless of threshold. This experiment
measures replay reduction, not downstream task accuracy or a recommended policy.
