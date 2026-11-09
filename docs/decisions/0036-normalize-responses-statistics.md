# Title

Normalize request statistics at the Responses terminal boundary

# Status

accepted

# Context

Issue #47 needs observations for a later request log and TUI. The legacy
`ModelUsage` accessor validates three API counts on demand. ADRs 0012, 0015
and 0034 govern stream assembly, semantic decoding and completion checks.
Strata supplies timing metadata on the SSE envelope, which is outside the
canonical response consumed by those paths.

# Decision

Attach an immutable `ResponseStatistics` to each received result. The semantic
adapter exposes it as `result.statistics`. A `ResponseFailure` carries the same
projection when a terminal response exists. A standalone stream error has no
snapshot. Error details keep their existing allowlist.

Read the standard usage object for every connection. For the timing extension,
recognize the field layout documented for Strata v0.2.7 Qwen. The schema label
identifies the interpretation of those fields. URL, deployment, model name and
provider identity are unnecessary. For JSON the roots are `usage` and `timings`.
For SSE they are `event.response.usage` and `event.timings`. When the outer
timings key is absent, accept `event.response.timings`. If both exist, the outer
block supplies the entire timing observation, including its missing or invalid
fields. Read only the terminal event for statistics.

| Normalized metric | Wire field or calculation | Unit |
| --- | --- | --- |
| Input, output | `usage.input_tokens`, `usage.output_tokens` | tokens |
| Total | `usage.total_tokens`, or input + output when absent and both counts are valid | tokens |
| Reasoning | `usage.output_tokens_details.reasoning_tokens`, a subset of output | tokens |
| Cached input | `usage.input_tokens_details.cached_tokens` | tokens |
| Cache writes | `usage.input_tokens_details.cache_write_tokens` when present | tokens |
| Input cache hit ratio | cached input / input for this request | fraction, 0–1 |
| Newly processed input, engine decode count | `timings.prompt_n`, `timings.predicted_n` | tokens |
| Prefill, decode duration | `timings.prompt_ms`, `timings.predicted_ms`, each divided by 1000 | seconds |
| Prefill, decode rate | `timings.prompt_per_second`, `timings.predicted_per_second` | tokens/s |

Each metric carries a value, unit and availability. Evidence identifies fixed
schema names, wire paths, original units and validated numeric operands.
Unit conversion retains server provenance and records the conversion formula.
Arithmetic across fields is marked `derived` with its formula and operands.
Only the allowlisted fields enter this projection.

* An absent or null field is `missing`. A reported zero remains available.
  Counts require nonnegative integers. Durations and rates require finite
  nonnegative numbers. Booleans, strings, malformed containers and invalid
  numbers yield `invalid` and a fixed reason. A subset larger than its reported
  parent count is invalid. Dependent metrics propagate invalid operands.
* Unavailable metrics have null values. A cached count is usable on its own.
  A ratio requires a known positive input count. Zero input makes the division
  inapplicable. Cache writes contribute only to their own metric.
* A reported rate takes precedence over a recomputation, including when the
  engine count and duration imply a different rate. Zero phase work or zero
  phase duration makes the rate inapplicable. Only an absent rate may be
  derived, using the same phase's engine count and server duration. Invalid
  reported rates retain their diagnostic. API usage, client latency and
  stream fragments supply no engine throughput operands.
* Strata's API output count, engine decode count and server rate can describe
  different accounting. Preserve each independently. Reasoning is already
  included in output, which may also include tool arguments.

## Fixture evidence

`tests/test_response_statistics.py` uses synthetic terminal payloads with
small numeric values and deliberate omissions. These are schema and adapter
tests. Live model verification is outstanding.

* OpenAI documents cached reads and cache writes under `input_tokens_details`.
  The fixture exercises both and the reasoning breakdown.
  [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
  · [Responses](https://developers.openai.com/api/reference/resources/responses/methods/create)
* Azure's v1 Responses endpoint uses the common response contract. Its partial
  usage fixture exercises a missing input breakdown.
  [Azure Responses](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses)
* LM Studio 0.3.39 shows cached input in Responses usage. The fixture omits
  timing fields. LM Studio native API stats supply no evidence for this request.
  [LM Studio Open Responses](https://lmstudio.ai/blog/openresponses)
* Strata v0.2.7 Qwen emits sibling timings on the terminal event and promotes
  them to the response root for JSON. The fixture includes intentionally
  divergent reported and calculated decode rates.
  [Responses source](https://github.com/Niko1221/Strata/blob/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d/serve/responses.py)
  · [Benchmark accounting](https://github.com/Niko1221/Strata/blob/v0.2.7/docs/BENCHMARK_PROTOCOL.md)
* Maya v1.1.56 emits the standard breakdown and a fixed zero for cache writes.
  This reported zero carries no evidence of measured write activity. Its
  Responses JSON and terminal event omit timings, so both speeds are missing.
  [Maya Responses source](https://github.com/mw00/project-maya/blob/22fc101f2c109b7e7174c975ff3c2d8e9ce0a867/serve/responses.py)

# Consequences

Callers can observe successful and unsuccessful requests using numeric metadata
alone. Completion validation still gates answers, tool dispatch and summaries.
The legacy usage accessor retains its validation and returns only reported
counts. Derived values and SSE envelope fields stay outside canonical session
payloads, model inputs and reasoning replay. No additional inference or
monitoring request is needed.
