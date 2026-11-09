# Title

Preserve Strata's reported decode count and rate independently

# Status

accepted. Supersedes the Strata decode accounting portion of ADR 0036.

# Context

In Strata v0.2.7 Qwen, `request_timings()` writes its `generated` argument to
`predicted_n`. It computes `predicted_per_second` using the engine's last
generated count when that count is nonzero. The two counts can differ.
Strata also rounds `predicted_ms` and the rate independently before transmission.
[Server implementation](https://github.com/Niko1221/Strata/blob/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d/serve/server.py)
· [Benchmark protocol](https://github.com/Niko1221/Strata/blob/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d/docs/BENCHMARK_PROTOCOL.md#measurements)

# Decision

Map each decode timing field independently for JSON and terminal SSE responses.

| Metric | Wire field | Meaning |
| --- | --- | --- |
| `reported_predicted_tokens` | `timings.predicted_n` | Provider-reported generated count in tokens |
| `decode_seconds` | `timings.predicted_ms` | Reported decode duration converted to seconds |
| `decode_rate` | `timings.predicted_per_second` | Server-reported engine throughput in tokens/s |

A missing or null rate yields `missing / not_reported`, including when both
count and duration are present. An invalid rate retains its validation reason.
Only the server reports its decode throughput under this schema.

# Consequences

With a reported count of 120, a duration of one second and a reported rate of
100 tokens/s, Gear exposes all three values. Omitting the rate leaves the count
and duration available. The exact numerator used by the engine is unobserved.
Consumers can inspect the provider count's wire path alongside API output usage.
For prefill, Strata uses `prompt_n` as the numerator, so the prefill derivation in
ADR 0036 still applies.
