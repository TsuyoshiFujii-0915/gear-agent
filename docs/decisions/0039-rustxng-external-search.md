# Title

Connect to an explicitly configured external RustXNG search server

# Status

accepted

# Context

Issue #50 adds keyless search against an already running RustXNG server. Its
v0.26.0 `/search` endpoint accepts form POST and reports engine failures even
with HTTP 200. ADRs 0023 and 0024 supply independent tool selection and bounded,
credential-safe HTTP. RustXNG needs explicit semantics for partial search results
and snippet limits rather than treating snippets as complete extracted pages.

# Decision

Implement only `mode = "external"`, with a required HTTP(S) base URL preserving
proxy path prefixes, required result/timeout limits and optional environment
Bearer authentication. Keep RustXNG configuration separate from key-required
providers. Reject unknown options, missing/unsupported modes, URL credentials,
query strings, fragments and invalid authentication before dispatch. Local/LAN
API endpoints are permitted; fetch target validation remains unchanged.

Use form POST with `q`, `format=json`, `max_results` and
`include_raw_content=0` through the shared streaming HTTP policy. Do not discover,
install, start, stop, update or reconfigure servers, or retry/redirect/fall back.

Normalize only query, URL, nullable title and snippet. For this search adapter,
explicitly marked snippet/result truncation and degraded results with engine
failures are permitted, refining ADR 0024's failure-only/no-truncation policy for
this new provider only. Empty results with engine failures raise a safe error
without asserting that every engine failed. Existing providers retain their
policies. Ignore raw content, scores, answers and internal metadata. Return only
bounded engine identities and categorized failures, never upstream error text.
Unreported usage stays empty. Hide the configured endpoint and token from repr,
run descriptions and diagnostics.

# Consequences

The model still calls `web_search(query)`. Search and fetch remain independent,
and disabled search requires no credentials or server dependencies. Server-side
search may send queries to upstream engines. Contract tests use a local HTTP
fixture; they do not certify live search quality or installation support. Managed
lifecycle and explicit setup are left to issues #51 and #52.
