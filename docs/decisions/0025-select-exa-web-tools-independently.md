# Title

Select Exa search and fetch through the configured tool factory

# Status

superseded by 0027-integrate-exa-with-web-provider-foundation.md

# Context

Issue #34 requires independent Exa search and fetch without changing the model
adapter. The checkout does not yet contain the provider foundation from #31.
ADRs 0015, 0021 and 0022 preserve model, observation and history boundaries.

# Decision

Add only the selection boundary needed for Exa in the existing configured tool
factory. Keep the Tavily tools and their existing request/result contracts intact.
Use separate Exa configuration types, with common limits and credentials in
`[web_search]` / `[web_fetch]` and provider options in the corresponding `.exa`
table. Reject unknown options, providers and mixed Tavily/Exa options. Omission
of `provider` is an explicit legacy-config rule selecting Tavily; it is never
a runtime fallback. Explicit `provider = "tavily"` uses the existing flat table.
Disabled tools require no keys. Keep `web_search(query)` / `web_fetch(url)`.

Use bounded JSON HTTP requests for Exa with no retries or redirects, safe errors,
and credential removal before results reach any consumer. Do not change canonical
JSONL, compaction, model adapters or observation contracts. Exa config metadata
identifies the provider and omits credentials through the existing run projection.

# Consequences

Both mixed-provider directions work without a main-model change. Existing Tavily
tests remain authoritative. This implements the prerequisite selection needed
by #34, not the complete Tavily transport migration and hardening in #31. That
work must reconcile this boundary rather than introduce a second Exa integration.
