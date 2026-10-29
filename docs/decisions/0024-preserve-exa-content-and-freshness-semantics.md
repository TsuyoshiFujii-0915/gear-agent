# Title

Preserve Exa content provenance and reject unsuccessful single-URL fetches

# Status

accepted

# Context

Exa Search and Contents return different content forms, optional metadata and
per-URL statuses. Successful HTTP status alone does not establish fetched text.
Official Search and Contents references were checked on 2026-10-29; Context7
could not serve documentation because its monthly quota was exhausted.

# Decision

Initially support Search `auto` (1–100 results), with explicit requests for
highlights by default or bounded text by configuration. Other search types are
rejected, including currently legacy `fast` and `instant`. Preserve highlights,
text and generated summary separately. Missing excerpts and optional metadata
are null; zero results are successful. Domain filters are explicit and queries
are not rewritten. Do not expose crawl-date filters, synthesis or subpages.

Contents requests one URL and explicit `text: true`. Require one successful
status for that input and one matching result by ID or URL; retain requested
URL, returned URL and ID. Reject missing, empty and over-limit text. Describe
the returned content as extracted text, without converting or summarizing it.
Preserve reported source only. `max_age_hours` omission delegates to Exa's
documented cache policy, -1 selects cache, 0 requests fresh retrieval, and 1–720
specifies cache age. Require explicit `livecrawl_timeout_ms` in 1–90000 and
strictly below the HTTP timeout converted from seconds.

Keep provider-reported cost under `provider_metadata.exa.estimated_cost_dollars`,
separate from credits and model tokens. Missing cost is null. Whitelist metadata
and never preserve upstream error bodies. Enforce content and HTTP body limits
without silent truncation; only explicitly requested Search maxCharacters can
limit text at the provider.

# Consequences

Live authentication, result quality and actual billing remain unverified until
an API key is available. Cache success is not a claim of current page freshness.
No new provider selection or retry occurs after an Exa error.

References: https://exa.ai/docs/reference/search and
https://exa.ai/docs/reference/get-contents.
