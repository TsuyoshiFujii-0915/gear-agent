# Title

Use an explicit Cloud v2 request policy for Firecrawl web tools

# Status

accepted

# Context

Issue #35 connects Firecrawl to the independent boundaries in ADR 0023 and the
HTTP/URL protections in ADR 0024. Provider defaults can enable extra processing,
reuse cached pages, skip target TLS verification or retry through another proxy.
Raw REST v2 envelopes differ from v1 and SDK return values.

# Decision

Use fixed Cloud v2 Search and Scrape endpoints through the shared HTTP client.
Search requests only web results, with no scrapeOptions or tool discovery.
The nested search option `highlights` defaults to false; enabling it labels
returned descriptions as highlights instead of snippets. Neither is page text.
Scrape requests only Markdown, explicitly verifies target TLS, disables PDF/image
parsers and LLM cleaning, and defaults to basic proxy and main content only.
Cache age defaults explicitly to 172800000 ms; cache writing defaults to false.
Proxy, cache age, cache writing, main-content filtering and wait duration are
explicit nested fetch settings. HTTP timeout is 2..301 seconds. API timeout is
`(timeout_seconds - 1) * 1000` ms, reserving one second for response overhead;
waitFor must be less than that API budget and at most 60000 ms.

Require success=true and the v2 data envelope, reject reported errors and
non-success target statuses, and accept only Markdown text from supported page
content types. Missing optional metadata remains null. Preserve sourceURL and
final metadata.url separately; when neither exists, the common url is explicitly
the requested URL and final_url remains null. A title array is joined with a
newline. Keep selected vendor details under `firecrawl`, and usage under the
existing `provider_metadata.firecrawl.usage`. Search credits come only from
top-level creditsUsed; Scrape credits come only from data.metadata.creditsUsed.
Warnings become a fixed diagnostic marker, without copying upstream warning
text into history. Unknown metadata is not copied.

# Consequences

Existing Tavily behavior and default configuration remain intact. Firecrawl
search and fetch are independently selectable, without a new SDK or model
change. There are no Gear retries or provider fallbacks. Explicit proxy=auto
permits Firecrawl's own retry behavior. The current OpenAPI says enhanced proxy
has no surcharge, unlike the issue's earlier research; pricing is not estimated
by Gear. PDF/image parsing, browser actions, cookies, batch/crawl, generated
summaries and extraction are not exposed. Cache writing=false is not a guarantee
of end-to-end zero data retention. Authenticated smoke tests require a user key.

Sources checked on 2026-10-29:

- https://docs.firecrawl.dev/api-reference/v2-openapi.json
- https://docs.firecrawl.dev/api-reference/endpoint/search
- https://docs.firecrawl.dev/api-reference/endpoint/scrape
- https://github.com/firecrawl/firecrawl/blob/main/apps/api/src/controllers/v2/scrape.ts
