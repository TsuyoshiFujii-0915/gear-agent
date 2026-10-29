# Title

Preserve DeepSeek search summaries, page bodies, and relevance scores separately

# Status

accepted

# Context

Issue #32 adds standalone DeepSeek Search and Fetch to the web tools. DeepSeek
returns a snippet, a generated summary, optional page content, and a relevance
score on a 0–3 scale. Existing Tavily results expose `content`, `raw_content`, and
`score`. Treating the summary as fetched content or reusing the common score
would change the meaning of persisted tool results and downstream compaction.

# Decision

Map the DeepSeek snippet to the common short `content` field and preserve it
explicitly as `snippet`. Preserve `summary` separately. Only page content maps
to `raw_content`; absent or empty page content maps to null. Leave the common
`score` null and retain `rerank_score` and optional source metadata under
`metadata.deepseek`. Retain the original search query when the API omits it.

Fetch results carry the requested URL, an explicitly declared Markdown format,
and the returned title and content. An empty title is valid; empty content and
content exceeding the configured limit are errors. Do not infer a final URL.
Neither API reports usage: do not manufacture credits, token usage, or cost.

# Consequences

The existing tool names and JSON-serializable result shape remain usable by
history, Jev, summary, and resume. Consumers can distinguish retrieved page
content from search summaries and cannot mistake DeepSeek relevance for a
Tavily score or a Jev probability. Pricing documentation is an estimate source,
not evidence of actual account charges.
