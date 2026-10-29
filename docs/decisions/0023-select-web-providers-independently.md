# Title

Select web search and fetch through independent provider boundaries

# Status

accepted

# Context

Issue #31 requires independent search and extraction selection without changing
model adapters, tool names, Jev eligibility, or canonical session semantics.
ADRs 0015, 0021 and 0022 remain in force. Tavily is the only implemented provider;
other vendors belong to separate issues.

# Decision

Use separate search and fetch protocols and factories. Tools own model argument
validation and shared result limits; adapters own vendor requests, normalization,
and reported usage. Preserve the existing transport seams for API test doubles.
Provider results carry the existing JSON payload plus typed, unit-labelled usage.
Tool results add `provider` and `provider_metadata.<provider>.usage`; existing
Tavily result fields retain their meaning. Search `content` is a snippet,
`answer` is generated, and `raw_content`/fetch `content` are extracted text.
Unreported titles, scores and usage remain null or absent. `requested_url` is the
fetch input, `url` is the provider-reported URL and `final_url` is null unless an
adapter can establish a redirect destination. Missing response query uses the
original input by explicit contract. External content remains tool data.

Place provider, api_key_env, timeout_seconds and result/content limits in each
web tool's table. Place vendor options in `[web_search.tavily]` or
`[web_fetch.tavily]`. The loader converts legacy flat Tavily options to the same
typed options. Omitted provider explicitly means Tavily for compatibility.
Any mixture of flat vendor options and a nested vendor table is an error, even
when values agree. Validate only enabled tools, including unknown keys, types,
provider support and vendor ranges, before resolving their credentials. Disabled
tools require no table or key and make no requests. No registry, dynamic plugins,
unimplemented vendor settings or speculative capabilities are introduced.

# Consequences

Existing TOML files and gear init defaults remain usable. Internal Python
constructors now accept typed vendor options and providers; fixtures migrate
with their request/response expectations intact. New providers require their own
options, loader branch, adapter and factory branch, without model changes.
Per-call observations retain only provider, unit-labelled usage, duration and
success/error identity; they do not aggregate credits, currency and tokens or
copy query/page content. Run metadata describes the enabled providers and options
without credentials. API keys remain environment-only and excluded from repr.
