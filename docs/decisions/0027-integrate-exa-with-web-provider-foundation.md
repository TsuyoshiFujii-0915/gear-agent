# Title

Integrate Exa through the shared web provider foundation

# Status

accepted

# Context

PR #41 implements issue #31 and is now merged into main. The initial Exa
implementation predated this boundary and duplicated tools, HTTP and URL safety.
ADR 0025 is superseded; the Exa content semantics in ADR 0026 remain in force.

# Decision

Use the shared WebSearchTool/WebFetchTool, provider factory, ProviderResult,
WebHttpClient and DNS-aware URL validation. Exa adapters own only the provider
payload, normalization and configured limits. Use common config base types with
typed Exa options and concrete subclasses for Exa's additional required limits;
preserve the existing Exa TOML layout. Validate options before resolving secrets.
Keep bearer authentication for Tavily and add an explicit header-based entry
point to the same HTTP implementation for Exa's x-api-key authentication.

Allow an explicitly null search snippet when a provider reports no excerpt,
while rejecting missing fields and invalid non-null types. Preserve namespaced
provider metadata when the common serializer adds usage. Exa keeps credential
removal before any model, history, presentation or observation consumer.

# Consequences

Exa uses the same DNS, HTTP error, observation and dispatch paths as Tavily.
The standalone Exa tools and duplicate HTTP/safety modules are removed. Existing
Tavily contracts and Exa payload/content tests remain authoritative. Config
Python callers use options instead of exa; TOML configuration does not change.
Search/fetch wire semantics, no-retry behavior and history formats are unchanged.
