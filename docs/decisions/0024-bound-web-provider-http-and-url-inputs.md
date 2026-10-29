# Title

Bound and sanitize host-side web provider requests

# Status

accepted

# Context

The original Tavily urllib transport reads unlimited bodies, follows redirects
with authentication and includes upstream error text in recoverable GearErrors.
Those errors reach canonical history and the main model. Provider separation
must not reproduce these risks. ADR 0021 requires content-free observations.

# Decision

Use the existing HTTPX dependency for both web transports. Disable redirects per
request, request identity encoding and reject other content encodings before
reading. Stream at most 8 MiB per response, enforcing declared and actual size.
Do not retry or fall back. Convert HTTP, timeout, network, encoding and JSON
failures to fixed safe classifications without upstream bodies, headers or
exception text. Treat embedded errors and partial failures as explicit failures.
Never truncate or summarize content to satisfy configured limits.

Validate fetch inputs before provider dispatch: only http/https, no userinfo,
control characters, backslashes, invalid ports, local hostnames, private or
special-use address literals. Resolve DNS and require all addresses to be public;
resolution failure is an explicit error. The provider performs extraction, so
Gear cannot pin the provider's DNS resolution or inspect its redirect chain.
Do not claim this client validation prevents provider-side DNS rebinding or
redirects into private networks; that enforcement also belongs to the upstream
service. No credentials, custom headers or cookies are sent to target pages by
Gear. Web API authentication is sent only to the fixed provider API endpoint.

# Consequences

Host-side web HTTP is independent of runtime.network, which controls Docker
shell networking. Explicitly enabling a web tool sends queries or URLs to its
selected provider and may incur charges. Compressed, oversized, malformed,
partially failed and redirected API responses now fail safely. DNS failures and
private targets are rejected before submission. Live authenticated provider
validation remains separate from deterministic contract tests.
