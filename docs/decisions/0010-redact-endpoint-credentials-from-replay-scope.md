# Title

Redact Endpoint Credentials from Reasoning Replay Scope

# Status

accepted

# Context

ADR-0008 scoped encrypted reasoning replay to the exact configured endpoint
URL and rejected every URL containing a query. Valid Responses-compatible
endpoints can require non-credential query parameters, such as Azure API
version selection. A query can also contain the configured API key, which must
not contribute to persisted metadata even through a derived fingerprint.

# Decision

Supersede ADR-0008's endpoint identity rule. Continue to scope replay by the
Responses protocol, configured model ID, and a SHA-256 endpoint identity. Allow
query strings in encrypted replay endpoint URLs. Before hashing, replace any
URL component whose decoded value contains the non-empty configured API key
with an unambiguous credential marker. Keep all other URL components, including
non-credential query components and their ordering, in the identity material.
When the API key is not present in the URL, retain the exact configured URL as
the identity material. The request transport continues to use the original URL
without modification.

Continue to reject user information and fragments for encrypted replay.
User information is itself credential-bearing, while fragments are not part of
the HTTP request endpoint and make scope identity ambiguous.

# Consequences

Responses-compatible endpoints with semantic query parameters can use
encrypted reasoning replay. Changing a non-credential query component changes
the replay scope, while rotating the configured API key in a credential-bearing
component does not. Persisted scope metadata is not derived from the configured
API key, and the actual request preserves the endpoint query exactly.
