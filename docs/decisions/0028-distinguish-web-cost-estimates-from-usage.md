# Title

Distinguish provider-reported cost estimates in web observations

# Status

accepted

# Context

The shared web result contract carries unit-labelled usage. Exa costDollars is
an endpoint-dependent estimate and must not be presented as billed USD or
combined with credits or model tokens.

# Decision

Represent Exa's reported total with a WebCostEstimate, a USD quantity explicitly
serialized with kind = "estimate". Preserve the full reported cost breakdown
under provider_metadata.exa.estimated_cost_dollars. If total is missing, emit
no usage quantity; do not invent zero or sum breakdown fields. Existing WebUsage
serialization is unchanged. Observations copy only the typed quantity, provider,
duration and outcome, never query/page data or the rest of provider metadata.

# Consequences

Exa costs become visible through the established tool observation and metrics
paths while remaining distinguishable from billed usage. Historical usage data
and existing Tavily payloads do not require migration.
