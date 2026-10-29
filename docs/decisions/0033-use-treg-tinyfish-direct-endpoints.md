# Title
Use only TinyFish direct endpoints through treg for Web access

# Status
Accepted

# Context
Issue #33 requires Web search and fetch through treg without implicit paid
routing. On 2026-10-29 the public catalog listed `tinyfish.web.search` and
`tinyfish.web.fetch` as free (USD 0); the catalog's own price check is dated
2026-09-23. A catalog statement is not a permanent billing guarantee.
The existing provider-selection work is tracked separately in #31.

# Decision
Call the two fixed direct endpoints using HTTPX on the host. Before every call,
read the public endpoint catalog without credentials and reject missing/nonzero
pricing, another provider/method, or advertised overflow. Send `X-Treg-Token`
and an explicitly configured optional `X-Treg-Org`, never Bearer authentication.
Send `X-Treg-Route-Max-Cost: 0`: the current server applies this ceiling to direct
marketplace reservations and overflow too. This does not select a routed tool.
Do not implement routed endpoints, retries, redirects, CLI installation, login,
top-ups, credential discovery or key uploads.

Inspect response cost and route headers and reject nonzero charges or unexpected
routes. Preserve unreported cost/upstream as null, separately from the configured
endpoint. Bound decoded response bodies to 8 MiB. Normalize snippets separately
from page text; enforce a local search limit and fetch content limit. Fetch uses
Markdown and explicit cache TTL (seconds) and per-URL timeout (milliseconds).
HTTP timeout remains a separate seconds setting. Results remain tool data.

# Consequences
Each invocation adds a public catalog request and fails if that check is
unavailable or incompatible. A catalog check and a subsequent call are not an
atomic price quote; the zero reservation ceiling adds protection, but depends on
the deployed service honoring it. Response diagnostics cannot undo a charge.
BYOK vendor billing is outside treg's reported balance charge. Live authenticated
validation requires a user-provided token; no local CLI credentials are reused.

Sources checked: [catalog](https://treg.to/catalog/endpoints/tinyfish.web.search),
[fetch catalog](https://treg.to/catalog/endpoints/tinyfish.web.fetch),
[server ceiling](https://github.com/superdesigndev/treg/blob/28ddd6d67a5b56b33ac3c932742cc035ebf4a905/src/treg/application/call/service.py),
[reservation](https://github.com/superdesigndev/treg/blob/28ddd6d67a5b56b33ac3c932742cc035ebf4a905/src/treg/application/call/reserve.py).
