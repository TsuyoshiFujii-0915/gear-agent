# Model metadata fixtures

These synthetic JSON projections cover the read-only routes used by both profiles.
Keys name paths relative to the reverse proxy prefix. Model identifiers, context
sizes, template text and file paths are synthetic. The tables omit unrelated fields.

- Strata v0.2.7, commit `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`, reports an engine ID and can list aliases as additional IDs. The tool hint comes from the chat template capabilities. [HTTP handlers](https://github.com/Niko1221/Strata/blob/v0.2.7/serve/server.py) · [Template capabilities](https://github.com/Niko1221/Strata/blob/v0.2.7/serve/frontend.py)
- Project Maya v1.1.56, commit `22fc101f2c109b7e7174c975ff3c2d8e9ce0a867`, can list an unloaded model and report a reload in health. [HTTP handlers](https://github.com/mw00/project-maya/blob/22fc101f2c109b7e7174c975ff3c2d8e9ce0a867/serve/server.py)

These files can be reused for the Project Maya compatibility tests in #54. Live model quality and throughput require a separate run.
