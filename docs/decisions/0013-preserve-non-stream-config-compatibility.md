# Title

Preserve non-stream behavior for existing configuration

# Status

accepted

# Context

Existing Gear Agent configuration files predate model streaming. Silently
enabling streaming would change provider compatibility and error behavior.
Streaming also requires an explicit idle timeout that has no meaning for the
legacy non-stream path.

# Decision

Add `model.stream` and `runtime.model_stream_idle_timeout_seconds`. A missing
`model.stream` is deliberately interpreted as `false`, preserving the existing
request mode. When streaming is enabled, the idle timeout is required and must
be a positive integer. When streaming is disabled, the idle timeout may be
omitted; if present it is still validated so configuration mistakes fail at
startup. Newly generated configuration writes both values explicitly.

# Consequences

Existing configuration remains valid and keeps non-stream behavior. Opting in
to streaming requires an explicit idle-timeout policy. The compatibility
default is limited to the new stream selector and is not a general fallback for
invalid configuration values.
