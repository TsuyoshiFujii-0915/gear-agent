# Title

Load Missing Reasoning Replay Configuration as Disabled

# Status

accepted

# Context

Gear Agent configurations created before encrypted reasoning replay was
introduced do not contain `model.reasoning_replay`. Requiring the new key would
prevent those otherwise valid configurations from starting. Replaying opaque
reasoning state must remain an explicit opt-in because providers and endpoints
do not all support it.

# Decision

Interpret an absent `model.reasoning_replay` key as `none`. Continue to require
that a present value is a string with the exact value `none` or `encrypted`.
Newly generated configurations continue to write `reasoning_replay = "none"`
explicitly.

# Consequences

Existing configurations retain their previous behavior and start without a
migration. Encrypted reasoning replay remains disabled until the user
explicitly selects it. Misspelled, mistyped, and unsupported explicit values
still fail during configuration loading.
