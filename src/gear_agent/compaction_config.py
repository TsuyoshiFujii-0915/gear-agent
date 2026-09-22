from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
import math
import re

from gear_agent.errors import GearError


@dataclass(frozen=True)
class JevPolicy:
    """Conservative selection thresholds and independent request limits."""

    keep_threshold: float
    preserve_recent_turns: int
    truncate_chars: int
    max_state_tokens: int
    max_request_tokens: int
    batch_size: int

    def __post_init__(self) -> None:
        if type(self.keep_threshold) not in (int, float) or not math.isfinite(self.keep_threshold) or not 0 < self.keep_threshold <= 1:
            raise _invalid('keep_threshold', 'Expected a finite number in (0, 1].')
        for key in ('preserve_recent_turns', 'truncate_chars', 'max_state_tokens', 'max_request_tokens', 'batch_size'):
            value = getattr(self, key)
            minimum = 0 if key == 'preserve_recent_turns' else 1
            if type(value) is not int or value < minimum:
                raise _invalid(key, f'Expected an integer >= {minimum}.')
        if self.max_state_tokens > 25_000 or self.max_request_tokens > 30_000:
            raise _invalid('limits', 'State/request limits must not exceed 25000/30000 estimated tokens.')


DEFAULT_JEV_POLICY = JevPolicy(0.2, 1, 160, 25_000, 30_000, 32)


@dataclass(frozen=True)
class JevConfig:
    """Pinned provider configuration; the credential is never printable."""

    model: str
    api_key: str = field(repr=False)
    timeout_seconds: int
    policy: JevPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or re.fullmatch(r'jev-\d+\.\d+\.\d+', self.model) is None:
            raise _invalid('model', 'Expected a pinned Jev version, such as jev-1.13.0.')
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise _invalid('api_key_env', 'Jev requires a nonempty API credential.')
        if type(self.timeout_seconds) is not int or self.timeout_seconds < 1:
            raise _invalid('timeout_seconds', 'Expected a positive integer.')


@dataclass(frozen=True)
class CompactionConfig:
    """Explicit strategy and fallback; absent legacy config uses summary."""

    strategy: str
    fallback: str
    jev: JevConfig | None

    def __post_init__(self) -> None:
        if self.strategy not in ('summary', 'jev') or self.fallback not in ('none', 'summary'):
            raise _invalid('strategy/fallback', 'Expected summary/jev and none/summary.')
        if (self.strategy == 'jev') != (self.jev is not None):
            raise _invalid('jev', 'Jev configuration is required only for strategy=jev.')
        if self.strategy == 'summary' and self.fallback != 'none':
            raise _invalid('fallback', 'Summary has no fallback strategy.')


SUMMARY_COMPACTION = CompactionConfig('summary', 'none', None)


def load_compaction_config(raw: dict[str, object], environment: Mapping[str, str]) -> CompactionConfig:
    """Loads opt-in selection with documented conservative policy defaults.

    Args:
        raw: Parsed application TOML.
        environment: Credential environment, never included in errors.

    Returns:
        Validated strategy, with summary compatibility when the table is absent.
    """
    if 'compaction' not in raw:
        return SUMMARY_COMPACTION
    table = raw['compaction']
    if not isinstance(table, dict) or set(table) - {'strategy', 'fallback', 'jev'}:
        raise _invalid('compaction', 'Invalid table or unknown key.')
    strategy = table.get('strategy')
    fallback = table.get('fallback', 'none')
    if strategy != 'jev':
        if 'jev' in table:
            raise _invalid('jev', 'Jev options require strategy=jev.')
        return CompactionConfig(strategy, fallback, None)
    options = table.get('jev')
    policy_keys = set(JevPolicy.__dataclass_fields__)
    if not isinstance(options, dict) or set(options) - policy_keys - {'model', 'api_key_env', 'timeout_seconds'}:
        raise _invalid('jev', 'Missing Jev table or unknown option.')
    env_name = options.get('api_key_env')
    if not isinstance(env_name, str) or not env_name:
        raise _invalid('api_key_env', 'Expected an environment variable name.')
    api_key = environment.get(env_name)
    if not api_key:
        raise GearError('config_secret_missing', 'Configured Jev API credential is missing.', 'config', True, {})
    policy = JevPolicy(**{key: options.get(key, getattr(DEFAULT_JEV_POLICY, key)) for key in policy_keys})
    jev = JevConfig(options.get('model'), api_key, options.get('timeout_seconds', 10), policy)
    return CompactionConfig(strategy, fallback, jev)


def _invalid(key: str, reason: str) -> GearError:
    return GearError('config_value_invalid', f'Invalid compaction.{key}: {reason}', 'config', True, {'key': key})
