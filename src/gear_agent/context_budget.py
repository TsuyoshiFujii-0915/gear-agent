from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol
import json

from gear_agent.errors import GearError, gear_error


@dataclass(frozen=True)
class ContextBudgetConfig:
    """Explicit capacity policy, independent of provider and tokenizer.

    Attributes:
        auto_compaction: Whether to enforce the budget and compact automatically.
        context_window_tokens: Configured model capacity, or unknown when disabled.
        reserved_tokens: Capacity reserved for output and reasoning.
        max_input_tokens: Earlier trigger, or window minus reserve when omitted.
    """

    auto_compaction: bool
    context_window_tokens: int | None
    reserved_tokens: int
    max_input_tokens: int | None

    def __post_init__(self) -> None:
        if type(self.auto_compaction) is not bool:
            raise _config_error('auto_compaction', 'Expected a boolean.')
        for key, value in (
            ('context_window_tokens', self.context_window_tokens),
            ('max_input_tokens', self.max_input_tokens),
        ):
            if value is not None and (type(value) is not int or value < 1):
                raise _config_error(key, 'Expected a positive integer.')
        if type(self.reserved_tokens) is not int or self.reserved_tokens < 0:
            raise _config_error('reserved_tokens', 'Expected a nonnegative integer.')
        if self.context_window_tokens is None:
            if self.auto_compaction:
                raise _config_error('context_window_tokens', 'Automatic compaction requires an explicit model capacity.')
            if self.max_input_tokens is not None or self.reserved_tokens != 0:
                raise _config_error('context_window_tokens', 'Capacity is required when specifying input limits or reserve.')
            return
        capacity = self.context_window_tokens - self.reserved_tokens
        if capacity <= 0:
            raise _config_error('reserved_tokens', 'Reserve must be smaller than the context window.')
        if self.max_input_tokens is not None and self.max_input_tokens > capacity:
            raise _config_error('max_input_tokens', 'Input limit must not exceed context window minus reserve.')


DISABLED_CONTEXT_BUDGET = ContextBudgetConfig(False, None, 0, None)


@dataclass(frozen=True)
class ContextRequest:
    """Effective adapter arguments plus component boundaries for accounting.

    Attributes:
        input_value: The exact input object dispatched to the adapter.
        tools: The exact tool schemas dispatched to the adapter.
        instructions: Complete dispatched instructions, including repository rules.
        base_instructions: Prefix of instructions belonging to Gear itself.
        history_item_count: Number of leading input items preceding the current turn.
            String inputs, used for textual compaction, belong entirely to history.
    """

    input_value: object
    tools: list[dict[str, object]]
    instructions: str
    base_instructions: str
    history_item_count: int


class TokenEstimator(Protocol):
    """Estimates a serializable request component without network calls."""

    def estimate(self, value: object) -> int:
        """Returns a conservative token estimate for the supplied component."""
        ...


class ByteTokenEstimator:
    """Charges compact JSON UTF-8 bytes plus 25%, rounded up per component."""

    def estimate(self, value: object) -> int:
        """Estimates a component including its JSON syntax and escaping.

        Args:
            value: Effective request component.

        Returns:
            Estimated tokens, not an exact tokenizer count.

        Raises:
            GearError: If the component cannot be serialized as UTF-8 JSON.
        """
        try:
            encoded = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
        except (TypeError, ValueError, UnicodeError) as exc:
            raise gear_error(
                'context_estimate_invalid', 'Cannot serialize effective request for context accounting.',
                'context_budget', True, {'reason': str(exc)},
            ) from exc
        return (len(encoded) * 5 + 3) // 4


@dataclass(frozen=True)
class ContextBudgetDiagnostic:
    """Serializable component estimates and effective capacity.

    All component fields are estimated tokens. Total excludes reserved_headroom,
    which has already been subtracted from input_limit_tokens. A None limit means
    capacity is unknown and no budget is enforced.
    """

    instructions: int
    repository_context: int
    history: int
    tool_schemas: int
    current_turn: int
    request_overhead: int
    reserved_headroom: int
    total_estimated_request_tokens: int
    input_limit_tokens: int | None
    context_window_tokens: int | None

    @property
    def fits(self) -> bool:
        """Returns whether the estimate is within the inclusive input limit."""
        return self.input_limit_tokens is None or self.total_estimated_request_tokens <= self.input_limit_tokens


def context_budget_error(diagnostic: ContextBudgetDiagnostic, phase: str) -> GearError:
    """Builds an explicit failure with content-free accounting details.

    Args:
        diagnostic: Estimate that exceeded the effective capacity.
        phase: Request construction phase that failed.

    Returns:
        Recoverable error containing estimates, limit and origin.
    """
    return gear_error(
        'context_budget_exceeded',
        f'Context estimate {diagnostic.total_estimated_request_tokens} exceeds input limit '
        f'{diagnostic.input_limit_tokens} during {phase}.',
        'context_budget', True, {'phase': phase, 'diagnostic': asdict(diagnostic)},
    )


class ContextBudgetManager:
    """Applies explicit capacity policy to the effective request representation."""

    def __init__(self, config: ContextBudgetConfig, estimator: TokenEstimator) -> None:
        self._config = config
        self._estimator = estimator

    def evaluate(self, request: ContextRequest) -> ContextBudgetDiagnostic:
        """Measures an agent request against the configured compaction trigger.

        Args:
            request: Effective adapter arguments and accounting boundaries.

        Returns:
            Component estimates and the regular input limit.
        """
        limit = self._config.max_input_tokens
        if limit is None:
            limit = self._capacity()
        return self._measure(request, limit)

    def evaluate_compaction(self, request: ContextRequest) -> ContextBudgetDiagnostic:
        """Measures a summary request against physical input capacity.

        Args:
            request: Effective textual summarization request.

        Returns:
            Component estimates and context window minus reserved tokens.
        """
        return self._measure(request, self._capacity())

    def _capacity(self) -> int | None:
        window = self._config.context_window_tokens
        if window is None:
            return None
        return window - self._config.reserved_tokens

    def _measure(self, request: ContextRequest, limit: int | None) -> ContextBudgetDiagnostic:
        if not request.instructions.startswith(request.base_instructions):
            raise gear_error(
                'context_request_invalid', 'Base instructions must be a prefix of effective instructions.',
                'context_budget', False, {},
            )
        estimate = self._estimator.estimate
        instructions = estimate(request.base_instructions)
        repository_suffix = request.instructions[len(request.base_instructions):]
        repository = estimate(repository_suffix) if repository_suffix else 0
        if isinstance(request.input_value, list):
            count = request.history_item_count
            if not 0 <= count <= len(request.input_value):
                raise gear_error(
                    'context_request_invalid', 'History boundary is outside the effective input.',
                    'context_budget', False, {'history_item_count': count},
                )
            history_items = request.input_value[:count]
            current_items = request.input_value[count:]
            history = estimate(history_items) if history_items else 0
            current = estimate(current_items) if current_items else 0
        else:
            history = estimate(request.input_value)
            current = 0
        schemas = estimate(request.tools) if request.tools else 0
        overhead = 256
        total = instructions + repository + history + current + schemas + overhead
        return ContextBudgetDiagnostic(
            instructions, repository, history, schemas, current, overhead,
            self._config.reserved_tokens, total, limit, self._config.context_window_tokens,
        )


def _config_error(key: str, message: str) -> GearError:
    return gear_error(
        'config_value_invalid', f'Invalid context_budget.{key}: {message}',
        'config', True, {'table': 'context_budget', 'key': key},
    )
