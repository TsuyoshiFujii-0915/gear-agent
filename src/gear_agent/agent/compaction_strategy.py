from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from gear_agent.agent.compaction import CompactionService
from gear_agent.context_budget import ContextBudgetManager, context_budget_error


@dataclass(frozen=True)
class CompactionCandidate:
    """Uncommitted checkpoint and content-free strategy observations."""

    kind: str
    payload: dict[str, Any]
    metrics: dict[str, Any]


class CompactionStrategy(Protocol):
    """Produces a checkpoint without mutating the audit store."""

    def compact(
        self, events: list[dict[str, Any]], timeout_seconds: int,
        stream_idle_timeout_seconds: int | None,
    ) -> CompactionCandidate:
        """Builds a complete candidate or raises an explicit error."""
        ...


class SummaryCompactionStrategy:
    """Uses the existing textual request with a physical-capacity preflight."""

    def __init__(self, service: CompactionService, budget: ContextBudgetManager) -> None:
        self._service = service
        self._budget = budget

    def compact(
        self, events: list[dict[str, Any]], timeout_seconds: int,
        stream_idle_timeout_seconds: int | None,
    ) -> CompactionCandidate:
        """Returns a summary checkpoint without persisting it.

        Args:
            events: Original effective history, including the current turn.
            timeout_seconds: Main model timeout.
            stream_idle_timeout_seconds: Main model streaming idle timeout.

        Returns:
            A textual checkpoint ready for the caller to commit.
        """
        request = self._service.prepare_request(events)
        diagnostic = self._budget.evaluate_compaction(request)
        if not diagnostic.fits:
            raise context_budget_error(diagnostic, 'compaction')
        summary = self._service.summarize_prepared(request, timeout_seconds, stream_idle_timeout_seconds)
        return CompactionCandidate('compaction_summary', {'text': summary, 'trigger': 'automatic'}, {'strategy': 'summary'})
