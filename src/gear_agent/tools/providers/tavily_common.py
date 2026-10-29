from __future__ import annotations

from typing import Any

from gear_agent.errors import gear_error


def reject_tavily_errors(response: dict[str, Any], origin: str, prefix: str) -> None:
    """Rejects embedded and partial failures without copying upstream text.

    Args:
        response: Parsed Tavily response.
        origin: Model-callable tool name.
        prefix: Legacy Tavily error namespace.

    Raises:
        GearError: On explicit upstream error or any failed result.
    """
    if 'error' in response or 'detail' in response:
        raise gear_error(f'{prefix}_failed', 'Tavily reported an upstream error.', origin, True, {})
    failures = response.get('failed_results')
    if failures is None:
        return
    if not isinstance(failures, list):
        raise gear_error(f'{prefix}_response_invalid', 'Tavily failed_results must be a list.', origin, True, {})
    if failures:
        raise gear_error(f'{prefix}_failed', 'Tavily could not complete all requested results.',
                         origin, True, {'failed_count': len(failures)})
