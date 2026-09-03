from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ContextStore(ABC):
    """Stores session events."""

    @abstractmethod
    def append(self, session_id: str, kind: str, payload: dict[str, Any]) -> None:
        """Appends one event.

        Args:
            session_id: Session identifier.
            kind: Event kind.
            payload: Event payload.
        """

    @abstractmethod
    def load(self, session_id: str) -> list[dict[str, Any]]:
        """Loads events for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Stored event objects.
        """
