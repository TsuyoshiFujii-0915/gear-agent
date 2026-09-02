from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from gear_agent.store.base import ContextStore


class MemoryContextStore(ContextStore):
    """In-memory store for tests and short-lived runs."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def append(self, session_id: str, kind: str, payload: dict[str, Any]) -> None:
        self.events.append(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "kind": kind,
                "payload": payload,
            }
        )

    def load(self, session_id: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["session_id"] == session_id]
