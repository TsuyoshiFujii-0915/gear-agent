from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from gear_agent.store.base import ContextStore


class JsonlContextStore(ContextStore):
    """JSONL-backed context store."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def append(self, session_id: str, kind: str, payload: dict[str, Any]) -> None:
        event = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "kind": kind,
            "payload": payload,
        }
        path = self._path_for_session(session_id)
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, ensure_ascii=False))
            output.write("\n")

    def load(self, session_id: str) -> list[dict[str, Any]]:
        path = self._path_for_session(session_id)
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                events.append(parsed)
        return events

    def _path_for_session(self, session_id: str) -> Path:
        safe_session_id = session_id.replace("/", "_")
        return self._root / f"{safe_session_id}.jsonl"
