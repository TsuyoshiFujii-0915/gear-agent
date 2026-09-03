from __future__ import annotations

from typing import Any
import json

from gear_agent.config import ModelConfig
from gear_agent.model.client import ModelClient
from gear_agent.model.responses import extract_output_text
from gear_agent.store.base import ContextStore


COMPACTION_INSTRUCTIONS = "Summarize stored Gear Agent session events for future continuation."


class CompactionService:
    """Creates explicit summaries for stored session history."""

    def __init__(self, client: ModelClient) -> None:
        self._client = client

    def compact(
        self,
        session_id: str,
        store: ContextStore,
        config: ModelConfig,
        timeout_seconds: int,
    ) -> str:
        """Compacts existing session events into a summary.

        Args:
            session_id: Session identifier.
            store: Context store.
            config: Model endpoint configuration.
            timeout_seconds: Request timeout in seconds.

        Returns:
            Summary text.
        """

        events = store.load(session_id)
        prompt = _build_compaction_prompt(events)
        response = self._client.create_response(
            config,
            prompt,
            [],
            COMPACTION_INSTRUCTIONS,
            timeout_seconds,
        )
        summary = extract_output_text(response)
        store.append(session_id, "compaction_summary", {"text": summary})
        return summary


def _build_compaction_prompt(events: list[dict[str, Any]]) -> str:
    serialized_events = json.dumps(events, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "Summarize this coding-agent session for future continuation.",
            "Include user goal, completed work, changed files, remaining work, constraints, and recent errors.",
            "Do not omit important tool results.",
            "",
            serialized_events,
        ]
    )
