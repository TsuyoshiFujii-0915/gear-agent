from __future__ import annotations

from typing import Any
import json

from gear_agent.agent.history import select_effective_events
from gear_agent.errors import gear_error
from gear_agent.model.adapter import ModelAdapter
from gear_agent.model.events import SilentModelProgressEventSink
from gear_agent.model.replay import strip_opaque_reasoning_from_event
from gear_agent.store.base import ContextStore


COMPACTION_INSTRUCTIONS = "Summarize effective Gear Agent session context for future continuation."


class CompactionService:
    """Creates explicit summaries for stored session history."""

    def __init__(self, adapter: ModelAdapter) -> None:
        self._adapter = adapter

    def compact(
        self,
        session_id: str,
        store: ContextStore,
        timeout_seconds: int,
        stream_idle_timeout_seconds: int | None = None,
    ) -> str:
        """Compacts existing session events into a summary.

        Args:
            session_id: Session identifier.
            store: Context store.
            timeout_seconds: Request timeout in seconds.
            stream_idle_timeout_seconds: Maximum idle time between stream bytes.

        Returns:
            Summary text.
        """

        events = store.load(session_id)
        effective_events = select_effective_events(events)
        sanitized_events = _strip_model_response_opaque_reasoning(effective_events)
        prompt = _build_compaction_prompt(sanitized_events)
        response = self._adapter.create_response(
            prompt,
            [],
            COMPACTION_INSTRUCTIONS,
            timeout_seconds,
            stream_idle_timeout_seconds,
            SilentModelProgressEventSink(),
        )
        summary = response.text
        if summary.strip() == "":
            raise gear_error(
                "compaction_summary_missing",
                "Compaction response did not contain a summary.",
                "compaction",
                True,
                {},
            )
        store.append(session_id, "compaction_summary", {"text": summary})
        return summary


def _build_compaction_prompt(effective_events: list[dict[str, Any]]) -> str:
    serialized_history = json.dumps(effective_events, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "Summarize this coding-agent context for future continuation.",
            "Include user goal, completed work, changed files, remaining work, constraints, and recent errors.",
            "Do not omit important tool results.",
            "",
            serialized_history,
        ]
    )


def _strip_model_response_opaque_reasoning(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sanitized_events: list[dict[str, Any]] = []
    for event in events:
        if event.get("kind") != "model_response":
            sanitized_events.append(event)
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            sanitized_events.append(event)
            continue
        sanitized_event = dict(event)
        sanitized_event["payload"] = strip_opaque_reasoning_from_event(payload)
        sanitized_events.append(sanitized_event)
    return sanitized_events
