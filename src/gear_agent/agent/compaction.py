from __future__ import annotations

from typing import Any, Literal
import json

from gear_agent.agent.history import select_effective_events
from gear_agent.errors import gear_error
from gear_agent.context_budget import ContextRequest
from gear_agent.model.adapter import ModelAdapter
from gear_agent.model.events import SilentModelProgressEventSink
from gear_agent.model.replay import strip_opaque_reasoning_from_event
from gear_agent.store.base import ContextStore
from gear_agent.observation import RunObserver, record_model_usage, request_model


COMPACTION_INSTRUCTIONS = "Summarize effective Gear Agent session context for future continuation."


class CompactionService:
    """Creates explicit summaries for stored session history."""

    def __init__(self, adapter: ModelAdapter, observer: RunObserver | None = None) -> None:
        """Binds services; existing interactive callers omit run-only observation."""
        self._adapter = adapter
        self._observer = observer

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

        request = self.prepare_request(store.load(session_id))
        return self.compact_prepared(
            session_id, store, request, timeout_seconds,
            stream_idle_timeout_seconds, 'manual',
        )

    def prepare_request(self, events: list[dict[str, Any]]) -> ContextRequest:
        """Builds a textual summary request that can be measured before dispatch.

        Args:
            events: Stored session events, including checkpoint boundaries.

        Returns:
            Effective request after checkpoint selection and opaque sanitization.
        """
        effective_events = select_effective_events(events)
        sanitized_events = _strip_model_response_opaque_reasoning(effective_events)
        prompt = _build_compaction_prompt(sanitized_events)
        return ContextRequest(prompt, [], COMPACTION_INSTRUCTIONS, COMPACTION_INSTRUCTIONS, 0)

    def compact_prepared(
        self,
        session_id: str,
        store: ContextStore,
        request: ContextRequest,
        timeout_seconds: int,
        stream_idle_timeout_seconds: int | None,
        trigger: Literal['manual', 'automatic'],
    ) -> str:
        """Executes an already prepared request and persists one checkpoint.

        Args:
            session_id: Session identifier.
            store: Append-only context store.
            request: Effective request, preflighted by automatic callers.
            timeout_seconds: Request timeout.
            stream_idle_timeout_seconds: Maximum idle time between stream bytes.
            trigger: Checkpoint origin, distinguishing automatic policy decisions.

        Returns:
            Nonempty summary text.

        Raises:
            GearError: If the model fails or returns no summary.
        """
        response = request_model(
            self._adapter, request, timeout_seconds, stream_idle_timeout_seconds,
            SilentModelProgressEventSink(), self._observer, 'compaction',
        )
        record_model_usage(response, self._observer)
        summary = response.text
        if summary.strip() == "":
            raise gear_error(
                "compaction_summary_missing",
                "Compaction response did not contain a summary.",
                "compaction",
                True,
                {},
            )
        payload = {'text': summary}
        if trigger == 'automatic':
            payload['trigger'] = trigger
        store.append(session_id, "compaction_summary", payload)
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
