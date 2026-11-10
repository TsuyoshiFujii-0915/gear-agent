"""Persist the same terminal event that the runtime delivers to the UI."""

import logging

from gear_agent.agent.events import AgentLoopEventSink
from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.request_events import RequestFinished, RequestLogWriteFailed, RequestStarted
from gear_agent.store.jsonl import JsonlContextStore


class SessionRequestLog:
    def __init__(self, store: JsonlContextStore, events: AgentLoopEventSink, privacy: ArtifactPrivacy) -> None:
        self._store = store
        self._events = events
        self._privacy = privacy

    def publish(self, event: RequestStarted | RequestFinished) -> None:
        if isinstance(event, RequestFinished):
            session_id = event.request.scope.session_id
            payload = event.payload()
            payload['session_id'] = self._privacy.text(session_id)
            try:
                self._store.append_request(session_id, payload)
            except (OSError, UnicodeError, ValueError) as error:
                logging.getLogger(__name__).warning(
                    'request_log_write_failed request_id=%s cause=%s',
                    event.request.request_id, type(error).__name__,
                )
                self._events.publish(RequestLogWriteFailed(event.request.request_id, session_id))
        self._events.publish(event)
