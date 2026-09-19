from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from gear_agent import observation
from gear_agent.agent.events import AgentLoopEvent, ContextBudgetEvaluated, ReasoningReplayEvaluated
from gear_agent.artifact_privacy import ArtifactPrivacy


class EvalRunCollector:
    """Collects structured run observations independently of canonical persistence."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._started: float | None = None

    def start(self) -> None:
        """Starts wall-duration measurement at the artifact initialization boundary."""
        self._started = observation.perf_counter()

    def record(self, kind: str, payload: dict[str, Any]) -> None:
        """Records a versioned, complete observation with a UTC correlation time."""
        self.events.append({'schema_version': 1, 'kind': kind, 'payload': payload,
                            'created_at': datetime.now(timezone.utc).isoformat()})

    def publish(self, event: AgentLoopEvent) -> None:
        """Accepts content-free diagnostics; ignores all streaming/presentation events."""
        if isinstance(event, ContextBudgetEvaluated):
            self.record('context_budget', asdict(event))
        elif isinstance(event, ReasoningReplayEvaluated):
            self.record('reasoning_replay', asdict(event))

    def metrics(
        self, status: str, error: dict[str, str] | None, final_text_produced: bool,
        session_events: list[dict[str, Any]], privacy: ArtifactPrivacy,
    ) -> dict[str, Any]:
        """Aggregates measured work without inferring unavailable provider usage.

        Args:
            status: Terminal execution outcome.
            error: Projected execution failure.
            final_text_produced: Whether a successful canonical answer exists.
            session_events: Independent canonical session snapshot.
            privacy: Central serialization policy.

        Returns:
            Versioned machine-readable run metrics.
        """
        if self._started is None:
            raise RuntimeError('Run collector was not started.')
        requests: list[dict[str, Any]] = []
        for event in self.events:
            if event['kind'] == 'model_request_finished':
                requests.append(dict(event['payload']))
            elif event['kind'] == 'model_usage':
                if not requests or requests[-1]['usage'] is not None:
                    raise RuntimeError('Model usage observation has no unmatched completed request.')
                requests[-1]['usage'] = event['payload']['usage']
        tool_calls = self._payloads('tool_finished')
        by_name: dict[str, dict[str, Any]] = {}
        for call in tool_calls:
            entry = by_name.setdefault(call['name'], {'call_count': 0, 'successful': 0,
                                                       'failed': 0, 'duration_seconds': 0.0})
            entry['call_count'] += 1
            entry['successful' if call['successful'] else 'failed'] += 1
            entry['duration_seconds'] += call['duration_seconds']
        model: dict[str, Any] = {'request_count': len(self._payloads('model_request_started')),
                                'requests': requests}
        for category in ('input_tokens', 'output_tokens', 'total_tokens'):
            counts = [r['usage'][category] if r['usage'] is not None else None for r in requests]
            model[category] = sum(counts) if counts and all(c is not None for c in counts) else None
        replay_fields = ('reused_encrypted_items', 'dropped_disabled_items',
                         'dropped_incompatible_scope_items', 'dropped_missing_scope_items')
        replay = {field: sum(p[field] for p in self._payloads('reasoning_replay')) for field in replay_fields}
        replay['dropped_encrypted_items'] = sum(replay[field] for field in replay_fields[1:])
        context = self._payloads('context_budget')
        estimates = [p['diagnostic']['total_estimated_request_tokens'] for p in context]
        checkpoints = [e['payload'] for e in session_events if e['kind'] == 'compaction_summary']
        iterations = self._payloads('iteration_completed')
        return privacy.serialize({
            'schema_version': 1,
            'outcome': {'status': status, 'error': error, 'completed_iterations': len(iterations),
                        'final_text_produced': final_text_produced},
            'model': model,
            'tools': {'call_count': len(self._payloads('tool_started')),
                      'successful': sum(c['successful'] for c in tool_calls),
                      'failed': sum(not c['successful'] for c in tool_calls),
                      'duration_seconds': sum(c['duration_seconds'] for c in tool_calls),
                      'by_name': by_name, 'calls': tool_calls},
            'reasoning_replay': replay,
            'context': {'automatic_compactions': sum(p.get('trigger') == 'automatic' for p in checkpoints),
                        'manual_compactions': sum(p.get('trigger') != 'automatic' for p in checkpoints),
                        'budget_failures': sum(p['failed'] for p in context),
                        'estimates': context,
                        'largest_estimated_request': max(estimates) if estimates else None,
                        'mean_estimated_request': sum(estimates) / len(estimates) if estimates else None,
                        'post_compaction_estimates': [p['diagnostic'] for p in context if p['phase'] == 'after']},
            'timing': {'wall_seconds': observation.perf_counter() - self._started,
                       'model_seconds': sum(r['duration_seconds'] for r in requests),
                       'tool_seconds': sum(c['duration_seconds'] for c in tool_calls)},
        })

    def _payloads(self, kind: str) -> list[dict[str, Any]]:
        return [event['payload'] for event in self.events if event['kind'] == kind]
