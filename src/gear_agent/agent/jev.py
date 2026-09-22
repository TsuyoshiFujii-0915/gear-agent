from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from time import perf_counter
from typing import Any, Protocol

from typesafe_sdk import Noul, RetryPolicy, TypeSafeClient, TypeSafeError

from gear_agent.agent.compaction_strategy import CompactionCandidate
from gear_agent.agent.history import select_effective_events, validate_tool_pairs
from gear_agent.compaction_config import JevConfig
from gear_agent.context_budget import ByteTokenEstimator
from gear_agent.errors import GearError
from gear_agent.model.replay import read_model_response_event


READ_ONLY_TOOLS = frozenset({'file_read', 'glob', 'grep', 'web_search', 'web_fetch'})
ARGUMENT_STATE_MAX_CHARS = 1000


@dataclass(frozen=True)
class JevAnswers:
    """Validated relevance scores and provider-reported token counts."""

    scores: dict[str, float]
    usage: dict[str, int | None]


class JevEvaluator(Protocol):
    """External batched relevance API boundary."""

    def evaluate(self, state: str, questions: dict[str, str]) -> JevAnswers:
        """Evaluates independent Noul questions against shared state."""
        ...


class JevClient:
    """Official TypeSafe SDK transport, isolated from the main model adapter."""

    def __init__(self, config: JevConfig) -> None:
        self._config = config

    def evaluate(self, state: str, questions: dict[str, str]) -> JevAnswers:
        """Calls the pinned model once without leaking provider error bodies.

        Args:
            state: Sanitized shared conversation context.
            questions: Stable question identifiers and relevance instructions.

        Returns:
            Noul scores and available usage counts.

        Raises:
            GearError: On transport, timeout, model mismatch or invalid answers.
        """
        try:
            with TypeSafeClient(api_key=self._config.api_key, model=self._config.model,
                                timeout=self._config.timeout_seconds,
                                retry=RetryPolicy(max_retries=0)) as client:
                response = client.system_one(state=state, questions={
                    key: Noul(instructions=value) for key, value in questions.items()
                }, model=self._config.model)
            if response.model != self._config.model:
                raise _error('model_mismatch')
            if set(response.answers) != set(questions) or set(response.nouls) != set(questions):
                raise _error('invalid_response')
            scores = {key: answer.noul for key, answer in response.nouls.items()}
            _validate_scores(scores, set(questions))
            usage = {'input_tokens': response.usage.input_tokens, 'output_tokens': response.usage.output_tokens}
            if any(value is not None and (type(value) is not int or value < 0) for value in usage.values()):
                raise _error('invalid_usage')
            return JevAnswers(scores, usage)
        except (TypeSafeError, TimeoutError, ValueError) as exc:
            reason = 'timeout' if isinstance(exc, TimeoutError) or 'Timeout' in type(exc).__name__ else 'provider_error'
            raise _error(reason) from None


@dataclass(frozen=True)
class Interaction:
    """A semantic call/result pair, independent of physical JSONL lines."""

    call_id: str
    call: dict[str, Any]
    result: dict[str, Any]
    turn: int


class JevCompactionStrategy:
    """Selects completed historical reads and returns an uncommitted snapshot."""

    def __init__(self, config: JevConfig, client: JevEvaluator) -> None:
        self._config = config
        self._client = client

    def compact(
        self, events: list[dict[str, Any]], timeout_seconds: int,
        stream_idle_timeout_seconds: int | None,
    ) -> CompactionCandidate:
        """Selects call/result units with conservative protection and batching.

        Args:
            events: Original session events including the active user turn.
            timeout_seconds: Main-model timeout; Jev uses its separate timeout.
            stream_idle_timeout_seconds: Main-model setting, unused by Jev.

        Returns:
            Flattened selected events and content-free counts.

        Raises:
            GearError: On invalid history, limits, provider failures or scores.
        """
        started = perf_counter()
        metrics: dict[str, Any] = {
            'strategy': 'jev', 'jev_model': self._config.model,
            'eligible_interactions': 0, 'pinned_interactions': 0,
            'kept_calls': 0, 'kept_results': 0, 'truncated_results': 0,
            'dropped_pairs': 0, 'jev_requests': 0, 'jev_usage': [],
        }
        try:
            effective = select_effective_events(events)
            validate_tool_pairs(effective)
            pairs, pinned_count = _eligible_interactions(effective, self._config.policy.preserve_recent_turns)
            metrics['eligible_interactions'] = len(pairs)
            metrics['pinned_interactions'] = pinned_count
            metrics['kept_calls'] = pinned_count
            metrics['kept_results'] = sum(event['kind'] == 'tool_result' for event in effective) - len(pairs)
            if not pairs:
                raise _error('no_eligible_interactions')
            state = _build_state(effective)
            estimator = ByteTokenEstimator()
            if estimator.estimate(state) > self._config.policy.max_state_tokens:
                raise _error('state_limit')
            batches = self._batches(state, pairs, estimator)
            scores: dict[str, float] = {}
            for questions in batches:
                metrics['jev_requests'] += 1
                answer = self._client.evaluate(state, questions)
                _validate_scores(answer.scores, set(questions))
                if set(answer.usage) - {'input_tokens', 'output_tokens'} or any(
                    value is not None and (type(value) is not int or value < 0)
                    for value in answer.usage.values()
                ):
                    raise _error('invalid_usage')
                metrics['jev_usage'].append(dict(answer.usage))
                scores.update(answer.scores)
            dropped: set[str] = set()
            truncated: dict[str, dict[str, object]] = {}
            for pair in pairs:
                keep_call = scores[f'{pair.call_id}:call'] >= self._config.policy.keep_threshold
                keep_result = scores[f'{pair.call_id}:result'] >= self._config.policy.keep_threshold
                if keep_result:
                    metrics['kept_calls'] += 1
                    metrics['kept_results'] += 1
                elif keep_call:
                    metrics['kept_calls'] += 1
                    serialized = json.dumps(pair.result, ensure_ascii=False)
                    replacement = {'truncated': True, 'original_json_chars': len(serialized),
                                   'preview': serialized[:self._config.policy.truncate_chars]}
                    if len(json.dumps(replacement, ensure_ascii=False)) < len(serialized):
                        truncated[pair.call_id] = replacement
                        metrics['truncated_results'] += 1
                    else:
                        metrics['kept_results'] += 1
                else:
                    dropped.add(pair.call_id)
                    metrics['dropped_pairs'] += 1
            selected = _select_events(effective, dropped, truncated)
            validate_tool_pairs(selected)
            _validate_selection(effective, selected, dropped, truncated)
            metrics['jev_latency_seconds'] = perf_counter() - started
            return CompactionCandidate('compaction_selective', {
                'schema': 'gear-agent.selective.v1', 'events': selected, 'trigger': 'automatic',
            }, metrics)
        except (GearError, TimeoutError) as exc:
            metrics['jev_latency_seconds'] = perf_counter() - started
            reason = exc.error_type if isinstance(exc, GearError) else 'jev_timeout'
            raise GearError(reason, 'Jev selective compaction could not produce a valid checkpoint.',
                            'compaction', True, {'metrics': metrics}) from None

    def _batches(
        self, state: str, pairs: list[Interaction], estimator: ByteTokenEstimator,
    ) -> list[dict[str, str]]:
        batches: list[dict[str, str]] = []
        batch: dict[str, str] = {}
        for pair in pairs:
            questions = {
                f'{pair.call_id}:call': f'Does knowing tool call {pair.call_id} and its inputs still matter for continuing the current user goal? Keep when uncertain.',
                f'{pair.call_id}:result': f'Does the full result of tool call {pair.call_id} still need to be available verbatim for the current goal, rather than re-reading it? Keep when uncertain.',
            }
            combined = {**batch, **questions}
            if batch and (len(combined) > self._config.policy.batch_size * 2 or not self._request_fits(state, combined, estimator)):
                batches.append(batch)
                combined = questions
            if not self._request_fits(state, combined, estimator):
                raise _error('request_limit')
            batch = combined
        if batch:
            batches.append(batch)
        return batches

    def _request_fits(self, state: str, questions: dict[str, str], estimator: ByteTokenEstimator) -> bool:
        request = {'model': self._config.model, 'state': state, 'questions': {
            key: {'type': 'noul', 'instructions': value} for key, value in questions.items()
        }}
        return estimator.estimate(request) + 256 <= self._config.policy.max_request_tokens


def _eligible_interactions(events: list[dict[str, Any]], recent_turns: int) -> tuple[list[Interaction], int]:
    calls: dict[str, tuple[dict[str, Any], int]] = {}
    results: dict[str, dict[str, Any]] = {}
    result_turns: dict[str, int] = {}
    completed_turns: set[int] = set()
    failed_turns: set[int] = set()
    turn = -1
    for event in events:
        kind, payload = event['kind'], event['payload']
        if kind == 'user_input':
            turn += 1
        elif kind == 'assistant_message':
            completed_turns.add(turn)
        elif kind == 'turn_error':
            failed_turns.add(turn)
        elif kind == 'model_response':
            for item in read_model_response_event(payload).response['output']:
                if item['type'] == 'function_call':
                    calls[item['call_id']] = (item, turn)
        elif kind == 'tool_result':
            results[payload['call_id']] = payload['result']
            result_turns[payload['call_id']] = turn
    for call_id, (_, call_turn) in calls.items():
        if call_id not in results:
            failed_turns.add(call_turn)
    eligible: list[Interaction] = []
    for call_id, (call, call_turn) in calls.items():
        result = results.get(call_id)
        if (result is None or result_turns[call_id] != call_turn
                or call_turn < 0 or call_turn >= turn - recent_turns
                or call_turn not in completed_turns or call_turn in failed_turns
                or call['name'] not in READ_ONLY_TOOLS or 'error' in result
                or result.get('timed_out', False) or result.get('exit_code', 0) != 0):
            continue
        eligible.append(Interaction(call_id, call, result, call_turn))
    return eligible, len(calls) - len(eligible)


def _build_state(events: list[dict[str, Any]]) -> str:
    state: list[dict[str, Any]] = []
    for event in events:
        kind, payload = event['kind'], event['payload']
        if kind in ('user_input', 'assistant_message', 'compaction_summary', 'continuation_instruction'):
            state.append({'kind': kind, 'text': payload['text']})
        elif kind == 'model_response':
            for item in read_model_response_event(payload).response['output']:
                if item['type'] == 'function_call':
                    arguments = item['arguments']
                    state.append({'kind': 'call', 'call_id': item['call_id'], 'name': item['name'],
                                  'arguments': arguments[:ARGUMENT_STATE_MAX_CHARS],
                                  'arguments_truncated': len(arguments) > ARGUMENT_STATE_MAX_CHARS})
                elif item['type'] == 'message':
                    for content in item['content']:
                        if content.get('type') == 'output_text':
                            state.append({'kind': 'assistant', 'text': content['text']})
        elif kind == 'tool_result':
            result = payload['result']
            state.append({'kind': 'result', 'call_id': payload['call_id'],
                          'status': 'error' if 'error' in result else 'ok',
                          'result_chars': len(json.dumps(result, ensure_ascii=False)), 'body_omitted': True})
        elif kind == 'turn_error':
            state.append({'kind': 'turn_error'})
    return json.dumps(state, ensure_ascii=False)


def _select_events(
    events: list[dict[str, Any]], dropped: set[str], truncated: dict[str, dict[str, object]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for event in deepcopy(events):
        kind, payload = event['kind'], event['payload']
        if kind in ('tool_call', 'tool_result') and payload.get('call_id') in dropped:
            continue
        if kind == 'model_response':
            response = read_model_response_event(payload).response
            response['output'] = [item for item in response['output']
                                  if not (item['type'] == 'function_call' and item['call_id'] in dropped)]
        elif kind == 'tool_result' and payload['call_id'] in truncated:
            payload['result'] = truncated[payload['call_id']]
            payload.pop('model_visible_output', None)
        selected.append(event)
    return selected


def _validate_selection(
    original: list[dict[str, Any]], selected: list[dict[str, Any]],
    dropped: set[str], truncated: dict[str, dict[str, object]],
) -> None:
    """Checks that all differences are authorized call/result decisions."""
    restored = iter(selected)
    for event in original:
        kind, payload = event['kind'], event['payload']
        if kind in ('tool_call', 'tool_result') and payload.get('call_id') in dropped:
            continue
        candidate = next(restored, None)
        if candidate is None:
            raise _error('selection_invalid')
        expected = deepcopy(event)
        if kind == 'model_response':
            response = read_model_response_event(expected['payload']).response
            response['output'] = [item for item in response['output'] if not (
                item['type'] == 'function_call' and item['call_id'] in dropped)]
        elif kind == 'tool_result' and payload['call_id'] in truncated:
            expected['payload']['result'] = truncated[payload['call_id']]
            expected['payload'].pop('model_visible_output', None)
        if expected != candidate:
            raise _error('selection_invalid')
    if next(restored, None) is not None:
        raise _error('selection_invalid')


def _validate_scores(scores: dict[str, float], expected: set[str]) -> None:
    if set(scores) != expected or any(type(value) not in (int, float) or not math.isfinite(value)
                                      or not 0 <= value <= 1 for value in scores.values()):
        raise _error('invalid_response')


def _error(reason: str) -> GearError:
    return GearError(f'jev_{reason}', 'Jev compaction failed: ' + reason + '.', 'compaction', True, {})
