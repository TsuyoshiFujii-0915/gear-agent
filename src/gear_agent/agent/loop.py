from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.compaction_strategy import CompactionStrategy, SummaryCompactionStrategy
from gear_agent.agent.history import build_model_history, select_effective_events
from gear_agent.agent.jev import JevClient, JevCompactionStrategy
from gear_agent.compaction_config import CompactionConfig, SUMMARY_COMPACTION
from gear_agent.agent.events import (
    AgentLoopEvent,
    AgentLoopEventSink,
    ContextBudgetEvaluated,
    ModelReasoningSummaryDelta,
    ModelRequestStarted,
    ModelTextDelta,
    ReasoningReplayEvaluated,
    ToolUseFinished,
    ToolUseStarted,
)
from gear_agent.errors import GearError, gear_error
from gear_agent.context_budget import (
    ByteTokenEstimator, ContextBudgetConfig, ContextBudgetManager, ContextRequest,
    DISABLED_CONTEXT_BUDGET, context_budget_error,
)
from gear_agent.model.adapter import ModelAdapter
from gear_agent.model.events import (
    ModelFunctionCallArgumentsDelta,
    ModelOutputItemCompleted,
    ModelProgressEvent,
    ModelProgressEventSink,
    ModelReasoningSummaryDelta as ProviderReasoningSummaryDelta,
    ModelReasoningTextDelta,
    ModelTextDelta as ProviderTextDelta,
)
from gear_agent.model.replay import ReasoningReplayDiagnostic
from gear_agent.repository import RepositoryContext
from gear_agent.observation import RunObserver, execute_tool, record_model_usage, request_model
from gear_agent.store.base import ContextStore
from gear_agent.tools.base import Tool
from gear_agent.tools.registry import ToolRegistry


AGENT_INSTRUCTIONS = "\n".join(
    [
        "You are Gear Agent, a coding assistant operating inside one explicit workspace.",
        "Use workspace-relative paths for every tool argument that accepts a path.",
        "The workspace root is '.'. Use workdir='.' when running shell commands at the root.",
        "Absolute paths such as /testbed, /workspace, or host filesystem paths are invalid.",
        "When a tool returns an error, correct the tool arguments or explain the blocker.",
    ]
)

FINALIZATION_RETRY_INSTRUCTION = "\n".join(
    [
        "The previous response did not contain a user-facing final answer as output_text.",
        "If no tool call is needed, return the final answer for the user as output_text now.",
    ]
)


@dataclass(frozen=True)
class TurnResult:
    """Result of one user turn.

    Attributes:
        final_text: Final assistant text.
        iterations: Number of model calls made.
    """

    final_text: str
    iterations: int


class AgentLoop:
    """Coordinates model responses and tool execution."""

    def __init__(
        self,
        adapter: ModelAdapter,
        tools: list[Tool],
        store: ContextStore,
        event_sink: AgentLoopEventSink,
        repository_context: RepositoryContext,
        context_budget: ContextBudgetConfig = DISABLED_CONTEXT_BUDGET,
        observer: RunObserver | None = None,
        compaction_config: CompactionConfig = SUMMARY_COMPACTION,
    ) -> None:
        """Binds runtime services and the effective context policy.

        Args:
            adapter: Configured model adapter.
            tools: Enabled tools.
            store: Session audit store.
            event_sink: Progress and diagnostic consumer.
            repository_context: Current workspace instruction loader.
            context_budget: Defaults to disabled for existing embedding callers,
                matching the documented legacy configuration behavior.
            observer: Omitted by existing TUI/embedding callers to disable run-only observations.
            compaction_config: Defaults to summary for legacy embedding callers.
        """
        self._adapter = adapter
        self._replay_policy = adapter.replay_policy
        self._registry = ToolRegistry(tools)
        self._store = store
        self._event_sink = event_sink
        self._repository_context = repository_context
        self._context_budget = context_budget
        self._budget_manager = ContextBudgetManager(context_budget, ByteTokenEstimator())
        self._observer = observer
        self._compaction = CompactionService(adapter, observer)
        self._compaction_config = compaction_config
        self._summary_strategy: CompactionStrategy = SummaryCompactionStrategy(self._compaction, self._budget_manager)
        self._selective_strategy: CompactionStrategy | None = None
        if compaction_config.jev is not None:
            self._selective_strategy = JevCompactionStrategy(compaction_config.jev, JevClient(compaction_config.jev))

    def run_turn(
        self,
        session_id: str,
        user_text: str,
        max_iterations: int,
        timeout_seconds: int,
        stream_idle_timeout_seconds: int | None = None,
    ) -> TurnResult:
        """Runs one user turn until final text or explicit failure.

        Args:
            session_id: Session identifier.
            user_text: User message.
            max_iterations: Maximum model calls for this turn.
            timeout_seconds: HTTP timeout for each model call.
            stream_idle_timeout_seconds: Maximum idle time between stream bytes.

        Returns:
            Turn result.
        """

        if max_iterations < 1:
            raise gear_error(
                "iteration_limit_invalid",
                "max_iterations must be at least 1.",
                "agent_loop",
                True,
                {"max_iterations": max_iterations},
            )
        model_input = self._adapter.prepare_history(
            self._store.load(session_id),
            user_text,
        )
        input_items = model_input.items
        history_item_count = len(input_items) - 1
        self._store.append(session_id, "user_input", {"text": user_text})
        tools = self._registry.schemas()
        finalization_retry_used = False
        pending_replay_diagnostic: ReasoningReplayDiagnostic | None = model_input.diagnostic

        for iteration in range(1, max_iterations + 1):
            instructions = self._repository_context.instructions(
                AGENT_INSTRUCTIONS, self._store.load(session_id),
            )
            request = ContextRequest(input_items, tools, instructions, AGENT_INSTRUCTIONS, history_item_count)
            if self._context_budget.auto_compaction:
                request, pending_replay_diagnostic = self._budgeted_request(
                    request, pending_replay_diagnostic,
                    session_id, iteration, user_text, finalization_retry_used,
                    timeout_seconds, stream_idle_timeout_seconds,
                )
                input_items = cast(list[object], request.input_value)
                history_item_count = request.history_item_count
            if pending_replay_diagnostic is not None:
                self._publish_replay_diagnostic(session_id, pending_replay_diagnostic)
                pending_replay_diagnostic = None
            self._event_sink.publish(
                ModelRequestStarted(session_id=session_id, iteration=iteration)
            )
            if self._observer is not None:
                self._observer.record('repository_instructions', {
                    'iteration': iteration, 'files': self._repository_context.instruction_metadata,
                })
            response = request_model(
                self._adapter, request, timeout_seconds, stream_idle_timeout_seconds,
                _AgentModelProgressSink(self._event_sink, session_id, iteration),
                self._observer, 'agent',
            )
            self._store.append(
                session_id,
                "model_response",
                response.persisted_payload,
            )
            record_model_usage(response, self._observer)
            replayed_output = response.replayed_output
            output_items = replayed_output.items
            function_calls = response.function_calls
            if len(function_calls) == 0:
                final_text = response.text
                if _has_final_text(final_text):
                    self._store.append(session_id, "assistant_message", {"text": final_text})
                    self._complete_iteration(iteration)
                    return TurnResult(final_text=final_text, iterations=iteration)
                if not finalization_retry_used and iteration < max_iterations:
                    finalization_retry_used = True
                    if self._selective_strategy is not None:
                        self._store.append(session_id, 'continuation_instruction', {'text': FINALIZATION_RETRY_INSTRUCTION})
                    input_items.extend(output_items)
                    input_items.append(
                        self._adapter.user_message_item(FINALIZATION_RETRY_INSTRUCTION)
                    )
                    pending_replay_diagnostic = _nonempty_replay_diagnostic(
                        replayed_output.diagnostic
                    )
                    self._complete_iteration(iteration)
                    continue
                raise gear_error(
                    "final_text_missing",
                    "Model returned neither a tool call nor a final output_text.",
                    "agent_loop",
                    True,
                    {"iteration": iteration, "retry_used": finalization_retry_used},
                )

            input_items.extend(output_items)
            pending_replay_diagnostic = _nonempty_replay_diagnostic(
                replayed_output.diagnostic
            )
            for function_call in function_calls:
                self._store.append(
                    session_id,
                    "tool_call",
                    {
                        "call_id": function_call.call_id,
                        "iteration": iteration,
                        "name": function_call.name,
                        "arguments": function_call.arguments,
                    },
                )
                self._event_sink.publish(
                    ToolUseStarted(
                        session_id=session_id,
                        iteration=iteration,
                        call_id=function_call.call_id,
                        name=function_call.name,
                        arguments=function_call.arguments,
                    )
                )
                try:
                    tool_result = execute_tool(
                        self._registry, function_call.name, function_call.arguments,
                        function_call.call_id, self._observer,
                    )
                except GearError as exc:
                    if not exc.recoverable:
                        raise
                    tool_result = _recoverable_tool_error_result(exc)
                self._store.append(
                    session_id,
                    "tool_result",
                    {
                        "call_id": function_call.call_id,
                        "iteration": iteration,
                        "name": function_call.name,
                        "result": tool_result,
                    },
                )
                self._event_sink.publish(
                    ToolUseFinished(
                        session_id=session_id,
                        iteration=iteration,
                        call_id=function_call.call_id,
                        name=function_call.name,
                        result=tool_result,
                    )
                )
                input_items.append(self._adapter.tool_result_item(function_call.call_id, tool_result))
            self._complete_iteration(iteration)

        raise gear_error(
            "iteration_limit_reached",
            "Model did not produce a final answer before max_iterations.",
            "agent_loop",
            True,
            {"max_iterations": max_iterations},
        )

    def _complete_iteration(self, iteration: int) -> None:
        if self._observer is not None:
            self._observer.record('iteration_completed', {'iteration': iteration})

    def _budgeted_request(
        self,
        request: ContextRequest,
        replay_diagnostic: ReasoningReplayDiagnostic | None,
        session_id: str,
        iteration: int,
        user_text: str,
        finalization_retry_used: bool,
        timeout_seconds: int,
        stream_idle_timeout_seconds: int | None,
    ) -> tuple[ContextRequest, ReasoningReplayDiagnostic | None]:
        diagnostic = self._budget_manager.evaluate(request)
        triggered = not diagnostic.fits
        self._event_sink.publish(ContextBudgetEvaluated(
            session_id, iteration, 'before', diagnostic, triggered, False,
        ))
        if not triggered:
            return request, replay_diagnostic

        if self._selective_strategy is not None:
            return self._selective_budgeted_request(
                request, replay_diagnostic, session_id, iteration, user_text,
                finalization_retry_used, timeout_seconds, stream_idle_timeout_seconds,
            )
        before = diagnostic.total_estimated_request_tokens
        metrics: dict[str, Any] = {'strategy': 'summary', 'estimated_input_before': before,
                                  'estimated_input_after': None, 'reduction_ratio': None,
                                  'outcome': 'failed'}
        try:
            result = self._summary_budgeted_request(
                request, session_id, iteration, user_text, finalization_retry_used,
                timeout_seconds, stream_idle_timeout_seconds,
            )
            after = self._budget_manager.evaluate(result[0]).total_estimated_request_tokens
            metrics.update(estimated_input_after=after, reduction_ratio=1 - after / before, outcome='succeeded')
            return result
        finally:
            self._observe_compaction(metrics)

    def _summary_budgeted_request(
        self, request: ContextRequest, session_id: str, iteration: int,
        user_text: str, finalization_retry_used: bool, timeout_seconds: int,
        stream_idle_timeout_seconds: int | None,
    ) -> tuple[ContextRequest, ReasoningReplayDiagnostic | None]:
        compaction_request = self._compaction.prepare_request(self._store.load(session_id))
        compaction_diagnostic = self._budget_manager.evaluate_compaction(compaction_request)
        self._event_sink.publish(ContextBudgetEvaluated(
            session_id, iteration, 'compaction', compaction_diagnostic, True, not compaction_diagnostic.fits,
        ))
        if not compaction_diagnostic.fits:
            raise context_budget_error(compaction_diagnostic, 'compaction')
        candidate = self._summary_strategy.compact(
            self._store.load(session_id), timeout_seconds, stream_idle_timeout_seconds,
        )
        self._store.append(session_id, candidate.kind, candidate.payload)

        events = self._store.load(session_id)
        rebuilt = self._adapter.prepare_history(events, user_text)
        history_item_count = len(rebuilt.items) - 1
        if finalization_retry_used:
            rebuilt.items.append(self._adapter.user_message_item(FINALIZATION_RETRY_INSTRUCTION))
        request = ContextRequest(
            rebuilt.items, request.tools,
            self._repository_context.instructions(AGENT_INSTRUCTIONS, events),
            AGENT_INSTRUCTIONS, history_item_count,
        )
        diagnostic = self._budget_manager.evaluate(request)
        self._event_sink.publish(ContextBudgetEvaluated(
            session_id, iteration, 'after', diagnostic, True, not diagnostic.fits,
        ))
        if not diagnostic.fits:
            raise context_budget_error(diagnostic, 'after')
        return request, rebuilt.diagnostic

    def _selective_budgeted_request(
        self, request: ContextRequest, replay_diagnostic: ReasoningReplayDiagnostic | None,
        session_id: str, iteration: int, user_text: str, finalization_retry_used: bool,
        timeout_seconds: int, stream_idle_timeout_seconds: int | None,
    ) -> tuple[ContextRequest, ReasoningReplayDiagnostic | None]:
        if self._selective_strategy is None:
            raise RuntimeError('Selective compaction strategy was not configured.')
        before = self._budget_manager.evaluate(request).total_estimated_request_tokens
        metrics: dict[str, Any] = {'strategy': 'jev', 'estimated_input_before': before,
                                  'estimated_input_after': None, 'reduction_ratio': None,
                                  'fallback_reason': None, 'fallback_outcome': 'not_needed'}
        try:
            candidate = self._selective_strategy.compact(
                self._store.load(session_id), timeout_seconds, stream_idle_timeout_seconds,
            )
            metrics.update(candidate.metrics)
            checkpoint = {'kind': candidate.kind, 'payload': candidate.payload}
            selected = select_effective_events([checkpoint])
            # Validate the complete replay, including assistant deduplication.
            build_model_history([checkpoint], self._replay_policy)
            current_index = next((i for i in range(len(selected) - 1, -1, -1)
                                  if selected[i]['kind'] == 'user_input'), None)
            if current_index is None or selected[current_index]['payload']['text'] != user_text:
                raise gear_error('jev_current_turn_invalid', 'Selective history lost the current user boundary.',
                                 'compaction', True, {})
            rebuilt = self._adapter.prepare_history(selected[:current_index], user_text)
            history_count = len(rebuilt.items) - 1
            # Preserve exact in-memory tool outputs, reasoning and synthetic input.
            active_items = cast(list[object], request.input_value)[request.history_item_count:]
            rebuilt.items[history_count:] = active_items
            candidate_request = ContextRequest(
                rebuilt.items, request.tools,
                self._repository_context.instructions(AGENT_INSTRUCTIONS, self._store.load(session_id)),
                request.base_instructions, history_count,
            )
            # Snapshot active outputs exactly, bypassing legacy replay truncation.
            active_outputs = {item['call_id']: item['output'] for item in active_items
                              if isinstance(item, dict) and item.get('type') == 'function_call_output'}
            for event in candidate.payload['events'][current_index:]:
                if event['kind'] == 'tool_result':
                    call_id = event['payload']['call_id']
                    if call_id not in active_outputs:
                        raise gear_error('jev_current_turn_invalid', 'Current tool result has no live replay item.',
                                         'compaction', True, {})
                    event['payload']['model_visible_output'] = active_outputs[call_id]
            replayed = build_model_history([checkpoint], self._replay_policy)
            if replayed.items != candidate_request.input_value:
                raise gear_error('jev_replay_mismatch', 'Selective checkpoint differs from the candidate request.',
                                 'compaction', True, {})
            after = self._budget_manager.evaluate(candidate_request)
            metrics['estimated_input_after'] = after.total_estimated_request_tokens
            metrics['reduction_ratio'] = 1 - after.total_estimated_request_tokens / before
            if not after.fits:
                raise gear_error('jev_insufficient_reduction', 'Selective history still exceeds the context budget.',
                                 'compaction', True, {})
        except GearError as error:
            if 'metrics' in error.details:
                metrics.update(error.details['metrics'])
            metrics['fallback_reason'] = error.error_type
            if self._compaction_config.fallback != 'summary':
                metrics['fallback_outcome'] = 'disabled'
                self._observe_compaction(metrics)
                raise
            try:
                result = self._summary_budgeted_request(
                    request, session_id, iteration, user_text, finalization_retry_used,
                    timeout_seconds, stream_idle_timeout_seconds,
                )
            except BaseException:
                metrics['fallback_outcome'] = 'failed'
                self._observe_compaction(metrics)
                raise
            metrics['fallback_outcome'] = 'succeeded'
            metrics['estimated_input_after'] = self._budget_manager.evaluate(result[0]).total_estimated_request_tokens
            metrics['reduction_ratio'] = 1 - metrics['estimated_input_after'] / before
            self._observe_compaction(metrics)
            return result
        # No selective state is persisted until all replay and budget checks pass.
        self._store.append(session_id, candidate.kind, candidate.payload)
        self._event_sink.publish(ContextBudgetEvaluated(session_id, iteration, 'after', after, True, False))
        self._observe_compaction(metrics)
        diagnostic = rebuilt.diagnostic
        if iteration > 1 and replay_diagnostic is not None:
            diagnostic = diagnostic.combine(replay_diagnostic)
        return candidate_request, diagnostic

    def _observe_compaction(self, metrics: dict[str, Any]) -> None:
        if self._observer is not None:
            self._observer.record('compaction_strategy', metrics)

    def _publish_replay_diagnostic(
        self,
        session_id: str,
        diagnostic: ReasoningReplayDiagnostic,
    ) -> None:
        self._event_sink.publish(
            ReasoningReplayEvaluated(
                session_id=session_id,
                mode=self._replay_policy.mode,
                reused_encrypted_items=diagnostic.reused_encrypted_items,
                dropped_disabled_items=diagnostic.dropped_disabled_items,
                dropped_incompatible_scope_items=(
                    diagnostic.dropped_incompatible_scope_items
                ),
                dropped_missing_scope_items=diagnostic.dropped_missing_scope_items,
            )
        )


def _nonempty_replay_diagnostic(
    diagnostic: ReasoningReplayDiagnostic,
) -> ReasoningReplayDiagnostic | None:
    handled_items = (
        diagnostic.reused_encrypted_items + diagnostic.dropped_encrypted_items
    )
    if handled_items == 0:
        return None
    return diagnostic


def _has_final_text(text: str) -> bool:
    return text.strip() != ""


def _recoverable_tool_error_result(error: GearError) -> dict[str, object]:
    return {
        "error": {
            "type": error.error_type,
            "message": error.message,
            "origin": error.origin,
            "details": error.details,
        }
    }


class _AgentModelProgressSink(ModelProgressEventSink):
    """Adds agent request context to safe, displayable model progress."""

    def __init__(
        self,
        event_sink: AgentLoopEventSink,
        session_id: str,
        iteration: int,
    ) -> None:
        self._event_sink = event_sink
        self._session_id = session_id
        self._iteration = iteration

    def publish(self, event: ModelProgressEvent) -> None:
        """Publishes only provider-neutral progress safe for presentation.

        Args:
            event: Model-layer progress event.

        Raises:
            ValueError: If a new model progress event type is not handled.
        """

        display_event: AgentLoopEvent
        if isinstance(event, ProviderTextDelta):
            display_event = ModelTextDelta(
                session_id=self._session_id,
                iteration=self._iteration,
                delta=event.delta,
            )
            self._event_sink.publish(display_event)
            return
        if isinstance(event, ProviderReasoningSummaryDelta):
            display_event = ModelReasoningSummaryDelta(
                session_id=self._session_id,
                iteration=self._iteration,
                delta=event.delta,
            )
            self._event_sink.publish(display_event)
            return
        if isinstance(event, ModelReasoningTextDelta):
            return
        if isinstance(event, ModelFunctionCallArgumentsDelta):
            return
        if isinstance(event, ModelOutputItemCompleted):
            return
        raise ValueError(f"Unsupported model progress event: {type(event).__name__}")
