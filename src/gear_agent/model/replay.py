from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any
from urllib.parse import unquote_plus, urlsplit

from gear_agent.config import ModelConfig, ReasoningReplayMode
from gear_agent.errors import GearError, gear_error


RESPONSES_PROTOCOL = "responses"
MODEL_RESPONSE_ENVELOPE_SCHEMA = "gear-agent.model-response.v1"


@dataclass(frozen=True)
class ModelReplayScope:
    """Identity required to reuse provider-generated opaque model state.

    Attributes:
        protocol: Model protocol family.
        endpoint_identity: Non-secret fingerprint of the configured endpoint URL.
        model: Exact configured model identifier.
    """

    protocol: str
    endpoint_identity: str
    model: str


@dataclass(frozen=True)
class ReasoningReplayPolicy:
    """Current request policy for opaque reasoning replay.

    Attributes:
        mode: Configured replay mode.
        current_scope: Scope of the active model connection.
    """

    mode: ReasoningReplayMode
    current_scope: ModelReplayScope


@dataclass(frozen=True)
class ReasoningReplayDiagnostic:
    """Counts how opaque reasoning state was handled.

    Attributes:
        reused_encrypted_items: Opaque reasoning items kept for replay.
        dropped_disabled_items: Opaque items dropped because replay is disabled.
        dropped_incompatible_scope_items: Opaque items dropped on scope mismatch.
        dropped_missing_scope_items: Opaque items dropped because source scope is absent.
    """

    reused_encrypted_items: int
    dropped_disabled_items: int
    dropped_incompatible_scope_items: int
    dropped_missing_scope_items: int

    @property
    def dropped_encrypted_items(self) -> int:
        """Returns the total number of dropped opaque reasoning items."""

        return (
            self.dropped_disabled_items
            + self.dropped_incompatible_scope_items
            + self.dropped_missing_scope_items
        )

    def combine(
        self,
        other: ReasoningReplayDiagnostic,
    ) -> ReasoningReplayDiagnostic:
        """Combines two replay diagnostics.

        Args:
            other: Diagnostic to add.

        Returns:
            Combined diagnostic counts.
        """

        return ReasoningReplayDiagnostic(
            reused_encrypted_items=(
                self.reused_encrypted_items + other.reused_encrypted_items
            ),
            dropped_disabled_items=(
                self.dropped_disabled_items + other.dropped_disabled_items
            ),
            dropped_incompatible_scope_items=(
                self.dropped_incompatible_scope_items
                + other.dropped_incompatible_scope_items
            ),
            dropped_missing_scope_items=(
                self.dropped_missing_scope_items + other.dropped_missing_scope_items
            ),
        )


@dataclass(frozen=True)
class StoredModelResponse:
    """Parsed model response event payload and its replay scope.

    Attributes:
        response: Raw Responses-compatible response.
        source_scope: Stored replay scope, or None for a legacy raw response.
    """

    response: dict[str, Any]
    source_scope: ModelReplayScope | None


@dataclass(frozen=True)
class ReplayedOutput:
    """Sanitized output items and their replay diagnostic.

    Attributes:
        items: Output items safe for the current request.
        diagnostic: Counts describing opaque state handling.
    """

    items: list[dict[str, Any]]
    diagnostic: ReasoningReplayDiagnostic


def reasoning_replay_policy(config: ModelConfig) -> ReasoningReplayPolicy:
    """Builds the replay policy for a model configuration.

    Args:
        config: Active model configuration.

    Returns:
        Replay policy for the active connection.
    """

    return ReasoningReplayPolicy(
        mode=config.reasoning_replay,
        current_scope=ModelReplayScope(
            protocol=RESPONSES_PROTOCOL,
            endpoint_identity=_endpoint_identity(config.url, config.api_key),
            model=config.model,
        ),
    )


def model_response_event_payload(
    response: dict[str, Any],
    policy: ReasoningReplayPolicy,
) -> dict[str, Any]:
    """Builds a persisted model response payload.

    Disabled replay keeps the legacy raw-response layout. Encrypted replay
    stores an envelope that atomically associates the response with its scope.

    Args:
        response: Raw Responses-compatible response.
        policy: Replay policy used to create the response.

    Returns:
        Model response event payload.
    """

    if policy.mode is ReasoningReplayMode.NONE:
        return response
    return {
        "schema": MODEL_RESPONSE_ENVELOPE_SCHEMA,
        "response": response,
        "source": {
            "reasoning_replay": ReasoningReplayMode.ENCRYPTED.value,
            "replay_scope": _scope_payload(policy.current_scope),
        },
    }


def read_model_response_event(payload: dict[str, Any]) -> StoredModelResponse:
    """Reads either a legacy raw response or a scoped response envelope.

    Args:
        payload: Stored model response event payload.

    Returns:
        Parsed response and optional source scope.

    Raises:
        GearError: If a scoped response envelope is malformed.
    """

    is_envelope = _is_model_response_envelope(payload)
    if not is_envelope:
        return StoredModelResponse(response=payload, source_scope=None)

    response = payload.get("response")
    if not isinstance(response, dict):
        raise _history_shape_error(
            "model_response.response must be an object.",
            "model_response.response",
        )
    source = payload.get("source")
    if not isinstance(source, dict):
        raise _history_shape_error(
            "model_response.source must be an object.",
            "model_response.source",
        )
    reasoning_replay = source.get("reasoning_replay")
    if reasoning_replay != ReasoningReplayMode.ENCRYPTED.value:
        raise _history_shape_error(
            "model_response.source.reasoning_replay must be 'encrypted'.",
            "model_response.source",
        )
    raw_scope = source.get("replay_scope")
    if not isinstance(raw_scope, dict):
        raise _history_shape_error(
            "model_response.source.replay_scope must be an object.",
            "model_response.source",
        )
    return StoredModelResponse(
        response=response,
        source_scope=_scope_from_payload(raw_scope),
    )


def replay_output_items(
    items: list[dict[str, Any]],
    source_scope: ModelReplayScope | None,
    policy: ReasoningReplayPolicy,
) -> ReplayedOutput:
    """Sanitizes opaque reasoning content for the active replay policy.

    Args:
        items: Validated model response output items.
        source_scope: Scope that created the items, or None for legacy data.
        policy: Active replay policy.

    Returns:
        Safe output items and structured handling counts.
    """

    replayed_items: list[dict[str, Any]] = []
    diagnostic = empty_replay_diagnostic()
    for item in items:
        if item.get("type") != "reasoning" or "encrypted_content" not in item:
            replayed_items.append(item)
            continue
        if not isinstance(item["encrypted_content"], str):
            raise gear_error(
                "reasoning_replay_shape_invalid",
                "reasoning.encrypted_content must be a string.",
                "reasoning_replay",
                True,
                {},
            )
        sanitized_item = dict(item)
        if policy.mode is ReasoningReplayMode.NONE:
            sanitized_item.pop("encrypted_content")
            diagnostic = diagnostic.combine(
                ReasoningReplayDiagnostic(0, 1, 0, 0)
            )
        elif source_scope is None:
            sanitized_item.pop("encrypted_content")
            diagnostic = diagnostic.combine(
                ReasoningReplayDiagnostic(0, 0, 0, 1)
            )
        elif source_scope != policy.current_scope:
            sanitized_item.pop("encrypted_content")
            diagnostic = diagnostic.combine(
                ReasoningReplayDiagnostic(0, 0, 1, 0)
            )
        else:
            diagnostic = diagnostic.combine(
                ReasoningReplayDiagnostic(1, 0, 0, 0)
            )
        replayed_items.append(sanitized_item)
    return ReplayedOutput(items=replayed_items, diagnostic=diagnostic)


def empty_replay_diagnostic() -> ReasoningReplayDiagnostic:
    """Returns a replay diagnostic with zero counts."""

    return ReasoningReplayDiagnostic(
        reused_encrypted_items=0,
        dropped_disabled_items=0,
        dropped_incompatible_scope_items=0,
        dropped_missing_scope_items=0,
    )


def strip_opaque_reasoning_from_event(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Copies a model response event while stripping output-item ciphertext.

    Args:
        payload: Legacy or scoped model response event payload.

    Returns:
        Copied payload with reasoning output ciphertext removed.

    Raises:
        GearError: If a scoped response envelope is malformed.
    """

    stored_response = read_model_response_event(payload)
    response = dict(stored_response.response)
    output = response.get("output")
    if isinstance(output, list):
        sanitized_output: list[object] = []
        for item in output:
            if (
                isinstance(item, dict)
                and item.get("type") == "reasoning"
                and "encrypted_content" in item
            ):
                sanitized_item = dict(item)
                sanitized_item.pop("encrypted_content")
                sanitized_output.append(sanitized_item)
            else:
                sanitized_output.append(item)
        response["output"] = sanitized_output
    if not _is_model_response_envelope(payload):
        return response
    copied_payload = dict(payload)
    copied_payload["response"] = response
    return copied_payload


def _endpoint_identity(url: str, api_key: str | None) -> str:
    identity_material = _endpoint_identity_material(url, api_key)
    digest = sha256(identity_material.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _endpoint_identity_material(url: str, api_key: str | None) -> str:
    if api_key is None or api_key == "":
        return url

    parsed_url = urlsplit(url)
    query_components = parsed_url.query.split("&") if parsed_url.query else []
    components = [
        _credential_free_component(parsed_url.scheme, api_key),
        _credential_free_component(parsed_url.netloc, api_key),
        _credential_free_component(parsed_url.path, api_key),
        *[
            _credential_free_component(component, api_key)
            for component in query_components
        ],
        _credential_free_component(parsed_url.fragment, api_key),
    ]
    if all(kind == "literal" for kind, _ in components):
        return url
    return json.dumps(components, ensure_ascii=True, separators=(",", ":"))


def _credential_free_component(value: str, api_key: str) -> tuple[str, str]:
    if api_key in unquote_plus(value):
        return ("credential", "")
    return ("literal", value)


def _is_model_response_envelope(payload: dict[str, Any]) -> bool:
    if payload.get("schema") == MODEL_RESPONSE_ENVELOPE_SCHEMA:
        return True
    if "output" in payload:
        return False
    return "response" in payload or "source" in payload


def _scope_payload(scope: ModelReplayScope) -> dict[str, str]:
    return {
        "protocol": scope.protocol,
        "endpoint_identity": scope.endpoint_identity,
        "model": scope.model,
    }


def _scope_from_payload(payload: dict[str, object]) -> ModelReplayScope:
    return ModelReplayScope(
        protocol=_required_scope_string(payload, "protocol"),
        endpoint_identity=_required_scope_string(payload, "endpoint_identity"),
        model=_required_scope_string(payload, "model"),
    )


def _required_scope_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise _history_shape_error(
            f"model_response.source.replay_scope.{key} must be a string.",
            "model_response.source",
        )
    return value


def _history_shape_error(message: str, origin: str) -> GearError:
    return gear_error(
        "history_shape_invalid",
        message,
        origin,
        True,
        {},
    )
