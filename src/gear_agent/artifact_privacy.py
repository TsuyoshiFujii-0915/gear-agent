from __future__ import annotations

from typing import Any, Literal, Mapping
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit, parse_qsl, urlencode
import json
import re

from gear_agent.config import AppConfig
from gear_agent.errors import GearError


MAX_ARGUMENT_JSON_BYTES = 1024 * 1024
MAX_ARTIFACT_JSON_DEPTH = 64
ProjectionContext = Literal['events', 'tool_payload', 'data']


_CREDENTIAL_KEY = re.compile(r'authorization|cookie|api.?key|secret|password|passwd|token|credential|private.?key', re.I)
_CREDENTIAL_FIELD = re.compile(
    r'authorization|cookie|api.?key|secret|password|passwd|credential|private.?key|(?:^|[_-])token(?:$|[_-])', re.I,
)
_HEADER = re.compile(
    r'''(?im)((?:proxy-)?authorization["']?\s*[=:]\s*["']?)(?:Bearer\s+|Basic\s+)?([^\s"'<>]+)'''
)
_ASSIGNMENT = re.compile(
    r'''(?i)((?:api[_-]?key|password|passwd|secret|access[_-]?token)["']?\s*[=:]\s*["']?)[^\s&,;"'<>]+'''
)


class ArtifactPrivacy:
    """Central policy for safe artifact metadata, diagnostics and session copies."""

    def __init__(self, config: AppConfig, environment: Mapping[str, str]) -> None:
        values = [config.model.api_key]
        if config.web_search is not None:
            values.append(config.web_search.api_key)
        if config.web_fetch is not None:
            values.append(config.web_fetch.api_key)
        values.extend(value for key, value in environment.items() if _CREDENTIAL_KEY.search(key))
        endpoint = urlsplit(config.model.url)
        values.extend((endpoint.username, endpoint.password, endpoint.fragment))
        values.extend(value for key, value in parse_qsl(endpoint.query) if _CREDENTIAL_KEY.search(key))
        secrets = {value for value in values if value}
        self.secrets = tuple(sorted(secrets, key=lambda value: (-len(value), value)))
        encoded = {variant for value in secrets for variant in (value, quote(value, safe=''), quote_plus(value))}
        # Text fields can contain a JSON representation of a credential. Protocol
        # argument strings are decoded separately before reaching this policy.
        encoded.update(json.dumps(value, ensure_ascii=ascii_only)[1:-1]
                       for value in tuple(encoded) for ascii_only in (False, True))
        self._variants = tuple(sorted(encoded, key=lambda value: (-len(value), value)))

    def text(self, value: str) -> str:
        """Removes known secrets and common embedded credential syntax."""
        header_values = {match.group(2) for match in _HEADER.finditer(value)}
        for secret in sorted(header_values, key=lambda item: (-len(item), item)):
            value = value.replace(secret, '[REDACTED]')
        for secret in self._variants:
            value = value.replace(secret, '[REDACTED]')
        value = _HEADER.sub(r'\1[REDACTED]', value)
        return _ASSIGNMENT.sub(r'\1[REDACTED]', value)

    def serialize(self, value: Any) -> Any:
        """Copies JSON values with credential fields and text centrally redacted.

        Args:
            value: JSON-compatible metadata or canonical event payload.

        Returns:
            Safe JSON-compatible copy. Encrypted reasoning state is preserved only
            in canonical event payloads supplied by the snapshot writer.

        Raises:
            GearError: If a protocol argument cannot be safely projected or the
                size/depth limit is exceeded. No unfiltered copy is returned.
        """
        return self._project(value, 0, 'events')

    def _project(self, value: Any, depth: int, context: ProjectionContext) -> Any:
        """Copies data while keeping decoded arguments outside protocol interpretation.

        Args:
            value: Current JSON value.
            depth: Container depth of the artifact projection, including envelopes.
            context: Only event objects can contain encoded protocol arguments;
                decoded argument data must not be interpreted as another event.

        Returns:
            Redacted copy with encoded protocol arguments remaining JSON strings.

        Raises:
            GearError: If the projection exceeds its structural depth limit.
        """
        if depth > MAX_ARTIFACT_JSON_DEPTH:
            raise _json_error('artifact_json_limit_exceeded', 'Artifact JSON exceeds the depth limit.')
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            function_call = context == 'events' and value.get('type') == 'function_call'
            arguments = self._arguments(value.get('arguments'), depth + 1) if function_call else None
            projected: dict[str, Any] = {}
            for key, item in value.items():
                child_context = context
                if context == 'events' and value.get('kind') in ('tool_call', 'tool_result') and key == 'payload':
                    child_context = 'tool_payload'
                elif context == 'tool_payload' and key in ('arguments', 'result'):
                    child_context = 'data'
                if _CREDENTIAL_FIELD.search(key):
                    projected[self.text(key)] = '[REDACTED]'
                elif function_call and key == 'arguments':
                    projected[self.text(key)] = arguments
                else:
                    projected[self.text(key)] = self._project(item, depth + 1, child_context)
            return projected
        if isinstance(value, (list, tuple)):
            return [self._project(item, depth + 1, context) for item in value]
        return value

    def _arguments(self, value: Any, depth: int) -> str:
        """Decodes exactly one protocol JSON object, redacts it, and re-encodes it.

        Args:
            value: Required function_call.arguments JSON string.
            depth: Structural depth at the decoded argument root.

        Returns:
            Valid JSON object text containing only the artifact-side projection.

        Raises:
            GearError: If arguments are invalid, oversized or too deeply nested.
        """
        if not isinstance(value, str):
            raise _json_error('artifact_json_invalid', 'function_call.arguments must be a JSON string.')
        if len(value) > MAX_ARGUMENT_JSON_BYTES:
            raise _json_error('artifact_json_limit_exceeded', 'function_call.arguments exceeds the byte limit.')
        try:
            size = len(value.encode('utf-8'))
        except UnicodeError as exc:
            raise _json_error('artifact_json_invalid', 'function_call.arguments is not valid UTF-8.') from exc
        if size > MAX_ARGUMENT_JSON_BYTES:
            raise _json_error('artifact_json_limit_exceeded', 'function_call.arguments exceeds the byte limit.')
        try:
            decoded = json.loads(value, parse_constant=_reject_json_constant, object_pairs_hook=_unique_argument_object)
        except ValueError as exc:
            raise _json_error('artifact_json_invalid', 'function_call.arguments is not valid JSON.') from exc
        except RecursionError as exc:
            raise _json_error('artifact_json_limit_exceeded', 'function_call.arguments exceeds the depth limit.') from exc
        if not isinstance(decoded, dict):
            raise _json_error('artifact_json_invalid', 'function_call.arguments must encode an object.')
        projected = self._project(decoded, depth, 'data')
        if projected == decoded:
            return value
        return json.dumps(projected, ensure_ascii=True, sort_keys=True, allow_nan=False)

    def exact_text(self, value: str, source: str) -> str:
        """Rejects sensitive exact-text artifacts instead of silently altering input.

        Args:
            value: Task or canonical final text that must be preserved exactly.
            source: Artifact field used to identify the failure.

        Returns:
            Original text when it can be persisted without redaction.

        Raises:
            GearError: If exact persistence would expose a recognized credential.
        """
        if self.text(value) != value:
            raise GearError('artifact_sensitive_text', f'{source} contains recognized credential material.',
                            'run_artifacts', True, {'source': source})
        return value

    def endpoint(self, url: str) -> str:
        """Returns a credential-free endpoint before any identity fingerprinting."""
        endpoint = urlsplit(url)
        query = urlencode([(key, '[REDACTED]' if _CREDENTIAL_KEY.search(key) else self.text(value))
                           for key, value in parse_qsl(endpoint.query, keep_blank_values=True)])
        return self.text(urlunsplit((endpoint.scheme, endpoint.netloc.rsplit('@', 1)[-1],
                                    endpoint.path, query, '')))


def _json_error(error_type: str, message: str) -> GearError:
    return GearError(error_type, message, 'run_artifacts', True, {})


def _reject_json_constant(value: str) -> None:
    """Rejects non-JSON numeric constants without copying untrusted content."""
    raise _json_error('artifact_json_invalid', 'function_call.arguments contains a non-JSON number.')


def _unique_argument_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Rejects duplicate keys instead of losing unchecked values during parsing."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _json_error('artifact_json_invalid', 'function_call.arguments contains duplicate object keys.')
        result[key] = value
    return result
