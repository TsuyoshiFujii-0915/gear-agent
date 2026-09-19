from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit, parse_qsl, urlencode
import re

from gear_agent.config import AppConfig
from gear_agent.errors import GearError


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
        """
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {
                self.text(key): ('[REDACTED]' if _CREDENTIAL_FIELD.search(key) else self.serialize(item))
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.serialize(item) for item in value]
        return value

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
