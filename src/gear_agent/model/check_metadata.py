"""Allowlisted projections of Strata and Maya's read-only metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from gear_agent.artifact_privacy import CredentialRedactor


def model_label(value: str, privacy: CredentialRedactor) -> str:
    """Displays short model IDs, including namespace/name, without rendering paths or URLs."""
    if not re.fullmatch(r'[\w-][\w.:+-]*(?:/[\w-][\w.:+-]*)?', value, re.ASCII) or len(value) > 200:
        return '[REDACTED]'
    return privacy.text(value)


@dataclass
class ServerMetadata:
    reported_model: str | None = None
    model_ids: list[str] = field(default_factory=list)
    load_state: str = 'unknown'
    context_window_tokens: int | None = None
    supports_tools_hint: bool | None = None


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError('Expected metadata object')
    return value


def _context(value: object) -> int | None:
    if value is None or (type(value) is int and value == 0):
        return None
    if type(value) is not int or value < 0:
        raise ValueError('Invalid context size')
    return value


def project_metadata(documents: dict[str, dict[str, object]], profile: str,
                     privacy: CredentialRedactor) -> ServerMetadata:
    """Combines reported fields from the same profile, ignoring image-only models and aliases' capacities."""
    server = ServerMetadata()
    health = documents.get('health', {})
    props = documents.get('props', {})
    models = documents.get('models')
    names: list[str] = []
    primary: set[str] = set()
    states: set[str] = set()
    contexts: set[int] = set()
    for value in (health.get('model'), props.get('model_alias')):
        if value is not None:
            if not isinstance(value, str):
                raise ValueError('Invalid model name')
            names.append(value)
            primary.add(value)
    for value in (health.get('max_context'), _object(props.get('default_generation_settings', {})).get('n_ctx')):
        context = _context(value)
        if context is not None:
            contexts.add(context)
    if models is not None:
        entries = models.get('data')
        if not isinstance(entries, list):
            raise ValueError('Invalid models list')
        text_models = 0
        for entry in entries:
            model = _object(entry)
            modalities = _object(model.get('architecture', {})).get('output_modalities')
            if isinstance(modalities, list) and 'text' not in modalities:
                continue
            name = model.get('id')
            if not isinstance(name, str):
                raise ValueError('Invalid model ID')
            names.append(name)
            text_models += 1
            if 'alias_of' in model:
                continue
            primary.add(name)
            state = _object(model.get('status', {})).get('value')
            if state in ('loaded', 'unloaded', 'loading', 'reloading'):
                states.add(str(state))
            context = _context(_object(model.get('meta', {})).get('n_ctx'))
            if context is not None:
                contexts.add(context)
        if not text_models:
            states.add('no_model')
    if len(primary) == 1:
        server.reported_model = model_label(primary.pop(), privacy)
    server.model_ids = list(dict.fromkeys(model_label(name, privacy) for name in names))
    loaded = health.get('loaded') if profile == 'strata' else None
    if profile == 'maya' and health.get('unloaded') is True:
        states.add('unloaded')
    if loaded is False or props.get('is_sleeping') is True:
        states.add('unloaded')
    if loaded is True:
        states.add('loaded')
    if health.get('status') in ('updating', 'reloading', 'loading'):
        server.load_state = str(health['status'])
    elif states == {'unloaded', 'no_model'}:
        server.load_state = 'unloaded'
    elif len(states) == 1:
        server.load_state = states.pop()
    elif states:
        server.load_state = 'inconsistent'
    if server.load_state in ('unknown', 'loaded') and len(contexts) == 1:
        server.context_window_tokens = contexts.pop()
    hint = _object(props.get('chat_template_caps', {})).get('supports_tools')
    if type(hint) is bool:
        server.supports_tools_hint = hint
    return server
