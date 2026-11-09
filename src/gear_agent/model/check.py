"""Standalone, deadline-bounded model diagnostics. Only --infer sends a POST."""

from __future__ import annotations

from dataclasses import asdict, replace
from multiprocessing import get_context
from multiprocessing.connection import Connection
from time import monotonic
from typing import Any, Mapping
from urllib.parse import urlsplit
import json
import socket
import ssl

import httpx

from gear_agent.artifact_privacy import CredentialRedactor
from gear_agent.config import ModelCheckConfig
from gear_agent.context_budget import validate_output_reserve
from gear_agent.errors import GearError
from gear_agent.model.check_metadata import ServerMetadata, model_label, project_metadata
from gear_agent.model.events import SilentModelProgressEventSink
from gear_agent.model.factory import build_model_adapter

MAX_METADATA_BYTES = 1024 * 1024
PROBE_OUTPUT_TOKENS = 32


def _network_failure(exc: BaseException) -> str:
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, ssl.SSLError):
            return 'tls'
        if isinstance(cause, socket.gaierror):
            return 'dns'
        if isinstance(cause, ConnectionRefusedError):
            return 'connection_refused'
        cause = cause.__cause__ or cause.__context__
    if isinstance(exc, httpx.TimeoutException):
        return 'http_timeout'
    return 'network_error'


def _http_result(status: str, code: int | None) -> dict[str, Any]:
    auth = 'unknown'
    if code in (401, 403):
        auth = 'denied'
    elif code is not None and 200 <= code < 300:
        auth = 'request_allowed'
    connection = 'reached' if code is not None else 'unknown'
    if status in ('dns', 'connection_refused', 'tls'):
        connection = 'failed'
    return {'status': status, 'http_status': code, 'authentication': auth, 'connection': connection}


def _read(pipe: Connection, client: httpx.Client, url: str,
          name: str) -> tuple[dict[str, Any], dict[str, object] | None]:
    pipe.send(('start', name))
    method = 'HEAD' if name == 'endpoint' else 'GET'
    code = None
    try:
        with client.stream(method, url) as response:
            code = response.status_code
            pipe.send((name, _http_result('headers_received', code)))
            if name == 'endpoint' and code in (405, 501):
                return _http_result('method_unsupported', code), None
            if name != 'endpoint' and code in (404, 405, 501):
                return _http_result('unavailable', code), None
            if not response.is_success:
                return _http_result('http_error', code), None
            mime = response.headers.get('content-type', '').partition(';')[0].strip().lower()
            if name == 'endpoint':
                status = 'unexpected_content_type' if mime == 'text/html' else 'ok'
                return _http_result(status, code), None
            if mime != 'application/json' and not mime.endswith('+json'):
                return _http_result('unexpected_content_type', code), None
            body = bytearray()
            for chunk in response.iter_bytes(chunk_size=16384):
                if len(body) + len(chunk) > MAX_METADATA_BYTES:
                    return _http_result('metadata_too_large', code), None
                body.extend(chunk)
            try:
                document = json.loads(body)
            except (ValueError, RecursionError):
                return _http_result('invalid_json', code), None
            if not isinstance(document, dict):
                return _http_result('invalid_metadata', code), None
            return _http_result('ok', code), document
    except httpx.RequestError as exc:
        return _http_result(_network_failure(exc), code), None


def _infer(pipe: Connection, config: ModelCheckConfig, cap: int) -> dict[str, Any]:
    started = monotonic()
    endpoint = _http_result('not_run', None)

    def record_status(code: int) -> None:
        nonlocal endpoint
        endpoint = _http_result('ok' if 200 <= code < 300 else 'http_error', code)
        pipe.send(('endpoint', endpoint))

    adapter = build_model_adapter(replace(config.model, max_output_tokens=cap), on_http_status=record_status)
    result: dict[str, Any] = {'status': 'succeeded', 'terminal_status': None}
    try:
        response = adapter.create_response('Reply with OK.', [], '', config.timeout_seconds,
                                           config.idle_timeout_seconds, SilentModelProgressEventSink())
        # The concrete adapter exposes the validated wire object, including legacy omitted status.
        result['terminal_status'] = response.response.get('status')
        if not response.text.strip() or response.function_calls:
            raise GearError('unexpected_probe_output', 'Expected a text response.', 'model_check', True, {})
        result['usage'] = asdict(response.usage)
    except GearError as exc:
        error = exc.error_type
        if endpoint['http_status'] is None:
            network = _network_failure(exc)
            endpoint = _http_result(error if network == 'network_error' else network, None)
            pipe.send(('endpoint', endpoint))
        if error in ('response_incomplete', 'response_failed'):
            result['terminal_status'] = error.removeprefix('response_')
            if 'reason' in exc.details:
                result['reason'] = exc.details['reason']
        result.update(status='failed', error=error)
    except (ValueError, RecursionError):
        result.update(status='failed', error='invalid_response')
    result.update(authentication=endpoint['authentication'], elapsed_seconds=round(monotonic() - started, 4))
    return result


def _worker(pipe: Connection, config: ModelCheckConfig, profile: str, cap: int | None,
            prefix: str | None, privacy: CredentialRedactor) -> None:
    """Sends allowlisted observations in request order so earlier results survive a deadline."""
    model = config.model
    try:
        headers = {'Authorization': f'Bearer {model.api_key}'} if model.api_key is not None else {}
        documents: dict[str, dict[str, object]] = {}
        with httpx.Client(headers=headers, timeout=config.timeout_seconds, follow_redirects=False) as client:
            if cap is None:
                endpoint, _ = _read(pipe, client, model.url, 'endpoint')
                pipe.send(('endpoint', endpoint))
            if profile != 'generic' and prefix is not None:
                for name, path in [('health', '/health'), ('models', '/v1/models'), ('props', '/props')]:
                    observation, document = _read(pipe, client, prefix + path, name)
                    if document is not None:
                        documents[name] = document
                        try:
                            server = project_metadata(documents, profile, privacy)
                        except ValueError:
                            del documents[name]
                            observation['status'] = 'invalid_metadata'
                        else:
                            pipe.send(('server', asdict(server)))
                    pipe.send((name, observation))
        if cap is not None:
            pipe.send(('start', 'inference'))
            pipe.send(('inference', _infer(pipe, config, cap)))
        pipe.send(('done', None))
    except (OSError, ValueError, OverflowError, httpx.HTTPError) as exc:
        # Transport setup errors can carry credentials or paths in their messages.
        pipe.send(('worker_error', type(exc).__name__))
    finally:
        pipe.close()


def _endpoint_prefix(url: str) -> str | None:
    try:
        parsed = httpx.URL(url)
        parts = urlsplit(url)
    except (httpx.InvalidURL, ValueError) as exc:
        raise GearError('invalid_model_url', 'Invalid model URL.', 'model_check', True, {}) from exc
    if parsed.scheme not in ('http', 'https') or not parsed.host or parts.username is not None or parts.fragment:
        raise GearError('invalid_model_url', 'Expected an HTTP(S) endpoint without userinfo or fragment.',
                        'model_check', True, {})
    if '?' in url or '#' in url or not parts.path.endswith('/v1/responses'):
        return None
    return url.removesuffix('/v1/responses')


def run_model_check(config: ModelCheckConfig, profile: str, infer: bool, deadline_seconds: int,
                    environment: Mapping[str, str]) -> tuple[int, dict[str, Any]]:
    prefix = _endpoint_prefix(config.model.url)
    model = config.model
    privacy = CredentialRedactor(model.url, [model.api_key, urlsplit(model.url).hostname], environment)
    if model.api_key is not None and (not model.api_key.isascii() or
                                       any(ord(c) < 33 or ord(c) == 127 for c in model.api_key)):
        raise GearError('invalid_bearer_token', 'Invalid model Bearer credential.', 'model_check', True, {})
    warnings: list[str] = []
    if profile != 'generic' and prefix is None:
        warnings.append('metadata_prefix_unknown')
    report: dict[str, Any] = {
        'schema_version': 1, 'profile': profile, 'warnings': warnings,
        'settings': {'requested_model': model_label(model.model, privacy), 'stream': model.stream,
                     'reasoning_effort': model.reasoning_effort, 'max_output_tokens': model.max_output_tokens,
                     'context_window_tokens': config.budget.context_window_tokens,
                     'reserved_tokens': config.budget.reserved_tokens, 'timeout_seconds': config.timeout_seconds,
                     'stream_idle_timeout_seconds': config.idle_timeout_seconds, 'deadline_seconds': deadline_seconds},
        'endpoint': _http_result('not_run', None), 'metadata': {}, 'server': asdict(ServerMetadata()),
        'inference': {'status': 'not_run'},
    }
    cap = None
    if infer:
        limits = [PROBE_OUTPUT_TOKENS]
        if model.max_output_tokens is not None:
            limits.append(model.max_output_tokens)
        if config.budget.auto_compaction and config.budget.reserved_tokens > 0:
            limits.append(config.budget.reserved_tokens)
        cap = min(limits)
        validate_output_reserve(config.budget, cap)
        report['inference'].update(max_output_tokens=cap, format='sse' if model.stream else 'json',
                                   terminal_status=None)
    complete = _collect(report, config, profile, cap, prefix, privacy, deadline_seconds)
    server = report['server']
    if server['reported_model'] is not None and report['settings']['requested_model'] != server['reported_model']:
        warnings.append('verify_model_alias')
    runtime_context = server['context_window_tokens']
    window = config.budget.context_window_tokens
    if window is not None and runtime_context is not None and window > runtime_context:
        warnings.append('configured_context_exceeds_runtime')
    ok = (complete and report['endpoint']['status'] in ('ok', 'method_unsupported')
          and all(item['status'] in ('ok', 'unavailable') for item in report['metadata'].values())
          and server['load_state'] in ('unknown', 'loaded')
          and report['inference']['status'] in ('not_run', 'succeeded'))
    return (0 if ok else 1), report


def _collect(report: dict[str, Any], config: ModelCheckConfig, profile: str, cap: int | None,
             prefix: str | None, privacy: CredentialRedactor, deadline_seconds: int) -> bool:
    """Supervises all network I/O, including blocking DNS and streams that never go idle."""
    ctx = get_context('spawn')
    reader, writer = ctx.Pipe(duplex=False)
    worker = ctx.Process(target=_worker, args=(writer, config, profile, cap, prefix, privacy))
    end = monotonic() + deadline_seconds
    active = 'endpoint'
    started = monotonic()
    worker.start()
    writer.close()
    try:
        while True:
            remaining = end - monotonic()
            if remaining <= 0 or not reader.poll(remaining):
                raise TimeoutError
            kind, value = reader.recv()
            if kind == 'worker_error':
                report['error'] = {'type': 'diagnostic_worker_failed', 'origin': active, 'cause': value}
                return False
            if kind == 'done':
                break
            if kind == 'start':
                active, started = value, monotonic()
                if active in ('health', 'models', 'props'):
                    report['metadata'][active] = _http_result('started', None)
            elif kind in ('health', 'models', 'props'):
                report['metadata'][kind] = value
            elif kind == 'inference':
                report[kind].update(value)
            else:
                report[kind] = value
    except (TimeoutError, EOFError) as exc:
        reason = 'deadline_exceeded' if isinstance(exc, TimeoutError) else 'worker_failed'
        if active == 'inference':
            report['inference'].update(status='failed', error=reason, elapsed_seconds=monotonic() - started,
                                       authentication=report['endpoint']['authentication'])
        elif active == 'endpoint':
            report['endpoint']['status'] = reason
        else:
            report['metadata'][active]['status'] = reason
        return False
    finally:
        reader.close()
        if worker.is_alive():
            worker.kill()
        worker.join()
        worker.close()
    return True


def format_report(report: dict[str, Any]) -> str:
    """Formats the public JSON fields for an interactive terminal."""
    if 'error' in report:
        return 'Error: ' + ' '.join(report['error'].values())

    settings = report['settings']
    server = report['server']
    inference = report['inference']

    def display(value: object) -> str:
        return 'unknown' if value is None else str(value)

    def http_line(name: str, result: dict[str, Any]) -> str:
        code = result['http_status']
        http = f'HTTP {code}, ' if code is not None else ''
        return (f"{name}: {http}{result['status']} "
                f"(connection={result['connection']}, auth={result['authentication']})")

    lines = [f"Model check ({report['profile']})", http_line('Endpoint', report['endpoint'])]
    for name, observation in report['metadata'].items():
        lines.append(http_line(f'Metadata {name}', observation))
    lines.extend([
        f"Model: requested={settings['requested_model']}, reported={display(server['reported_model'])}, "
        f"state={server['load_state']}",
        f"Listed IDs: {', '.join(server['model_ids']) or '(none reported)'}",
        f"Context (tokens): configured={display(settings['context_window_tokens'])}, "
        f"reserved={settings['reserved_tokens']}, runtime={display(server['context_window_tokens'])}",
        f"Generation: stream={settings['stream']}, effort={display(settings['reasoning_effort'])}, "
        f"configured cap={display(settings['max_output_tokens'])}",
        f"Timeouts (s): network={settings['timeout_seconds']}, "
        f"stream read idle={display(settings['stream_idle_timeout_seconds'])}, total={settings['deadline_seconds']}",
        f"Tool support (server hint): {display(server['supports_tools_hint'])}",
        f"Small inference: {inference['status']}",
    ])
    if 'format' in inference:
        lines.append(f"  {inference['format'].upper()}, probe cap={inference['max_output_tokens']}, "
                     f"terminal={display(inference['terminal_status'])}, "
                     f"elapsed (s)={display(inference.get('elapsed_seconds'))}")
    if 'usage' in inference:
        lines.append('  Reported usage: ' + ', '.join(
            f'{key}={display(count)}' for key, count in inference['usage'].items()))
    if 'error' in inference:
        lines.append('  ' + inference['error'] + (f" ({inference['reason']})" if 'reason' in inference else ''))
    messages = {
        'verify_model_alias': 'Requested and reported model IDs differ. Check the server alias.',
        'configured_context_exceeds_runtime': 'Configured context exceeds the reported runtime context.',
        'metadata_prefix_unknown': 'Cannot derive metadata paths from this endpoint.',
    }
    lines.extend('Warning: ' + messages[warning] for warning in report['warnings'])
    return '\n'.join(lines)
