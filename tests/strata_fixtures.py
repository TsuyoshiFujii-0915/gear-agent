"""Synthetic protocol fixtures, not recordings or live Strata verification.

Based on Niko1221/Strata v0.2.7, commit
5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d, serve/responses.py and
docs/DETAILS.md#the-responses-api-and-codex-cli (Qwen chat path only).
This models Stream.start/in_progress/feed/close/finish, reasoning_text with
empty summary, and the optional strata.r1 Base64 representation. It does not
import upstream code, run a model, or assert cache performance. Omitted usage
and terminal output are deliberate compatibility variants beyond Strata's
normal complete snapshots. No credentials or real conversations are included.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
import base64
import json


REASONING_TEXT = 'Inspect the local file, then write and search.'
OPAQUE_REASONING = 'strata.r1:' + base64.b64encode(
    json.dumps({'text': REASONING_TEXT}, ensure_ascii=False).encode('utf-8'),
).decode('ascii')


def reasoning_item() -> dict[str, Any]:
    return {
        'id': 'rs_fixture', 'type': 'reasoning', 'status': 'completed',
        'summary': [], 'content': [{'type': 'reasoning_text', 'text': REASONING_TEXT}],
        'encrypted_content': OPAQUE_REASONING,
    }


def function_item(call_id: str, name: str, arguments: dict[str, str]) -> dict[str, Any]:
    return {'id': f'fc_{call_id}', 'type': 'function_call', 'status': 'completed',
            'call_id': call_id, 'name': name, 'arguments': json.dumps(arguments)}


def text_item(text: str) -> dict[str, Any]:
    return {'id': 'msg_fixture', 'type': 'message', 'status': 'completed', 'role': 'assistant',
            'content': [{'type': 'output_text', 'text': text, 'annotations': [], 'logprobs': []}]}


def completed_response(output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'id': 'resp_fixture', 'object': 'response', 'created_at': 1, 'completed_at': 2,
        'status': 'completed', 'background': False, 'error': None, 'incomplete_details': None,
        'model': 'qwen-fixture', 'output': output, 'store': False,
        'usage': {'input_tokens': 120, 'output_tokens': 40, 'total_tokens': 160,
                  'input_tokens_details': {'cached_tokens': 10},
                  'output_tokens_details': {'reasoning_tokens': 20}},
    }


def tool_exchanges() -> list[dict[str, Any]]:
    return [
        completed_response([reasoning_item(), text_item('Inspecting the file.'),
                            function_item('read', 'file_read', {'path': 'input.txt'})]),
        completed_response([
            function_item('write', 'file_write', {'path': 'output.txt', 'content': 'written'}),
            function_item('search', 'web_search', {'query': 'fixture query'}),
        ]),
        completed_response([text_item('Task complete.')]),
    ]


def stream_events(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Builds Strata-shaped lifecycle, item and fragmented delta events.

    Args:
        response: Synthetic terminal response including complete output items.

    Returns:
        Ordered wire events, including repeated prefill keepalives.
    """
    initial = {**deepcopy(response), 'status': 'in_progress', 'output': [], 'usage': None}
    initial.pop('completed_at', None)
    events: list[dict[str, Any]] = [{'type': 'response.created', 'response': initial}]
    events.extend({'type': 'response.in_progress', 'response': deepcopy(initial)} for _ in range(3))
    for index, item in enumerate(response['output']):
        opened = {**deepcopy(item), 'status': 'in_progress'}
        kind = item['type']
        if kind == 'reasoning':
            opened.pop('encrypted_content', None)
            opened['content'] = [{'type': 'reasoning_text', 'text': ''}]
        elif kind == 'message':
            opened['content'] = []
        elif kind == 'function_call':
            opened['arguments'] = ''
        else:
            raise ValueError(f'Unsupported synthetic fixture item: {kind}')
        events.append({'type': 'response.output_item.added', 'output_index': index, 'item': opened})
        fields: dict[str, Any] = {'item_id': item['id'], 'output_index': index}
        if kind == 'function_call':
            prefix, value, done_key = 'response.function_call_arguments', item['arguments'], 'arguments'
        else:
            fields['content_index'] = 0
            prefix = 'response.reasoning_text' if kind == 'reasoning' else 'response.output_text'
            value, done_key = item['content'][0]['text'], 'text'
        if kind == 'message':
            events.append({'type': 'response.content_part.added', **fields,
                           'part': {'type': 'output_text', 'text': '', 'annotations': [], 'logprobs': []}})
        midpoint = len(value) // 2
        events.extend({'type': f'{prefix}.delta', **fields, 'delta': fragment}
                      for fragment in (value[:midpoint], value[midpoint:]))
        events.append({'type': f'{prefix}.done', **fields, done_key: value})
        if kind == 'message':
            events.append({'type': 'response.content_part.done', **fields, 'part': deepcopy(item['content'][0])})
        events.append({'type': 'response.output_item.done', 'output_index': index, 'item': deepcopy(item)})
    events.append({'type': f'response.{response["status"]}', 'response': deepcopy(response)})
    for sequence, event in enumerate(events):
        event['sequence_number'] = sequence
    return events
