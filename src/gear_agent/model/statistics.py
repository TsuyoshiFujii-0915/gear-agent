"""Request-local observations from Responses usage and Strata-compatible timings."""

from dataclasses import dataclass, replace
from math import isfinite
from typing import Literal


Unit = Literal['tokens', 'tokens/s', 'ms', 's', 'fraction']
Schema = Literal['responses_usage', 'strata_qwen_timings']
Availability = Literal['available', 'missing', 'invalid', 'not_applicable']
Reason = Literal['not_reported', 'invalid_number', 'invalid_container', 'exceeds_input_tokens',
                 'exceeds_output_tokens', 'zero_input_tokens', 'zero_duration', 'zero_tokens',
                 'out_of_range']


@dataclass(frozen=True)
class SourceField:
    """Allowlisted wire field and its validated numeric operand, in original units."""

    schema: Schema
    path: str
    unit: Unit
    value: int | float | None


@dataclass(frozen=True)
class Metric:
    """One observation with its unit, availability and reproducible derivation."""

    value: int | float | None
    unit: Unit
    availability: Availability
    reason: Reason | None
    origin: Literal['server_reported', 'derived'] | None
    formula: str | None
    sources: tuple[SourceField, ...]


@dataclass(frozen=True)
class ResponseStatistics:
    """Safe numeric observations for one response, independent of semantic output."""

    input_tokens: Metric
    output_tokens: Metric
    total_tokens: Metric
    reasoning_tokens: Metric
    cached_tokens: Metric
    cache_write_tokens: Metric
    cache_hit_ratio: Metric
    prefill_tokens: Metric
    decode_tokens: Metric
    prefill_seconds: Metric
    decode_seconds: Metric
    prefill_rate: Metric
    decode_rate: Metric


@dataclass(frozen=True)
class _Fields:
    schema: Schema
    path: str
    data: object

    def read(self, name: str, unit: Unit) -> Metric:
        field = SourceField(self.schema, f'{self.path}.{name}', unit, None)
        value = self.data
        for part in name.split('.'):
            if value is None:
                break
            if not isinstance(value, dict):
                return _unavailable(unit, 'invalid', 'invalid_container', (field,))
            value = value.get(part)
        if value is None:
            return _unavailable(unit, 'missing', 'not_reported', (field,))
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            return _unavailable(unit, 'invalid', 'invalid_number', (field,))
        if unit == 'tokens' and not isinstance(value, int):
            return _unavailable(unit, 'invalid', 'invalid_number', (field,))
        if isinstance(value, float) and not isfinite(value):
            return _unavailable(unit, 'invalid', 'invalid_number', (field,))
        return Metric(value, unit, 'available', None, 'server_reported', None,
                      (replace(field, value=value),))


def _unavailable(unit: Unit, state: Availability, reason: Reason,
                 sources: tuple[SourceField, ...]) -> Metric:
    return Metric(None, unit, state, reason, None, None, sources)


def _missing_operand(unit: Unit, left: Metric, right: Metric) -> Metric:
    sources = tuple(dict.fromkeys((*left.sources, *right.sources)))
    for operand in (left, right):
        if operand.availability == 'invalid' and operand.reason is not None:
            return _unavailable(unit, 'invalid', operand.reason, sources)
    return _unavailable(unit, 'missing', 'not_reported', sources)


def _subset(count: Metric, total: Metric, reason: Reason) -> Metric:
    if count.value is not None and total.value is not None and count.value > total.value:
        return _unavailable('tokens', 'invalid', reason, count.sources + total.sources)
    return count


def _cache_ratio(cached: Metric, inputs: Metric) -> Metric:
    if cached.value is None or inputs.value is None:
        return _missing_operand('fraction', cached, inputs)
    sources = cached.sources + inputs.sources
    if inputs.value == 0:
        return _unavailable('fraction', 'not_applicable', 'zero_input_tokens', sources)
    fraction = cached.value / inputs.value
    if fraction == 0 and cached.value != 0:
        return _unavailable('fraction', 'invalid', 'out_of_range', sources)
    return Metric(fraction, 'fraction', 'available', None, 'derived',
                  'cached_tokens / input_tokens', sources)


def _total(reported: Metric, inputs: Metric, outputs: Metric) -> Metric:
    if reported.availability != 'missing':
        return reported
    if inputs.value is None or outputs.value is None:
        return _missing_operand('tokens', inputs, outputs)
    return Metric(inputs.value + outputs.value, 'tokens', 'available', None, 'derived',
                  'input_tokens + output_tokens', inputs.sources + outputs.sources)


def _seconds(milliseconds: Metric) -> Metric:
    if milliseconds.value is None:
        return replace(milliseconds, unit='s')
    try:
        seconds = milliseconds.value / 1000
    except OverflowError:
        return _unavailable('s', 'invalid', 'out_of_range', milliseconds.sources)
    if seconds == 0 and milliseconds.value != 0:
        return _unavailable('s', 'invalid', 'out_of_range', milliseconds.sources)
    return replace(milliseconds, value=seconds, unit='s', formula='ms / 1000')


def _rate(reported: Metric, tokens: Metric, seconds: Metric) -> Metric:
    sources = tokens.sources + seconds.sources
    if tokens.value == 0:
        return _unavailable('tokens/s', 'not_applicable', 'zero_tokens', tokens.sources)
    if seconds.value == 0:
        return _unavailable('tokens/s', 'not_applicable', 'zero_duration', seconds.sources)
    if reported.availability != 'missing':
        return reported
    if tokens.value is None or seconds.value is None:
        return _missing_operand('tokens/s', tokens, seconds)
    try:
        rate = tokens.value / seconds.value
    except OverflowError:
        return _unavailable('tokens/s', 'invalid', 'out_of_range', sources)
    if not isfinite(rate):
        return _unavailable('tokens/s', 'invalid', 'out_of_range', sources)
    return Metric(rate, 'tokens/s', 'available', None, 'derived', 'tokens / (ms / 1000)', sources)


@dataclass(frozen=True)
class _Phase:
    tokens: Metric
    seconds: Metric
    rate: Metric


def _strata_phase(fields: _Fields, phase: Literal['prompt', 'predicted']) -> _Phase:
    tokens = fields.read(f'{phase}_n', 'tokens')
    seconds = _seconds(fields.read(f'{phase}_ms', 'ms'))
    rate = _rate(fields.read(f'{phase}_per_second', 'tokens/s'), tokens, seconds)
    return _Phase(tokens, seconds, rate)


def _normalize(usage: _Fields, timings: _Fields) -> ResponseStatistics:
    inputs = usage.read('input_tokens', 'tokens')
    outputs = usage.read('output_tokens', 'tokens')
    cached = _subset(usage.read('input_tokens_details.cached_tokens', 'tokens'),
                     inputs, 'exceeds_input_tokens')
    reasoning = _subset(usage.read('output_tokens_details.reasoning_tokens', 'tokens'),
                        outputs, 'exceeds_output_tokens')
    writes = _subset(usage.read('input_tokens_details.cache_write_tokens', 'tokens'),
                     inputs, 'exceeds_input_tokens')
    prefill = _strata_phase(timings, 'prompt')
    decode = _strata_phase(timings, 'predicted')
    return ResponseStatistics(
        input_tokens=inputs, output_tokens=outputs,
        total_tokens=_total(usage.read('total_tokens', 'tokens'), inputs, outputs),
        reasoning_tokens=reasoning, cached_tokens=cached, cache_write_tokens=writes,
        cache_hit_ratio=_cache_ratio(cached, inputs),
        prefill_tokens=prefill.tokens, decode_tokens=decode.tokens,
        prefill_seconds=prefill.seconds, decode_seconds=decode.seconds,
        prefill_rate=prefill.rate, decode_rate=decode.rate,
    )


def json_statistics(response: dict[str, object]) -> ResponseStatistics:
    """Reads the usage and timings objects of one JSON response."""
    return _normalize(_Fields('responses_usage', 'usage', response.get('usage')),
                      _Fields('strata_qwen_timings', 'timings', response.get('timings')))


def terminal_statistics(event: dict[str, object], response: dict[str, object]) -> ResponseStatistics:
    """Reads a terminal SSE snapshot, preferring the whole event-level timing block."""
    if 'timings' in event:
        timings = _Fields('strata_qwen_timings', 'event.timings', event['timings'])
    else:
        timings = _Fields('strata_qwen_timings', 'event.response.timings', response.get('timings'))
    return _normalize(_Fields('responses_usage', 'event.response.usage', response.get('usage')), timings)
