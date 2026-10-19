"""Rerun Jev on one saved session at explicitly chosen retention thresholds."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

from examples.compaction_comparison import (
    CURRENT, RecordingJev, checkpoint_input, copy_source_run, interaction_changes,
    measurements, write_json,
)
from gear_agent.agent.history import build_model_history
from gear_agent.agent.jev import JevClient, JevCompactionStrategy, JevEvaluator
from gear_agent.agent.loop import AGENT_INSTRUCTIONS
from gear_agent.compaction_config import DEFAULT_JEV_POLICY, JevConfig
from gear_agent.config import load_config
from gear_agent.errors import GearError
from gear_agent.model.adapter import ModelAdapter
from gear_agent.model.factory import build_model_adapter
from gear_agent.repository import RepositoryContext
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.filesystem import FileReadTool


def run_case(
    events: list[dict[str, Any]], adapter: ModelAdapter, config: JevConfig,
    evaluator: JevEvaluator, output: Path, schemas: list[dict[str, object]],
    instructions: str, current: str,
) -> dict[str, Any]:
    """Runs a fresh selection and records the actual replay changes.

    Args:
        events: Original events, never a prior threshold's selected history.
        adapter: Replay adapter; no main-model inference is dispatched.
        config: Jev configuration for this threshold.
        evaluator: External Jev API boundary.
        output: Existing empty directory for this case.
        schemas: Constant main-model tool schemas for measurement.
        instructions: Constant main-model instructions for measurement.
        current: Pending current user message.

    Returns:
        Measured selection, metrics and per-interaction changes.
    """
    before = build_model_history(events, adapter.replay_policy).items
    baseline = measurements(before, schemas, instructions)
    started = perf_counter()
    candidate = JevCompactionStrategy(config, evaluator).compact(events, config.timeout_seconds, None)
    elapsed = perf_counter() - started
    after = checkpoint_input(adapter, candidate, current)
    measured = measurements(after, schemas, instructions)
    write_json(output / 'checkpoint.json', asdict(candidate))
    write_json(output / 'model-input.json', after)
    row = {'threshold': config.policy.keep_threshold, 'duration_seconds': elapsed,
           'measurements': measured, 'metrics': candidate.metrics,
           'reduction_ratio': 1 - measured['estimated_request_tokens'] / baseline['estimated_request_tokens'],
           'interactions': interaction_changes(before, after)}
    write_json(output / 'result.json', row)
    return row


def render_report(directory: Path, result: dict[str, Any]) -> None:
    """Writes the threshold comparison and all actual per-call scores."""
    lines = ['# Jev保持閾値の比較', '',
             'スコア >= 閾値で保持。閾値が低いほど残しやすく、高いほど削除・短縮されやすくなります。',
             'resultが閾値以上なら呼び出し・本文を保持。resultが未満でcallが以上なら本文を短縮',
             '（短縮表現が小さくなる場合）。両方未満なら呼び出し・結果を削除します。',
             '直近ターンなどの保護対象は閾値によらず保持します。', '',
             f"元の推定リクエスト: {result['baseline']['estimated_request_tokens']} tokens", '',
             '| 閾値 | 全文保持 | 短縮 | 削除ペア | 推定tokens | 削減率 | 秒 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in result['cases']:
        m = row['metrics']
        lines.append(f"| {row['threshold']} | {m['kept_results']} | {m['truncated_results']} | "
                     f"{m['dropped_pairs']} | {row['measurements']['estimated_request_tokens']} | "
                     f"{row['reduction_ratio']:.1%} | {row['duration_seconds']:.3f} |")
    lines += ['', '保持数は保護対象1件を含みます。推定値は実トークナイザーの値ではありません。',
              '同じ元履歴から毎回独立して実APIを呼び、前回の選別結果は次の入力に使っていません。',
              '閾値はAPI応答後のローカル判定に使用され、APIへの状態・質問は同一です。',
              '各水準1回の観測です。スコアの変動がある場合、閾値以外の影響も含まれます。',
              '通常のユーザー・アシスタントメッセージは削除せず、継続回答の品質は評価していません。', '']
    for row in result['cases']:
        name = f"threshold-{row['threshold']:g}"
        batches = json.loads((directory / name / 'jev-requests.json').read_text())
        scores = {key: value for batch in batches for key, value in batch['response']['scores'].items()}
        lines += [f"## 閾値 {row['threshold']}", '',
                  '| 引数 | call | result | 処理 |', '|---|---:|---:|---|']
        for item in row['interactions']:
            call_id = item['call_id']
            call = scores[f'{call_id}:call'] if f'{call_id}:call' in scores else '保護'
            score = scores[f'{call_id}:result'] if f'{call_id}:result' in scores else '保護'
            lines.append(f"| `{item['arguments']}` | {call} | {score} | {item['action']} |")
        lines += ['', f'- [変換後の入力]({name}/model-input.json)',
                  f'- [実スコア・状態・質問・使用量]({name}/jev-requests.json)', '']
    lines += ['- [元のJSONL](sessions/source.jsonl)', '- [設定・測定結果](results.json)', '']
    (directory / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    """Runs explicit thresholds once each, stopping at the first failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--thresholds', type=float, nargs='+', required=True)
    parser.add_argument('--jev-model', required=True)
    parser.add_argument('--jev-timeout', type=int, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    directory = args.output.resolve()
    subprocess.run(['git', 'check-ignore', '-q', str(directory.relative_to(project))], cwd=project, check=True)
    if len(set(args.thresholds)) != len(args.thresholds):
        raise ValueError('Thresholds must be unique.')
    configs = [JevConfig(args.jev_model, os.environ['TYPESAFE_API_KEY'], args.jev_timeout,
                        replace(DEFAULT_JEV_POLICY, keep_threshold=value)) for value in args.thresholds]
    config = load_config(args.config.resolve(), os.environ)
    directory.mkdir(parents=True, exist_ok=False)
    stage = 'copy_source'
    write_json(directory / 'status.json', {'status': 'running', 'stage': stage})
    try:
        copy_source_run(args.source_run.resolve(), directory)
        source = directory / 'sessions/source.jsonl'
        raw = source.read_bytes()
        events = JsonlContextStore(directory / 'sessions').load('source')
        adapter = build_model_adapter(config.model)
        schemas = [FileReadTool(directory / 'workspace').schema()]
        instructions = RepositoryContext(directory / 'workspace').instructions(AGENT_INSTRUCTIONS, events)
        baseline = build_model_history(events, adapter.replay_policy).items
        write_json(directory / 'baseline-input.json', baseline)
        result: dict[str, Any] = {'source_sha256': sha256(raw).hexdigest(),
            'source_run': str(args.source_run.resolve()), 'model': config.model.model,
            'jev_model': args.jev_model, 'jev_timeout_seconds': args.jev_timeout,
            'base_policy': asdict(DEFAULT_JEV_POLICY), 'baseline': measurements(baseline, schemas, instructions),
            'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip(),
            'runner_sha256': sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': []}
        for jev_config in configs:
            stage = f'threshold-{jev_config.policy.keep_threshold:g}'
            print(stage, flush=True)
            output = directory / stage
            output.mkdir()
            evaluator = RecordingJev(JevClient(jev_config), output)
            row = run_case(events, adapter, jev_config, evaluator, output, schemas, instructions, CURRENT)
            result['cases'].append(row)
            (output / 'source.jsonl').write_bytes(raw)
            checkpoint = json.loads((output / 'checkpoint.json').read_text())
            JsonlContextStore(output).append('source', checkpoint['kind'], checkpoint['payload'])
            write_json(directory / 'results.json', result)
        if source.read_bytes() != raw or events != [json.loads(line) for line in raw.decode().splitlines()]:
            raise RuntimeError('Source history changed during threshold comparison.')
        render_report(directory, result)
        write_json(directory / 'status.json', {'status': 'complete'})
        print(f'complete: {directory / "report.md"}', flush=True)
    except Exception as error:
        write_json(directory / 'status.json', {'status': 'stopped', 'stage': stage,
            'error_type': type(error).__name__, 'error_code': error.error_type if isinstance(error, GearError) else None})
        print(f'STOPPED at {stage}; inspect status.json. No retry or fallback.', flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
