"""Compare real summary and Jev compaction on one isolated coding session."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from typing import Any

from gear_agent.agent.compaction import CompactionService
from gear_agent.agent.compaction_strategy import CompactionCandidate, SummaryCompactionStrategy
from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.history import build_model_history
from gear_agent.agent.jev import JevAnswers, JevClient, JevCompactionStrategy
from gear_agent.agent.loop import AGENT_INSTRUCTIONS, AgentLoop
from gear_agent.compaction_config import DEFAULT_JEV_POLICY, JevConfig
from gear_agent.config import load_config
from gear_agent.context_budget import ByteTokenEstimator, ContextBudgetManager, ContextRequest, DISABLED_CONTEXT_BUDGET
from gear_agent.model.adapter import ModelAdapter
from gear_agent.repository import RepositoryContext
from gear_agent.model.factory import build_model_adapter
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.filesystem import FileReadTool


CURRENT = (
    'Continue the retry-limit investigation. Using the collected evidence, propose the exact '
    'minimal code change and boundary tests for retry.py. Preserve the exponential backoff '
    'and the public API. The completed documentation build investigation is unrelated; '
    'do not revisit it. Do not edit files yet.'
)
TURNS = (
    'Read docs-build.log, docs-release.md, retry.py, and retry-contract.md with file_read. '
    'First explain why the old documentation build failed, then identify the retry boundary '
    'bug for later work. Do not change files. Finish with at most 120 words.',
    'The documentation issue is closed. Read retry-ticket.md with file_read. Confirm the '
    'retry acceptance criteria and what remains to investigate. Do not reread other files '
    'or change files. Finish with at most 80 words.',
)
FIXTURES = {
    'retry.py': '''from collections.abc import Callable
from time import sleep


def send_with_retry(send: Callable[[], str], max_attempts: int) -> str:
    """Send with a bounded number of attempts."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    attempt = 0
    while attempt <= max_attempts:
        attempt += 1
        try:
            return send()
        except TimeoutError:
            if attempt > max_attempts:
                raise
            sleep(0.25 * 2 ** (attempt - 1))
    raise RuntimeError("unreachable")
''',
    'retry-contract.md': '''# Retry contract

The public entry point is send_with_retry(send, max_attempts).
max_attempts counts ALL sends, including the first attempt, not retries alone.
For max_attempts=1, a timeout must propagate after exactly one send and zero sleeps.
For max_attempts=3, three consecutive timeouts cause exactly three sends and two sleeps.
The delay sequence is 0.25, 0.5, 1.0 seconds; never sleep after the final failure.
A successful send returns the original response immediately, with no extra attempts.
Only TimeoutError is retryable. Other exceptions propagate immediately.
Reject max_attempts less than 1 before invoking send.
Keep the current function name, arguments, return type, and delay formula.
Tests must use a deterministic send callable and replace the nondeterministic sleep.
Do not add jitter, environment configuration, retry libraries, or new API parameters.
''',
    'retry-ticket.md': '''# Ticket RETRY-42

Production observed four outbound sends when max_attempts was set to three.
The contract in retry-contract.md is authoritative: the limit includes the first send.
Fix only the attempt boundary; preserve the exponential backoff and public API.
Regression cases: immediate success; one allowed attempt that times out; success on
the third attempt; three timeouts; non-timeout failure; invalid max_attempts.
Documentation build work was completed independently and is out of scope.
''',
    'docs-release.md': '''# Historical docs release

Release docs-2026.04 failed because the navigation referenced an old guide path.
The source page moved from guide/retry.html to reference/retry.html.
Resolution: update the navigation link and rebuild the static documentation.
This changes documentation navigation only, not retry behavior or Python code.
The release was subsequently completed; no outstanding documentation task remains.
''',
    'docs-build.log': '\n'.join(
        f'2026-04-10T09:00:{i:02d}Z INFO docs-builder rendered reference/page-{i:02d}.html'
        for i in range(45)
    ) + '\nERROR navigation: missing target guide/retry.html\n'
        'INFO actual page: reference/retry.html\nERROR build failed: 1 invalid link\n',
}


def write_json(path: Path, value: object) -> None:
    """Writes a readable artifact, rejecting non-finite measurements."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


class ObservationLog:
    """Persists observations and stops immediately on a failed tool execution."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, kind: str, payload: dict[str, Any]) -> None:
        """Records an observation before rejecting any recoverable tool failure."""
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'kind': kind, 'payload': payload}, ensure_ascii=False) + '\n')
        if kind == 'tool_finished' and not payload['successful']:
            raise RuntimeError('Experiment stopped at tool_finished; inspect observations.jsonl.')


class RecordingJev:
    """Records the actual external selection evidence for this synthetic workspace."""

    def __init__(self, client: JevClient, directory: Path) -> None:
        self.client = client
        self.directory = directory
        self.batches: list[dict[str, Any]] = []

    def evaluate(self, state: str, questions: dict[str, str]) -> JevAnswers:
        """Saves a request before dispatch and its scores only after success."""
        batch: dict[str, Any] = {'state': state, 'questions': questions}
        self.batches.append(batch)
        write_json(self.directory / 'jev-requests.json', self.batches)
        result = self.client.evaluate(state, questions)
        batch['response'] = asdict(result)
        write_json(self.directory / 'jev-requests.json', self.batches)
        return result


def checkpoint_input(adapter: ModelAdapter, candidate: CompactionCandidate, current: str) -> list[dict[str, Any]]:
    """Rebuilds the next input with the same current-user semantics as AgentLoop.

    Args:
        adapter: Main model adapter and replay policy.
        candidate: Completed compaction checkpoint.
        current: Pending current user message, already present in selective events.

    Returns:
        Model-visible replay with the current user message included exactly once.
    """
    checkpoint = {'kind': candidate.kind, 'payload': candidate.payload}
    if candidate.kind == 'compaction_summary':
        return adapter.prepare_history([checkpoint], current).items
    if candidate.kind != 'compaction_selective':
        raise ValueError(f'Unknown checkpoint kind: {candidate.kind}')
    return build_model_history([checkpoint], adapter.replay_policy).items


def interaction_changes(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compares model-visible tool results by call ID, including verbatim equality.

    Args:
        before: Uncompacted replay items.
        after: Compacted replay items.

    Returns:
        Per-call disposition and output UTF-8 byte counts.
    """
    old = {item['call_id']: item['output'] for item in before if item.get('type') == 'function_call_output'}
    new = {item['call_id']: item['output'] for item in after if item.get('type') == 'function_call_output'}
    rows = []
    for item in before:
        if item.get('type') != 'function_call':
            continue
        call_id = item['call_id']
        action = 'removed' if call_id not in new else ('kept' if old[call_id] == new[call_id] else 'changed')
        rows.append({'call_id': call_id, 'name': item['name'], 'arguments': item['arguments'],
                     'action': action, 'before_bytes': len(old[call_id].encode('utf-8')),
                     'after_bytes': len(new[call_id].encode('utf-8')) if call_id in new else 0})
    return rows


def measurements(items: list[dict[str, Any]], tools: list[dict[str, object]], instructions: str) -> dict[str, int]:
    """Measures comparable effective inputs without claiming an actual tokenizer count."""
    request = ContextRequest(items, tools, instructions, AGENT_INSTRUCTIONS, len(items) - 1)
    diagnostic = ContextBudgetManager(DISABLED_CONTEXT_BUDGET, ByteTokenEstimator()).evaluate(request)
    return {'input_items': len(items),
            'input_json_bytes': len(json.dumps(items, ensure_ascii=False, separators=(',', ':')).encode('utf-8')),
            'estimated_request_tokens': diagnostic.total_estimated_request_tokens}


def render_report(directory: Path, results: dict[str, Any]) -> None:
    """Writes a local comparison with links to the actual evidence and output text."""
    lines = ['# PR #26: 実セッションのコンパクション比較', '',
             'コード・課題は架空ですが、会話・ツール呼び出し・読み取り結果は実際のAgentLoopで生成しました。',
             '同一のJSONLに現在の依頼を追加し、従来のsummaryとJevのstrategyをそれぞれ1回実行しています。',
             '自動発火の閾値到達・継続回答の品質・反復試行のベンチマークは今回の対象外です。', '',
             f"Main model: `{results['model']}` / Jev: `{results['jev_model']}`", '',
             '| 指標 | 元の履歴 | 従来の要約 | Jev |', '|---|---:|---:|---:|']
    for label, key in [('入力項目数', 'input_items'), ('入力JSON UTF-8 bytes', 'input_json_bytes'),
                       ('リクエスト推定トークン数', 'estimated_request_tokens')]:
        lines.append(f"| {label} | {results['baseline'][key]} | {results['summary'][key]} | {results['jev'][key]} |")
    lines += ['', '推定値はGearのByteTokenEstimator（JSONバイト数×1.25等）です。実トークン数ではありません。',
              'instructions・tool schemaは両方式で同一です。API使用量は各observations.jsonlとjev-requests.jsonにあります。',
              '', '| 方法 | 処理秒数 | 推定リクエスト削減率 |', '|---|---:|---:|']
    for name in ('summary', 'jev'):
        row = results[name]
        lines.append(f"| {name} | {row['duration_seconds']:.3f} | {row['reduction_ratio']:.1%} |")
    lines += ['', '## Jevの選別', '',
              'kept = 元の出力を保持、changed = 出力を変更（短縮）、removed = 呼び出しと結果を除外。',
              '従来の要約ではツール対が直接リプレイされず、情報を自然言語の要約へ置き換えます。', '',
              '| file_read引数 | Jev結果 | 元の出力bytes | 選別後bytes |', '|---|---|---:|---:|']
    for row in results['jev']['interactions']:
        lines.append(f"| `{row['arguments']}` | {row['action']} | {row['before_bytes']} | {row['after_bytes']} |")
    lines += ['', '## 従来の要約全文', '', (directory / 'summary' / 'summary.txt').read_text(encoding='utf-8'),
              '', '## 証跡', '', '- [元のセッションJSONL](sessions/source.jsonl)',
              '- [元のモデル入力](baseline-input.json)', '- [従来の要約後のモデル入力](summary/model-input.json)',
              '- [Jev選別後のモデル入力](jev/model-input.json)', '- [Jevに渡した状態・質問・実スコア](jev/jev-requests.json)',
              '- [数値・設定・入力ハッシュ](results.json)', '',
              '生JSONLは削除しません。チェックポイントを追加するため、ディスク使用量の削減とは異なります。',
              'Jevは結果本文を受け取らず、会話・引数・結果のサイズ等から判断します。',
              '保持設定による保護とJevスコアによる保持は区別してください。直近1ターンは既定で保護されます。',
              '各方式1回の観察結果であり、一般的な品質・速度・コストの優劣を示すものではありません。', '']
    (directory / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def copy_source_run(source: Path, target: Path) -> None:
    """Copies an existing scenario without generating different model responses.

    Args:
        source: Prior run with the completed scenario and pending current message.
        target: New output directory; previous results are never copied.
    """
    raw = (source / 'sessions/source.jsonl').read_bytes()
    events = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    if not events or events[-1]['kind'] != 'user_input' or events[-1]['payload']['text'] != CURRENT:
        raise ValueError('Source must end with the scenario pending user message.')
    shutil.copytree(source / 'workspace', target / 'workspace')
    (target / 'sessions').mkdir()
    (target / 'sessions/source.jsonl').write_bytes(raw)
    shutil.copyfile(source / 'scenario.json', target / 'scenario.json')


def run(config_path: Path, directory: Path, jev_model: str, jev_timeout: int, source_run: Path | None) -> None:
    """Generates live evidence and runs both production compaction strategies once.

    Args:
        config_path: Existing Gear config for the main model endpoint.
        directory: New, Git-ignored output directory inside this project.
        jev_model: Explicit pinned Jev model version.
        jev_timeout: Explicit Jev request timeout in seconds.
        source_run: Prior run to reuse, or None when explicitly generating a new session.
    """
    project = Path(__file__).resolve().parents[1]
    directory = directory.resolve()
    relative = directory.relative_to(project)
    subprocess.run(['git', 'check-ignore', '-q', str(relative)], cwd=project, check=True)
    config = load_config(config_path.resolve(), os.environ)
    jev_config = JevConfig(jev_model, os.environ['TYPESAFE_API_KEY'], jev_timeout, DEFAULT_JEV_POLICY)
    directory.mkdir(parents=True, exist_ok=False)
    stage = 'generate_session'
    try:
        workspace = directory / 'workspace'
        if source_run is None:
            workspace.mkdir()
            for name, content in FIXTURES.items():
                (workspace / name).write_text(content, encoding='utf-8')
            write_json(directory / 'scenario.json', {'turns': TURNS, 'current': CURRENT})
        else:
            stage = 'copy_source_session'
            copy_source_run(source_run.resolve(), directory)
        adapter = build_model_adapter(config.model)
        tool = FileReadTool(workspace)
        store = JsonlContextStore(directory / 'sessions')
        observer = ObservationLog(directory / 'generation-observations.jsonl')
        loop = AgentLoop(adapter, [tool], store, SilentAgentLoopEventSink(), RepositoryContext(workspace),
                         observer=observer)
        if source_run is None:
            for index, prompt in enumerate(TURNS, 1):
                print(f'generate_session: turn {index}/{len(TURNS)}', flush=True)
                loop.run_turn('source', prompt, config.runtime.max_iterations,
                              config.runtime.model_timeout_seconds, config.runtime.model_stream_idle_timeout_seconds)
        calls = [event['payload']['arguments']['path'] for event in store.load('source') if event['kind'] == 'tool_call']
        if set(calls) != set(FIXTURES):
            raise RuntimeError(f'Generated session did not read the required fixture files: {calls}')
        if source_run is None:
            store.append('source', 'user_input', {'text': CURRENT})
        source = directory / 'sessions' / 'source.jsonl'
        original_bytes = source.read_bytes()
        events = store.load('source')
        before = build_model_history(events, adapter.replay_policy).items
        instructions = RepositoryContext(workspace).instructions(AGENT_INSTRUCTIONS, events)
        write_json(directory / 'baseline-input.json', before)
        results: dict[str, Any] = {
            'source_sha256': sha256(original_bytes).hexdigest(),
            'source_run': str(source_run.resolve()) if source_run is not None else None,
            'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip(),
            'model': config.model.model, 'jev_model': jev_model, 'policy': asdict(DEFAULT_JEV_POLICY),
            'jev_timeout_seconds': jev_timeout, 'main_timeout_seconds': config.runtime.model_timeout_seconds,
            'baseline': measurements(before, [tool.schema()], instructions),
            'scope': 'strategy comparison at the start of a user turn; no automatic trigger or continuation quality test',
        }
        write_json(directory / 'results.json', results)
        for name in ('summary', 'jev'):
            stage = name
            print(f'compact: {name}', flush=True)
            output = directory / name
            output.mkdir()
            service = CompactionService(adapter, ObservationLog(output / 'observations.jsonl'))
            if name == 'summary':
                write_json(output / 'request.json', asdict(service.prepare_request(events)))
                strategy = SummaryCompactionStrategy(service, ContextBudgetManager(DISABLED_CONTEXT_BUDGET, ByteTokenEstimator()))
            else:
                strategy = JevCompactionStrategy(jev_config, RecordingJev(JevClient(jev_config), output))
            started = perf_counter()
            candidate = strategy.compact(events, config.runtime.model_timeout_seconds,
                                         config.runtime.model_stream_idle_timeout_seconds)
            duration = perf_counter() - started
            items = checkpoint_input(adapter, candidate, CURRENT)
            write_json(output / 'checkpoint.json', asdict(candidate))
            write_json(output / 'model-input.json', items)
            if name == 'summary':
                (output / 'summary.txt').write_text(candidate.payload['text'], encoding='utf-8')
            (output / 'source.jsonl').write_bytes(original_bytes)
            JsonlContextStore(output).append('source', candidate.kind, candidate.payload)
            measured = measurements(items, [tool.schema()], instructions)
            results[name] = {**measured, 'duration_seconds': duration, 'strategy_metrics': candidate.metrics,
                             'reduction_ratio': 1 - measured['estimated_request_tokens'] / results['baseline']['estimated_request_tokens'],
                             'interactions': interaction_changes(before, items)}
            write_json(directory / 'results.json', results)
        if source.read_bytes() != original_bytes:
            raise RuntimeError('Source JSONL changed during comparison.')
        render_report(directory, results)
        write_json(directory / 'status.json', {'status': 'complete'})
        print(f'complete: {directory / "report.md"}', flush=True)
    except Exception as error:
        write_json(directory / 'status.json', {'status': 'stopped', 'stage': stage,
                   'error_type': type(error).__name__, 'origin': getattr(error, 'origin', None),
                   'error_code': getattr(error, 'error_type', None)})
        print(f'STOPPED at {stage}: {type(error).__name__}; inspect status.json. No retry or fallback.', flush=True)
        raise SystemExit(1) from None


def main() -> None:
    """Runs an explicitly configured comparison without overwriting earlier artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jev-model', required=True)
    parser.add_argument('--jev-timeout', type=int, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--generate', action='store_true', help='Generate a new live session.')
    mode.add_argument('--source-run', type=Path, help='Reuse the exact history from a prior run.')
    args = parser.parse_args()
    run(args.config, args.output, args.jev_model, args.jev_timeout, args.source_run)


if __name__ == '__main__':
    main()
