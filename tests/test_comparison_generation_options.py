from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch
import json

import pytest

from examples.compaction_comparison import CURRENT, FIXTURES, TURNS, run
from examples.jev_threshold_sweep import main as sweep_main
from gear_agent.agent.jev import JevClient
from gear_agent.agent.loop import AGENT_INSTRUCTIONS
from gear_agent.config import DEFAULT_CONFIG_TEXT, load_config
from gear_agent.context_budget import (
    ByteTokenEstimator, ContextBudgetManager, ContextRequest, DISABLED_CONTEXT_BUDGET,
)
from gear_agent.model.factory import build_model_adapter
from gear_agent.model.transport import HttpxHttpTransport
from gear_agent.repository import RepositoryContext
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.filesystem import FileReadTool
from tests.test_context_budget import response
from tests.test_jev_compaction import JevAPI


def saved_source(root: Path) -> Path:
    source = root / 'source'
    workspace = source / 'workspace'
    workspace.mkdir(parents=True)
    for name, content in FIXTURES.items():
        (workspace / name).write_text(content, encoding='utf-8')
    (source / 'scenario.json').write_text(json.dumps({'current': CURRENT}), encoding='utf-8')
    store = JsonlContextStore(source / 'sessions')
    for prompt, names in zip(TURNS, [
        ['docs-build.log', 'docs-release.md', 'retry.py', 'retry-contract.md'], ['retry-ticket.md'],
    ], strict=True):
        store.append('source', 'user_input', {'text': prompt})
        calls = [{'type': 'function_call', 'call_id': name, 'name': 'file_read',
                  'arguments': json.dumps({'path': name})} for name in names]
        store.append('source', 'model_response', {'output': calls})
        for name in names:
            store.append('source', 'tool_call', {'call_id': name, 'name': 'file_read', 'arguments': {'path': name}})
            store.append('source', 'tool_result', {'call_id': name, 'name': 'file_read',
                                                  'result': FileReadTool(workspace).run({'path': name})})
        store.append('source', 'assistant_message', {'text': 'Investigation complete.'})
    store.append('source', 'user_input', {'text': CURRENT})
    return source


@pytest.mark.parametrize('runner', ['comparison', 'sweep'])
@pytest.mark.parametrize('settings', ['', 'reasoning_effort = "medium"\nmax_output_tokens = 8192'])
def test_reports_measure_before_and_after_with_effective_generation_options(
    tmp_path: Path, runner: str, settings: str,
) -> None:
    path = tmp_path / 'config.toml'
    path.write_text(DEFAULT_CONFIG_TEXT.replace('[model]', f'[model]\n{settings}'), encoding='utf-8')
    source = saved_source(tmp_path)
    original = (source / 'sessions/source.jsonl').read_bytes()
    adapter = build_model_adapter(load_config(path, {}).model)
    api = JevAPI({'docs-build.log': (0.25, 0.25), 'docs-release.md': (0.25, 0.25),
                  'retry.py': (0.7, 0.3), 'retry-contract.md': (0.7, 0.2)})
    output_root = Path(__file__).resolve().parents[1] / 'test_outputs'
    output_root.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output_root) as directory:
        output = Path(directory) / runner
        with patch.dict('os.environ', {'TYPESAFE_API_KEY': 'private-jev-key'}), \
                patch.object(HttpxHttpTransport, 'post_json', return_value=response('Investigation summary.')), \
                patch.object(JevClient, 'evaluate', side_effect=api.evaluate):
            if runner == 'comparison':
                run(path, output, 'jev-1.13.0', 10, source)
            else:
                with patch('sys.argv', ['jev_threshold_sweep', '--config', str(path),
                                       '--source-run', str(source), '--output', str(output),
                                       '--thresholds', '0.3', '0.7', '--jev-model', 'jev-1.13.0',
                                       '--jev-timeout', '10']):
                    sweep_main()
        results = json.loads((output / 'results.json').read_text())
        events = JsonlContextStore(output / 'sessions').load('source')
        instructions = RepositoryContext(output / 'workspace').instructions(AGENT_INSTRUCTIONS, events)
        schemas = [FileReadTool(output / 'workspace').schema()]
        budget = ContextBudgetManager(DISABLED_CONTEXT_BUDGET, ByteTokenEstimator(), adapter.generation_options)

        def expected_tokens(input_path: Path) -> int:
            items: list[dict[str, Any]] = json.loads(input_path.read_text())
            request = ContextRequest(items, schemas, instructions, instructions, len(items) - 1)
            return budget.evaluate(request).total_estimated_request_tokens

        baseline = expected_tokens(output / 'baseline-input.json')
        assert results['baseline']['estimated_request_tokens'] == baseline
        if runner == 'comparison':
            rows = [(name, results[name], results[name]) for name in ('summary', 'jev')]
        else:
            rows = [(f'threshold-{row["threshold"]:g}', row['measurements'], row) for row in results['cases']]
        for name, measured, row in rows:
            after = expected_tokens(output / name / 'model-input.json')
            assert measured['estimated_request_tokens'] == after
            assert row['reduction_ratio'] == 1 - after / baseline
        assert json.loads((output / 'status.json').read_text())['status'] == 'complete'
    assert (source / 'sessions/source.jsonl').read_bytes() == original
