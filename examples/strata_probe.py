"""Check a live Strata file_read round trip and JSONL resume with Gear's headless runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import secrets
import sys

from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.config import AppConfig, load_config
from gear_agent.errors import GearError
from gear_agent.headless import TaskPrompt, describe_run, run_task, validate_headless_runtime
from gear_agent.run_collector import EvalRunCollector
from gear_agent.runtime import build_agent_runtime
from gear_agent.workspace_snapshot import require_git


PROJECT = Path(__file__).resolve().parents[1]
READ = TaskPrompt('Call file_read exactly once on marker.txt. Reply with just the file contents.', 'inline')
RECALL = TaskPrompt('Repeat the marker from the previous turn. Use no tools. Reply with just the marker.', 'inline')


def run_probe(config: AppConfig, output: Path, deployment: str) -> int:
    """Exercises the same endpoint twice with a newly opened store for the second turn.

    Args:
        config: Gear settings with only file_read enabled.
        output: New directory under the ignored test_outputs tree.
        deployment: Operator-supplied route label for the report.

    Returns:
        Zero for a verified round trip and resume, one for a failed turn or assertion.
    """
    output = output.resolve()
    if not output.is_relative_to(PROJECT / 'test_outputs') or output == PROJECT / 'test_outputs':
        raise ValueError('Choose a fresh output directory inside test_outputs/.')
    require_git(PROJECT, ['check-ignore', '-q', '--', str(output)])
    if {name for name, enabled in asdict(config.tool).items() if enabled} != {'file_read'}:
        raise ValueError('The probe requires only file_read to be enabled.')
    if config.model.api_key is None or config.compaction.jev is not None:
        raise ValueError('The probe requires model Bearer authentication and no external Jev service.')

    revision = require_git(PROJECT, ['rev-parse', 'HEAD']).strip()
    dirty = bool(require_git(PROJECT, ['status', '--porcelain']))
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    runtime = replace(config.runtime, workdir=output / 'workspace', session_dir=output / 'sessions')
    runtime.workdir.mkdir()
    marker = secrets.token_hex(16)
    (runtime.workdir / 'marker.txt').write_text(marker, encoding='utf-8')
    observations = EvalRunCollector()
    agent = build_agent_runtime(config, runtime, observations, observations)
    validate_headless_runtime(agent)
    privacy = ArtifactPrivacy(config, os.environ)
    report = {'deployment': deployment, 'gear_commit': revision, 'gear_dirty': dirty,
              'settings': privacy.serialize(asdict(describe_run(agent, 'probe', READ)))}
    stage = 'tool_round_trip'
    error = None
    observations.start()
    try:
        turn = run_task(agent, 'probe', READ).turn
        history = agent.store.load('probe')
        results = [e['payload'] for e in history if e['kind'] == 'tool_result']
        if (len(results) != 1 or results[0]['name'] != 'file_read'
                or results[0]['result'].get('path') != 'marker.txt'
                or results[0]['result'].get('content') != marker
                or turn.iterations != 2 or turn.final_text.strip() != marker):
            raise GearError('tool_round_trip_unverified', 'Expected one successful file_read and its marker.',
                            'strata_probe', False, {})

        stage = 'resume'
        agent = build_agent_runtime(config, runtime, observations, observations)
        resumed = run_task(agent, 'probe', RECALL).turn
        after = agent.store.load('probe')
        appended = after[len(history):]
        if (after[:len(history)] != history or resumed.iterations != 1 or resumed.final_text.strip() != marker
                or [e['kind'] for e in appended] != ['user_input', 'model_response', 'assistant_message']):
            raise GearError('resume_unverified', 'Expected one appended turn recalling the marker without tools.',
                            'strata_probe', False, {})
    except GearError as exc:
        error = {'type': exc.error_type, 'origin': exc.origin}
    status = 'passed' if error is None else 'failed'
    events = agent.store.load('probe')
    final_text_produced = bool(events and events[-1]['kind'] == 'assistant_message')
    metrics = observations.metrics(status, error, final_text_produced, events, privacy)
    report.update(status=status, stage=stage, error=error, metrics=metrics)
    (output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'{status} ({stage}): {output / "result.json"}')
    return int(error is not None)


def main(argv: list[str]) -> int:
    """Runs the opt-in CLI with explicit configuration and a fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True, help='Allow real model requests.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deployment', choices=('local', 'docker', 'ssh', 'ssh-docker'), required=True)
    args = parser.parse_args(argv)
    return run_probe(load_config(args.config, os.environ), args.output, args.deployment)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
