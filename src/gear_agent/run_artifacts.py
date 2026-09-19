from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4
import json
import os
import shutil

from gear_agent.artifact_privacy import ArtifactPrivacy
from gear_agent.config import DEFAULT_DOCKER_IMAGE
from gear_agent.errors import GearError
from gear_agent.headless import RunResult, TaskPrompt, describe_run, run_task
from gear_agent.observation import error_identity
from gear_agent.run_collector import EvalRunCollector
from gear_agent.runtime import AgentRuntime
from gear_agent.workspace_snapshot import git_command, require_git, snapshot_workspace


class RunArtifacts:
    """Owns one exclusively reserved directory with a terminal manifest commit marker."""

    def __init__(
        self, agent: AgentRuntime, session_id: str, prompt: TaskPrompt,
        destination: Path | None, collector: EvalRunCollector, environment: Mapping[str, str],
    ) -> None:
        """Initializes artifacts before execution without mutating workspace Git state.

        Args:
            agent: Effective production harness.
            session_id: Canonical session identifier.
            prompt: Exact task and source.
            destination: Exact run directory; None selects workspace/.gear/runs/UUID.
            collector: Run observer already attached to execution services.
            environment: CLI environment used only to identify secrets for redaction.

        Raises:
            GearError: If destination exists, metadata is unsafe, or writing fails.
        """
        self.agent = agent
        self.session_id = session_id
        self.collector = collector
        self.privacy = ArtifactPrivacy(agent.config, environment)
        self.run_id = str(uuid4())
        self.path = destination if destination is not None else agent.workspace / '.gear/runs' / self.run_id
        self.baseline = snapshot_workspace(agent.workspace)
        self.manifest: dict[str, Any] = {}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise _write_error(exc) from exc
        try:
            self.path.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise GearError('artifact_destination_exists', 'Run artifact destination already exists.',
                            'run_artifacts', True, {'path': str(self.path)}) from exc
        except OSError as exc:
            raise _write_error(exc) from exc
        collector.start()
        spec = asdict(describe_run(agent, session_id, prompt))
        public_endpoint = self.privacy.endpoint(agent.config.model.url)
        spec['model']['endpoint_identity'] = sha256(public_endpoint.encode('utf-8')).hexdigest()
        spec['model']['endpoint'] = public_endpoint
        scope = agent.adapter.replay_policy.current_scope
        spec['model']['protocol'] = scope.protocol
        spec['model']['reasoning_replay_scope'] = {
            'protocol': spec['model']['protocol'], 'model': agent.config.model.model,
            'endpoint_identity': scope.endpoint_identity if public_endpoint == agent.config.model.url else None,
            'endpoint_identity_omitted': public_endpoint != agent.config.model.url,
        }
        spec.pop('prompt_source')
        spec['runtime']['shell_docker_image'] = DEFAULT_DOCKER_IMAGE if spec['tools']['shell_tool'] else None
        self.manifest = {
            **spec, 'schema_version': 1, 'run_id': self.run_id,
            'started_at': _utc_now(), 'finished_at': None, 'status': 'running', 'error': None,
            'gear': _gear_identity(),
            'task': {'text': self.privacy.exact_text(prompt.text, 'task.text'),
                     'source': 'inline' if prompt.source == 'inline' else 'file',
                     'path': None if prompt.source == 'inline' else prompt.source},
            'enabled_tools': sorted('shell' if name == 'shell_tool' else name
                                    for name, enabled in spec['tools'].items() if enabled),
            'workspace_git': self.baseline.metadata(), 'repository_instructions': [],
        }
        self._json('run.json', self.manifest)

    def finish(self, result: RunResult | None, error: BaseException | None) -> None:
        """Writes a complete snapshot before publishing the terminal manifest.

        Args:
            result: Successful canonical result, absent for failure.
            error: Original execution failure, absent for success.

        Raises:
            GearError: Distinct artifact failure; a success manifest is never written.
        """
        try:
            self._finish(result, error)
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            raise _write_error(exc) from exc

    def _finish(self, result: RunResult | None, error: BaseException | None) -> None:
        status = 'success' if error is None else 'failure'
        diagnostic = error_identity(error) if error is not None else None
        events = self.agent.store.load(self.session_id)
        observations = [{**event, 'run_id': self.run_id, 'session_id': self.session_id}
                        for event in self.collector.events]
        snapshot = ''.join(_json_text(self.privacy.serialize(event)) + '\n' for event in events + observations)
        self._text('events.jsonl', snapshot)
        ending_status = None
        if self.baseline.available:
            ending_status = require_git(self.agent.workspace, ['status', '--porcelain=v1', '--untracked-files=all'])
            self._text('workspace-status.txt', self.privacy.text(
                'Starting git status --porcelain=v1:\n' + str(self.baseline.status)
                + '\nEnding git status --porcelain=v1:\n' + ending_status))
            if self.baseline.head is not None:
                diff = require_git(self.agent.workspace, [
                    'diff', '--no-ext-diff', '--no-textconv', self.baseline.head, '--', '.',
                ])
                self._text('workspace.diff', self.privacy.text(diff))
        else:
            self._text('workspace-status.txt', f'Git unavailable: {self.baseline.reason}\n')
        self.manifest['workspace_git']['ending_status'] = ending_status
        self.manifest['repository_instructions'] = [event['payload'] for event in observations
                                                    if event['kind'] == 'repository_instructions']
        metrics = self.collector.metrics(status, diagnostic, result is not None, events, self.privacy)
        self._json('metrics.json', metrics)
        if result is not None:
            self._text('final.txt', self.privacy.exact_text(result.turn.final_text, 'final.txt'))
        self.manifest.update({'status': status, 'error': diagnostic, 'finished_at': _utc_now()})
        self._json('run.json', self.manifest)

    def _json(self, name: str, value: dict[str, Any]) -> None:
        self._text(name, _json_text(self.privacy.serialize(value)) + '\n')

    def _text(self, name: str, value: str) -> None:
        temporary = self.path / f'.{name}.{uuid4()}.tmp'
        try:
            with temporary.open('x', encoding='utf-8', newline='') as output:
                output.write(value)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(self.path / name)
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            raise _write_error(exc) from exc


def run_with_artifacts(
    agent: AgentRuntime, session_id: str, prompt: TaskPrompt, destination: Path | None,
    collector: EvalRunCollector, environment: Mapping[str, str],
) -> RunResult:
    """Runs one task and records both successful and failed terminal outcomes.

    Args:
        agent: Production harness composed with the collector.
        session_id: Canonical session identifier.
        prompt: Task text and source.
        destination: Exact run directory, or documented workspace default.
        collector: Attached observation and event sink.
        environment: Effective CLI environment for secret redaction.

    Returns:
        Original successful task result after artifact completion.

    Raises:
        BaseException: Original execution failure, after recording available data.
        GearError: Distinct artifact error if initialization or completion fails.
    """
    artifacts = RunArtifacts(agent, session_id, prompt, destination, collector, environment)
    try:
        result = run_task(agent, session_id, prompt)
    except BaseException as error:
        artifacts.finish(None, error)
        raise
    artifacts.finish(result, None)
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _write_error(error: BaseException) -> GearError:
    return GearError('artifact_write_failed', f'Run artifact writing failed ({type(error).__name__}).',
                     'run_artifacts', True, {})


def _gear_identity() -> dict[str, str | None]:
    try:
        installed_version = version('gear-agent')
    except PackageNotFoundError:
        installed_version = None
    source_root = Path(__file__).resolve().parents[2]
    commit = None
    # Installed wheels do not identify the user's workspace as Gear source.
    if (shutil.which('git') is not None and (source_root / '.git').exists()
            and (source_root / 'pyproject.toml').is_file()):
        result = git_command(source_root, ['rev-parse', '--verify', 'HEAD'])
        if result.returncode != 0:
            raise GearError('artifact_git_failed', 'Cannot determine Gear source revision.',
                            'run_artifacts', True, {})
        commit = result.stdout.strip()
    return {'version': installed_version, 'commit': commit}
