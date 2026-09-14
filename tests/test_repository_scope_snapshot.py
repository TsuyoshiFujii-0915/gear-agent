from pathlib import Path
from typing import Any
import json
import subprocess

import pytest

from gear_agent.agent.events import SilentAgentLoopEventSink
from gear_agent.agent.loop import AgentLoop
from gear_agent.config import ReasoningReplayMode
from gear_agent.errors import GearError
from gear_agent.model.client import ModelClient
from gear_agent.model.responses_adapter import ResponsesModelAdapter
from gear_agent.repository import RepositoryContext
from gear_agent.store.jsonl import JsonlContextStore
from gear_agent.tools.base import Tool
from gear_agent.tools.filesystem import FileReadTool, FileWriteTool
from gear_agent.tools.filesystem_search import GlobTool, GrepTool
from gear_agent.tools.patch import ApplyPatchTool
from gear_agent.tools.runtimes import ShellRuntime
from gear_agent.tools.shell import ShellTool
from tests.test_repository import tool_events, write_instruction
from tests.test_repository_integration import (
    InstructionTransport, final_response, model_config, no_change,
)


class LocalTestShellRuntime(ShellRuntime):
    """Executes real shell commands in temporary test workspaces."""

    def run(self, command: str, workdir: Path, timeout_seconds: int) -> dict[str, object]:
        completed = subprocess.run(
            ['sh', '-c', command], cwd=workdir, timeout=timeout_seconds,
            text=True, capture_output=True, check=False,
        )
        return {
            'exit_code': completed.returncode, 'stdout': completed.stdout,
            'stderr': completed.stderr, 'timed_out': False,
        }


def alias_workspace(root: Path) -> None:
    for directory in ('backend', 'frontend'):
        write_instruction(root, directory, directory.upper() + '_RULES')
        (root / directory / 'service.py').write_text('old\n', encoding='utf-8')
    (root / 'alias').symlink_to('backend', target_is_directory=True)


def tool_activity(root: Path, name: str) -> tuple[Tool, dict[str, object]]:
    if name == 'file_read':
        return FileReadTool(root), {'path': 'alias/service.py'}
    if name == 'file_write':
        return FileWriteTool(root), {'path': 'alias/service.py', 'content': 'new\n'}
    if name == 'glob':
        return GlobTool(root), {'pattern': 'alias', 'max_results': 10}
    if name == 'grep':
        return GrepTool(root), {'path': 'alias/service.py', 'pattern': 'old', 'max_results': 10}
    if name == 'apply_patch':
        return ApplyPatchTool(root), {
            'patch': '--- alias/service.py\n+++ alias/service.py\n@@ -1 +1 @@\n-old\n+new\n',
        }
    if name == 'shell':
        return ShellTool(root, LocalTestShellRuntime()), {
            'workdir': 'alias', 'command': 'cat service.py', 'timeout_seconds': 10,
        }
    raise ValueError(name)


@pytest.mark.parametrize('name', ['file_read', 'file_write', 'glob', 'grep', 'apply_patch', 'shell'])
@pytest.mark.parametrize('change', ['retarget', 'delete'])
def test_completed_activity_keeps_physical_scope_after_alias_change(tmp_path: Path, name: str, change: str) -> None:
    alias_workspace(tmp_path)
    tool, arguments = tool_activity(tmp_path, name)
    result = tool.run(arguments)
    (tmp_path / 'alias').unlink()
    if change == 'retarget':
        (tmp_path / 'alias').symlink_to('frontend', target_is_directory=True)
    events = tool_events(name, arguments, result)
    instructions = RepositoryContext(tmp_path).instructions('base', events)
    assert 'BACKEND_RULES' in instructions
    assert 'FRONTEND_RULES' not in instructions
    assert result['resolved_scope_paths'] == ['backend']
    assert str(tmp_path) not in json.dumps(result)


def test_shell_records_workdir_before_command_retargets_alias(tmp_path: Path) -> None:
    alias_workspace(tmp_path)
    tool = ShellTool(tmp_path, LocalTestShellRuntime())
    arguments = {
        'workdir': 'alias', 'timeout_seconds': 10,
        'command': 'rm ../alias && ln -s frontend ../alias',
    }
    result = tool.run(arguments)
    assert result['exit_code'] == 0
    assert (tmp_path / 'alias').resolve() == tmp_path / 'frontend'
    instructions = RepositoryContext(tmp_path).instructions('base', tool_events('shell', arguments, result))
    assert 'BACKEND_RULES' in instructions
    assert 'FRONTEND_RULES' not in instructions


def function_call(call_id: str, name: str, arguments: dict[str, object]) -> dict[str, Any]:
    return {'type': 'function_call', 'call_id': call_id, 'name': name, 'arguments': json.dumps(arguments)}


@pytest.mark.parametrize('compacted', [False, True])
@pytest.mark.parametrize('change', ['retarget', 'delete'])
def test_request_and_resume_retain_scope_after_later_tool_in_same_iteration(tmp_path: Path, compacted: bool, change: str) -> None:
    alias_workspace(tmp_path)
    config = model_config(False, ReasoningReplayMode.NONE)
    command = 'rm alias'
    if change == 'retarget':
        command += ' && ln -s frontend alias'
    response = {'output': [
        function_call('read_alias', 'file_read', {'path': 'alias/service.py'}),
        function_call('change_alias', 'shell', {'workdir': '.', 'command': command, 'timeout_seconds': 10}),
    ]}
    transport = InstructionTransport([response, final_response()], no_change)
    store = JsonlContextStore(tmp_path / 'sessions')
    tools = [FileReadTool(tmp_path), ShellTool(tmp_path, LocalTestShellRuntime())]
    loop = AgentLoop(
        ResponsesModelAdapter(ModelClient(transport), config), tools, store,
        SilentAgentLoopEventSink(), RepositoryContext(tmp_path),
    )
    assert loop.run_turn('session', 'inspect code', 3, 30).final_text == 'done'
    assert 'BACKEND_RULES' in transport.payloads[1]['instructions']
    assert 'FRONTEND_RULES' not in transport.payloads[1]['instructions']
    saved_results = [event['payload']['result'] for event in store.load('session') if event['kind'] == 'tool_result']
    assert saved_results[0]['resolved_scope_paths'] == ['backend']
    if compacted:
        store.append('session', 'compaction_summary', {'text': 'Read the backend code.'})
    write_instruction(tmp_path, 'backend', 'BACKEND_CURRENT_RULES')
    resumed = InstructionTransport([final_response()], no_change)
    AgentLoop(
        ResponsesModelAdapter(ModelClient(resumed), config), tools,
        JsonlContextStore(tmp_path / 'sessions'), SilentAgentLoopEventSink(), RepositoryContext(tmp_path),
    ).run_turn('session', 'continue', 1, 30)
    assert 'BACKEND_CURRENT_RULES' in resumed.payloads[0]['instructions']
    assert 'FRONTEND_RULES' not in resumed.payloads[0]['instructions']


@pytest.mark.parametrize('scopes', [None, 'backend', [17], ['../outside'], ['/tmp'], ['bad\x00path']])
def test_invalid_snapshot_never_uses_legacy_raw_path(tmp_path: Path, scopes: object) -> None:
    alias_workspace(tmp_path)
    events = tool_events('file_read', {}, {'path': 'alias/service.py', 'resolved_scope_paths': scopes})
    with pytest.raises(GearError):
        RepositoryContext(tmp_path).instructions('base', events)


def test_empty_snapshot_does_not_infer_scopes_from_raw_matches(tmp_path: Path) -> None:
    alias_workspace(tmp_path)
    events = tool_events('glob', {}, {
        'matches': [{'path': 'alias', 'type': 'directory'}], 'resolved_scope_paths': [],
    })
    assert RepositoryContext(tmp_path).instructions('base', events) == 'base'


def test_snapshot_directory_replaced_by_symlink_fails_without_redirecting_rules(tmp_path: Path) -> None:
    alias_workspace(tmp_path)
    result = FileReadTool(tmp_path).run({'path': 'alias/service.py'})
    (tmp_path / 'backend').rename(tmp_path / 'old_backend')
    (tmp_path / 'backend').symlink_to('frontend', target_is_directory=True)
    with pytest.raises(GearError) as error:
        RepositoryContext(tmp_path).instructions('base', tool_events('file_read', {}, result))
    assert error.value.error_type == 'repository_instruction_read_failed'


def test_legacy_activity_explicitly_resolves_current_raw_path(tmp_path: Path) -> None:
    alias_workspace(tmp_path)
    events = tool_events('file_read', {}, {'path': 'alias/service.py', 'content': 'old'})
    (tmp_path / 'alias').unlink()
    (tmp_path / 'alias').symlink_to('frontend', target_is_directory=True)
    instructions = RepositoryContext(tmp_path).instructions('base', events)
    assert 'FRONTEND_RULES' in instructions
    assert 'BACKEND_RULES' not in instructions


@pytest.mark.parametrize('name', ['glob', 'grep'])
def test_search_snapshot_excludes_unreturned_matches(tmp_path: Path, name: str) -> None:
    alias_workspace(tmp_path)
    if name == 'glob':
        result = GlobTool(tmp_path).run({'pattern': '*/service.py', 'max_results': 1})
    else:
        result = GrepTool(tmp_path).run({'path': '.', 'pattern': 'old', 'max_results': 1})
    assert result['truncated'] is True
    assert result['resolved_scope_paths'] == ['backend']
    assert 'FRONTEND_RULES' not in RepositoryContext(tmp_path).instructions('base', tool_events(name, {}, result))
