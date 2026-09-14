from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
import os
import stat

from gear_agent.errors import GearError, gear_error


MAX_INSTRUCTION_FILE_BYTES = 32 * 1024
MAX_REPOSITORY_INSTRUCTION_BYTES = 128 * 1024
_SCOPED_TOOLS = frozenset({'file_read', 'file_write', 'apply_patch', 'glob', 'grep', 'shell'})


@dataclass(frozen=True)
class RepositoryInstruction:
    """One current instruction file within the workspace.

    Attributes:
        path: Workspace-relative physical instruction path.
        scope: Workspace-relative directory governed by the instructions.
        content: Unmodified, strictly decoded UTF-8 file content.
    """

    path: Path
    scope: Path
    content: str


class RepositoryContext:
    """Discovers and renders current repository rules before model execution."""

    def __init__(self, workspace: Path) -> None:
        """Binds discovery to the effective workspace.

        Args:
            workspace: Workspace selected by configuration and CLI overrides.

        Raises:
            GearError: If the workspace cannot be resolved to a directory.
        """
        try:
            self._workspace = workspace.resolve(strict=True)
            if not stat.S_ISDIR(self._workspace.stat().st_mode):
                raise NotADirectoryError(str(workspace))
        except (OSError, RuntimeError, ValueError) as exc:
            raise _error(
                'repository_workspace_invalid', 'Workspace is not an accessible directory.',
                {'path': str(workspace), 'reason': str(exc)},
            ) from exc

    def discover(self, directories: tuple[Path, ...]) -> tuple[RepositoryInstruction, ...]:
        """Loads root rules and ancestor chains for the given directory scopes.

        Args:
            directories: Workspace-relative directory paths, including missing ones.

        Returns:
            Unique records ordered by depth and then POSIX relative path.

        Raises:
            GearError: If a path escapes, a file is invalid, or a size limit is exceeded.
        """
        return self._discover_physical(
            tuple(self._resolve_path(str(directory)) for directory in directories)
        )

    def _discover_physical(self, directories: tuple[Path, ...]) -> tuple[RepositoryInstruction, ...]:
        scopes = {Path('.')}
        for directory in directories:
            physical = _relative_scope_path(str(directory))
            scopes.add(physical)
            scopes.update(physical.parents)
        records: list[RepositoryInstruction] = []
        total_bytes = 0
        for scope in sorted(scopes, key=lambda path: (len(path.parts), path.as_posix())):
            content = self._read_instruction(scope, MAX_REPOSITORY_INSTRUCTION_BYTES - total_bytes)
            if content is None:
                continue
            total_bytes += len(content)
            path = scope / 'AGENTS.md'
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError as exc:
                raise _error(
                    'repository_instruction_invalid_utf8', 'Instruction file is not valid UTF-8.',
                    {'path': path.as_posix(), 'reason': str(exc)},
                ) from exc
            records.append(RepositoryInstruction(path, scope, text))
        return tuple(records)

    def instructions(self, base: str, events: list[dict[str, Any]]) -> str:
        """Builds the current request instructions without modifying session history.

        Args:
            base: Gear's own instructions, retained verbatim at the beginning.
            events: Complete persisted session activity, including before compaction.

        Returns:
            Base instructions followed by delimited repository instruction blocks.

        Raises:
            GearError: If applicable context cannot be constructed safely.
        """
        records = self._discover_physical(self._activity_directories(events))
        if not records:
            return base
        blocks = [
            base,
            'Repository instructions apply only within their declared directory scopes. '
            'Within a scope, deeper instructions take precedence over broader instructions.',
        ]
        for record in records:
            blocks.append(
                f'<repository_instructions path="{escape(record.path.as_posix(), quote=True)}" '
                f'scope="{escape(record.scope.as_posix(), quote=True)}">\n'
                f'{escape(record.content, quote=False)}\n</repository_instructions>'
            )
        return '\n\n'.join(blocks)

    def _resolve_path(self, raw_path: str) -> Path:
        path = _relative_scope_path(raw_path)
        try:
            resolved = (self._workspace / path).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise _error(
                'repository_path_invalid', 'Cannot resolve repository scope.',
                {'path': raw_path, 'reason': str(exc)},
            ) from exc
        if not resolved.is_relative_to(self._workspace):
            raise _error(
                'repository_path_outside_workspace', 'Repository scope resolves outside the workspace.',
                {'path': raw_path},
            )
        return resolved.relative_to(self._workspace)

    def _read_instruction(self, scope: Path, remaining_bytes: int) -> bytes | None:
        path = (scope / 'AGENTS.md').as_posix()
        directory_fd: int | None = None
        file_fd: int | None = None
        try:
            directory_fd = os.open(self._workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            for part in scope.parts:
                try:
                    child_fd = os.open(
                        part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd,
                    )
                except FileNotFoundError:
                    return None
                os.close(directory_fd)
                directory_fd = child_fd
            try:
                file_fd = os.open(
                    'AGENTS.md', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd,
                )
            except FileNotFoundError:
                return None
            metadata = os.fstat(file_fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError('Instruction file must be a regular file.')
            with os.fdopen(file_fd, 'rb', closefd=False) as source:
                _check_size(path, metadata.st_size, remaining_bytes)
                content = source.read(min(MAX_INSTRUCTION_FILE_BYTES, remaining_bytes) + 1)
                _check_size(path, len(content), remaining_bytes)
                return content
        except OSError as exc:
            raise _error(
                'repository_instruction_read_failed', 'Cannot safely read instruction file.',
                {'path': path, 'reason': str(exc)},
            ) from exc
        finally:
            if file_fd is not None:
                os.close(file_fd)
            if directory_fd is not None:
                os.close(directory_fd)

    def _activity_directories(self, events: list[dict[str, Any]]) -> tuple[Path, ...]:
        directories: set[Path] = set()
        shell_workdirs: dict[str, object] = {}
        for event in events:
            kind = event.get('kind')
            if kind not in ('tool_call', 'tool_result'):
                continue
            payload = _scope_object(event.get('payload'), 'payload')
            name = _scope_string(payload.get('name'), 'name')
            if name not in _SCOPED_TOOLS:
                continue
            if kind == 'tool_call':
                if name == 'shell':
                    call_id = _scope_string(payload.get('call_id'), 'call_id')
                    arguments = _scope_object(payload.get('arguments'), 'arguments')
                    # Failed calls may have invalid arguments; validate only after execution.
                    shell_workdirs[call_id] = arguments.get('workdir')
                continue
            result = _scope_object(payload.get('result'), 'result')
            if 'error' in result:
                continue
            if 'resolved_scope_paths' in result:
                for scope in _scope_list(result['resolved_scope_paths'], 'resolved_scope_paths'):
                    directories.add(_relative_scope_path(_scope_string(scope, 'resolved_scope_paths')))
                continue
            # Only the legacy result schema lacks execution-time physical scopes.
            if name in ('file_read', 'file_write'):
                directories.add(self._file_directory(result.get('path')))
            elif name == 'apply_patch':
                for path in _scope_list(result.get('changed_files'), 'changed_files'):
                    directories.add(self._file_directory(path))
            elif name in ('glob', 'grep'):
                for item in _scope_list(result.get('matches'), 'matches'):
                    match = _scope_object(item, 'match')
                    if name == 'grep' or match.get('type') == 'file':
                        directories.add(self._file_directory(match.get('path')))
                    elif match.get('type') == 'directory':
                        directories.add(self._resolve_path(_scope_string(match.get('path'), 'path')))
                    else:
                        raise _scope_error('match.type', match.get('type'))
            else:
                call_id = _scope_string(payload.get('call_id'), 'call_id')
                directories.add(self._resolve_path(_scope_string(shell_workdirs.get(call_id), 'shell.workdir')))
        return tuple(directories)

    def _file_directory(self, value: object) -> Path:
        return self._resolve_path(_scope_string(value, 'path')).parent


def _relative_scope_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or '..' in path.parts:
        raise _error(
            'repository_path_outside_workspace', 'Repository scope must be workspace-relative.',
            {'path': raw_path},
        )
    try:
        raw_path.encode('utf-8')
        if '\x00' in raw_path:
            raise ValueError('Path contains a null byte.')
    except ValueError as exc:
        raise _error(
            'repository_path_invalid', 'Invalid repository scope path.',
            {'path': raw_path, 'reason': str(exc)},
        ) from exc
    return path


def _check_size(path: str, size_bytes: int, remaining_bytes: int) -> None:
    if size_bytes > MAX_INSTRUCTION_FILE_BYTES:
        raise _error(
            'repository_instruction_file_too_large', 'Instruction file exceeds the byte limit.',
            {'path': path, 'limit_bytes': MAX_INSTRUCTION_FILE_BYTES, 'size_bytes': size_bytes},
        )
    if size_bytes > remaining_bytes:
        raise _error(
            'repository_instructions_too_large', 'Repository instructions exceed the total byte limit.',
            {'path': path, 'limit_bytes': MAX_REPOSITORY_INSTRUCTION_BYTES},
        )


def _scope_object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _scope_error(field, value)
    return value


def _scope_string(value: object, field: str) -> str:
    if not isinstance(value, str) or value == '':
        raise _scope_error(field, value)
    return value


def _scope_list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise _scope_error(field, value)
    return value


def _scope_error(field: str, value: object) -> GearError:
    return _error(
        'repository_scope_invalid', 'Completed tool activity has invalid repository scope metadata.',
        {'field': field, 'value_type': type(value).__name__},
    )


def _error(error_type: str, message: str, details: dict[str, object]) -> GearError:
    return gear_error(error_type, message, 'repository_context', True, details)
