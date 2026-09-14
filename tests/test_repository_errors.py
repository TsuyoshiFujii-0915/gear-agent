from pathlib import Path

import pytest

from gear_agent.errors import GearError
from gear_agent.repository import RepositoryContext


@pytest.mark.parametrize('path', ['bad\x00path', 'bad\ud800path'])
def test_malformed_scope_paths_have_structured_errors(tmp_path: Path, path: str) -> None:
    with pytest.raises(GearError) as error:
        RepositoryContext(tmp_path).discover((Path(path),))
    assert error.value.error_type == 'repository_path_invalid'
    assert error.value.origin == 'repository_context'
    assert error.value.details['path'] == path
    assert error.value.__cause__ is not None


def test_workspace_replaced_by_symlink_is_rejected_on_next_read(tmp_path: Path) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    context = RepositoryContext(workspace)
    assert context.discover(()) == ()
    workspace.rmdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'AGENTS.md').write_text('outside instructions', encoding='utf-8')
    workspace.symlink_to(outside, target_is_directory=True)
    with pytest.raises(GearError) as error:
        context.discover(())
    assert error.value.error_type == 'repository_instruction_read_failed'
