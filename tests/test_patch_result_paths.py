from pathlib import Path

import pytest

from gear_agent.tools.patch import ApplyPatchTool


@pytest.mark.parametrize('filename', ['service.py', 'file name.py', "quote's.py", "'quoted'.py"])
def test_patch_reports_actual_changed_path_and_physical_scope(tmp_path: Path, filename: str) -> None:
    (tmp_path / 'backend').mkdir()
    (tmp_path / 'alias').symlink_to('backend', target_is_directory=True)
    (tmp_path / 'backend' / filename).write_text('old\n', encoding='utf-8')
    path = 'alias/' + filename
    patch_text = f'--- {path}\t2026-01-01\n+++ {path}\t2026-01-01\n@@ -1 +1 @@\n-old\n+new\n'
    result = ApplyPatchTool(tmp_path).run({'patch': patch_text})
    assert (tmp_path / 'backend' / filename).read_text(encoding='utf-8') == 'new\n'
    assert result['changed_files'] == [path]
    assert result['resolved_scope_paths'] == ['backend']
