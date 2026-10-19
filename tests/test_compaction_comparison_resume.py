import json
from pathlib import Path

import pytest

from examples.compaction_comparison import CURRENT, copy_source_run


def test_resume_preserves_exact_source_and_workspace(tmp_path: Path) -> None:
    source = tmp_path / 'old'
    target = tmp_path / 'new'
    (source / 'sessions').mkdir(parents=True)
    (source / 'workspace').mkdir()
    target.mkdir()
    original = (json.dumps({'kind': 'user_input', 'payload': {'text': CURRENT}}) + '\n').encode()
    (source / 'sessions/source.jsonl').write_bytes(original)
    (source / 'workspace/retry.py').write_text('original code\n')
    (source / 'scenario.json').write_text(json.dumps({'current': CURRENT}))
    copy_source_run(source, target)
    assert (target / 'sessions/source.jsonl').read_bytes() == original
    assert (source / 'sessions/source.jsonl').read_bytes() == original
    assert (target / 'workspace/retry.py').read_text() == 'original code\n'
    assert not (target / 'summary').exists()


def test_resume_rejects_unfinished_session_before_copying(tmp_path: Path) -> None:
    source = tmp_path / 'old'
    target = tmp_path / 'new'
    (source / 'sessions').mkdir(parents=True)
    target.mkdir()
    (source / 'sessions/source.jsonl').write_text(json.dumps({
        'kind': 'assistant_message', 'payload': {'text': 'Previous answer'},
    }) + '\n')
    with pytest.raises(ValueError, match='pending user message'):
        copy_source_run(source, target)
    assert list(target.iterdir()) == []
