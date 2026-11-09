from pathlib import Path

from tests.test_model_check import Endpoint, check, config, endpoint


def test_probe_cap_must_fit_automatic_reserve(endpoint: Endpoint, tmp_path: Path) -> None:
    path = config(tmp_path / 'gear.toml', endpoint.base + '/v1/responses', False)
    path.write_text(path.read_text().replace('max_output_tokens = 96\n', '')
                    .replace('reserved_tokens = 128', 'reserved_tokens = 0')
                    .replace('auto_compaction = false', 'auto_compaction = true'))
    code, report = check(path, '--infer')
    assert code == 1 and report['error']['field'] == 'context_budget.reserved_tokens'
    assert endpoint.requests == []
