"""WIN2-R3 seeded A/A pilot recipe: two starts, PASS on identical B1 streams, measured diff CV and the A/A band input."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.gb10_budget import window_cells as wc
from offload.gb10_budget.test_window_cells_seeded import SeededFake, plan, texts
from offload.launch.windows import aa_pilot as ap
from offload.launch.windows import dist_gate as dg
from offload.launch.windows.tests.test_dist_gate import FakeDist, materials
from offload.prefill_maxlen.evidence import Json, number, record

CELLS = ('seeded-short-b1', 'seeded-32k-b1', 'seeded-32k-b3')


@pytest.fixture(scope='module')
def setup() -> tuple[str, dict[str, Json]]:
    text = materials()
    prompts = dg.selected(['short'])[:3]
    return text, record(dg.cuts(FakeDist(), text, {'x': 'y'}, prompts)['prompts'])


def _start(tmp_path: Path, setup: tuple[str, dict[str, Json]], seeded: SeededFake, dist: FakeDist) -> dict[str, Json]:
    text, rows = setup
    return ap.run(seeded, dist, texts(), plan(tmp_path), text, rows, cells=CELLS)


def test_identical_starts_pass_and_give_the_band_input(tmp_path: Path, setup: tuple[str, dict[str, Json]]) -> None:
    a = _start(tmp_path, setup, SeededFake(), FakeDist())
    b = _start(tmp_path, setup, SeededFake(speed=1.01), FakeDist(shift=0.001))
    r = ap.compare(a, b)
    assert r['verdict'] == 'PASS'
    cells = record(r['cells'])
    short = record(cells['seeded-short-b1'])
    assert short['identical'] is True and short['pairs_same_rounds'] == 6 and short['pairs_same_output'] == 6
    tps = record(record(short['metrics'])['decode_tps'])
    assert number(tps['mean_diff']) == pytest.approx(0.01) and number(tps['diff_cv']) < 1e-9
    band = dg.register_band([record(r['dist'])])
    assert band['positions'] == 3 * dg.POSITIONS and band['aa_comparisons'] == 1


def test_b1_divergence_fails_but_multi_divergence_is_reported(tmp_path: Path, setup: tuple[str, dict[str, Json]]) -> None:
    a = _start(tmp_path, setup, SeededFake(), FakeDist())
    b = _start(tmp_path, setup, SeededFake(shift=1), FakeDist())
    r = ap.compare(a, b)
    assert r['verdict'] == 'FAIL' and 'B1 seeded cell differs' in str(r['reason'])
    multi_only = {**a, 'cells': {k: v for k, v in record(a['cells']).items() if k == 'seeded-32k-b3'}}
    multi_b = {**b, 'cells': {k: v for k, v in record(b['cells']).items() if k == 'seeded-32k-b3'}}
    assert ap.compare(multi_only, multi_b)['verdict'] == 'FAIL'                 # no B1 cell ran: cannot PASS
    with pytest.raises(ValueError, match='different pilot cells'):
        ap.compare(a, multi_b)


def test_cli_dry_plan_compare_and_coordinator_gate(tmp_path: Path, setup: tuple[str, dict[str, Json]],
                                                   monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    common = ['--url', wc.URL, '--cuts', 'c', '--dist-cuts', 'd', '--salt', 's', '--deadline-utc', '1', '--out',
              str(tmp_path / 'x.json')]
    assert ap.main(['run', *common]) == 0 and 'DRY RUN' in capsys.readouterr().out
    monkeypatch.delenv('WINDOW_COORDINATOR', raising=False)
    assert ap.main(['run', '--execute', *common]) == 2
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(json.dumps(_start(tmp_path, setup, SeededFake(), FakeDist())))
    b.write_text(json.dumps(_start(tmp_path, setup, SeededFake(), FakeDist())))
    assert ap.main(['compare', str(a), str(b), '--out', str(tmp_path / 'cmp.json')]) == 0
    assert json.loads((tmp_path / 'cmp.json').read_text())['verdict'] == 'PASS'
