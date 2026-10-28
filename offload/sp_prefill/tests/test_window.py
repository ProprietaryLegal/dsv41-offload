"""R-SP window: plan, win2-v1 decision, archive rows, chain close, arm config and microbenchmark logic (CPU only)."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import cast

import pytest
import torch

from offload.launch.windows import chain
from offload.launch.windows import decision_v2 as d2
from offload.launch.windows.tests import test_pf1_config as pf1
from offload.prefill_maxlen.evidence import Json
from offload.sp_prefill import build_overlay as bo
from offload.sp_prefill import rs_bench
from offload.sp_prefill import window as w
from offload.sp_prefill.tests.fake_tp import FakeTP

CONFIG = 'offload/launch/configs/RSP-window.env'
LIB = '/usr/local/lib/python3.12/dist-packages'


def measured(ttft_gain: float = 0.0, noise: float = 0.002) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Synthetic per-repeat values: candidate TTFT lower by ttft_gain; decode unchanged."""
    base = {'32k-b1': {'ttft_s': 14.8, 'committed_tps': 52.0}, '150k-b1': {'ttft_s': 69.5, 'committed_tps': 48.0},
            'short-b1': {'committed_tps': 55.0, 'ms_per_round': 60.0},
            'mixed150k-2decode': {'mean_completion_s': 120.0, 'ttft_s': 110.0}}
    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for k, arm in enumerate(('c1', 'x1', 'c2')):
        out[arm] = {}
        for cell, metrics in base.items():
            n = w.repeats(cell)
            out[arm][cell] = {}
            for metric, value in metrics.items():
                scale = 1 - ttft_gain if arm == 'x1' and metric == 'ttft_s' else 1.0
                out[arm][cell][metric] = [value * scale * (1 + noise * ((i + k) % 3 - 1)) for i in range(n)]
    return out


def test_plan_registers_two_ttft_decision_cells_three_repeats() -> None:
    p = w.plan()
    assert p['decision_cells'] == ['32k-b1', '150k-b1'] and p['rule'] == 'win2-v1'
    assert p['repeats'] == {'32k-b1': 3, '150k-b1': 3, 'short-b1': 1, 'mixed150k-2decode': 1}
    arms = cast(list[dict[str, Json]], p['arms'])
    assert [a['env'] for a in arms] == [{'RSP_SP': '0'}, {'RSP_SP': '1'}, {'RSP_SP': '0'}]
    assert w.METRICS['32k-b1'][0] == ('32k-b1', 'ttft_s', False) and w.METRICS['150k-b1'][0][1] == 'ttft_s'
    est = cast(dict[str, int], p['estimated_s'])
    assert est['hold_chained'] == 120 + 300 + 3 * w.start_seconds() + 120
    assert est['hold_last'] == est['hold_chained'] - 120 + 720 + 900
    assert 60 * 60 < est['hold_chained'] < 75 * 60


@pytest.mark.parametrize(('gain', 'status'), [(0.05, 'CANDIDATE'), (0.0, 'INCONCLUSIVE'), (-0.05, 'NOT A CANDIDATE')])
def test_decision_is_win2(gain: float, status: str) -> None:
    result = w.decide(measured(gain), {'verdict': 'IDENTICAL'})
    assert result['rule'] == 'win2-v1' and result['status'] == status
    assert result['decision_cells'] == ['32k-b1', '150k-b1']
    cells = cast(dict[str, dict[str, Json]], result['cells'])
    assert cells['short-b1']['verdict'] == 'REPORTED' and cells['32k-b1:server']['verdict'] == 'REPORTED'
    assert result['groups'] == {'single-user': status}


def test_greedy_failure_overrides_a_speed_win() -> None:
    result = w.decide(measured(0.05), {'verdict': 'DIVERGED', 'first': 3})
    assert result['status'] == 'NOT A CANDIDATE' and 'greedy DIVERGED' in str(result['greedy_stop'])
    assert w.decide(measured(0.05), {'verdict': 'NEAR-TIE'})['status'] == 'CANDIDATE'


def test_decision_refuses_short_or_missing_arms() -> None:
    m = measured()
    m['x1']['32k-b1']['ttft_s'] = m['x1']['32k-b1']['ttft_s'][:2]
    with pytest.raises(ValueError, match='needs 3 repeats'):
        w.decide(m, {'verdict': 'IDENTICAL'})
    del m['c2']
    with pytest.raises(ValueError, match='arms'):
        w.series(m)


PROD = '21bed31' + 'a' * 33


def test_archive_rows_carry_provenance_and_append(tmp_path: Path) -> None:
    caps = {'c1': 2300, 'c2': 2150}
    rows = w.archive_rows('RSP-W1', measured(), commit=PROD, clock_cap_mhz=caps, date='2026-10-28')
    assert rows and all(r['max_tokens'] == 512 and r['commit'] == PROD and r['date'] == '2026-10-28' for r in rows)
    assert {(r['start'], r['clock_cap_mhz']) for r in rows} == {('c1', 2300), ('c2', 2150)}, "cap of the row's own start"
    assert len([r for r in rows if r['cell'] == '150k-b1' and r['metric'] == 'ttft_s']) == 6
    assert sorted(int(str(r['repeat'])) for r in rows if r['cell'] == '32k-b1' and r['metric'] == 'ttft_s') == [0, 0, 1, 1, 2, 2]
    archive = tmp_path / 'control-archive.jsonl'
    d2.archive_append(archive, rows)
    assert len(d2.archive_rows(archive)) == len(rows)
    assert set(w.spreads(archive, commit=PROD, clock_cap_mhz=2150)) == {'32k-b1', '150k-b1'}
    for bad in (dict(commit='abc'), dict(date='28.10.2026'), dict(clock_cap_mhz={'c1': 2150}),
                dict(clock_cap_mhz={'c1': 2150, 'c2': True})):
        kwargs = {'commit': PROD, 'clock_cap_mhz': caps, 'date': '2026-10-28', **bad}
        with pytest.raises(ValueError, match='R-SP archive'):
            w.archive_rows('RSP-W1', measured(), **kwargs)  # type: ignore[arg-type]


def test_close_follows_the_chain_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ('WINDOW_CHAIN', 'WINDOW_CHAIN_LAST'):
        monkeypatch.delenv(var, raising=False)
    assert w.close(stopped=False)['chain'] == 'OFF'
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chained = w.close(stopped=False)
    assert chained['chain'] == 'CHAINED' and chained['post_close_watch'] is False
    stopped = w.close(stopped=True)
    assert stopped['restore'] == {'status': 'W8 strict restore'} and stopped['clear_marker'] is True
    monkeypatch.setenv('WINDOW_CHAIN_LAST', '1')
    assert w.close(stopped=False)['chain'] == 'LAST'


def test_preflight_modes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(chain, 'MARKER', tmp_path / 'CHAINED')
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    assert w.preflight(True, False)['mode'] == 'production'
    (tmp_path / 'CHAINED').write_text(json.dumps({'schema': chain.SCHEMA}))
    assert w.preflight(False, False) == {'mode': 'chained', 'baseline': 'c1 control start'}
    with pytest.raises(ValueError, match='HOLD'):
        w.preflight(False, True)


def test_memory_floor() -> None:
    assert w.memory_ok({'sparkhead': w.MEM_FLOOR_KB, 'sparkworker': w.MEM_FLOOR_KB + 1})
    assert not w.memory_ok({'sparkhead': w.MEM_FLOOR_KB - 1, 'sparkworker': w.MEM_FLOOR_KB})
    assert not w.memory_ok({})


# ---------------------------------------------------------------- arm config (PF1-window.env OFF + R-SP)


def source(home: Path, prod_text: str, **env: str) -> dict[str, str]:
    (home / 'dsv41-deploy').mkdir(parents=True, exist_ok=True)
    (home / 'dsv41-deploy/dsv41.env').write_text(prod_text)
    printer = ' '.join(f'"${{{f}-}}"' for f in pf1.FIELDS)
    script = (f'set -euo pipefail; set -a; . ./dsv41.env; PORT=8001; . {CONFIG}; set +a; test "$PORT" = 8001; '
              f'test -z "$(compgen -v | grep -E "^(pf1|rsp)_" || true)"; printf "%s\\0" {printer}')
    result = subprocess.run(['bash', '-c', script], cwd=pf1.ROOT, text=True, capture_output=True,
                            env={'HOME': str(home), 'PATH': os.environ['PATH'], **env})
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return dict(zip(pf1.FIELDS, result.stdout.split('\0')))


def built(tmp_path: Path) -> Path:
    out = tmp_path / 'rsp'
    bo.build(out)
    return out


def test_control_arm_is_pf1_window_off(tmp_path: Path) -> None:
    text = pf1.production()
    got = source(tmp_path, text)
    off = pf1.source(tmp_path, text)
    assert {k: v for k, v in got.items() if k != 'LABEL'} == {k: v for k, v in off.items() if k != 'LABEL'}
    assert got['LABEL'] == 'RSP-window-327680-sp0'


def test_candidate_arm_swaps_model_py_and_adds_the_switch(tmp_path: Path) -> None:
    text = pf1.production()
    root = built(tmp_path)
    got = source(tmp_path / 'home', text, RSP_SP='1', RSP_ROOT=str(root))
    off = pf1.source(tmp_path / 'home', text)
    assert got['EXP_DOCKER_ENV'].split() == off['EXP_DOCKER_ENV'].split() + ['VLLM_DSV41_PREFILL_SP=1']
    assert sorted(got['OVERLAYS'].split()) == sorted(f'{root}/{p}={LIB}/{p}' for p in (bo.MODEL, bo.HELPER))
    kept, before = got['EXTRA_DOCKER_ARGS'].split()[1::2], off['EXTRA_DOCKER_ARGS'].split()[1::2]
    removed = [x for x in before if x not in kept]
    assert len(removed) == 1 and removed[0].endswith(f':{LIB}/{bo.MODEL}:ro')
    assert [x for x in before if x in kept] == kept
    assert got['LABEL'] == 'RSP-window-327680-sp1'


def test_candidate_arm_refuses_unpinned_sources_and_pf1_switches(tmp_path: Path) -> None:
    text = pf1.production()
    root = built(tmp_path)
    (root / bo.HELPER).write_text((root / bo.HELPER).read_text() + '\n')
    with pytest.raises(RuntimeError, match='differ from OVERLAY.sha256'):
        source(tmp_path / 'home', text, RSP_SP='1', RSP_ROOT=str(root))
    with pytest.raises(RuntimeError, match='trace/eager'):
        source(tmp_path / 'home', text, PF1_TRACE='1')
    with pytest.raises(RuntimeError, match='RSP_SP must be'):
        source(tmp_path / 'home', text, RSP_SP='2')


# ---------------------------------------------------------------- microbenchmark logic


def test_rs_bench_logic_on_fake_tp2() -> None:
    tp = FakeTP(2)
    ticks = iter(range(10**6))
    results = tp.run(lambda r: rs_bench.run(_Coll(tp), lambda fn: (fn(), float(next(ticks)))[1],
                                            rows=(33, 64), hidden=8, iters=3, warmup=1))
    for res in results:
        assert res['rs_equals_ar_bitwise'] is True and res['verdict'] == 'SP ARM MAY RUN'
        cells = res['cells']
        assert isinstance(cells, list) and [c['padded_rows'] for c in cells] == [34, 64]


def test_rs_bench_stops_the_sp_arm_on_a_non_commutative_reduce() -> None:
    tp = FakeTP(2)
    results = tp.run(lambda r: rs_bench.run(_Coll(tp, skew=True), lambda fn: (fn(), 1.0)[1],
                                            rows=(16,), hidden=8, iters=1, warmup=0))
    assert all(str(res['verdict']).startswith('STOP SP ARM') for res in results)


class _Coll:
    def __init__(self, tp: FakeTP, skew: bool = False) -> None:
        self.tp, self.skew = tp, skew

    @property
    def world_size(self) -> int:
        return self.tp.world_size

    @property
    def rank(self) -> int:
        return self.tp.rank

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.tp.all_reduce(x)

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tp.reduce_scatter(x)
        return out + 1e-2 if self.skew else out

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.tp.all_gather(x)


def test_rs_bench_plan_is_dry(capsys: pytest.CaptureFixture[str]) -> None:
    assert rs_bench.main(['--plan']) == 0
    assert json.loads(capsys.readouterr().out)['rows'] == [4096, 8192, 16384]
