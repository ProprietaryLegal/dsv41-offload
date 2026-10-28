"""R-UPX groups in the UP3 runner (upx-0064, upx-pin, upx-0067): admission, gates, band cells, decisions; CPU fakes only."""
from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import decision_v2 as dv2, pf1_window as pf, up3_window as w
from offload.launch.windows import nbw_cells as greedy_driver
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, record
from offload.up3 import build as b, upx_plan, upx_driver
from offload.up3.tests import test_up3_window as t
from offload.up3.tests import test_up3_win2 as v2
from offload.upx import band, windows as upx

# Fake server probabilities: one 0.805 token and nineteen 0.01 tokens (top-20 mass 0.995 >= greedy_kl's 0.99).
TOP20: dict[str, Json] = {f'token_id:{i}': math.log(.805 if i == 0 else .01) for i in range(20)}
PINS: dict[str, Json] = {name: {'tokens': 10, 'sha256': name * 8} for name in band.REGIONS}


class Remote(v2.Remote):
    fault = ''

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        if '-engagement-' in label:
            assert band.SPLIT_LOG in command and command.endswith('|| true')
            engaged = cast(w.Arm, self.arm).role == 'split' and self.fault != 'not-engaged'
            return f'INFO EXL3 routed experts: a 40-row call -> two smoe chunks {band.SPLIT_LOG}\n' if engaged else ''
        return super().remote(label, host, command, stdin)

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        if argv[1:3] != ['-m', 'offload.up3.upx_driver']:
            return super().call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)
        self.sup.poll()
        self.commands.append(label)
        self.argvs[label] = argv
        arm = cast(w.Arm, self.arm)
        mode = argv[argv.index('--mode') + 1]
        output = Path(argv[argv.index('--output') + 1])
        data: dict[str, Json]
        if mode == 'greedy':
            data = {**t.greedy_result(), 'near_tie_nats': band.NEAR_TIE_NATS, 'greedy_registration': upx_plan.REGISTRATION}
        elif mode == 'regions':
            data = {'pins': PINS, 'regions': {name: list(range(10)) for name in band.REGIONS}}
        elif mode == 'band-probes':
            top = dict(TOP20)
            if arm.role == 'split' and self.fault == 'screen':
                top['token_id:0'], top['token_id:1'] = math.log(.01), math.log(.805)
            data = {'probes': {str(n): top for n in band.PROBE_LENGTHS}}
        elif mode in ('band-greedy', 'long-greedy'):
            reference = '--reference' in argv
            data = {'checks': [{'i': i, 'output_token_ids': [1, 2, 3], **({'identical': True} if reference else {})}
                               for i in range(2 if mode == 'long-greedy' else 6)],
                    'identical': (2 if mode == 'long-greedy' else 6) if reference else None, 'threshold_nats': band.NEAR_TIE_NATS}
        elif mode == 'band-cell':
            name = argv[argv.index('--cell') + 1]
            ms, acc = self.speed.get(arm.role, (1., 1.))
            data = {'cell': name, 'max_tokens': band.BAND_CELLS[name].archive_max_tokens, 'repetitions': [
                {'repeat': r, 'ms_per_round': 40 * ms * (1 + .003 * ((r * 7) % 5 - 2)),
                 'accepted_per_round': 3 * acc, 'short_latency_median_s': .2, 'band_ok': True, 'band_steps': 50,
                 'band_fraction': .4} for r in range(int(argv[argv.index('--repeats') + 1]))]}
            self.rounds += 10
        elif mode == 'kl':
            data = {'rows': [], 'bound': float(argv[argv.index('--bound') + 1]), 'compared': 0, 'pass': self.fault != 'kl'}
        else:
            assert mode == 'needle'
            data = {'context': 150000, 'recall': 6, 'of': 6, 'foreign_codes': 0}
        output.write_text(json.dumps(data))
        return ''


Rig = Callable[..., tuple[w.Live, Remote]]


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    import offload.prefill_trace.window_requests as wr
    monkeypatch.setattr(greedy_driver, 'greedy_inputs', lambda: (['public']*6, [{'i': i, 'text': 'public'} for i in range(6)]))
    monkeypatch.setattr(wr, 'corpus', lambda root: ('public fixture', {}))
    monkeypatch.setattr(pf, 'CAMPAIGN', tmp_path)
    (tmp_path / 'D2-GEN.PAUSE').touch()
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    sups: list[pf.Supervisor] = []

    def make(group: str, *, speed: dict[str, tuple[float, float]] | None = None, fault: str = '') -> tuple[w.Live, Remote]:
        clock = base.Clock()
        out = tmp_path / group
        out.mkdir()
        sup = base.NoTunnel(out, clock=clock, demand=lambda: 0., monitor=lambda host: [
            os.environ.get('UP3_TEST_PYTHON', '/home/operator/miniconda3/bin/python3'), str(monitor), 'low', str(tmp_path / 'trip')])
        sups.append(sup)
        remote = Remote(out, sup, clock, '', trip=tmp_path / 'trip')
        remote.speed, remote.argvs, remote.timeouts, remote.fault = speed or {}, {}, {}, fault
        live = w.Live(out, sup, group=group, remote='dsv41-deploy-offload', expected='a' * 40, data_dir=tmp_path, tag='tag',
                      call=remote.call, plain=remote.plain, clock=clock)
        remote.live = live
        return live, remote
    try:
        yield make
    finally:
        for sup in sups:
            sup.close()


def test_upx_groups_follow_the_design_and_admission_stays_scoped() -> None:
    for group, spec in upx_plan.GROUPS.items():
        roles = [a.role for a in w.arms(group)]
        assert roles == ['control', spec.candidate, 'control'] and w.cells_for(group) == spec.cells
        assert [tuple(g for g in upx_plan.gates(group, i) if g != 'regions') for i in range(3)] == [
            tuple(s.kind for s in start.steps if s.kind not in ('warmup', 'cell', 'band', 'greedy', 'census', 'regions'))
            for start in spec.window.starts]
        w.admit(group)
        assert {w.near_tie_nats(group, a) for a in w.arms(group)} == {0.9 + 1e-3}
    # split (0064) and vor (0067) stay parked in every other group; cpu-pin keeps its host-gate admission.
    for parked in ('split', 'vor'):
        with pytest.raises(ValueError, match='parked'):
            w.admit(parked)
    assert [upx_plan.gates('upx-0064', i) for i in range(3)] == [
        ('regions', 'band-probes', 'band-greedy', 'engagement'), ('regions', 'band-probes', 'band-greedy', 'engagement'),
        ('regions', 'band-probes', 'band-greedy', 'kl', 'engagement')]
    # long-greedy needs the token regions that UPX-0067's design forgot to list: the runner adds them (reported gap).
    assert [upx_plan.gates('upx-0067', i) for i in range(3)] == [
        ('regions', 'long-greedy'), ('regions', 'long-greedy', 'needle-150k'), ('regions', 'long-greedy', 'kl')]
    assert [upx_plan.repeats('upx-0064', 'band-32k-4user', i) for i in range(3)] == [2, 3, 2]
    # UPX-0064 needs the lifted chained HOLD bounds (coordinator 2026-10-27); upx-pin and upx-0067 fit 85 minutes.
    assert w.budget_minutes('upx-0064') > 85 and w.budget_minutes('upx-0064') <= w.CHAINED_RESTORE_S / 60
    assert w.budget_minutes('upx-pin') <= 85 and w.budget_minutes('upx-0067') <= 85
    assert 'registration upx-0p9' in w.plan('upx-0064') and 'ESTIMATED HOLD' in w.plan('upx-pin')


@pytest.mark.parametrize('group,speed,verdict', [
    ('upx-0064', {'split': (.97, 1.)}, 'CANDIDATE FOR PROMOTION'),
    ('upx-pin', {}, 'NO GAIN SHOWN (stays off)'),
    ('upx-0067', {'vor': (.97, 1.)}, 'CANDIDATE FOR PROMOTION'),
])
def test_full_upx_window(rig: Rig, monkeypatch: pytest.MonkeyPatch, group: str, speed: dict[str, tuple[float, float]],
                         verdict: str) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')   # UPX-0064's 73 min of starts needs the chained HOLD bounds
    live, fake = rig(group, speed=speed)
    result = w.window(live, group, lambda row: None)
    assert 'stop' not in result, result.get('stop')
    assert result['status'] == 'COMPLETE' and result['chain'] == 'CHAINED'
    decision = record(result['decision'])
    assert decision['rule'] == 'win2-v1' and decision['verdict'] == verdict, decision
    assert decision['registration'] == 'upx-0p9' and decision['design'] == upx_plan.group_for(group).design
    names = [a.name for a in w.arms(group)]
    # Every start: six greedy through the upx driver (0.9 nats, recorded distributions), then the design's gates in order.
    for i, name in enumerate(names):
        assert f'{name}-greedy' in fake.argvs
        kinds = [label.removeprefix(f'{name}-upx-') for label in fake.commands if label.startswith(f'{name}-upx-')]
        # Driver gates in design order, bracketed by the stats snapshots; engagement reads both ranks' logs over ssh.
        assert kinds == ['before', *[g if g != 'needle-150k' else 'needle' for g in upx_plan.gates(group, i)
                                     if g != 'engagement'], 'after']
    if 'regions' in upx_plan.gates(group, 0):
        assert '--pins' not in fake.argvs[f'{names[0]}-upx-regions'] and '--pins' in fake.argvs[f'{names[2]}-upx-regions']
    kl = fake.argvs[f'{names[2]}-upx-kl']
    candidates = kl[kl.index('--candidate') + 1:kl.index('--bound')]
    assert candidates[0].endswith(f'{names[1]}-greedy.json') and all(names[1] in c for c in candidates)
    gates = record(record(record(result['starts'])[names[1]])['gates'])
    if group == 'upx-0064':
        assert record(record(gates['upx_gates'])['engagement']) == {h: 40 for h in pf.HOSTS}
        assert record(record(record(record(result['starts'])[names[2]])['gates'])['upx_gates'])['band-screen']
    if group == 'upx-pin':
        assert set(gates) >= set(pf.HOSTS)          # Round 8 receipt census through the cpu-pin host gate
    # Controls are archived per start with WIN2-R2 provenance at each cell's own max_tokens.
    rows = dv2.archive_rows(pf.CAMPAIGN / 'control-archive.jsonl')
    tokens = {c: band.BAND_CELLS[c].archive_max_tokens if c in band.BAND_CELLS else 512 for c in upx_plan.group_for(group).cells}
    assert rows and all(r['max_tokens'] == tokens[str(r['cell'])] and r['commit'] == b.PRODUCTION for r in rows)
    assert {r['start'] for r in rows} == {names[0], names[2]}


@pytest.mark.parametrize('fault,message', [
    ('not-engaged', '0064 engagement'),
    ('screen', 'band KL screen failed'),
    ('kl', 'greedy-divergence KL above the registered bound'),
])
def test_failed_upx_gate_stops_and_restores(rig: Rig, monkeypatch: pytest.MonkeyPatch, fault: str, message: str) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    live, fake = rig('upx-0064', fault=fault)
    result = w.window(live, 'upx-0064', lambda row: None)
    assert message in str(result.get('stop')), result.get('stop')
    assert str(record(result['restore'])['status']).startswith('RESTORED') and fake.restores == ['start', 'check', 'close']
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'


def test_unchained_upx_0064_does_not_fit_the_73_minute_ceiling(rig: Rig) -> None:
    live, fake = rig('upx-0064')
    result = w.window(live, 'upx-0064', lambda row: None)
    assert 'cannot fit' in str(result.get('stop')) and fake.restores == ['check', 'close']


def test_kl_bound_follows_the_band_aa_floor() -> None:
    screen: dict[str, Json] = {'rows': {'33': {'aa_kl': .004}, '42': {'aa_kl': .001}}}
    assert upx_plan.kl_bound(None) == band.KL_FLOOR and upx_plan.kl_bound(screen) == .008


def test_driver_cli_refuses_incomplete_requests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    def run(*args: str) -> int:
        monkeypatch.setattr(sys, 'argv', ['upx', '--output', str(tmp_path / 'x.json'), '--data-dir', str(tmp_path),
                                          '--deadline', '0', '--salt', 'tag', *args])
        return upx_driver.main()
    assert run('--mode', 'regions') == 0 and 'DRY' in capsys.readouterr().out
    for bad in (('--mode', 'band-cell', '--cell', 'band-32k-4user'), ('--mode', 'kl', '--bound', '0.01'),
                ('--mode', 'band-probes'), ('--mode', 'band-cell', '--cell', 'band-32k-4user', '--repeats', '13',
                                            '--regions', 'r', '--cuts', 'c')):
        with pytest.raises(ValueError):
            run(*bad)
    assert not (tmp_path / 'x.json').exists()
