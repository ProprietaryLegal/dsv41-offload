"""UP3 on WIN2: ring-depth2, defer (0051) and k5-batch windows, win2-v1 decisions, archive; CPU fakes only."""
from __future__ import annotations

import copy
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import chain, decision_v2 as dv2, pf1_window as pf, up3_window as w
from offload.launch.windows import nbw_cells as greedy_driver
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, number, record
from offload.up3 import build as b, cells, host_gates, win2_plan as plan
from offload.up3.tests import test_up3_window as t

GROUP = 'ring-depth2'
# Per-role multipliers for the fake server: (server ms per round, accepted per round).
Speed = dict[str, tuple[float, float]]


def measured(name: str, repeats: int, ms: float, acc: float, salt: str) -> dict[str, Json]:
    """One cell artifact with `repeats` repeats; small deterministic repeat-to-repeat noise, identical prompts/seeds."""
    batch = cells.batch_for(name)
    rows: list[Json] = []
    for r in range(repeats):
        noise = 1 + .003 * ((r * 7) % 5 - 2)
        streams: list[Json] = [{'prompt_sha256': str(i) * 64, 'prompt_tokens': 150000 if name == cells.CELLS[-1] and i == 2 else 32000,
            'budget': 512, 'decode_tps': 10.0, 'ttft_s': 1. + .01 * r, 'p99_inter_token_delivery_gap_ms': 100.,
            'finish': 'length', 'sampling': {'temperature': 1, 'top_p': 1, 'seed': 42 + r, 'priority': 10, 'max_tokens': 512},
            'cache_salt_sha256': f'{salt}-{name}-{r}-{i}'} for i in range(batch)]
        rounds = 1_000_000
        rows.append({'repeat': r, 'batch': batch, 'streams': streams, 'spec_request_rounds': rounds,
                     'accepted_tokens': round(rounds * 3 * acc * noise), 'server_decode_ms': rounds * 40 * ms * noise,
                     'completed_requests': batch, 'mixed_three_active_overlap': name == cells.CELLS[-1]})
    return {'cell': name, 'batch': batch, 'repetitions': rows}


class Remote(t.Remote):
    speed: Speed = {}
    argvs: dict[str, list[str]]
    timeouts: dict[str, float]
    cap = 2300

    def identity(self, rank: int) -> dict[str, Json]:
        ident = t.identity(cast(w.Arm | None, self.arm), rank, self.serial)
        if self.arm is not None and cast(w.Arm, self.arm).role == 'fixed2':
            argv = shlex.split(str(ident['argv']).replace('\\\n', ' '))
            argv[argv.index('--speculative-config') + 1] = b.SPEC
            ident['argv'] = shlex.join(argv)
        return ident

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        if label.endswith('-start'):
            super().remote(label, host, command, stdin)
            text = ''.join(f'{node} clock cap {self.cap} MHz\n' for node in ('spark', 'sparkworker'))
            # The Supervisor keeps every command's output as cmd/NNNN-<label>.log; win2_history.start_clock_cap reads it.
            logs = self.out / 'cmd'
            logs.mkdir(exist_ok=True)
            (logs / f'{len(list(logs.iterdir())) + 1:04d}-{label}.log').write_text(text)
            return text
        if '-host-gate-' in label and cast(w.Arm, self.arm).role in ('dual-ring', 'ring-defer'):
            rank = pf.HOSTS.index(host)
            return json.dumps({'arm': 'dual-ring', 'rank': rank, 'container': self.live.ids[rank],
                'rdma': [f'EXL3 RDMA all-reduce: rank {rank} of 2 on rocep1s0f1,roceP2p1s0f1, gid 3, proxy on cpu 14, '
                         'up to 256 KB (slots 3072 KB, all-gathers up to 3072 KB)'],
                'gids': [{'device': dev, 'index': 3, 'type': 'RoCE v2', 'ipv4': f'10.202.{26+i}.{rank+1}', 'netdev': 'eth'+str(i)}
                         for i, dev in enumerate(host_gates.DEVICES)]})
        return super().remote(label, host, command, stdin)

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        self.timeouts[label] = timeout
        text = super().call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)
        if argv[1:3] == ['-m', 'offload.up3.cells']:
            self.argvs[label] = argv
            output = Path(argv[argv.index('--output') + 1])
            if argv[argv.index('--mode') + 1] == 'greedy' and '--near-tie-nats' in argv:
                data = json.loads(output.read_text())
                data['near_tie_nats'] = float(argv[argv.index('--near-tie-nats') + 1])
                output.write_text(json.dumps(data))
            elif '--repeats' in argv:
                arm = cast(w.Arm, self.arm)
                ms, acc = self.speed.get(arm.role, (1., 1.))
                output.write_text(json.dumps(measured(argv[argv.index('--cell') + 1], int(argv[argv.index('--repeats') + 1]),
                                                      ms, acc, arm.name)))
        return text


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

    def make(name: str = 'W', *, speed: Speed | None = None, group: str = GROUP, fail: str = '') -> tuple[w.Live, Remote]:
        clock = base.Clock()
        out = tmp_path / name
        out.mkdir()
        sup = base.NoTunnel(out, clock=clock, demand=lambda: 0., monitor=lambda host: [
            os.environ.get('UP3_TEST_PYTHON', '/home/operator/miniconda3/bin/python3'), str(monitor), 'low', str(tmp_path / 'trip')])
        sups.append(sup)
        remote = Remote(out, sup, clock, fail, trip=tmp_path / 'trip')
        remote.speed = speed or {}
        remote.argvs, remote.timeouts = {}, {}
        live = w.Live(out, sup, group=group, remote='dsv41-deploy-offload', expected='a' * 40, data_dir=tmp_path, tag='tag',
                      call=remote.call, plain=remote.plain, clock=clock)
        remote.live = live
        return live, remote
    try:
        yield make
    finally:
        for sup in sups:
            sup.close()


BLOCKS = {'ring-depth2': [981, 981, 1011, 981], 'defer-shares': [981] * 4, 'k5-batch': [1256] * 3}


def test_plan_shape_and_measured_budget() -> None:
    assert set(plan.GROUPS) == set(BLOCKS) and set(plan.GROUPS) <= set(w.GROUPS)
    for group, p in plan.GROUPS.items():
        assert [a.role for a in w.arms(group)] == list(p.roles) and p.roles[0] == p.roles[-1] == 'control'
        assert [w.block_s(group, a, i == 0) for i, a in enumerate(w.arms(group))] == BLOCKS[group]
        # Every mandatory start fits the HOLD+73 submit ceiling with > 6 minutes left; whole HOLD <= 85 minutes.
        assert w.SUBMIT_S - sum(BLOCKS[group]) > 360 and w.budget_minutes(group) <= 85
        assert len(p.decision_cells) == 2 and set(p.decision_cells) <= set(p.cells) == set(p.repeats)
        assert all(p.repeats[c] >= 3 for c in p.decision_cells) and w.cells_for(group) == p.cells
        assert 'DRY RUN' in w.plan(group) and f'ESTIMATED HOLD {w.budget_minutes(group):.2f}min' in w.plan(group)
        for role in p.roles[1:-1]:
            w.admit(group)
            assert role in plan.DECIDE and role in plan.REPORTED and role in b.ENV
    # K5 is decided only where its schedule differs from production (B3/B4); 0051 arms keep NDM's 0.9 limit.
    assert plan.GROUPS['k5-batch'].decision_cells == ('32k-b3', '32k-b4')
    assert b.ENV['defer'] == {'VLLM_DSV41_INDEXER_DEFER': '1'}
    assert b.ENV['ring-defer'] == {**b.ENV['dual-ring'], **b.ENV['defer']}
    assert w.cells_for('k5') == cells.CELLS and w.repeats_for('k5', '32k-b3', w.arms('k5')[1]) == cells.REPEATS['32k-b3']
    assert w.near_tie_nats('k5', w.arms('k5')[1]) == greedy_driver.NEAR_TIE_NATS
    assert [w.near_tie_nats(GROUP, a) for a in w.arms(GROUP)] == [0.9 + 1e-3, 0.9 + 1e-3, 1.1 + 1e-3, 0.9 + 1e-3]
    assert {w.near_tie_nats('defer-shares', a) for a in w.arms('defer-shares')} == {0.9 + 1e-3}
    # Group names never collide with role names (a group is admitted as a window, a role as an arm).
    assert not set(plan.GROUPS) & set(b.ARMS)


def test_fixed2_runtime_is_exactly_the_dep1_switch() -> None:
    arm = w.arms(GROUP)[2]
    assert arm.role == 'fixed2' and b.ENV['fixed2'] == {'VLLM_DSPARK_DEPTH_POLICY': 'fixed:2', 'VLLM_DSV41_ADAPTIVE_K': '0'}
    prod = t.identity(None)
    live = t.identity(arm)
    argv = shlex.split(str(live['argv']).replace('\\\n', ' '))
    argv[argv.index('--speculative-config') + 1] = b.SPEC
    live['argv'] = shlex.join(argv)
    w.runtime(prod, live, arm)
    for field, value in (('env', 'VLLM_DSV41_ADAPTIVE_K'), ('argv', None)):
        bad = copy.deepcopy(live)
        if value is None:
            bad['argv'] = t.identity(arm)['argv']  # production schedule instead of [[1,4,5]]
        else:
            record(bad['env'])[value] = '1'
        with pytest.raises(ValueError):
            w.runtime(prod, bad, arm)
    assert b.verify('fixed2')['files'] == {} and 'fixed2' in host_gates.CONFIG_ARMS


def starts(speed: Speed, *, group: str = GROUP, limits: dict[str, float] | None = None) -> dict[str, Json]:
    p = plan.plan_for(group)
    rows: dict[str, Json] = {}
    for i, role in enumerate(p.roles, 1):
        ms, acc = speed.get(role, (1., 1.))
        rows[f's{i}-{role}'] = {'status': 'COMPLETE', 'role': role, 'checks': {'clock_caps_mhz': [2300]}, 'greedy': {
            **t.greedy_result(), 'near_tie_nats': (limits or plan.NEAR_TIE_NATS)[role]},
            'cells': {name: measured(name, p.repeats[name], ms, acc, f's{i}') for name in p.cells}}
    return rows


def test_decision_per_arm_under_win2(tmp_path: Path) -> None:
    archive = tmp_path / 'archive.jsonl'
    # dual ring 3 % faster per round, acceptance unchanged; fixed2 4 % fewer ms/round with unchanged acceptance.
    result = plan.decide(GROUP, starts({'dual-ring': (.97, 1.), 'fixed2': (.96, 1.)}), archive, b.PRODUCTION)
    assert result['rule'] == 'win2-v1' and result['status'] == 'CANDIDATE'
    arms = record(result['arms'])
    for role in ('dual-ring', 'fixed2'):
        verdict = record(arms[role])
        assert verdict['status'] == 'CANDIDATE' and record(verdict['groups']) == {'single-user': 'CANDIDATE', 'multi-user': 'CANDIDATE'}
        assert record(record(verdict['cells'])['mixed150k-2decode'])['decides'] is False
    assert record(record(record(arms['dual-ring'])['cells'])['short-b1'])['primary_metric'] == 'ms_per_round'
    assert record(record(record(arms['fixed2'])['cells'])['32k-b3'])['primary_metric'] == 'committed_per_s'
    # A null is INCONCLUSIVE, never a win.
    assert plan.decide(GROUP, starts({}), archive, b.PRODUCTION)['status'] == 'INCONCLUSIVE'
    # Dual ring: an acceptance drop beyond the control spread is a LOSS even with faster rounds.
    lost = plan.decide(GROUP, starts({'dual-ring': (.97, .9)}), archive, b.PRODUCTION)
    assert record(record(lost['arms'])['dual-ring'])['status'] == 'NOT A CANDIDATE'
    # fixed2: fewer ms/round with an acceptance fall that costs committed throughput is a LOSS, not a win on ms/round.
    fixed = plan.decide(GROUP, starts({'fixed2': (.8, .7)}), archive, b.PRODUCTION)
    verdict = record(record(fixed['arms'])['fixed2'])
    assert verdict['status'] == 'NOT A CANDIDATE'
    components = record(record(verdict['components_reported'])['short-b1'])
    assert float(cast(float, record(components['ms_per_round'])['point_gain'])) > .15
    # Greedy evidence must carry each role's registered near-tie limit.
    with pytest.raises(ValueError, match='registered near-tie'):
        plan.decide(GROUP, starts({}, limits={**plan.NEAR_TIE_NATS, 'fixed2': 0.9 + 1e-3}), archive, b.PRODUCTION)
    partial = starts({})
    record(partial['s4-control'])['status'] = 'STARTED'
    assert plan.decide(GROUP, partial, archive, b.PRODUCTION)['status'] == 'NOT A CANDIDATE'
    # 0051 alone and with the ring, in one window: each judged on round speed against the same controls.
    defer = plan.decide('defer-shares', starts({'defer': (.99, 1.), 'ring-defer': (.97, 1.)}, group='defer-shares'), archive, b.PRODUCTION)
    assert {k: record(v)['status'] for k, v in record(defer['arms']).items()} == {'defer': 'INCONCLUSIVE', 'ring-defer': 'CANDIDATE'}
    # K5 at B3/B4: more accepted per round at a higher round cost; committed throughput decides.
    k5 = plan.decide('k5-batch', starts({'k5': (1.1, 1.25)}, group='k5-batch'), archive, b.PRODUCTION)
    verdict = record(record(k5['arms'])['k5'])
    assert verdict['status'] == 'CANDIDATE' and record(verdict['groups']) == {'multi-user': 'CANDIDATE'}
    assert record(k5['aa_calibration'])['interval95'] == [-0.085, -0.004]


@pytest.mark.parametrize('group,speed,status', [
    ('ring-depth2', {'dual-ring': (.97, 1.), 'fixed2': (.96, 1.)}, {'dual-ring': 'CANDIDATE', 'fixed2': 'CANDIDATE'}),
    ('defer-shares', {'ring-defer': (.97, 1.)}, {'defer': 'INCONCLUSIVE', 'ring-defer': 'CANDIDATE'}),
    ('k5-batch', {'k5': (1.1, 1.25)}, {'k5': 'CANDIDATE'}),
])
def test_full_window(rig: Rig, group: str, speed: Speed, status: dict[str, str]) -> None:
    p = plan.plan_for(group)
    live, fake = rig(group=group, speed=speed)
    result = w.window(live, group, lambda row: None)
    assert 'stop' not in result, result.get('stop')
    assert result['status'] == 'COMPLETE' and str(record(result['restore'])['status']).startswith('RESTORED')
    assert list(record(result['starts'])) == [a.name for a in w.arms(group)]
    decision = record(result['decision'])
    assert decision['rule'] == 'win2-v1' and {k: record(v)['status'] for k, v in record(decision['arms']).items()} == status
    for a in w.arms(group):
        argv = fake.argvs[f'{a.name}-greedy']
        assert float(argv[argv.index('--near-tie-nats') + 1]) == plan.NEAR_TIE_NATS[a.role]
        for name in p.cells:
            argv = fake.argvs[f'{a.name}-{name}']
            assert argv[argv.index('--repeats') + 1] == str(p.repeats[name])
        gates = json.dumps(record(record(result['starts'])[a.name])['gates'])
        # Only ring roles carry host RDMA evidence; every other role is config-only.
        assert ('rdma' in gates) == (a.role in ('dual-ring', 'ring-defer')), (a.role, gates[:200])
    # Every control repeat of both decision cells is archived at 512 tokens.
    rows = dv2.archive_rows(pf.CAMPAIGN / 'control-archive.jsonl')
    per_start = sum(p.repeats[c] for c in p.decision_cells) * len(plan.ARCHIVE_METRICS)
    assert len(rows) == 2 * per_start and {r['max_tokens'] for r in rows} == {512}
    assert {r['arm'] for r in rows} == {w.arms(group)[0].name, w.arms(group)[-1].name}
    assert record(result['archive'])['rows'] == 2 * per_start
    # WIN2-R2 provenance (schema v3): production commit, the row's own start's proven cap from its start log, YYYYMMDD.
    assert {(r['commit'], r['clock_cap_mhz'], r['clock_source']) for r in rows} == {(b.PRODUCTION, 2300, 'start-log')}
    assert all(len(str(r['date'])) == 8 and str(r['date']).isdigit() for r in rows)
    heads = record(result['archive'])
    assert number(record(heads['head_after'])['rows']) == number(record(heads['head_before'])['rows']) + len(rows)
    # Depth 2's greedy call may run past GREEDY_S (near-tie diagnostics); time alone never stops it.
    assert {k: v for k, v in fake.timeouts.items() if k.endswith('-greedy') and 'fixed2' in k} in (
        {}, {'s3-fixed2-greedy': w.FIXED2_GREEDY_S}) and w.FIXED2_GREEDY_S > w.GREEDY_S
    # reanalyze (WIN2-R2) keeps every candidate role apart: one arm per role, never pooled.
    (live.out / 'summary.json').write_text(json.dumps(result))
    audit = w.reanalyze_arms(live.out, group)
    roles = [a.role for a in w.arms(group)[1:-1]]
    assert (set(record(audit['arms'])) == set(roles)) if len(roles) > 1 else audit['arm'] == roles[0]
    assert {r['start'] for r in rows} == {r['arm'] for r in rows} and {r['repeat'] for r in rows} >= {0, 1, 2}


def test_stopped_window_archives_only_finished_controls(rig: Rig) -> None:
    live, fake = rig(fail='STOP')
    result = w.window(live, GROUP, lambda row: None)
    assert 'stop' in result and record(result['decision'])['status'] == 'NOT A CANDIDATE'
    assert record(result['archive'])['rows'] == 0


def test_chained_ring_depth2_window(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    live, fake = rig()
    result = w.window(live, GROUP, lambda row: None)
    assert 'stop' not in result, result.get('stop')
    assert result['chain'] == 'CHAINED' and str(record(result['restore'])['status']).startswith('CHAINED: production left down')
    assert fake.restores == [] and chain.read_marker() is not None


def test_cells_cli_registers_limits_and_repeats(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def run(*args: str) -> None:
        monkeypatch.setattr(sys, 'argv', ['cells', '--output', str(tmp_path / 'x.json'), '--data-dir', str(tmp_path),
                                          '--deadline', '0', *args])
        cells.main()
    run('--mode', 'greedy', '--near-tie-nats', repr(1.1 + 1e-3))
    run('--mode', 'cell', '--cell', '32k-b3', '--repeats', '4')
    for bad in (('--mode', 'greedy', '--near-tie-nats', '0.7'), ('--mode', 'cell', '--near-tie-nats', repr(0.9 + 1e-3)),
                ('--mode', 'greedy', '--repeats', '3'), ('--mode', 'cell', '--cell', 'short-b1', '--repeats', '13')):
        with pytest.raises(ValueError):
            run(*bad)
    assert not (tmp_path / 'x.json').exists()


def test_window_env_fixed2_is_the_dep1_switch(tmp_path: Path) -> None:
    prod = tmp_path / 'dsv41.env'
    prod.write_bytes(b.production_file('dsv41.env'))
    script = 'UP3_ARM=$1 UP3_PROD_ENV=$2; source offload/up3/window.env; printf "%s\\n" "$EXP_DOCKER_ENV" "${SPEC_OVERRIDE-}" "$OVERLAYS"'
    control, fixed2 = (subprocess.check_output(['bash', '-eu', '-c', script, 'up3', arm, str(prod)], cwd=b.ROOT, text=True)
                       .splitlines() for arm in ('control', 'fixed2'))
    assert fixed2[0] == control[0] + ' VLLM_DSPARK_DEPTH_POLICY=fixed:2 VLLM_DSV41_ADAPTIVE_K=0'
    assert fixed2[1] == b.SPEC and fixed2[2] == '' and control[1] == ''


def test_clock_cap_change_or_absence_refuses_the_decision(tmp_path: Path) -> None:
    archive = tmp_path / 'archive.jsonl'
    rows = starts({'dual-ring': (.97, 1.)})
    record(record(rows['s3-fixed2'])['checks'])['clock_caps_mhz'] = [2150]   # THERMAL-WATCH dial-back mid-window
    assert 'clock cap' in str(plan.decide(GROUP, rows, archive, b.PRODUCTION)['reason'])
    record(record(rows['s3-fixed2'])['checks'])['clock_caps_mhz'] = []
    assert plan.decide(GROUP, rows, archive, b.PRODUCTION)['status'] == 'NOT A CANDIDATE'
    # Archive spreads never pool 2150 MHz rows (before NDM's 2300 MHz ceiling) with this window's 2300 MHz cell.
    dv2.archive_append(archive, [{'window': f'old{i}', 'cell': 'short-b1', 'metric': 'ms_per_round', 'value': 40. + i,
                                  'max_tokens': 512, 'commit': b.PRODUCTION, 'clock_cap_mhz': 2150, 'date': '20261027',
                                  'start': 's1-control', 'repeat': 0} for i in range(3)])
    result = plan.decide(GROUP, starts({}), archive, b.PRODUCTION)
    assert result['clock_caps_mhz'] == [2300] and record(result['archive_spread_512'])['short-b1/ms_per_round'] is None


def test_live_records_the_applied_clock_cap(rig: Rig) -> None:
    live, fake = rig(speed={'dual-ring': (.97, 1.)})
    fake.cap = 2150
    result = w.window(live, GROUP, lambda row: None)
    assert record(record(record(result['starts'])['s2-dual-ring'])['checks'])['clock_caps_mhz'] == [2150]
    assert record(result['decision'])['clock_caps_mhz'] == [2150]
    assert {json.dumps(r['clock_caps_mhz']) for r in dv2.archive_rows(pf.CAMPAIGN / 'control-archive.jsonl')} == {'[2150]'}
