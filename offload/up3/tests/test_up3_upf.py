"""R-UPF groups in the UP3 runner (upf-topk, upf-0091; upf-w1 refused): T0-T2, C1-relative identity, needles, soak,
C2 KL, arm-level stops, no-loss verdicts and archive rows; CPU fakes only."""
from __future__ import annotations

import copy
import json
import math
import os
import shlex
import subprocess
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import cast

import pytest

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import decision_v2 as dv2, pf1_window as pf, up3_window as w
from offload.launch.windows import nbw_cells as greedy_driver
from offload.launch.windows import up2_request
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, record
from offload.up3 import build as b, cells, upf_driver, upf_plan
from offload.up3.tests import test_up3_window as t
from offload.up3.tests import test_up3_win2 as v2
from offload.upfix import configs as uc, windows as uw

# UPF.env starts from D-kv6-async-pack.env, not production's dsv41.env: the fake C1 differs from production by one key.
C1_ENV = {'VLLM_FIXTURE_K1_BASE': '1'}
LIBRARY = 'f' * 64


class Remote(v2.Remote):
    fault = ''

    def identity(self, rank: int) -> dict[str, Json]:
        arm = cast(w.Arm | None, self.arm)
        prod = t.identity(None, rank)
        if arm is None:
            return prod
        live = copy.deepcopy(prod)
        live['id'] = arm.name + str(rank) + str(self.serial)
        argv = shlex.split(str(live['argv']).replace('\\\n', ' '))
        argv[argv.index('--port') + 1] = '8001'
        live['argv'] = shlex.join(argv)
        record(live['env']).update({**C1_ENV, **upf_plan.env_delta(arm.role)})
        if self.fault == 'drift' and arm.role == 'upf-0091':
            record(live['env'])['VLLM_UNDOCUMENTED'] = '1'
        record(live['mounted']).update(upf_plan.mount_delta(arm.role))
        return live

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        if label.startswith('upf-image-bases-'):
            assert stdin == pf.IMAGE_BASES
            bases = upf_plan.image_bases(self.live.group)
            return json.dumps({k: ('0' * 64 if self.fault == 'image-base' else v) for k, v in bases.items()})
        if label == 'upf-t0':
            assert './stop.sh' in command and 'nvidia-smi --query-compute-apps' in command
            return '' if self.fault == 't0' else w.UPF_T0_IDLE + '\n'
        if label == 'upf-t1':
            assert 'offload.upfix.topk.jit_build' in command
            return 'UPF T1 RC 1\n' if self.fault == 't1' else 'UPF T1 RC 0\n'
        if label == 'upf-t2':
            assert 'upfix-t2' in command and '--gpus all' in command
            return {'t2-timeout': 'UPF T2 RC 124\n', 't2-fail': 'UPF T2 RC 2\n'}.get(self.fault, 'UPF T2 RC 0\n')
        if label == 'upf-receipts':
            receipt = {'library_sha256': LIBRARY, 'arch': '12.1'}
            compare = {'library_sha256': LIBRARY, 'verdict': 'PASS', 'complete': True, 'cases': 149, 'cases_done': 149,
                       'capability': [12, 1]}
            if self.fault == 't2-timeout':
                compare.update(complete=False, cases_done=140, verdict=None)
            verify = 1 if self.fault in ('verify', 't2-timeout', 't2-fail') else 0
            return (f'UPF RECEIPT receipt\n{json.dumps(receipt)}\nUPF RECEIPT bitcompare\n{json.dumps(compare)}\n'
                    f'UPF_TOPK start gate: ...\nUPF VERIFY RC {verify}\n')
        if '-sources-' in label and label.split('-sources-')[0] in uc.ARMS:
            return json.dumps(self.live.upf_pins(label.split('-sources-')[0]))
        if label.endswith('-start') and 'start_exp.sh' in command:
            arm = next(a for a in w.arms(self.live.group) if label == a.name + '-start')
            assert f'{upf_plan.start_env(arm.role)} timeout --foreground' in command and uc.CONFIG in command
        return super().remote(label, host, command, stdin)

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        cell = argv[argv.index('--cell') + 1] if '--cell' in argv else ''
        if argv[1:3] == ['-m', 'offload.up3.cells'] and cell in cells.WC_CELLS:
            self.sup.poll()
            self.commands.append(label)
            self.argvs[label], self.timeouts[label] = argv, timeout
            arm = cast(w.Arm, self.arm)
            ms, acc = self.speed.get(arm.role, (1., 1.))
            Path(argv[argv.index('--output') + 1]).write_text(json.dumps(
                v2.measured(cell, int(argv[argv.index('--repeats') + 1]), ms, acc, arm.name)))
            self.rounds += 10
            return ''
        if argv[1:3] != ['-m', 'offload.up3.upf_driver']:
            return super().call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)
        self.sup.poll()
        self.commands.append(label)
        self.argvs[label], self.timeouts[label] = argv, timeout
        arm = cast(w.Arm, self.arm)
        mode = argv[argv.index('--mode') + 1]
        output = Path(argv[argv.index('--output') + 1])
        data: dict[str, Json]
        if mode == 'greedy':
            data = {**t.greedy_result(), 'near_tie_nats': uc.GREEDY_NEAR_TIE_NATS,
                    'greedy_registration': upf_driver.REGISTRATION}
            checks = cast(list[Json], data['checks'])
            if self.fault == 'c1-near-tie' and arm.name == 'C1':
                record(checks[2]).update(identical=False, near_tie=True)
                data.update(identical=5, near_tie=1)
            if self.fault == 'cand-greedy' and not upf_plan.is_control(arm.role):
                record(checks[3])['identical'] = False
                data.update(identical=5, near_tie=0)
        elif mode == 'kl':
            data = {'rows': [], 'compared': 0, 'bound': float(argv[argv.index('--bound') + 1]), 'pass': self.fault != 'kl'}
        elif mode == 'needle':
            target = int(argv[argv.index('--target') + 1])
            ok = not (self.fault == 'needle-300k' and target == 300000)
            data = {'target': target, 'pass': ok, **({'recall': 6, 'of': 6, 'foreign_codes': 0} if ok else
                                                      {'error': 'ValueError: STOP: needle6/6/foreign-code failure'})}
        else:
            assert mode == 'soak' and inflight == upf_driver.SOAK_STREAMS
            grown = 400. if self.fault == 'soak' and not upf_plan.is_control(arm.role) else 3.
            data = {'growth_mib': {'sparkhead:vllm serve': grown, 'sparkworker-wired:vllm worker': 1., upf_driver.SHM_ROLE: 0.},
                    'requests': 300, 'samples': [{}] * 61, 'transient_roles': []}
            self.rounds += 10
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
        out = tmp_path / f'UP3-{group}-W-20261027T230000Z'
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


def run(live: w.Live, group: str) -> dict[str, Json]:
    return w.window(live, group, lambda row: None)


def starts(result: dict[str, Json]) -> dict[str, Json]:
    return record(result['starts'])


def verdict(result: dict[str, Json], label: str) -> str:
    return str(record(record(record(result['decision'])['arms'])[label])['verdict'])


def test_upf_groups_follow_the_design_and_the_hold_ceiling() -> None:
    assert upf_plan.QUEUE == ('upf-topk', 'upf-0091')
    for group, design in upf_plan.GROUPS.items():
        window = uw.WINDOWS[design]
        assert [(a.name, a.role) for a in w.arms(group)] == [(s.label, s.arm) for s in window.starts]
        assert w.cells_for(group) == uw.DECISION_CELLS
        assert all(a.config == uc.CONFIG for a in w.arms(group))
        for i, start in enumerate(window.starts):
            for step in start.steps:
                if isinstance(step, uw.CellStep) and step.cell in uw.DECISION_CELLS:
                    assert w.repeats_for(group, step.cell, w.arms(group)[i]) == step.repeats
            assert upf_plan.gates(group, i) == tuple(s.kind for s in start.steps if s.kind.startswith('needle'))
        assert {w.near_tie_nats(group, a) for a in w.arms(group)} == {0.9 + 1e-3}
    # Coordinator 2026-10-27: each group's own bounds from its budget, restore <= HOLD+132 with a >= 10-min reserve.
    assert w.hold_bounds('upf-topk') == (108 * 60, 108 * 60, 120 * 60)
    assert w.hold_bounds('upf-0091') == (110 * 60, 110 * 60, 122 * 60)
    for group in upf_plan.QUEUE:
        submit, abort, restore = w.hold_bounds(group)
        assert upf_plan.hold_s(group) <= submit == abort and restore - submit >= 600 and restore <= 132 * 60
        assert w.budget_minutes(group) <= restore / 60
        w.admit(group)
        assert 'a SINGLE window' in w.plan(group) and f'HOLD+{restore // 60}' in w.plan(group)
    assert upf_plan.pre('upf-topk') == ('stop_idle', 'jit_build', 'bitcompare') and upf_plan.pre('upf-0091') == ()
    # The combined window cannot restore production by HOLD+132: refused before any host access, everywhere.
    with pytest.raises(ValueError, match='HOLD\\+176 > the HOLD\\+132 ceiling'):
        w.admit('upf-w1')
    assert 'REFUSED' in w.plan('upf-w1')
    assert w.main(['--group', 'upf-w1']) == 0   # the dry plan still prints


def test_upf_w1_execute_refuses_before_any_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(pf, 'CAMPAIGN', tmp_path)
    out = tmp_path / 'UP3-upf-w1-W-20261027T230000Z'
    assert w.main(['--execute', '--group', 'upf-w1', '--out', str(out)]) == 1
    assert 'HOLD+132 ceiling' in capsys.readouterr().err and not out.exists()


def test_t_scripts_carry_the_window_md_commands() -> None:
    t1, t2 = w.upf_t1('dsv41-deploy-up3w2'), w.upf_t2('dsv41-deploy-up3w2')
    for script, name in ((t1, 'upfix-t1'), (t2, 'upfix-t2')):
        assert f'timeout -k 30 600 docker run --rm --name {name} --pull never --network none' in script
        assert f'case $rc in 124|137) docker rm -f {name} >/dev/null 2>&1 || true ;; esac' in script
        assert '--user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/opt/repo:ro" -v "$PWD/$B:/opt/repo/$B"' in script
    assert '--cpus 16' in t1 and '--gpus all' not in t1 and pf.IMAGE in t1
    assert 'rm -rf $B/jit $B/receipt.json $B/bitcompare.json' in t1
    assert '--gpus all' in t2 and '--library $B/jit/dsv41_topk.so --out $B/bitcompare.json' in t2
    assert './stop.sh' in w.upf_t0('r') and 'nvidia-smi --query-compute-apps' in w.upf_t0('r')
    assert 'python3 -m offload.upfix.configs verify-topk' in w.upf_receipts('r')
    for script in (w.upf_t0('r'), t1, t2, w.upf_receipts('r')):
        assert subprocess.run(['bash', '-n', '-c', script]).returncode == 0


@pytest.mark.parametrize('group', upf_plan.QUEUE)
def test_full_upf_window(rig: Rig, group: str) -> None:
    live, fake = rig(group)
    result = run(live, group)
    assert 'stop' not in result, result.get('stop')
    assert result['status'] == 'COMPLETE' and fake.restores == ['start', 'check', 'close']
    candidate = next(s for s in upf_plan.starts(group) if not upf_plan.is_control(s.arm))
    decision = record(result['decision'])
    assert decision['status'] == f'PROMOTABLE: {candidate.label}' and verdict(result, candidate.label) == 'PROMOTABLE (no loss)'
    assert decision['rule'] == 'win2-v1' and decision['rule_revision'] == 2 and decision['registration'] == 'upf-0p9'
    assert all(v is True for v in record(record(record(decision['arms'])[candidate.label])['gates']).values())
    order = fake.commands
    if group == 'upf-topk':
        # T0 -> T1 -> T2 -> receipts, all before C1's start; the library sha is pinned for X-topk's installed bytes.
        assert order.index('upf-t0') < order.index('upf-t1') < order.index('upf-t2') < order.index('upf-receipts') \
            < order.index('C1-start')
        assert record(record(result['pre'])['topk'])['status'] == 'READY' and live.upf_library == LIBRARY
        assert live.upf_pins('topk-fix')[uc.TOPK_SO_CONTAINER] == LIBRARY
        assert [order.index(f'X-topk-{k}') for k in ('needle-150k', 'needle-300k')] == sorted(
            order.index(f'X-topk-{k}') for k in ('needle-150k', 'needle-300k'))
    else:
        assert 'upf-t0' not in order and 'pre' not in result
        # The soak follows the cells (OFF in C1, ON in X-0091) and the candidate passes soak_ok.
        for label in ('C1', 'X-0091'):
            assert order.index(f'{label}-ctl-32k-b4') < order.index(f'{label}-soak')
        assert record(record(starts(result)['X-0091'])['checks'])['soak'] == {
            'growth_mib': {'sparkhead:vllm serve': 3., 'sparkworker-wired:vllm worker': 1., upf_driver.SHM_ROLE: 0.},
            'requests': 300, 'samples': 61, 'transient_roles': [], 'mode': 'ON', 'ok': True}
    # Greedy through the UPF driver at 0.9 nats; C2's step is the KL re-probe of the candidate's recorded divergences.
    assert all(f'{s.label}-greedy' in fake.argvs for s in upf_plan.starts(group)[:-1]) and 'C2-greedy' not in fake.argvs
    kl = fake.argvs[f'C2-kl-{candidate.label}']
    assert kl[kl.index('--candidate') + 1].endswith(f'{candidate.label}-greedy.json') and kl[kl.index('--bound') + 1] == '0.005'
    # Decision cells at 512 tokens with the design's repeats; identity relative to C1 with the C1 diff recorded.
    for i, s in enumerate(upf_plan.starts(group)):
        for c in uw.DECISION_CELLS:
            argv = fake.argvs[f'{s.label}-{c}']
            assert argv[argv.index('--repeats') + 1] == str(upf_plan.repeats(group, i, c))
    diff = record(record(record(starts(result)['C1'])['gates'])['c1_vs_production'])
    assert {h: record(diff[h])['env_added'] for h in pf.HOSTS} == {h: sorted(C1_ENV) for h in pf.HOSTS}
    assert all(record(diff[h])['argv_equal_but_port'] is True for h in pf.HOSTS)
    assert json.loads((live.out / 'upf-image-bases.json').read_text())['status'] == 'MATCH'
    # Archive (main() after summary.json): every C1/C2 repeat, WIN2-R2 v3 provenance, at 512 tokens.
    assert record(result['archive'])['status'] == 'DEFERRED'
    (live.out / 'summary.json').write_text(json.dumps({'production': b.PRODUCTION}))
    archived = w.upf_archive(live.out, group, result)
    rows = dv2.archive_rows(pf.CAMPAIGN / 'control-archive.jsonl')
    assert archived['rows'] == len(rows) == 2 * sum(2 * len(uw.METRICS[c]) for c in uw.DECISION_CELLS)
    assert {r['start'] for r in rows} == {'C1', 'C2'} and {r['cell'] for r in rows} == set(uw.DECISION_CELLS)
    assert all(r['max_tokens'] == 512 and r['commit'] == b.PRODUCTION and r['clock_cap_mhz'] == 2300
               and r['clock_source'] == 'start-log' and r['date'] == '20261027' for r in rows)


@pytest.mark.parametrize('fault,reason', [
    ('t1', 'T1 jit_build rc 1'),
    ('t2-timeout', 'T2 bit-compare rc 124 (TIMED OUT)'),
    ('t2-fail', 'T2 bit-compare rc 2 (FAIL)'),
    ('verify', 'verify-topk refused the receipts (rc 1)'),
])
def test_t1_t2_failure_skips_only_x_topk(rig: Rig, fault: str, reason: str) -> None:
    live, fake = rig('upf-topk', fault=fault)
    result = run(live, 'upf-topk')
    assert 'stop' not in result, result.get('stop')
    assert record(starts(result)['X-topk']) == {'status': 'SKIPPED', 'role': 'topk-fix', 'reason': reason}
    assert record(starts(result)['C1'])['status'] == record(starts(result)['C2'])['status'] == 'COMPLETE'
    assert 'X-topk-start' not in fake.commands and ('upf-t2' in fake.commands) == (fault != 't1')
    assert verdict(result, 'X-topk') == f'SKIPPED: {reason}' and record(result['decision'])['status'] == 'NO PROMOTABLE ARM'
    assert fake.restores == ['start', 'check', 'close'] and not live.upf_library


@pytest.mark.parametrize('group,fault,label,expected', [
    ('upf-topk', 'needle-300k', 'X-topk', 'STOP: needle-300k'),
    ('upf-0091', 'cand-greedy', 'X-0091', 'STOP: greedy (not run: needle-150k, soak)'),
    ('upf-topk', 'kl', 'X-topk', 'STOP: greedy'),
    ('upf-0091', 'soak', 'X-0091', 'STOP: soak'),
])
def test_candidate_gate_failure_is_an_arm_stop(rig: Rig, group: str, fault: str, label: str, expected: str) -> None:
    live, fake = rig(group, fault=fault)
    result = run(live, group)
    assert 'stop' not in result, result.get('stop')
    assert result['status'] == 'COMPLETE' and record(starts(result)['C2'])['status'] == 'COMPLETE'
    assert verdict(result, label) == expected and record(result['decision'])['status'] == 'NO PROMOTABLE ARM'
    row = record(starts(result)[label])
    if fault in ('needle-300k', 'cand-greedy'):
        # The arm stops at its gate: no cells for it, and the window goes on to C2.
        assert row['status'] == 'STOPPED' and row['failed_gate'] == expected.split()[1] and f'{label}-ctl-150k-b1' not in fake.commands
    else:
        assert row['status'] == 'COMPLETE'
    assert fake.restores == ['start', 'check', 'close']


@pytest.mark.parametrize('group,fault,message,restores', [
    ('upf-topk', 'c1-near-tie', 'greedy is not 6/6 IDENTICAL to the K1 reference', ['start', 'check', 'close']),
    ('upf-0091', 'drift', 'container env differs from C1', ['start', 'check', 'close']),
    ('upf-topk', 't0', 'GPU-idle check on both Sparks unproved', ['start', 'check', 'close']),
    ('upf-topk', 'image-base', 'differ from their manifests', ['check', 'close']),
])
def test_control_or_identity_failure_stops_the_window(rig: Rig, group: str, fault: str, message: str,
                                                       restores: list[str]) -> None:
    live, fake = rig(group, fault=fault)
    result = run(live, group)
    assert message in str(result.get('stop')), result.get('stop')
    assert fake.restores == restores and str(record(result['restore'])['status']).startswith('RESTORED')
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'


def test_a_loss_is_not_a_candidate(rig: Rig) -> None:
    live, _ = rig('upf-topk', speed={'topk-fix': (1.05, 1.)})
    result = run(live, 'upf-topk')
    assert 'stop' not in result, result.get('stop')
    assert verdict(result, 'X-topk') == 'NOT A CANDIDATE'


def test_wc_cells_run_at_512_tokens_with_explicit_repeats() -> None:
    assert cells.keys_for('ctl-150k-b1') == ('150k',) and cells.keys_for('ctl-32k-b4') == ('32k-0', '32k-1', '32k-2', '32k-3')
    assert cells.ESTIMATES['ctl-150k-b1'] == cells.ESTIMATES['ctl-32k-b4'] == 180 and cells.BUDGET == uw.MAX_TOKENS
    assert [cells.keys_for(n) for n in cells.CELLS[:4]] == [('short',), ('32k-0',), ('32k-0', '32k-1', '32k-2'),
                                                            ('32k-0', '32k-1', '32k-2', '32k-3')]
    with pytest.raises(ValueError, match='explicit repeats'):
        cells.repeated(cast(wc.Api, None), 'ctl-32k-b4', {}, deadline=math.inf, salt='s', ledger=Path('/dev/null'),
                       output=Path('/dev/null'))


def test_soak_roles_and_growth() -> None:
    text = ('  1  2048 vllm serve --port 8001\n 40  1024 python3 -c worker\n 41  4096 python3 -c worker\n'
            ' 99   512 ps -eo pid=,rss=,args=\n')
    assert upf_driver.process_roles('sparkhead', text) == {
        'sparkhead:vllm serve --port 8001': 2., 'sparkhead:python3 -c worker#0': 1., 'sparkhead:python3 -c worker#1': 4.}
    samples = [{'a': float(i), 'b': 5.} for i in range(10)] + [{'a': 9., 'b': 5., 'c': 1.}]
    assert upf_driver.growth(samples) == ({'a': 8., 'b': 0.}, ['c'])
    with pytest.raises(ValueError, match='at least 6'):
        upf_driver.growth(samples[:5])
    assert uw.soak_ok({'a': 1., 'b': 0.}, {'a': 16., 'b': 0.}) and not uw.soak_ok({'a': 1.}, {'a': 17.})


class FakeApi:
    def __init__(self, fail: bool = False) -> None:
        self.fail, self.calls, self.lock = fail, 0, threading.Lock()

    def model(self) -> dict[str, Json]:
        return {'id': 'm'}

    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        with self.lock:
            self.calls += 1
        assert body['max_tokens'] == 512 and ':soak:' in str(body['cache_salt'])
        if self.fail:
            raise OSError('connection reset')
        return {'usage': {'completion_tokens': 512, 'prompt_tokens': 1000, 'prompt_tokens_details': {'cached_tokens': 0}},
                'finish': 'length'}


def test_soak_drives_four_streams_and_samples() -> None:
    now = [0.]
    def clock() -> float:
        return now[0]
    def sleep(seconds: float) -> None:
        now[0] += seconds
    def sample() -> dict[str, float]:
        return {'sparkhead:vllm serve': 100. + now[0] / 100}
    api = FakeApi()
    result = upf_driver.soak(cast(wc.Api, api), ('short prompt', 1000), seconds=60, salt='s', sample=sample,
                             clock=clock, sleep=sleep)
    assert result['streams'] == 4 and len(cast(list[Json], result['samples'])) == 7
    assert record(result['growth_mib'])['sparkhead:vllm serve'] == pytest.approx(.4)
    with pytest.raises(ValueError, match='soak load failed: stream'):
        now[0] = 0.
        upf_driver.soak(cast(wc.Api, FakeApi(fail=True)), ('p', 1000), seconds=60, salt='s',
                        sample=sample, clock=clock, sleep=lambda s: None)


def test_remote_sample_reads_both_sparks_and_shm() -> None:
    seen: list[Sequence[str]] = []
    def runner(argv: Sequence[str]) -> str:
        seen.append(argv)
        if 'df' in argv[-1]:
            return '     Used\n1048576\n'
        return '7 2048 vllm serve\n'
    sample = upf_driver.remote_sample(runner)
    assert sample == {'sparkhead:vllm serve': 2., 'sparkworker-wired:vllm serve': 2., upf_driver.SHM_ROLE: 1.}
    assert [a[-2] for a in seen] == [*pf.HOSTS, pf.HOSTS[0]] and all(a[0] == 'ssh' for a in seen)


@pytest.mark.parametrize('message,gate', [('needle6/6/foreign-code failure', True),
                                          ('needle did not complete normally', True),
                                          ('needle submission deadline 1.0 reached', False)])
def test_needle_gate_failures_are_verdicts_and_others_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           message: str, gate: bool) -> None:
    def needle(*args: object, **kwargs: object) -> None:
        raise ValueError('STOP: ' + message)
    monkeypatch.setattr(up2_request, 'needle', needle)
    import argparse
    args = argparse.Namespace(output=tmp_path / 'n.json', salt='s', deadline=math.inf, target=300000,
                              ledger=tmp_path / 'l.jsonl', data_dir=tmp_path)
    if gate:
        assert upf_driver.run('needle', cast(cells.Http, None), args) == {
            'target': 300000, 'pass': False, 'error': 'ValueError: STOP: ' + message}
    else:
        with pytest.raises(ValueError, match='deadline'):
            upf_driver.run('needle', cast(cells.Http, None), args)
