"""UPX-0085B runner, CPU only: pure gates, the request driver over a fake server, and the real Live/Supervisor over the
NBW/PF1 rehearsal fakes (no SSH, HTTP, Spark, x86host or GPU). Every stop must end in the strict W8 restore; a clean
chained, non-last window must leave production down with its own HOLD closed and the chain marker written."""
from __future__ import annotations

import datetime
import json
import shlex
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import chain
from offload.launch.windows import decision_v2 as dv
from offload.launch.windows import nbw_cells as cells
from offload.launch.windows import nbw_window as n
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import scoped_remote as owner
from offload.launch.windows.tests import test_nbw_rehearsal as rehearsal_base
from offload.launch.windows.tests import test_nbw_window as pure
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, record
from offload.upx import battery_cells as bc
from offload.upx import battery_window as bw
from offload.upx import build_0085b as b
from offload.upx import configs
from offload.upx import measure_0085b as m
from offload.upx.tests import test_measure_0085b as tm

MANIFEST = json.loads(b.MANIFEST.read_text())


# ---------------------------------------------------------------- design, pure gates

def test_arms_budgets_and_schedule_come_from_the_window_design() -> None:
    assert [(a.name, a.role) for a in bw.ARMS] == [('C1', 'C'), ('X-release', 'X'), ('C2', 'C')]
    assert bw.ARMS[0].env == configs.ARMS['C-prod'] and bw.ARMS[1].env == configs.ARMS['release-file-prod']
    assert all(a.config == configs.CONFIG and a.cap == 327680 for a in bw.ARMS)
    assert bw.schedule(bw.ARMS[0]) == [('image', 0)] * 2
    assert bw.schedule(bw.ARMS[1]) == [('on', 131072), ('off', 0)] * 3
    assert {k: round(v / 60, 1) for k, v in bw.BUDGET_S.items()} == {'C1': 15.5, 'X-release': 29.1, 'C2': 16.5}
    assert sum(bw.BUDGET_S.values()) < n.SUBMIT_S, 'every mandatory block fits before the 72-minute submission bound'
    for path in bw.PIN_FILES:
        assert (pf.ROOT / path).is_file(), path


def identity(port: str = '8001', rank: int = 0) -> dict[str, Json]:
    row = pure.production(rank)
    row['argv'] = str(row['argv']).replace('--port 8000', f'--port {port}')
    return row


def x_identity(rank: int = 0) -> dict[str, Json]:
    row = identity('8001', rank)
    record(row['mounted']).update({r['container_path']: r['overlay_sha256'] for r in MANIFEST['files']})
    record(row['env'])['VLLM_DSV41_RELEASE_FILE'] = configs.RELEASE_FILE_CONTAINER
    return row


@pytest.mark.parametrize('from_port', ['8000', '8001'])
def test_runtime_identity_equals_the_baseline_except_port_and_the_declared_0085b_change(from_port: str) -> None:
    base_row = identity(from_port)
    bw.upx_runtime(base_row, identity(), bw.ARMS[0], from_port=from_port)
    bw.upx_runtime(base_row, x_identity(), bw.ARMS[1], from_port=from_port)
    with pytest.raises(ValueError, match='mounted engine bytes'):
        bw.upx_runtime(base_row, x_identity(), bw.ARMS[0], from_port=from_port)
    with pytest.raises(ValueError, match='mounted engine bytes'):
        bw.upx_runtime(base_row, identity(), bw.ARMS[1], from_port=from_port)
    drift = x_identity()
    record(drift['env'])['VLLM_DSV41_RELEASE_TOKENS'] = '131072'
    with pytest.raises(ValueError, match='env drift'):
        bw.upx_runtime(base_row, drift, bw.ARMS[1], from_port=from_port)
    argv = identity()
    argv['argv'] = str(argv['argv']).replace('--max-num-seqs 4', '--max-num-seqs 8')
    with pytest.raises(ValueError, match='argv differs'):
        bw.upx_runtime(base_row, argv, bw.ARMS[0], from_port=from_port)
    engram = identity()
    engram['engram_identity'] = {'shim': '0' * 64}
    with pytest.raises(ValueError, match='engram_identity'):
        bw.upx_runtime(base_row, engram, bw.ARMS[0], from_port=from_port)


def test_in_container_files_are_overlay_in_x_and_image_in_c() -> None:
    def lines(key: str) -> str:
        return ''.join(f"{r[key]}  {r['container_path']}\n" for r in MANIFEST['files'])
    bw.in_container(lines('overlay_sha256'), bw.ARMS[1])
    bw.in_container(lines('base_sha256'), bw.ARMS[0])
    with pytest.raises(ValueError, match='overlay'):
        bw.in_container(lines('base_sha256'), bw.ARMS[1])
    with pytest.raises(ValueError, match='image'):
        bw.in_container(lines('overlay_sha256'), bw.ARMS[0])


def synthetic_starts() -> dict[str, Json]:
    t = 1_800_000_000.0
    on, off = [], []
    for i in range(3):
        on.append(tm.result('on', t, idle_gain=1_800_000, low_gain=500_000))
        off.append(tm.result('off', t + 400))
        t += 800
    x = [r for pair in zip(on, off) for r in pair]
    image = [tm.result('image', t), tm.result('image', t + 400)]
    ok: dict[str, Json] = {'ok': True}
    return {'C1': {'status': 'COMPLETE', 'greedy': ok, 'batteries': cast(Json, image), 'date': '20261028'},
            'X-release': {'status': 'COMPLETE', 'greedy': ok, 'batteries': cast(Json, x),
                          'checks': {'release_census': {'ok': True}}, 'date': '20261028'},
            'C2': {'status': 'COMPLETE', 'greedy': ok, 'kl': {'pass': True}, 'batteries': cast(Json, image),
                   'date': '20261028'}}


def test_decision_gates_verdict_and_archive_rows(tmp_path: Path) -> None:
    starts = synthetic_starts()
    result = bw.decision(starts)
    assert record(result['decision'])['status'] == 'CANDIDATE'
    assert result['verdict'] == 'CANDIDATE FOR PROMOTION'
    assert result['gates'] == {'greedy': True, 'release-census': True, 'kl': True}
    record(record(starts['C2'])['kl'])['pass'] = False
    assert bw.decision(starts)['verdict'] == 'STOP: kl'
    caps = {'C1': (2300, 'start-log'), 'C2': (2150, 'clock-log')}
    rows = bw.archive_rows('UPX-W', synthetic_starts(), caps.__getitem__)
    assert len(rows) == 2 * 2 * (2 * 2 + 1)
    assert {r['start'] for r in rows} == {'C1', 'C2'} and {r['cell'] for r in rows} == {'battery-150k'}
    assert all(r['max_tokens'] == bc.MAX_TOKENS and r['commit'] == n.PROD_COMMIT and r['date'] == '20261028' for r in rows)
    assert {(r['start'], r['clock_cap_mhz'], r['clock_source']) for r in rows} == {('C1', 2300, 'start-log'),
                                                                                  ('C2', 2150, 'clock-log')}
    assert sorted({int(cast(int, r['repeat'])) for r in rows}) == [0, 1]
    dv.archive_append(tmp_path / 'archive.jsonl', rows)            # WIN2-R2 schema v3 accepts every row
    assert len(dv.archive_rows(tmp_path / 'archive.jsonl')) == len(rows)
    # one clock per pooled spread: the two starts differ, so only `where` with one cap pools (WIN2-R2)
    metric = 'next_ttft_s'
    with pytest.raises(ValueError, match='mixes'):
        dv.archive_spread(tmp_path / 'archive.jsonl', 'battery-150k', metric)
    incomplete = synthetic_starts()
    record(incomplete['C2'])['status'] = 'STARTED'
    assert {r['start'] for r in bw.archive_rows('UPX-W', incomplete, caps.__getitem__)} == {'C1'}
    with pytest.raises(ValueError, match='complete'):
        bw.decision(incomplete)


# ---------------------------------------------------------------- the request driver over a fake server

class FakeServer:
    def __init__(self, *, cached: int = 0, busy_after: int = 99) -> None:
        self.now = 1_800_000_000.0
        self.cached = cached
        self.busy_after = busy_after
        self.metric_calls = 0
        self.bodies: list[dict[str, Json]] = []

    def wall(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s

    def model(self) -> dict[str, Json]:
        return {'id': 'DeepSeek', 'max_model_len': wc.CAP}

    def metrics(self) -> str:
        self.metric_calls += 1
        busy = 1 if self.metric_calls > self.busy_after else 0
        return f'vllm:num_requests_running {busy}\nvllm:num_requests_waiting 0\n'

    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]:
        self.bodies.append(body)
        self.now += 70.0
        return {'usage': {'prompt_tokens': 149000, 'completion_tokens': bc.MAX_TOKENS,
                          'prompt_tokens_details': {'cached_tokens': self.cached}}, 'finish': 'length', 'ttft_s': 69.0}


def test_battery_driver_times_and_idles() -> None:
    api = FakeServer()
    t0 = api.now
    out = bc.battery(api, 'text', 149000, 'W:X:b0', t0 + 1000, wall=api.wall, sleep=api.sleep)
    assert out['a_sent'] == t0 and out['a_done'] == t0 + 70
    assert out['b_sent'] == pytest.approx(t0 + 70 + m.IDLE_WAIT_S)
    assert out['b_first'] == pytest.approx(float(cast(float, out['b_sent'])) + 69)
    assert api.now == pytest.approx(float(cast(float, out['b_done'])) + m.POST_B_IDLE_S)
    battery = bc.times(out, 'on')
    assert battery.b_sent - battery.a_done >= m.IDLE_WAIT_S
    salts = [str(record(body)['cache_salt']) for body in api.bodies]
    assert salts == ['W:X:b0:A', 'W:X:b0:B']
    assert all(body['max_tokens'] == bc.MAX_TOKENS and body['ignore_eos'] is True for body in api.bodies)
    json.dumps(out, allow_nan=False)


def test_battery_driver_refuses_cache_hits_foreign_requests_and_the_deadline() -> None:
    api = FakeServer(cached=4096)
    with pytest.raises(ValueError, match='cache'):
        bc.battery(api, 't', 149000, 's', api.now + 1000, wall=api.wall, sleep=api.sleep)
    api = FakeServer(busy_after=1)
    with pytest.raises(ValueError, match='idle period'):
        bc.battery(api, 't', 149000, 's', api.now + 1000, wall=api.wall, sleep=api.sleep)
    api = FakeServer()
    with pytest.raises(ValueError, match='deadline'):
        bc.battery(api, 't', 149000, 's', api.now + 10, wall=api.wall, sleep=api.sleep)


# ---------------------------------------------------------------- the real Live backend through fake commands

def stamp(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + '000Z'


class Remote(rehearsal_base.Remote):
    """NBW's rehearsal remote with this window's own commands."""

    def __init__(self, out: Path, sup: pf.Supervisor, clock: base.Clock, fail: str, trip: Path, *, chained: bool) -> None:
        super().__init__(out, sup, clock, fail, trip)
        self.chained = chained
        self.upx_arm: bw.Arm | None = None
        self.threshold: dict[str, str] = {}
        self.batteries: list[tuple[str, bw.Arm, m.Battery]] = []
        self.started: list[str] = []
        self.allow_unreachable: bool | None = None
        # hold placeholders as each view sees them: plain-ssh probe at preflight / at the chained end, and from a scope
        self.placeholders = {'preflight': True, 'chain-end': True}
        self.placeholders_scoped = True
        self.foreign_hold = fail == 'hold-refused'      # chain.abort's HOLD probe: another window's HOLD exists
        self.aborts: list[Path] = []

    def identity(self, rank: int) -> dict[str, Json]:
        if self.upx_arm is None:
            return pure.production(rank)
        row = x_identity(rank) if self.upx_arm.role == 'X' else identity('8001', rank)
        row['id'] = f'{self.upx_arm.name}-{rank}'
        if self.fail == 'identity' and self.upx_arm.role == 'X':
            record(row['env'])['VLLM_DSV41_RELEASE_TOKENS'] = '1'
        return row

    def release_lines(self, host: str, since: str) -> str:
        floor = datetime.datetime.fromisoformat(since.replace('Z', '+00:00')).timestamp()
        times = [(mode, battery.a_done + 5.2) for mode, _, battery in self.batteries]
        times += [(mode, battery.b_done + 5.1) for mode, _, battery in self.batteries]
        x = [battery for _, arm, battery in self.batteries if arm.role == 'X']
        if self.fail == 'census' and x:
            times.append(('on', x[0].a_sent - 30))      # a release outside every battery: only the start census sees it
        lines = [f'{stamp(at)} (Worker_TP0 pid=9) INFO dsv41 patch 0085b: idle, after 150000 prefill tokens released '
                 '1748 MiB of cached memory (reserved 9000 -> 7252 MiB) in 41.3 ms'
                 for mode, at in sorted(times, key=lambda pair: pair[1]) if mode == 'on' and at >= floor]
        return '\n'.join([*lines, bw.RELEASE_MARKER]) + '\n'

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        if label == 'hold-open':
            tag, payload = rehearsal_base.scoped_argv(command)
            assert tag == self.tag + '-hold' and f'UPX-0085B {self.tag}:' in payload[-1]
            # the admit argv ends the script's first command: '... <purpose> <drain_s> allow-unreachable|require-serving;'
            self.allow_unreachable = ' allow-unreachable; set -a;' in payload[-1]
            assert self.allow_unreachable or ' require-serving; set -a;' in payload[-1]
            if self.fail == 'hold-refused':
                raise ValueError('STOP: HOLD already exists or drain expired; refusing overwrite')
            assert 'os.link(candidate, hold)' in payload[-1]
            self.hold = f'{base.iso(self.clock())} DSV41-OFFLOAD (session coordinator): UPX-0085B {self.tag}: batteries'
            self.original_hold = self.hold
            self.events.append('hold')
            return 'opened'
        if label == 'prod-models' and self.chained:
            raise ValueError('STOP: prod-models failed rc7')
        if label.startswith('chain-quiet-'):
            return 'UPX SPARKS QUIET\n'
        if label.startswith('X-release-release-read-'):
            return self.threshold.get(host, 'absent') + '\n'
        if label.startswith('X-release-release-'):
            tag, payload = rehearsal_base.scoped_argv(command)
            assert tag.startswith(f'{self.tag}-X-release-release-') and payload[:2] == ['bash', '-c']
            self.threshold[host] = payload[-1].split("printf '%d\\n' ")[1].split(' ')[0]
            return ''
        if label.endswith('-start'):
            arm = next(a for a in bw.ARMS if label == a.name + '-start')
            self.upx_arm = arm
            self.started.append(arm.name)
            self.events.append('start-' + arm.name)
            tag, payload = rehearsal_base.scoped_argv(command)
            assert tag == f'{self.tag}-{arm.name}-engine' and configs.CONFIG in payload[-1]
            assert all(f'{k}={v}' in payload[-1] for k, v in arm.env.items())
            if self.fail == 'start' and arm.role == 'X':
                raise ValueError('Spark start failed rc7')
            worker = 2150 if self.fail == 'clock-split' else 2300
            text = f'spark clock cap 2300 MHz\nsparkworker clock cap {worker} MHz\nready\n'
            # Supervisor.run's command log, which win2_history.start_clock_cap reads for the archive
            log = self.out / 'cmd' / f'{len(list((self.out / "cmd").iterdir())) + 1:04d}-{label}.log'
            log.write_text(text)
            return text
        if label.endswith(tuple(f'-0085b-files-{h}' for h in pf.HOSTS)):
            assert self.upx_arm is not None
            key = 'overlay_sha256' if self.upx_arm.role == 'X' else 'base_sha256'
            return ''.join(f"{r[key]}  {r['container_path']}\n" for r in MANIFEST['files'])
        if '-releases-' in label or '-census-' in label:
            assert shlex.quote(bw.RELEASE_FILTER) in command
            return self.release_lines(host, shlex.split(command)[-1])
        return super().remote(label, host, command, stdin)

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        if argv[1:3] != ['-m', 'offload.upx.battery_cells']:
            return super().call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)
        self.sup.poll()
        mode, path = argv[argv.index('--mode') + 1], Path(argv[argv.index('--output') + 1])
        arm = cast(bw.Arm, self.upx_arm)
        if mode == 'build':
            path.write_text('{}\n')
        elif mode == 'greedy':
            self.events.append('greedy-' + arm.name)
            diverged = self.fail == 'greedy-divergence' and arm.role == 'X'
            path.write_text(json.dumps({'identical': 5 if diverged else 6, 'near_tie': 0,
                                        'checks': [{'i': i, 'identical': not (diverged and i == 3)} for i in range(6)]}))
        elif mode == 'kl':
            assert argv[argv.index('--candidate') + 1].endswith('X-release-greedy.json')
            path.write_text(json.dumps({'rows': [], 'bound': bc.KL_BOUND, 'pass': self.fail != 'kl'}))
        else:
            assert mode == 'battery' and inflight == 1
            if self.fail == 'STOP' and arm.role == 'X':
                (self.out / 'STOP').touch()
                self.sup.poll()
            t0 = self.clock()
            battery = m.Battery('', t0, t0 + 70, t0 + 90, t0 + 159, t0 + 160, 149000, 149000)
            if arm.role == 'X':
                assert self.threshold['sparkhead'] == self.threshold['sparkworker-wired'], 'both ranks carry the same threshold'
                mode_of = 'on' if self.threshold['sparkhead'] == str(m.THRESHOLD) else 'off'
            else:
                mode_of = 'image'
            self.batteries.append((mode_of, arm, battery))
            path.write_text(json.dumps({'a_sent': battery.a_sent, 'a_done': battery.a_done, 'b_sent': battery.b_sent,
                                        'b_first': battery.b_first, 'b_done': battery.b_done, 'a_prompt_tokens': 149000,
                                        'b_prompt_tokens': 149000}))
            self.clock.now = t0 + 160 + m.POST_B_IDLE_S + 1
        return ''

    def plain(self, step: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if '-placeholders-' in step:
            view, host = step.split('-placeholders-')
            assert argv == [*pf.SSH, host, chain.PLACEHOLDER_PROBE] and host in chain.PLACEHOLDER_HOSTS
            self.events.append(step)
            return (0, '') if self.placeholders[view] else (1, '')
        if step == 'placeholder-check':
            self.events.append(step)
            tag, payload = rehearsal_base.scoped_argv(argv[-1])
            assert tag.startswith(self.tag + '-restore-placeholder-check-')
            assert payload == [*pf.SSH, 'sparkhead', bw.PLACEHOLDER_CHECK]
            return (0, bw.PLACEHOLDERS_OK + '\n') if self.placeholders_scoped else (1, 'hold placeholder missing on spark\n')
        if step.startswith('chain-quiet-'):
            self.events.append(step)
            assert 'UPX CHAINED QUIET' in argv[-1]
            return 0, 'UPX CHAINED QUIET\n'
        if step == 'chained-close':
            self.events.append(step)
            assert argv[:-1] == [*pf.SSH, 'sparkhead'], 'the close scope runs on sparkhead, where its rm shim guards HOLD'
            tag, payload = rehearsal_base.scoped_argv(argv[-1])
            assert tag == self.tag + '-restore-close'
            assert payload == ['bash', '-c', chain.CHAINED_CLOSE]
            return 0, chain.CHAINED_CLOSE_OK + '\n'
        return super().plain(step, argv, timeout)

    def telemetry(self, host: str) -> list[dict[str, Json]]:
        """1 s MemAvailable over every recorded battery: ON releases show at idle and at the next prefill's low point."""
        rows: list[dict[str, Json]] = []
        k = pf.HOSTS.index(host)
        for mode, _, battery in self.batteries:
            gain = (1_800_000, 500_000) if mode == 'on' else (0, 0)
            t = battery.a_sent
            while t < battery.b_done + m.POST_B_IDLE_S:
                base_kb = 6_600_000 + 400_000 * k
                if battery.b_sent <= t < battery.b_first:
                    kb = base_kb - 200_000 + gain[1]
                elif battery.a_done + 6 <= t < battery.b_sent:
                    kb = base_kb + gain[0]
                else:
                    kb = base_kb - 100_000
                rows.append({'kind': 'memory', 'utc': t, 'kb': {'MemAvailable': kb, 'SwapFree': 1, 'SwapTotal': 2}})
                t += 1.0
        return rows


@pytest.fixture
def rehearsal(tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[bw.Live, Remote]]:
    params = cast(dict[str, str], getattr(request, 'param', {}))
    fail, mode = params.get('fail', ''), params.get('chain', 'off')
    monkeypatch.setattr(chain, 'MARKER', tmp_path / 'CHAINED')
    monkeypatch.delenv('WINDOW_CHAIN', raising=False)
    monkeypatch.delenv('WINDOW_CHAIN_LAST', raising=False)
    if mode != 'off':
        monkeypatch.setenv('WINDOW_CHAIN', '1')
        if mode == 'last':
            monkeypatch.setenv('WINDOW_CHAIN_LAST', '1')
    marker = params.get('marker')
    if marker is not None:              # a previous chained window left production down (yes / no-snapshot / foreign)
        prev = tmp_path / 'UPX-prev-W'
        prev.mkdir()
        commit = '1' * 40 if marker == 'foreign' else n.PROD_COMMIT
        (prev / 'rollback.json').write_text(json.dumps({'production_commit': commit, 'tag': n.PROD_TAG}) + '\n')
        snapshot = None if marker == 'no-snapshot' else chain.snapshot_ref(prev / 'rollback.json')
        chain.MARKER.write_text(json.dumps({'schema': chain.SCHEMA, 'window': str(prev), 'quiet_proof': {'x': 'y'},
                                            'production_snapshot': snapshot, 'placeholders': chain.PLACEHOLDERS_KEPT}) + '\n')
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    out = tmp_path / 'UPX-0085B-fake'
    out.mkdir()
    clock = base.Clock()
    trip = tmp_path / 'trip'
    sup = base.NoTunnel(out, clock=clock, monitor=lambda host: [sys.executable, str(monitor), 'low', str(trip)],
                        demand=lambda: 0.0)
    fake = Remote(out, sup, clock, fail, trip, chained=marker is not None)
    real_abort = chain.abort

    def abort(out_dir: Path) -> dict[str, Json]:
        """chain.abort itself, with its HOLD probe, strict W8 recovery and watch faked (never a real host)."""
        fake.aborts.append(out_dir)
        return real_abort(out_dir, hold=lambda: fake.foreign_hold,
                          recover=lambda where, log: {'start': 'PASS', 'check': 'PASS', 'close': 'PASS'},
                          watch=lambda where, log: {'status': 'PASS'}, log=lambda line: None)
    monkeypatch.setattr(chain, 'abort', abort)

    def no_host() -> bool:
        raise AssertionError('the rehearsal must never reach a real chain host probe')
    for name in ('hold_exists', 'placeholders_present', 'production_serving'):
        monkeypatch.setattr(chain, name, no_host)
    live = bw.Live(out, sup, remote='dsv41-deploy-offload', expected='0' * 40, data_dir=tmp_path, tag='tag',
                   archive=tmp_path / 'archive.jsonl', call=fake.call, plain=fake.plain, clock=clock)
    monkeypatch.setattr(live, 'telemetry', fake.telemetry)
    import offload.prefill_trace.window_requests as wr
    monkeypatch.setattr(wr, 'corpus', lambda root: ('public fake data', {}))
    monkeypatch.setattr(cells, 'greedy_inputs', lambda: (['public prompt'] * 6, [{'i': i, 'text': 'ref'} for i in range(6)]))
    try:
        yield live, fake
    finally:
        sup.close()


def assert_w8(result: dict[str, Json], fake: Remote, *, stopped: bool = True) -> None:
    restore = record(result['restore'])
    assert fake.restores == (['start', 'check', 'close'] if stopped else ['check', 'close'])
    assert str(restore['status']).startswith('RESTORED'), result
    assert str(restore['pair_close']).startswith('not opened')
    assert 'pair-close' not in fake.events, 'no native pair: the x86host production worker is never touched'
    assert not fake.hold
    assert fake.events.index('close') < fake.events.index('commit-close') < fake.events.index('cancel-close')


def test_full_production_window_measures_decides_restores_and_archives(rehearsal: tuple[bw.Live, Remote]) -> None:
    live, fake = rehearsal
    result = bw.window(live, lambda row: None)
    assert 'stop' not in result, result
    assert result['status'] == 'COMPLETE' and result['chain_mode'] == 'production'
    assert fake.allow_unreachable is False
    assert fake.started == ['C1', 'X-release', 'C2']
    assert [mode for mode, _, _ in fake.batteries] == ['image'] * 2 + ['on', 'off'] * 3 + ['image'] * 2
    assert record(result['decision'])['status'] == 'CANDIDATE' and result['verdict'] == 'CANDIDATE FOR PROMOTION'
    census = record(record(record(record(result['starts'])['X-release'])['checks'])['release_census'])
    assert census['expected_per_rank'] == 6 and census['ok'] is True
    assert fake.threshold == {'sparkhead': '0', 'sparkworker-wired': '0'}, 'X leaves the threshold at 0'
    assert result['archived_rows'] == 20 and len(dv.archive_rows(live.archive)) == 20
    assert {(r['clock_cap_mhz'], r['clock_source']) for r in dv.archive_rows(live.archive)} == {(2300, 'start-log')}
    head = record(result['archive_head'])
    assert record(head['before'])['rows'] == 0 and record(head['after'])['rows'] == 20
    assert result['end_action'] == 'restore' and bw.exit_status(dict(result, post_close_watch={'status': 'PASS'})) == (0, '')
    assert result.get('chain') is None and not chain.MARKER.exists()
    assert_w8(result, fake)
    rollback = json.loads((live.out / 'rollback.json').read_text())
    assert rollback['production_commit'] == n.PROD_COMMIT


@pytest.mark.parametrize('rehearsal', [{'chain': 'on', 'marker': 'yes'}], indirect=True)
def test_chained_window_leaves_production_down_and_closes_its_own_hold(rehearsal: tuple[bw.Live, Remote]) -> None:
    live, fake = rehearsal
    result = bw.window(live, lambda row: None)
    assert 'stop' not in result, result
    assert result['chain_mode'] == 'chained' and result['chain'] == 'CHAINED' and result['end_action'] == 'chained'
    assert fake.allow_unreachable is True
    assert record(result['restore'])['status'] == 'CHAINED: production left down'
    assert record(record(result['restore'])['chain_end'])['ok'] is True
    assert not fake.restores, 'no W8 in a chained, non-last window'
    assert {'preflight-placeholders-sparkhead', 'preflight-placeholders-sparkworker-wired'} <= set(fake.events)
    assert (fake.events.index('chain-end-placeholders-sparkworker-wired') < fake.events.index('placeholder-check')
            < fake.events.index('chain-quiet-sparkhead') < fake.events.index('chained-close') < fake.events.index('commit-close'))
    for host in (*owner.NODES, 'localhost'):
        assert fake.events.index(f'owned-absent-{fake.tag}-restore-close-{host}') < fake.events.index('commit-close')
    assert not fake.hold, 'own HOLD closed'
    marker = chain.read_marker()
    assert marker is not None and marker['window'] == str(live.out)
    prev = live.out.parent / 'UPX-prev-W' / 'rollback.json'
    assert record(marker['production_snapshot'])['path'] == str(prev), 'the chain keeps its first production capture'
    rollback = json.loads((live.out / 'rollback.json').read_text())
    assert rollback['chained_from'] == str(prev) and rollback['chained_from_sha256'] == pf.digest(prev)
    assert rollback['production_commit'] == n.PROD_COMMIT
    baseline = json.loads((live.out / 'baseline.json').read_text())
    assert baseline['start'] == 'C1', 'C1 is the identity baseline when production is down'
    assert result['verdict'] == 'CANDIDATE FOR PROMOTION'


@pytest.mark.parametrize('rehearsal', [{'chain': 'on', 'marker': 'yes'}], indirect=True)
@pytest.mark.parametrize('view', ['probe', 'scoped'])
def test_chained_end_without_both_placeholder_views_restores_production(rehearsal: tuple[bw.Live, Remote], view: str) -> None:
    """Today's harness (test_chain_placeholders): the scoped placeholders are invisible to the chain gate's probe and to
    CHAINED_CLOSE's scope. The window must not leave production down: strict W8, never a chained close."""
    live, fake = rehearsal
    if view == 'probe':
        fake.placeholders['chain-end'] = False
    else:
        fake.placeholders_scoped = False
    result = bw.window(live, lambda row: None)
    assert 'stop' not in result and result['end_action'] == 'chained' and result.get('chain') is None
    assert_w8(result, fake)
    ready = record(record(result['restore'])['chain_end'])
    assert ready['ok'] is False and 'placeholders' in str(ready['reason'])
    assert ('placeholder-check' in fake.events) is (view == 'scoped')
    assert 'chained-close' not in fake.events and 'experiment-stop' in fake.events
    assert record(chain.read_marker() or {})['window'] != str(live.out), 'no marker for a restored window'
    assert result['verdict'] == 'CANDIDATE FOR PROMOTION'


@pytest.mark.parametrize('rehearsal,message', [
    ({'chain': 'on', 'marker': 'yes'}, 'hold placeholders missing'),
    ({'chain': 'on', 'marker': 'no-snapshot'}, 'carries no production snapshot'),
    ({'chain': 'on', 'marker': 'foreign'}, 'is not production')], indirect=['rehearsal'])
def test_chained_preflight_refusal_takes_the_chain_abort(rehearsal: tuple[bw.Live, Remote], message: str) -> None:
    """WIN2-R2 end_action 'abort': production is down from the previous window and this one owns no HOLD."""
    live, fake = rehearsal
    fake.placeholders['preflight'] = 'missing' not in message
    result = bw.window(live, lambda row: None)
    assert message in str(result.get('stop')), result
    assert result['end_action'] == 'abort' and fake.aborts == [live.out / 'chain-abort']
    assert str(record(result['restore'])['status']).startswith('RESTORED: chain ended by abort')
    assert record(result['post_close_watch'])['status'] == 'PASS' and chain.read_marker() is None
    assert not fake.started and fake.allow_unreachable is None, 'no HOLD and no start'
    assert bw.exit_status(result) == (1, '')


@pytest.mark.parametrize('rehearsal', [{'chain': 'last', 'marker': 'yes'}], indirect=True)
def test_last_chained_window_restores_production(rehearsal: tuple[bw.Live, Remote]) -> None:
    live, fake = rehearsal
    result = bw.window(live, lambda row: None)
    assert 'stop' not in result, result
    assert result['chain_mode'] == 'chained' and result.get('chain') is None and result['end_action'] == 'restore'
    assert 'chain-end-placeholders-sparkhead' not in fake.events
    assert_w8(result, fake)


@pytest.mark.parametrize('rehearsal,message', [
    ({'fail': 'census'}, 'expected 6, 0'), ({'fail': 'identity'}, 'env drift'), ({'fail': 'start'}, 'rc7'),
    ({'fail': 'greedy-divergence'}, 'near-tie'), ({'fail': 'kl'}, 'KL at a greedy divergence'),
    ({'fail': 'STOP'}, 'STOP file'), ({'fail': 'census', 'chain': 'on', 'marker': 'yes'}, 'expected 6, 0')],
    indirect=['rehearsal'])
def test_every_stop_restores_production_even_in_a_chain(rehearsal: tuple[bw.Live, Remote], message: str) -> None:
    live, fake = rehearsal
    result = bw.window(live, lambda row: None)
    assert message in str(result.get('stop')), result
    assert str(result['verdict']).startswith('STOP')
    assert result.get('chain') is None and result['end_action'] == 'restore'
    assert_w8(result, fake)


@pytest.mark.parametrize('rehearsal', [{'chain': 'on'}], indirect=True)
def test_chain_without_marker_and_production_down_refuses_before_any_hold(rehearsal: tuple[bw.Live, Remote]) -> None:
    live, fake = rehearsal
    fake.chained = True                  # production does not answer, yet no previous window left the marker
    result = bw.window(live, lambda row: None)
    assert 'production not serving and no chain marker' in str(result.get('stop'))
    assert not fake.events and not fake.restores and not fake.aborts
    assert result['end_action'] == 'none' and str(record(result['restore'])['status']).startswith('NOT NEEDED')


@pytest.mark.parametrize('rehearsal', [{'marker': 'yes'}], indirect=True)
def test_marker_without_chain_mode_restores_production_by_chain_abort(rehearsal: tuple[bw.Live, Remote]) -> None:
    live, fake = rehearsal
    result = bw.window(live, lambda row: None)
    assert 'WINDOW_CHAIN=1' in str(result.get('stop'))
    assert result['end_action'] == 'abort' and fake.aborts == [live.out / 'chain-abort']
    assert str(record(result['restore'])['status']).startswith('RESTORED: chain ended by abort')
    assert not fake.started and not fake.restores


def test_plan_and_dry_main(capsys: pytest.CaptureFixture[str]) -> None:
    assert bw.main([]) == 0
    text = capsys.readouterr().out
    assert 'DRY RUN' in text and 'ESTIMATED HOLD 66 min chained, 89 min as the last window' in text


# chain.CHAINED_CLOSE's unlink step as it runs (errexit on); its placeholder half needs the Sparks (test_chain_placeholders)
UNLINK = 'set -euo pipefail; ' + bw.CHAINED_UNLINK


def test_chained_close_records_then_commits_only_this_windows_hold(tmp_path: Path) -> None:
    """The sparkhead side of Live.chained_close with the REAL scoped owner on a fake HOME: in the -restore-close scope the
    rm shim records the HOLD bytes of CHAINED_CLOSE's unlink instead of deleting them; commit-close removes exactly that
    HOLD after fence + absence."""
    assert chain.CHAINED_CLOSE.startswith('set -euo pipefail; ') and chain.CHAINED_CLOSE.endswith(bw.CHAINED_UNLINK)
    import os
    import subprocess
    import uuid
    node = tmp_path / 'fake-sparkhead'
    (node / 'dsv41-deploy').mkdir(parents=True)
    bin_dir = node / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'python3').symlink_to(sys.executable)
    env = {**os.environ, 'HOME': str(node), 'PATH': str(bin_dir) + ':' + os.environ['PATH'], 'CUDA_VISIBLE_DEVICES': ''}
    window_tag = 'own1test-upx-' + uuid.uuid4().hex[:10]
    tag = window_tag + '-restore-close'
    hold = node / 'dsv41-deploy/HOLD'
    hold.write_text(f'2026-10-28T00:00:00Z DSV41-OFFLOAD (session coordinator): UPX-0085B {window_tag}: batteries\n')
    deadline = time.time() + 120

    def run(command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(['bash', '-c', command], env=env, capture_output=True, text=True, timeout=60)
    try:
        step = run(owner.command('launch', tag, ['bash', '-c', UNLINK], deadline=deadline))
        assert step.returncode == 0 and chain.CHAINED_CLOSE_OK in step.stdout.splitlines(), step.stdout + step.stderr
        assert hold.exists(), 'the rm shim must record, never delete, the HOLD'
        assert (node / owner.CONTROL / tag / 'CLOSE_HOLD').read_text() == hold.read_text()
        for mode, marker in (('fence', owner.FENCED), ('reap', owner.ABSENT), ('absent', owner.ABSENT)):
            done = run(owner.command(mode, tag))
            assert done.returncode == 0 and marker in done.stdout.splitlines(), done.stdout + done.stderr
        closed = run(owner.command('commit-close', tag, deadline=deadline))
        assert closed.returncode == 0 and owner.CLOSED in closed.stdout.splitlines(), closed.stdout + closed.stderr
        assert not hold.exists()
    finally:
        for mode in ('fence', 'reap'):
            run(owner.command(mode, tag))


def test_chained_close_refuses_another_windows_hold(tmp_path: Path) -> None:
    import os
    import subprocess
    import uuid
    node = tmp_path / 'fake-sparkhead'
    (node / 'dsv41-deploy').mkdir(parents=True)
    bin_dir = node / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'python3').symlink_to(sys.executable)
    env = {**os.environ, 'HOME': str(node), 'PATH': str(bin_dir) + ':' + os.environ['PATH'], 'CUDA_VISIBLE_DEVICES': ''}
    tag = 'own1test-upx-' + uuid.uuid4().hex[:10] + '-restore-close'
    hold = node / 'dsv41-deploy/HOLD'
    hold.write_text('2026-10-28T00:00:00Z DSV41-OFFLOAD (session coordinator): NBW-WINDOW other-window: native\n')
    try:
        step = subprocess.run(['bash', '-c', owner.command('launch', tag, ['bash', '-c', UNLINK], deadline=time.time() + 120)],
                              env=env, capture_output=True, text=True, timeout=60)
        assert step.returncode != 0 and 'HOLD belongs to another window' in step.stderr
        assert chain.CHAINED_CLOSE_OK not in step.stdout
        assert hold.exists() and not (node / owner.CONTROL / tag / 'CLOSE_HOLD').exists()
    finally:
        for mode in ('fence', 'reap'):
            subprocess.run(['bash', '-c', owner.command(mode, tag)], env=env, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize('rehearsal,status,code', [({'fail': 'hold-refused'}, 'NOT NEEDED', 1),
                                                   ({'fail': 'hold-refused', 'chain': 'on', 'marker': 'yes'}, 'ALERT', 3)],
                         indirect=['rehearsal'])
def test_refused_hold_never_runs_w8(rehearsal: tuple[bw.Live, Remote], status: str, code: int) -> None:
    """Another window's HOLD: production mode touches nothing; in a chain, chain.abort alerts and touches nothing."""
    live, fake = rehearsal
    result = bw.window(live, lambda row: None)
    assert 'refusing overwrite' in str(result.get('stop'))
    assert str(record(result['restore'])['status']).startswith(status), result
    assert not fake.restores and not fake.started
    assert f'owned-absent-{fake.tag}-hold-sparkhead' in fake.events, 'the opener is fenced before deciding'
    assert fake.events.index(f'owned-absent-{fake.tag}-hold-sparkhead') < fake.events.index('hold-reread')
    assert bw.exit_status(result)[0] == code
    if code == 3:
        assert result['verdict'] == 'NOT A CANDIDATE: production restore unverified' and chain.read_marker() is not None


@pytest.mark.parametrize('rehearsal', [{'fail': 'clock-split'}], indirect=True)
def test_unproven_start_clock_blocks_the_archive_not_the_window(rehearsal: tuple[bw.Live, Remote]) -> None:
    live, fake = rehearsal
    result = bw.window(live, lambda row: None)
    assert 'stop' not in result and result['verdict'] == 'CANDIDATE FOR PROMOTION'
    assert 'does not prove one clock cap on both Sparks' in str(result['archive_error'])
    assert not live.archive.exists()
    assert_w8(result, fake)


def test_exit_status_maps_every_end() -> None:
    def end(status: str, watch: str = '', stop: bool = False) -> dict[str, Json]:
        result: dict[str, Json] = {'restore': {'status': status}}
        if watch:
            result['post_close_watch'] = {'status': watch}
        if stop:
            result['stop'] = 'ValueError: x'
        return result
    assert bw.exit_status(end('CHAINED: production left down'))[0] == 0
    assert bw.exit_status(end('RESTORED: K1 START VERIFIED', 'PASS'))[0] == 0
    assert bw.exit_status(end('RESTORED: K1 START VERIFIED', 'PASS', stop=True))[0] == 1
    assert bw.exit_status(end('RESTORED: chain ended by abort; watch PASS; marker cleared', 'PASS', stop=True))[0] == 1
    assert bw.exit_status(end('NOT NEEDED: no owned HOLD and no chain marker', stop=True))[0] == 1
    assert bw.exit_status(end('RESTORED BUT POST-CLOSE WATCH FAILED', 'FAILED: x', stop=True))[0] == 2
    assert bw.exit_status(end('RESTORED: K1 START VERIFIED'))[0] == 2, 'a restore is never reported unwatched'
    assert bw.exit_status(end('FAILED: strict W8 recovery unverified', stop=True))[0] == 2
    assert bw.exit_status(end('FAILED', stop=True))[0] == 2
    assert bw.exit_status(end('ALERT: a HOLD exists', stop=True))[0] == 3
