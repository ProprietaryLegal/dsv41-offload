"""PF2-LOWKV item 5: CPU rehearsal of the real PF2Live / Supervisor / window() against SP1's and PF1's fake command layer.
No ssh, scp, HTTP or service. Every case that may have written a HOLD must end in W8 restore start|check|close."""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any, cast

import pytest

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pf2_window as p
from offload.launch.windows import sp1_window as s
from offload.launch.windows.tests import test_pf1_window as base
from offload.launch.windows.tests import test_sp1_window as sp1t
from offload.launch.windows.tests.test_pf2_window import PLATFORM, parity
from offload.pf2_lowkv import analyze, arith
from offload.pf2_lowkv import cells as pc
from offload.pf2_lowkv.test_analyze import kineto
from offload.prefill_maxlen.evidence import Json, record

RESTORED = ['start', 'check', 'close']
SLOW, FAST = 95.0, 70.0
SAMPLER = ('import json, sys, time\nsys.stdin.read()\n'
           'while True:\n'
           '    print(json.dumps({"kind": "procmem", "utc": time.time(), "procs": [{"pid": 7, "role": "worker", "Pss": 5, "rollup_ms": 1.0}]}), flush=True)\n'
           '    time.sleep(.2)\n')


def extended(available: int = 9_000_000) -> str:
    vm: dict[str, int] = {k: 0 for k in p.VMSTAT}
    vm['allocstall_normal'] = 0
    return (json.dumps({'kind': 'memory', 'utc': time.time(), 'kb': {'MemAvailable': available, 'SwapFree': 1, 'SwapTotal': 2},
                        'vmstat': vm, 'psi': {'some': {'avg10': 0.0, 'total': 1.0}, 'full': {'avg10': 0.0, 'total': 0.0}}}) + '\n' +
            json.dumps({'kind': 'gpu', 'utc': time.time(), 'csv': 'GPU-x,2100,4000,50.1,100,60,90,0x0'}) + '\n')


class PF2Remote(sp1t.SP1Remote):
    """SP1's fake profile directory and PF1's fake Sparks/x86host, plus PF2's KV reservation, host identity, extended
    telemetry probe and cell driver. TTFT: H starts slow, L starts fast (the HEADROOM_IMPLICATED pattern)."""

    def __init__(self, out: Path, sup: pf.Supervisor, clock: base.Clock, fail: str = '', trip: Path | None = None) -> None:
        super().__init__(out, sup, clock, fail, trip=trip)
        self.kv = arith.H_KV
        self.starts: list[int] = []
        self.kernel = dict(PLATFORM.kernel)

    def _kv_argv(self, argv: str, kv: int) -> str:
        return argv.replace('--max-num-seqs 4', f'--max-num-seqs 4 --kv-cache-memory-bytes {kv}')

    def identity(self, rank: int) -> dict[str, Json]:
        ident = super().identity(rank)
        kv = arith.H_KV if self.arm is None else (arith.H_KV if self.fail == 'kv-argv' and self.kv != arith.H_KV else self.kv)
        ident['argv'] = self._kv_argv(str(ident['argv']), kv)
        return ident

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        if argv[1:3] == ['-m', 'offload.pf2_lowkv.cells']:
            self.sup.poll()
            self.commands.append(label)
            return self.pf2_driver(argv, inflight)
        return super().call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)

    def role(self) -> str:
        return 'H' if self.kv == arith.H_KV else 'L'

    def pf2_driver(self, argv: list[str], inflight: int) -> str:
        args = {argv[i]: argv[i + 1] for i in range(4, len(argv) - 1) if argv[i].startswith('--')}
        command = argv[argv.index('--data-dir') + 2]
        if command == 'build':
            Path(args['--output']).write_text(json.dumps({'prompts': {k: {'tokens': v.target - 3} for k, v in pc.PROMPTS.items()}}))
            return ''
        assert float(args['--deadline-utc']) == self.sup.abort_at - 4 * 60
        if command == 'needle':
            self.cells.append(f'{self.role()}:needle-480k')
            if self.fail == 'needle':
                raise ValueError('STOP: needle driver rc1')
            Path(args['--output']).write_text(json.dumps({'type': 'needle', 'status': 'RAN', 'recall': 6, 'of': 6,
                                                          'recalled_all': True, 'prompt_tokens': 479000, 'cached_tokens': 0,
                                                          'elapsed_s': 240.0}))
            self.rounds += 5
            self.steps += 5
            return ''
        name = args['--cell']
        cell = pc.CELLS[name]
        assert inflight == cell.batch
        self.cells.append(f'{self.role()}:{name}')
        if self.trip is not None and cell.captures:
            self.trip.touch()
        if self.fail == 'late-abort' and cell.captures:
            self.clock.now += 67 * 60
        if self.fail == 'late-submit' and cell.captures and len(self.starts) == 2:
            self.clock.now += 50 * 60            # the 3rd start (13.5 min) no longer fits before T+62
        end = time.monotonic() + .2
        self.sup.inflight = inflight
        try:
            while time.monotonic() < end:
                self.sup.poll()
        finally:
            self.sup.inflight = 0
        ttft = (SLOW if self.role() == 'H' else FAST) * (1.2 if cell.captures else 1.0)
        captures: list[Json] = []
        for kind in cell.captures:
            captures.append({'kind': kind, 'status': 'TAKEN', 'start_call_rel_s': 35.0, 'start_return_rel_s': 35.01,
                             'stop_call_rel_s': 40.0, 'stop_return_rel_s': 40.5, 'mem_available_kb': {}})
            for rank, host in enumerate(pf.HOSTS):
                files = self.dirs[host]
                assert files is not None, 'capture without a profile directory'
                text = kineto(8000.0 if self.role() == 'H' else 6000.0, [96, 1, 1]).text().encode()
                self.ns += 1
                files[f'dp0_pp0_tp{rank}_dcp0_ep{rank}_rank{rank}.{self.ns}.pt.trace.json.gz'] = gzip.compress(text, mtime=0)
                self.started[host] += 1
        stream = {'stream': 0, 'submitted_rel_s': .1, 'ttft_s': ttft, 'total_s': ttft + 27.5,
                  'usage': {'completion_tokens': 64 if cell.warmup else 1500, 'prompt_tokens': 149_000}}
        Path(args['--output']).write_text(json.dumps({'type': 'cell', 'cell': name, 'batch': cell.batch, 'origin_wall': time.time(),
                                                      'streams': [stream], 'captures': captures}))
        self.rounds += 50
        self.steps += 50
        return ''

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        if 'window.sh open' in command:
            self.hold = f'{base.iso(self.clock())} DSV41-OFFLOAD (session coordinator): {command.split("PF2-LOWKV ")[1][:-1]}'
            return 'window open'
        if command == pf.HOST_PARITY:
            kernel = self.kernel[host]
            if self.fail == 'kernel-drift' and len(self.starts) >= 2 and host == 'sparkhead':
                kernel = '7.0.0-9999-nvidia'
            return parity(kernel, PLATFORM.driver[host], 9_000_000)
        if command == p.telemetry_command(1):
            return extended()
        if 'bash offload/launch/start_exp.sh' in command:
            assert f'bash offload/launch/start_exp.sh {p.CONFIG}' in command
            env = dict(x.split('=') for x in command.split(' && ')[1].split(' timeout ')[0].split())
            assert (env['PF1_CAP'], env['PF1_TRACE'], env['PF1_EAGER']) == ('524288', '0', '0')
            self.kv = int(env['PF2_KV_BYTES'])
            self.starts.append(self.kv)
            if self.fail == 'start2' and len(self.starts) == 2:
                raise ValueError('STOP: s2-L-start failed rc1; see log')
            self.started = {h: 0 for h in pf.HOSTS}           # a fresh container's log
            return super().remote(label, host, command.replace(f' PF2_KV_BYTES={self.kv}', ''), stdin)
        if 'kv_tokens' in command:
            if self.arm is None:
                return json.dumps(base.log_summary(327680))
            pool = arith.expected_tokens(self.kv)
            if self.fail == 'kv-log' and self.kv != arith.H_KV:
                pool = 2_820_756
            return json.dumps(base.log_summary(524288, kv_tokens=[pool]))
        if f'mkdir ~/{s.PROFILE_DIR}' in command:
            assert self.dirs[host] is None
            self.dirs[host] = {}
            return ''
        return super().remote(label, host, command, stdin)

    def plain(self, step: str, command: list[str], timeout: float) -> tuple[int, str]:
        if command[0] == 'ssh' and step == 'reap':
            self.unsupervised.append((step, command[-2], command[-1]))
            assert "reap 'window[.]sh open PF2-LOWKV tag:' 'HOLD opener'" in command[-1]
            assert 'start_exp[.]sh offload/launch/configs/PF2-lowkv[.]env' in command[-1]
            return 0, 'reap done\n'
        return super().plain(step, command, timeout)


def spawn(command: list[str], out: IO[str], err: IO[str]) -> subprocess.Popen[str]:
    assert command[-1] == 'docker exec -i dsv41-serve python3 -u -'
    return subprocess.Popen([sys.executable, '-u', '-c', SAMPLER], stdin=subprocess.PIPE, stdout=out, stderr=err, text=True,
                            start_new_session=True)


@pytest.fixture
def rehearsal(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[tuple[p.PF2Live, PF2Remote, base.Clock]]:
    params = cast(dict[str, str], getattr(request, 'param', {}))
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    out = tmp_path / 'PF2-W-rehearsal'
    out.mkdir()
    clock = base.Clock()
    trip = tmp_path / 'trip'
    mode = params.get('monitor', 'ok')
    sup = base.NoTunnel(out, clock=clock, monitor=lambda host: [sys.executable, str(monitor), mode, str(trip)],
                        demand=lambda: 0.0)
    fake = PF2Remote(out, sup, clock, params.get('fail', ''), trip=trip if 'monitor' in params else None)
    live = p.PF2Live(out, sup, remote='dsv41-deploy-offload', expected='0' * 40, data_dir=tmp_path, tag='tag',
                     stage_bundle=False, low=arith.LOW_KV, platform=PLATFORM, call=fake.call, plain=fake.plain, clock=clock,
                     spawn=spawn)
    import offload.prefill_trace.window_requests as requests_module
    original = requests_module.corpus
    requests_module.corpus = lambda root: ('x', {})
    try:
        yield live, fake, clock
    finally:
        requests_module.corpus = original
        live.stop_samplers()
        sup.close()


def rehearse(live: p.PF2Live) -> dict[str, Any]:
    rows: list[dict[str, Json]] = []
    result = p.window(live, p.arms(arith.LOW_KV), rows.append)
    (live.out / 'summary.json').write_text(json.dumps(result, indent=1))
    return result


def started(fake: PF2Remote) -> list[str]:
    return [c for c in fake.commands if c.endswith('-start')]


def test_full_sequence_restores_and_feeds_the_analysis(rehearsal: tuple[p.PF2Live, PF2Remote, base.Clock]) -> None:
    live, fake, _clock = rehearsal
    result = rehearse(live)
    assert 'stop' not in result, result.get('stop')
    assert result['status'] == 'COMPLETE' and fake.restores == RESTORED
    assert str(result['restore']['status']).startswith('RESTORED: K1 START VERIFIED')
    assert started(fake) == ['s1-H-start', 's2-L-start', 's3-H-start', 's4-L-start']
    assert fake.starts == [arith.H_KV, arith.LOW_KV, arith.H_KV, arith.LOW_KV]
    assert fake.cells == [f'{r}:{c}' for r in 'HLHL' for c in pc.ORDER][:12] + [f'L:{c}' for c in pc.ORDER] + ['L:needle-480k']
    assert result['starts']['s4-L']['cells']['needle-480k']['status'] == 'RAN'
    assert all(result['starts'][n]['kv_tokens'] == arith.expected_tokens(kv) for n, kv in
               zip(('s1-H', 's2-L'), (arith.H_KV, arith.LOW_KV)))
    assert all(d is None for d in fake.dirs.values())                              # collected, removed, rmdir'd
    manifest = json.loads((live.out / 'traces-manifest.json').read_text())
    assert len(manifest) == 8 and {r['cell'] for r in manifest} == {f'{n}-p150' for n in ('s1-H', 's2-L', 's3-H', 's4-L')}
    deadlines = json.loads((live.out / 'deadlines.json').read_text())
    assert deadlines['submission'] - deadlines['opened'] == 62 * 60 and deadlines['abort'] - deadlines['opened'] == 66 * 60
    assert all((live.out / f'procmem-{n}-{h}.jsonl').stat().st_size > 0 for n in ('s1-H', 's4-L') for h in pf.HOSTS)
    assert json.loads((live.out / 'hosts-preflight.json').read_text())['sparkhead']['kernel'] == PLATFORM.kernel['sparkhead']
    assert set(json.loads((live.out / 'arm-memory.json').read_text())) == {'s1-H', 's2-L', 's3-H', 's4-L'}
    got: Any = analyze.analyze(live.out, 70.0, 'test')                              # item 4 on the rehearsal output
    assert got['reading']['verdict'] == 'HEADROOM_IMPLICATED'
    assert got['dispatch']['diff']['label'] == 'fast vs slow'
    assert not live.samplers


@pytest.mark.parametrize('rehearsal', [{'fail': 'needle'}], indirect=True)
def test_needle_stop_keeps_the_last_low_start(rehearsal: tuple[p.PF2Live, PF2Remote, base.Clock]) -> None:
    """S44 FIX 1: the last L start is classified before the 480K needle, so a needle STOP restores production and
    still leaves four complete starts for the reading (before the fix: s4-L STARTED, reading INCOMPLETE)."""
    live, fake, _clock = rehearsal
    result = rehearse(live)
    assert 'needle driver rc1' in result['stop'] and fake.restores == RESTORED
    assert result['starts']['s4-L']['status'] == 'COMPLETE' and 'needle_checks' not in result['starts']['s4-L']
    got: Any = analyze.analyze(live.out, 70.0, 'test')
    assert got['reading']['verdict'] == 'HEADROOM_IMPLICATED'


@pytest.mark.parametrize('rehearsal', [{'fail': 'start2'}], indirect=True)
def test_failed_start_restores(rehearsal: tuple[p.PF2Live, PF2Remote, base.Clock]) -> None:
    live, fake, _clock = rehearsal
    result = rehearse(live)
    assert 's2-L-start failed' in result['stop'] and fake.restores == RESTORED
    assert fake.cells == [f'H:{c}' for c in pc.ORDER]
    assert result['restore']['profile_dir_sparkhead'] == 'rc0: CLEAN' and not live.samplers


@pytest.mark.parametrize('rehearsal', [{'monitor': 'low'}], indirect=True)
def test_memory_guard_trip_restores(rehearsal: tuple[p.PF2Live, PF2Remote, base.Clock]) -> None:
    live, fake, _clock = rehearsal
    result = rehearse(live)
    assert 'below 4 GiB' in result['stop'] and fake.restores == RESTORED
    assert fake.cells[-1] == 'H:p150'                                              # tripped in the first profiled cell
    minima = json.loads((live.out / 'arm-memory.json').read_text())['s1-H']
    assert min(minima.values()) < 4 * 1024**2


@pytest.mark.parametrize('rehearsal', [{'fail': 'late-abort'}], indirect=True)
def test_abort_deadline_restores(rehearsal: tuple[p.PF2Live, PF2Remote, base.Clock]) -> None:
    live, fake, _clock = rehearsal
    result = rehearse(live)
    assert '66-minute PF2 abort bound' in result['stop'] and fake.restores == RESTORED
    assert fake.cells[-1] == 'H:p150'


@pytest.mark.parametrize('rehearsal', [{'fail': 'late-submit'}], indirect=True)
def test_submission_bound_skips_later_starts_then_restores(rehearsal: tuple[p.PF2Live, PF2Remote, base.Clock]) -> None:
    live, fake, _clock = rehearsal
    result = rehearse(live)
    assert 'stop' not in result and result['status'] == 'COMPLETE' and fake.restores == RESTORED
    assert started(fake) == ['s1-H-start', 's2-L-start']
    assert result['starts']['s3-H']['status'] == result['starts']['s4-L']['status'] == 'SKIPPED_TIME'
    got: Any = analyze.analyze(live.out, 70.0, 'test')
    assert got['reading']['verdict'] == 'INCOMPLETE'


@pytest.mark.parametrize('rehearsal,message,cells', [
    ({'fail': 'identity'}, 'mounted engine bytes differ from production', 0),
    ({'fail': 'profiler-json'}, 'differs from the reviewed JSON', 0),
    ({'fail': 'mount-ro'}, 'not the read-write bind', 0),
    ({'fail': 'early-profiler'}, 'profiler already started', 0),
    ({'fail': 'kv-argv'}, "experiment KV reservation is not the arm's 3221225472", 4),
    ({'fail': 'kv-log'}, 'KV capacity is not 1410348', 4),
    ({'fail': 'kernel-drift'}, 'sparkhead kernel', 8)], indirect=['rehearsal'])
def test_identity_mismatch_restores(rehearsal: tuple[p.PF2Live, PF2Remote, base.Clock], message: str, cells: int) -> None:
    live, fake, _clock = rehearsal
    result = rehearse(live)
    assert message in result['stop'], result['stop']
    assert fake.restores == RESTORED and len(fake.cells) == cells


def test_platform_mismatch_before_the_hold_touches_nothing(rehearsal: tuple[p.PF2Live, PF2Remote, base.Clock]) -> None:
    live, fake, _clock = rehearsal
    fake.kernel['sparkworker-wired'] = '7.0.0-1004-nvidia'                                # sparkworker is not the reviewed kernel
    result = rehearse(live)
    assert 'sparkworker-wired kernel' in result['stop'] and fake.restores == [] and fake.hold == ''
    assert str(result['restore']['status']).startswith('NOT NEEDED')


def test_restore_strings_are_the_real_ones() -> None:
    for step in RESTORED:
        pf.restored(step, 0, base.RESTORE_OK[step])
    with pytest.raises(ValueError, match='6/6'):
        pf.restored('check', 0, base.RESTORE_OK['check'].replace('greedy K1-post vs restored: 5 IDENTICAL 105\n', ''))
    with pytest.raises(ValueError):
        pf.restored('check', 0, '6/6 IDENTICAL\nPRODUCTION HOLD OK\nCHECK: DONE\n')     # no literal "6/6" accepted
    assert '6/6' not in base.RESTORE_OK['check']
