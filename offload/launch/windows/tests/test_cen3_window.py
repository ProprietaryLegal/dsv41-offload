"""CEN3 window (bandwidth, Nsight Compute bytes, profiled start with a drafter-down segment), CPU only: the config over
fixtures, the pure gates, and the real CEN3Live/Supervisor over the NBW/PF1 rehearsal fakes with simulated bench
containers, profile directories and proxy counters (no SSH, HTTP, Spark, x86host or GPU)."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from offload.census import cen3_cells as cells
from offload.launch.windows import cen3_window as w
from offload.launch.windows import cen_common as cc
from offload.launch.windows import chain
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import sp1_window as sp1
from offload.launch.windows import up2_checks as ck
from offload.launch.windows.tests import test_nbw_config as nbw_config
from offload.launch.windows.tests import test_nbw_rehearsal as rb
from offload.launch.windows.tests import test_nbw_window as pure
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, record

CONFIG = pf.ROOT / w.CONFIG
NCU_DIR = '/opt/nvidia/nsight-compute/2025.3.1'


# ---------------------------------------------------------------- config and pure gates

def source(tmp_path: Path, extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    prod = tmp_path / 'production.env'
    prod.write_text(f'OFFLOAD_DOCKER_ENV={shlex.quote(nbw_config.ENV)}\nOFFLOAD_DSPARK_ENV={shlex.quote(nbw_config.DRAFT)}\n'
                    'OFFLOAD_DOCKER_ARGS="-v /production/code:/engine:ro"\n'
                    'PROD_KV_BYTES=6442450944\nPROD_MAX_LEN=327680\nPROD_K1=1\nPROD_K1_PROFILE=327680\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('NBW_', 'CEN3_', 'EXP_', 'EXTRA_'))}
    env.update({'NBW_PROD_ENV': str(prod), 'CEN3_TAG': 'CEN3-W-20261028T000000Z', **(extra or {})})
    script = f'source {shlex.quote(str(CONFIG))} || exit\n' + '\n'.join(
        f'printf "%s\\n" "${{{key}-}}"' for key in ('EXP_DOCKER_ENV', 'EXTRA_DOCKER_ARGS', 'EXP_EXTRA_ARGS', 'LABEL'))
    return subprocess.run(['bash', '-c', script], env=env, cwd=pf.ROOT, capture_output=True, text=True, check=False)


def test_config_is_production_plus_the_reviewed_profiler_and_mount(tmp_path: Path) -> None:
    result = source(tmp_path)
    assert result.returncode == 0, result.stderr
    env, mounts, extra, label = result.stdout.splitlines()
    assert env == nbw_config.ENV + ' ' + nbw_config.DRAFT, 'production env verbatim'
    assert mounts == ('-v /production/code:/engine:ro '
                      '-v $HOME/dsv41-offload-run/cen3/CEN3-W-20261028T000000Z/profile:/cen3-profile:rw')
    assert shlex.split(extra) == ['--profiler-config', w.PROFILER_JSON], 'byte-identical to the runner JSON'
    assert record(cast(Json, json.loads(w.PROFILER_JSON)))['max_iterations'] == 64 and label == 'CEN3-profile'


@pytest.mark.parametrize('extra', [{'CEN3_TAG': ''}, {'EXP_EXTRA_ARGS': '--x'}, {'CEN3_PROFILER_CONFIG': '{}'}])
def test_config_refuses_inherited_or_missing_settings(tmp_path: Path, extra: dict[str, str]) -> None:
    result = source(tmp_path, extra)
    assert result.returncode != 0 and 'CEN3 STOP' in result.stderr


def test_bench_scripts_and_container_commands() -> None:
    bw = w.docker_command('bw', 1, 'dsv41-offload-run/cen3/t', None)
    assert '--user "$(id -u):$(id -g)"' in bw and 'SYS_ADMIN' not in bw and '--network none' in bw
    assert f'--name {w.CONTAINER}-bw ' in bw and 'timeout --signal=TERM' in bw and '--rank 1' in bw
    ncu = w.docker_command('ncu', 0, 'dsv41-offload-run/cen3/t', NCU_DIR)
    assert '--cap-add SYS_ADMIN' in ncu and f'src={NCU_DIR},dst=/ncu-host,readonly' in ncu
    script = w.bench_script('ncu', 0)
    assert script.count('--cache-control all --clock-control none') == len(w.NCU_GROUPS)
    assert all(f'--group {g} ' in script for g in w.NCU_GROUPS) and 'chown -R' in script
    smoke = w.bench_script('ncu-smoke', 1)
    assert 'exit 3' in smoke and '--query-metrics' in smoke and '--group smoke' in smoke
    with pytest.raises(ValueError):
        w.bench_script('nope', 0)
    assert w.bench_files('f 10 bw.json\nf 0 ncu-failures.txt\n') == {'bw.json': 10, 'ncu-failures.txt': 0}
    for bad in ('d 1 bw.json', 'f 1 secret.txt', f'f {w.BENCH_FILE_CAP + 1} bw.json'):
        with pytest.raises(ValueError):
            w.bench_files(bad)
    assert w.ncu_host_dir('/opt/nvidia/nsight-compute/2024.1\n/opt/nvidia/nsight-compute/2025.3.1\n') == NCU_DIR
    assert w.ncu_host_dir('\n') is None


def stats(rounds: int, fnr: int, **extra: int) -> dict[str, Json]:
    return {'counters': {'rounds_ok': rounds, 'fallback_not_ready': fnr, **extra}, 'policy': 'capacity', 'async': True,
            'async_proxy': {'mode': 'process'}, 'async_handoff': {'mlock': 'ok'},
            'async_watchdog': {'fires': 0, 'running': True}, 'proxy_torch_loaded': False}


def test_target_only_rules() -> None:
    assert w.target_only(stats(10, 0), stats(10, 64))['fallback_not_ready_delta'] == 64
    with pytest.raises(ValueError, match='pair window not in force'):
        w.target_only(stats(10, 0), stats(11, 64))
    with pytest.raises(ValueError, match='no target-only round'):
        w.target_only(stats(10, 0), stats(10, 0))
    with pytest.raises(ValueError, match='watchdog_fires'):
        w.target_only(stats(10, 0, watchdog_fires=0), stats(10, 5, watchdog_fires=1))


# ---------------------------------------------------------------- rehearsal

def trace_name(rank: int, ns: int) -> str:
    return f'dp0_pp0_tp{rank}_dcp0_ep0_rank{rank}.{ns}.pt.trace.json.gz'


class Remote(rb.Remote):
    def __init__(self, out: Path, sup: pf.Supervisor, clock: base.Clock, fail: str, trip: Path) -> None:
        super().__init__(out, sup, clock, fail, trip)
        self.engine = False
        self.pair = False
        self.fnr = 0
        self.ns = 1000
        self.bench_out: dict[str, dict[str, bytes]] = {h: {} for h in pf.HOSTS}
        self.profile: dict[str, dict[str, bytes]] = {h: {} for h in pf.HOSTS}
        self.parts: list[str] = []
        self.cells: list[tuple[str, bool]] = []

    def identity(self, rank: int) -> dict[str, Json]:
        row = pure.production(rank)
        if self.engine:
            row['id'] = f'cen3-{rank}'
            row['argv'] = str(row['argv']).replace(
                '--port 8000', '--port 8001 --profiler-config ' + shlex.quote(w.PROFILER_JSON))
            record(row['mounted'])['/serve.sh'] = '9' * 64
        return row

    def listing(self, files: dict[str, bytes]) -> str:
        return ''.join(f'f {len(v)} {k}\n' for k, v in sorted(files.items()))

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        run = f'dsv41-offload-run/cen3/{self.tag}'
        if label == 'hold-open':
            _, payload = rb.scoped_argv(command)
            assert f'CEN3-WINDOW {self.tag}:' in payload[-1]
            self.hold = f'{base.iso(self.clock())} DSV41-OFFLOAD (session coordinator): CEN3-WINDOW {self.tag}: x'
            self.original_hold = self.hold
            self.events.append('hold')
            return 'opened'
        if label.startswith('run-dirs-'):
            return '/dev/x 100 1 99999999 1% /home\n'
        if label.startswith('cen3-plan-'):
            return json.dumps({w.PLAN: 'a' * 64})
        if label.startswith('cen3-ncu-host-'):
            return NCU_DIR + '\n'
        if label in ('cen3-x86host-window',) or label.startswith(('cen3-image-', 'cen3-mkdir-', 'cen3-idle-')):
            return ''
        if label == 'cen3-stop':
            tag, payload = rb.scoped_argv(command)
            assert tag == f'{self.tag}-cen3-stop' and 'cd ~/dsv41-deploy && ./stop.sh' in payload[-1]
            assert ' stop tag-hold ' in payload[-1] or 'stop' in payload[-1]
            self.events.append('production-stop')
            return ''
        if label.startswith('cen3-memory-'):
            return f'{pf.KERNEL if hasattr(pf, "KERNEL") else "k"}\nMemAvailable: {50 * 1024**2} kB\n'
        if label.startswith('cen3-absent-'):
            return 'CEN3 BENCH ABSENT\n'
        if label.startswith('cen3-out-list-'):
            return self.listing(self.bench_out[host])
        if label.startswith('cen3-out-hash-'):
            names = shlex.split(command.split('python3 - ', 1)[1])
            return json.dumps({n: hashlib.sha256(self.bench_out[host][n]).hexdigest() for n in names})
        if label == 'cen3-start':
            tag, payload = rb.scoped_argv(command)
            assert tag == f'{self.tag}-cen3-engine' and f'CEN3_TAG={self.tag}' in payload[-1] and w.CONFIG in payload[-1]
            self.engine = True
            self.events.append('start-cen3')
            (self.out / 'cmd').mkdir(exist_ok=True)
            (self.out / 'cmd' / f'{len(list((self.out / "cmd").iterdir())) + 1:04d}-{label}.log').write_text(
                'spark clock cap 2300 MHz\nsparkworker clock cap 2300 MHz\n')
            return 'ready'
        if label.startswith('cen3-mounts-'):
            return json.dumps([{'Type': 'bind', 'Destination': w.MOUNT, 'RW': True, 'Source': f'/home/u/{run}/profile'}])
        if label.startswith(('cen3-profile-empty-', 'final-list-')) or (label.endswith(f'-list-{host}') and
                                                                       'find . -mindepth' in command):
            return self.listing(self.profile[host]) + 'FREE 99999999\n'
        if label.startswith('log-count-'):
            return '1\n' if shlex.quote(sp1.PROFILER_ON) in command else '0\n'
        if label.endswith(f'-hash-{host}') and '/profile' in command:
            names = shlex.split(command.split('python3 - ', 1)[1])
            return json.dumps({n: hashlib.sha256(self.profile[host][n]).hexdigest() for n in names})
        if label.endswith(f'-remove-{host}'):
            for n in shlex.split(command.split('rm -f -- ', 1)[1]):
                self.profile[host].pop(n)
            return ''
        if label == 'cen3-pair':
            tag, payload = rb.scoped_argv(command)
            assert tag == f'{self.tag}-cen3-pair' and 'prod_window.sh pair COORD' in payload[-1]
            self.pair = True
            self.events.append('pair-open')
            return 'CEN3 PAIR OPEN\n'
        if label == 'pair-close':
            self.pair = False
        if '--pid' in command and stdin is not None:
            if not self.pair and self.engine:
                self.rounds += 1
            if self.pair or self.fail == 'pair-ignored':
                self.fnr += 64
                if self.fail == 'pair-ignored':
                    self.rounds += 1
            return json.dumps(stats(self.rounds, self.fnr))
        return super().remote(label, host, command, stdin)

    def bench(self, part: str) -> None:
        self.parts.append(part)
        self.sup.poll()
        files: dict[str, bytes]
        if part == 'bw-smoke':
            row = {'op': 'read', 'gbps_median': 0.0 if self.fail == 'bw-zero' else 200.0}
            files = {'bw-smoke.json': json.dumps({'ops': [row]}).encode()}
        elif part == 'bw':
            files = {'bw.json': json.dumps({'ops': [{'op': o, 'gbps_median': 220.0, 'gbps_max': 230.0}
                                                    for o in ('read', 'write', 'copy')]}).encode(),
                     'bw-gpu-before.csv': b'x\n', 'bw-gpu-after.csv': b'x\n'}
        elif part == 'ncu-smoke':
            if self.fail == 'ncu-absent':
                files = {'ncu-path.txt': b'ABSENT\n'}
            else:
                csv = ('"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size",'
                       '"Grid Size","Device","CC","dram__bytes_read.sum"\n'
                       '"0","1","p","h","void (anonymous namespace)::mrow_kernel<5, false>(int)","1","7","(1, 1, 1)",'
                       '"(1, 1, 1)","0","12.1","1000"\n')
                files = {'ncu-path.txt': b'/usr/local/cuda/bin/ncu\n', 'ncu-version.txt': b'2025.3\n',
                         'ncu-query.txt': b'dram__bytes_read\n',
                         'ncu-metrics.json': json.dumps({'read': 'dram__bytes_read.sum', 'write': None,
                                                         'metrics': ['dram__bytes_read.sum']}).encode(),
                         'ncu-smoke.csv': csv.encode(), 'ncu-smoke.app.txt': b'', 'ncu-smoke.err': b'',
                         'ncu-smoke-target.json': b'{}'}
        else:
            files = {f'ncu-{g}{s}': b'{}' for g in w.NCU_GROUPS for s in ('.csv', '.app.txt', '.err', '-target.json')}
            files['ncu-failures.txt'] = b''
        for host in pf.HOSTS:
            self.bench_out[host].update(files)
        if (self.fail == 'bw-smoke' and part == 'bw-smoke') or (self.fail == 'bw-main' and part == 'bw') \
                or (self.fail == 'ncu-absent' and part == 'ncu-smoke'):
            raise ValueError(f'STOP: cen3-{part} failed rc3; see 0001-cen3-{part}.log')

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        part = label.removeprefix('cen3-')
        if label.startswith('cen3-') and part in w.TIMEOUT_S:
            assert argv[:2] == ['bash', '-c'] and argv[2].count(w.CONTAINER) >= 2
            self.bench(part)
            return ''
        if argv[0] == 'scp' and '/out/' in argv[-2]:
            host, name = argv[-2].split(':')[0], argv[-2].rsplit('/', 1)[1]
            (Path(argv[-1]) / name).write_bytes(self.bench_out[host][name])
            return ''
        if argv[0] == 'scp' and '/profile/' in argv[-2]:
            host, name = argv[-2].split(':')[0], argv[-2].rsplit('/', 1)[1]
            (Path(argv[-1]) / name).write_bytes(self.profile[host][name])
            return ''
        if argv[1:3] == ['-m', 'offload.census.cen3_cells']:
            self.sup.poll()
            output = Path(argv[argv.index('--output') + 1])
            if 'build' in argv:
                output.write_text(json.dumps({'prompts': {'short': {'tokens': 1000}}}))
                return ''
            name = argv[argv.index('--cell') + 1]
            cell = cells.CELLS[name]
            assert inflight == cell.batch
            self.cells.append((name, self.pair))
            caps: list[Json] = [{'kind': k, 'status': 'TAKEN', 'stop_return_rel_s': 2.0} for k in cell.captures]
            for _ in cell.captures:
                self.ns += 1
                for rank, host in enumerate(pf.HOSTS):
                    self.profile[host][trace_name(rank, self.ns)] = b'trace-bytes-' + str(self.ns).encode()
            output.write_text(json.dumps({'cell': name, 'streams': [{'stream': i} for i in range(cell.batch)],
                                          'captures': caps}))
            self.clock.now += 60
            return ''
        return super().call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)

    def plain(self, step: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if step.startswith('cen3-rm-'):
            self.events.append(step)
            return 0, 'CEN3 BENCH ABSENT\n'
        if step.startswith('cen3-dir-clean-'):
            self.events.append(step)
            return 0, 'CLEAN\n'
        if step == 'pair-close':
            self.pair = False
        return super().plain(step, argv, timeout)


@pytest.fixture
def rehearsal(tmp_path: Path, request: pytest.FixtureRequest,
              monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[w.CEN3Live, Remote]]:
    fail = str(getattr(request, 'param', ''))
    monkeypatch.setattr(chain, 'MARKER', tmp_path / 'CHAINED')
    monkeypatch.delenv('WINDOW_CHAIN', raising=False)
    monkeypatch.delenv('WINDOW_CHAIN_LAST', raising=False)
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    out = tmp_path / 'CEN3-W-fake'
    out.mkdir()
    clock = base.Clock()
    trip = tmp_path / 'trip'
    sup = base.NoTunnel(out, clock=clock, monitor=lambda host: [sys.executable, str(monitor), 'low', str(trip)],
                        demand=lambda: 0.0)
    fake = Remote(out, sup, clock, fail, trip)
    live = w.CEN3Live(out, sup, remote='dsv41-deploy-offload', expected='0' * 40, data_dir=tmp_path, tag='tag',
                      call=fake.call, plain=fake.plain, clock=clock)
    import offload.prefill_trace.window_requests as wr
    monkeypatch.setattr(wr, 'corpus', lambda root: ('public fake data', {}))
    try:
        yield live, fake
    finally:
        sup.close()


def assert_w8(result: dict[str, Json], fake: Remote, *, pair: bool) -> None:
    restore = record(result['restore'])
    assert fake.restores == ['start', 'check', 'close'], result
    assert str(restore['status']).startswith('RESTORED'), result
    assert not fake.hold and not fake.pair
    assert str(restore['pair_close']).startswith('PRODUCTION HOLD OK' if pair else 'not opened')
    assert fake.events.index('cen3-rm-sparkhead') < fake.events.index('experiment-stop'), \
        'bench containers go before the quiet proof'


def test_full_window_runs_all_three_parts_and_the_drafter_down_segment(rehearsal: tuple[w.CEN3Live, Remote]) -> None:
    live, fake = rehearsal
    result = cc.window(live, w.body(live))
    assert 'stop' not in result, result.get('stop')
    parts = record(result['parts'])
    assert {k: record(v)['status'] for k, v in parts.items()} == {'c': 'DONE', 'a': 'DONE', 'b': 'DONE'}
    assert fake.parts == ['bw-smoke', 'bw', 'ncu-smoke', 'ncu']
    assert fake.events.index('production-stop') < fake.events.index('start-cen3')
    assert [c for c, _ in fake.cells] == [*cells.VERIFY, cells.TARGET_ONLY, cells.RECOVERY]
    assert dict(fake.cells)[cells.TARGET_ONLY] is True and dict(fake.cells)[cells.RECOVERY] is False
    down = record(record(parts['b'])['drafter_down'])
    assert down['pair'] == 'CLOSED: PRODUCTION HOLD OK'
    assert record(record(down[cells.TARGET_ONLY])['target_only'])['fallback_not_ready_delta'] == 64
    assert fake.events.index('pair-open') < fake.events.index('pair-close') < fake.events.index('experiment-stop')
    traces = json.loads((live.out / 'traces-manifest.json').read_text())
    assert len(traces) == 2 * 4 and {t['cell'] for t in traces} == {'cen3-short-b1', 'cen3-32k-b1', 'cen3-32k-b4',
                                                                     cells.TARGET_ONLY}
    assert not any(fake.profile.values()), 'verified trace copies leave the Sparks'
    bench = json.loads((live.out / 'bench-manifest.json').read_text())
    assert {b['file'] for b in bench} >= {'bench/sparkhead/bw.json', 'bench/sparkworker-wired/ncu-dense.csv'}
    assert record(result['clock'])['status'] == 'OK'
    assert_w8(result, fake, pair=True)
    assert record(result['restore'])['run_dir_sparkhead'] == 'rc0: CLEAN'


@pytest.mark.parametrize('rehearsal,part,reason', [('ncu-absent', 'a', 'absent'), ('bw-smoke', 'c', 'smoke failed'),
                                                   ('bw-zero', 'c', 'no positive rate')], indirect=['rehearsal'])
def test_a_failed_smoke_skips_its_part_and_the_window_goes_on(rehearsal: tuple[w.CEN3Live, Remote], part: str,
                                                              reason: str) -> None:
    live, fake = rehearsal
    result = cc.window(live, w.body(live))
    assert 'stop' not in result, result.get('stop')
    parts = record(result['parts'])
    assert str(record(parts[part])['status']).startswith('SKIPPED') and reason in str(record(parts[part])['status'])
    assert record(parts['b'])['status'] == 'DONE'
    assert_w8(result, fake, pair=True)


@pytest.mark.parametrize('rehearsal', ['bw-main'], indirect=True)
def test_a_failure_after_a_passing_smoke_stops_and_restores(rehearsal: tuple[w.CEN3Live, Remote]) -> None:
    live, fake = rehearsal
    result = cc.window(live, w.body(live))
    assert 'bandwidth bench failed after a passing smoke' in str(result['stop'])
    assert 'start-cen3' not in fake.events
    assert_w8(result, fake, pair=False)


@pytest.mark.parametrize('rehearsal', ['pair-ignored'], indirect=True)
def test_remote_rounds_during_the_pair_window_stop_and_still_close_the_pair(rehearsal: tuple[w.CEN3Live, Remote]) -> None:
    live, fake = rehearsal
    result = cc.window(live, w.body(live))
    assert 'pair window not in force' in str(result['stop'])
    assert fake.events.count('pair-close') == 2, 'closed in the segment and again by the restore'
    assert cells.RECOVERY not in dict(fake.cells)
    assert_w8(result, fake, pair=True)


def test_supervisor_stops_are_never_read_as_a_failed_smoke(rehearsal: tuple[w.CEN3Live, Remote]) -> None:
    live, fake = rehearsal
    real = fake.bench

    def stopped(part: str) -> None:
        real(part)
        raise ValueError('STOP: STOP file present')
    fake.bench = stopped        # type: ignore[method-assign]
    result = cc.window(live, w.body(live))
    assert 'STOP file present' in str(result['stop'])
    assert_w8(result, fake, pair=False)
    assert ck.STOP_COUNTERS
