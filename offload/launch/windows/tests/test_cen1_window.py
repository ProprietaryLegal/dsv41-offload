"""CEN1 eager expert-count window, CPU only: the config over fixtures, the smoke rule, and the real CEN1Live/Supervisor
over the NBW/PF1 rehearsal fakes with a simulated VLLM_EXL3_MOE_STATS file per rank (no SSH, HTTP, Spark or GPU)."""
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

from offload.census import cen1_cells as cells
from offload.census import experts
from offload.launch.windows import cen1_window as w
from offload.launch.windows import cen_common as cc
from offload.launch.windows import chain
from offload.launch.windows import pf1_window as pf
from offload.launch.windows.tests import test_nbw_config as nbw_config
from offload.launch.windows.tests import test_nbw_rehearsal as rb
from offload.launch.windows.tests import test_nbw_window as pure
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, record

CONFIG = pf.ROOT / w.CONFIG


# ---------------------------------------------------------------- config (shell, CPU fixtures)

def source(tmp_path: Path, extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    prod = tmp_path / 'production.env'
    prod.write_text(f'OFFLOAD_DOCKER_ENV={shlex.quote(nbw_config.ENV)}\nOFFLOAD_DSPARK_ENV={shlex.quote(nbw_config.DRAFT)}\n'
                    'OFFLOAD_DOCKER_ARGS="-v /production/code:/engine:ro"\n'
                    'PROD_KV_BYTES=6442450944\nPROD_MAX_LEN=327680\nPROD_K1=1\nPROD_K1_PROFILE=327680\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('NBW_', 'CEN1_', 'AK_', 'EXP_', 'EXTRA_'))}
    env.update({'NBW_PROD_ENV': str(prod), 'CEN1_TAG': 'CEN1-W-20261028T000000Z', **(extra or {})})
    script = f'source {shlex.quote(str(CONFIG))} || exit\n' + '\n'.join(
        f'printf "%s\\n" "${{{key}-}}"' for key in
        ('EXP_DOCKER_ENV', 'EXTRA_DOCKER_ARGS', 'EXP_EXTRA_ARGS', 'AK_MAX_REQS', 'MAX_LEN_OVERRIDE', 'KV_BYTES_OVERRIDE',
         'OVERLAYS', 'LABEL'))
    return subprocess.run(['bash', '-c', script], env=env, cwd=pf.ROOT, capture_output=True, text=True, check=False)


def test_config_is_production_plus_exactly_the_cen1_additions(tmp_path: Path) -> None:
    result = source(tmp_path)
    assert result.returncode == 0, result.stderr
    env, mounts, extra, ak, cap, kv, overlays, label = result.stdout.splitlines()
    added = env.removeprefix(nbw_config.ENV + ' ' + nbw_config.DRAFT + ' ')
    assert dict(token.split('=', 1) for token in added.split()) == {k: v for k, v in w.ENV.items()
                                                                   if k != 'VLLM_DSV41_AK_MAX_REQS'}
    assert ak == '0', '0047 off through serve_exp.sh AK_MAX_REQS (no duplicate VLLM_DSV41_AK_MAX_REQS key)'
    assert mounts == '-v /production/code:/engine:ro -v $HOME/dsv41-offload-run/cen1/CEN1-W-20261028T000000Z:/cen1:rw'
    assert (extra, cap, kv, overlays, label) == ('--enforce-eager', '327680', '6442450944', '', 'CEN1-eager-stats')


@pytest.mark.parametrize('extra', [{'CEN1_TAG': ''}, {'CEN1_TAG': 'bad/tag'}, {'AK_MAX_REQS': '4'},
                                   {'EXP_EXTRA_ARGS': '--x'}, {'NBW_ROLE': 'I'}])
def test_config_refuses_inherited_or_missing_settings(tmp_path: Path, extra: dict[str, str]) -> None:
    result = source(tmp_path, extra)
    assert result.returncode != 0 and 'CEN1 STOP' in result.stderr, result.stdout


# ---------------------------------------------------------------- simulated stats file

class Stats:
    """One rank's VLLM_EXL3_MOE_STATS file: forward.py's cumulative per-pid line every 2,000 calls."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.lines: list[str] = []
        self.sums: dict[int, list[float]] = {}

    def flush(self, m: int, live: int) -> None:
        distinct = max(6.0 * live / m, 0.5 * experts.uniform(live))
        b = self.sums.setdefault(m, [0.0, 0.0, 0.0])
        b[0] += experts.FLUSH
        b[1] += experts.FLUSH * distinct
        b[2] += experts.FLUSH * 6 * live
        self.lines.append(json.dumps({'pid': self.pid, 'calls_by_tokens': {str(k): int(v[0]) for k, v in sorted(self.sums.items())},
                                      'mean_distinct_experts': {str(k): round(v[1] / v[0], 2) for k, v in sorted(self.sums.items())},
                                      'mean_routed_slots': {str(k): round(v[2] / v[0], 2) for k, v in sorted(self.sums.items())}}))

    @property
    def text(self) -> str:
        return ''.join(x + '\n' for x in self.lines)


class Remote(rb.Remote):
    LINES_PER_BATCH = 3

    def __init__(self, out: Path, sup: pf.Supervisor, clock: base.Clock, fail: str, trip: Path) -> None:
        super().__init__(out, sup, clock, fail, trip)
        self.engine = False
        self.depth = 5
        self.stats = {host: Stats(100 + i) for i, host in enumerate(pf.HOSTS)}
        self.batches: list[tuple[str, int]] = []
        self.dirs: set[str] = set()

    def identity(self, rank: int) -> dict[str, Json]:
        row = pure.production(rank)
        record(row['env'])['VLLM_DSV41_AK_MAX_REQS'] = '4'
        if self.engine:
            row['id'] = f'cen1-{rank}'
            row['argv'] = str(row['argv']).replace('--port 8000', '--port 8001 --enforce-eager')
            record(row['mounted'])['/serve.sh'] = '9' * 64
            record(row['env']).update(w.ENV)
        return row

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        run = f'~/{w.RUN}/{self.tag}'
        if label == 'hold-open':
            tag, payload = rb.scoped_argv(command)
            assert f'CEN1-WINDOW {self.tag}:' in payload[-1]
            self.hold = f'{base.iso(self.clock())} DSV41-OFFLOAD (session coordinator): CEN1-WINDOW {self.tag}: x'
            self.original_hold = self.hold
            self.events.append('hold')
            return 'opened'
        if label.startswith('run-dirs-'):
            assert f'test ! -e {run}' in command
            return '/dev/x 100 1 99999999 1% /home\n'
        if label.startswith('cen1-dir-'):
            assert f'mkdir {run}' in command
            self.dirs.add(host)
            return 'fixed:5\n'
        if label == 'cen1-start':
            tag, payload = rb.scoped_argv(command)
            assert tag == f'{self.tag}-cen1-engine' and f'CEN1_TAG={self.tag}' in payload[-1] and w.CONFIG in payload[-1]
            assert self.dirs == set(pf.HOSTS), 'the /cen1 bind sources exist before docker run'
            self.engine = True
            (self.out / 'cmd').mkdir(exist_ok=True)
            (self.out / 'cmd' / f'{len(list((self.out / "cmd").iterdir())) + 1:04d}-{label}.log').write_text(
                'spark clock cap 2300 MHz\nsparkworker clock cap 2300 MHz\n')
            return 'ready'
        if label.startswith('cen1-mounts-'):
            return json.dumps([{'Type': 'bind', 'Destination': '/cen1', 'RW': True,
                                'Source': f'/home/u/{w.RUN}/{self.tag}'}])
        if '-depth-' in label:
            policy = shlex.split(command.split(' > ')[0])[-1]
            self.depth = int(policy.split(':')[1])
            return policy + '\n'
        if label.endswith(f'-lines-{host}'):
            return f'{len(self.stats[host].lines)}\n'
        if label.endswith(f'-read-{host}'):
            return self.stats[host].text
        if label.startswith('stats-size-'):
            return f'{len(self.stats[host].text.encode())}\n'
        if label.startswith('stats-hash-'):
            return json.dumps({w.STATS: hashlib.sha256(self.stats[host].text.encode()).hexdigest()})
        return super().remote(label, host, command, stdin)

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        if argv[1:3] == ['-m', 'offload.census.cen1_cells']:
            self.sup.poll()
            output = Path(argv[argv.index('--output') + 1])
            if 'build' in argv:
                output.write_text(json.dumps({'prompts': {k: {'tokens': 1000} for k in ('short', '32k-0')}}))
                return ''
            cell = argv[argv.index('--cell') + 1]
            size = len(cells.CELLS[cell])
            assert inflight == size
            self.batches.append((cell, self.depth))
            m = cells.window_rows(size)
            live = m if self.fail == 'depth-ignored' else cells.live_rows(size, min(self.depth, cells.speculative_k(size)))
            for stats in self.stats.values():
                for _ in range(0 if self.fail == 'no-stats' else self.LINES_PER_BATCH):
                    stats.flush(m, live)
            self.clock.now += 100 if self.fail == 'slow' else 20
            output.write_text(json.dumps({'cell': cell, 'streams': [{'stream': i} for i in range(size)]}))
            return ''
        if argv[0] == 'scp' and w.STATS in argv[-2]:
            host = argv[-2].split(':')[0]
            (Path(argv[-1]) / w.STATS).write_text(self.stats[host].text)
            return ''
        return super().call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)

    def plain(self, step: str, argv: list[str], timeout: float) -> tuple[int, str]:
        if step.startswith('cen1-dir-clean-'):
            self.events.append(step)
            return 0, 'CLEAN\n'
        return super().plain(step, argv, timeout)


@pytest.fixture
def rehearsal(tmp_path: Path, request: pytest.FixtureRequest,
              monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[w.CEN1Live, Remote]]:
    fail = str(getattr(request, 'param', ''))
    monkeypatch.setattr(chain, 'MARKER', tmp_path / 'CHAINED')
    monkeypatch.delenv('WINDOW_CHAIN', raising=False)
    monkeypatch.delenv('WINDOW_CHAIN_LAST', raising=False)
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    out = tmp_path / 'CEN1-W-fake'
    out.mkdir()
    clock = base.Clock()
    trip = tmp_path / 'trip'
    sup = base.NoTunnel(out, clock=clock, monitor=lambda host: [sys.executable, str(monitor), 'low', str(trip)],
                        demand=lambda: 0.0)
    fake = Remote(out, sup, clock, fail, trip)
    live = w.CEN1Live(out, sup, remote='dsv41-deploy-offload', expected='0' * 40, data_dir=tmp_path, tag='tag',
                      call=fake.call, plain=fake.plain, clock=clock)
    import offload.prefill_trace.window_requests as wr
    monkeypatch.setattr(wr, 'corpus', lambda root: ('public fake data', {}))
    try:
        yield live, fake
    finally:
        sup.close()


def assert_w8(result: dict[str, Json], fake: Remote) -> None:
    assert fake.restores == ['start', 'check', 'close'], result
    assert str(record(result['restore'])['status']).startswith('RESTORED'), result
    assert 'pair-close' not in fake.events and not fake.hold


def test_smoke_rule_over_simulated_files() -> None:
    a, b = Stats(1), Stats(2)
    for s in (a, b):
        for _ in range(4):
            s.flush(6, 6)
    assert w.smoke_check([a.text, b.text], [0, 0], 1, 5)['expected_live_rows'] == 6
    with pytest.raises(ValueError, match='pure at live rows'):
        w.smoke_check([a.text, b.text], [0, 0], 1, 0)
    with pytest.raises(ValueError, match='no whole decode interval'):
        w.smoke_check(['', ''], [0, 0], 1, 0)
    assert [w.settle_batches(d) for d in range(6)] == [1, 1, 1, 2, 2, 2]


def test_full_window_cuts_every_depth_collects_and_analyzes(rehearsal: tuple[w.CEN1Live, Remote]) -> None:
    live, fake = rehearsal
    result = cc.window(live, w.body(live))
    assert 'stop' not in result, result
    assert result['status'] == 'COMPLETE' and record(result['clock'])['status'] == 'OK'
    rows = record(result['cells'])
    assert set(rows) == {f'{c}-d{d}' for c in cells.ORDER for d in cells.DEPTHS[c]}
    assert all(record(r)['status'] == 'OK' for r in rows.values()), rows
    assert all(min(cast(list[int], record(r)['flush_lines'])) >= w.NEED_LINES for r in rows.values())
    smoke = record(result['smoke'])
    assert set(smoke) == {'fixed:5', 'fixed:0'}
    assert fake.batches[0] == ('short-b1', 5) and ('short-b1', 0) in fake.batches[:8]
    assert_w8(result, fake)
    assert record(result['restore'])['run_dir_sparkhead'] == 'rc0: CLEAN'
    analysis = w.analyze(live.out)
    doc = json.loads((live.out / 'cen1-experts.json').read_text())
    assert record(analysis['cross_rank'])['verdict'] == doc['cross_rank']['verdict'] == 'CONSISTENT'
    seg = record(doc['segments'])
    for c in cells.ORDER:
        for d in cells.DEPTHS[c]:
            curve = cast(list[Json], record(seg[f'{c}-d{d}:rank0'])['curve'])
            assert [record(x)['live_rows'] for x in curve] == [cells.live_rows(len(cells.CELLS[c]), d)], (c, d)
            assert record(seg[f'{c}-d{d}:rank1'])['off_depth_calls'] == 0
    assert 'public' not in (live.out / 'cen1-experts.json').read_text()


@pytest.mark.parametrize('rehearsal,match', [('no-stats', 'no VLLM_EXL3_MOE_STATS line'),
                                             ('depth-ignored', 'not in force')], indirect=['rehearsal'])
def test_smoke_failures_stop_and_restore(rehearsal: tuple[w.CEN1Live, Remote], match: str) -> None:
    live, fake = rehearsal
    result = cc.window(live, w.body(live))
    assert match in str(result['stop'])
    assert_w8(result, fake)
    assert str(record(result['restore'])['run_dir_sparkhead']).startswith('KEPT'), 'uncollected stats stay as evidence'
    assert 'cells' not in result


@pytest.mark.parametrize('rehearsal', ['slow'], indirect=True)
def test_depths_that_cannot_fit_are_skipped_and_the_window_completes(rehearsal: tuple[w.CEN1Live, Remote]) -> None:
    live, fake = rehearsal
    result = cc.window(live, w.body(live))
    assert 'stop' not in result, result
    rows = record(result['cells'])
    skipped = [k for k, v in rows.items() if record(v)['status'] == 'SKIPPED_TIME']
    assert skipped and skipped[-1] == '32k-b4-d4', 'the B4 cell is the first to go'
    measured = [k for k, v in rows.items() if record(v)['status'] != 'SKIPPED_TIME']
    assert measured[0] == 'short-b1-d0'
    assert_w8(result, fake)
    cuts = json.loads((live.out / 'cuts.json').read_text())['cuts']
    assert {c['cell'] for c in cuts} == set(measured)


def test_entry_point_is_dry_by_default_and_refuses_without_the_coordinator(
        capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    assert w.main([]) == 0
    assert 'DRY RUN' in capsys.readouterr().out
    monkeypatch.delenv('WINDOW_COORDINATOR', raising=False)
    assert w.main(['--execute']) == 1
    assert 'WINDOW_COORDINATOR' in capsys.readouterr().err
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    assert w.main(['--precheck']) == 1
    assert 'run single' in capsys.readouterr().err
