"""SP1-WINDOW runner: pure checks (item 3) and the CPU rehearsal with fakes (item 5). No ssh, HTTP or service."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import shlex
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from offload.gb10_budget import test_trace_budget as synth
from offload.gb10_budget import trace_budget as tb
from offload.gb10_budget import window_cells as wc
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import sp1_window as s
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, record

PROD_ARGV = ("#!/usr/bin/env bash\nset -e\nexec vllm serve '/srv/dsv41/legal-v2a' --port 8000 \\\n"
             "  --max-model-len 327680 --max-num-seqs 4 \\\n  --limit-mm-per-prompt '{\"image\":4}' --enable-prompt-tokens-details")
RANK0 = 'dp0_pp0_tp0_dcp0_ep0_rank0.1792999000000000001.pt.trace.json.gz'
RANK1 = 'dp0_pp0_tp1_dcp0_ep1_rank1.1792999000000000002.pt.trace.json.gz'


def experiment_argv(extra: str = '') -> str:
    return PROD_ARGV + f" --profiler-config {shlex.quote(s.PROFILER_JSON)}{extra}"


def test_strip_profiler_removes_exactly_the_reviewed_flag() -> None:
    stripped = s.strip_profiler(experiment_argv())
    assert '--profiler-config' not in stripped
    assert shlex.split(stripped) == shlex.split(PROD_ARGV.replace('\\\n', ' '))
    with pytest.raises(ValueError, match='exactly one'):
        s.strip_profiler(PROD_ARGV)
    with pytest.raises(ValueError, match='exactly one'):
        s.strip_profiler(experiment_argv(f" --profiler-config {shlex.quote(s.PROFILER_JSON)}"))
    other = s.PROFILER_JSON.replace('"max_iterations":16', '"max_iterations":64')
    with pytest.raises(ValueError, match='differs from the reviewed JSON'):
        s.strip_profiler(PROD_ARGV + f' --profiler-config {shlex.quote(other)}')
    spaced = json.dumps(s.PROFILER_CONFIG)     # same object, other bytes: refused (byte identity is pinned)
    with pytest.raises(ValueError, match='differs from the reviewed JSON'):
        s.strip_profiler(PROD_ARGV + f' --profiler-config {shlex.quote(spaced)}')


def test_trace_mount_must_be_the_one_rw_bind() -> None:
    good: dict[str, Json] = {'Type': 'bind', 'Source': '/home/u/dsv41-offload-run/sp1-profile', 'Destination': '/sp1-profile', 'RW': True}
    ro: dict[str, Json] = {'Type': 'bind', 'Source': '/home/u/x', 'Destination': '/srv/dsv41/legal-v2a', 'RW': False}
    assert s.trace_mount([ro, good])['rw'] is True
    bads: list[dict[str, Json]] = [{**good, 'RW': False}, {**good, 'Source': '/home/u/dsv41-offload-run/pf1-trace/x'}, {**good, 'Type': 'volume'}]
    for bad in bads:
        with pytest.raises(ValueError):
            s.trace_mount([ro, bad])
    with pytest.raises(ValueError, match='exactly one'):
        s.trace_mount([ro])
    with pytest.raises(ValueError, match='exactly one'):
        s.trace_mount([good, good])


def test_listing_and_caps() -> None:
    files, free = s.listing(f'f 1000 {RANK0}\nFREE {20 * 1024**2}\n')
    assert files == {RANK0: 1000} and free == 20 * 1024**2
    assert s.new_traces(files, set(), 0, 1, free) == [(RANK0, 1000)]
    assert s.new_traces(files, {RANK0}, 0, 0, free) == []
    with pytest.raises(ValueError, match='unexpected directory'):
        s.listing(f'd 4096 sub\nFREE 1\n')
    with pytest.raises(ValueError, match='unexpected file'):
        s.listing('f 10 notes.txt\nFREE 1\n')
    with pytest.raises(ValueError, match='FREE'):
        s.listing(f'f 1000 {RANK0}\n')
    with pytest.raises(ValueError, match='not global rank 0'):
        s.new_traces({RANK1: 10}, set(), 0, 1, 20 * 1024**2)
    with pytest.raises(ValueError, match='new trace files, expected 2'):
        s.new_traces({RANK0: 10}, set(), 0, 2, 20 * 1024**2)
    with pytest.raises(ValueError, match='size'):
        s.new_traces({RANK0: s.FILE_CAP + 1}, set(), 0, 1, 20 * 1024**2)
    with pytest.raises(ValueError, match='exceeds'):
        s.new_traces({RANK0: 1, 'rank0.1.pt.trace.json.gz': s.DIR_CAP}, set(), 0, 2, 20 * 1024**2)
    with pytest.raises(ValueError, match='free disk'):
        s.new_traces({RANK0: 1}, set(), 0, 1, s.DISK_FLOOR_KB - 1)
    older = 'dp0_pp0_tp0_dcp0_ep0_rank0.1792999000000000000.pt.trace.json.gz'
    assert [n for n, _ in s.new_traces({RANK0: 1, older: 1}, set(), 0, 2, 20 * 1024**2)] == [older, RANK0]


def test_taken_counts_only_started_captures() -> None:
    cell: dict[str, Json] = {'captures': [{'kind': 'prefill-early', 'status': 'SKIPPED_MEMORY'},
                         {'kind': 'decode', 'status': 'TAKEN', 'stop_return_rel_s': 3.0}]}
    assert [c['kind'] for c in s.taken(cell)] == ['decode']
    with pytest.raises(ValueError, match='no completed stop'):
        s.taken({'captures': [{'kind': 'decode', 'status': 'TAKEN'}]})
    with pytest.raises(ValueError, match='unknown'):
        s.taken({'captures': [{'kind': 'decode', 'status': 'WEIRD', 'stop_return_rel_s': 1}]})


def test_reap_and_hold_text_match_each_other() -> None:
    live = object.__new__(s.SP1Live)
    live.tag = 'SP1-W-20261026T200000Z'
    reap = live.reap_script()
    assert 'PF1' not in reap and reap.count('SP1-profile[.]env') == 2
    purpose = live.hold_purpose()
    opener = re.search(r"reap '(window\[\.\]sh open SP1-WINDOW [^']+)'", reap)
    assert opener is not None and re.search(opener[1], f'offload/launch/window.sh open {purpose}')
    assert re.search(r'start_exp[.]sh offload/launch/configs/SP1-profile[.]env', f'bash offload/launch/start_exp.sh {s.CONFIG}')
    assert pf.REAP.count('@TAG@') == 2 and '@TAG@' not in reap


def test_dry_run_prints_plan_without_side_effects(capsys: pytest.CaptureFixture[str]) -> None:
    assert s.main(['--dry-run']) == 0
    out = capsys.readouterr().out
    assert out.startswith('DRY RUN') and 'restore.sh start' in out and 'prof-150k-b1' in out
    assert s.main([]) == 0


def test_execute_refuses_before_any_remote_action(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv('WINDOW_COORDINATOR', raising=False)
    assert s.main(['--execute']) == 1
    assert 'STOP before any remote action' in capsys.readouterr().err
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    assert s.main(['--execute', '--out', '/tmp/not-the-campaign', '--expected-experiment-commit', 'a' * 40]) == 1
    assert s.main(['--execute', '--out', str(pf.CAMPAIGN / 'SP1-W-new'), '--expected-experiment-commit', 'a' * 40,
                   '--remote-repo', 'dsv41-deploy']) == 1


# ---------------------------------------------------------------- item 5: CPU rehearsal of the live backend with fakes
# The real SP1Live/Supervisor/window() run against PF1's fake command layer (no ssh, scp, HTTP or service); every case
# after the HOLD must end in W8 restore start|check|close with its four verified strings.

DF = '/dev/nvme0n1p2 3800000000 1000000000 {free} 21% /home\n'


class SP1Remote(base.FakeRemote):
    """PF1's fake Sparks/x86host plus the SP1 profile directory, mounts, profiler log lines and the cell driver."""

    def __init__(self, out: Path, sup: pf.Supervisor, clock: base.Clock, fail: str = '', trip: Path | None = None) -> None:
        super().__init__(out, sup, clock, fail, trip=trip)
        self.dirs: dict[str, dict[str, bytes] | None] = {h: None for h in pf.HOSTS}
        self.started = {h: 0 for h in pf.HOSTS}
        self.ns = 1792999000000000000
        self.cells: list[str] = []

    def identity(self, rank: int) -> dict[str, Json]:
        ident = super().identity(rank)
        if self.arm is not None:
            config = s.PROFILER_JSON.replace('16', '64') if self.fail == 'profiler-json' else s.PROFILER_JSON
            ident['argv'] = str(ident['argv']).rstrip('\n') + f' --profiler-config {shlex.quote(config)}\n'
            if self.fail == 'identity':
                record(ident['mounted'])[pf.DIST + 'vllm/models/deepseek_v41/attention.py'] = '0' * 64
        return ident

    def call(self, label: str, argv: list[str], *, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        if argv[1:3] == ['-m', 'offload.gb10_budget.window_cells']:
            self.sup.poll()
            self.commands.append(label)
            return self.driver(argv, inflight)
        if argv[0] == 'scp' and s.PROFILE_DIR in argv[-2]:
            self.sup.poll()
            self.commands.append(label)
            host, name = argv[-2].split(':')[0], argv[-2].rsplit('/', 1)[1]
            files = self.dirs[host]
            assert files is not None
            (Path(argv[-1]) / name).write_bytes(files[name])
            return ''
        return super().call(label, argv, stdin=stdin, timeout=timeout, inflight=inflight)

    def driver(self, argv: list[str], inflight: int) -> str:
        args = {argv[i]: argv[i + 1] for i in range(4, len(argv) - 1) if argv[i].startswith('--')}
        assert '--execute' in argv and argv[argv.index('--data-dir') + 2] in ('build', 'run')
        if 'build' in argv:
            Path(args['--output']).write_text(json.dumps({'prompts': {p.key: {'tokens': p.target - 3, 'cut': p.target * 3}
                                                                      for p in wc.PROMPTS.values()}}))
            return ''
        name = args['--cell']
        cell = wc.CELLS[name]
        assert inflight == cell.batch and float(args['--deadline-utc']) == self.sup.abort_at - 5 * 60
        self.cells.append(name)
        if self.trip is not None and cell.captures:
            self.trip.touch()          # the fake Spark monitors now report MemAvailable below 4 GiB
        if self.fail.startswith('late') and cell.captures:
            self.clock.now += (46 if self.fail == 'late-abort' else 39) * 60
        end = time.monotonic() + .3
        self.sup.inflight = inflight
        try:
            while time.monotonic() < end:
                self.sup.poll()
        finally:
            self.sup.inflight = 0
        captures: list[Json] = []
        t = 5.0
        for kind in cell.captures:
            captures.append({'kind': kind, 'status': 'TAKEN', 'start_call_rel_s': t, 'start_return_rel_s': t + .01,
                             'stop_call_rel_s': t + 3, 'stop_return_rel_s': t + 3.5, 'mem_available_kb': {}})
            for rank, host in enumerate(pf.HOSTS):
                files = self.dirs[host]
                assert files is not None, 'capture without a profile directory'
                trace = synth.decode_trace(tid=11 + rank) if kind == 'decode' else synth.prefill_trace()
                text = trace.text().encode()
                self.ns += 1
                files[f'dp0_pp0_tp{rank}_dcp0_ep{rank}_rank{rank}.{self.ns}.pt.trace.json.gz'] = gzip.compress(text, mtime=0)
                self.started[host] += 1
            t += 20
        streams = [synth.stream(.1 * i, 10.0, 30, 30.0) for i in range(cell.batch)]
        Path(args['--output']).write_text(json.dumps({'type': 'cell', 'cell': name, 'batch': cell.batch,
                                                      'profiled': bool(cell.captures), 'warmup': cell.warmup,
                                                      'streams': streams, 'captures': captures}))
        return ''

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        files = self.dirs.get(host)
        if 'window.sh open' in command:
            self.hold = f'{base.iso(self.clock())} DSV41-OFFLOAD (session coordinator): {command.split("SP1-WINDOW ")[1][:-1]}'
            return 'window open'
        if f'test ! -e ~/{s.PROFILE_DIR}' in command:
            assert files is None
            return DF.format(free=(9 if self.fail == 'preflight-disk' else 300) * 1024**2)
        if f'mkdir ~/{s.PROFILE_DIR}' in command:
            assert files is None
            self.dirs[host] = {}
            return ''
        if '{{json .Mounts}}' in command:
            rw = self.fail != 'mount-ro'
            return json.dumps([{'Type': 'bind', 'Source': '/home/u/models', 'Destination': '/srv/dsv41/legal-v2a', 'RW': False},
                               {'Type': 'bind', 'Source': f'/home/u/{s.PROFILE_DIR}', 'Destination': s.MOUNT, 'RW': rw}])
        if command.startswith(f'cd ~/{s.PROFILE_DIR} && find'):
            assert files is not None
            free = (7 if self.fail == 'trace-disk' and self.started[host] else 300) * 1024**2
            listed = {n: (s.FILE_CAP + 1 if self.fail == 'trace-size' else len(b)) for n, b in files.items()}
            return ''.join(f'f {size} {n}\n' for n, size in listed.items()) + f'FREE {free}\n'
        if stdin == pf.IMAGE_BASES and command.startswith(f'cd ~/{s.PROFILE_DIR} && python3 - '):
            assert files is not None
            return json.dumps({n: hashlib.sha256(files[n]).hexdigest() for n in shlex.split(command.split('python3 - ', 1)[1])})
        if command.startswith(f'cd ~/{s.PROFILE_DIR} && rm -f -- '):
            assert files is not None
            for n in shlex.split(command.split('rm -f -- ', 1)[1]):
                del files[n]
            return ''
        if command == f'rmdir ~/{s.PROFILE_DIR}':
            assert files == {}
            self.dirs[host] = None
            return ''
        if 'docker logs dsv41-serve' in command and 'grep -c -F' in command:
            if shlex.quote(s.PROFILER_ON) in command:
                return '1\n' if host == 'sparkhead' else '0\n'
            assert shlex.quote(s.PROFILER_STARTED) in command
            return f'{self.started[host] + (1 if self.fail == "early-profiler" else 0)}\n'
        return super().remote(label, host, command, stdin)

    def plain(self, step: str, command: list[str], timeout: float) -> tuple[int, str]:
        if command[0] == 'ssh':
            self.unsupervised.append((step, command[-2], command[-1]))
            if step == 'reap':
                assert "reap 'window[.]sh open SP1-WINDOW tag:' 'HOLD opener'" in command[-1]
                return 0, 'reap done\n'
            if step.startswith('jit-wait-'):
                return 0, 'no JIT prebuild running\n'
            if step.startswith('profile-dir-'):
                files = self.dirs[command[-2]]
                assert step == f'profile-dir-{command[-2]}' and files is not None
                if files:
                    return 0, f'LEFT {len(files)} files\n'
                self.dirs[command[-2]] = None
                return 0, 'CLEAN\n'
            assert step == 'hold-reread'
            return 0, self.hold
        assert command == ['bash', str(pf.RESTORE), step]
        self.restores.append(step)
        return 0, base.RESTORE_OK[step]


@pytest.fixture
def rehearsal(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[tuple[s.SP1Live, SP1Remote, base.Clock]]:
    params = cast(dict[str, str], getattr(request, 'param', {}))
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    out = tmp_path / 'SP1-W-rehearsal'
    out.mkdir()
    clock = base.Clock()
    trip = tmp_path / 'trip'
    mode = params.get('monitor', 'ok')
    sup = base.NoTunnel(out, clock=clock, monitor=lambda host: [sys.executable, str(monitor), mode, str(trip)],
                        demand=lambda: 0.0)
    fake = SP1Remote(out, sup, clock, params.get('fail', ''), trip=trip if 'monitor' in params else None)
    live = s.SP1Live(out, sup, remote='dsv41-deploy-offload', expected='0' * 40, data_dir=tmp_path, tag='tag',
                     stage_bundle=False, call=fake.call, plain=fake.plain, clock=clock)
    import offload.prefill_trace.window_requests as requests_module
    original = requests_module.corpus
    requests_module.corpus = lambda root: ('x', {})
    try:
        yield live, fake, clock
    finally:
        requests_module.corpus = original
        sup.close()


def rehearse(live: s.SP1Live) -> dict[str, Any]:
    rows: list[dict[str, Json]] = []
    return s.window(live, rows.append)


RESTORED = ['start', 'check', 'close']


def test_rehearsal_full_sequence_restores_and_feeds_the_analysis(rehearsal: tuple[s.SP1Live, SP1Remote, base.Clock]) -> None:
    live, fake, _clock = rehearsal
    result = rehearse(live)
    assert 'stop' not in result, result.get('stop')
    assert result['status'] == 'COMPLETE' and fake.restores == RESTORED
    assert str(result['restore']['status']).startswith('RESTORED: K1 START VERIFIED')
    assert fake.cells == list(wc.ORDER)
    assert result['finish']['never_started_before_profiled_cells'] == {h: 0 for h in pf.HOSTS}
    assert all(d is None for d in fake.dirs.values())                                # collected, removed, rmdir'd
    manifest = json.loads((live.out / 'traces-manifest.json').read_text())
    assert len(manifest) == 2 * sum(len(wc.CELLS[n].captures) for n in wc.ORDER) == 12
    deadlines = json.loads((live.out / 'deadlines.json').read_text())
    assert deadlines['submission'] - deadlines['opened'] == 40 * 60 and deadlines['abort'] - deadlines['opened'] == 45 * 60
    assert [c for c in fake.commands if c.endswith('-start')] == ['sp1-327-start']
    analysis: Any = tb.analyze_window(live.out)                                       # item 4 on the rehearsal output
    assert set(analysis['captures']) == {f'{r["cell"]}:{r["capture"]}' for r in manifest}
    assert analysis['next_work']['ranking'] == 'OK' and analysis['next_work']['next'][0] == 'prefill-grouped-kernel'


@pytest.mark.parametrize('rehearsal', [{'fail': 'start'}], indirect=True)
def test_rehearsal_failed_start_restores(rehearsal: tuple[s.SP1Live, SP1Remote, base.Clock]) -> None:
    live, fake, _clock = rehearsal
    original = fake.remote
    def failing(label: str, host: str, command: str, stdin: str | None) -> str:
        if 'bash offload/launch/start_exp.sh offload/launch/configs/SP1-profile.env' in command:
            raise ValueError('STOP: sp1-327-start failed rc1; see log')
        return original(label, host, command, stdin)
    fake.remote = failing  # type: ignore[method-assign]
    result = rehearse(live)
    assert 'sp1-327-start failed' in result['stop'] and fake.restores == RESTORED and fake.cells == []
    assert all(result['restore'][f'profile_dir_{h}'] == 'rc0: CLEAN' for h in pf.HOSTS)


@pytest.mark.parametrize('rehearsal', [{'monitor': 'low'}], indirect=True)
def test_rehearsal_memory_guard_trip_restores(rehearsal: tuple[s.SP1Live, SP1Remote, base.Clock]) -> None:
    live, fake, _clock = rehearsal
    result = rehearse(live)
    assert 'below 4 GiB' in result['stop'] and fake.restores == RESTORED
    assert fake.cells[-1] == 'prof-short-b1'                                          # tripped in the first profiled cell
    minima = json.loads((live.out / 'arm-memory.json').read_text())['sp1-327']
    assert min(minima.values()) < 4 * 1024**2                                     # whichever Spark tripped first


@pytest.mark.parametrize('rehearsal,message', [({'fail': 'trace-size'}, 'size'), ({'fail': 'trace-disk'}, 'free disk')],
                         indirect=['rehearsal'])
def test_rehearsal_trace_size_and_disk_trips_restore_and_keep_uncollected_traces(
        rehearsal: tuple[s.SP1Live, SP1Remote, base.Clock], message: str) -> None:
    live, fake, _clock = rehearsal
    result = rehearse(live)
    assert message in result['stop'] and fake.restores == RESTORED and fake.cells[-1] == 'prof-short-b1'
    assert result['restore']['profile_dir_sparkhead'] == 'rc0: LEFT 1 files'           # evidence kept for the coordinator
    assert not (live.out / 'traces-manifest.json').exists()


@pytest.mark.parametrize('rehearsal', [{'fail': 'preflight-disk'}], indirect=True)
def test_rehearsal_low_disk_before_the_hold_touches_nothing(rehearsal: tuple[s.SP1Live, SP1Remote, base.Clock]) -> None:
    live, fake, _clock = rehearsal
    result = rehearse(live)
    assert '< 10 GiB free' in result['stop'] and fake.restores == [] and fake.hold == ''
    assert str(result['restore']['status']).startswith('NOT NEEDED')


@pytest.mark.parametrize('rehearsal,message', [({'fail': 'late-abort'}, '45-minute SP1 abort bound'),
                                               ({'fail': 'late-submit'}, 'does not fit before the submission bound')],
                         indirect=['rehearsal'])
def test_rehearsal_deadline_expiry_restores(rehearsal: tuple[s.SP1Live, SP1Remote, base.Clock], message: str) -> None:
    live, fake, _clock = rehearsal
    result = rehearse(live)
    assert message in result['stop'] and fake.restores == RESTORED
    assert fake.cells[-1] == 'prof-short-b1'


@pytest.mark.parametrize('rehearsal,message', [
    ({'fail': 'identity'}, 'mounted engine bytes differ from production'),
    ({'fail': 'profiler-json'}, 'differs from the reviewed JSON'),
    ({'fail': 'mount-ro'}, 'not the read-write bind'),
    ({'fail': 'early-profiler'}, 'profiler already started')], indirect=['rehearsal'])
def test_rehearsal_identity_mismatch_restores(rehearsal: tuple[s.SP1Live, SP1Remote, base.Clock], message: str) -> None:
    live, fake, _clock = rehearsal
    result = rehearse(live)
    assert message in result['stop'] and fake.restores == RESTORED and fake.cells == []
