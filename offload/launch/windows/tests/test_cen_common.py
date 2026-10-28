"""CEN shared window lifecycle (cen_common), CPU only: the real CenLive/Supervisor over the NBW/PF1 rehearsal fakes
(no SSH, HTTP, Spark, x86host or GPU). Single windows only; every path after the HOLD ends in the strict W8 restore."""
from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import cen_common as cc
from offload.launch.windows import chain
from offload.launch.windows import nbw_window as n
from offload.launch.windows import pf1_window as pf
from offload.launch.windows.tests import test_nbw_rehearsal as rb
from offload.launch.windows.tests import test_nbw_window as pure
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, record

PRODUCTION_COMMIT = 'a' * 40
PRODUCTION_TAG = 'prod-test-20261027'
ENV = {'VLLM_TEST_KNOB': '1'}


class Probe(cc.CenLive):
    MARKER = 'CENT-WINDOW'
    TIMING = cc.Timing(50 * 60, 55 * 60, 75 * 60)

    @property
    def pin_files(self) -> tuple[str, ...]:
        return cc.SHARED_PINS

    def run_dirs(self) -> tuple[str, ...]:
        return (f'dsv41-offload-run/cent/{self.tag}',)

    def start_cent(self) -> None:
        current = self.start_engine('cent', 'CENT=1', 'offload/launch/configs/NBW-native.env', budget_s=600)
        for rank, row in enumerate(current):
            cc.runtime_identity(self.production[rank], row, env=ENV, eager=True)
        self.accept_start('cent', current)


class Remote(rb.Remote):
    """NBW's rehearsal remote with the CEN commands: own HOLD marker, run-directory probe, generic engine start."""

    def __init__(self, out: Path, sup: pf.Supervisor, clock: base.Clock, fail: str, trip: Path) -> None:
        super().__init__(out, sup, clock, fail, trip)
        self.engine = False

    def identity(self, rank: int) -> dict[str, Json]:
        row = pure.production(rank)
        if self.engine:
            row['id'] = f'cent-{rank}'
            row['argv'] = str(row['argv']).replace('--port 8000', '--port 8001 --enforce-eager')
            record(row['mounted'])['/serve.sh'] = '9' * 64
            record(row['env']).update(ENV if self.fail != 'env-drift' else {'VLLM_TEST_KNOB': '2'})
        return row

    def remote(self, label: str, host: str, command: str, stdin: str | None) -> str:
        if label == 'ledger':
            assert PRODUCTION_COMMIT in command and PRODUCTION_TAG in command and n.PROD_COMMIT not in command
            if self.fail == 'existing-HOLD':
                raise ValueError('existing HOLD')
            return ''
        if label == 'hold-open':
            tag, payload = rb.scoped_argv(command)
            assert tag == self.tag + '-hold' and f'CENT-WINDOW {self.tag}:' in payload[-1]
            assert 'os.link(candidate, hold)' in payload[-1] and ' require-serving; set -a;' in payload[-1]
            self.hold = f'{base.iso(self.clock())} DSV41-OFFLOAD (session coordinator): CENT-WINDOW {self.tag}: test'
            self.original_hold = self.hold
            self.events.append('hold')
            if self.fail == 'hold-reply':
                raise ValueError('lost HOLD reply')
            return 'opened'
        if label.startswith('run-dirs-'):
            assert f'test ! -e ~/dsv41-offload-run/cent/{self.tag}' in command
            return '/dev/x 100 1 99999999 1% /home\n'
        if label == 'cent-start':
            tag, payload = rb.scoped_argv(command)
            assert tag == f'{self.tag}-cent-engine' and 'CENT=1' in payload[-1] and 'start_exp.sh' in payload[-1]
            self.engine = True
            self.events.append('start-cent')
            worker = 2150 if self.fail == 'clock' else 2300
            text = f'spark clock cap 2300 MHz\nsparkworker clock cap {worker} MHz\nready\n'
            (self.out / 'cmd').mkdir(exist_ok=True)
            (self.out / 'cmd' / f'{len(list((self.out / "cmd").iterdir())) + 1:04d}-{label}.log').write_text(text)
            if self.fail == 'start':
                raise ValueError('Spark start failed rc7')
            return text
        return super().remote(label, host, command, stdin)


@pytest.fixture
def rehearsal(tmp_path: Path, request: pytest.FixtureRequest,
              monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Probe, Remote]]:
    fail = str(getattr(request, 'param', ''))
    monkeypatch.setattr(chain, 'MARKER', tmp_path / 'CHAINED')
    monkeypatch.delenv('WINDOW_CHAIN', raising=False)
    monkeypatch.delenv('WINDOW_CHAIN_LAST', raising=False)

    def no_host() -> bool:
        raise AssertionError('a single CEN window never probes chain hosts')
    for name in ('hold_exists', 'placeholders_present', 'production_serving'):
        monkeypatch.setattr(chain, name, no_host)
    monitor = tmp_path / 'monitor.py'
    monitor.write_text(base.MONITOR)
    out = tmp_path / 'CENT-fake'
    out.mkdir()
    clock = base.Clock()
    trip = tmp_path / 'trip'
    sup = base.NoTunnel(out, clock=clock, monitor=lambda host: [sys.executable, str(monitor), 'low', str(trip)],
                        demand=lambda: 0.0)
    fake = Remote(out, sup, clock, fail, trip)
    live = Probe(out, sup, remote='dsv41-deploy-offload', expected='0' * 40, data_dir=tmp_path, tag='tag',
                 production_commit=PRODUCTION_COMMIT, production_tag=PRODUCTION_TAG, call=fake.call, plain=fake.plain,
                 clock=clock)
    import offload.prefill_trace.window_requests as wr
    monkeypatch.setattr(wr, 'corpus', lambda root: ('public fake data', {}))
    try:
        yield live, fake
    finally:
        sup.close()


def body(live: Probe, *, after: Callable[[], None] | None = None) -> Callable[[dict[str, Json]], None]:
    def run(result: dict[str, Json]) -> None:
        live.start_cent()
        result['clock'] = live.clock_cap('cent')
        if after is not None:
            after()
    return run


def assert_w8(result: dict[str, Json], fake: Remote, *, stopped: bool) -> None:
    restore = record(result['restore'])
    assert fake.restores == (['start', 'check', 'close'] if stopped else ['check', 'close']), result
    assert str(restore['status']).startswith('RESTORED'), result
    assert str(restore['pair_close']).startswith('not opened')
    assert 'pair-close' not in fake.events, 'no drafter-down segment: the x86host production worker is never touched'
    assert not fake.hold
    assert fake.events.index('close') < fake.events.index('commit-close') < fake.events.index('cancel-close')


def test_timing_and_argv_helpers() -> None:
    cc.Timing(1, 2, 3).check()
    for bad in (cc.Timing(0, 2, 3), cc.Timing(3, 2, 4), cc.Timing(1, 3, 3), cc.Timing(1, 2, n.RESTORE_S + 1)):
        with pytest.raises(ValueError, match='deadlines'):
            bad.check()
    argv = "vllm serve m --port 8001 \\\n --profiler-config '{\"a\":1}' --x 1"
    assert shlex.split(cc.strip_flag(argv, '--profiler-config', '{"a":1}')) == ['vllm', 'serve', 'm', '--port', '8001',
                                                                              '--x', '1']
    with pytest.raises(ValueError, match='value differs'):
        cc.strip_flag(argv, '--profiler-config', '{"a":2}')
    with pytest.raises(ValueError, match='exactly one'):
        cc.strip_flag(argv, '--enforce-eager', '')


def test_runtime_identity_allows_only_the_declared_changes() -> None:
    prod = pure.production(0)
    live = pure.production(0)
    live['id'] = 'x'
    live['argv'] = str(live['argv']).replace('--port 8000', '--port 8001 --enforce-eager')
    record(live['mounted'])['/serve.sh'] = '9' * 64
    record(live['env']).update(ENV)
    cc.runtime_identity(prod, live, env=ENV, eager=True)
    with pytest.raises(ValueError, match='lacks --enforce-eager'):
        cc.runtime_identity(prod, {**live, 'argv': str(live['argv']).replace(' --enforce-eager', '')}, env=ENV,
                            eager=True)
    with pytest.raises(ValueError, match='argv differs'):
        cc.runtime_identity(prod, live, env=ENV, eager=False)
    with pytest.raises(ValueError, match='VLLM_TEST_KNOB'):
        cc.runtime_identity(prod, live, env={}, eager=True)
    record(live['mounted'])['/opt/dsv41-ofl/libofl.so'] = '0' * 64
    with pytest.raises(ValueError, match='mounted engine bytes'):
        cc.runtime_identity(prod, live, env=ENV, eager=True)


def test_full_single_window_restores_with_w8_and_reads_the_clock(rehearsal: tuple[Probe, Remote]) -> None:
    live, fake = rehearsal
    result = cc.window(live, body(live))
    assert 'stop' not in result, result
    assert result['status'] == 'COMPLETE' and result['chain_mode'] == 'production' and result['end_action'] == 'restore'
    assert record(result['clock'])['status'] == 'OK' and record(result['clock'])['mhz'] == 2300
    assert_w8(result, fake, stopped=True)
    assert cc.exit_status(dict(result, post_close_watch={'status': 'PASS'})) == (0, '')
    assert cc.exit_status(result)[0] == 2, 'a restore without the post-close watch is never reported clean'
    rollback = json.loads((live.out / 'rollback.json').read_text())
    assert rollback['production_commit'] == PRODUCTION_COMMIT and rollback['tag'] == PRODUCTION_TAG
    deadlines = json.loads((live.out / 'deadlines.json').read_text())
    assert deadlines['restore_by'] - deadlines['opened'] == 75 * 60
    assert deadlines['submit'] - deadlines['opened'] == 50 * 60


@pytest.mark.parametrize('rehearsal', ['clock'], indirect=True)
def test_a_start_off_the_registered_clock_is_inconclusive_never_a_stop(rehearsal: tuple[Probe, Remote]) -> None:
    live, fake = rehearsal
    result = cc.window(live, body(live))
    assert 'stop' not in result and str(record(result['clock'])['status']).startswith('INCONCLUSIVE')
    assert_w8(result, fake, stopped=True)


@pytest.mark.parametrize('rehearsal', ['env-drift', 'start'], indirect=True)
def test_failure_after_the_stop_restores_production(rehearsal: tuple[Probe, Remote]) -> None:
    live, fake = rehearsal
    result = cc.window(live, body(live))
    assert 'differs from production' in str(result['stop']) or 'Spark start failed' in str(result['stop'])
    assert_w8(result, fake, stopped=True)
    assert cc.exit_status(dict(result, post_close_watch={'status': 'PASS'}))[0] == 1


def test_failure_before_any_stop_checks_and_closes_without_a_start(rehearsal: tuple[Probe, Remote]) -> None:
    live, fake = rehearsal

    def boom(result: dict[str, Json]) -> None:
        raise ValueError('body refused before any engine start')
    result = cc.window(live, boom)
    assert 'refused' in str(result['stop'])
    assert_w8(result, fake, stopped=False)


@pytest.mark.parametrize('rehearsal', ['hold-reply'], indirect=True)
def test_a_lost_hold_reply_is_found_by_the_reread_and_restored(rehearsal: tuple[Probe, Remote]) -> None:
    live, fake = rehearsal
    result = cc.window(live, body(live))
    assert 'lost HOLD reply' in str(result['stop'])
    assert result['end_action'] == 'restore'
    assert_w8(result, fake, stopped=False)


@pytest.mark.parametrize('rehearsal', ['existing-HOLD'], indirect=True)
def test_preflight_failure_touches_nothing(rehearsal: tuple[Probe, Remote]) -> None:
    live, fake = rehearsal
    result = cc.window(live, body(live))
    assert 'existing HOLD' in str(result['stop'])
    assert result['end_action'] == 'none' and str(record(result['restore'])['status']).startswith('NOT NEEDED')
    assert 'hold' not in fake.events and not fake.restores
    assert cc.exit_status(result) == (1, '')


def test_single_window_rules_refuse_chain_mode_and_a_chain_marker(rehearsal: tuple[Probe, Remote],
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    live, fake = rehearsal
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    result = cc.window(live, body(live))
    assert 'never chained' in str(result['stop']) and result['end_action'] == 'none'
    monkeypatch.delenv('WINDOW_CHAIN')
    chain.MARKER.write_text('{}\n')
    live2 = Probe(live.out, live.sup, remote='dsv41-deploy-offload', expected='0' * 40, data_dir=live.data_dir,
                  tag='tag', production_commit=PRODUCTION_COMMIT, production_tag=PRODUCTION_TAG, call=fake.call,
                  plain=fake.plain, clock=live.clock)
    with pytest.raises(ValueError, match='chain marker present'):
        live2.preflight()
    assert 'hold' not in fake.events


def test_restore_failure_stays_loud(rehearsal: tuple[Probe, Remote]) -> None:
    live, fake = rehearsal
    fake.fail = 'restore-greedy'
    result = cc.window(live, body(live))
    assert record(result['restore'])['status'] == 'FAILED'
    code, message = cc.exit_status(result)
    assert code == 2 and 'NOT VERIFIED' in message
    assert fake.hold, 'a failed restore leaves the HOLD in place'


def test_exit_status_after_a_mid_window_chain_abort() -> None:
    restored = {'status': 'RESTORED: ... (chain abort)', 'post_close_watch': {'status': 'PASS'}}
    assert cc.exit_status({'end_action': 'abort', 'restore': cast(Json, restored)})[0] == 1
    assert cc.exit_status({'end_action': 'abort', 'restore': {'status': 'ALERT: a HOLD exists'}})[0] == 3
    assert cc.exit_status({'end_action': 'abort', 'restore': {'status': 'RESTORED: x'}})[0] == 2
