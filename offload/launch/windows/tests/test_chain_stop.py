"""WIN2-R2 / S92 FIX 1: a chain that stops outside a window's own HOLD restores production (CPU only).

S92's counterexamples (scratch/S92/work/test_s92_counterexamples.py at 48bc7a7) are kept here inverted: each now shows the
fixed behaviour. Every host probe is a fake: production on an unreachable loopback port, ssh a script in tmp_path.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from offload.launch.windows import chain, hold_admission
from offload.prefill_maxlen.evidence import Json, record

REPO = Path(__file__).resolve().parents[4]
WINDOWS = REPO / 'offload/launch/windows'
OWNER = 'chain-20261027T220000Z-0123abcd'


@pytest.fixture(autouse=True)
def isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    marker = tmp_path / 'CHAINED'
    monkeypatch.setattr(chain, 'MARKER', marker)
    for k in ('WINDOW_CHAIN', 'WINDOW_CHAIN_LAST'):
        monkeypatch.delenv(k, raising=False)
    return marker


def _chain_left_production_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W-prev', {'sparkhead': 'QUIET', 'sparkworker-wired': 'QUIET'}, placeholder_owner=OWNER)


# ---------------------------------------------------------------- end decision

def test_end_action_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert chain.end_action(stopped=True, hold_written=False) == 'none'          # no chain, nothing touched
    assert chain.end_action(stopped=True, hold_written=True) == 'restore'
    assert not chain.restore_required(stopped=True, hold_written=False)
    _chain_left_production_down(tmp_path, monkeypatch)
    assert chain.end_action(stopped=False, hold_written=True) == 'chained'
    assert chain.end_action(stopped=True, hold_written=True) == 'restore'
    assert chain.end_action(stopped=True, hold_written=False) == 'abort'          # S92: production down, nothing restored
    assert chain.restore_required(stopped=True, hold_written=False)
    monkeypatch.setenv('WINDOW_CHAIN_LAST', '1')
    assert chain.end_action(stopped=False, hold_written=True) == 'restore'


def test_torn_marker_still_requires_restore(isolate: Path) -> None:
    isolate.write_text('{"schema": "dsv41-chain-v1"')
    assert chain.marker_present() and chain.end_action(stopped=True, hold_written=False) == 'abort'


# ---------------------------------------------------------------- chained start with production down

def test_chained_start_admits_hold_with_production_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """S92: runners opened the admission HOLD in 'require-serving' mode, so a chained window failed before its HOLD."""
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('SCOPED_REMOTE_DEADLINE', '9999999999')
    (tmp_path / 'dsv41-deploy').mkdir()

    def down(path: str) -> str:
        raise subprocess.CalledProcessError(7, ['curl', path])
    monkeypatch.setattr(hold_admission, 'fetch', down)
    monkeypatch.setattr(time, 'sleep', lambda s: None)              # hold_admission's drain poll
    with pytest.raises(subprocess.CalledProcessError):
        hold_admission.admit('r2-tag', '0' * 32, 'WIN2-R2 test', 30.0, allow_unreachable=chain.allow_unreachable('production'))
    assert not (tmp_path / 'dsv41-deploy/HOLD').exists()
    hold_admission.admit('r2-tag', '1' * 32, 'WIN2-R2 test', 30.0, allow_unreachable=chain.allow_unreachable('chained'))
    assert (tmp_path / 'dsv41-deploy/HOLD').exists()
    with pytest.raises(ValueError, match='unknown chain preflight mode'):
        chain.allow_unreachable('serving')


# ---------------------------------------------------------------- abort helper

class Fakes:
    def __init__(self, *, hold: bool = False, recover_error: Exception | None = None, watch: str = 'PASS') -> None:
        self.hold, self.recover_error, self.watch_status = hold, recover_error, watch
        self.recovered = self.watched = self.ended = 0
        self.lines: list[str] = []

    def hold_probe(self) -> bool:
        return self.hold

    def recover(self, out: Path, log: Callable[[str], None]) -> Mapping[str, Json]:
        self.recovered += 1
        if self.recover_error is not None:
            raise self.recover_error
        return {'start': 'PASS', 'check': 'PASS', 'close': 'PASS'}

    def watch_fn(self, out: Path, log: Callable[[str], None]) -> Mapping[str, Json]:
        self.watched += 1
        return {'status': self.watch_status, 'polls': 30, 'failures': 0, 'recoveries': 0}

    def end(self) -> Mapping[str, Json]:
        self.ended += 1
        return chain.end_chain(run=lambda host, command: (0, 'PLACEHOLDER DOWN\n'))

    def run(self, out: Path) -> dict[str, Json]:
        return chain.abort(out, hold=self.hold_probe, recover=self.recover, watch=self.watch_fn, end=self.end,
                           log=self.lines.append)


def _prev_ok_accepts(summary: Path) -> bool:
    """The exact summary test of chain_prev_ok.sh / prev-ok.sh."""
    s = json.loads(summary.read_text())
    chained = s.get('chain') == 'CHAINED'
    restored = (str(s.get('restore', {}).get('status', '')).startswith('RESTORED')
                and str(s.get('post_close_watch', {}).get('status', '')) == 'PASS')
    return chained or restored


def test_pre_hold_failure_in_chain_restores_and_clears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """S92: a chained window failing before its HOLD said 'NOT NEEDED' and the next preflight said 'chained' again."""
    _chain_left_production_down(tmp_path, monkeypatch)
    stopped, hold_written = True, False
    assert chain.end_action(stopped, hold_written) == 'abort'
    fakes = Fakes()
    result = fakes.run(tmp_path / 'W-now' / 'chain-abort')
    assert fakes.recovered == 1 and fakes.watched == 1 and fakes.ended == 1
    assert str(result['status']).startswith('RESTORED') and chain.read_marker() is None
    assert 'placeholders down' in str(result['status']) and record(result['end_chain'])['placeholder_owner'] == OWNER
    assert record(result['restore'])['status'] == chain.ABORT_RESTORED
    assert _prev_ok_accepts(tmp_path / 'W-now' / 'chain-abort' / 'summary.json')
    with pytest.raises(ValueError, match='not serving and no chain marker'):
        chain.preflight(production_serving=False, hold_exists=False)       # never 'chained' again
    assert chain.preflight(production_serving=True, hold_exists=False) == 'production'


def test_abort_with_hold_alerts_and_touches_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _chain_left_production_down(tmp_path, monkeypatch)
    fakes = Fakes(hold=True)
    result = fakes.run(tmp_path / 'abort')
    assert str(result['status']).startswith('ALERT') and fakes.recovered == fakes.watched == 0
    assert chain.read_marker() is not None and not _prev_ok_accepts(tmp_path / 'abort' / 'summary.json')


@pytest.mark.parametrize('fakes', [Fakes(recover_error=ValueError('W8 check: greedy is not 6/6')), Fakes(watch='FAILED: x')])
def test_abort_keeps_marker_unless_restored_and_watch_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                          fakes: Fakes) -> None:
    _chain_left_production_down(tmp_path, monkeypatch)
    result = fakes.run(tmp_path / 'abort')
    assert not str(result['status']).startswith('RESTORED: chain ended')
    assert chain.read_marker() is not None and not _prev_ok_accepts(tmp_path / 'abort' / 'summary.json')
    assert fakes.ended == 0                       # production unproved: the chain placeholders stay up


def test_abort_without_marker_does_nothing(tmp_path: Path) -> None:
    fakes = Fakes()
    result = fakes.run(tmp_path / 'abort')
    assert str(result['status']).startswith('NOT NEEDED') and fakes.recovered == 0


def test_abort_cli_is_coordinator_only_and_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('WINDOW_COORDINATOR', raising=False)
    assert chain.main(['abort', '--out', str(tmp_path / 'a')]) == 2
    monkeypatch.setenv('WINDOW_COORDINATOR', chain.COORDINATOR)
    assert chain.main(['abort', '--out', str(tmp_path / 'b')]) == 0           # no marker: NOT NEEDED
    _chain_left_production_down(tmp_path, monkeypatch)
    fakes = Fakes(hold=True)
    real = chain.abort
    monkeypatch.setattr(chain, 'abort', lambda out: real(out, hold=fakes.hold_probe, recover=fakes.recover,
                                                         watch=fakes.watch_fn, end=fakes.end, log=fakes.lines.append))
    assert chain.main(['abort', '--out', str(tmp_path / 'c')]) == 4
    fakes.hold, fakes.watch_status = False, 'FAILED: recurrence'
    assert chain.main(['abort', '--out', str(tmp_path / 'd')]) == 3
    fakes.watch_status = 'PASS'
    assert chain.main(['abort', '--out', str(tmp_path / 'e')]) == 0 and chain.read_marker() is None


# ---------------------------------------------------------------- chain_prev_ok.sh end to end (fakes only)

def _fake_campaign(tmp_path: Path, stale_rc: int) -> Path:
    camp = tmp_path / 'campaign'
    (camp / 'launch').mkdir(parents=True)
    script = camp / 'launch/no-stale-containers.sh'
    script.write_text(f'#!/bin/bash\necho stale-check rc={stale_rc}\nexit {stale_rc}\n')
    script.chmod(0o755)
    return camp


def _gate_env(tmp_path: Path, marker: Path, *, stale_rc: int, hold: bool = False) -> dict[str, str]:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    ssh = bin_dir / 'ssh'
    # HOLD probe: exit 1 = a HOLD exists. Placeholder probe (absolute tmux): this chain's placeholder is up.
    ssh.write_text(f'#!/bin/bash\necho "fake ssh $*" >> {tmp_path}/ssh.log\n'
                   f'case "$*" in *"{chain.PLACEHOLDER_TMUX}"*) echo "PLACEHOLDER UP owner={OWNER} pid=4242"; exit 0 ;; esac\n'
                   f'exit {1 if hold else 0}\n')
    ssh.chmod(0o755)
    abort = bin_dir / 'abort.sh'
    abort.write_text(f'#!/bin/sh\necho ABORT-CALLED >> {tmp_path}/abort.log\necho fake abort ran\nexit 0\n')
    abort.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ('WINDOW_CHAIN_LAST', 'TMUX', 'TMUX_PANE')}
    env.update({'PATH': f'{bin_dir}:{env["PATH"]}', 'WINDOW_CHAIN': '1', 'DSV41_CHAIN_MARKER': str(marker),
                'DSV41_CHAIN_PRODUCTION_URL': 'http://127.0.0.1:9/v1/models', 'DSV41_CHAIN_PY': sys.executable,
                'DSV41_CHAIN_CAMPAIGN': str(_fake_campaign(tmp_path, stale_rc)), 'DSV41_CHAIN_ABORT': str(abort)})
    return env


def _prev(tmp_path: Path, summary: dict[str, Json]) -> Path:
    prev = tmp_path / 'W-prev'
    prev.mkdir(exist_ok=True)
    (prev / 'summary.json').write_text(json.dumps(summary))
    prev_file = tmp_path / 'prev.txt'
    prev_file.write_text(str(prev) + '\n')
    return prev_file


def _gate(env: dict[str, str], prev_file: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(['bash', str(WINDOWS / 'chain_prev_ok.sh'), 'win2r2-no-such-tmux', str(prev_file)],
                          capture_output=True, text=True, env=env, timeout=120, cwd=REPO)


CHAINED_SUMMARY: dict[str, Json] = {'chain': 'CHAINED', 'restore': {'status': 'CHAINED: production left down'}}


def test_gate_refusal_in_chained_state_exits_3_and_takes_abort_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                   isolate: Path) -> None:
    """S92: the stale-container refusal in the chained state exited 2, ran no W8 and left production down."""
    _chain_left_production_down(tmp_path, monkeypatch)
    r = _gate(_gate_env(tmp_path, isolate, stale_rc=2), _prev(tmp_path, CHAINED_SUMMARY))
    assert 'chain gate: chained' in r.stdout, r.stdout + r.stderr
    assert r.returncode == 3 and 'CHAIN STOPPED: PRODUCTION DOWN' in r.stdout
    assert (tmp_path / 'abort.log').read_text().splitlines() == ['ABORT-CALLED']


@pytest.mark.parametrize('case', ['bad-summary', 'never-started', 'hold'])
def test_every_refusal_with_marker_exits_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolate: Path, case: str) -> None:
    _chain_left_production_down(tmp_path, monkeypatch)
    env = _gate_env(tmp_path, isolate, stale_rc=0, hold=case == 'hold')
    summary: dict[str, Json] = {'restore': {'status': 'FAILED'}} if case == 'bad-summary' else CHAINED_SUMMARY
    prev_file: Path | str = _prev(tmp_path, summary)
    if case == 'never-started':
        prev_file = tmp_path / 'empty.txt'
        prev_file.write_text('')
    r = _gate(env, prev_file)
    assert r.returncode == 3 and 'CHAIN STOPPED: PRODUCTION DOWN' in r.stdout, r.stdout + r.stderr
    assert (tmp_path / 'abort.log').exists()


def test_refusal_without_marker_exits_2_and_never_aborts(tmp_path: Path, isolate: Path) -> None:
    r = _gate(_gate_env(tmp_path, isolate, stale_rc=0), '-')          # production down, no marker: STOP as prev-ok
    assert r.returncode == 2 and 'CHAIN STOPPED' not in r.stdout and not (tmp_path / 'abort.log').exists()


def test_clean_chained_gate_admits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolate: Path) -> None:
    _chain_left_production_down(tmp_path, monkeypatch)
    r = _gate(_gate_env(tmp_path, isolate, stale_rc=0), _prev(tmp_path, CHAINED_SUMMARY))
    assert r.returncode == 0 and 'chain gate: chained' in r.stdout and not (tmp_path / 'abort.log').exists()


def test_chain_abort_sh_without_marker_is_a_no_op(tmp_path: Path, isolate: Path) -> None:
    env = _gate_env(tmp_path, isolate, stale_rc=0)
    env['WINDOW_COORDINATOR'] = chain.COORDINATOR
    r = subprocess.run(['bash', str(WINDOWS / 'chain_abort.sh'), str(tmp_path / 'abort-out')], capture_output=True,
                       text=True, env=env, timeout=120, cwd=REPO)
    assert r.returncode == 0 and 'NOT NEEDED' in r.stdout, r.stdout + r.stderr
    assert json.loads((tmp_path / 'abort-out/summary.json').read_text())['marker_path'] == str(isolate)
    env.pop('WINDOW_COORDINATOR')
    r = subprocess.run(['bash', str(WINDOWS / 'chain_abort.sh'), str(tmp_path / 'abort-out2')], capture_output=True,
                       text=True, env=env, timeout=120, cwd=REPO)
    assert r.returncode == 2 and 'coordinator-only' in r.stderr


# ---------------------------------------------------------------- S92 positive checks kept as regression tests

def test_corrupt_or_partial_marker_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolate: Path) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    for text in ('', '{"schema": "dsv41-chain-v1"', 'null'):
        isolate.write_text(text)
        with pytest.raises(ValueError):
            chain.preflight(production_serving=False, hold_exists=False)
    isolate.unlink()
    (tmp_path / 'CHAINED.new').write_text('{}')            # crash between temp write and rename
    with pytest.raises(ValueError, match='no chain marker'):
        chain.preflight(production_serving=False, hold_exists=False)
    chain.clear()
    assert not (tmp_path / 'CHAINED.new').exists()


@pytest.mark.parametrize('serving,hold', [(True, False), (False, True), (True, True)])
def test_marker_never_admits_live_production_or_hold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                     serving: bool, hold: bool) -> None:
    _chain_left_production_down(tmp_path, monkeypatch)
    for last in ('0', '1'):
        monkeypatch.setenv('WINDOW_CHAIN_LAST', last)
        with pytest.raises(ValueError, match='STOP'):
            chain.preflight(production_serving=serving, hold_exists=hold)


def test_failed_hold_probe_counts_as_hold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'ssh').write_text('#!/bin/sh\nexit 255\n')
    (bin_dir / 'ssh').chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}:{os.environ["PATH"]}')
    _chain_left_production_down(tmp_path, monkeypatch)
    assert chain.hold_exists() is True
    with pytest.raises(ValueError, match='HOLD'):
        chain.gate(lambda: False)


def test_recovery_tags_are_unique_and_valid() -> None:
    a = chain._tag(Path('/c/PC1-threshold2048-W-20261027T153817Z/chain-abort'), 'abort')
    b = chain._tag(Path('/c/NBW2-W-20261027T210000Z/chain-abort'), 'abort')
    long = chain._tag(Path('/c/' + 'X' * 120 + '/chain-abort'), 'watch')
    assert a != b and a.endswith('-chain-abort-abort') and len(long) <= 64
    assert all(re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_-]{0,63}', t) for t in (a, b, long))
