"""S92's counterexamples as regression tests (WIN2-R2). One test per S92 test, same name, asserting the FIXED behaviour.

Sources: scratch/S92/work/test_s92_counterexamples.py and test_s92_clock.py, which reproduce the defects at 48bc7a7.
CPU only: production on an unreachable loopback port, ssh a script in tmp_path; the real campaign windows are read only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from offload.launch.windows import chain, decision_v2 as d, hold_admission, win2_history as h
from offload.prefill_maxlen.evidence import Json, record

C = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')
REPO = Path(__file__).resolve().parents[4]
PROD = '596ba1cf4e3b11be29b41a57e7f69a5d8efa4799'
PC1 = C / 'PC1-threshold2048-W-20261027T153817Z'


@pytest.fixture(autouse=True)
def isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    marker = tmp_path / 'CHAINED'
    monkeypatch.setattr(chain, 'MARKER', marker)
    for k in ('WINDOW_CHAIN', 'WINDOW_CHAIN_LAST'):
        monkeypatch.delenv(k, raising=False)
    return marker


def _row(window: str, value: float, **extra: Json) -> dict[str, Json]:
    row: dict[str, Json] = {'window': window, 'cell': 'short-b1', 'metric': 'ms_per_round', 'value': value,
                            'max_tokens': 512, 'commit': PROD, 'clock_cap_mhz': 2150, 'date': '20261027',
                            'start': 's1-control', 'repeat': 0}
    row.update(extra)
    return row


def _fake_bin(tmp_path: Path, ssh_rc: int = 0) -> Path:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / 'ssh').write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/ssh.log\nexit {ssh_rc}\n')
    (bin_dir / 'ssh').chmod(0o755)
    (bin_dir / 'abort.sh').write_text(f'#!/bin/sh\necho ABORT >> {tmp_path}/abort.log\n')
    (bin_dir / 'abort.sh').chmod(0o755)
    return bin_dir


def _chained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W-prev', {'sparkhead': 'QUIET', 'sparkworker-wired': 'QUIET'})


# ---------------------------------------------------------------- FIX 1

def test_pre_hold_admission_fails_when_production_is_down_unless_allow_unreachable(tmp_path: Path,
                                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('SCOPED_REMOTE_DEADLINE', '9999999999')
    (tmp_path / 'dsv41-deploy').mkdir()

    def down(path: str) -> str:
        raise subprocess.CalledProcessError(7, ['curl', path])
    monkeypatch.setattr(hold_admission, 'fetch', down)
    monkeypatch.setattr(time, 'sleep', lambda s: None)               # hold_admission's drain poll
    with pytest.raises(subprocess.CalledProcessError):
        hold_admission.admit('s92-tag', '0' * 32, 'S92 test', 30.0, allow_unreachable=chain.allow_unreachable('production'))
    hold_admission.admit('s92-tag', '1' * 32, 'S92 test', 30.0, allow_unreachable=chain.allow_unreachable('chained'))
    assert (tmp_path / 'dsv41-deploy/HOLD').exists()


def test_chain_state_after_pre_hold_failure_is_accepted_by_next_runner_preflight(monkeypatch: pytest.MonkeyPatch,
                                                                               tmp_path: Path) -> None:
    """Fixed: the end decision is 'abort' (restore), never 'NOT NEEDED'; after the abort no preflight says 'chained'."""
    _chained(tmp_path, monkeypatch)
    assert chain.end_action(stopped=True, hold_written=False) == 'abort'
    assert chain.restore_required(stopped=True, hold_written=False)
    result = chain.abort(tmp_path / 'abort', hold=lambda: False, recover=lambda out, log: {'start': 'PASS'},
                         watch=lambda out, log: {'status': 'PASS'}, log=lambda m: None)
    assert str(result['status']).startswith('RESTORED') and chain.read_marker() is None
    with pytest.raises(ValueError, match='no chain marker'):
        chain.preflight(production_serving=False, hold_exists=False, placeholders_present=True)


def test_chain_prev_ok_refusal_in_chained_state_restores_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 isolate: Path) -> None:
    """Fixed: the stale-container refusal in the chained state exits 3 and runs the chain abort."""
    _chained(tmp_path, monkeypatch)
    camp = tmp_path / 'campaign'
    (camp / 'launch').mkdir(parents=True)
    (camp / 'launch/no-stale-containers.sh').write_text('#!/bin/bash\necho stale-check rc=2\nexit 2\n')
    (camp / 'launch/no-stale-containers.sh').chmod(0o755)
    prev = tmp_path / 'W-prev'
    prev.mkdir()
    (prev / 'summary.json').write_text(json.dumps({'chain': 'CHAINED', 'restore': {'status': 'CHAINED: production left down'}}))
    (tmp_path / 'prev.txt').write_text(str(prev) + '\n')
    bin_dir = _fake_bin(tmp_path)
    env = {**{k: v for k, v in os.environ.items() if k not in ('TMUX', 'TMUX_PANE', 'WINDOW_CHAIN_LAST')},
           'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'WINDOW_CHAIN': '1', 'DSV41_CHAIN_MARKER': str(isolate),
           'DSV41_CHAIN_PRODUCTION_URL': 'http://127.0.0.1:9/v1/models', 'DSV41_CHAIN_PY': sys.executable,
           'DSV41_CHAIN_CAMPAIGN': str(camp), 'DSV41_CHAIN_ABORT': str(bin_dir / 'abort.sh')}
    r = subprocess.run(['bash', str(REPO / 'offload/launch/windows/chain_prev_ok.sh'), 's92-no-such-tmux',
                        str(tmp_path / 'prev.txt')], capture_output=True, text=True, env=env, timeout=120, cwd=REPO)
    assert 'chain gate: chained' in r.stdout, r.stdout + r.stderr
    assert r.returncode == 3 and 'CHAIN STOPPED: PRODUCTION DOWN' in r.stdout
    assert (tmp_path / 'abort.log').read_text().splitlines() == ['ABORT']


# ---------------------------------------------------------------- FIX 2

def test_backfill_drops_the_production_commit_of_pc1_and_up3_windows(tmp_path: Path) -> None:
    if not (PC1 / 'summary.json').exists():
        pytest.skip('historical window not on this host')
    summary = json.loads((PC1 / 'summary.json').read_text())
    assert summary.get('production_commit') is None and summary['production'] == PROD
    h.backfill([PC1], tmp_path / 'a.jsonl')
    rows = d.archive_rows(tmp_path / 'a.jsonl')
    assert rows and all(r['commit'] == PROD for r in rows)


def test_backfill_would_record_an_experiment_commit_as_production(tmp_path: Path) -> None:
    w = tmp_path / 'X-W-20261027T000000Z'
    w.mkdir()
    rep = {'server_decode_ms': 4000.0, 'spec_request_rounds': 100.0, 'streams': [{'sampling': {'max_tokens': 512}}]}
    (w / 's1-control-short-b1.json').write_text(json.dumps({'repetitions': [rep]}))
    (w / 'summary.json').write_text(json.dumps({'starts': {'s1-control': {'role': 'control'}}, 'experiment_commit': 'e' * 40}))
    with pytest.raises(ValueError, match='no production commit'):
        h.backfill([w], tmp_path / 'a.jsonl')


def test_archive_accepts_rows_without_commit_and_spread_pools_commits(tmp_path: Path) -> None:
    p = tmp_path / 'a.jsonl'
    without = _row('W1', 40.0)
    del without['commit']
    with pytest.raises(ValueError, match='lacks'):
        d.archive_append(p, [without])
    d.archive_append(p, [_row('W2', 40.2), _row('W3', 37.0, commit='b' * 40)])
    with pytest.raises(ValueError, match='commit'):
        d.archive_spread(p, 'short-b1', 'ms_per_round')


def test_backfill_twice_in_one_call_duplicates(tmp_path: Path) -> None:
    if not (PC1 / 'summary.json').exists():
        pytest.skip('historical window not on this host')
    archive = tmp_path / 'a.jsonl'
    assert h.backfill([PC1, PC1], archive) == 64 and len(d.archive_rows(archive)) == 64
    assert h.backfill([PC1], archive) == 0


def _clock_window(tmp: Path, caps: dict[str, int | None]) -> Path:
    w = tmp / 'Z-W-20261027T200000Z'
    (w / 'cmd').mkdir(parents=True)
    starts = {}
    for n, (start, cap) in enumerate(caps.items()):
        starts[start] = {'role': 'control'}
        rep = {'server_decode_ms': 4000.0 + n, 'spec_request_rounds': 100.0, 'streams': [{'sampling': {'max_tokens': 512}}]}
        (w / f'{start}-short-b1.json').write_text(json.dumps({'repetitions': [rep]}))
        line = '' if cap is None else f'spark clock cap {cap} MHz\nsparkworker clock cap {cap} MHz\n'
        (w / f'cmd/00{n}0-{start}-start.log').write_text('starting\n' + line)
    (w / 'summary.json').write_text(json.dumps({'starts': starts, 'production': 'a' * 40}))
    return w


def test_spread_pools_2150_and_2300_rows_silently(tmp_path: Path) -> None:
    p = tmp_path / 'a.jsonl'
    d.archive_append(p, [_row('W1', 39.0), _row('W2', 39.2), _row('W3', 37.0, clock_cap_mhz=2300)])
    with pytest.raises(ValueError, match='clock_cap_mhz'):
        d.archive_spread(p, 'short-b1', 'ms_per_round')
    only = d.archive_spread(p, 'short-b1', 'ms_per_round', where={'clock_cap_mhz': 2150})
    assert only is not None and abs(only - 0.2 / 39.1) < 1e-12


def test_rows_without_clock_cap_are_accepted(tmp_path: Path) -> None:
    row = _row('W', 1.0)
    del row['clock_cap_mhz']
    with pytest.raises(ValueError, match='lacks'):
        d.archive_append(tmp_path / 'a.jsonl', [row])


def test_backfill_records_empty_caps_when_the_start_log_lacks_the_line(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='does not prove one clock cap'):
        h.backfill([_clock_window(tmp_path, {'s1-control': None})], tmp_path / 'a.jsonl')
    assert not (tmp_path / 'a.jsonl').exists()


def test_backfill_unions_caps_across_starts_losing_per_start_provenance(tmp_path: Path) -> None:
    h.backfill([_clock_window(tmp_path, {'s1-control': 2150, 's3-control': 2300})], tmp_path / 'a.jsonl')
    rows = d.archive_rows(tmp_path / 'a.jsonl')
    assert {(r['start'], r['clock_cap_mhz'], r['clock_source']) for r in rows} == {('s1-control', 2150, 'start-log'),
                                                                                  ('s3-control', 2300, 'start-log')}
    assert {json.dumps(r['clock_caps_mhz']) for r in rows} == {'[2150, 2300]'}      # window audit field only


# ---------------------------------------------------------------- FIX 3

def test_reanalyze_ignores_guard_loss_that_decide_window_counts(tmp_path: Path) -> None:
    w = tmp_path / 'Y-W-20261027T000000Z'
    w.mkdir()

    def rep(ms: float, acc: float) -> dict[str, object]:
        return {'server_decode_ms': ms * 100, 'spec_request_rounds': 100.0, 'acceptance_per_round': acc,
                'streams': [{'sampling': {'max_tokens': 512}}]}
    (w / 's1-control-short-b1.json').write_text(json.dumps({'repetitions': [rep(40.0, 1.50), rep(40.2, 1.52), rep(40.1, 1.51)]}))
    (w / 's2-x-short-b1.json').write_text(json.dumps({'repetitions': [rep(38.0, 1.20), rep(38.1, 1.22), rep(38.2, 1.21)]}))
    (w / 'summary.json').write_text(json.dumps({'starts': {'s1-control': {'role': 'control'}, 's2-x': {'role': 'x'}}}))
    data = h.extract(w)['short-b1']
    series = [d.Series('short-b1', m, hib, tuple(data['control'][m]), tuple(data['candidate'][m]), paired=False)
              for m, hib in (('ms_per_round', False), ('accepted_per_round', True))]
    assert h.reanalyze(w, ['short-b1'])['win2_status'] == d.decide_window(series, ['short-b1'])['status'] == 'NOT A CANDIDATE'


# ---------------------------------------------------------------- S92 positive checks

def test_corrupt_or_partial_marker_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolate: Path) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    for text in ('', '{"schema": "dsv41-chain-v1"', 'null'):
        isolate.write_text(text)
        with pytest.raises(ValueError):
            chain.preflight(production_serving=False, hold_exists=False, placeholders_present=True)
    isolate.unlink()
    (tmp_path / 'CHAINED.new').write_text('{}')
    with pytest.raises(ValueError, match='no chain marker'):
        chain.preflight(production_serving=False, hold_exists=False, placeholders_present=True)


@pytest.mark.parametrize('serving,hold', [(True, False), (False, True), (True, True)])
def test_marker_never_admits_live_production_or_hold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                     serving: bool, hold: bool) -> None:
    _chained(tmp_path, monkeypatch)
    for last in ('0', '1'):
        monkeypatch.setenv('WINDOW_CHAIN_LAST', last)
        with pytest.raises(ValueError, match='STOP'):
            chain.preflight(production_serving=serving, hold_exists=hold, placeholders_present=True)


def test_gate_probe_failures_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('PATH', f'{_fake_bin(tmp_path, ssh_rc=255)}:{os.environ["PATH"]}')
    _chained(tmp_path, monkeypatch)
    assert chain.hold_exists() is True and chain.placeholders_present() is False
    with pytest.raises(ValueError, match='HOLD'):
        chain.gate(lambda: False)
    assert record(chain.read_marker() or {})['schema'] == chain.SCHEMA
