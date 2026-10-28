"""S96's counterexamples for the WIN2-R2 chain stop path (scratch/S96/work/test_s96_chain.py, written against 2b48eec),
kept here inverted: each now shows the fixed behaviour. CPU only; every path under pytest tmp; production is a loopback
fake or an unreachable loopback port.

CX-1 (S96 FIX 1): chain_prev_ok.sh's automatic abort was refused as coordinator-only when WINDOW_COORDINATOR was not
     exported before the gate (exit 3, nothing restored, marker kept). The gate is now coordinator-only and refuses at
     once; with it, a refusal in the chained state runs the REAL chain_abort.sh.
CX-2 (S96 FIX 2a): with the marker present and production SERVING, the abort drained production, opened a HOLD (K24
     pauses) and failed at the guarded stop. It now ALERTs and touches nothing.
CX-3 (S96 FIX 2b): the abort was not single-shot; every gate re-ran the W8 recovery after a failed abort. It now runs once
     per stopped chain (the post-close watch rule); `abort --again` by hand, or the chain's end, resets it.
"""
from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import pytest

from offload.launch.windows import chain
from offload.prefill_maxlen.evidence import Json

WINDOWS = Path(chain.__file__).resolve().parent
REPO = WINDOWS.parents[2]
OWNER = 'chain-20261027T220000Z-0123abcd'


@pytest.fixture(autouse=True)
def isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    marker = tmp_path / 'CHAINED'
    monkeypatch.setattr(chain, 'MARKER', marker)
    for k in ('WINDOW_CHAIN', 'WINDOW_CHAIN_LAST', 'WINDOW_COORDINATOR'):
        monkeypatch.delenv(k, raising=False)
    return marker


def _chained_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    chain.finish_chained(tmp_path / 'W-prev', {'sparkhead': 'QUIET', 'sparkworker-wired': 'QUIET'}, placeholder_owner=OWNER)
    monkeypatch.delenv('WINDOW_CHAIN')


def _env(tmp_path: Path, marker: Path, *, url: str, stale_rc: int, ssh_rc: int, coordinator: bool) -> dict[str, str]:
    """Fake ssh: rc `ssh_rc` for the HOLD probe; this chain's placeholders UP when it answers at all (rc 0)."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / 'ssh').write_text(f'#!/bin/bash\necho "fake ssh" >> {tmp_path}/ssh.log\n'
                                 f'case "$*" in *"{chain.PLACEHOLDER_TMUX}"*) [ {ssh_rc} = 0 ] && '
                                 f'echo "PLACEHOLDER UP owner={OWNER} pid=4242" ;; esac\nexit {ssh_rc}\n')
    (bin_dir / 'ssh').chmod(0o755)
    camp = tmp_path / 'campaign'
    (camp / 'launch').mkdir(parents=True, exist_ok=True)
    (camp / 'launch/no-stale-containers.sh').write_text(f'#!/bin/bash\necho stale-check rc={stale_rc}\nexit {stale_rc}\n')
    (camp / 'launch/no-stale-containers.sh').chmod(0o755)
    env = {k: v for k, v in os.environ.items()
           if k not in ('WINDOW_CHAIN_LAST', 'TMUX', 'TMUX_PANE', 'WINDOW_COORDINATOR', 'DSV41_CHAIN_ABORT')}
    env.update({'PATH': f'{bin_dir}:{env["PATH"]}', 'WINDOW_CHAIN': '1', 'DSV41_CHAIN_MARKER': str(marker),
                'DSV41_CHAIN_PRODUCTION_URL': url, 'DSV41_CHAIN_PY': sys.executable, 'DSV41_CHAIN_CAMPAIGN': str(camp)})
    if coordinator:
        env['WINDOW_COORDINATOR'] = chain.COORDINATOR
    return env


def _gate(tmp_path: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    prev = tmp_path / 'W-prev'
    prev.mkdir(exist_ok=True)
    (prev / 'summary.json').write_text(json.dumps({'chain': 'CHAINED', 'restore': {'status': 'CHAINED: production left down'}}))
    (tmp_path / 'prev.txt').write_text(str(prev) + '\n')
    return subprocess.run(['bash', str(WINDOWS / 'chain_prev_ok.sh'), 's96-no-such-tmux', str(tmp_path / 'prev.txt')],
                          capture_output=True, text=True, env=env, timeout=120, cwd=REPO)


def _abort_summaries(tmp_path: Path) -> list[dict[str, Json]]:
    return [json.loads(p.read_text()) for p in sorted((tmp_path / 'campaign').glob('CHAIN-ABORT-*/summary.json'))]


# ---------------------------------------------------------------- CX-1 / FIX 1

def test_cx1_gate_refuses_at_once_without_window_coordinator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             isolate: Path) -> None:
    """Fixed: no probe, no wait, no abort that would be refused later; the RUN template must export it first."""
    _chained_marker(tmp_path, monkeypatch)
    r = _gate(tmp_path, _env(tmp_path, isolate, url='http://127.0.0.1:9/v1/models', stale_rc=2, ssh_rc=0,
                             coordinator=False))
    assert r.returncode == 2 and 'chain_prev_ok.sh is coordinator-only' in r.stdout, r.stdout + r.stderr
    assert 'CHAIN STOPPED' not in r.stdout and not (tmp_path / 'ssh.log').exists() and not _abort_summaries(tmp_path)
    assert isolate.exists()


def test_cx1_refusal_in_chained_state_runs_the_real_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                          isolate: Path) -> None:
    """The REAL chain_abort.sh (no DSV41_CHAIN_ABORT fake). Its HOLD probe fails (fake ssh rc 255), so it ALERTs and
    touches nothing (exit 4); the gate reports it and stops the chain (exit 3). At 2b48eec this abort was refused."""
    _chained_marker(tmp_path, monkeypatch)
    r = _gate(tmp_path, _env(tmp_path, isolate, url='http://127.0.0.1:9/v1/models', stale_rc=2, ssh_rc=255,
                             coordinator=True))
    assert r.returncode == 3 and 'CHAIN STOPPED: PRODUCTION DOWN' in r.stdout, r.stdout + r.stderr
    assert 'chain abort exit 4' in r.stdout and 'coordinator-only' not in r.stdout + r.stderr
    [summary] = _abort_summaries(tmp_path)
    assert str(summary['status']).startswith('ALERT: a HOLD exists') and isolate.exists()


# ---------------------------------------------------------------- CX-2 / FIX 2a: production serving with the marker

@pytest.fixture
def serving_production() -> Iterator[str]:
    """A loopback /v1/models that answers like K1 production (max_model_len 327680)."""
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps({'data': [{'id': 'k1', 'max_model_len': chain.K1_CAP}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}/v1/models'
    finally:
        server.shutdown()


def test_cx2_gate_with_serving_production_alerts_without_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                  isolate: Path, serving_production: str) -> None:
    """Real chain_abort.sh: no HOLD (fake ssh rc 0), production serving -> ALERT, exit 4; no recovery phase, no abort
    record, marker kept for inspection."""
    _chained_marker(tmp_path, monkeypatch)
    r = _gate(tmp_path, _env(tmp_path, isolate, url=serving_production, stale_rc=0, ssh_rc=0, coordinator=True))
    assert 'STOP: chain marker present but production is serving' in r.stderr, r.stdout + r.stderr
    assert r.returncode == 3 and 'chain abort exit 4' in r.stdout, r.stdout + r.stderr
    [summary] = _abort_summaries(tmp_path)
    assert 'production is serving' in str(summary['status'])
    assert not chain.abort_attempt().exists() and isolate.exists()
    assert not list((tmp_path / 'campaign').glob('CHAIN-ABORT-*/abort-commands'))       # no recovery phase ran


class Rec:
    """abort() with injected recovery, watch and end (no host, no :8000)."""

    def __init__(self, *, error: bool = False, watch: str = 'PASS') -> None:
        self.error, self.watch_status, self.recovered, self.ended = error, watch, 0, 0

    def recover(self, out: Path, log: Callable[[str], None]) -> Mapping[str, Json]:
        self.recovered += 1
        if self.error:
            raise RuntimeError('W8 check: greedy 5/6 IDENTICAL')
        return {'start': 'PASS', 'check': 'PASS', 'close': 'PASS'}

    def watch(self, out: Path, log: Callable[[str], None]) -> Mapping[str, Json]:
        return {'status': self.watch_status}

    def end(self) -> Mapping[str, Json]:
        self.ended += 1
        return chain.end_chain(run=lambda host, command: (0, 'PLACEHOLDER DOWN\n'))

    def run(self, out: Path, *, serving: bool = False) -> dict[str, Json]:
        return chain.abort(out, hold=lambda: False, recover=self.recover, watch=self.watch, end=self.end,
                           log=lambda m: None, serving=lambda: serving)


def test_cx2_abort_alerts_and_touches_nothing_while_production_serves(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    _chained_marker(tmp_path, monkeypatch)
    rec = Rec()
    result = rec.run(tmp_path / 'a', serving=True)
    assert str(result['status']).startswith('ALERT: chain marker present but production is serving')
    assert rec.recovered == rec.ended == 0 and chain.marker_present() and not chain.abort_attempt().exists()


# ---------------------------------------------------------------- CX-3 / FIX 2b: one automatic abort per stopped chain

@pytest.mark.parametrize('first', ['failed', 'watch-failed'])
def test_cx3_one_automatic_abort_per_stopped_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: str) -> None:
    _chained_marker(tmp_path, monkeypatch)
    rec = Rec(error=first == 'failed', watch='PASS' if first == 'failed' else 'FAILED: recurrence after the single recovery')
    assert not str(rec.run(tmp_path / 'a')['status']).startswith('RESTORED: chain ended')
    assert chain.abort_attempt().exists() and chain.marker_present()
    rec.error, rec.watch_status = False, 'PASS'
    second = rec.run(tmp_path / 'b')
    assert str(second['status']).startswith('ALERT: a previous chain abort') and rec.recovered == 1
    assert record_status(tmp_path / 'b') == 'NOT RUN: a previous chain abort did not restore'
    real = chain.abort
    with monkeypatch.context() as m:                                               # by hand: --again runs it once more
        m.setenv('WINDOW_COORDINATOR', chain.COORDINATOR)
        m.setattr(chain, 'abort', lambda out: real(out, hold=lambda: False, recover=rec.recover, watch=rec.watch,
                                                   end=rec.end, log=lambda msg: None, serving=lambda: False))
        assert chain.main(['abort', '--again', '--out', str(tmp_path / 'c')]) == 0 and rec.recovered == 2
    assert not chain.abort_attempt().exists() and not chain.marker_present() and rec.ended == 1
    _chained_marker(tmp_path, monkeypatch)                                         # a later chain stops again: one abort
    assert str(rec.run(tmp_path / 'd')['status']).startswith('RESTORED: chain ended') and rec.recovered == 3


def record_status(out: Path) -> object:
    return json.loads((out / 'summary.json').read_text())['restore']['status']


def test_clear_and_end_chain_remove_the_abort_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _chained_marker(tmp_path, monkeypatch)
    chain.abort_attempt().write_text('{}\n')
    chain.end_chain(run=lambda host, command: (0, 'PLACEHOLDER DOWN\n'))
    assert not chain.abort_attempt().exists() and not chain.marker_present()
    chain.abort_attempt().write_text('{}\n')
    chain.clear()
    assert not chain.abort_attempt().exists()
