"""WIN2 back-to-back (chain) mode (NDM 2026-10-27: "We do not need to keep restoring production -- go window after window").

Coordinator-only. A chained window still opens and closes its own HOLD and keeps every fence, quiet and absence proof;
at a SUCCESSFUL end it leaves production DOWN (no W8 restore, no post-close watch) and writes MARKER. The next chained
window's preflight accepts "MARKER + production down + no HOLD" instead of "production serving". Any stop or failure,
or the last window of a chain (WINDOW_CHAIN_LAST=1), runs the normal strict W8 restore + watch and then clear().

S92 FIX 1 (WIN2-R2): production is down for the whole gap between chained windows, so a stop OUTSIDE a window's own
HOLD must restore too. `end_action` / `restore_required` decide it from the marker as well as the HOLD; a stop with no
owned HOLD while the marker is present takes `abort`: when no HOLD exists, the strict W8 start|check|close (under the
reviewed scoped recovery's own HOLD) + post-close watch, clear() only on RESTORED + watch PASS; when a HOLD exists, alert
and touch nothing. chain_prev_ok.sh runs that abort on any refusal while the marker is present (exit 3).

Hold placeholders (WIN2-R2, coordinator rule from R-PC1/R-PP4): the tmux `dsv41-hold` session and its
"vllm serve [DSV41-OFFLOAD hold]" sleep process stay up on both Sparks for the whole chain, so other users (NVO drama3,
the local-law queue) never see the Sparks as free while production is down. A chained end closes only the HOLD file
(CHAINED_CLOSE), the next chained preflight requires the placeholders present, and only the final strict W8 restore
(window.sh close) or `abort` (whose W8 close runs window.sh close) takes them down.

The marker also carries the chain's production snapshot (path + sha256 of the rollback.json captured by the window that
started from serving production), so any runner can follow any other runner's chained window (R-NBW request).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, record

# DSV41_CHAIN_MARKER / DSV41_CHAIN_PRODUCTION_URL are TEST-ONLY overrides (S92 note: chain_prev_ok.sh could not be
# rehearsed end to end without touching the live marker). `status` prints both; production runs never set them.
MARKER = Path(os.environ.get('DSV41_CHAIN_MARKER') or '/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/CHAINED')
SCHEMA = 'dsv41-chain-v1'
PRODUCTION_MODELS = os.environ.get('DSV41_CHAIN_PRODUCTION_URL') or 'http://HEAD_IP:8000/v1/models'
K1_CAP = 327680
COORDINATOR = 'Deepseek/coordinator'
PLACEHOLDER_HOSTS = ('sparkhead', 'sparkworker-wired')
# Both the tmux session and the process: `exec -a "vllm serve [DSV41-OFFLOAD hold]" sleep 86400` (hold_admission.open_script,
# window.sh holdproc). The bracketed pattern does not match this probe's own shell, whose command line has backslashes.
PLACEHOLDER_PROBE = "tmux has-session -t dsv41-hold 2>/dev/null && pgrep -f 'vllm serve \\[DSV41-OFFLOAD hold\\]' >/dev/null"
# Shared "close our own HOLD without production" (R-NBW request; window.sh close refuses without :8000 and would also kill
# the placeholders). It removes ONLY the HOLD file and first proves both placeholders are still up. Run it ONLY inside the
# window's own `-restore-close` scope (owner.command('launch', tag, ['bash', '-c', CHAINED_CLOSE], ...)), whose rm shim
# RECORDS the unlink; the runner's commit_close removes HOLD after every phase is fenced and proved absent.
CHAINED_CLOSE = ('set -euo pipefail; cd ~/dsv41-deploy; set -a; . ./dsv41.env; set +a; '
                 'for node in "$HEAD" "$WORKER"; do ssh -o BatchMode=yes "$node" "tmux has-session -t dsv41-hold" '
                 '|| { echo "hold placeholder missing on $node"; exit 1; }; done; '
                 'rm -f "$HOME/dsv41-deploy/HOLD"; echo CHAINED CLOSE RECORDED')
CHAINED_CLOSE_OK = 'CHAINED CLOSE RECORDED'
PLACEHOLDERS_KEPT = ('kept up between chained windows; only the final strict W8 restore (window.sh close) or chain.abort '
                     'takes them down')
ABORT_RESTORED = 'RESTORED: K1 START VERIFIED, greedy 6/6 IDENTICAL, PRODUCTION HOLD OK, window closed (chain abort)'


def enabled() -> bool:
    return os.environ.get('WINDOW_CHAIN') == '1'


def last() -> bool:
    return os.environ.get('WINDOW_CHAIN_LAST') == '1'


def skip_restore(stopped: bool) -> bool:
    """True only for a chained, non-last window that ended without any stop."""
    return enabled() and not last() and not stopped


def marker_present() -> bool:
    """The marker exists in any form (a torn or unreadable marker still means a chain left production down)."""
    return MARKER.exists() or MARKER.is_symlink()


def end_action(stopped: bool, hold_written: bool) -> str:
    """How a window must end (S92 FIX 1). Chained runners use this, never `hold_written` alone:

    'chained' - chained, non-last, no stop: stop the experiment, prove the Sparks quiet, close the own HOLD with
                CHAINED_CLOSE, finish_chained; production stays down.
    'restore' - the window owns a HOLD: its own strict W8 restore + post-close watch, then clear().
    'abort'   - no owned HOLD but the chain marker is present (production is down from the previous window, e.g. a
                pre-HOLD failure of a chained window): abort(out), which restores with the strict W8 + watch when no
                HOLD exists and only alerts when one does. Record its summary as the window's `restore`.
    'none'    - no HOLD, no marker: production was never touched by this window or the chain.
    """
    if skip_restore(stopped):
        return 'chained'
    if hold_written:
        return 'restore'
    return 'abort' if marker_present() else 'none'


def restore_required(stopped: bool, hold_written: bool) -> bool:
    """True whenever production must be brought back: the window's own HOLD, or the chain marker without a clean
    chained end (S92 FIX 1). The window's own restore applies when it holds a HOLD; otherwise `abort`."""
    return end_action(stopped, hold_written) in ('restore', 'abort')


def allow_unreachable(mode: str) -> bool:
    """HOLD admission rule for a runner start: in mode 'chained' production is down by design, so the admission HOLD
    opens with hold_admission.open_script(..., allow_unreachable=True); in mode 'production' it must see :8000 serving."""
    if mode not in ('chained', 'production'):
        raise ValueError(f'unknown chain preflight mode {mode!r}')
    return mode == 'chained'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_commit(path: Path) -> str:
    data = record(cast(Json, json.loads(path.read_text())))
    commit = data.get('production_commit') or data.get('commit')
    if not (isinstance(commit, str) and re.fullmatch('[0-9a-f]{40}', commit)):
        raise ValueError(f'{path}: production snapshot lacks a 40-hex production commit')
    return commit


def snapshot_ref(path: Path) -> dict[str, Json]:
    """{'path', 'sha256', 'production_commit'} of a production capture (a window's rollback.json)."""
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f'production snapshot {path} is not an existing absolute file')
    return {'path': str(path), 'sha256': _sha256(path), 'production_commit': _snapshot_commit(path)}


def finish_chained(window_out: Path, quiet_proof: Mapping[str, Json], *,
                   production_snapshot: Path | None = None) -> None:
    """Write MARKER (temp + fsync + rename) at a clean chained end; the hold placeholders stay up.

    The marker's production snapshot is, in order: the explicit path; the current marker's snapshot when this window ran
    chained (marker present at its start, so it could not capture production); else this window's own rollback.json (it
    started from serving production). With none of them the marker records null and `production_snapshot()` refuses
    later; an inherited snapshot whose bytes changed refuses now."""
    if not (enabled() and not last()):
        raise ValueError('finish_chained outside a non-last chained window')
    if not quiet_proof:
        raise ValueError('chained finish requires the Spark quiet proof')
    snapshot: Json = None
    if production_snapshot is not None:
        snapshot = snapshot_ref(production_snapshot)
    elif marker_present():
        current = read_marker()
        inherited = current.get('production_snapshot') if current is not None else None
        if isinstance(inherited, dict):
            snapshot = snapshot_ref(Path(str(inherited.get('path'))))
            if snapshot['sha256'] != inherited.get('sha256'):
                raise ValueError('chain production snapshot changed since it was recorded')
    elif (window_out / 'rollback.json').is_file():
        snapshot = snapshot_ref(window_out / 'rollback.json')
    data: dict[str, Json] = {'schema': SCHEMA, 'window': str(window_out), 'time': time.time(),
                             'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                             'quiet_proof': cast(Json, dict(quiet_proof)), 'production_snapshot': snapshot,
                             'placeholders': PLACEHOLDERS_KEPT}
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    temp = MARKER.with_name(MARKER.name + '.new')
    with temp.open('w') as handle:
        handle.write(json.dumps(data, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, MARKER)


def read_marker() -> dict[str, Json] | None:
    if not MARKER.exists():
        return None
    data = record(cast(Json, json.loads(MARKER.read_text())))
    if data.get('schema') != SCHEMA:
        raise ValueError(f'{MARKER}: unknown chain marker schema')
    return data


def production_snapshot() -> tuple[Path, dict[str, Json]]:
    """The chain's production capture for a chained window (which cannot capture production while it is down):
    (path, parsed rollback.json), after checking the sha256 recorded in the marker. Refuses without a marker."""
    marker = read_marker()
    if marker is None:
        raise ValueError('no chain marker: capture production from the serving server instead')
    ref = marker.get('production_snapshot')
    if not isinstance(ref, dict):
        raise ValueError('chain marker carries no production snapshot (none captured, or written before WIN2-R2)')
    path = Path(str(ref.get('path')))
    if not path.is_absolute() or not path.is_file() or _sha256(path) != ref.get('sha256'):
        raise ValueError(f'chain production snapshot {path} missing or changed since the marker was written')
    return path, record(cast(Json, json.loads(path.read_text())))


def preflight(production_serving: bool, hold_exists: bool, *, placeholders_present: bool | None = None) -> str:
    """'chained' (marker, production down, no HOLD, both hold placeholders up) or 'production' (no marker, production
    serving); else STOP. A chained preflight must pass the placeholder probe result (`placeholders_present()`)."""
    if hold_exists:
        raise ValueError('STOP: a HOLD exists; no window may start')
    marker = read_marker()
    if marker is not None:
        if not enabled():
            raise ValueError('STOP: production is down in a chain (marker present); run with WINDOW_CHAIN=1 or restore W8')
        if production_serving:
            raise ValueError('STOP: chain marker present but production is serving; inspect before continuing')
        if placeholders_present is None:
            raise ValueError('STOP: a chained preflight must prove the hold placeholders on both Sparks '
                             '(placeholders_present=chain.placeholders_present())')
        if not placeholders_present:
            raise ValueError('STOP: hold placeholders missing between chained windows; the Sparks looked free while '
                             'production was down')
        return 'chained'
    if not production_serving:
        raise ValueError('STOP: production not serving and no chain marker')
    return 'production'


def clear() -> None:
    """After a strict W8 restore verified with post-close watch PASS (chain end or any failure)."""
    MARKER.unlink(missing_ok=True)
    MARKER.with_name(MARKER.name + '.new').unlink(missing_ok=True)


def production_serving(url: str = PRODUCTION_MODELS, timeout: float = 15.0) -> bool:
    """Production answers /v1/models with the K1 cap (the same test the coordinator's prev-ok gate uses)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = record(cast(Json, json.loads(response.read(1024 * 1024))))
    except (OSError, ValueError):
        return False
    rows = data.get('data')
    return isinstance(rows, list) and any(isinstance(r, dict) and r.get('max_model_len') == K1_CAP for r in rows)


def hold_exists() -> bool:
    """HOLD file on sparkhead (read-only ssh test); an ssh failure counts as 'exists' so nothing starts blind."""
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'sparkhead', 'test ! -e ~/dsv41-deploy/HOLD'],
                            capture_output=True, timeout=30, check=False)
    return result.returncode != 0


def placeholders_present() -> bool:
    """Both Sparks run the hold placeholder (tmux dsv41-hold + its sleep process); any ssh failure counts as absent."""
    for host in PLACEHOLDER_HOSTS:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, PLACEHOLDER_PROBE],
                                capture_output=True, timeout=30, check=False)
        if result.returncode != 0:
            return False
    return True


def gate(serving: Callable[[], bool] = production_serving, hold: Callable[[], bool] = hold_exists,
         placeholders: Callable[[], bool] = placeholders_present) -> str:
    """Between-window gate: 'chained' or 'production' (see preflight); raises otherwise. The placeholders are probed
    only while the chain marker is present."""
    present = placeholders() if marker_present() else None
    return preflight(production_serving=serving(), hold_exists=hold(), placeholders_present=present)


Recover = Callable[[Path, Callable[[str], None]], Mapping[str, Json]]
Watch = Callable[[Path, Callable[[str], None]], Mapping[str, Json]]


def _tag(out: Path, suffix: str) -> str:
    """Scoped-recovery id, unique per window (parent and own directory names), at most 64 characters."""
    base = re.sub('[^A-Za-z0-9_-]', '-', f'{out.parent.name}-{out.name}')
    return (base[-(63 - len(suffix)):] + '-' + suffix).lstrip('-')


def _recover_production(out: Path, log: Callable[[str], None]) -> Mapping[str, Json]:
    """The reviewed post_close_watch.ScopedRecovery: its own HOLD (admission allow-unreachable), production stop, both
    Sparks proved quiet, W8 start|check|close each verified by restore_verified (six distinct IDENTICAL), every phase
    fenced, HOLD released by commit-close. Any failure raises with that HOLD retained."""
    from offload.launch.windows import post_close_watch as watchdog
    recovery = watchdog.ScopedRecovery(_tag(out, 'abort'), out / 'abort-commands')
    log('CHAIN ABORT: strict W8 recovery under its own scoped HOLD (start, check, close)')
    recovery.recover(time.monotonic() + watchdog.RECOVERY_S)
    return {'start': 'PASS', 'check': 'PASS', 'close': 'PASS', 'hold': 'own scoped HOLD opened, released by commit-close',
            'phases': cast(Json, list(recovery.phases))}


def _watch_production(out: Path, log: Callable[[str], None]) -> Mapping[str, Json]:
    from offload.launch.windows import post_close_watch as watchdog
    result = watchdog.watch(watchdog.ProductionHealth(),
                            watchdog.ScopedRecovery(_tag(out, 'watch'), out / 'post-close-commands'), log=log)
    return cast(Mapping[str, Json], dict(result.__dict__))


def abort(out: Path, *, hold: Callable[[], bool] = hold_exists, recover: Recover = _recover_production,
          watch: Watch = _watch_production, log: Callable[[str], None] = print) -> dict[str, Json]:
    """Coordinator chain abort (S92 FIX 1): bring production back after a chain stopped outside a window's own HOLD.

    No marker -> nothing to do. A HOLD exists (or its probe fails) -> ALERT and touch nothing: its owner, or the
    coordinator by hand, restores. Otherwise the strict W8 restore (start, check, close; `_recover_production`), then the
    15-minute post-close watch; clear() ONLY when every W8 step verified AND the watch PASSED. Writes out/summary.json in
    the shape prev-ok.sh / chain_prev_ok.sh read (restore.status RESTORED..., post_close_watch.status PASS).
    """
    out.mkdir(parents=True, exist_ok=False)
    result: dict[str, Json] = {'schema': 'dsv41-chain-abort-v1', 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                               'marker_path': str(MARKER), 'chain': 'ABORT', 'restore': {'status': 'NOT RUN'}}

    def finish(status: str) -> dict[str, Json]:
        result['status'] = status
        (out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
        log(f'CHAIN ABORT: {status}')
        return result

    if not marker_present():
        result['restore'] = {'status': 'NOT NEEDED: no chain marker'}
        return finish('NOT NEEDED: no chain marker; production was not left down by a chain')
    try:
        result['marker'] = read_marker()
    except (OSError, ValueError) as exc:
        result['marker'] = f'unreadable: {type(exc).__name__}: {exc}'
    if hold():
        result['restore'] = {'status': 'NOT RUN: a HOLD exists'}
        return finish('ALERT: a HOLD exists (or its probe failed); production is down; chain abort touched nothing; '
                      'the HOLD owner (or the coordinator by hand) must restore')
    try:
        steps = dict(recover(out, log))
    except (Exception, KeyboardInterrupt) as exc:
        result['restore'] = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
        return finish('FAILED: strict W8 recovery unverified; production down; its HOLD (if opened) retained; marker kept')
    result['restore'] = {'status': ABORT_RESTORED, **steps}
    watched = dict(watch(out, log))
    result['post_close_watch'] = cast(Json, watched)
    if watched.get('status') != 'PASS':
        return finish('RESTORED BUT POST-CLOSE WATCH FAILED: marker kept; coordinator action required')
    clear()
    result['chain'] = 'ABORTED: strict restore verified, watch PASS, marker cleared'
    return finish('RESTORED: chain ended by abort; watch PASS; marker cleared')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cmd', choices=('gate', 'status', 'clear', 'present', 'abort'))
    parser.add_argument('--out', type=Path, help='abort: fresh output directory (summary.json, W8 logs, watch)')
    args = parser.parse_args(argv)
    if args.cmd == 'status':
        print(json.dumps({'enabled': enabled(), 'last': last(), 'marker_path': str(MARKER),
                          'production_url': PRODUCTION_MODELS, 'marker': read_marker()}))
        return 0
    if args.cmd == 'present':
        print('PRESENT' if marker_present() else 'ABSENT')
        return 0
    if args.cmd == 'clear':
        clear()
        print('chain marker cleared')
        return 0
    if args.cmd == 'abort':
        if os.environ.get('WINDOW_COORDINATOR') != COORDINATOR:
            print(f'STOP: chain abort is coordinator-only (WINDOW_COORDINATOR={COORDINATOR})', file=sys.stderr)
            return 2
        if args.out is None:
            print('STOP: abort needs --out', file=sys.stderr)
            return 2
        status = str(abort(args.out)['status'])
        if status.startswith(('RESTORED: chain ended', 'NOT NEEDED')):
            return 0
        return 4 if status.startswith('ALERT') else 3
    try:
        print(gate())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
