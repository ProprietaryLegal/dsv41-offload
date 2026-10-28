"""WIN2 back-to-back (chain) mode (NDM 2026-10-27: "We do not need to keep restoring production -- go window after window").

Coordinator-only. A chained window still opens and closes its own HOLD and keeps every fence, quiet and absence proof;
at a SUCCESSFUL end it leaves production DOWN (no W8 restore, no post-close watch) and writes MARKER. The next chained
window's preflight accepts "MARKER + production down + no HOLD" instead of "production serving". Any stop or failure,
or the last window of a chain (WINDOW_CHAIN_LAST=1), runs the normal strict W8 restore + watch and then end_chain().

S92 FIX 1 (WIN2-R2): production is down for the whole gap between chained windows, so a stop OUTSIDE a window's own
HOLD must restore too. `end_action` / `restore_required` decide it from the marker as well as the HOLD; a stop with no
owned HOLD while the marker is present takes `abort`: when no HOLD exists, the strict W8 start|check|close (under the
reviewed scoped recovery's own HOLD) + post-close watch, end_chain() only on RESTORED + watch PASS; when a HOLD exists, alert
and touch nothing. chain_prev_ok.sh runs that abort on any refusal while the marker is present (exit 3).

Hold placeholders (WIN2-R2 rule from R-PC1/R-PP4; WIN2-R3 root fix of R-NBW's defect): a tmux `dsv41-hold` session
running "vllm serve [DSV41-OFFLOAD hold] owner=<chain owner>" stays up on both Sparks for the whole chain, so other users
(NVO drama3, the local-law queue) never see the Sparks as free while production is down. These CHAIN placeholders live on
the DEFAULT tmux socket and outlive every window scope: the runner starts them from its own unscoped process over plain
ssh (`placeholders_up`, refused inside a scoped_remote scope) before the first fence of its `-hold` scope, whose own
placeholder sits on that scope's private socket and dies with the fence. Every probe uses the absolute tmux binary on the
default socket (PLACEHOLDER_TMUX), so it sees them from anywhere, including inside a `-restore-close` scope whose tmux shim
would look at its private socket. The owner is in the process title on the Spark and in the marker; start, probe and stop
refuse a placeholder of any other owner (or of none, e.g. window.sh open's). A chained end closes only the HOLD file
(`chained_close(owner)`), the next chained preflight requires this chain's placeholders, and they come down only in
`end_chain()`: after the strict W8 restore verified and the post-close watch passed (a window's own restore, or `abort`).
window.sh close cannot remove them when it runs under a scope (its kill reaches only the scope's private socket).

The marker also carries the chain's production snapshot (path + sha256 of the rollback.json captured by the window that
started from serving production), so any runner can follow any other runner's chained window (R-NBW request).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
import uuid
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
PLACEHOLDER_SESSION = 'dsv41-hold'
PLACEHOLDER_TITLE = 'vllm serve [DSV41-OFFLOAD hold]'
# The absolute binary on the explicit default socket: never a scoped_remote tmux shim (which sends any `dsv41-hold` command
# to the scope's private socket, R-NBW scratch/RNBW/placeholder-proof.txt) and never the server of an inherited $TMUX.
PLACEHOLDER_TMUX = '/usr/bin/tmux -L default'
OWNER_PATTERN = '[A-Za-z0-9_-]{1,64}'
# One Spark's chain-placeholder step: `probe`, `up` (start unless this owner's is up) or `down` (stop this owner's). The last
# line is "PLACEHOLDER <state>": UP owner=<o> pid=<p> | ABSENT | DOWN | FOREIGN owner=<o|none> pid=<p> | BROKEN <why> |
# REFUSED <why> (exit 3). A foreign or broken placeholder is never touched. `up` and `down` refuse inside any scoped_remote
# window scope (dsv41w-*.scope): a default-socket server born there would die with the scope's fence. Each action's script
# carries only its own branch, so a probe contains no mutation. No `${...}` anywhere: systemd >= 258 expands it in the
# command line of a `systemd-run --scope` (scoped_remote's launch; Controller runs 259), unbraced `$x` and `$(...)` it leaves.
_PLACEHOLDER_STATE = r"""set -u
owner=@OWNER@
title='@TITLE@'
t() { @TMUX@ "$@"; }
state() {
  t has-session -t =@SESSION@ 2>/dev/null || { echo ABSENT; return; }
  local panes cmd who
  panes=$(t list-panes -s -t =@SESSION@ -F '#{pane_pid}' 2>/dev/null) || { echo 'BROKEN pane query failed'; return; }
  case "$panes" in ''|*[!0-9]*) echo "BROKEN panes: $(echo $panes)"; return ;; esac
  cmd=$(tr '\0' '\n' < "/proc/$panes/cmdline" 2>/dev/null | head -n 1)
  case "$cmd" in
    "$title owner=$owner") echo "UP owner=$owner pid=$panes" ;;
    "$title owner="*) who=$(printf '%s' "$cmd" | cut -c@OWNER_FROM@-); echo "FOREIGN owner=$who pid=$panes" ;;
    "$title") echo "FOREIGN owner=none pid=$panes" ;;
    '') echo "BROKEN pane runs: nothing" ;;
    *) echo "BROKEN pane runs: $cmd" ;;
  esac
}
s=$(state)
"""
_PLACEHOLDER_UNSCOPED = r"""if grep -Eq '/dsv41w-[^/]*\.scope(/|$)' /proc/self/cgroup; then
  echo "PLACEHOLDER REFUSED: @ACTION@ inside a scoped_remote window scope; start and stop only from an unscoped process"
  exit 3
fi
"""
_PLACEHOLDER_ACTIONS = {
    'probe': '',
    'up': _PLACEHOLDER_UNSCOPED + r"""if [ "$s" = ABSENT ]; then
  t new-session -d -s @SESSION@ "exec -a '$title owner=$owner' sleep 86400" >/dev/null 2>&1
  # The pane runs tmux's fork, then the shell, until the exec lands.
  for _ in $(seq 50); do s=$(state); case "$s" in UP*|FOREIGN*|ABSENT) break ;; esac; sleep 0.1; done
fi
""",
    'down': _PLACEHOLDER_UNSCOPED + r"""case "$s" in
  UP*)
    pid=$(printf '%s' "$s" | sed 's/.*pid=//')
    t kill-session -t =@SESSION@ 2>/dev/null
    for _ in $(seq 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
    if kill -0 "$pid" 2>/dev/null; then s="BROKEN pid $pid survived kill-session"; else s=DOWN; fi ;;
esac
""",
}
CHAINED_CLOSE_OK = 'CHAINED CLOSE RECORDED'
PLACEHOLDERS_KEPT = ('chain placeholders (default tmux socket, owner in the process title) kept up between chained windows; '
                     'only end_chain() takes them down, after the strict W8 restore and a post-close watch PASS')
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

    'chained' - chained, non-last, no stop: owner = chain_owner(); placeholders_up(owner) BEFORE the first fence of the
                window's scopes; stop the experiment, prove the Sparks quiet, close the own HOLD with
                chained_close(owner) in its `-restore-close` scope, finish_chained(..., placeholder_owner=owner);
                production stays down.
    'restore' - the window owns a HOLD: its own strict W8 restore + post-close watch, then end_chain() on watch PASS
                (end_chain(owner) when this window started chain placeholders before any marker recorded them).
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


def finish_chained(window_out: Path, quiet_proof: Mapping[str, Json], *, placeholder_owner: str,
                   production_snapshot: Path | None = None) -> None:
    """Write MARKER (temp + fsync + rename) at a clean chained end; the chain placeholders of `placeholder_owner` stay up.

    The owner is the one `placeholders_up` proved on both Sparks (`chain_owner()`); a chain keeps one owner, so a marker
    that already records another owner refuses.

    The marker's production snapshot is, in order: the explicit path; the current marker's snapshot when this window ran
    chained (marker present at its start, so it could not capture production); else this window's own rollback.json (it
    started from serving production). With none of them the marker records null and `production_snapshot()` refuses
    later; an inherited snapshot whose bytes changed refuses now."""
    if not (enabled() and not last()):
        raise ValueError('finish_chained outside a non-last chained window')
    if not quiet_proof:
        raise ValueError('chained finish requires the Spark quiet proof')
    check_owner(placeholder_owner)
    recorded = marker_owner() if marker_present() else None
    if recorded is not None and recorded != placeholder_owner:
        raise ValueError(f'chain placeholder owner {placeholder_owner} differs from the marker\'s {recorded}')
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
                             'placeholder_owner': placeholder_owner, 'placeholders': PLACEHOLDERS_KEPT}
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


def clear(*, placeholders_released: bool = False) -> None:
    """Remove the marker. Runners call end_chain() instead, which takes the chain placeholders down first; a marker that
    records a placeholder owner refuses here unless the caller states the placeholders were released (end_chain does,
    as does the coordinator's `clear --placeholders-released` after removing them by hand)."""
    if not placeholders_released and marker_present() and marker_owner() is not None:
        raise ValueError('the chain marker records placeholder owner '
                         f'{marker_owner()}: end the chain with chain.end_chain() (placeholders down, then the marker)')
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


def check_owner(owner: str) -> str:
    if not isinstance(owner, str) or re.fullmatch(OWNER_PATTERN, owner) is None:
        raise ValueError(f'invalid chain placeholder owner {owner!r} ({OWNER_PATTERN} required)')
    return owner


def new_owner() -> str:
    return 'chain-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]


def marker_owner() -> str | None:
    """The placeholder owner the marker records (None without a marker, or for a marker written before WIN2-R3)."""
    marker = read_marker()
    owner = marker.get('placeholder_owner') if marker is not None else None
    return check_owner(owner) if isinstance(owner, str) else None


def chain_owner() -> str:
    """The owner a chained end starts or keeps: the marker's while this window ran chained, else a new chain's.
    Call it once per window and keep the value (end_chain needs it if the window fails before finish_chained)."""
    if marker_present():
        owner = marker_owner()
        if owner is None:
            raise ValueError('the chain marker records no placeholder owner (written before the WIN2-R3 placeholder fix); '
                             'restore W8 and start a new chain')
        return owner
    return new_owner()


def placeholder_script(action: str, owner: str) -> str:
    """One Spark's step as a self-contained `bash -c` command, for plain ssh or a scope's ssh shim alike."""
    if action not in _PLACEHOLDER_ACTIONS:
        raise ValueError(f'unknown placeholder action {action!r}')
    script = ((_PLACEHOLDER_STATE + _PLACEHOLDER_ACTIONS[action] + 'echo "PLACEHOLDER $s"\n')
              .replace('@OWNER@', check_owner(owner)).replace('@ACTION@', action)
              .replace('@OWNER_FROM@', str(len(PLACEHOLDER_TITLE + ' owner=') + 1))
              .replace('@TITLE@', PLACEHOLDER_TITLE).replace('@TMUX@', PLACEHOLDER_TMUX)
              .replace('@SESSION@', PLACEHOLDER_SESSION))
    return shlex.join(['bash', '-c', script])


Run = Callable[[str, str], tuple[int, str]]


def plain_ssh(host: str, command: str) -> tuple[int, str]:
    """An ordinary (unscoped) ssh from the calling process, as the runners and the coordinator gate use."""
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, command], capture_output=True, text=True,
                                timeout=60, check=False)
    except subprocess.TimeoutExpired:
        return 124, 'ssh timed out after 60 s'
    return result.returncode, result.stdout + result.stderr


def placeholder_states(action: str, owner: str, *, run: Run = plain_ssh,
                       hosts: tuple[str, ...] = PLACEHOLDER_HOSTS) -> dict[str, str]:
    """{host: state} for one action on every Spark; a missing state line is 'ERROR rc=<rc>: <output tail>'."""
    command = placeholder_script(action, owner)
    states: dict[str, str] = {}
    for host in hosts:
        rc, text = run(host, command)
        lines = [line for line in text.splitlines() if line.startswith('PLACEHOLDER ')]
        states[host] = lines[-1].removeprefix('PLACEHOLDER ') if lines else f'ERROR rc={rc}: {text.strip()[-300:]}'
    return states


def placeholders_up(owner: str, *, run: Run = plain_ssh) -> dict[str, str]:
    """Start (or keep) this chain's placeholders on both Sparks and prove them; raises unless both are UP.

    Call from the runner's own unscoped process (plain ssh) BEFORE the first fence of the window's `-hold` scope, so the
    Sparks never look free. A placeholder of another owner (or of none) refuses and is left alone."""
    states = placeholder_states('up', owner, run=run)
    if not all(state.startswith('UP ') for state in states.values()):
        raise ValueError(f'chain placeholders of {owner} not up on both Sparks: {states}')
    return states


def placeholders_present(owner: str | None = None, *, run: Run = plain_ssh) -> bool:
    """Both Sparks run THIS chain's placeholder (default socket; owner from the marker unless given). Absent, foreign,
    broken or unreachable counts as missing; so does a marker without an owner."""
    if owner is None:
        owner = marker_owner()
        if owner is None:
            return False
    return all(state.startswith('UP ') for state in placeholder_states('probe', owner, run=run).values())


def placeholders_down(owner: str, *, run: Run = plain_ssh) -> dict[str, str]:
    """Stop this chain's placeholders (from an unscoped process). DOWN or ABSENT per host counts as removed; a foreign or
    broken placeholder is left alone and reported."""
    return placeholder_states('down', owner, run=run)


def released(states: Mapping[str, str]) -> bool:
    return bool(states) and all(state in ('DOWN', 'ABSENT') for state in states.values())


def chained_close(owner: str) -> str:
    """Shared "close our own HOLD without production" (R-NBW request): window.sh close refuses without :8000 and would kill
    placeholders. Run it ONLY inside the window's own `-restore-close` scope on sparkhead
    (owner.command('launch', f'{tag}-restore-close', ['bash', '-c', chain.chained_close(owner)], ...)): it proves this
    chain's placeholders on both Sparks, then removes ONLY the HOLD file, which that scope's rm shim RECORDS (refusing a
    HOLD of another window); the runner's commit-close removes it after every phase is fenced and proved absent.

    Through the scope's ssh shim each Spark's check runs in a scope of the same tag whose `tmux` is the shim (private
    socket); the probe calls PLACEHOLDER_TMUX, so it sees the default-socket chain placeholders."""
    probe = shlex.quote(placeholder_script('probe', owner))
    return ('set -euo pipefail; cd ~/dsv41-deploy; set -a; . ./dsv41.env; set +a; '
            'for node in "$HEAD" "$WORKER"; do '
            f'state=$(ssh -o BatchMode=yes "$node" {probe} | tail -n 1) '
            '|| { echo "hold placeholder probe failed on $node"; exit 1; }; '
            'case "$state" in "PLACEHOLDER UP "*) ;; '
            f'*) echo "hold placeholder of {owner} missing on $node: $state"; exit 1 ;; esac; done; '
            f'rm -f "$HOME/dsv41-deploy/HOLD"; echo {CHAINED_CLOSE_OK}')


def end_chain(owner: str | None = None, *, run: Run = plain_ssh) -> dict[str, Json]:
    """The chain's last step, ONLY after the strict W8 restore verified AND the post-close watch PASSED (a window's own
    restore, or abort): take this chain's placeholders down on both Sparks, then clear the marker.

    `owner` defaults to the marker's; a window that started placeholders before any marker recorded them (its chained end
    failed before finish_chained) passes its own. A foreign, broken or unreachable placeholder is left alone and the result
    says ALERT; the marker is cleared either way, because production is verified serving."""
    result: dict[str, Json] = {}
    if owner is None and marker_present():
        try:
            owner = marker_owner()
        except (OSError, ValueError) as exc:
            result['marker_read'] = f'unreadable: {type(exc).__name__}: {exc}'
    result['placeholder_owner'] = owner
    if owner is None:
        result['placeholders'] = 'NONE: no placeholder owner recorded'
        result['status'] = ('ENDED WITH ALERT: marker unreadable, its placeholders (if any) not removed'
                            if 'marker_read' in result else 'ENDED: no chain placeholders to remove')
    else:
        try:
            states = placeholders_down(check_owner(owner), run=run)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            states = {'error': f'{type(exc).__name__}: {exc}'}
        result['placeholders'] = cast(Json, dict(states))
        result['status'] = ('ENDED: chain placeholders down on both Sparks' if released(states) else
                            'ENDED WITH ALERT: chain placeholders not removed (left alone); coordinator action required')
    clear(placeholders_released=True)
    result['marker'] = 'cleared'
    return result


def gate(serving: Callable[[], bool] = production_serving, hold: Callable[[], bool] = hold_exists,
         placeholders: Callable[[], bool] = placeholders_present) -> str:
    """Between-window gate: 'chained' or 'production' (see preflight); raises otherwise. The placeholders (this chain's
    owner, default socket) are probed only while the chain marker is present."""
    present: bool | None = None
    if marker_present():
        if marker_owner() is None:
            raise ValueError('STOP: the chain marker records no placeholder owner (written before the WIN2-R3 placeholder '
                             'fix), so its placeholders cannot be proved; restore W8')
        present = placeholders()
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
          watch: Watch = _watch_production, end: Callable[[], Mapping[str, Json]] = end_chain,
          log: Callable[[str], None] = print) -> dict[str, Json]:
    """Coordinator chain abort (S92 FIX 1): bring production back after a chain stopped outside a window's own HOLD.

    No marker -> nothing to do. A HOLD exists (or its probe fails) -> ALERT and touch nothing: its owner, or the
    coordinator by hand, restores. Otherwise the strict W8 restore (start, check, close; `_recover_production`), then the
    15-minute post-close watch; end_chain() (chain placeholders down, then the marker) ONLY when every W8 step verified
    AND the watch PASSED. Until then the chain placeholders stay up: production is not proved. Writes out/summary.json in
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
    ended = dict(end())
    result['end_chain'] = cast(Json, ended)
    result['chain'] = 'ABORTED: strict restore verified, watch PASS, marker cleared'
    # Exit 0 for any 'RESTORED: chain ended' (production is verified); an ALERT about placeholders rides in the text.
    return finish(f'RESTORED: chain ended by abort; watch PASS; marker cleared; {ended.get("status")}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cmd', choices=('gate', 'status', 'clear', 'present', 'abort', 'placeholders', 'end'))
    parser.add_argument('--out', type=Path, help='abort: fresh output directory (summary.json, W8 logs, watch)')
    parser.add_argument('--owner', help='placeholders/end: chain placeholder owner (default: the marker\'s)')
    parser.add_argument('--placeholders-released', action='store_true',
                        help='clear: the coordinator removed this chain\'s placeholders by hand')
    args = parser.parse_args(argv)
    if args.cmd == 'status':
        print(json.dumps({'enabled': enabled(), 'last': last(), 'marker_path': str(MARKER),
                          'production_url': PRODUCTION_MODELS, 'marker': read_marker()}))
        return 0
    if args.cmd == 'present':
        print('PRESENT' if marker_present() else 'ABSENT')
        return 0
    if args.cmd == 'clear':
        try:
            clear(placeholders_released=args.placeholders_released)
        except ValueError as exc:
            print(f'STOP: {exc}', file=sys.stderr)
            return 2
        print('chain marker cleared')
        return 0
    if args.cmd == 'placeholders':
        owner = args.owner if args.owner is not None else marker_owner()
        if owner is None:
            print('STOP: no placeholder owner (marker has none; pass --owner)', file=sys.stderr)
            return 2
        states = placeholder_states('probe', owner)
        print(json.dumps({'owner': owner, 'states': states}))
        return 0 if all(state.startswith('UP ') for state in states.values()) else 1
    if args.cmd == 'end':
        if os.environ.get('WINDOW_COORDINATOR') != COORDINATOR:
            print(f'STOP: end is coordinator-only (WINDOW_COORDINATOR={COORDINATOR}); only after a verified restore + '
                  'watch PASS', file=sys.stderr)
            return 2
        ended = end_chain(args.owner)
        print(json.dumps(ended))
        return 0 if str(ended['status']).startswith('ENDED:') else 4
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
