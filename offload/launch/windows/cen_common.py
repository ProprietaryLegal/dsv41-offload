"""CEN window lifecycle shared by the CEN1 and CEN3 runners (documents/plans/lanes/CEN-RUN.plan.md).

nbw_window.Live without NBW's native pair, as R-UPX's battery runner does it: owned HOLD admission with the window's own
marker (private placeholder kept up by the hold phase), scoped ownership and fences on every host, quiet proof, strict
six-IDENTICAL W8 restore, then the 15-minute post-close watch. SINGLE windows only (coordinator 2026-10-27: the chain
placeholder defect, scratch/RNBW/placeholder-proof.txt, is open): the preflight refuses WINDOW_CHAIN and a chain marker,
and the end follows chain.end_action(stopped=True, ...), which without WINDOW_CHAIN is 'restore' (own HOLD), 'abort'
(a marker appeared mid-window: chain.abort) or 'none'. The production commit and tag are arguments proved on sparkhead.
Each start's clock cap comes from its own start log (win2_history.start_clock_cap); a cap other than the registered one
marks the measurement INCONCLUSIVE and never touches the restore.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from offload.launch.windows import chain
from offload.launch.windows import hold_admission
from offload.launch.windows import nbw_window as n
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import post_close_watch as watchdog
from offload.launch.windows import up2_checks as ck
from offload.launch.windows import win2_history as wh
from offload.launch.windows.up2_low import X86HOST_PROBE
from offload.prefill_maxlen.evidence import Json, record

# Never validated or used: no CEN start has NBW's native role (probe_worker and start are overridden).
NO_NATIVE = n.Native('dsv41-offload/none/repo', '0' * 40, '1' * 64, '2' * 64)
CLOCK_MHZ = 2300                        # registered per-start cap (CEN briefs: "2300 MHz now")
PRE_DISK_KB = 10 * 1024**2
SHARED_PINS = ('offload/launch/start_exp.sh', 'offload/launch/serve_exp.sh', 'offload/launch/dsv41_tp2_offload.sh',
               'offload/kvpack/MANIFEST.json', 'offload/launch/windows/cen_common.py',
               'offload/launch/windows/nbw_window.py', 'offload/launch/windows/scoped_remote.py',
               'offload/launch/windows/post_close_watch.py', 'offload/launch/windows/spark_quiet.py',
               'offload/launch/windows/production_health.py', 'offload/launch/windows/hold_admission.py',
               'offload/launch/windows/chain.py')


@dataclass(frozen=True)
class Timing:
    """Deadlines from the HOLD's publication, seconds: last cell submission, measurement abort, restore completed."""
    submit: float
    abort: float
    restore: float

    def check(self) -> None:
        ck.require(0 < self.submit <= self.abort < self.restore <= n.RESTORE_S,
                   'CEN deadlines must satisfy submit <= abort < restore <= the NBW placeholder lease (HOLD+90)')


def strip_flag(argv: str, flag: str, value: str) -> str:
    """The experiment's /serve.sh argv minus exactly one reviewed `flag value` pair; anything else refuses."""
    tokens = shlex.split(argv.replace('\\\n', ' '))
    ck.require(tokens.count(flag) == 1, f'experiment argv lacks exactly one {flag}')
    i = tokens.index(flag)
    ck.require(i + 1 < len(tokens) and tokens[i + 1] == value, f'{flag} value differs from the reviewed one')
    del tokens[i:i + 2]
    return shlex.join(tokens)


def runtime_identity(prod: dict[str, Json], live: dict[str, Json], *, env: dict[str, str], eager: bool,
                     strip: tuple[str, str] | None = None) -> None:
    """Experiment rank == saved production rank except port, the declared env changes, --enforce-eager (eager) and one
    reviewed `flag value` pair (strip). Env equality is checked exactly first; PF1's comparator then checks image,
    engine, target files, Engram identity, mounted code bytes and the argv."""
    before, after = record(prod['env']), record(live['env'])
    expected: dict[str, Json] = {**before, **env}
    drift = sorted(k for k in expected.keys() | after.keys() if expected.get(k) != after.get(k))
    ck.require(not drift, 'CEN runtime env differs from production plus the declared changes: ' + ', '.join(drift))
    argv = str(live['argv']) if strip is None else strip_flag(str(live['argv']), *strip)
    # PF1's comparator encodes REQUIRED 0 -> 1; normalise its inputs only after the exact check above (as NBW does).
    a = {**prod, 'env': {**before, 'VLLM_ENGRAM_REMOTE_REQUIRED': '0'}}
    b = {**live, 'argv': argv, 'env': {**before, 'VLLM_ENGRAM_REMOTE_REQUIRED': '1'}}
    pf.runtime(a, b, pf.Arm('cen', 327680, 0, eager=int(eager)), 'cen')


class CenLive(n.Live):
    """Shared CEN backend; subclasses set MARKER, TIMING, pin_files and hold_purpose, and add their phases."""

    MARKER = 'CEN-WINDOW'
    TIMING = Timing(50 * 60, 55 * 60, 75 * 60)

    def __init__(self, out: Path, sup: pf.Supervisor, *, remote: str, expected: str, data_dir: Path, tag: str,
                 production_commit: str = n.PROD_COMMIT, production_tag: str = n.PROD_TAG,
                 call: pf.Call | None = None,
                 plain: Callable[[str, list[str], float], tuple[int, str]] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        super().__init__(out, sup, native=NO_NATIVE, remote=remote, expected=expected, data_dir=data_dir, tag=tag,
                         call=call, plain=plain, clock=clock)
        self.TIMING.check()
        ck.require(re.fullmatch('[0-9a-f]{40}', production_commit) is not None, 'full production commit required')
        ck.require(re.fullmatch('[A-Za-z0-9_.-]+', production_tag) is not None, 'production tag required')
        self.production_commit, self.production_tag = production_commit, production_tag
        self.mode = ''
        self.stopped = False            # set BEFORE anything that may stop production (a lost reply may have stopped it)
        self.pair_used = False          # set BEFORE prod_window.sh pair (CEN3's drafter-down segment)
        self.started_at: dict[str, float] = {}

    # -- names
    def hold_marker(self) -> str:
        return f'{self.MARKER} {self.tag}:'

    def hold_purpose(self) -> str:
        return f'{self.hold_marker()} CEN measurement window; no production change'

    def run_dirs(self) -> tuple[str, ...]:
        """Per-node directories this window creates (relative to HOME); each must be absent before the HOLD."""
        return ()

    # -- before the HOLD (read-only, production serving)
    def production_serving(self) -> bool:
        models = ck.rows(self.json('prod-models', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/v1/models')['data'])
        ck.require(len(models) == 1 and models[0].get('max_model_len') == 327680
                   and models[0].get('root') == '/srv/dsv41/legal-v2a', 'production model/cap differs')
        return True

    def preflight(self) -> None:
        ck.require(not chain.enabled() and not chain.last(),
                   'CEN windows run single, never chained: unset WINDOW_CHAIN and WINDOW_CHAIN_LAST')
        self.sup.begin()
        ck.require(pf.digest(pf.RESTORE) == pf.RESTORE_SHA256, 'W8 restore changed')
        from offload.prefill_trace.window_requests import corpus
        corpus(self.data_dir)
        ck.require((pf.CAMPAIGN / 'D2-GEN.PAUSE').exists() or subprocess.run(
            ['tmux', 'has-session', '-t', '=d2-gen'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0,
            'D2 generation active without pause')
        self.ssh('ledger', 'sparkhead', 'cd ~/dsv41-deploy && test ! -e HOLD && '
                 f'test "$(git rev-parse HEAD)" = {self.production_commit} && '
                 f'test "$(git rev-parse HEAD)" = "$(git rev-list -n1 {self.production_tag})" && '
                 'git diff --exit-code HEAD -- dsv41.env serve.sh start.sh '
                 'offload/launch/warmup.py offload/launch/warmup_serve.sh')
        self.ssh('checkout', 'sparkhead', f'cd ~/{shlex.quote(self.remote)} && test "$(git rev-parse HEAD)" = {self.expected} && '
                 'test -z "$(git status --porcelain --untracked-files=no)"')
        remote = self.ssh('pins', 'sparkhead', f'cd ~/{shlex.quote(self.remote)} && sha256sum ' + shlex.join(self.pin_files))
        ck.require(remote.strip() == '\n'.join(f'{pf.digest(pf.ROOT / p)}  {p}' for p in self.pin_files),
                   'staged source pins differ')
        # A chain marker means production is down from a chain: refuse before probing production (say why).
        ck.require(not chain.marker_present(), 'STOP: chain marker present (production down in a chain); CEN windows '
                   'run single: finish or abort the chain (W8) first')
        self.mode = chain.preflight(production_serving=self.production_serving(), hold_exists=False)
        ck.require(self.mode == 'production', 'CEN single window needs production serving')
        self.ssh('prod-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/health')
        pf.worker(self.json('worker', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        for host in pf.HOSTS:
            n.host_check(self.ssh(f'preflight-host-{host}', host, pf.HOST_PARITY))
            if self.run_dirs():
                text = self.ssh(f'run-dirs-{host}', host, ' && '.join(f'test ! -e ~/{d}' for d in self.run_dirs())
                                + ' && df -Pk ~ | tail -1')
                ck.require(int(text.split()[-3]) >= PRE_DISK_KB, f'{host} home has < 10 GiB free')
        self.extra_preflight()

    def extra_preflight(self) -> None:
        """Subclass hook, still read-only and before the HOLD."""

    # -- HOLD (nbw_window.Live.open_hold with this window's marker and deadlines)
    def open_hold(self) -> float:
        ck.require(not self.open_admitted, f'{self.MARKER} HOLD cannot be reopened or renewed')
        ck.require(self.mode == 'production', 'preflight decides the mode first')
        self.open_admitted = True
        self.open_deadline = self.clock() + pf.DRAIN_S + 60
        self.hold_scope_deadline = self.open_deadline + n.RESTORE_S
        text = ''
        try:
            self.owned('hold-open', 'sparkhead', self.tag + '-hold',
                       hold_admission.open_script(self.tag + '-hold', self.admission_token, self.hold_purpose(),
                                                  pf.DRAIN_S, allow_unreachable=False),
                       timeout=pf.DRAIN_S + 60)
        finally:
            text = self.ssh('hold-read', 'sparkhead', hold_admission.command('check', self.tag + '-hold', self.admission_token))
            self.hold_written = self.hold_marker() in text
            if self.hold_written:
                self.set_deadlines(pf.hold_opened(text, self.tag, self.clock()))
        ck.require(self.hold_written, f'{self.MARKER} exact owned HOLD marker required')
        opened = pf.hold_opened(text, self.tag, self.clock())
        ck.require(opened <= self.open_deadline and self.restore_by <= self.hold_scope_deadline,
                   'HOLD published after its opening ceiling; placeholder lease would not cover the window')
        self.sup.tunnel([*pf.SSH, '-N', '-o', 'ExitOnForwardFailure=yes', '-L', '127.0.0.1:18001:127.0.0.1:8001', 'sparkhead'])
        return opened

    def set_deadlines(self, opened: float) -> None:
        t = self.TIMING
        self.submit, self.sup.abort_at, self.restore_by = opened + t.submit, opened + t.abort, opened + t.restore
        self.sup.bound = f'{t.abort / 60:.0f}-minute {self.MARKER} abort bound'
        (self.out / 'deadlines.json').write_text(json.dumps({'opened': opened, 'submit': self.submit,
                                                             'abort': self.sup.abort_at, 'restore_by': self.restore_by}) + '\n')

    def snapshot(self) -> None:
        super().snapshot()       # NBW's: production capture, idle wait, rollback.json
        path = self.out / 'rollback.json'
        data = ck.read(path)
        data.update({'production_commit': self.production_commit, 'tag': self.production_tag})
        path.write_text(json.dumps(data, indent=1) + '\n')

    # -- engine starts (battery_window.Live.start without its roles)
    def probe_worker(self, label: str, arm: n.Arm) -> dict[str, Json]:
        return pf.worker(self.json(label, 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())

    def start_engine(self, name: str, env: str, config: str, *, budget_s: float) -> list[dict[str, Json]]:
        """start_exp.sh under the HOLD admission in an owned scope (its cmd log NNNN-<name>-start.log keeps start_exp.sh's
        "<node> clock cap N MHz" lines), then health, model and both ranks' identities. The caller checks identity."""
        ck.require(self.clock() + budget_s <= self.submit, f'{name}: does not fit before the submission bound')
        self.sup.unready()
        for host in pf.HOSTS:
            n.host_check(self.ssh(f'{name}-host-{host}', host, pf.HOST_PARITY))
        self.sup.arm_minima = {}
        self.sup.arm_memory[name] = self.sup.arm_minima
        start = f'cd ~/{self.remote} && {env} timeout {pf.START_S} bash offload/launch/start_exp.sh {config}'
        start = hold_admission.command('stop', self.tag + '-hold', self.admission_token, ['bash', '-c', start])
        self.stopped = True
        self.started_at[name] = self.clock()
        self.owned(f'{name}-start', 'sparkhead', f'{self.tag}-{name}-engine', start, timeout=pf.START_S + 60)
        self.ssh(f'{name}-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/health')
        models = ck.rows(self.json(f'{name}-models', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/v1/models')['data'])
        ck.require(len(models) == 1 and models[0].get('max_model_len') == 327680
                   and models[0].get('root') == '/srv/dsv41/legal-v2a', 'experiment model/cap differs')
        return [self.capture(host, name) for host in pf.HOSTS]

    def accept_start(self, name: str, current: list[dict[str, Json]]) -> None:
        """After the caller's identity check: fresh containers, in-container pins, clean logs, proxy pid, worker."""
        for rank, host in enumerate(pf.HOSTS):
            ck.require(str(current[rank]['id']) not in self.seen, 'experiment reused a container')
            self.verify(host, name, 0)
            logs = pf.logs(self.json(f'{name}-logs-{host}', host, 'python3 -c ' + shlex.quote(pf.LOG_FILTER)
                                     + ' ' + shlex.quote(ck.STOP_LOG.pattern)), rank, 327680)
            if rank == 0:
                self.proxy_pid = int(cast(int, cast(list[Json], logs['proxy_pids'])[0]))
        self.ids = [str(row['id']) for row in current]
        self.seen.update(self.ids)
        self.probe_worker(f'{name}-worker', n.ARMS[0])
        (self.out / f'{name}-identity.json').write_text(json.dumps({'ranks': cast(Json, current)}, indent=1) + '\n')
        self.sup.mark_ready()

    def clock_cap(self, start: str) -> dict[str, Json]:
        """The start's own cap (start log); INCONCLUSIVE unless it is the registered cap on both Sparks."""
        try:
            cap, source = wh.start_clock_cap(self.out, start)
        except ValueError as exc:
            return {'status': f'INCONCLUSIVE: {exc}', 'registered_mhz': CLOCK_MHZ}
        status = 'OK' if cap == CLOCK_MHZ else f'INCONCLUSIVE: cap {cap} MHz, registered {CLOCK_MHZ}'
        return {'status': status, 'mhz': cap, 'source': source, 'registered_mhz': CLOCK_MHZ}

    # -- end
    def settle(self) -> bool:
        """Before chain.end_action: fence every owned phase, then reread this window's HOLD ownership."""
        for tag in list(self.owner_tags):
            self.fence_phase(tag)
        if not self.hold_written:
            rc, text = self.plain('hold-reread', [*pf.SSH, 'sparkhead',
                                                  hold_admission.command('check', self.tag + '-hold', self.admission_token)], 60)
            ck.require(rc == 0, f'{self.MARKER} HOLD ownership unreadable; HOLD stays open')
            self.hold_written = self.hold_marker() in text
            if self.hold_written:
                self.restore_by = pf.hold_opened(text, self.tag, self.clock()) + self.TIMING.restore
        return self.hold_written

    def pair_close(self, *, guarded: bool) -> None:
        """NBW's prod_window.sh close only when this window stopped the production worker (pair_used); otherwise a
        no-op, and W8's own check still requires PRODUCTION HOLD OK."""
        if not self.pair_used:
            self.pair_open = False
            return
        super().pair_close(guarded=guarded)

    def restore(self, *, production_stopped: bool) -> dict[str, Json]:
        self.before_restore()
        result = super().restore(production_stopped=production_stopped)
        result['pair_close'] = ('PRODUCTION HOLD OK (drafter-down segment closed)' if self.pair_used else
                                'not opened (no drafter-down segment); W8 check proves PRODUCTION HOLD OK')
        result.update(self.after_restore())
        return result

    def before_restore(self) -> None:
        """Subclass hook: remove what dockerd keeps alive outside the owned scopes (bench containers)."""

    def after_restore(self) -> dict[str, Json]:
        """Subclass hook: per-node directory cleanup after the HOLD is closed; returns what it did."""
        return {}


class Backend(Protocol):
    @property
    def mode(self) -> str: ...
    @property
    def stopped(self) -> bool: ...
    @property
    def out(self) -> Path: ...
    def preflight(self) -> None: ...
    def open_hold(self) -> float: ...
    def snapshot(self) -> None: ...
    def settle(self) -> bool: ...
    def restore(self, *, production_stopped: bool) -> dict[str, Json]: ...


def window(b: Backend, body: Callable[[dict[str, Json]], None]) -> dict[str, Json]:
    """Preflight, HOLD, snapshot, the window's body; whatever happens after the HOLD may exist, the end follows
    chain.end_action and a failed restore is never hidden."""
    result: dict[str, Json] = {'status': 'INCOMPLETE'}
    hold = False
    try:
        b.preflight()
        result['chain_mode'] = b.mode
        hold = True             # from here a HOLD may exist (an open can time out after writing it)
        result['hold_opened'] = b.open_hold()
        b.snapshot()
        body(result)
        result['status'] = 'COMPLETE'
    except (Exception, KeyboardInterrupt) as exc:
        result['stop'] = f'{type(exc).__name__}: {exc}'
    finally:
        saved = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            try:
                action = chain.end_action(stopped=True, hold_written=b.settle() if hold else False)
                result['end_action'] = action
                if action == 'restore':
                    result['restore'] = b.restore(production_stopped=b.stopped)
                elif action == 'abort':
                    result['restore'] = chain.abort(b.out / 'chain-abort')
                else:
                    ck.require(action == 'none', f'unexpected end action {action!r} for a single window')
                    result['restore'] = {'status': 'NOT NEEDED: no owned HOLD and no chain marker; production untouched'
                                                   + ('; every owned phase fenced' if hold else '')}
            except (Exception, KeyboardInterrupt) as exc:
                result['restore'] = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
    return result


def exit_status(result: dict[str, Json]) -> tuple[int, str]:
    """0 done; 1 stopped with production restored or untouched; 2 production NOT verified; 3 chain abort alerted."""
    status = str(record(result['restore']).get('status', ''))
    if result.get('end_action') == 'abort':          # chain.abort ran the W8 restore and its own watch
        watched = record(record(result['restore']).get('post_close_watch', {})).get('status') == 'PASS'
        if status.startswith('ALERT'):
            return 3, 'PRODUCTION DOWN and a HOLD exists; chain abort touched nothing. Coordinator action required.'
        if status.startswith('RESTORED') and watched:
            return 1, 'chain marker appeared mid-window; chain abort restored and watched production'
        return 2, 'PRODUCTION NOT VERIFIED after chain abort. Inspect chain-abort/summary.json.'
    if 'post_close_watch' in result and record(result['post_close_watch']).get('status') != 'PASS':
        return 2, 'POST-CLOSE PRODUCTION HEALTH FAILED; inspect watchdog log and HOLD before further work.'
    if status.startswith('ALERT'):
        return 3, 'PRODUCTION DOWN and a HOLD exists; chain abort touched nothing. Coordinator action required.'
    if status.startswith('RESTORED') and 'post_close_watch' not in result:
        return 2, 'PRODUCTION RESTORED BUT NOT WATCHED; inspect before further work.'
    if status.startswith(('RESTORED', 'NOT NEEDED')):
        return (0 if 'stop' not in result else 1), ''
    return 2, 'PRODUCTION NOT VERIFIED; a HOLD may remain open. Inspect cmd/restore-*.log or chain-abort/summary.json.'


@dataclass(frozen=True)
class Entry:
    """What a runner gives the shared entry point."""
    description: str
    plan: Callable[[], str]
    make: Callable[[Path, pf.Supervisor, argparse.Namespace, str], CenLive]
    body: Callable[[CenLive], Callable[[dict[str, Json]], None]]
    config: str
    analyze: Callable[[Path], dict[str, Json]] | None = None


def parse(entry: Entry, argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=entry.description, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--precheck', action='store_true', help='read-only preflight; opens no HOLD')
    mode.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--remote-repo', default='dsv41-deploy-offload')
    ap.add_argument('--expected-experiment-commit', default='')
    ap.add_argument('--production-commit', default=n.PROD_COMMIT)
    ap.add_argument('--production-tag', default=n.PROD_TAG)
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    return ap.parse_args(argv)


def main(entry: Entry, argv: list[str] | None = None) -> int:
    args = parse(entry, argv)
    if not (args.execute or args.precheck):
        print(entry.plan())
        return 0
    try:
        ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'WINDOW_COORDINATOR=Deepseek/coordinator required')
        ck.require(not chain.enabled() and not chain.last(), 'CEN windows run single: unset WINDOW_CHAIN and WINDOW_CHAIN_LAST')
        ck.require(args.out is not None and args.out.resolve().parent == pf.CAMPAIGN and not args.out.exists(),
                   'fresh --out directly inside the campaign directory required')
        ck.require(re.fullmatch('[0-9a-f]{40}', args.expected_experiment_commit) is not None, 'full reviewed experiment commit required')
        ck.require(re.fullmatch('[0-9a-f]{40}', args.production_commit) is not None, 'full production commit required')
        ck.require(re.fullmatch('[A-Za-z0-9_.-]+', args.remote_repo) is not None and args.remote_repo != 'dsv41-deploy',
                   'separate experiment checkout required, never ~/dsv41-deploy')
        ck.require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pf.ROOT, text=True).strip() ==
                   args.expected_experiment_commit, 'local HEAD differs from the reviewed commit')
        ck.require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=pf.ROOT,
                                               text=True).strip(), 'local tracked source dirty')
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP before any remote action: {exc}', file=sys.stderr)
        return 1
    out = cast(Path, args.out)
    out.mkdir()
    tag = re.sub('[^A-Za-z0-9_-]', '-', out.name)

    def interrupted(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f'signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    sup = pf.Supervisor(out)
    live = entry.make(out, sup, args, tag)
    if args.precheck:
        try:
            live.preflight()
            result: dict[str, Json] = {'status': 'PRECHECK PASS', 'restore': {'status': 'NOT NEEDED: read-only precheck'}}
        except (Exception, KeyboardInterrupt) as exc:
            result = {'status': 'PRECHECK FAILED', 'stop': f'{type(exc).__name__}: {exc}',
                      'restore': {'status': 'NOT NEEDED: read-only precheck'}}
        finally:
            sup.close()
        (out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
        print(json.dumps(result))
        return 0 if result['status'] == 'PRECHECK PASS' else 1
    result = window(live, entry.body(live))
    sup.close()
    status = str(record(result['restore']).get('status', ''))
    if status.startswith('RESTORED') and result.get('end_action') != 'abort':     # chain.abort watched itself
        def watch_log(message: str) -> None:
            print(message, flush=True)
            with (out / 'post-close-watch.log').open('a') as handle:
                handle.write(message + '\n')
        watched = watchdog.watch(watchdog.ProductionHealth(),
                                 watchdog.ScopedRecovery(tag + '-watch', out / 'post-close-commands'), log=watch_log)
        result['post_close_watch'] = cast(Json, dict(watched.__dict__))
        if watched.status == 'PASS':
            chain.clear()
    if entry.analyze is not None and result.get('status') == 'COMPLETE':
        try:
            result['analysis'] = entry.analyze(out)          # CPU on Controller; never touches a host
        except (Exception, KeyboardInterrupt) as exc:
            result['analysis_error'] = f'{type(exc).__name__}: {exc}'
    result.update({'experiment_commit': args.expected_experiment_commit, 'production_commit': args.production_commit,
                   'production_tag': args.production_tag, 'restore_sha256': pf.RESTORE_SHA256,
                   'config_sha256': pf.digest(pf.ROOT / entry.config),
                   'artifact_sha256': {str(p.relative_to(out)): pf.digest(p) for p in sorted(out.rglob('*.json*'))
                                       if p.is_file() and not p.name.startswith('telemetry-')}})
    (out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'stop': result.get('stop'), 'end_action': result.get('end_action'),
                      'restore': record(result['restore']).get('status')}))
    code, message = exit_status(result)
    if message:
        print(message, file=sys.stderr)
    return code
