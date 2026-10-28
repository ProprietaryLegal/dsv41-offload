"""Coordinator-only post-close watchdog; a dry plan unless --execute is explicit.

Poll production for fifteen minutes. Two confirmed failures admit at most one
recovery. Every mutating shell and nested SSH hop uses scoped_remote. W8's HOLD
unlink is deferred until all recovery scopes are fenced and proved absent.
This module does not start itself, contact production during import, or install a
service. CPU tests supply fake health, transport, time and recovery backends.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time
import uuid
from typing import Protocol, cast
import urllib.request

from offload.launch.windows import spark_quiet, production_health, hold_admission
from offload.launch.windows.production_health import (
    BusyHealth as BusyHealth, ProductionHealth as ProductionHealth,
    WatchError as WatchError, MODEL_ROOT as MODEL_ROOT,
    PROBE_TIMEOUT_S as PROBE_TIMEOUT_S, require as require,
)

WATCH_S = 15 * 60
POLL_S = 30
RECOVERY_S = 30 * 60
RESTORE = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/W8/restore.sh')
RESTORE_SHA256 = 'd1c4479dbfbd90bc2036c7bf1561db5b564c2b162a7cde4a121e46bfb9f86a49'
PRODUCTION = 'http://HEAD_IP:8000'
HOSTS = ('localhost', 'sparkhead', 'sparkworker-wired', 'x86host')
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
       '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2']


def restore_verified(step: str, rc: int, text: str) -> None:
    """W8 does not use pipefail; its zero exit status alone proves very little."""
    require(rc == 0, f'W8 {step} rc{rc}; HOLD stays open')
    require(not re.search(r'\b(?:STOP:|DIVERGES\b|CHECK FAILED\b)', text),
            f'W8 {step} reported a failure; HOLD stays open')
    if step == 'start':
        require('START: K1 START VERIFIED' in text, 'W8 start lacks K1 START VERIFIED')
    elif step == 'check':
        matches = re.findall(r'^greedy K1-post vs restored: ([0-9]+) IDENTICAL ([0-9]+)\s*$',
                             text, re.MULTILINE)
        rows = [line for line in text.splitlines() if line.startswith('greedy K1-post vs restored:')]
        require(len(rows) == 6 and len(matches) == 6 and sorted(int(i) for i, _ in matches) == list(range(6)),
                'W8 greedy check requires six distinct IDENTICAL indices 0..5; HOLD stays open')
        require('PRODUCTION HOLD OK' in text and 'CHECK: DONE' in text,
                'W8 check lacks PRODUCTION HOLD OK / CHECK: DONE')
    elif step == 'close':
        require('CLOSE: window closed' in text, 'W8 close not confirmed')
    else:
        raise ValueError('unknown W8 step')


class Health(Protocol):
    def probe(self) -> None: ...


class Recovery(Protocol):
    def recover(self, deadline: float) -> None: ...


@dataclass(frozen=True)
class Result:
    status: str
    polls: int
    failures: int
    recoveries: int


def watch(health: Health, recovery: Recovery, *, log: Callable[[str], None],
          clock: Callable[[], float] = time.monotonic,
          sleep: Callable[[float], None] = time.sleep,
          duration: float = WATCH_S, interval: float = POLL_S) -> Result:
    """Fixed fifteen-minute monitoring horizon; recovery gets a separate fixed lease.

    Successful polls reset the consecutive-failure count. A failed recovery exits
    immediately, and a later second failure after recovery never restarts again.
    The recovery allowance is consumed before any mutation, including a lost reply.
    """
    require(math.isfinite(duration) and duration > 0 and math.isfinite(interval) and interval > 0,
            'invalid watchdog schedule')
    until = clock() + duration
    next_poll = clock()
    consecutive = polls = failures = recoveries = 0
    log(f'WATCH START: duration={duration:g}s interval={interval:g}s priority=20 max_tokens=1 greedy')
    while clock() < until:
        if clock() < next_poll:
            sleep(min(next_poll - clock(), until - clock()))
            if clock() >= until:
                break
        polls += 1
        try:
            health.probe()
        except BusyHealth as exc:
            consecutive = 0
            log(f'WATCH BUSY/INCONCLUSIVE poll={polls}: {exc}; no recovery admitted')
        except Exception as exc:
            failures += 1
            consecutive += 1
            log(f'WATCH FAILURE {consecutive}/2 poll={polls}: {type(exc).__name__}: {exc}')
        else:
            consecutive = 0
            log(f'WATCH HEALTHY poll={polls}')
        if consecutive >= 2:
            if recoveries:
                log('CRITICAL: production failed again; recovery allowance exhausted; coordinator action required')
                return Result('FAILED: recurrence after the single recovery', polls, failures, recoveries)
            recoveries = 1
            # Monotonic deadline never extends when a phase retries or hangs.
            recovery_deadline = clock() + RECOVERY_S
            log('CRITICAL: two consecutive production failures; starting the ONE permitted W8 recovery')
            try:
                with masked_signals():
                    recovery.recover(recovery_deadline)
                require(clock() < recovery_deadline, 'immutable recovery deadline reached')
            except Exception as exc:
                log(f'CRITICAL: RECOVERY FAILED; HOLD stays open: {type(exc).__name__}: {exc}')
                return Result('FAILED: recovery unverified; retain HOLD', polls, failures, recoveries)
            consecutive = 0
            log('RECOVERED: W8 start/check/close verified; six distinct IDENTICAL checks; owned HOLD closed')
        next_poll += interval
        # Do not issue a burst of catch-up probes after slow HTTP or recovery.
        if next_poll <= clock():
            next_poll = clock() + interval
    log(f'WATCH COMPLETE: polls={polls} failures={failures} recoveries={recoveries}')
    return Result('PASS', polls, failures, recoveries)


@contextmanager
def masked_signals() -> Iterator[None]:
    saved = {sig: signal.signal(sig, signal.SIG_IGN)
             for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


class Call(Protocol):
    def __call__(self, label: str, argv: list[str], timeout: float) -> tuple[int, str]: ...


class ScopedRecovery:
    """Production adapter. Instantiation is inert; recover() is coordinator-only.

    No experiment is owned by this watchdog. It refuses to remove any experiment
    container or unknown compute app; those require the original window owner.
    X86host's live TP2 worker remains running and is checked by W8 itself.
    """

    def __init__(self, tag: str, out: Path, *, call: Call | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time, restore: Path = RESTORE,
                 restore_sha256: str | None = None) -> None:
        require(re.fullmatch('[A-Za-z0-9_-]{1,64}', tag) is not None, 'invalid watchdog run id')
        self.tag, self.out, self.clock, self.wall, self.restore = tag, out, clock, wall, restore
        self.restore_sha256 = RESTORE_SHA256 if restore_sha256 is None else restore_sha256
        self.call: Call = call or self.local
        self.deadline = 0.0
        self.wall_deadline = 0.0
        self.phases: list[str] = []
        self.admission_token = uuid.uuid4().hex
        self.local_serial = 0

    def local(self, label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        self.out.mkdir(parents=True, exist_ok=True)
        # Phases are fenced again after close; a serial keeps each exclusive log
        # distinct (TV1-COMPACT-W-20261027T031304Z kept HOLD on a repeated name).
        self.local_serial += 1
        path = self.out / f'{self.local_serial:04d}-{label}.log'
        with path.open('x') as handle:
            process = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                try:
                    rc = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    rc = 124
            finally:
                if process.poll() is None:
                    # Only the local group this method created. Remote descendants
                    # are fenced and killed separately on all receiving hosts.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        return rc, path.read_text()

    def invoke(self, label: str, host: str, command: str, limit: float = 60) -> tuple[int, str]:
        if label in ('cancel-close', 'retain-close'):
            argv = ['bash', '-c', command] if host == 'localhost' else [*SSH, host, command]
            return self.call(label, argv, min(30, limit))
        remaining = self.deadline - self.clock()
        require(remaining > 1, f'immutable recovery deadline reached during {label}; HOLD stays open')
        argv = ['bash', '-c', command] if host == 'localhost' else [*SSH, host, command]
        result = self.call(label, argv, min(limit, remaining))
        require(self.clock() < self.deadline, f'immutable recovery deadline reached during {label}; HOLD stays open')
        return result

    def owned(self, phase: str, host: str, argv: list[str], limit: float, *,
              close_owner: str | None = None) -> tuple[int, str]:
        from offload.launch.windows import scoped_remote as owner
        tag = self.tag + '-' + phase
        require(tag not in self.phases, 'watchdog recovery phase reused')
        self.phases.append(tag)  # ownership survives a lost launch acknowledgement
        command = owner.command('launch', tag, argv, deadline=self.wall_deadline, close_owner=close_owner)
        try:
            return self.invoke('run-' + phase, host, command, limit)
        finally:
            self.quiesce(tag)

    def quiesce(self, tag: str) -> None:
        from offload.launch.windows import scoped_remote as owner
        def control(host: str, command: str) -> tuple[int, str]:
            mode = shlex.split(command)[-2]
            try:
                return self.invoke(mode + '-' + tag + '-' + host, host, command)
            except Exception as exc:
                # Preserve the shared API's all-host attempt within each stage,
                # even when one transport raises instead of returning nonzero.
                return 1, f'{type(exc).__name__}: {exc}'
        try:
            owner.fence_all(tag, control, hosts=HOSTS)
        except Exception as exc:
            raise WatchError(f'remote absence unproved: {exc}; HOLD stays open') from exc

    def recover(self, deadline: float) -> None:
        from offload.launch.windows import scoped_remote as owner
        require(not self.phases, 'watchdog recovery was already admitted')
        require(math.isfinite(deadline) and deadline > self.clock() + 90, 'recovery lease lacks close reserve')
        self.deadline = deadline
        self.wall_deadline = self.wall() + deadline - self.clock() - 5
        require(hashlib.sha256(self.restore.read_bytes()).hexdigest() == self.restore_sha256,
                'W8 restore.sh changed; coordinator must review it before recovery')
        marker = 'POST-CLOSE-WATCH ' + self.tag + ':'
        purpose = marker + ' watchdog recovery after two consecutive production failures'
        opened = hold_admission.open_script(self.tag + '-hold-open', self.admission_token,
            purpose, 19 * 60, allow_unreachable=True)
        rc, text = self.owned('hold-open', 'sparkhead', ['bash', '-c', opened], 20 * 60)
        require(rc == 0 and 'window open:' in text and marker in text, 'watchdog HOLD open unverified; HOLD retained')
        # Stop only the production named by its reviewed checkout; unknown
        # experiments/apps are refused below rather than signalled by name.
        rc, text = self.owned('stop', 'sparkhead', ['bash', '-c',
            hold_admission.command('stop', self.tag + '-hold-open', self.admission_token,
                ['bash', '-c', production_health.stop_command()])], 180)
        require(rc == 0, f'production stop rc{rc}; HOLD stays open')
        for host in ('sparkhead', 'sparkworker-wired'):
            quiet = spark_quiet.command('WATCH SPARK QUIET',
                deadline=min(self.wall_deadline, self.wall() + 350))
            rc, text = self.invoke('quiet-' + host, host, quiet, 360)
            require(rc == 0 and 'WATCH SPARK QUIET' in text.splitlines(), host + ': compute/container absence unproved')
        for step, limit in (('start', 900), ('check', 600), ('close', 120)):
            # Local scope owns the W8 shell; inherited SSH shim scopes each
            # receiving host, including nested start.sh/stop.sh second hops.
            rc, text = self.owned('restore-' + step, 'localhost',
                                  ['bash', str(self.restore), step], limit)
            restore_verified(step, rc, text)
        # W8 close ran under the reviewed rm shim. Its HOLD unlink is deferred.
        # Prove every earlier phase again immediately before releasing HOLD.
        for tag in self.phases:
            self.quiesce(tag)
        require(self.deadline - self.clock() >= 90, 'final HOLD close lacks 90s cancellation/retention reserve')
        tag = self.tag + '-restore-close'
        try:
            try:
                rc, text = self.invoke('commit-close', 'sparkhead',
                                       owner.command('commit-close', tag, deadline=min(self.wall_deadline, self.wall() + 25)), 30)
                require(rc == 0 and owner.CLOSED in text.splitlines(), 'owned HOLD removal unconfirmed')
            finally:
                rc, text = self.invoke('cancel-close', 'sparkhead', owner.command('cancel-close', tag), 30)
                require(rc == 0 and 'CLOSE COMMIT FENCED' in text, 'close commit fence unconfirmed')
            require(self.clock() < self.deadline, 'final HOLD close completed after immutable deadline')
        except Exception:
            rc, text = self.invoke('retain-close', 'sparkhead',
                                   owner.command('retain-close', tag, deadline=self.wall_deadline), 30)
            require(rc == 0 and owner.RETAINED in text.splitlines(), 'original HOLD retention unconfirmed')
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='coordinator only: contact production and permit ONE recovery')
    parser.add_argument('--tag', default='', help='unique closed-window id with watchdog suffix')
    parser.add_argument('--out', type=Path, help='new directory for watchdog event and command logs')
    args = parser.parse_args(argv)
    if not args.execute:
        print('PLAN ONLY: after close, poll :8000 /v1/models + 1-token greedy priority-20 probe every 30s for 15min; '
              'busy/unknown -> inconclusive; two confirmed idle/unreachable failures -> ONE scoped HOLD + W8 recovery; '
              'strict six-distinct-IDENTICAL, all-host absence before HOLD release. No network or mutation.')
        return 0
    require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator',
            'WINDOW_COORDINATOR=Deepseek/coordinator required before watchdog execution')
    require(bool(args.tag) and args.out is not None, '--execute requires --tag and --out')
    out = cast(Path, args.out)
    out.mkdir(parents=True, exist_ok=False)
    def log(message: str) -> None:
        text = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + message
        print(text, flush=True)
        with (out / 'watch.log').open('a') as handle:
            handle.write(text + '\n')
    result = watch(ProductionHealth(), ScopedRecovery(args.tag, out / 'commands'), log=log)
    (out / 'result.json').write_text(json.dumps(result.__dict__, indent=2) + '\n')
    return 0 if result.status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
