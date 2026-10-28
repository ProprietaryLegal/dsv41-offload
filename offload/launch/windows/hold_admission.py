"""Shared post-drain HOLD admission, independent of production window.sh.

Publish complete bytes by exclusive hard link; a private nonce and saved inode
prove which caller won, even if an acknowledgement was lost. The global lock
also covers the exact ownership check and first stop command. This source is
sent inline inside scoped_remote, including nested receiving-host SSH shims.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

CONTROL = 'dsv41-offload-run/scoped-control'
NOT_OWNED = 'HOLD NOT OWNED BY THIS ADMISSION'


def expiry() -> float:
    deadline = float(os.environ['SCOPED_REMOTE_DEADLINE'])
    if not math.isfinite(deadline) or time.time() >= deadline:
        raise ValueError('immutable admission/stop deadline expired; HOLD retained')
    return deadline


def paths(tag: str, token: str) -> tuple[Path, Path, Path]:
    if re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_-]{0,95}', tag) is None or re.fullmatch('[0-9a-f]{32}', token) is None:
        raise ValueError('invalid admission identity')
    root = Path.home()/CONTROL/tag
    return Path.home()/'dsv41-deploy/HOLD', root/('hold-' + token), root/('receipt-' + token + '.json')


@contextmanager
def transaction() -> Iterator[None]:
    root = Path.home()/CONTROL
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root/'HOLD.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def owned(tag: str, token: str) -> bool:
    hold, candidate, receipt = paths(tag, token)
    try:
        saved = json.loads(receipt.read_text())
        current = hold.lstat()
        original = candidate.lstat()
        if not hold.is_file() or hold.is_symlink() or not candidate.is_file() or candidate.is_symlink():
            return False
        return bool(saved == {'dev': current.st_dev, 'ino': current.st_ino,
                              'sha256': hashlib.sha256(hold.read_bytes()).hexdigest()}
                    and (original.st_dev, original.st_ino) == (current.st_dev, current.st_ino))
    except FileNotFoundError:
        return False


def fetch(path: str) -> str:
    return subprocess.check_output(['curl', '-fsS', '--max-time', '5', 'http://localhost:8000' + path],
                                   text=True, timeout=6)


def idle() -> bool:
    metrics = fetch('/metrics')
    totals = []
    for name in ('num_requests_running', 'num_requests_waiting'):
        values = re.findall(r'^vllm:' + name + r'(?:\{[^\n]*\})?\s+([^\s]+)\s*$', metrics, re.M)
        if not values:
            raise ValueError('drain lacks both request gauges; HOLD not admitted')
        numbers = [float(value) for value in values]
        if not all(math.isfinite(value) and value >= 0 for value in numbers):
            raise ValueError('invalid drain gauges; HOLD not admitted')
        totals.append(sum(numbers))
    return sum(totals) == 0


def admit(tag: str, token: str, purpose: str, drain_s: float, *, allow_unreachable: bool = False) -> None:
    if not math.isfinite(drain_s) or drain_s <= 0 or '\n' in purpose or '\r' in purpose:
        raise ValueError('invalid admission purpose/drain bound')
    hold, candidate, receipt = paths(tag, token)
    until = min(expiry(), time.time() + drain_s)
    # An unreachable watchdog target may already be stopped; it is revalidated
    # immediately before stop. NBW must see working production to admit.
    try:
        fetch('/v1/models')
    except (OSError, subprocess.SubprocessError):
        if not allow_unreachable:
            raise
    unreachable = 0
    while True:
        if time.time() >= until:
            raise ValueError('production drain exhausted; HOLD not admitted')
        try:
            drained = idle()
        except (OSError, subprocess.SubprocessError):
            if not allow_unreachable:
                raise
            # A failed models GET alone cannot bypass active traffic. Only
            # twice-unreachable metrics AND models count as a stopped target;
            # successful metrics always require both gauges to be zero.
            try:
                fetch('/v1/models')
            except (OSError, subprocess.SubprocessError):
                unreachable += 1
                drained = unreachable >= 2
            else:
                raise ValueError('drain metrics unavailable while models responds; HOLD not admitted')
        else:
            unreachable = 0
        if drained:
            break
        time.sleep(min(5, max(0, until - time.time())))
    # No pre-drain check can confer ownership. Exclusive publication and all
    # receipt bytes happen only here, AFTER the drain, under the shared lock.
    with transaction():
        expiry()
        if time.time() >= until or hold.exists() or hold.is_symlink():
            raise ValueError('HOLD already exists or drain expired; refusing overwrite')
        candidate.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                + ' DSV41-OFFLOAD (session coordinator): ' + purpose + '\n').encode()
        with candidate.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        original = candidate.stat()
        with receipt.open('x') as handle:
            json.dump({'dev': original.st_dev, 'ino': original.st_ino,
                       'sha256': hashlib.sha256(data).hexdigest()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        expiry()
        os.link(candidate, hold)  # atomic, complete, exclusive; EEXIST never overwrites
        print('window open: ' + hold.read_text(), end='', flush=True)


def stop(tag: str, token: str, argv: list[str]) -> int:
    if not argv:
        raise ValueError('first stop command missing')
    with transaction():
        expiry()
        if not owned(tag, token):
            raise ValueError('first stop refuses unowned/changed HOLD')
        # Hold the global lock through the actual stop, including a nested SSH
        # command. A different phase's lock cannot claim this global resource.
        return subprocess.call(argv)


def command(mode: str, tag: str, token: str, argv: list[str] | None = None) -> str:
    paths(tag, token)
    return shlex.join(['python3', '-c', Path(__file__).read_text(), mode, tag, token, *(argv or [])])


def open_script(tag: str, token: str, purpose: str, drain_s: float, *, allow_unreachable: bool = False) -> str:
    admit_command = command('admit', tag, token,
        [purpose, str(drain_s), 'allow-unreachable' if allow_unreachable else 'require-serving'])
    placeholder = ('tmux has-session -t dsv41-hold 2>/dev/null || '
                   'tmux new-session -d -s dsv41-hold '
                   + shlex.quote('exec -a "vllm serve [DSV41-OFFLOAD hold]" sleep 86400'))
    return ('set -euo pipefail; cd ~/dsv41-deploy; ' + admit_command
            + '; set -a; . ./dsv41.env; set +a; for node in "$HEAD" "$WORKER"; do '
            + 'ssh -o BatchMode=yes "$node" ' + shlex.quote(placeholder) + '; done')


def main(args: list[str]) -> int:
    mode, tag, token, *argv = args
    if mode == 'admit':
        purpose, drain_s, allow = argv
        admit(tag, token, purpose, float(drain_s), allow_unreachable=allow == 'allow-unreachable')
    elif mode == 'check':
        with transaction():
            hold, _, _ = paths(tag, token)
            print(hold.read_text() if owned(tag, token) else NOT_OWNED, end='\n', flush=True)
    elif mode == 'stop':
        return stop(tag, token, argv)
    else:
        raise ValueError('unknown HOLD admission operation')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
