"""Run-scoped KL1 remote launcher/fence. Stdlib only; sent inline, never a remote checkout dependency.

The registration lock is held until exec (CLOEXEC). Abort takes the same lock before reading the PID record.
Keep this tiny control directory after restore: an SSH session arriving late must still see its ABORT fence.
"""
from __future__ import annotations

import fcntl
import os
import re
import shlex
import signal
import sys
import time
from pathlib import Path

CONTROL = 'dsv41-offload-run/kl1-control'
FENCED = 'KL1 REMOTE FENCED'
ABSENT = 'KL1 REMOTE LAUNCHER ABSENT'


def state_dir(tag: str) -> Path:
    if re.fullmatch('[A-Za-z0-9_-]+', tag) is None:
        raise ValueError('invalid KL1 run tag')
    return Path.home() / CONTROL / tag


def command(mode: str, tag: str, argv: list[str] | None = None) -> str:
    """Inline the reviewed local source; no staging/SSH race can change the remote cleanup implementation."""
    prefix = ['setsid', '--wait'] if mode == 'launch' else []
    return shlex.join([*prefix, 'python3', '-c', Path(__file__).read_text(), mode, tag, *(argv or [])])


def proc_stat(pid: int) -> tuple[str, int, int]:
    # comm can contain spaces and parentheses; pgrp is field 5, starttime is field 22.
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return fields[0], int(fields[2]), int(fields[19])


def group_members(pgid: int) -> list[tuple[int, int]]:
    members: list[tuple[int, int]] = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            state, group, ticks = proc_stat(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if group == pgid and state not in ('Z', 'X'):
            members.append((int(path.name), ticks))
    return members


def owned_members(pgid: int, ticks: int, tag: str) -> list[tuple[int, int]]:
    members = group_members(pgid)
    token = f'KL1_LAUNCH_TAG={tag}'.encode()
    for pid, started in members:
        try:
            path = Path('/proc') / str(pid)
            if path.stat().st_uid != os.getuid() or started < ticks:
                raise ValueError('KL1 recorded group ownership differs; refusing to signal')
            if pid == pgid:
                if started != ticks:
                    raise ValueError('KL1 launcher PID reused; refusing to signal')
            elif token not in (path / 'environ').read_bytes().split(b'\0'):
                # A process can exit between the stat scan and environ read; zombies have an empty environ.
                if proc_stat(pid)[0] in ('Z', 'X'):
                    continue
                raise ValueError('KL1 group member lacks run identity; refusing to signal')
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def reap(root: Path, tag: str, *, term_s: float = 15, kill_s: float = 5) -> None:
    if not (root / 'ABORT').exists():
        raise ValueError('KL1 reap requires ABORT fence')
    pidfile = root / 'launcher.pid'
    if not pidfile.exists():
        print(ABSENT, flush=True)
        return
    # Integers/text only, exact schema. A stale/reused identity fails closed without touching another group.
    saved = pidfile.read_text().splitlines()
    if len(saved) != 3:
        raise ValueError('invalid KL1 launcher PID record')
    pgid, ticks = int(saved[0]), int(saved[1])
    if pgid <= 1 or ticks <= 0 or saved[2] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('KL1 launcher identity/boot differs; refusing to signal')
    if pgid == os.getpgrp():
        raise ValueError('KL1 cleanup shares launcher group; refusing to signal')
    for sig, seconds in ((signal.SIGTERM, term_s), (signal.SIGKILL, kill_s)):
        if not owned_members(pgid, ticks, tag):
            break
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        until = time.monotonic() + seconds
        while owned_members(pgid, ticks, tag) and time.monotonic() < until:
            time.sleep(.05)
    if owned_members(pgid, ticks, tag):
        raise ValueError('KL1 remote launcher still alive; W8 must not start')
    print(ABSENT, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    mode, tag, *child = args
    if mode not in ('launch', 'fence', 'reap'):
        raise ValueError('unknown KL1 remote operation')
    root = state_dir(tag)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if mode == 'fence':
            (root / 'ABORT').write_text('KL1 restore: refuse this run forever\n')
            print(FENCED, flush=True)
            return 0
        if mode == 'launch':
            if (root / 'ABORT').exists():
                raise ValueError('KL1 launch fenced by ABORT')
            if (root / 'launcher.pid').exists():
                raise ValueError('KL1 run already has a launcher')
            pid = os.getpid()
            if not child or os.getpgrp() != pid or os.getsid(0) != pid:
                raise ValueError('KL1 launcher requires its own session/process group')
            ticks = proc_stat(pid)[2]
            (root / 'launcher.pid').write_text(f'{pid}\n{ticks}\n' + Path('/proc/sys/kernel/random/boot_id').read_text())
            # This lock's non-inheritable FD closes only at exec. Fence cannot miss a registered launcher.
            if (root / 'ABORT').exists():
                raise ValueError('KL1 launch fenced by ABORT')
            os.execvpe(child[0], child, {**os.environ, 'KL1_LAUNCH_TAG': tag})
    reap(root, tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
