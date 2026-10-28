"""Remote stdin lease: EOF, signal or timeout terminates only our child process group."""
from __future__ import annotations

import argparse
import os
import selectors
import signal
import subprocess
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--timeout', type=float, required=True)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    command: list[str] = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command or a.timeout <= 0:
        raise SystemExit('positive timeout and command required')
    def interrupted(signum: int, frame: object) -> None:
        raise RuntimeError(f'owned command interrupted by signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        with selectors.DefaultSelector() as sel:
            sel.register(sys.stdin, selectors.EVENT_READ)
            deadline = time.monotonic() + a.timeout
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    raise RuntimeError('owned command timed out')
                for key, _ in sel.select(.2):
                    if os.read(key.fd, 4096) == b'':
                        raise RuntimeError('coordinator lease closed; stopping owned command')
            raise SystemExit(child.returncode)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == '__main__':
    main()
