"""Read capture error telemetry and latch the existing generation PAUSE file."""
from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import threading
import shlex


def error_count(raw: str) -> int:
    value: object = json.loads(raw)
    if not isinstance(value, dict) or value.get('schema') != 'd2.decode-error-count.v1':
        raise ValueError('missing capture error-count telemetry')
    count = value.get('errors')
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError('invalid capture error count')
    return count


class ErrorMonitor:
    """A telemetry failure also pauses. Only the coordinator removes PAUSE."""
    def __init__(self, read: Callable[[], str], pause: Path, *, every_s: float = 0.5) -> None:
        self.read, self.pause, self.every_s = read, pause, every_s
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name='d2-capture-errors')

    def poll(self) -> bool:
        try:
            count = error_count(self.read())
            reason = f'capture errors={count}' if count else ''
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            reason = 'capture error telemetry unavailable: ' + str(exc)
        if reason:
            self.pause.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation preserves an existing coordinator PAUSE reason.
            try:
                with self.pause.open('x') as fh:
                    fh.write(reason + '\n')
            except FileExistsError:
                pass
            return True
        return False

    def run(self) -> None:
        while not self.stopped.is_set():
            if self.poll():
                return
            self.stopped.wait(self.every_s)

    def start(self) -> None:
        self.poll()  # No submissions precede the first telemetry check.
        self.thread.start()

    def close(self) -> None:
        self.stopped.set()
        self.thread.join(timeout=4)
        if self.thread.is_alive():
            raise RuntimeError('capture error monitor did not stop')


def remote_monitor(host: str, path: str, pause: str) -> ErrorMonitor:
    if not path.startswith('/') or not host or host.startswith('-'):
        raise ValueError('capture error telemetry needs a host and absolute path')
    def read() -> str:
        return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=2', host,
                                       shlex.join(['cat', '--', path])], text=True, timeout=3)
    return ErrorMonitor(read, Path(pause))
