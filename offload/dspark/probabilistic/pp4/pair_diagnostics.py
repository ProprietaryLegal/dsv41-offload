"""Stdlib-only progress and traceback timer for the untimed physical probe."""
from __future__ import annotations

import atexit
import faulthandler
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

DUMP_INTERVAL_S = 30.0  # before either the 60s or 240s parent subprocess limit
_START = time.monotonic()


def progress(label: str) -> None:
    print(f'PP4 PAIR +{time.monotonic()-_START:.3f}s {label}', flush=True)


@contextmanager
def phase(label: str) -> Iterator[None]:
    progress(f'BEGIN {label}')
    try:
        yield
    except BaseException as error:
        progress(f'FAIL {label} ({type(error).__name__})')
        raise
    else:
        progress(f'END {label}')


def start_watchdog() -> None:
    """Dump Python stacks without extending deadlines, killing or resetting CUDA.

    Armed before torch import in the CLI process only. Repeated dumps also
    cover a blocked extension load/driver call where a Python timer cannot run.
    It is diagnostic only: the fenced parent still owns the hard child timeout.
    """
    faulthandler.enable(file=sys.stderr)
    faulthandler.dump_traceback_later(DUMP_INTERVAL_S, repeat=True, file=sys.stderr, exit=False)
    atexit.register(stop_watchdog)  # includes an exception during module imports
    progress(f'watchdog armed interval={DUMP_INTERVAL_S:g}s')


def stop_watchdog() -> None:
    faulthandler.cancel_dump_traceback_later()
