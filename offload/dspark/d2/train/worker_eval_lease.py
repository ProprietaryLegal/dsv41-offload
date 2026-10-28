"""Explicit solo-card leases for isolated TP2 adapter evaluation. No production pair."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

from .safety import GPU, SOLO_B, assert_gpu_lease


def require_evaluation_lease(emulated: bool) -> None:
    if emulated:
        assert_gpu_lease()
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU+","+SOLO_B or os.environ.get("D2_ADAPTER_EVAL") != "1":
        raise RuntimeError("TP2 adapter evaluation permits only both named solos in order with explicit evaluation flag")
    for fd, suffix in ((8, "240876ad"), (9, "bc62df2a")):
        path = Path.home()/f"dsv41-offload/locks/gpu-{suffix}.lock"
        held, expected = os.fstat(fd), path.stat()
        if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError("adapter evaluation did not inherit the correct solo flock")
        with path.open("a") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(probe, fcntl.LOCK_UN)
                raise RuntimeError("adapter evaluation solo flock was not already held")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
