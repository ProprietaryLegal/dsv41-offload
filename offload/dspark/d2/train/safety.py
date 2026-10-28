"""Lease validation shared by GPU entrypoints. No service/power mutation."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess

GPU = "GPU-240876ad-ea42-47de-1b8d-f6153810b7e1"
SOLO_B = "GPU-bc62df2a-59cf-9828-7d43-558d1355bab7"


def leased_gpu() -> str:
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    allowed = (GPU, SOLO_B) if os.environ.get("D2_PILOT_LEASE") == "1" else (GPU,)
    if selected not in allowed:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must equal a permitted solo UUID: {allowed}")
    return selected


def assert_gpu_lease() -> None:
    selected = leased_gpu()
    fd = int(os.environ.get("D2_LOCK_FD", "-1"))
    expected = Path.home()/f"dsv41-offload/locks/gpu-{selected.split('-')[1]}.lock"
    if fd < 0 or (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (expected.stat().st_dev, expected.stat().st_ino):
        raise RuntimeError("missing inherited D2 exclusive GPU lock")
    # A new file description must be unable to lock the inode while ours holds it.
    with expected.open("a") as probe:
        try:
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("inherited descriptor does not own the GPU lock") from exc
            return
        fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
    raise RuntimeError("GPU lock descriptor is open but not held")


def power_logger_running() -> bool:
    return subprocess.run(["tmux", "has-session", "-t", "=dsv41-powerlog"], capture_output=True).returncode == 0
