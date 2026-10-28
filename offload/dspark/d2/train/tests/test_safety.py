from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from offload.dspark.d2.train.safety import GPU, assert_gpu_lease
from offload.dspark.d2.train import lease


def test_only_named_gpu_and_actual_exclusive_inherited_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    lock = tmp_path/"dsv41-offload/locks/gpu-240876ad.lock"
    lock.parent.mkdir(parents=True)
    with lock.open("w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        monkeypatch.setenv("D2_LOCK_FD", str(f.fileno()))
        assert_gpu_lease()
        for forbidden in ("GPU-c4e7a371", "GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc"):
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", forbidden)
            with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
                assert_gpu_lease()


def test_dedicated_lease_without_marker_checkpoints_on_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    ready, checkpoint = tmp_path/"ready", tmp_path/"checkpoint"
    child = tmp_path/"child.py"
    child.write_text("import signal,time,sys\nfrom pathlib import Path\n"
                     f"def done(*args):\n Path({str(checkpoint)!r}).write_text('saved'); sys.exit(0)\n"
                     "signal.signal(signal.SIGTERM,done)\n"
                     f"Path({str(ready)!r}).touch()\nwhile True: time.sleep(.01)\n")
    original_sleep = time.sleep
    def revoke() -> None:
        deadline = time.monotonic()+10
        while not ready.exists() and time.monotonic() < deadline:
            original_sleep(.01)
        os.kill(os.getpid(), signal.SIGTERM)
    monkeypatch.setattr(lease, "assert_gpu_lease", lambda: None)
    monkeypatch.setattr(lease, "leased_gpu", lambda: GPU)
    monkeypatch.setattr(lease, "power_logger_running", lambda: True)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(time, "sleep", lambda _: original_sleep(.01))
    monkeypatch.setattr(sys, "argv", ["lease", "--", sys.executable, str(child)])
    with (tmp_path/"fd").open("w") as f:
        monkeypatch.setenv("D2_LOCK_FD", str(f.fileno()))
        thread = threading.Thread(target=revoke)
        thread.start()
        begin = time.monotonic()
        lease.main()
        thread.join()
    assert checkpoint.read_text() == "saved"
    assert time.monotonic()-begin < 60
