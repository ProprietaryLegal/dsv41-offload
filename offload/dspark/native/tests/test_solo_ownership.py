from __future__ import annotations

import fcntl
from pathlib import Path

import pytest

from offload.dspark.kernels.fp8dense.api import require_lock
from offload.dspark.worker.tp2 import check_visible

SOLO = "GPU-bc62df2a-59cf-9828-7d43-558d1355bab7"


def test_new_solo_requires_its_actual_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", SOLO)
    directory = tmp_path / "dsv41-offload/locks"
    directory.mkdir(parents=True)
    lock = directory / "gpu-bc62df2a.lock"
    with lock.open("a") as held:
        monkeypatch.setenv("KX_FP8DENSE_LOCK_FD", str(held.fileno()))
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_lock()
        with lock.open("a") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert check_visible(pair=False) == SOLO
        with pytest.raises(SystemExit):
            check_visible(pair=True)
        with (directory / "wrong.lock").open("a") as other:
            monkeypatch.setenv("KX_FP8DENSE_LOCK_FD", str(other.fileno()))
            with pytest.raises(RuntimeError, match="not the GPU lock"):
                require_lock()
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        with pytest.raises(RuntimeError, match="pin"):
            require_lock()
