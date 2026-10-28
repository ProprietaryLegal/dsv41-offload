# SPDX-License-Identifier: Apache-2.0
"""S11 P-2: a process proxy that cannot start falls back to synchronous rounds (unless the start is strict), and every
refusal path releases the shared region: the watchdog stops, both of the region's fds and its mapping are gone.

S11's Q2 probe (a proxy that exits at once made RemoteRound raise and the engine refuse to start) is run at the engine
start: the speculator's _make_round with stubbed vLLM, the CPU stand-in of the hand-off on a real shared region (the
CUDA refusal paths of _make_stream_sync are covered on the GPU: tests_gpu/gpu_refusal_probe.py)."""

import gc
import os
import shutil
import time

import pytest
import torch

import sim
import vllm_stubs

vllm_stubs.install()

from remote_dspark import proxy_proc  # noqa: E402
from remote_dspark import speculator as S  # noqa: E402
from remote_dspark.shmregion import RegionLayout, SharedRegion  # noqa: E402
from remote_dspark.stats import Stats  # noqa: E402
from remote_dspark.streamsync import CpuHostFlagSync  # noqa: E402

from test_round import cfg_for, hello, run_step  # noqa: E402
from test_speculator import propose, remote_env, server, truth_drafts  # noqa: E402,F401

pytestmark = pytest.mark.skipif(shutil.which("cc") is None and shutil.which("gcc") is None,
                                reason="no C compiler for hostflag.c")
MEMFD = "memfd:dsv41-remote-dspark"


def held(ino: int) -> tuple[int, int]:
    """(fds, mappings) of this process that refer to the memfd with inode ino."""
    fds = 0
    for f in os.listdir("/proc/self/fd"):
        try:
            if MEMFD in os.readlink(f"/proc/self/fd/{f}") and os.stat(f"/proc/self/fd/{f}").st_ino == ino:
                fds += 1
        except OSError:
            pass
    with open("/proc/self/maps") as m:
        maps = sum(1 for line in m if MEMFD in line and int(line.split()[4]) == ino)
    return fds, maps


class Recorder:
    """_make_stream_sync stand-in: the CPU hand-off on a fresh shared region, recorded for the leak check."""

    def __init__(self):
        self.syncs = []

    def __call__(self, cfg, device, stats, hello_):
        region = SharedRegion.create(RegionLayout.for_hello(hello_))
        sync = CpuHostFlagSync(timeout_s=2.0, region=region)
        self.syncs.append((sync, region, os.fstat(region.fd).st_ino))
        return sync


@pytest.fixture
def proxy_exits(monkeypatch):
    monkeypatch.setattr(proxy_proc, "BOOT", "import sys\nsys.exit(3)\n")
    rec = Recorder()
    monkeypatch.setattr(S, "_make_stream_sync", rec)
    return rec


def test_proxy_start_failure_falls_back_to_synchronous_rounds(remote_env, proxy_exits):
    """S11 Q2 reversed: the engine starts with synchronous rounds; the watchdog is stopped, the region released."""
    remote_env("capacity", VLLM_DSPARK_REMOTE_ASYNC="1", VLLM_DSPARK_REMOTE_ASYNC_PROXY="process",
               VLLM_DSPARK_REMOTE_STRICT_START="0")
    spec = S.init_remote_dspark_speculator(vllm_stubs.vllm_config(), torch.device("cpu"))
    (sync, region, ino), = proxy_exits.syncs
    rnd = spec.remote
    assert not rnd.async_mode and rnd.proxy is None and rnd.client is not None
    why = spec.stats.snapshot()["async_fallback_reason"]
    assert "exited with 3 before it started" in why, why
    assert sync._wd is None and sync.page == 0                      # watchdog stopped, atexit hook inert
    assert held(ino) == (0, 0), held(ino)                            # both fds and the mapping are gone
    eng = sim.Engine(seed=31)
    eng.new_request(0, prompt_len=40)
    exact = 0
    for _ in range(8):                                                # synchronous rounds, exact drafts
        sd = eng.step()
        out = propose(spec, sd)
        ids, _ = truth_drafts(eng, sd)
        assert torch.equal(out, torch.from_numpy(ids))
        exact += int(any(sd.need_draft))
    assert exact >= 4
    rnd.client.close()


def test_strict_start_failure_raises_and_releases(remote_env, proxy_exits):
    remote_env("capacity", VLLM_DSPARK_REMOTE_ASYNC="1", VLLM_DSPARK_REMOTE_ASYNC_PROXY="process",
               VLLM_DSPARK_REMOTE_STRICT_START="1")
    with pytest.raises(RuntimeError, match="exited with 3 before it started"):
        S.init_remote_dspark_speculator(vllm_stubs.vllm_config(), torch.device("cpu"))
    (sync, region, ino), = proxy_exits.syncs
    gc.collect()
    assert sync._wd is None and held(ino) == (0, 0), held(ino)


def test_start_timeout_falls_back_and_kills_the_proxy(server, monkeypatch):
    """A proxy that never reports its start (hangs in its imports): killed at the start timeout, synchronous rounds."""
    from remote_dspark.round import make_remote_round

    monkeypatch.setattr(proxy_proc, "BOOT", "import time\ntime.sleep(60)\n")
    monkeypatch.setattr(proxy_proc, "START_TIMEOUT_S", 0.5)
    import remote_dspark.round as R
    monkeypatch.setattr(R, "START_TIMEOUT_S", 0.5)
    region = SharedRegion.create(RegionLayout.for_hello(hello()))
    ino = os.fstat(region.fd).st_ino
    sync = CpuHostFlagSync(timeout_s=2.0, region=region)
    stats = Stats()
    cfg = cfg_for(server.port, VLLM_DSPARK_REMOTE_ASYNC="1", VLLM_DSPARK_REMOTE_ASYNC_PROXY="process",
                  VLLM_DSPARK_REMOTE_CONNECT_TIMEOUT_MS="100")
    rnd = make_remote_round(cfg, hello(), device=torch.device("cpu"), tp_rank=0, broadcast_fn=None, stats=stats,
                            background=True, stream_sync=sync)
    assert not rnd.async_mode and "did not start within" in stats.snapshot()["async_fallback_reason"]
    assert region.closed and held(ino) == (0, 0)
    t = time.monotonic() + 3
    while not rnd.client.ready and time.monotonic() < t:
        time.sleep(0.005)
    eng = sim.Engine(seed=8)
    eng.new_request(0, prompt_len=30)
    for _ in range(5):
        run_step(rnd, eng, eng.step())
    rnd.client.close()


def test_release_hand_off_unmaps_the_region():
    """The refusal helper on its own: watchdog stopped, both fds and the mapping gone, idempotent."""
    from remote_dspark.streamsync import release_hand_off

    region = SharedRegion.create(RegionLayout.for_hello(hello()))
    ino = os.fstat(region.fd).st_ino
    sync = CpuHostFlagSync(timeout_s=1.0, region=region)
    sync.attach(64, 2, 100)
    assert held(ino) == (2, 1)            # the memfd and mmap's dup of it, one mapping
    assert release_hand_off(sync) is None
    assert sync._wd is None and sync.page == 0 and region.closed and held(ino) == (0, 0)
    assert release_hand_off(None, None, region) is None


def test_release_reports_a_view_that_is_still_alive():
    """A numpy view holds a buffer export of the mapping, so it cannot be unmapped under it: reported, not crashed,
    and released once the view is gone. (torch.frombuffer keeps no export: the failed constructor drops its torch views
    of the region itself, and nothing touches them after the release.)"""
    import numpy as np

    from remote_dspark.streamsync import release_hand_off

    region = SharedRegion.create(RegionLayout.for_hello(hello()))
    ino = os.fstat(region.fd).st_ino
    view = np.frombuffer(region.mm, dtype=np.uint8, count=16)
    why = release_hand_off(None, None, region)
    assert why and "still mapped" in why and held(ino) == (1, 1)     # our fd is closed; mmap's dup + mapping stay
    del view
    assert release_hand_off(None, None, region) is None and held(ino) == (0, 0)


def test_watchdog_start_failure_falls_back_too(server, monkeypatch):
    """Every failure of the async path's start falls back, the watchdog thread's start included."""
    from remote_dspark.round import make_remote_round
    from remote_dspark.streamsync import StreamSyncError

    region = SharedRegion.create(RegionLayout.for_hello(hello()))
    ino = os.fstat(region.fd).st_ino
    sync = CpuHostFlagSync(timeout_s=2.0, region=region)

    def refuse(*a, **k):
        raise StreamSyncError("hostflag watchdog did not start (errno 11, injected)")

    monkeypatch.setattr(sync, "attach", refuse)
    stats = Stats()
    cfg = cfg_for(server.port, VLLM_DSPARK_REMOTE_ASYNC="1", VLLM_DSPARK_REMOTE_ASYNC_PROXY="process")
    rnd = make_remote_round(cfg, hello(), device=torch.device("cpu"), tp_rank=0, broadcast_fn=None, stats=stats,
                            background=True, stream_sync=sync)
    assert not rnd.async_mode and "watchdog did not start" in stats.snapshot()["async_fallback_reason"]
    assert region.closed and held(ino) == (0, 0)
    rnd.client.close()
