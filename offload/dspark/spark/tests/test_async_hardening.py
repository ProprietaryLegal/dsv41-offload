# SPDX-License-Identifier: Apache-2.0
"""Async hand-off hardening after review S9 (F2, F3, F5): S9's probes P1-P3 with the expectation reversed, on CPU with
the real C words, publish and native watchdog (CPU only; the GPU side is tests_gpu/)."""

import ctypes
import shutil
import threading
import time

import numpy as np
import pytest
import torch

from remote_dspark import streamsync
from remote_dspark.mock_worker import TcpStubServer, WorkerCore
from remote_dspark.round import STATUS_OK, STATUS_ROUND_FAILED, RemoteRound
from remote_dspark.streamsync import FAR_AHEAD, CpuHostFlagSync, HostFlagCore

import sim
from test_round import cfg_for, churn, hello, run_step

pytestmark = pytest.mark.skipif(shutil.which("cc") is None and shutil.which("gcc") is None,
                                reason="no C compiler for hostflag.c")
M32 = 0xFFFFFFFF


def at_or_after(a: int, b: int) -> bool:
    return ((a - b) & M32) < (1 << 31)


class GpuLikeSync(HostFlagCore):
    """The engine side modelled as a GPU: gathered() is the stream's (monotonic) write, wait_release() an enqueue that
    returns at once, copy_reply() reads the page. The proxy side and the watchdog are the production ones."""

    def __init__(self):
        super().__init__("", spin_us=200, nap_us=20)
        # a native wait that HOLDS the GIL, as a pybind11 extension (m.def, no gil_scoped_release) does when its first
        # kernel launch waits for the parked stream (S9 P1)
        self.gil_holding = ctypes.PyDLL(self.lib._name)
        self.gil_holding.dsv41_hf_wait_geq.restype = ctypes.c_int
        self.gil_holding.dsv41_hf_wait_geq.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int64,
                                                       ctypes.c_int64, ctypes.c_int64]

    def gathered(self, v, stream=None):
        self.lib.dsv41_hf_release_max(ctypes.c_void_p(self.gathered_host), v & M32)

    def wait_release(self, v, stream=None):
        pass

    def copy_reply(self, dst, nbytes=None, stream=None):
        ctypes.memmove(dst.data_ptr(), self.reply_host, self.reply_bytes if nbytes is None else nbytes)

    def self_test(self, timeout_s=2.0):
        return True, "gpu-like"

    def reply_status(self) -> int:
        return int(np.frombuffer((ctypes.c_int32 * 1).from_address(self.reply_host), dtype=np.int32)[0])

    def engine_waits_for_stream(self, t, timeout_s, hold_gil):
        f = self.gil_holding.dsv41_hf_wait_geq if hold_gil else self.lib.dsv41_hf_wait_geq
        t0 = time.perf_counter()
        r = f(ctypes.c_void_p(self.released_host), t & M32, int(timeout_s * 1e9), 0, 0)
        return r, time.perf_counter() - t0


@pytest.fixture
def server():
    srv = TcpStubServer(WorkerCore(hello()), "127.0.0.1", 0).start()
    yield srv
    srv.stop()


def _round(port, sync, **kw):
    cfg = cfg_for(port, VLLM_DSPARK_REMOTE_ASYNC="1", **kw)
    rnd = RemoteRound(cfg, hello(), device=torch.device("cpu"), tp_rank=0, broadcast_fn=None,
                      background=True, stream_sync=sync)
    t = time.monotonic() + 3
    while not rnd.client.ready and time.monotonic() < t:
        time.sleep(0.002)
    assert rnd.client.ready
    return rnd


def _submit(rnd, eng):
    sd = eng.step()
    return rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected, sd.last_sampled, sd.next_prefill_tokens)


def _wait(pred, s=5.0):
    t = time.monotonic() + s
    while not pred() and time.monotonic() < t:
        time.sleep(0.005)
    return pred()


# ------------------------------------------------------------------------------------------- P1 reversed

@pytest.mark.parametrize("hold_gil", [False, True])
def test_P1r_gil_held_wait_is_released_by_the_watchdog(server, hold_gil):
    """A module load waiting for the parked stream: with the GIL released the proxy releases it (~ the worker delay);
    while the waiting thread HOLDS the GIL the proxy cannot run, and the native watchdog fails and releases the round
    within its deadline. The proxy then finds the round taken, marks the slots dirty and carries on."""
    sync = GpuLikeSync()
    rnd = _round(server.port, sync, VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS="150")
    server.core.inject.delay_ms = 40
    eng = sim.Engine(seed=5)
    eng.new_request(0, prompt_len=200)
    _submit(rnd, eng)
    t = rnd.rounds
    r, dt = sync.engine_waits_for_stream(t, 3.0, hold_gil)
    assert r == 0, f"never released (words {sync.values()})"
    if hold_gil:
        assert 0.14 <= dt < 1.0, dt                       # the watchdog's deadline, not the 3 s timeout
        assert sync.reply_status() == STATUS_ROUND_FAILED   # fail-closed reply for that round
        assert _wait(lambda: rnd.stats.snapshot()["counters"].get("watchdog_preempted", 0) == 1)
        assert sync.watchdog_stats()["fires"] == 1
    else:
        assert dt < 0.14, dt                              # the proxy released it, about the worker delay
        assert sync.watchdog_stats()["fires"] == 0
    # the proxy is alive and releases later rounds itself, without the watchdog
    server.core.inject.delay_ms = 0
    fires = sync.watchdog_stats()["fires"]
    for _ in range(20):
        _submit(rnd, eng)
        assert _wait(lambda: at_or_after(sync.values()[1], rnd.rounds), 2.0), sync.values()
    assert rnd._thread.is_alive() and sync.watchdog_stats()["fires"] == fires
    rnd.close()


# ------------------------------------------------------------------------------------------- P2 reversed

@pytest.mark.parametrize("exc", [RuntimeError("s9: injected error after the release"),
                                 SystemExit("s9: BaseException in the proxy")])
def test_P2r_proxy_death_fails_closed(server, monkeypatch, exc):
    """An exception outside the round's handler (here _maybe_log, after the publish) or a BaseException ends the
    proxy thread: every pending and later round is released at once with a failed reply, later rounds skip the
    hand-off (zero drafts, no wait), and the death is recorded."""
    sync = CpuHostFlagSync(timeout_s=2.0)
    rnd = _round(server.port, sync)
    eng = sim.Engine(seed=7)
    eng.new_request(0, prompt_len=200)
    for _ in range(3):
        _submit(rnd, eng)
    calls = {"n": 0}
    real = rnd._maybe_log

    def boom():
        calls["n"] += 1
        if calls["n"] == 1:
            raise exc
        real()

    monkeypatch.setattr(rnd, "_maybe_log", boom)
    _submit(rnd, eng)                 # published, then the thread dies
    assert _wait(lambda: not rnd._thread.is_alive())
    assert rnd._dead is not None and type(exc).__name__ in rnd._dead
    g, r, c = sync.values()
    assert at_or_after(r, rnd.rounds + FAR_AHEAD - 1)       # released far ahead
    for _ in range(5):
        t0 = time.monotonic()
        res = _submit(rnd, eng)       # never waits (CpuHostFlagSync would time out after 2 s)
        assert time.monotonic() - t0 < 0.5
        assert res.read_status() == STATUS_ROUND_FAILED and not res.drafts.any() and not res.conf.any()
    snap = rnd.stats.snapshot()
    assert snap["counters"]["fallback_proxy_dead"] == 5 and snap["async_dead"] == rnd._dead
    rnd.close()


def test_P2r_dead_before_any_publish_still_fails_closed(server):
    """A proxy that dies while the engine had already enqueued a wait: fail_all's far-ahead release covers it."""
    sync = GpuLikeSync()
    rnd = _round(server.port, sync)
    eng = sim.Engine(seed=8)
    eng.new_request(0, prompt_len=100)
    rnd._q.put("not an item")         # unpacking fails outside the round handler: the thread dies
    assert _wait(lambda: not rnd._thread.is_alive())
    _submit(rnd, eng)
    assert at_or_after(sync.values()[1], rnd.rounds) and sync.reply_status() == STATUS_ROUND_FAILED
    rnd.close()


# ------------------------------------------------------------------------------------------- P3 reversed

def test_P3r_close_never_moves_the_release_word_backwards(server):
    """close() while the proxy is still inside a slow round: the release word is claimed and released far ahead and
    stays there; the late proxy can neither publish nor move it back."""
    sync = GpuLikeSync()
    rnd = _round(server.port, sync, VLLM_DSPARK_REMOTE_TIMEOUT_MS="9000",
                 VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS="60000")
    eng = sim.Engine(seed=9)
    eng.new_request(0, prompt_len=200)
    server.core.inject.delay_ms = 6500
    for _ in range(3):
        _submit(rnd, eng)             # GPU-like: the engine runs ahead; round 1 is slow on the worker
    assert rnd.rounds == 3
    rnd.close()                       # join times out after 5 s, then fail_all far ahead
    after_close = sync.values()[1]
    assert at_or_after(after_close, 3 + FAR_AHEAD - 1)
    assert _wait(lambda: not rnd._thread.is_alive(), 15)
    final = sync.values()[1]
    assert final == after_close       # never backwards
    assert sync.reply_status() == STATUS_ROUND_FAILED


# ------------------------------------------------------------------------------------------- normal operation

def test_watchdog_silent_in_normal_operation(server):
    """Through ok rounds, failures, resyncs and a worker restart the proxy releases every round itself: the watchdog
    never fires, and every usable draft stays exact."""
    sync = CpuHostFlagSync(timeout_s=5)
    rnd = _round(server.port, sync)
    eng = sim.Engine(seed=31)
    for sl in range(sim.M):
        eng.new_request(sl, prompt_len=400)
    rng = np.random.default_rng(31)
    exact = 0
    for step in range(250):
        churn(eng, step, rng)
        if not any(eng.slots):
            eng.new_request(0)
        if step == 60:
            server.core.inject.delay_ms = 300      # past the 150 ms timeout, below the 600 ms watchdog
        if step == 61:
            server.core.inject.delay_ms = 0
        if step == 150:
            server.core = WorkerCore(hello())
        res, checked = run_step(rnd, eng, eng.step())
        exact += checked
    st = rnd.sync.watchdog_stats()
    assert st["running"] and st["fires"] == 0 and st["deadline_ms"] == 600
    assert exact > 300
    c = rnd.stats.snapshot()["counters"]
    assert c.get("watchdog_preempted", 0) == 0 and c.get("resync_rounds", 0) >= 1
    rnd.close()


# ------------------------------------------------------------------------------------------- the C primitives

@pytest.fixture
def core():
    c = HostFlagCore("", spin_us=0, nap_us=0)
    yield c
    c.close()


def _ld(c, addr):
    return int(c.lib.dsv41_hf_load(ctypes.c_void_p(addr)))


def test_release_max_is_monotonic_and_cyclic(core):
    r = ctypes.c_void_p(core.released_host)
    core.lib.dsv41_hf_release_max(r, 10)
    core.lib.dsv41_hf_release_max(r, 7)                    # backwards: ignored
    assert _ld(core, core.released_host) == 10
    core.lib.dsv41_hf_store(r, 3)                           # the v1 name is monotonic too (S9 F3)
    assert _ld(core, core.released_host) == 10
    core.lib.dsv41_hf_set(r, 0xFFFFFFF0)
    core.lib.dsv41_hf_release_max(r, 5)                     # 5 is after 0xFFFFFFF0 cyclically
    assert _ld(core, core.released_host) == 5


def test_publish_and_fail_are_exclusive_per_round(core):
    core.attach(16, STATUS_ROUND_FAILED, 10_000)
    assert core.publish(1, np.array([0, 1, 2, 3], np.int32))
    assert not core.publish(1, np.array([9, 9, 9, 9], np.int32))       # already claimed: nothing written
    got = np.frombuffer((ctypes.c_int32 * 4).from_address(core.reply_host), dtype=np.int32).tolist()
    assert got == [0, 1, 2, 3] and core.values()[1:] == (1, 1)
    assert core.fail_all(2)
    assert not core.publish(2, np.array([5, 5, 5, 5], np.int32))
    got = np.frombuffer((ctypes.c_int32 * 4).from_address(core.reply_host), dtype=np.int32).tolist()
    assert got == [STATUS_ROUND_FAILED, 0, 0, 0]
    assert core.fail_ahead() and at_or_after(core.values()[1], 2 + FAR_AHEAD)


def test_watchdog_fires_only_after_the_deadline(core):
    core.attach(16, STATUS_ROUND_FAILED, 80)
    g = ctypes.c_void_p(core.gathered_host)
    # round 1 released in time by publish: no fire
    core.lib.dsv41_hf_release_max(g, 1)
    time.sleep(0.03)
    assert core.publish(1, np.array([0, 4, 4, 4], np.int32))
    time.sleep(0.15)
    assert core.watchdog_stats()["fires"] == 0
    # round 2 never released: failed and released after ~80 ms
    t0 = time.monotonic()
    core.lib.dsv41_hf_release_max(g, 2)
    assert _wait(lambda: core.values()[1] == 2, 2.0)
    dt = time.monotonic() - t0
    assert 0.075 <= dt < 0.5, dt
    st = core.watchdog_stats()
    assert st["fires"] == 1 and st["last_failed_round"] == 2 and st["max_late_ms"] >= 75
    got = np.frombuffer((ctypes.c_int32 * 4).from_address(core.reply_host), dtype=np.int32).tolist()
    assert got == [STATUS_ROUND_FAILED, 0, 0, 0]
    assert not core.publish(2, np.array([1, 1, 1, 1], np.int32))       # the late proxy loses the claim


def test_attach_writes_a_failed_reply_and_refuses_misuse(core):
    with pytest.raises(streamsync.StreamSyncError):
        core.attach(streamsync.REPLY_CAPACITY + 4, STATUS_ROUND_FAILED, 100)
    core.attach(16, STATUS_ROUND_FAILED, 100)
    got = np.frombuffer((ctypes.c_int32 * 4).from_address(core.reply_host), dtype=np.int32).tolist()
    assert got == [STATUS_ROUND_FAILED, 0, 0, 0]
    with pytest.raises(streamsync.StreamSyncError, match="already attached"):
        core.attach(16, STATUS_ROUND_FAILED, 100)
    with pytest.raises(streamsync.StreamSyncError, match="watchdog running"):
        core.reset()
    core.stop_watchdog()
    core.reset()
    assert core.values() == (0, 0, 0)
