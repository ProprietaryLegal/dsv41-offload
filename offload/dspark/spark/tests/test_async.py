# SPDX-License-Identifier: Apache-2.0
"""Async capacity mode (proxy thread + host-word hand-off), on CPU with CpuHostFlagSync.

CpuHostFlagSync runs the real C words, publish and watchdog; the engine thread's "stream wait"
blocks until the round is released, which exercises the thread, the FIFO hand-off, the
always-release rule, staged resync windows and the exactness of drafts. The CUDA side (stream memory ops on registered
host memory) is self-tested at startup (streamsync.HostFlagSync.self_test) and covered by
tests_gpu/ on a real GPU."""

import time

import numpy as np
import pytest
import torch

from remote_dspark import wire
from remote_dspark.mock_worker import TcpStubServer, WorkerCore
from remote_dspark.round import STATUS_OK, STATUS_ROUND_FAILED, RemoteRound
import threading
import sys

from remote_dspark.streamsync import CpuHostFlagSync

import sim
from test_round import cfg_for, churn, hello, run_step


@pytest.fixture
def server():
    srv = TcpStubServer(WorkerCore(hello()), "127.0.0.1", 0).start()
    yield srv
    srv.stop()


def make_async_round(port):
    cfg = cfg_for(port, VLLM_DSPARK_REMOTE_ASYNC="1")
    rnd = RemoteRound(cfg, hello(), device=torch.device("cpu"), tp_rank=0, broadcast_fn=None,
                      background=True, stream_sync=CpuHostFlagSync(timeout_s=5))
    t = time.monotonic() + 3
    while not rnd.client.ready and time.monotonic() < t:
        time.sleep(0.002)
    assert rnd.client.ready and rnd.async_mode
    return rnd


def test_async_exact_through_failures(server):
    rnd = make_async_round(server.port)
    eng = sim.Engine(seed=13)
    for s in range(sim.M):
        eng.new_request(s, prompt_len=500)
    rng = np.random.default_rng(13)
    statuses, exact_after = [], 0
    for step in range(220):
        churn(eng, step, rng)
        if not any(eng.slots):
            eng.new_request(0)
        if step == 40:
            server.core.inject.delay_ms = 400
        if step == 41:
            server.core.inject.delay_ms = 0
        if step == 90:
            server.core = WorkerCore(hello())
        if step == 140:
            server.core.inject.drop_next_step = True
        res, checked = run_step(rnd, eng, eng.step())
        st = res.read_status()
        statuses.append(st)
        if step > 40 and st == STATUS_OK:
            exact_after += checked
        if st != STATUS_OK:
            time.sleep(0.01)
    c = rnd.stats.snapshot()["counters"]
    assert c.get("fallback_timeout", 0) >= 1 and c.get("resync_rounds", 0) >= 2
    # resync rows come from windows the engine side staged after a request (never the device)
    assert c.get("fallback_resync_pending", 0) >= 1 and c.get("resync_windows_staged", 0) >= 1
    assert exact_after > 150
    assert statuses[-15:] == [STATUS_OK] * 15
    assert rnd.stats.snapshot()["async"] is True
    rnd.close()


def test_async_releases_stream_when_thread_fails(server, monkeypatch):
    rnd = make_async_round(server.port)
    eng = sim.Engine(seed=17)
    eng.new_request(0, prompt_len=300)
    eng.new_request(2, prompt_len=40)
    for _ in range(5):
        run_step(rnd, eng, eng.step())
    calls = {"n": 0}
    real_step = rnd.client.step

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected proxy-thread failure")
        return real_step(*a, **k)

    monkeypatch.setattr(rnd.client, "step", boom)
    t0 = time.monotonic()
    res, _ = run_step(rnd, eng, eng.step())
    assert time.monotonic() - t0 < 2.0                     # never left waiting
    assert res.read_status() == STATUS_ROUND_FAILED
    # the ring had this round's rows. Async resync rows come from windows the engine side stages
    # on request: the first round after the failure asks for them (one more failed round), then
    # every round is exact again
    res, _ = run_step(rnd, eng, eng.step())
    assert res.read_status() == STATUS_ROUND_FAILED
    assert rnd.stats.snapshot()["counters"]["fallback_resync_pending"] == 1
    for _ in range(10):
        res, _ = run_step(rnd, eng, eng.step())
        assert res.read_status() == STATUS_OK
    c = rnd.stats.snapshot()["counters"]
    assert c["fallback_internal_error"] == 1 and c["fallback_resync_pending"] == 1
    assert c["resync_rounds"] >= 1
    rnd.close()


def test_async_releases_stream_when_enqueue_fails(server, monkeypatch):
    rnd = make_async_round(server.port)
    eng = sim.Engine(seed=19)
    eng.new_request(1, prompt_len=200)
    for _ in range(4):
        run_step(rnd, eng, eng.step())
    real = rnd.gatherer.enqueue
    state = {"n": 0}

    def flaky(*a, **k):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("injected enqueue failure")
        return real(*a, **k)

    monkeypatch.setattr(rnd.gatherer, "enqueue", flaky)
    sd = eng.step()
    res = rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected, sd.last_sampled,
                  sd.next_prefill_tokens)
    assert res.read_status() == STATUS_ROUND_FAILED and not res.drafts.any()
    # rows of that round never reached the ring: the window is genuinely short until it
    # refills, so later rounds succeed (usable drafts) without the exactness check
    oks = 0
    for _ in range(40):
        sd = eng.step()
        res = rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected, sd.last_sampled,
                      sd.next_prefill_tokens)
        oks += res.read_status() == STATUS_OK
    assert oks == 40
    rnd.close()


def _torch_call(frame_arg) -> str | None:
    """Name of a torch call (a torch function or a Tensor method), else None."""
    mod = getattr(frame_arg, "__module__", None) or ""
    owner = getattr(frame_arg, "__self__", None)
    if mod.startswith("torch") or isinstance(owner, torch.Tensor) or owner is torch:
        return getattr(frame_arg, "__qualname__", repr(frame_arg))
    return None


def test_proxy_thread_makes_no_torch_call(server, monkeypatch):
    """The proxy thread must never call into torch (on GPU every such call can be a CUDA call that
    deadlocks against a module load while the engine stream is parked; streamsync.py). Profiles
    the real thread through ok rounds, a failed round, resyncs from staged windows, a worker restart
    and a timeout."""
    seen: list[str] = []
    name = "dsv41-remote-dspark-round"

    def prof(frame, event, arg):
        if event == "c_call" and threading.current_thread().name == name:
            hit = _torch_call(arg)
            if hit:
                seen.append(f"{hit} from {frame.f_code.co_name}:{frame.f_lineno}")

    threading.setprofile(prof)
    try:
        rnd = make_async_round(server.port)
    finally:
        threading.setprofile(None)
    # setprofile applies to threads started after it: the round's thread was started inside
    assert rnd._thread.name == name
    eng = sim.Engine(seed=23)
    for s in range(sim.M):
        eng.new_request(s, prompt_len=300)
    rng = np.random.default_rng(23)
    real_step = rnd.client.step
    calls = {"n": 0}

    def once_failing(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected proxy-thread failure")
        return real_step(*a, **k)

    for step in range(140):
        churn(eng, step, rng)
        if not any(eng.slots):
            eng.new_request(0)
        if step == 30:
            monkeypatch.setattr(rnd.client, "step", once_failing)   # deterministic resync next
        if step == 70:
            server.core = WorkerCore(hello())      # worker lost its windows -> resync
        if step == 100:
            server.core.inject.delay_ms = 400        # past the timeout
        if step == 101:
            server.core.inject.delay_ms = 0
        run_step(rnd, eng, eng.step())
    c = rnd.stats.snapshot()["counters"]
    assert c.get("resync_rounds", 0) >= 1 and c.get("resync_windows_staged", 0) >= 1, c
    assert c.get("fallback_internal_error", 0) == 1
    rnd.close()
    assert seen == [], seen[:20]


def test_staged_resync_rows_equal_device_gather(server):
    """The CPU rebuild from staged windows gives byte-identical rows to the sync path's
    gather_from_ring on the same ring."""
    rnd = make_async_round(server.port)
    g = rnd.gatherer
    W = rnd.W
    gen = torch.Generator().manual_seed(5)
    g.ring.copy_(torch.randn(g.ring.shape, generator=gen).to(g.ring.dtype))
    for sl in range(rnd.M):
        rnd.resync_host[sl].copy_(g.ring[sl * W:(sl + 1) * W])
    spans = [(2, 250, 128), (0, 7, 30), (3, 1000, 1), (1, 120, 20)]
    total = sum(k for _, _, k in spans)
    g.msg_np[:] = 0xA5
    got = rnd._rows_from_staged(spans)
    # built in the proxy's private buffer: the shared STEP staging is not written (S11 P-1)
    assert (g.msg_np == 0xA5).all()
    assert got.shape == (total, g.row_bytes) and not np.shares_memory(got, g.msg_np)
    assert g.gather_from_ring(4, spans) == total
    assert np.array_equal(got.reshape(-1), g.rows_view(4, total).view(torch.uint8).numpy().reshape(-1))
    rnd.close()


def test_close_releases_a_parked_round(server):
    rnd = make_async_round(server.port)
    eng = sim.Engine(seed=29)
    eng.new_request(0, prompt_len=50)
    run_step(rnd, eng, eng.step())
    rnd.close()
    g, r, c = rnd.sync.values()
    assert ((r - rnd.rounds) & 0xFFFFFFFF) < (1 << 31) and not rnd._thread.is_alive()   # r at/after rounds
