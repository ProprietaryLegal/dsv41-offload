# SPDX-License-Identifier: Apache-2.0
"""Async capacity mode with the out-of-process proxy (VLLM_DSPARK_REMOTE_ASYNC_PROXY=process), on CPU.

The engine side is RemoteRound with a CpuHostFlagSync on the shared region's page (or a GPU-like stand-in that does not
wait); the proxy is the real child process (proxy_proc.py) talking TCP to the reference worker. Covers: exact drafts
through worker faults and resyncs from staged windows, round-for-round equivalence with the thread proxy, the success
path not needing the engine's GIL (S9 P1 with the expectation the directive asks for: the proxy releases the round, no
watchdog), proxy death and stalls (fail closed, never a parked stream), enqueue failure, close, strict start."""

import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from remote_dspark import wire
from remote_dspark.hostflag import FAR_AHEAD, HostFlagCore
from remote_dspark.mock_worker import TcpStubServer, WorkerCore
from remote_dspark.round import STATUS_OK, STATUS_ROUND_FAILED, RemoteRound
from remote_dspark.shmregion import RING, RegionLayout, SharedRegion
from remote_dspark.streamsync import CpuHostFlagSync

import sim
from test_round import cfg_for, churn, hello, run_step

pytestmark = pytest.mark.skipif(shutil.which("cc") is None and shutil.which("gcc") is None,
                                reason="no C compiler for hostflag.c")
M32 = 0xFFFFFFFF


def at_or_after(a: int, b: int) -> bool:
    return ((a - b) & M32) < (1 << 31)


@pytest.fixture
def server():
    srv = TcpStubServer(WorkerCore(hello()), "127.0.0.1", 0).start()
    yield srv
    srv.stop()


class ProcGpuLikeSync(HostFlagCore):
    """test_async_hardening.GpuLikeSync on the shared region's page: the engine side modelled as a GPU (gathered is the
    stream's write, wait_release an enqueue that returns at once), with a wait that can HOLD the GIL (S9 P1)."""

    def __init__(self, region: SharedRegion):
        super().__init__("", spin_us=200, nap_us=20, page=region.page_addr)
        self.region = region
        self.shared = region
        self.gil_holding = ctypes.PyDLL(self.lib._name)
        self.gil_holding.dsv41_hf_wait_geq.restype = ctypes.c_int
        self.gil_holding.dsv41_hf_wait_geq.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int64,
                                                       ctypes.c_int64, ctypes.c_int64]

    def d2h(self, dst, src):
        dst.copy_(src)

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


def proxy_stats(rnd) -> dict:
    rnd.write_stats()
    with open(rnd.cfg.stats_file) as f:
        return json.load(f)


def _wait(pred, s=5.0):
    t = time.monotonic() + s
    while not pred() and time.monotonic() < t:
        time.sleep(0.005)
    return pred()


def make_proc_round(port, tmp_path, *, gpu_like=False, ring=RING, ready=True, timeout_s=5.0, **kw):
    region = SharedRegion.create(RegionLayout.for_hello(hello(), ring))
    sync = ProcGpuLikeSync(region) if gpu_like else CpuHostFlagSync(timeout_s=timeout_s, region=region)
    env = {"VLLM_DSPARK_REMOTE_ASYNC": "1", "VLLM_DSPARK_REMOTE_ASYNC_PROXY": "process",
           "VLLM_DSPARK_REMOTE_STATS_FILE": str(tmp_path / "remote.json")}
    env.update(kw)
    cfg = cfg_for(port, **env)
    rnd = RemoteRound(cfg, hello(), device=torch.device("cpu"), tp_rank=0, broadcast_fn=None,
                      background=not cfg.strict_start, stream_sync=sync)
    assert rnd.proxy is not None and rnd.client is None and rnd.async_mode
    if ready:
        assert _wait(lambda: proxy_stats(rnd)["counters"].get("hellos", 0) >= 1, 5.0)
    return rnd


def _submit(rnd, eng):
    sd = eng.step()
    return rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected, sd.last_sampled, sd.next_prefill_tokens)


# ------------------------------------------------------------------------------------------- shared region

def test_region_layout_and_descriptor_round_trip():
    L = RegionLayout.for_hello(hello(), 4)
    assert L.size % 4096 == 0 and L.off_msg % 4096 == 0 and L.off_resync % 4096 == 0
    assert L.off_resync - L.off_msg >= L.msg_capacity
    reg = SharedRegion.create(L)
    other = SharedRegion.attach(L, reg.fd)            # a second mapping of the same memory
    ids = ["req-a", "req-b-" + "x" * 300, "req-c"]
    for t in range(1, 11):                            # wraps the 4-slot ring twice
        reg.write_descriptor(t, 3, 17 + t, 0, np.array([2, 0, 3]), np.array([0, 5, 400]),
                             np.array([True, False, True]), frozenset({0, 3}) if t % 2 else frozenset(), ids)
        d = other.read_descriptor(t)
        assert (d.t, d.n, d.R, d.flags) == (t, 3, 17 + t, 0)
        assert d.slots.tolist() == [2, 0, 3] and d.k_window.tolist() == [0, 5, 400]
        assert d.need_draft.tolist() == [True, False, True]
        assert d.staged == (frozenset({0, 3}) if t % 2 else frozenset())
        assert len(set(d.owners)) == 3 and d.owners[0] != d.owners[2]
    with pytest.raises(RuntimeError, match="holds round 10"):
        other.read_descriptor(6)                      # slot 6 % 4 now holds round 10
    reg.msg_np[:4] = [1, 2, 3, 4]
    assert other.msg_np[:4].tolist() == [1, 2, 3, 4]


# ------------------------------------------------------------------------------------------- exactness

def test_process_exact_through_failures(server, tmp_path):
    rnd = make_proc_round(server.port, tmp_path)
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
    snap = proxy_stats(rnd)
    c = snap["counters"]
    assert c.get("fallback_timeout", 0) >= 1 and c.get("resync_rounds", 0) >= 2
    assert c.get("fallback_resync_pending", 0) >= 1 and c.get("resync_windows_staged", 0) >= 1
    assert exact_after > 150
    assert statuses[-15:] == [STATUS_OK] * 15
    assert snap["async"] is True and snap["async_proxy"]["mode"] == "process"
    assert snap["proxy_torch_loaded"] is False
    assert {"rank0_submit_us", "rank0_round_us", "rtt_us"} <= set(snap["latency"])
    rnd.close()
    assert rnd.proxy.exit_code() == 0


def _corrupt_vocab(rep, req):
    """S5R reply validation: a draft id outside the vocabulary in the first drafted entry."""
    for i, e in enumerate(req.entries):
        if e.flags & wire.RF_NEED_DRAFT:
            rep.draft_ids[i, 0] = sim.VOCAB
            return True
    return False


def _drive(rnd, seed, steps=160, fail_at=(50, 51, 90, 120), frame_at=(30,), corrupt_at=(110,), server=None):
    eng = sim.Engine(seed=seed)
    for s in range(sim.M):
        eng.new_request(s, prompt_len=400)
    rng = np.random.default_rng(seed)
    out = []
    for step in range(steps):
        churn(eng, step, rng)
        if not any(eng.slots):
            eng.new_request(0)
        if server is not None and step in fail_at:
            # deterministic worker-side failure of one live slot -> dirty -> NEW_GEN with a resync from staged windows
            live = [i for i, r in enumerate(eng.slots) if r is not None]
            server.core.inject.req_status_once[live[step % len(live)]] = wire.RS_INTERNAL
        if server is not None and step in frame_at:
            server.core.inject.frame_status_once = wire.ST_INTERNAL     # a failed STEP frame
        if server is not None and step in corrupt_at:
            server.core.inject.reply_hook_once = _corrupt_vocab         # refused reply -> reconnect + resync
        sd = eng.step()
        res = rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected, sd.last_sampled, sd.next_prefill_tokens)
        out.append((res.read_status(), res.ok_mask.numpy().copy(), res.drafts.numpy().copy(),
                    res.conf.numpy().copy()))
        if step in frame_at or step in corrupt_at:
            time.sleep(0.15)          # any reconnect finishes before the next round, in both modes alike
    return out


def test_process_matches_thread_round_for_round(tmp_path):
    """Same simulated engine, same worker events: the process proxy produces the thread proxy's reply in every round
    (status, ok mask, drafts, confidences) and the same round counters - a move of the reviewed code, not a rewrite."""
    runs = {}
    for mode in ("thread", "process"):
        srv = TcpStubServer(WorkerCore(hello()), "127.0.0.1", 0).start()
        try:
            if mode == "process":
                rnd = make_proc_round(srv.port, tmp_path)
            else:
                rnd = RemoteRound(cfg_for(srv.port, VLLM_DSPARK_REMOTE_ASYNC="1",
                                          VLLM_DSPARK_REMOTE_STATS_FILE=str(tmp_path / "thread.json")),
                                  hello(), device=torch.device("cpu"), tp_rank=0, broadcast_fn=None,
                                  background=True, stream_sync=CpuHostFlagSync(timeout_s=5))
                assert _wait(lambda: rnd.client.ready)
            out = _drive(rnd, seed=31, server=srv)
            if mode == "process":
                c = proxy_stats(rnd)["counters"]
            else:
                rnd.write_stats()
                c = json.load(open(tmp_path / "thread.json"))["counters"]
            rnd.close()
        finally:
            srv.stop()
        runs[mode] = (out, c)
    (a, ca), (b, cb) = runs["thread"], runs["process"]
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert x[0] == y[0], f"round {i + 1}: status {x[0]} vs {y[0]}"
        for j in (1, 2, 3):
            assert np.array_equal(x[j], y[j]), f"round {i + 1}: field {j} differs"
    keys = [k for k in set(ca) | set(cb) if k.startswith(("rounds_", "fallback_", "req_status_", "resync_",
                                                          "breaker_"))
            or k in ("rows_sent", "stale_generation_new_sessions", "hellos")]
    assert {k: ca.get(k, 0) for k in keys} == {k: cb.get(k, 0) for k in keys}
    assert ca.get("resync_rounds", 0) >= 1 and ca.get("rounds_ok", 0) > 100
    assert ca.get("fallback_bad_reply", 0) == 1 and ca.get("hellos", 0) >= 2


# ------------------------------------------------------------------------------------------- GIL independence

WORKER_MAIN = """
import sys, time
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from remote_dspark.mock_worker import TcpStubServer, WorkerCore
from test_round import hello
srv = TcpStubServer(WorkerCore(hello()), "127.0.0.1", 0).start()
srv.core.inject.delay_ms = float(sys.argv[3])
print(srv.port, flush=True)
sys.stdin.read()
srv.stop()
"""


@pytest.fixture
def worker_process():
    """The reference worker in a process of its own (a worker thread in the test process would need the GIL the test
    holds); yields a function delay_ms -> port."""
    procs = []

    def start(delay_ms: float) -> int:
        here = os.path.dirname(os.path.abspath(__file__))
        p = subprocess.Popen([sys.executable, "-c", WORKER_MAIN, here, os.path.dirname(here), str(delay_ms)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        procs.append(p)
        return int(p.stdout.readline())

    yield start
    for p in procs:
        p.stdin.close()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def test_process_proxy_releases_while_the_engine_holds_the_gil(worker_process, tmp_path):
    """S9 P1 with the directive's expectation: the engine thread waits for the parked stream inside native code that
    HOLDS the GIL. The thread proxy cannot run then and the watchdog fails the round
    (test_async_hardening.test_P1r_...[True]); the process proxy has its own interpreter, so it completes the round
    itself: released within about the worker's delay, a usable reply, no watchdog fire, no preempted round."""
    port = worker_process(40)
    rnd = make_proc_round(port, tmp_path, gpu_like=True, VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS="150")
    sync = rnd.sync
    eng = sim.Engine(seed=5)
    eng.new_request(0, prompt_len=200)
    for _ in range(5):
        _submit(rnd, eng)
        r, dt = sync.engine_waits_for_stream(rnd.rounds, 3.0, hold_gil=True)
        assert r == 0 and dt < 0.14, (r, dt, sync.values())      # well inside the 150 ms watchdog deadline
        assert sync.reply_status() != STATUS_ROUND_FAILED
    snap = proxy_stats(rnd)
    assert snap["async_watchdog"]["fires"] == 0
    assert snap["counters"].get("watchdog_preempted", 0) == 0 and snap["counters"]["rounds_ok"] == 5
    rnd.close()


def test_thread_proxy_needs_the_watchdog_when_the_engine_holds_the_gil(worker_process):
    """The contrast, same out-of-process worker: the thread proxy cannot publish while the engine thread holds the GIL,
    so every such round is failed by the watchdog at its deadline (the GB10 async runs' 31 fires)."""
    from test_async_hardening import GpuLikeSync

    port = worker_process(40)
    sync = GpuLikeSync()
    rnd = RemoteRound(cfg_for(port, VLLM_DSPARK_REMOTE_ASYNC="1", VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS="150"),
                      hello(), device=torch.device("cpu"), tp_rank=0, broadcast_fn=None, background=True,
                      stream_sync=sync)
    assert _wait(lambda: rnd.client.ready)
    eng = sim.Engine(seed=5)
    eng.new_request(0, prompt_len=200)
    for i in range(3):
        _submit(rnd, eng)
        r, dt = sync.engine_waits_for_stream(rnd.rounds, 3.0, hold_gil=True)
        assert r == 0 and dt < 1.0, (r, dt)
        if i == 0:
            assert dt >= 0.14, dt                     # released by the watchdog at its deadline
        # failed either way: the watchdog, or the client's own timeout (the proxy could not read the reply in
        # time) and the reconnect it causes
        assert sync.reply_status() == STATUS_ROUND_FAILED
    assert sync.watchdog_stats()["fires"] >= 1
    rnd.close()


# ------------------------------------------------------------------------------------------- failures

def test_process_proxy_killed_fails_closed(server, tmp_path):
    rnd = make_proc_round(server.port, tmp_path, timeout_s=3.0)
    eng = sim.Engine(seed=7)
    eng.new_request(0, prompt_len=200)
    for _ in range(4):
        assert _submit(rnd, eng).read_status() == STATUS_OK
    assert proxy_stats(rnd)["counters"]["rounds_ok"] == 4      # the proxy's last stats file before it dies
    os.kill(rnd.proxy.pid, signal.SIGKILL)
    assert _wait(lambda: rnd.proxy.exit_code() is not None)
    for _ in range(5):
        t0 = time.monotonic()
        res = _submit(rnd, eng)       # never waits for the dead proxy
        assert time.monotonic() - t0 < 0.5
        assert res.read_status() == STATUS_ROUND_FAILED and not res.drafts.any() and not res.conf.any()
    assert rnd._dead is not None and "exited with -9" in rnd._dead
    g, r, c = rnd.sync.values()
    assert at_or_after(r, rnd.rounds) and at_or_after(c, 4 + FAR_AHEAD - 1)
    rnd.write_stats()
    snap = json.load(open(rnd.cfg.stats_file))
    assert snap["counters"]["fallback_proxy_dead"] == 5 and snap["async_dead"] == rnd._dead
    assert snap["counters"]["rounds_ok"] == 4          # the proxy's counts survive, the engine's are added
    rnd.close()


def test_process_proxy_stalled_then_resumes(server, tmp_path):
    """SIGSTOP: the engine's watchdog fails each round within its deadline (never a parked stream); after SIGCONT the
    proxy finds those rounds taken, re-announces every window and is exact again."""
    rnd = make_proc_round(server.port, tmp_path, timeout_s=3.0, VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS="60")
    eng = sim.Engine(seed=9)
    eng.new_request(0, prompt_len=300)
    eng.new_request(1, prompt_len=50)
    for _ in range(5):
        run_step(rnd, eng, eng.step())
    os.kill(rnd.proxy.pid, signal.SIGSTOP)
    try:
        for _ in range(4):
            t0 = time.monotonic()
            res, _ = run_step(rnd, eng, eng.step())
            assert res.read_status() == STATUS_ROUND_FAILED and 0.05 <= time.monotonic() - t0 < 1.0
    finally:
        os.kill(rnd.proxy.pid, signal.SIGCONT)
    ok = 0
    for _ in range(30):
        res, checked = run_step(rnd, eng, eng.step())   # exactness checked on every usable draft
        ok += res.read_status() == STATUS_OK
    assert ok >= 25 and rnd._dead is None
    snap = proxy_stats(rnd)
    assert snap["async_watchdog"]["fires"] >= 4 and snap["counters"]["watchdog_preempted"] >= 1
    rnd.close()


def test_thread_proxy_late_rounds_skip_rewritten_staging(server, monkeypatch):
    """The same late-proxy case on the thread proxy: the proxy reaches rounds the watchdog already completed while
    the engine has gathered later rounds into the same staging. It must not read their rows (they belong to a later
    round): ring bookkeeping from the round's own metadata slot, the round's slots resend, every later draft exact."""
    rnd = RemoteRound(cfg_for(server.port, VLLM_DSPARK_REMOTE_ASYNC="1", VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS="100"),
                      hello(), device=torch.device("cpu"), tp_rank=0, broadcast_fn=None, background=True,
                      stream_sync=CpuHostFlagSync(timeout_s=3.0))
    assert _wait(lambda: rnd.client.ready)
    eng = sim.Engine(seed=11)
    eng.new_request(0, prompt_len=300)
    eng.new_request(2, prompt_len=60)
    for _ in range(5):
        run_step(rnd, eng, eng.step())
    real = rnd._wait_gathered
    late = {6, 7, 8}

    def slow(t):
        if t in late:
            time.sleep(0.35)          # the proxy arrives after the 100 ms watchdog completed round t
        return real(t)

    monkeypatch.setattr(rnd, "_wait_gathered", slow)
    statuses = []
    for _ in range(40):
        res, _ = run_step(rnd, eng, eng.step())     # exactness checked on every usable draft
        statuses.append(res.read_status())
    # the late rounds and the backlog behind them are completed by the watchdog (each a skipped round), then one
    # round asks for the windows, one resyncs, and every later round is exact (100 ms watchdog: a busy test host
    # must not fire it on ordinary rounds)
    assert statuses[:3] == [STATUS_ROUND_FAILED] * 3 and statuses[-15:] == [STATUS_OK] * 15, statuses
    c = rnd.stats.snapshot()["counters"]
    assert c["watchdog_skipped_rounds"] >= 3 and c["watchdog_preempted"] >= 3 and c["resync_rounds"] >= 1, c
    assert "watchdog_skipped_unknown_ring" not in c
    ev = rnd.stats.snapshot()["watchdog_events"]
    assert {e["round"] for e in ev} >= {6, 7, 8} and all(e["how"] == "skipped" for e in ev)
    rnd.close()


@pytest.mark.parametrize("proxy", ["thread", "process"])
def test_missing_consumer_ack_fails_closed(server, tmp_path, proxy):
    """The explicit consumer acknowledgment: an engine side that never acknowledges a reply copy (broken ordering)
    makes the proxy stop before it writes the page again; every later round is target-only, nothing waits."""

    def no_ack(self, value, stream=None):
        pass

    if proxy == "process":
        rnd = make_proc_round(server.port, tmp_path, timeout_s=2.0)
        rnd.sync.consumed = no_ack.__get__(rnd.sync)
    else:
        sync = CpuHostFlagSync(timeout_s=2.0)
        sync.consumed = no_ack.__get__(sync)
        rnd = RemoteRound(cfg_for(server.port, VLLM_DSPARK_REMOTE_ASYNC="1"), hello(), device=torch.device("cpu"),
                          tp_rank=0, broadcast_fn=None, background=True, stream_sync=sync)
        assert _wait(lambda: rnd.client.ready)
    eng = sim.Engine(seed=4)
    eng.new_request(0, prompt_len=100)
    assert _submit(rnd, eng).read_status() == STATUS_OK          # round 1 needs no earlier acknowledgment
    statuses = []
    for _ in range(4):
        t0 = time.monotonic()
        statuses.append(_submit(rnd, eng).read_status())
        assert time.monotonic() - t0 < 1.0
    assert statuses == [STATUS_ROUND_FAILED] * 4
    if proxy == "process":
        assert _wait(lambda: rnd.proxy.exit_code() is not None)
        _submit(rnd, eng)
    else:
        assert _wait(lambda: not rnd._thread.is_alive())
    assert rnd._dead is not None
    rnd.close()


def test_timer_slack_is_set_for_the_calling_thread():
    import threading

    from remote_dspark.hostflag import set_timer_slack

    out = {}

    def probe():
        out["ok"] = set_timer_slack(1000)
        with open(f"/proc/{threading.get_native_id()}/timerslack_ns") as f:
            out["ns"] = int(f.read())

    th = threading.Thread(target=probe)
    th.start()
    th.join()
    assert out == {"ok": True, "ns": 1000}


def test_process_proxy_stalled_past_the_ring_is_declared_lost(server, tmp_path):
    rnd = make_proc_round(server.port, tmp_path, ring=4, timeout_s=3.0, VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS="30")
    eng = sim.Engine(seed=10)
    eng.new_request(0, prompt_len=100)
    _submit(rnd, eng)
    os.kill(rnd.proxy.pid, signal.SIGSTOP)
    try:
        for _ in range(8):
            _submit(rnd, eng)
    finally:
        if rnd.proxy.exit_code() is None:
            os.kill(rnd.proxy.pid, signal.SIGCONT)
    assert rnd._dead is not None and "stalled" in rnd._dead
    assert _wait(lambda: rnd.proxy.exit_code() is not None)
    t0 = time.monotonic()
    assert _submit(rnd, eng).read_status() == STATUS_ROUND_FAILED and time.monotonic() - t0 < 0.5
    rnd.close()


def test_process_enqueue_failure_fails_one_round(server, tmp_path, monkeypatch):
    rnd = make_proc_round(server.port, tmp_path)
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
    res = _submit(rnd, eng)
    assert res.read_status() == STATUS_ROUND_FAILED and not res.drafts.any()
    oks = sum(_submit(rnd, eng).read_status() == STATUS_OK for _ in range(40))
    assert oks == 40
    assert proxy_stats(rnd)["counters"]["fallback_internal_error"] == 1
    rnd.close()


def test_process_close_stops_the_proxy_and_releases(server, tmp_path):
    rnd = make_proc_round(server.port, tmp_path, gpu_like=True, VLLM_DSPARK_REMOTE_TIMEOUT_MS="9000",
                          VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS="60000")
    eng = sim.Engine(seed=29)
    eng.new_request(0, prompt_len=50)
    server.core.inject.delay_ms = 3000
    for _ in range(3):
        _submit(rnd, eng)             # GPU-like: the engine runs ahead; round 1 is slow on the worker
    t0 = time.monotonic()
    rnd.close()                       # stop request; the proxy is inside round 1: killed after 5 s
    assert time.monotonic() - t0 < 8 and rnd.proxy.exit_code() is not None
    g, r, c = rnd.sync.values()
    assert at_or_after(r, 3 + FAR_AHEAD - 1) and rnd.sync.reply_status() == STATUS_ROUND_FAILED


def test_process_strict_start_failure_raises(tmp_path):
    with pytest.raises(RuntimeError, match="failed to start.*connect/HELLO"):
        make_proc_round(1, tmp_path, ready=False, VLLM_DSPARK_REMOTE_STRICT_START="1")


def test_process_strict_start_connects_first(server, tmp_path):
    rnd = make_proc_round(server.port, tmp_path, ready=False, VLLM_DSPARK_REMOTE_STRICT_START="1")
    assert proxy_stats(rnd)["counters"]["hellos"] == 1      # connected before RemoteRound returned
    eng = sim.Engine(seed=3)
    eng.new_request(0, prompt_len=80)
    run_step(rnd, eng, eng.step())
    rnd.close()
