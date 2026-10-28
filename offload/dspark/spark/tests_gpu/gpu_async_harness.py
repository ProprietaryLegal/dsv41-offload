#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU regression and soak for the async remote round (lane A2), on one CUDA device.

    CUDA_VISIBLE_DEVICES=GPU-<uuid> python3 gpu_async_harness.py --pkg <dir with remote_dspark> \\
        --tests <dir with sim.py> --mode regress|soak|selftest [--rounds N] [--out result.json]

--pkg selects the implementation, so the same harness runs the first async design (an export of 57c6474: proxy thread
with a side stream, CudaStreamSync) and the host-word design (HostFlagSync). The engine side is the offline simulator
(tests/sim.py) with its tensors on the GPU and the reference worker (mock_worker.TcpStubServer) on localhost.

regress: after each round is submitted (the engine stream parked on the round), the engine thread loads a module the
         process has never seen (cuModuleLoadData of a fresh PTX kernel), as Triton does at a kernel's first use. The
         first design deadlocks there (run it under `timeout -s KILL`); the host-word design must complete every round.
soak:    N rounds, pipelined (the engine thread runs up to --depth rounds ahead of verification), with fresh module loads
         while rounds are parked, injected worker delays past the timeout, dropped replies and worker restarts
         (resyncs). Every usable draft must equal the fake drafter over the exact window (bit-exact ids and
         confidences); failed requests must carry zeros.
gilwait: S9 P1 on the GPU. Every --gil-every-th round, right after submitting it, the engine thread waits for the
         parked stream inside a native call that HOLDS the GIL (cuStreamSynchronize through ctypes.PyDLL, as a pybind11
         extension launch without gil_scoped_release would). The proxy cannot run; the v2 native watchdog must fail and
         release that round within its deadline (zero drafts), and every other round must stay bit-exact.
selftest: the hand-off's own startup self-test only.

--proxy process (lane R2): the same modes with the out-of-process proxy (VLLM_DSPARK_REMOTE_ASYNC_PROXY=process): the
         hand-off page, gather staging and descriptor ring in one registered shared region, the round in a torch-free
         child process. gilwait then expects the opposite of the thread proxy: the child publishes while the engine
         thread holds the GIL, so the GIL-held rounds complete normally (no watchdog fire, bit-exact drafts). Use
         --worker-proc there: a worker thread in this process would itself need the GIL the engine thread holds.
--kill-proxy-every K (lane R3, process proxy): SIGKILL the proxy process every K rounds; the engine respawns it
         (S11 P-4: event-drained word reset, new session). Requests in flight across a respawn are then checked
         against the window gathered since (their first live round under the new proxy), new ones against their whole
         window; every usable draft must still be bit-exact.
GPU protocol (2026-10-23): run under `flock ~/dsv41-offload/locks/gpu-7744b44b.lock` with CUDA_VISIBLE_DEVICES set to
         the development card; --allow-device names the accepted UUID prefix.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import os
import sys
import threading
import time

import numpy as np
import torch


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------- fresh modules (PTX)

_PTX = """.version 7.0
.target sm_80
.address_size 64
.visible .entry {name}(.param .u64 p)
{{
  .reg .u64 %rd<3>;
  .reg .u32 %r<2>;
  ld.param.u64 %rd1, [p];
  cvta.to.global.u64 %rd2, %rd1;
  mov.u32 %r1, {val};
  st.global.u32 [%rd2], %r1;
  ret;
}}
"""


class ModuleLoader:
    def __init__(self):
        self.cu = ctypes.CDLL("libcuda.so.1")
        self.cu.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        self.cu.cuModuleLoadData.restype = ctypes.c_int
        self.cu.cuModuleUnload.argtypes = [ctypes.c_void_p]
        self.cu.cuModuleUnload.restype = ctypes.c_int
        self.n = 0
        self.loaded: list[ctypes.c_void_p] = []
        self.durations: list[float] = []

    def load_fresh(self) -> float:
        """Load a module this process has never loaded; returns the call's duration (s)."""
        self.n += 1
        ptx = _PTX.format(name=f"a2_fresh_{os.getpid()}_{self.n}", val=self.n).encode() + b"\0"
        mod = ctypes.c_void_p()
        t0 = time.perf_counter()
        r = self.cu.cuModuleLoadData(ctypes.byref(mod), ptx)
        dt = time.perf_counter() - t0
        if r != 0:
            raise RuntimeError(f"cuModuleLoadData failed: CUresult {r}")
        self.loaded.append(mod)
        self.durations.append(dt)
        return dt

    def unload_all(self) -> None:
        torch.cuda.synchronize()
        for m in self.loaded:
            r = self.cu.cuModuleUnload(m)
            if r != 0:
                raise RuntimeError(f"cuModuleUnload failed: CUresult {r}")
        self.loaded.clear()


# ------------------------------------------------------------------------------------------- setup

def setup(args):
    sys.path.insert(0, args.tests)
    sys.path.insert(0, args.pkg)
    import sim  # noqa: E402

    wire = importlib.import_module("remote_dspark.wire")
    cfgmod = importlib.import_module("remote_dspark.config")
    mw = importlib.import_module("remote_dspark.mock_worker")
    rmod = importlib.import_module("remote_dspark.round")
    ss = importlib.import_module("remote_dspark.streamsync")
    ident = bytes(range(32))
    hello = lambda: wire.Hello(hidden=sim.H, row_bytes=sim.N_TAPS * sim.H * 2, drafter_identity=ident)  # noqa: E731
    env = {"VLLM_DSPARK_REMOTE": "tcp://127.0.0.1:{port}", "VLLM_DSPARK_REMOTE_IDENTITY": ident.hex(),
           "VLLM_DSPARK_REMOTE_POLICY": "capacity", "VLLM_DSPARK_REMOTE_TIMEOUT_MS": "150",
           "VLLM_DSPARK_REMOTE_BACKOFF_MS": "2", "VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS": "4",
           "VLLM_DSPARK_REMOTE_LOG_EVERY": "0", "VLLM_DSPARK_REMOTE_STATS_FILE": "",
           "VLLM_DSPARK_REMOTE_ASYNC": "1"}
    if args.watchdog_ms:
        env["VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS"] = str(args.watchdog_ms)
    if args.proxy == "process":
        env["VLLM_DSPARK_REMOTE_ASYNC_PROXY"] = "process"
        env["VLLM_DSPARK_REMOTE_STATS_FILE"] = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), f"r2-gpu-harness-{os.getpid()}.json")
    if args.spin_us is not None:
        env["VLLM_DSPARK_REMOTE_ASYNC_SPIN_US"] = str(args.spin_us)
    if args.nap_us is not None:
        env["VLLM_DSPARK_REMOTE_ASYNC_NAP_US"] = str(args.nap_us)
    return sim, wire, cfgmod, mw, rmod, ss, hello, env


def make_sync(ss, dev, args, hello=None):
    if args.proxy == "process":
        shm = importlib.import_module("remote_dspark.shmregion")
        region = ss.RegisteredRegion(shm.SharedRegion.create(shm.RegionLayout.for_hello(hello())), dev)
        sync = ss.HostFlagSync(dev, lib_path=args.hostflag_lib, spin_us=args.spin_us if args.spin_us is not None
                               else 1000, nap_us=args.nap_us or 0, region=region)
        impl = "host_flags + process proxy (shared region)"
    elif hasattr(ss, "HostFlagSync"):
        sync = ss.HostFlagSync(dev, lib_path=args.hostflag_lib)
        impl = "host_flags"
    else:
        sync = ss.CudaStreamSync(dev)
        impl = "side_stream (first design)"
    ok, why = sync.self_test()
    log(f"hand-off {impl}: self-test {ok} ({why})")
    if not ok:
        raise SystemExit(f"self-test failed: {why}")
    return sync, impl


def to_dev(sd, dev):
    """The simulator's step on the GPU: pinned + non_blocking copies (never a pageable copy, which would block the CPU
    on the parked stream and hide run-ahead)."""
    def d(t):
        return t.pin_memory().to(dev, non_blocking=True)
    b = sd.batch
    b.positions = d(b.positions)
    b.idx_mapping = d(b.idx_mapping)
    sd.taps = [d(t) for t in sd.taps]
    sd.num_sampled = d(sd.num_sampled)
    sd.num_rejected = d(sd.num_rejected)
    sd.last_sampled = d(sd.last_sampled)
    sd.next_prefill_tokens = d(sd.next_prefill_tokens)
    return sd


def churn(sim, eng, rng):
    for slot, r in enumerate(eng.slots):
        if r is None:
            if rng.random() < 0.5:
                eng.new_request(slot, hit=int(rng.choice([0, 0, 300, 2000])))
            continue
        if not r.prefilling and r.decode_left <= 0:
            if rng.random() < 0.3:
                eng.readd(slot)
                r.decode_left = int(rng.integers(2, 7))
            else:
                eng.slots[slot] = None


def expected(sim, mw, eng, sd, since=None):
    """Per batch row: None (no draft needed) or (ids, conf) the worker must return. since: req_id -> p0 for requests in
    flight across a proxy respawn: their window holds only rows gathered under the new proxy, i.e. from the lowest first
    position of their rounds since (a re-add replays from below and extends it down)."""
    out = []
    for i, r in enumerate(sd.reqs):
        if not sd.need_draft[i]:
            out.append(None)
            continue
        win = eng.expected_window(r)
        p = (since or {}).get(r.req_id)
        if p is not None:
            win = [(pos, b) for pos, b in win if pos >= p]
        out.append(mw.fake_draft(win, r.anchor, r.computed, sim.N, sim.VOCAB))
    return out


def verify(sim, layout, snap: np.ndarray, n: int, exp) -> tuple[int, int, int]:
    """(checked exact, failed-with-zeros, status) for one round's broadcast snapshot."""
    N = sim.N
    status = int(snap[0])
    okm = snap[layout.o_ok:layout.o_ok + n]
    ids = snap[layout.o_ids:layout.o_ids + n * N].reshape(n, N)
    conf = snap[layout.o_conf:layout.o_conf + n * N].view(np.float32).reshape(n, N)
    checked = zeros = 0
    for i, e in enumerate(exp):
        if e is None or not okm[i]:
            if ids[i].any() or conf[i].any():
                raise AssertionError(f"row {i}: no usable draft but non-zero ids/conf")
            zeros += e is not None
            continue
        want_ids, want_conf = e
        if not np.array_equal(ids[i], want_ids):
            raise AssertionError(f"row {i}: drafts {ids[i].tolist()} != {want_ids.tolist()}")
        if not np.array_equal(conf[i].view(np.int32), want_conf.view(np.int32)):
            raise AssertionError(f"row {i}: confidences differ")
        checked += 1
    if status == 2 and okm.any():
        raise AssertionError("round failed but ok_mask set")
    return checked, zeros, status


def stats_of(rnd) -> dict:
    """The round's stats: the shared object (thread proxy) or the proxy process's file (process proxy)."""
    if getattr(rnd, "proxy", None) is None:
        return rnd.stats.snapshot()
    rnd.write_stats()
    with open(rnd.cfg.stats_file) as f:
        return json.load(f)


def wait_ready(rnd, s=5.0):
    t = time.monotonic() + s
    if getattr(rnd, "proxy", None) is not None:
        while stats_of(rnd)["counters"].get("hellos", 0) < 1:
            if time.monotonic() > t:
                raise SystemExit("worker connection not ready (process proxy)")
            time.sleep(0.02)
        return
    while not rnd.client.ready and time.monotonic() < t:
        time.sleep(0.002)
    if not rnd.client.ready:
        raise SystemExit("worker connection not ready")


WORKER_MAIN = """
import sys, time
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import sim
from remote_dspark import wire
from remote_dspark.mock_worker import TcpStubServer, WorkerCore
h = wire.Hello(hidden=sim.H, row_bytes=sim.N_TAPS * sim.H * 2, drafter_identity=bytes(range(32)))
srv = TcpStubServer(WorkerCore(h), "127.0.0.1", 0).start()
srv.core.inject.delay_ms = float(sys.argv[3])
print(srv.port, flush=True)
sys.stdin.read()
srv.stop()
"""


class WorkerProc:
    """The reference worker in a process of its own (no GIL shared with the engine thread)."""

    def __init__(self, args, delay_ms: float):
        import subprocess

        self.p = subprocess.Popen([sys.executable, "-c", WORKER_MAIN, args.tests, args.pkg, str(delay_ms)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.port = int(self.p.stdout.readline())

    def stop(self):
        self.p.stdin.close()
        self.p.wait(timeout=10)


# ------------------------------------------------------------------------------------------- modes

def run(args):
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    sim, wire, cfgmod, mw, rmod, ss, hello, env = setup(args)
    sync, impl = make_sync(ss, dev, args, hello)
    result = {"impl": impl, "pkg": args.pkg, "mode": args.mode, "proxy": args.proxy,
              "device": torch.cuda.get_device_name(dev)}
    if args.mode == "selftest":
        result["ok"] = True
        return result
    wproc = None
    if args.worker_proc:
        if args.mode == "soak":
            raise SystemExit("--worker-proc: soak injects faults into an in-process worker")
        wproc = WorkerProc(args, args.regress_delay_ms if args.mode in ("regress", "gilwait") else 0)
        srv = None
        port = wproc.port
    else:
        srv = mw.TcpStubServer(mw.WorkerCore(hello()), "127.0.0.1", 0).start()
        port = srv.port
    e = dict(env)
    e["VLLM_DSPARK_REMOTE"] = env["VLLM_DSPARK_REMOTE"].format(port=port)
    cfg = cfgmod.RemoteConfig.from_env(e)
    rnd = rmod.RemoteRound(cfg, hello(), device=dev, tp_rank=0, broadcast_fn=None, background=True,
                           stream_sync=sync)
    wait_ready(rnd)
    loader = ModuleLoader()
    eng = sim.Engine(seed=args.seed)
    for sl in range(sim.M):
        eng.new_request(sl, prompt_len=400)
    rng = np.random.default_rng(args.seed)
    L = rnd.layout
    depth = 1 if args.mode in ("regress", "gilwait") else args.depth
    snaps = [torch.zeros(L.size, dtype=torch.int32, device=dev) for _ in range(depth)]
    snaps_host = [torch.zeros(L.size, dtype=torch.int32, pin_memory=True) for _ in range(depth)]
    pending: list[tuple[int, int, list, int]] = []   # (round, n, expected, snapshot slot)
    totals = {"checked": 0, "zeros": 0, "ok": 0, "some_failed": 0, "failed": 0, "loads_parked": 0}
    gil_dts: list[float] = []
    gil_rounds: set[int] = set()
    gil_status: list[int] = []
    if args.mode == "gilwait":
        cu_gil = ctypes.PyDLL("libcuda.so.1")      # PyDLL: the GIL stays held for the whole call
        cu_gil.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
        cu_gil.cuStreamSynchronize.restype = ctypes.c_int
    t_start = time.monotonic()
    last_progress = t_start
    events: list[str] = []
    since: dict[str, int] = {}                  # req_id -> lowest first position of its rounds under the current proxy
    gen_seen: dict[str, int] = {}
    if args.kill_proxy_every:
        if args.proxy != "process":
            raise SystemExit("--kill-proxy-every needs --proxy process")
        rmod.RESPAWN_DELAY_S = args.respawn_delay_s

    def drain():
        torch.cuda.synchronize(dev)
        for (t, n, exp, k) in pending:
            c, z, st = verify(sim, L, snaps_host[k].numpy(), n, exp)
            if t in gil_rounds:
                gil_status.append(st)
            totals["checked"] += c
            totals["zeros"] += z
            totals[{0: "ok", 1: "some_failed", 2: "failed"}[st]] += 1
        pending.clear()

    for step in range(args.rounds):
        churn(sim, eng, rng)
        if not any(eng.slots):
            eng.new_request(0)
        if args.mode == "soak" and not args.no_faults:
            if step % 400 == 200:
                srv.core.inject.delay_ms = 400            # past the 150 ms timeout: a failed round
                events.append(f"{step}: delay 400 ms")
            elif step % 400 == 201:
                srv.core.inject.delay_ms = 0
            if step % 1000 == 500:
                srv.core = mw.WorkerCore(hello())          # worker restart: its windows are gone -> resync
                events.append(f"{step}: worker restart")
            if step % 700 == 350:
                srv.core.inject.drop_next_step = True
                events.append(f"{step}: drop reply")
        elif args.mode == "regress" and srv is not None:
            srv.core.inject.delay_ms = args.regress_delay_ms   # the proxy is mid-round when the module loads
        if (args.kill_proxy_every and step % args.kill_proxy_every == args.kill_proxy_every // 2
                and rnd._dead is None):
            os.kill(rnd.proxy.pid, 9)
            events.append(f"{step}: SIGKILL proxy {rnd.proxy.pid}")
        before = {r.req_id: r.computed for r in eng.slots if r is not None}
        sd = eng.step()
        sd = to_dev(sd, dev)
        k = step % depth
        res = rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected, sd.last_sampled, sd.next_prefill_tokens)
        if args.kill_proxy_every and rnd._dead is None and rnd._generation:
            for r in sd.reqs:               # under a respawned proxy a request's window starts at its rows gathered since
                if gen_seen.get(r.req_id, 0) != rnd._generation:
                    gen_seen[r.req_id] = rnd._generation
                    since[r.req_id] = before[r.req_id]
                else:
                    since[r.req_id] = min(since[r.req_id], before[r.req_id])
        exp = expected(sim, mw, eng, sd, since)
        snaps[k].copy_(res.bcast)                   # stream-ordered after the reply copy
        snaps_host[k].copy_(snaps[k], non_blocking=True)
        pending.append((step, sd.batch.num_reqs, exp, k))
        if args.mode == "gilwait" and step % args.gil_every == args.gil_every // 2:
            t0 = time.perf_counter()
            r = cu_gil.cuStreamSynchronize(ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream))
            gil_dts.append(time.perf_counter() - t0)
            gil_rounds.add(step)
            if r != 0:
                raise RuntimeError(f"cuStreamSynchronize failed: CUresult {r}")
        load_now = args.mode != "gilwait" and (
            (args.mode == "regress") or (args.load_every and step % args.load_every == args.load_every // 2))
        if load_now:
            if args.mode == "regress":
                log(f"round {step + 1}: submitted, stream parked; loading a fresh module")
            dt = loader.load_fresh()
            totals["loads_parked"] += 1
            if args.mode == "regress":
                log(f"round {step + 1}: module loaded in {dt * 1e3:.1f} ms")
        if len(pending) == depth:
            drain()
        if args.mode == "soak" and step % 2000 == 1999:
            drain()
            loader.unload_all()
        now = time.monotonic()
        if now - last_progress > 10:
            last_progress = now
            log(f"step {step + 1}/{args.rounds}: {totals}")
    drain()
    if srv is not None:
        srv.core.inject.delay_ms = 0
    snap = stats_of(rnd)
    rnd.close()
    if srv is not None:
        srv.stop()
    if wproc is not None:
        wproc.stop()
    loader.unload_all()
    elapsed = time.monotonic() - t_start
    lat = snap.get("latency", {})
    durs = sorted(loader.durations)
    result.update({
        "ok": True, "rounds": args.rounds, "elapsed_s": round(elapsed, 2), "totals": totals,
        "counters": snap.get("counters", {}), "events": events[:50],
        "module_loads": {"n": len(durs), "p50_ms": round(durs[len(durs) // 2] * 1e3, 3) if durs else None,
                         "max_ms": round(durs[-1] * 1e3, 3) if durs else None},
        "latency_us": {k: lat.get(k) for k in ("rank0_round_us", "rank0_submit_us", "gather_us", "rtt_us")},
        "words": list(sync.values()) if hasattr(sync, "values") else None,
        "watchdog": sync.watchdog_stats() if hasattr(sync, "watchdog_stats") else None,
        "watchdog_events": snap.get("watchdog_events"),
        "proxy_torch_loaded": snap.get("proxy_torch_loaded"),
        "latency_us_all": {k: v for k, v in lat.items() if k.startswith(("rank0_", "proxy_parked", "rtt_us"))},
        "async_respawns": snap.get("async_respawns"), "async_respawn": snap.get("async_respawn"),
        "async_dead": snap.get("async_dead"), "async_proxy_losses": snap.get("async_proxy_losses"),
        "async_stall_detection": snap.get("async_stall_detection"),
    })
    if args.mode == "gilwait":
        d = sorted(gil_dts)
        result["gil_waits"] = {"n": len(d), "min_ms": round(d[0] * 1e3, 1), "p50_ms": round(d[len(d) // 2] * 1e3, 1),
                               "max_ms": round(d[-1] * 1e3, 1), "statuses": sorted(set(gil_status))}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--mode", choices=("regress", "soak", "gilwait", "selftest"), required=True)
    ap.add_argument("--gil-every", type=int, default=5)
    ap.add_argument("--watchdog-ms", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--load-every", type=int, default=37)
    ap.add_argument("--regress-delay-ms", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hostflag-lib", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--proxy", choices=("thread", "process"), default="thread")
    ap.add_argument("--worker-proc", action="store_true", help="reference worker in its own process")
    ap.add_argument("--spin-us", type=int, default=None)
    ap.add_argument("--kill-proxy-every", type=int, default=0, help="process proxy: SIGKILL it every K rounds (R3)")
    ap.add_argument("--no-faults", action="store_true",
                    help="soak without worker faults (steady-state round overhead; module loads still run)")
    ap.add_argument("--respawn-delay-s", type=float, default=0.05, help="with --kill-proxy-every: RESPAWN_DELAY_S")
    ap.add_argument("--nap-us", type=int, default=None)
    ap.add_argument("--allow-device", default="GPU-7744b44b",
                    help="accepted CUDA_VISIBLE_DEVICES UUID prefix (one device)")
    ap.add_argument("--gate-file", default="",
                    help="optional: GPU work only while this file exists; checked every 0.5 s")
    args = ap.parse_args()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not vis.startswith(args.allow_device) or "," in vis:
        raise SystemExit(f"refusing: CUDA_VISIBLE_DEVICES must be exactly {args.allow_device}-... (is {vis!r})")
    if args.gate_file:
        if not os.path.exists(args.gate_file):
            raise SystemExit(f"refusing: {args.gate_file} does not exist (no GPU slot from the coordinator)")

        def gate_watch():
            while True:
                time.sleep(0.5)
                if not os.path.exists(args.gate_file):
                    log(f"GATE CLOSED: {args.gate_file} disappeared; exiting now")
                    os._exit(3)          # also works while the main thread is stuck in a driver call

        threading.Thread(target=gate_watch, daemon=True, name="gate-watch").start()
    log(f"start {args.mode} pkg={args.pkg} rounds={args.rounds}")
    res = run(args)
    log("RESULT " + json.dumps(res))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
