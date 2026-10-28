"""DW1 item 2 (F-DWAIT option A1): the rank-0 proxy's own time per decode round, on Controller's CPU.

The production stat that says how long the engine stream stays parked on a round is ``proxy_parked_us_decode``
(gathered seen -> reply published); ``rtt_us_decode`` is the worker round trip inside it. Their difference is the
proxy's own work: metadata collect, ring bookkeeping, STEP entries and prefix packing, reply decode and validation,
the packed reply, stats, publish. NBW-W3's K arms (production, Spark CPU) put it at 0.10-0.15 ms per B1 round.

This runs the production proxy mode (VLLM_DSPARK_REMOTE_ASYNC_PROXY=process) against the reference mock worker in
its own process (TCP loopback, no delay) with the offline engine simulator (tests/sim.py, tiny rows) and reports the
same two stats for decode rounds of 1-4 requests. ``--breakdown`` instead runs the thread proxy (same RoundCore code,
in this process) with perf_counter accumulators on the RoundCore stages. Relative numbers only: Controller is an x86 EPYC
host, the Sparks are Arm.

Usage (offload/dspark/spark): python -m tools.dw1_proxy_bench OUT.json [rounds] [--breakdown]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPARK = os.path.dirname(HERE)
TESTS = os.path.join(SPARK, "tests")
for p in (SPARK, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import sim  # noqa: E402
from remote_dspark import roundcore, wire  # noqa: E402
from remote_dspark.round import RemoteRound  # noqa: E402
from remote_dspark.streamsync import CpuHostFlagSync  # noqa: E402
from test_async_process import WORKER_MAIN, make_proc_round, proxy_stats  # noqa: E402
from test_round import cfg_for, hello  # noqa: E402


def worker(delay_ms: float = 0.0) -> tuple[subprocess.Popen, int]:
    p = subprocess.Popen([sys.executable, "-c", WORKER_MAIN, TESTS, SPARK, str(delay_ms)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    return p, int(p.stdout.readline())


def drive(rnd, n_req: int, rounds: int, seed: int) -> None:
    """n_req requests prefilled, then `rounds` decode rounds of all n_req (requests are re-created when done)."""
    eng = sim.Engine(seed=seed)
    for s in range(n_req):
        eng.new_request(s, prompt_len=200)
    done = 0
    while done < rounds:
        for s in range(n_req):
            r = eng.slots[s]
            if r is None or (not r.prefilling and r.decode_left <= 0):
                eng.new_request(s, prompt_len=200)
            else:
                r.decode_left = max(r.decode_left, 2)
        sd = eng.step(schedule=list(range(n_req)))
        rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected, sd.last_sampled, sd.next_prefill_tokens)
        if not any(eng.slots[s].prefilling for s in range(n_req)):
            done += 1


def lat(stats: dict, key: str) -> dict:
    v = stats.get("latency", {}).get(key, {})
    return {k: v.get(k) for k in ("n", "mean_us", "p50_us", "p90_us")}


def bench_process(rounds: int) -> dict:
    out = {}
    for n_req in (1, 2, 3, 4):
        p, port = worker()
        tmp = tempfile.mkdtemp(prefix="dw1-proxy-")
        try:
            rnd = make_proc_round(port, _P(tmp))
            drive(rnd, n_req, 20, seed=n_req)                  # warm-up, then a fresh stats window
            before = proxy_stats(rnd)
            drive(rnd, n_req, rounds, seed=100 + n_req)
            after = proxy_stats(rnd)
            rnd.close()
        finally:
            p.stdin.close()
            p.wait(timeout=5)
        cell = {}
        for key in ("proxy_parked_us_decode", "rtt_us_decode"):
            b, a = lat(before, key), lat(after, key)
            n = (a["n"] or 0) - (b["n"] or 0)
            cell[key] = {"n": n, "mean_us": round(((a["mean_us"] or 0) * (a["n"] or 0)
                                                   - (b["mean_us"] or 0) * (b["n"] or 0)) / max(n, 1), 1)}
        cell["proxy_own_mean_us"] = round(cell["proxy_parked_us_decode"]["mean_us"] - cell["rtt_us_decode"]["mean_us"],
                                          1)
        out[f"{n_req}req"] = cell
        print(n_req, cell, flush=True)
    return out


class _P(str):
    """tmp_path stand-in (supports the / operator)."""

    def __truediv__(self, other: str) -> str:
        return os.path.join(self, other)


def bench_breakdown(rounds: int) -> dict:
    """Thread proxy in this process; perf_counter accumulators around the RoundCore stages."""
    acc: dict[str, list[float]] = {}

    def wrap(obj, name, tag):
        fn = getattr(obj, name)

        def run(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                acc.setdefault(tag, []).append((time.perf_counter() - t0) * 1e6)
        setattr(obj, name, run)

    out = {}
    for n_req in (1, 4):
        p, port = worker()
        tmp = tempfile.mkdtemp(prefix="dw1-proxy-")
        try:
            rnd = RemoteRound(cfg_for(port, VLLM_DSPARK_REMOTE_ASYNC="1",
                                      VLLM_DSPARK_REMOTE_STATS_FILE=os.path.join(tmp, "t.json")),
                              hello(), device=torch.device("cpu"), tp_rank=0, broadcast_fn=None, background=True,
                              stream_sync=CpuHostFlagSync(timeout_s=5))
            t_end = time.monotonic() + 5
            while not rnd.client.ready and time.monotonic() < t_end:
                time.sleep(0.005)
            drive(rnd, n_req, 20, seed=n_req)
            acc.clear()
            for obj, name, tag in ((rnd, "_after_gather", "after_gather_total"),
                                   (rnd.gatherer, "collect", "collect"),
                                   (rnd, "_apply_ring", "apply_ring"),
                                   (rnd.client, "step", "client_step_total"),
                                   (rnd, "_accept_reply", "accept_reply"),
                                   (rnd.sync, "publish", "publish")):
                wrap(obj, name, tag)
            drive(rnd, n_req, rounds, seed=100 + n_req)
            rnd.write_stats()
            st = json.load(open(os.path.join(tmp, "t.json")))
            rnd.close()
        finally:
            p.stdin.close()
            p.wait(timeout=5)
        cell = {k: {"n": len(v), "mean_us": round(float(np.mean(v)), 1), "p50_us": round(float(np.median(v)), 1)}
                for k, v in acc.items()}
        cell["rtt_us_decode"] = lat(st, "rtt_us_decode")
        cell["proxy_parked_us_decode"] = lat(st, "proxy_parked_us_decode")
        out[f"{n_req}req"] = cell
        print(n_req, json.dumps(cell), flush=True)
    return out


def main() -> None:
    out_path = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 400
    rep: dict = {"host": os.uname().nodename, "rounds": rounds, "wire": wire.VERSION if hasattr(wire, "VERSION") else None,
                 "roundcore": os.path.abspath(roundcore.__file__)}
    if "--breakdown" in sys.argv:
        rep["breakdown_thread"] = bench_breakdown(rounds)
    else:
        rep["process"] = bench_process(rounds)
    with open(out_path, "w") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()
