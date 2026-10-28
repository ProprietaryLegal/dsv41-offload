#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""S11 P-1 on a CUDA device: a process proxy preempted by the watchdog mid-round must not corrupt the next round's rows.

    CUDA_VISIBLE_DEVICES=GPU-<uuid> python3 gpu_late_proxy_probe.py --pkg <dir with remote_dspark> \\
        --tests <dir with sim.py> [--at after_gather|step|rows_from_staged] [--out result.json]

S11's Q1 schedule (tests/test_late_proxy.py) with the engine side on the GPU: the real registered shared region, the
driver D2H of the gather into it, the stream's gathered / wait / reply-copy words, the watchdog in this process, the
proxy in its own process with one injected stall (150 ms, watchdog 100 ms) at round S. Rounds 1..S: requests in slots
0-2; then one fresh request alone for 12 rounds (its gathered rows ARE its window), then all four for 12 rounds. Every
usable draft is compared bit-exactly with the fake drafter over the request's window. --pkg selects the code: the
pre-R3 package shows the defect (wrong drafts with status OK), the R3 package must show none.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

S = 6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--at", default="after_gather", choices=("after_gather", "step", "rows_from_staged"))
    # between one and two deadlines: the watchdog fails round S while the proxy is inside it, and the proxy is back
    # before round S+1's deadline, so it handles S+1 itself (a longer stall makes S+1 a skipped round too, whose rows are
    # never sent: the defect would not show)
    ap.add_argument("--stall-ms", type=int, default=150)
    ap.add_argument("--watchdog-ms", type=int, default=100)
    ap.add_argument("--allow-device", default="GPU-7744b44b")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not vis.startswith(args.allow_device) or "," in vis:
        raise SystemExit(f"refusing: CUDA_VISIBLE_DEVICES must be exactly {args.allow_device}-... (is {vis!r})")
    sys.path[:0] = [args.pkg, args.tests]
    import sim

    wire = importlib.import_module("remote_dspark.wire")
    cfgmod = importlib.import_module("remote_dspark.config")
    mw = importlib.import_module("remote_dspark.mock_worker")
    R = importlib.import_module("remote_dspark.round")
    ss = importlib.import_module("remote_dspark.streamsync")
    shm = importlib.import_module("remote_dspark.shmregion")
    pp = importlib.import_module("remote_dspark.proxy_proc")
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    ident = bytes(range(32))
    hello = wire.Hello(hidden=sim.H, row_bytes=sim.N_TAPS * sim.H * 2, drafter_identity=ident)
    srv = mw.TcpStubServer(mw.WorkerCore(hello), "127.0.0.1", 0).start()
    stats_file = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"r3-late-probe-{os.getpid()}.json")
    cfg = cfgmod.RemoteConfig.from_env({
        "VLLM_DSPARK_REMOTE": f"tcp://127.0.0.1:{srv.port}", "VLLM_DSPARK_REMOTE_IDENTITY": ident.hex(),
        "VLLM_DSPARK_REMOTE_POLICY": "capacity", "VLLM_DSPARK_REMOTE_TIMEOUT_MS": "150",
        "VLLM_DSPARK_REMOTE_BACKOFF_MS": "2", "VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS": "4",
        "VLLM_DSPARK_REMOTE_LOG_EVERY": "0", "VLLM_DSPARK_REMOTE_STATS_FILE": stats_file,
        "VLLM_DSPARK_REMOTE_ASYNC": "1", "VLLM_DSPARK_REMOTE_ASYNC_PROXY": "process",
        "VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS": str(args.watchdog_ms)})
    # the stall, injected into the proxy's interpreter (the same patch as tests/test_late_proxy.py)
    pp.BOOT = pp.BOOT.replace(
        "from remote_dspark.proxy_proc import main",
        "import os, time\n"
        "import remote_dspark.roundcore as _rc\n"
        "import remote_dspark.client as _cl\n"
        f"_W, _S, _D = {args.at!r}, {S}, {args.stall_ms / 1000}\n"
        "_cur = [0]\n"
        "_ag = _rc.RoundCore._after_gather\n"
        "def _after_gather(self, bv, g, res, step_id):\n"
        "    _cur[0] = step_id\n"
        "    if _W == 'after_gather' and step_id == _S:\n"
        "        time.sleep(_D)\n"
        "    return _ag(self, bv, g, res, step_id)\n"
        "_rc.RoundCore._after_gather = _after_gather\n"
        "_st = _cl.RemoteDraftClient.step\n"
        "def _step(self, *a, **k):\n"
        "    if _W == 'step' and k.get('step_id') == _S:\n"
        "        time.sleep(_D)\n"
        "    return _st(self, *a, **k)\n"
        "_cl.RemoteDraftClient.step = _step\n"
        "_rs = _rc.RoundCore._rows_from_staged\n"
        "def _rows(self, *a, **k):\n"
        "    if _W == 'rows_from_staged' and _cur[0] == _S:\n"
        "        time.sleep(_D)\n"
        "    return _rs(self, *a, **k)\n"
        "_rc.RoundCore._rows_from_staged = _rows\n"
        "from remote_dspark.proxy_proc import main")
    region = ss.RegisteredRegion(shm.SharedRegion.create(shm.RegionLayout.for_hello(hello)), dev)
    sync = ss.HostFlagSync(dev, region=region)
    ok, why = sync.self_test()
    if not ok:
        raise SystemExit(f"self-test failed: {why}")
    rnd = R.RemoteRound(cfg, hello, device=dev, tp_rank=0, broadcast_fn=None, background=True, stream_sync=sync)
    t_end = time.monotonic() + 5
    while time.monotonic() < t_end:
        rnd.write_stats()
        if json.load(open(stats_file))["counters"].get("hellos", 0) >= 1:
            break
        time.sleep(0.02)
    L = rnd.layout

    def run(eng, schedule):
        sd = eng.step(schedule=schedule)
        sd.batch.positions = sd.batch.positions.to(dev)
        sd.batch.idx_mapping = sd.batch.idx_mapping.to(dev)
        t0 = time.monotonic()
        res = rnd.run(sd.batch, [t.to(dev) for t in sd.taps], sd.num_sampled.to(dev), sd.num_rejected.to(dev),
                      sd.last_sampled.to(dev), sd.next_prefill_tokens.to(dev))
        snap = res.bcast.to("cpu").numpy()          # waits for the round (the stream's wait, then the reply copy)
        wall = time.monotonic() - t0
        n, N = sd.batch.num_reqs, sim.N
        okm = snap[L.o_ok:L.o_ok + n]
        ids = snap[L.o_ids:L.o_ids + n * N].reshape(n, N)
        conf = snap[L.o_conf:L.o_conf + n * N].view(np.float32).reshape(n, N)
        exact = wrong = 0
        for i, r in enumerate(sd.reqs):
            if not sd.need_draft[i] or not okm[i]:
                continue
            w_ids, w_conf = mw.fake_draft(eng.expected_window(r), r.anchor, r.computed, N, sim.VOCAB)
            if np.array_equal(ids[i], w_ids) and np.array_equal(conf[i].view(np.int32), w_conf.view(np.int32)):
                exact += 1
            else:
                wrong += 1
        return int(snap[0]), wall, exact, wrong

    eng = sim.Engine(seed=3)
    for s in range(3):
        eng.new_request(s, prompt_len=5)
        eng.slots[s].decode_left = 10_000
    pre = []
    for t in range(1, S):
        if args.at == "rows_from_staged" and t == S - 2:
            srv.core.inject.req_status_once[0] = wire.RS_INTERNAL      # round S becomes a resync round
        pre.append(run(eng, [0, 1, 2]))
    preempted = run(eng, [0, 1, 2])
    eng.new_request(3, prompt_len=5).decode_left = 10_000
    fresh = [run(eng, [3])[2:] for _ in range(12)]
    joint = [run(eng, [0, 1, 2, 3])[2:] for _ in range(12)]
    rnd.write_stats()
    snap = json.load(open(stats_file))
    rnd.close()
    srv.stop()
    c = snap["counters"]
    res = {"pkg": args.pkg, "at": args.at, "device": torch.cuda.get_device_name(dev),
           "pre_wrong": sum(x[3] for x in pre), "preempted_round": {"status": preempted[0],
                                                                 "wall_ms": round(preempted[1] * 1e3, 1)},
           "fresh_exact_wrong": fresh, "joint_exact_wrong": joint,
           "fresh_wrong": sum(w for _, w in fresh), "fresh_exact": sum(e for e, _ in fresh),
           "joint_wrong": sum(w for _, w in joint),
           "watchdog": snap.get("async_watchdog"), "watchdog_preempted": c.get("watchdog_preempted", 0),
           "fallback_preempted": c.get("fallback_preempted", 0), "resync_rounds": c.get("resync_rounds", 0)}
    res["ok"] = (res["pre_wrong"] == 0 and preempted[0] == 2 and res["watchdog_preempted"] >= 1
                 and res["fresh_wrong"] == 0 and res["fresh_exact"] == 12 and res["joint_wrong"] == 0)
    print("RESULT " + json.dumps(res), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
