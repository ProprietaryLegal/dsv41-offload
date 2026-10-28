#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The STEP's two-part send over lane O1's RDMA backend (R3, S11 P-1): O1Client.request_parts copies the private headers
and the staged rows straight into ofl's registered send slot (ofl_client_payload_buf) and posts it. CPU only, on x86host,
over an RDMA loopback on its own direct-link address (as lane E's server selftest, worker/run/e_pair_suite.sh step 3).

    python3 rdma_loopback_probe.py --pkg <dir with remote_dspark> --tests <dir with sim.py> --ofl-dir <libofl.so + ofl.py>
        [--bind 10.202.28.1] [--port 18637] [--rounds 120] [--out result.json]

The reference worker serves on ofl (RDMA + TCP backends) at bind:port; RemoteRound (device cpu) drafts over
o1rdma://bind:port?bind=bind, first synchronously (engine thread), then through the process proxy (CPU hand-off on a real
shared region); a 600 ms worker delay mid-run makes ofl time out and break the QP, then the client reconnects. Every
usable draft must equal the fake drafter over the request's exact window.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--ofl-dir", required=True)
    ap.add_argument("--bind", default="10.202.28.1")
    ap.add_argument("--port", type=int, default=18637)
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.bind not in ("10.202.28.1", "10.202.29.1"):
        raise SystemExit("refusing: listeners only on x86host's direct-link addresses (COMMON.md rule 9)")
    os.environ["VLLM_DSPARK_REMOTE_OFL_DIR"] = args.ofl_dir
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.path[:0] = [args.pkg, args.tests]
    import torch

    import sim
    from remote_dspark.config import RemoteConfig
    from remote_dspark.mock_worker import WorkerCore, fake_draft, serve_ofl
    from remote_dspark.round import STATUS_OK, RemoteRound
    from remote_dspark.shmregion import RegionLayout, SharedRegion
    from remote_dspark.streamsync import CpuHostFlagSync
    from test_round import hello

    core = WorkerCore(hello())
    srv = serve_ofl(core, args.bind, args.port, args.ofl_dir, rdma=True)
    stats_file = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"r3-rdma-probe-{os.getpid()}.json")
    env = {"VLLM_DSPARK_REMOTE": f"o1rdma://{args.bind}:{args.port}?bind={args.bind}",
           "VLLM_DSPARK_REMOTE_IDENTITY": hello().drafter_identity.hex(), "VLLM_DSPARK_REMOTE_POLICY": "capacity",
           "VLLM_DSPARK_REMOTE_TIMEOUT_MS": "300", "VLLM_DSPARK_REMOTE_BACKOFF_MS": "2",
           "VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS": "4", "VLLM_DSPARK_REMOTE_LOG_EVERY": "0",
           "VLLM_DSPARK_REMOTE_STATS_FILE": stats_file, "VLLM_DSPARK_REMOTE_OFL_DIR": args.ofl_dir}
    out = {"endpoint": env["VLLM_DSPARK_REMOTE"], "modes": {}}

    def drive(rnd, ready) -> dict:
        t_end = time.monotonic() + 10
        while not ready() and time.monotonic() < t_end:
            time.sleep(0.01)
        if not ready():
            raise SystemExit("ofl RDMA connection not ready")
        eng = sim.Engine(seed=21)
        eng.new_request(0, prompt_len=700)
        eng.new_request(1, prompt_len=60)
        exact = wrong = ok_rounds = 0
        for step in range(args.rounds):
            if step == args.rounds // 2:
                core.inject.delay_ms = 600              # > 300 ms: ofl times out and breaks the QP
            if step == args.rounds // 2 + 1:
                core.inject.delay_ms = 0
            sd = eng.step()
            res = rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected, sd.last_sampled,
                          sd.next_prefill_tokens)
            ok_rounds += res.read_status() == STATUS_OK
            ids, conf, okm = res.drafts.numpy(), res.conf.numpy(), res.ok_mask.numpy()
            for i, r in enumerate(sd.reqs):
                if sd.need_draft[i] and okm[i]:
                    w_ids, w_conf = fake_draft(eng.expected_window(r), r.anchor, r.computed, sim.N, sim.VOCAB)
                    if (ids[i] == w_ids).all() and (conf[i] == w_conf).all():
                        exact += 1
                    else:
                        wrong += 1
            if res.read_status() != STATUS_OK:
                time.sleep(0.01)
        return {"exact": exact, "wrong": wrong, "ok_rounds": ok_rounds, "rounds": args.rounds}

    cfg = RemoteConfig.from_env(env)
    rnd = RemoteRound(cfg, hello(), device=torch.device("cpu"), tp_rank=0, broadcast_fn=None, background=True)
    r = drive(rnd, lambda: rnd.client.ready)
    r["counters"] = {k: v for k, v in rnd.stats.snapshot()["counters"].items()
                     if k.startswith(("fallback_", "hellos", "disconnects", "rounds_"))}
    rnd.client.close()
    out["modes"]["sync"] = r

    env2 = dict(env, VLLM_DSPARK_REMOTE_ASYNC="1", VLLM_DSPARK_REMOTE_ASYNC_PROXY="process")
    cfg2 = RemoteConfig.from_env(env2)
    region = SharedRegion.create(RegionLayout.for_hello(hello()))
    rnd2 = RemoteRound(cfg2, hello(), device=torch.device("cpu"), tp_rank=0, broadcast_fn=None, background=True,
                       stream_sync=CpuHostFlagSync(timeout_s=5.0, region=region))

    def proxy_ready():
        rnd2.write_stats()
        return json.load(open(stats_file))["counters"].get("hellos", 0) >= 1

    r2 = drive(rnd2, proxy_ready)
    rnd2.write_stats()
    snap = json.load(open(stats_file))
    r2["counters"] = {k: v for k, v in snap["counters"].items()
                      if k.startswith(("fallback_", "hellos", "disconnects", "rounds_"))}
    r2["proxy_torch_loaded"] = snap.get("proxy_torch_loaded")
    rnd2.close()
    out["modes"]["process"] = r2
    srv.stop()
    out["ok"] = all(m["wrong"] == 0 and m["exact"] >= args.rounds // 2 and m["counters"].get("hellos", 0) >= 2
                    for m in out["modes"].values())
    print("RESULT " + json.dumps(out), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
