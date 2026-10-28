#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rank-0 per-round overhead of the remote round, outside the worker (indicative only).

Runs RemoteRound on CPU tensors at the real shapes (3 taps x 5120 bf16, W = 128) against
the reference worker with a no-op drafter over the contract's TCP stub (or ofl with
--ofl-dir), and reports per request count: rank0_round_us (whole round), gather_us (CPU
torch here, the GPU on a Spark), rtt_us (transport + worker), worker_us, and the Python /
packing overhead = round - gather - rtt. Not a GB10 measurement.

  python3 tools/bench_round.py [--rounds 400] [--ofl-dir .build/ofl]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from remote_dspark import wire  # noqa: E402
from remote_dspark.config import RemoteConfig  # noqa: E402
from remote_dspark.gather import BatchView  # noqa: E402,F401
from remote_dspark.mock_worker import TcpStubServer, WorkerCore, serve_ofl  # noqa: E402
from remote_dspark.round import RemoteRound  # noqa: E402
from remote_dspark.stats import Stats  # noqa: E402

H, NT, W, N = 5120, 3, 128, 5
ID = bytes(range(32))


class RoundTimingStats(Stats):
    """Derive host overhead from nested timings of the SAME successful round.

    Separate percentile subtraction is not a percentile of the difference (even
    without Stats' logarithmic binning). This benchmark keeps exact residuals.
    """

    def __init__(self) -> None:
        super().__init__()
        self._parts: dict[str, float] = {}
        self.python_pack_us: list[float] = []

    def begin_round(self) -> None:
        self._parts.clear()

    def observe(self, key: str, us: float) -> None:
        super().observe(key, us)
        if key in ("gather_us", "rtt_us"):
            self._parts[key] = us
        elif key == "rank0_round_us":
            missing = {"gather_us", "rtt_us"} - self._parts.keys()
            if missing:
                raise RuntimeError(f"benchmark round lacks paired timing: {sorted(missing)}")
            self.python_pack_us.append(us - self._parts["gather_us"] - self._parts["rtt_us"])


def noop_draft(window, anchor_token, anchor_pos, num_spec, vocab):
    return np.zeros(num_spec, np.int32), np.full(num_spec, 0.5, np.float32)


class Batch:
    pass


def make_batch(n, rows_per_req, base_pos):
    b = Batch()
    q = np.full(n, rows_per_req, dtype=np.int64)
    b.num_reqs = n
    b.req_ids = [f"r{i}" for i in range(n)]
    b.idx_mapping_np = np.arange(n, dtype=np.int64)
    b.idx_mapping = torch.arange(n, dtype=torch.int32)
    b.query_start_loc_np = np.concatenate([[0], np.cumsum(q)])
    b.positions = torch.cat([torch.arange(base_pos[i], base_pos[i] + rows_per_req)
                             for i in range(n)])
    b.is_prefilling_np = np.zeros(n, bool)
    b.num_computed_prefill_tokens_np = np.zeros(n, np.int64)
    b.prefill_len_np = np.zeros(n, np.int64)
    b.num_draft_tokens_per_req = np.full(n, rows_per_req - 1, np.int32)
    b.num_tokens = int(q.sum())
    return b


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--ofl-dir", default="")
    args = ap.parse_args()
    expected = wire.Hello(drafter_identity=ID)
    core = WorkerCore(expected, draft_fn=noop_draft)
    port = free_port()
    if args.ofl_dir:
        srv = serve_ofl(core, "127.0.0.1", port, args.ofl_dir, rdma=False)
        ep = f"o1tcp://127.0.0.1:{port}?bind=127.0.0.1"
        os.environ["VLLM_DSPARK_REMOTE_OFL_DIR"] = args.ofl_dir
    else:
        srv = TcpStubServer(core, "127.0.0.1", port).start()
        ep = f"tcp://127.0.0.1:{port}"
    out = {"transport": ep.split(":")[0], "note": "Controller CPU, Python; not GB10"}
    for n in (1, 2, 4):
        cfg = RemoteConfig.from_env({"VLLM_DSPARK_REMOTE": ep,
                                     "VLLM_DSPARK_REMOTE_IDENTITY": ID.hex(),
                                     "VLLM_DSPARK_REMOTE_POLICY": "capacity",
                                     "VLLM_DSPARK_REMOTE_TIMEOUT_MS": "1000",
                                     "VLLM_DSPARK_REMOTE_LOG_EVERY": "0",
                                     "VLLM_DSPARK_REMOTE_STATS_FILE": ""})
        timing = RoundTimingStats()
        rnd = RemoteRound(cfg, expected, device=torch.device("cpu"), tp_rank=0,
                          broadcast_fn=None, background=False, stats=timing)
        assert rnd.client.connect_now()
        rows = 6 if n <= 2 else 5
        pos = [1000 * (i + 1) for i in range(n)]
        taps_all = [torch.randn(rows * n, H).to(torch.bfloat16) for _ in range(NT)]
        ls = torch.zeros(4, 1, dtype=torch.int64)
        npf = torch.zeros(1, 4, dtype=torch.int32)
        for r in range(args.rounds):
            timing.begin_round()
            b = make_batch(n, rows, pos)
            nrej = torch.randint(0, rows, (n,), dtype=torch.int32)
            nsamp = rows - nrej
            res = rnd.run(b, taps_all, nsamp, nrej, ls, npf)
            assert res.read_status() == 0, rnd.stats.snapshot()["counters"]
            pos = [p + int(rows - k) for p, k in zip(pos, nrej.tolist())]
        lat = rnd.stats.snapshot()["latency"]
        p50 = {k: v["p50_us"] for k, v in lat.items()}
        out[f"{n}req"] = {**p50, "python_pack_us_p50": round(
            statistics.median(timing.python_pack_us), 1),
            "python_pack_timing": "exact median of paired round - gather - rtt"}
        rnd.client.close()
    srv.stop()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
