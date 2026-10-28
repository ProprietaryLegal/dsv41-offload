"""DW1 identity harness: a candidate STEP path must give byte-identical draft ids and confidences to GpuBackend.

One model on the granted card (GPU-bc62df2a); the reference backend (GpuBackend: flush + draft) owns model slots 0-3,
the candidate (LeanGpuBackend: flush_draft) owns slots 4-7, and both receive the same rows, anchors, positions and
windows in the same order. Every drafted request's ids (int32) and confidences (float32) are compared as bytes.

Sets:
  synthetic  the self-test's synthetic windows (tools/server_selftest.taps_for): 4 requests, a 128-row first window
             each, then decode rounds of 1-6 rows per request at batch 1-4 and n_out 4/5 (seeded)
  v1         the V1 replay set (A-capture-20261022 on x86host): each captured round's valid rows (real taps where the
             round carries them, decode rounds; seeded synthetic taps for prefill rows, which carry only main_x),
             the captured anchors and positions, windows max(0, P - 127) .. P as parity_capture builds them
Usage (from offload/dspark): python -m worker.tools.dw1_identity ROOT OUT.json [capture_dir]
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch

from ..lean import LeanGpuBackend
from ..server import GpuBackend
from .dw1_stages import build_one_card
from .parity_capture import Slots
from .server_selftest import Shadow1, taps_for


class Lo:
    """WindowRange stand-in: the window's low bound for an anchor position (max(0, apos - W))."""

    def __init__(self, W: int) -> None:
        self.W = W

    def read_lo(self, apos: int) -> int:
        return max(0, apos - self.W)


class Pair:
    def __init__(self, m, exp) -> None:
        self.ref = GpuBackend(m, exp.max_rows_per_step, exp.row_bytes, graphs=True)
        self.cand = LeanGpuBackend(m, exp.max_rows_per_step, exp.row_bytes, exp.num_spec, exp.max_slots)
        self.N, self.W = exp.num_spec, exp.window
        self.rng = Lo(exp.window)
        self.counts = {"steps": 0, "requests": 0, "ids_equal": 0, "conf_equal": 0, "lean_steps": 0}
        self.first_diff: list[dict] = []

    def reset(self, slot: int) -> None:
        self.ref.reset(slot)
        self.cand.reset(slot + 4)

    def step(self, staged: list[tuple[int, int, np.ndarray]], reqs: list[tuple[int, int, int, int]], tag: str
             ) -> None:
        """staged: (slot 0-3, first_pos, rows uint8 [n, row_bytes]); reqs: (slot 0-3, anchor tok, anchor pos, k)."""
        for slot, p0, rows in staged:
            self.ref.stage(slot, p0, rows)
            self.cand.stage(slot + 4, p0, rows)
        self.ref.flush()
        a_ids, a_conf, _ = self.ref.draft([(s, t, p, self.rng, k) for s, t, p, k in reqs], self.N, False)
        lean_before = self.cand.n_lean
        b_ids, b_conf, _ = self.cand.flush_draft([(s + 4, t, p, self.rng, k) for s, t, p, k in reqs], self.N, False)
        self.counts["lean_steps"] += self.cand.n_lean - lean_before
        self.counts["steps"] += 1
        for j, (_s, _t, _p, k) in enumerate(reqs):
            self.counts["requests"] += 1
            ie = a_ids[j, :k].tobytes() == b_ids[j, :k].tobytes()
            ce = a_conf[j, :k].tobytes() == b_conf[j, :k].tobytes()
            self.counts["ids_equal"] += int(ie)
            self.counts["conf_equal"] += int(ce)
            if not (ie and ce) and len(self.first_diff) < 20:
                self.first_diff.append({"set": tag, "step": self.counts["steps"], "req": j, "k": k,
                                        "ids_equal": ie, "conf_equal": ce})


def synthetic(pair: Pair, m, rounds: int, seed: int) -> None:
    sh = Shadow1(m)
    gen = torch.Generator(device=m.dev).manual_seed(seed)
    rnd = random.Random(seed)
    W, N = pair.W, pair.N
    L = W + 8 * (rounds + 8)
    toks = torch.randint(0, 120000, (L,), generator=gen, device=m.dev)
    taps = taps_for(sh, toks, gen)
    apos = [W + 1] * 4
    for s in range(4):
        pair.reset(s)
        pair.step([(s, 1, taps[1:W + 1])], [(s, int(toks[W + 1]), W + 1, N)], "synthetic")
    for _ in range(rounds):
        B = rnd.randint(1, 4)
        k = N if B <= 2 else 4
        staged, reqs = [], []
        for s in rnd.sample(range(4), B):
            acc = rnd.randint(1, 6)
            staged.append((s, apos[s], taps[apos[s]:apos[s] + acc]))
            apos[s] += acc
            reqs.append((s, int(toks[apos[s]]), apos[s], k))
        pair.step(staged, reqs, "synthetic")


def v1(pair: Pair, cap: str) -> dict:
    files = sorted(glob.glob(os.path.join(cap, "round*.pt")))
    if not files:
        raise FileNotFoundError(f"no round*.pt under {cap}")
    slots = Slots(0, 4)
    W, N = pair.W, pair.N
    info = {"rounds": 0, "rounds_with_taps": 0, "rows_real_taps": 0, "rows_synthetic_taps": 0, "skipped_noncontig": 0}
    for f in files:
        r = torch.load(f, map_location="cpu", weights_only=False)
        req_ids = list(r["req_ids"])
        nr = len(req_ids)
        k = int(r["num_speculative_steps"])
        nq = int(r["num_query_per_req"])
        if k != N or nq != N:
            raise RuntimeError(f"round {r['round']}: k {k} nq {nq} != N {N}")
        qsl = [int(x) for x in r["query_start_loc"]]
        nrej = r["num_rejected"].tolist()
        tpos = r["target_positions"].long()
        qids = r["query_input_ids"].view(nr, nq).long()
        qpos = r["query_positions"].view(nr, nq).long()
        taps = None
        if r["taps"] is not None:
            taps = torch.cat([t.to(torch.bfloat16) for t in r["taps"]], dim=-1).contiguous()
            info["rounds_with_taps"] += 1
        g = torch.Generator().manual_seed(int(r["round"]))
        staged, reqs = [], []
        live = set(req_ids)
        for b, req in enumerate(req_ids):
            slot, new = slots.get(req, int(r["round"]), live)
            if new:
                pair.reset(slot)
            st, en = qsl[b], qsl[b + 1]
            nv = en - st - nrej[b]
            if nv > 0:
                lo = max(st, st + nv - W)
                pos = tpos[lo:st + nv]
                if not torch.equal(pos, pos[0] + torch.arange(pos.shape[0])):
                    info["skipped_noncontig"] += 1
                    continue
                if taps is not None:
                    rows = taps[lo:st + nv]
                    info["rows_real_taps"] += rows.shape[0]
                else:
                    rows = (torch.randn(st + nv - lo, 3 * 5120, generator=g) * 4).to(torch.bfloat16)
                    info["rows_synthetic_taps"] += rows.shape[0]
                staged.append((slot, int(pos[0]), rows.view(torch.uint8).numpy().reshape(rows.shape[0], -1)))
            kk = N if nr <= 2 else 4
            reqs.append((slot, int(qids[b, 0]), int(qpos[b, 0]), kk))
        pair.step(staged, reqs, "v1")
        info["rounds"] += 1
    return info


def main() -> None:
    root, out_path = sys.argv[1], sys.argv[2]
    cap = sys.argv[3] if len(sys.argv) > 3 else os.path.join(root, "captures/A-capture-20261022/dspark-captures")
    t0 = time.time()
    core, m = build_one_card(root)
    pair = Pair(m, core.expected)
    rep: dict = {"device": torch.cuda.get_device_name(0)}
    synthetic(pair, m, rounds=400, seed=11)
    rep["synthetic"] = dict(pair.counts)
    rep["v1"] = v1(pair, cap)
    rep["totals"] = dict(pair.counts)
    rep["first_differences"] = pair.first_diff
    rep["identical"] = pair.counts["ids_equal"] == pair.counts["requests"] == pair.counts["conf_equal"]
    rep["seconds"] = round(time.time() - t0, 1)
    with open(out_path, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: rep[k] for k in ("synthetic", "v1", "totals", "identical")}), flush=True)


if __name__ == "__main__":
    main()
