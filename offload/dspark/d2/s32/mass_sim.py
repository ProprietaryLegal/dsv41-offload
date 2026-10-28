"""S32: fp32 emulation of the pinned V2 kernel `_topk_log_softmax_kernel` (vllm/v1/worker/gpu/sample/logprob.py:16-58).

Synthetic logits only (no model data). Emulates: fp32 max, exp(x - max), per-1024-block tree sum, sequential block
accumulation `se += tl.sum(e)`, lse = log(se), output `logits - max_val - lse`. Exact numpy exp/log stand in for the
Triton approximations (an ESTIMATE of their extra error is reported separately). Prints aggregate counts only.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

V, BLOCK, TOPK = 129280, 1024, 20


def bf16(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000            # round to nearest even
    return u.view(np.float32)


def tree(a: np.ndarray) -> np.ndarray:
    while a.shape[-1] > 1:
        h = a.shape[-1] // 2
        a = (a[..., :h] + a[..., h:]).astype(np.float32)
    return a[..., 0]


def rows(rng: np.random.Generator, n: int, gap_lo: float, gap_hi: float, sigma: float) -> np.ndarray:
    x = rng.normal(0.0, sigma, size=(n, V)).astype(np.float32)
    top = rng.integers(0, V, size=n)
    x[np.arange(n), top] = x.max(-1) + rng.uniform(gap_lo, gap_hi, size=n).astype(np.float32)
    return bf16(x)


def kernel(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    m = x.max(-1, keepdims=True)
    nb = -(-V // BLOCK)
    pad = np.full((n, nb * BLOCK), -np.inf, dtype=np.float32)
    pad[:, :V] = x
    e = np.exp((pad - m).astype(np.float32)).astype(np.float32).reshape(n, nb, BLOCK)
    se = np.zeros(n, dtype=np.float32)
    for b in range(nb):
        se = (se + tree(e[:, b])).astype(np.float32)
    lse = np.log(se).astype(np.float32)
    idx = np.argpartition(-x, TOPK - 1, axis=-1)[:, :TOPK]
    lp = ((np.take_along_axis(x, idx, -1) - m).astype(np.float32) - lse[:, None]).astype(np.float32)
    return lp, lse


def main() -> None:
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
    batches = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    out = {}
    for name, lo, hi, sigma in (("peaked_gap16-30", 16, 30, 1.5), ("confident_gap8-16", 8, 16, 1.5),
                                ("mixed_gap0-12", 0, 12, 2.5)):
        cnt = gt1 = gt1e6 = zero_lp1 = 0
        worst = -1.0
        for _ in range(batches):
            lp, _ = kernel(rows(rng, 64, lo, hi, sigma))
            for r in lp.astype(np.float64):
                mass = math.fsum(math.exp(v) for v in r)
                cnt += 1
                gt1 += mass > 1
                gt1e6 += mass > 1 + 1e-6
                zero_lp1 += float(r.max()) == 0.0
                worst = max(worst, mass - 1)
        out[name] = {"rows": cnt, "mass_gt_1": gt1, "mass_gt_1_plus_1e-6": gt1e6, "top1_lp_exactly_0": zero_lp1,
                     "max_mass_minus_1": worst}
    print(json.dumps({"schema": "s32.mass-sim.v1", "classification": "MEASURED (synthetic fp32 emulation)",
                      "results": out}, indent=1))


if __name__ == "__main__":
    main()
