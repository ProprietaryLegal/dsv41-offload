"""R-SP in-window collective microbenchmark for the TP2 RoCE link: all-reduce vs reduce-scatter + all-gather at
prompt-chunk sizes, plus a bit-identity check of NCCL's reduce-scatter against its all-reduce.

Coordinator-only, inside the HOLD with the Sparks quiet, before the first control start (WINDOW.md, Microbenchmark):
one process per Spark in the experiment image (torch only, no vLLM import), NCCL with the serving environment's NCCL
variables, `torchrun --nnodes 2 --nproc-per-node 1 ... -m offload.sp_prefill.rs_bench --out <json>`. Rank 0 writes one
JSON of timings and booleans; no model, prompt or text is involved. Dry `--plan` prints the sizes and touches nothing.

What it settles before any SP arm runs:
- exactness: every reduce-scatter row equals the all-reduce's row bit for bit (two-rank sum a + b in both). If not,
  the SP arm cannot reproduce greedy output and the window skips it (STOP SP, keep the controls);
- cost: per full-row decoder layer K1 runs 2 all-reduces of [T, 5120] bf16, SP runs 2 all-gathers of [L, 5120] and
  2 reduce-scatters of [Tp, 5120]; ring traffic is equal, so the difference is per-collective latency and the
  RoCE / NCCL protocol choice at these sizes. The replay-row fetch (4 x [R, ...] all-gathers) is timed too.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

ROWS = (4096, 8192, 16384)  # production chunk 4,096; PC1's 8,192 / 16,384
HIDDEN, HC_MULT = 5120, 4
FULL_ROW_LAYERS = 20  # layers 0-19 run on every prompt row; 20-39 replay 128-row windows (patch 0034)
REPLAY_ROWS = (128, 512)  # one / four prefill requests' replay windows
ITERS, WARMUP = 20, 5


class Collectives(Protocol):
    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor: ...

    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class Timing:
    median_ms: float
    p90_ms: float

    @classmethod
    def of(cls, samples: Sequence[float]) -> Timing:
        ordered = sorted(samples)
        return cls(statistics.median(ordered), ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))])


def timed(fn: Callable[[], torch.Tensor], clock: Callable[[Callable[[], torch.Tensor]], float],
          iters: int, warmup: int) -> Timing:
    for _ in range(warmup):
        fn()
    return Timing.of([clock(fn) for _ in range(iters)])


def bench_rows(coll: Collectives, rows: int, clock: Callable[[Callable[[], torch.Tensor]], float], *,
               hidden: int = HIDDEN, iters: int = ITERS, warmup: int = WARMUP, device: str = 'cpu') -> dict[str, object]:
    n, r = coll.world_size, coll.rank
    padded = -(-rows // n) * n
    local = padded // n
    gen = torch.Generator(device=device).manual_seed(1234 + r)
    partial = torch.randn(rows, hidden, generator=gen, device=device).to(torch.bfloat16)
    buf = torch.zeros(padded, hidden, dtype=torch.bfloat16, device=device)
    buf[:rows].copy_(partial)
    reduced = coll.all_reduce(partial)
    scattered = coll.reduce_scatter(buf)
    real = max(0, min(local, rows - r * local))
    exact = bool(torch.equal(scattered[:real], reduced[r * local : r * local + real])) and not bool(scattered[real:].any())
    gathered = coll.all_gather(scattered)
    exact = exact and bool(torch.equal(gathered[:rows], reduced))
    ar = timed(lambda: coll.all_reduce(partial), clock, iters, warmup)
    rs = timed(lambda: coll.reduce_scatter(buf), clock, iters, warmup)
    ag = timed(lambda: coll.all_gather(scattered), clock, iters, warmup)
    k1_layer = 2 * ar.median_ms
    sp_layer = 2 * (ag.median_ms + rs.median_ms)
    return {'rows': rows, 'padded_rows': padded, 'local_rows': local, 'message_bytes': rows * hidden * 2,
            'rs_equals_ar_bitwise': exact,
            'all_reduce_ms': ar.__dict__, 'reduce_scatter_ms': rs.__dict__, 'all_gather_ms': ag.__dict__,
            'per_layer_ms': {'k1': k1_layer, 'sp': sp_layer},
            'per_chunk_delta_ms': FULL_ROW_LAYERS * (sp_layer - k1_layer)}


def bench_replay(coll: Collectives, rows: int, clock: Callable[[Callable[[], torch.Tensor]], float], *,
                 hidden: int = HIDDEN, iters: int = ITERS, warmup: int = WARMUP, device: str = 'cpu') -> dict[str, object]:
    """SPRows.select's all-gathers when a trimming step leaves SP: residual [R, 4, H] bf16, post/res/pre mixes fp32."""
    tensors = [torch.zeros(rows, HC_MULT, hidden, dtype=torch.bfloat16, device=device),
               torch.zeros(rows, hidden, dtype=torch.bfloat16, device=device),
               torch.zeros(rows, HC_MULT, device=device), torch.zeros(rows, HC_MULT, HC_MULT, device=device),
               torch.zeros(rows, HC_MULT, device=device)]

    def once() -> torch.Tensor:
        out = tensors[0]
        for t in tensors:
            out = coll.all_gather(t)
        return out

    t = timed(once, clock, iters, warmup)
    return {'replay_rows': rows, 'select_ms': t.__dict__}


def run(coll: Collectives, clock: Callable[[Callable[[], torch.Tensor]], float], *, rows: Sequence[int] = ROWS,
        hidden: int = HIDDEN, iters: int = ITERS, warmup: int = WARMUP, device: str = 'cpu') -> dict[str, object]:
    cells = [bench_rows(coll, t, clock, hidden=hidden, iters=iters, warmup=warmup, device=device) for t in rows]
    replay = [bench_replay(coll, r, clock, hidden=hidden, iters=iters, warmup=warmup, device=device)
              for r in REPLAY_ROWS]
    exact = all(bool(c['rs_equals_ar_bitwise']) for c in cells)
    return {'schema': 'rsp-rs-bench-v1', 'world_size': coll.world_size, 'hidden': hidden, 'iters': iters,
            'cells': cells, 'replay': replay, 'rs_equals_ar_bitwise': exact,
            'verdict': 'SP ARM MAY RUN' if exact else 'STOP SP ARM: reduce-scatter differs from all-reduce'}


class _Dist:
    """torch.distributed NCCL (no vLLM)."""

    def __init__(self) -> None:
        import torch.distributed as dist

        self.dist = dist
        dist.init_process_group('nccl')
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        self.dist.all_reduce(out)
        return out

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        out = x.new_empty((x.shape[0] // self.world_size, *x.shape[1:]))
        self.dist.reduce_scatter_tensor(out, x)
        return out

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        out = x.new_empty((x.shape[0] * self.world_size, *x.shape[1:]))
        self.dist.all_gather_into_tensor(out, x.contiguous())
        return out


def cuda_clock(fn: Callable[[], torch.Tensor]) -> float:
    start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def wall_clock(fn: Callable[[], torch.Tensor]) -> float:
    began = time.perf_counter()
    fn()
    return 1000 * (time.perf_counter() - began)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='print the sizes only')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--iters', type=int, default=ITERS)
    args = parser.parse_args(argv)
    if args.plan:
        print(json.dumps({'rows': ROWS, 'hidden': HIDDEN, 'replay_rows': REPLAY_ROWS, 'iters': args.iters,
                          'warmup': WARMUP, 'full_row_layers': FULL_ROW_LAYERS}))
        return 0
    if args.out is None:
        parser.error('--out is required to run')
    coll = _Dist()
    result = run(coll, cuda_clock, iters=args.iters, device='cuda')
    if coll.rank == 0:
        args.out.write_text(json.dumps(result, indent=2) + '\n')
    coll.dist.destroy_process_group()
    return 0 if result['rs_equals_ar_bitwise'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
