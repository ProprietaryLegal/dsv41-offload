"""NB1 items 2/3: per-shard draft latency on the solo card (MEASURED, CUDA-graph replays).

One TP2 shard at a time (``solo_rank``; collectives are local stand-ins, so this is shard compute, not pair
latency). Arms: K2, native kx2 default, native kx2 batch-invariant level 2 (built in the same process with an
explicit ``batch_invariant`` level; its draft graphs are level 1's, its insert graphs use 8-row sub-batches), and
drafter-only microbatching = each of B requests replayed as its own B1 graph of the
native default arm, back to back, with each request's outputs copied out before the next replay. Every timed cell
is (warmup, then ``iters`` replays bracketed by CUDA events); repetitions interleave every arm and cell in a rotated
order. Window-insertion graphs are timed per bucket (8..128 rows, written to the scratch slot). One rank per
process (three native-size models fill the card; ``latency_summary.py`` merges the per-rank logs). Emits one
``RESULT_JSON`` line of timings and the card's clocks; no tensors or token IDs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

UUID = 'GPU-bc62df2a-59cf-9828-7d43-558d1355bab7'
K = 5
BUCKETS = (8, 16, 32, 64, 128)


def clocks() -> str:
    return subprocess.run(['nvidia-smi', '-i', UUID, '--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,'
                           'power.limit', '--format=csv,noheader'], text=True, capture_output=True,
                          check=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=Path.home() / 'dsv41-offload/n4')
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, default=Path('/opt/dsv41/x86host-data/nb1'))
    parser.add_argument('--root', default='/opt/dsv41/x86host-data/dspark')
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=40)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    try:
        from stage_probe import bootstrap
    except ImportError:
        from offload.dspark.native.tools.nb1.stage_probe import bootstrap
    binaries = bootstrap(args.runtime, args.scratch)
    import torch
    from offload.dspark.worker.checkpoint import Checkpoint
    from offload.dspark.worker.tp2 import TPDrafter, check_visible
    from offload.dspark.native.tp2_native import NativeTPDrafter

    check_visible(pair=False)
    if os.environ.get('DSPARK_NATIVE_BATCH_INVARIANT') not in (None, '0'):
        raise RuntimeError('run the latency probe with the env switch unset; arms set the flag explicitly')
    torch.set_num_threads(4)
    start = time.monotonic()
    dev = torch.device('cuda', 0)
    ck = Checkpoint(args.root)
    cells: dict[str, list[float]] = {}
    loads: dict[str, float] = {}
    clock_log: list[str] = [clocks()]
    memory: dict[str, int] = {}

    rank = args.rank
    builders: dict[str, Callable[[], Any]] = {
        'k2': lambda: TPDrafter(ck, [dev, dev], slots=4, solo_rank=rank),
        'native': lambda: NativeTPDrafter(ck, [dev, dev], 'native-hybrid-kx2', batch_invariant=False, slots=4,
                                          solo_rank=rank),
        'native-invariant': lambda: NativeTPDrafter(ck, [dev, dev], 'native-hybrid-kx2', batch_invariant=2,
                                                    slots=4, solo_rank=rank),
    }
    models: dict[str, Any] = {}
    for name, make in builders.items():
        t0 = time.monotonic()
        before = torch.cuda.memory_allocated(dev)
        models[name] = make()
        models[name].capture(batch_sizes=(1, 2, 3, 4), ks=(K,), insert_rows=BUCKETS, n_outs=(4, 5),
                             margins=False)
        loads[f'r{rank}/{name}'] = time.monotonic() - t0
        memory[f'r{rank}/{name}'] = torch.cuda.memory_allocated(dev) - before
        print(f'PHASE r{rank} {name} built+captured {loads[f"r{rank}/{name}"]:.0f}s', flush=True)
    generator = torch.Generator().manual_seed(20261026)
    lasts = torch.tensor([32767, 32800, 33000, 40000])
    windows = []
    for slot in range(4):
        pos = torch.arange(int(lasts[slot]) - 127, int(lasts[slot]) + 1)
        windows.append((pos, torch.randn(len(pos), ck.shape.hidden, generator=generator).to(torch.bfloat16)))
    anchors = torch.randint(100, 10000, (4,), generator=generator)
    for m in models.values():
        m.reset_slots([0, 1, 2, 3])
        for slot, (pos, mx) in enumerate(windows):
            m.insert_mx_eager(torch.full_like(pos, slot), pos, mx)
    inputs = torch.stack((torch.arange(4), anchors, lasts, lasts - 127)).contiguous().pin_memory()
    singles = [inputs[:, b:b + 1].contiguous().pin_memory() for b in range(4)]
    packed = {b: inputs[:, :b].contiguous().pin_memory() for b in range(1, 5)}
    scratch_rows = {b: (torch.full((b,), 4, dtype=torch.int64).pin_memory(),
                        torch.arange(b, dtype=torch.int64).pin_memory(),
                        (torch.randn(b, ck.shape.hidden * len(ck.shape.taps), generator=generator) * 0.05
                         ).to(torch.bfloat16).pin_memory()) for b in BUCKETS}
    sink_ids = torch.empty(4, K, dtype=torch.int64, device=dev)
    sink_conf = torch.empty(4, K, dtype=torch.float32, device=dev)

    def timed(fn: Callable[[], None]) -> float:
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize(dev)
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))  # type: ignore[no-untyped-call]
                  for _ in range(args.iters)]
        for e0, e1 in events:
            e0.record()
            fn()
            e1.record()
        torch.cuda.synchronize(dev)
        return float(statistics.median(e0.elapsed_time(e1) * 1000.0 for e0, e1 in events))

    def packed_fn(m: Any, b: int, n: int) -> Callable[[], None]:
        def fn() -> None:
            m.draft_graphed(packed[b], K, n_out=n)
        return fn

    def micro_fn(m: Any, b: int, n: int) -> Callable[[], None]:
        def fn() -> None:
            for i in range(b):
                ids, conf, _ = m.draft_graphed(singles[i], K, n_out=n)
                if ids is not None:                 # rank 0 owns the outputs; rank 1's graph returns none
                    sink_ids[i].copy_(ids[0])
                if conf is not None:
                    sink_conf[i].copy_(conf[0])
        return fn

    def insert_fn(m: Any, b: int) -> Callable[[], None]:
        def fn() -> None:
            m.insert_taps_graphed(*scratch_rows[b])
        return fn

    plan: list[tuple[str, Callable[[], None]]] = []
    for name, m in models.items():
        for b in BUCKETS:
            plan.append((f'r{rank}/{name}/insert{b}', insert_fn(m, b)))
    for name, m in models.items():
        for b in range(1, 5):
            for n in (4, 5):
                plan.append((f'r{rank}/{name}/B{b}/n{n}', packed_fn(m, b, n)))
    for b in range(2, 5):
        for n in (4, 5):
            plan.append((f'r{rank}/native-microbatched/B{b}/n{n}', micro_fn(models['native'], b, n)))
    for rep in range(args.reps):
        clock_log.append(clocks())
        shift = rep * len(plan) // args.reps
        for key, fn in plan[shift:] + plan[:shift]:
            cells.setdefault(key, []).append(timed(fn))
        print(f'PHASE r{rank} rep {rep} t={time.monotonic() - start:.0f}s', flush=True)
    clock_log.append(clocks())

    summary = {k: dict(median_us=statistics.median(v), run_medians_us=v, min_us=min(v), max_us=max(v))
               for k, v in cells.items()}
    print('RESULT_JSON ' + json.dumps(dict(
        label='MEASURED', mode='per-shard (solo_rank) CUDA-graph replay on bc62df2a, 225 W, flock -n',
        reps=args.reps, iters=args.iters, warmup=args.warmup, cells=summary, loads_seconds=loads,
        model_bytes=memory, clocks=clock_log, prebuilt_extensions_sha256=binaries,
        torch_version=torch.__version__, seconds=time.monotonic() - start)), flush=True)


if __name__ == '__main__':
    main()
