"""Isolated S10 reproducer. Run each mode in a fresh process; a CUDA fault poisons that process.

GPU ownership/locks are enforced by the lane runner, before importing torch. No service or target model access.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Callable
from typing import TypeVar

import torch

from . import P2PPair

T = TypeVar("T")


def boundary(label: str, devs: list[torch.device], fn: Callable[[], T]) -> T:
    start = time.monotonic()
    print(json.dumps({"boundary": label, "state": "start", "current": torch.cuda.current_device()}), flush=True)
    value = fn()
    for dev in dict.fromkeys(devs):
        torch.cuda.synchronize(dev)
    print(json.dumps({"boundary": label, "state": "ok", "seconds": time.monotonic() - start,
                      "allocated": [torch.cuda.memory_allocated(d) for d in devs]}), flush=True)
    return value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="Use existing checkpoint, not downloaded weights")
    ap.add_argument("--swapped", action="store_true", help="Swap logical rank devices within the authorized pair")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--legacy-capture", action="store_true", help="Reproduce model.py's implicit capture stream")
    ap.add_argument("--foreign-default", action="store_true", help="Seed Torch's class-wide stream on the other device")
    ap.add_argument("--skip-one-capture", action="store_true", help="Separate construction from capture as the trigger")
    ap.add_argument("--allocator-import", action="store_true", help="Minimal trigger: exllamav3 Python import, no model")
    args = ap.parse_args()
    devs = [torch.device("cuda", i) for i in range(min(2, torch.cuda.device_count()))]
    if not devs:
        raise RuntimeError("lifecycle requires the authorized development card or pair")
    if len(devs) == 1:
        devs *= 2
    if args.swapped:
        devs.reverse()
    if args.allocator_import:
        with torch.cuda.device(devs[1]):
            import exllamav3  # noqa: F401 -- intentionally invokes its runtime allocator default setter
            probe = torch.empty(16 << 20, dtype=torch.uint8, device=devs[1])
        print(json.dumps({"allocator_segments": [
            {"device": s["device"], "is_expandable": s.get("is_expandable", False)}
            for s in torch.cuda.memory_snapshot()]}), flush=True)
        del probe
    if args.full:
        from ..checkpoint import Checkpoint
        from ..model import DSparkDrafter
        from ..tp2 import TPDrafter
        ck = boundary("checkpoint", devs, lambda: Checkpoint("/opt/dsv41/x86host-data/dspark"))
        for cycle in range(args.cycles):
            with torch.cuda.device(devs[1]):
                one = boundary(f"{cycle}:one-build", devs, lambda: DSparkDrafter(
                    ck, devs[1], slots=1, markov_cache_rows=256))
            tp = boundary(f"{cycle}:tp-build", devs, lambda: TPDrafter(
                ck, devs, slots=1, markov_cache_rows=256))
            if args.foreign_default:
                # Separate this known Torch global-stream hazard from the original build-order reproducer.
                torch.cuda.graphs.graph.default_capture_stream = torch.cuda.Stream(devs[0])
            graph = torch.cuda.graph
            if args.legacy_capture:
                def implicit_graph(*a: object, **kw: object) -> object:
                    kw.pop("stream", None)
                    return graph(*a, **kw)
                torch.cuda.graph = implicit_graph
            try:
                if not args.skip_one_capture:
                    with torch.cuda.device(devs[1]):
                        boundary(f"{cycle}:one-capture", devs, lambda: one.capture(
                            batch_sizes=(1,), ks=(5,), n_outs=(5,), insert_rows=(8,), margins=False))
            finally:
                torch.cuda.graph = graph
            boundary(f"{cycle}:tp-capture", devs, lambda: tp.capture(
                batch_sizes=(1,), ks=(5,), n_outs=(5,), insert_rows=(8,), margins=False))
            inp = torch.tensor([[0], [123], [0], [0]], dtype=torch.int64).pin_memory()
            eager = boundary(f"{cycle}:eager", devs, lambda: tp.draft(inp, 5))
            ids = eager[0].cpu().clone()
            conf = eager[1].cpu().clone()
            got = boundary(f"{cycle}:replay", devs, lambda: tp.draft_graphed(inp, 5))
            assert torch.equal(ids, got[0].cpu()) and torch.equal(conf, got[1].cpu())
            del eager, got, tp, one
            boundary(f"{cycle}:teardown", devs, lambda: (gc.collect(), torch.cuda.empty_cache()))
    else:
        from ..tests.test_p2p import MAXW, SameDevicePair, run_sequence
        for cycle in range(args.cycles):
            if devs[0] == devs[1]:
                pr = SameDevicePair(devs[0])
            else:
                # run_sequence includes 32 * 5120-word payloads. Match its capacity in both modes.
                pr = boundary(f"{cycle}:pair-build", devs, lambda: P2PPair(devs, MAXW))
            pr.streams = [torch.cuda.Stream(d) for d in devs]
            boundary(f"{cycle}:collectives", devs, lambda: run_sequence(pr, seed=cycle, n_ops=60))
            del pr
            boundary(f"{cycle}:teardown", devs, lambda: (gc.collect(), torch.cuda.empty_cache()))
    print(json.dumps({"result": "PASS", "cycles": args.cycles, "swapped": args.swapped, "full": args.full}), flush=True)


if __name__ == "__main__":
    main()
