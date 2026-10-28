"""Solo synthetic qualification of fused Markov/q/graph semantics (no weights)."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import resource
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch

from ...worker import sampled_fused
from ...worker.sampled import MASS, Top32
from ...worker.sampled_fused import FusedContext
from ...worker.tp2 import MarkovShard

UUID = "GPU-bc62df2a-59cf-9828-7d43-558d1355bab7"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_fixture() -> Any:
    dev = torch.device("cuda:0")
    vocab, rank_dim = 129280, 256
    w1 = (torch.randn(vocab, rank_dim, device=dev) / 8).bfloat16()
    slots = torch.full((vocab,), -1, dtype=torch.int64, device=dev)
    slots[:4] = torch.arange(4, device=dev)
    ranks: list[Any] = []
    for rank, (lo, hi) in enumerate(((0, 64768), (64768, vocab))):
        mb = MarkovShard.__new__(MarkovShard)
        mb.w1 = w1
        mb.w2 = (torch.randn(hi - lo, rank_dim, device=dev) / 8).bfloat16()
        mb.slot_of_token = slots
        mb.cache = (torch.randn(4, hi - lo, device=dev) / 4).bfloat16()
        ranks.append(SimpleNamespace(r=rank, v0=lo, v1=hi, markov=mb, dev=dev))
    return SimpleNamespace(ALL=(0, 1), devs=[dev, dev], ranks=ranks, emulated=True)


def qualify(model: Any, batch: int, depth: int, mask: torch.Tensor) -> dict[str, Any]:
    dev = model.devs[0]
    ctx = FusedContext(model, batch, depth)
    keys = torch.tensor([[4104 + j, 1, 32768] for j in range(batch)], device=dev, dtype=torch.int64)
    ctx.update(keys, mask)
    logits = {r: torch.randn(batch, depth, rk.v1 - rk.v0, device=dev).bfloat16()
              for r, rk in enumerate(model.ranks)}
    output = {r: torch.zeros(batch, depth, device=dev, dtype=torch.int64) for r in model.ALL}
    sampler = Top32(batch, 129280, dev)
    checks = 0
    # Cached, uncached and mixed rows; sequential positions use
    # the actually selected previous token, exercising conditional Markov q.
    for initial in (0, 7):
        anchor = torch.tensor([initial if j % 2 == 0 else 2 for j in range(batch)], device=dev)
        for pos in range(depth):
            previous = {r: anchor if pos == 0 else output[r][:, pos - 1] for r in model.ALL}
            ctx.select(model.ALL, logits, previous, output, pos)
            support, q = ctx.result()
            parts = []
            for r in model.ALL:
                mb = model.ranks[r].markov
                prev = previous[r].contiguous()
                parts.append((logits[r][:, pos] + mb.bias(prev, mb.embed(prev))).bfloat16())
            full = torch.cat(parts, -1)
            poskeys = keys.clone()
            poskeys[:, 2] += pos
            token, ref_ids, ref_q = sampler(full, poskeys, torch.full((batch,), .7, device=dev))
            expected = torch.where(mask, token.long(), full.argmax(-1))
            torch.testing.assert_close(output[0][:, pos], expected, rtol=0, atol=0)
            torch.testing.assert_close(output[1][:, pos], expected, rtol=0, atol=0)
            torch.testing.assert_close(support[:, pos], ref_ids, rtol=0, atol=0)
            torch.testing.assert_close(q[:, pos], ref_q * mask[:, None], rtol=0, atol=0)
            if not torch.equal((q[:, pos].double() * MASS).sum(-1), mask.double() * MASS):
                raise AssertionError("dyadic mass differs")
            checks += batch
    # Capture the complete sequential selector. Replaying reads fresh RNG
    # keys and masks; candidate retirement counters must reset each invocation.
    anchor = torch.zeros(batch, device=dev, dtype=torch.int64)

    def run() -> None:
        for pos in range(depth):
            previous = {r: anchor if pos == 0 else output[r][:, pos - 1] for r in model.ALL}
            ctx.select(model.ALL, logits, previous, output, pos)

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    times: list[float] = []
    event_factory = cast(Callable[..., torch.cuda.Event], torch.cuda.Event)
    start, end = event_factory(enable_timing=True), event_factory(enable_timing=True)
    for rnd in range(100):
        ctx.state[0]["keys"][:, 1].fill_(rnd)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        ctx.result()
        times.append(start.elapsed_time(end))
    # Fixed RNG keys imply exact graph/eager equality, including sparse q.
    before = (output[0].clone(), ctx.state[0]["ids"].clone(), ctx.state[0]["q"].clone())
    run()
    for actual, expected in zip((output[0], ctx.state[0]["ids"], ctx.state[0]["q"]), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    times.sort()
    return {"batch": batch, "depth": depth, "sampled_rows": int(mask.sum().item()),
            "exact_position_checks": checks, "graph_replays": 100,
            "selector_median_ms": times[50], "selector_p95_ms": times[95],
            "scope": "MEASURED two logical TP2 shards on solo; selector including Markov, excludes backbone/pair"}


def compile_peer_branches(model: Any) -> None:
    """Compile both real-rank specializations without executing peer traffic.

    This catches JIT/assembly errors; it does not qualify physical-pair numerics,
    visibility, timeout or graph interoperability.
    """
    # JIT warmup is a dynamic compiler API absent from the narrow launch stub.
    kernel: Any = sampled_fused._kernel

    class CompileOnly:
        def warmup(self, *args: Any, **kwargs: Any) -> Any:
            return kernel.warmup(*args, **kwargs)

        def __getitem__(self, grid: Any) -> Any:
            def compile_call(*args: Any, **kwargs: Any) -> Any:
                return kernel.warmup(*args, grid=grid, **kwargs)
            return compile_call

    model.emulated = False
    try:
        ctx = FusedContext(model, 1, 5)
        logits = {r: torch.zeros(1, 5, rk.v1 - rk.v0, device=rk.dev, dtype=torch.bfloat16)
                  for r, rk in enumerate(model.ranks)}
        previous = {r: torch.zeros(1, device=rk.dev, dtype=torch.int64) for r, rk in enumerate(model.ranks)}
        outputs = {r: torch.zeros(1, 5, device=rk.dev, dtype=torch.int64) for r, rk in enumerate(model.ranks)}
        with patch.object(sampled_fused, "_kernel", CompileOnly()):
            ctx.select((0, 1), logits, previous, outputs, 0)
    finally:
        model.emulated = True


def edge_cases(model: Any) -> dict[str, bool]:
    dev = model.devs[0]
    ctx = FusedContext(model, 2, 5)
    keys = torch.tensor([[4104, 7, 100], [4105, 8, 101]], device=dev, dtype=torch.int64)
    mask = torch.tensor([True, False], device=dev)
    ctx.update(keys, mask)
    logits = {r: torch.zeros(2, 5, rk.v1 - rk.v0, device=dev, dtype=torch.bfloat16)
              for r, rk in enumerate(model.ranks)}
    previous = {r: torch.zeros(2, device=dev, dtype=torch.int64) for r in model.ALL}
    output = {r: torch.zeros(2, 5, device=dev, dtype=torch.int64) for r in model.ALL}
    # Force exact global ties, including the shard boundary. Signed zero must
    # not move a larger ID ahead of a smaller ID.
    saved = [rk.markov.cache[0].clone() for rk in model.ranks]
    try:
        for rk in model.ranks:
            rk.markov.cache[0].zero_()
        logits[0][:, 0, 1] = -0.
        ctx.select(model.ALL, logits, previous, output, 0)
        ids, q = ctx.result()
        expected_ids = torch.arange(32, device=dev, dtype=torch.int32).expand(2, 32)
        torch.testing.assert_close(ids[:, 0], expected_ids, rtol=0, atol=0)
        assert output[0][1, 0].item() == 0
        torch.testing.assert_close(q[0, 0], torch.full((32,), 1 / 32, device=dev), rtol=0, atol=0)
        assert torch.count_nonzero(q[1, 0]).item() == 0
        for bad in (float("nan"), float("inf")):
            logits[1][0, 0, 3] = bad
            ctx.select(model.ALL, logits, previous, output, 0)
            for pos in range(1, 5):
                continuation = {r: output[r][:, pos - 1] for r in model.ALL}
                ctx.select(model.ALL, logits, continuation, output, pos)
            try:
                ctx.result()
            except RuntimeError:
                pass
            else:
                raise AssertionError("invalid fused logit accepted")
        for rank in model.ALL:
            logits[rank][:, 0].fill_(float("-inf"))
        ctx.select(model.ALL, logits, previous, output, 0)
        try:
            ctx.result()
        except RuntimeError:
            pass
        else:
            raise AssertionError("all-masked fused logits accepted")
    finally:
        for rk, cache in zip(model.ranks, saved, strict=True):
            rk.markov.cache[0].copy_(cache)
    return {"lower_id_ties": True, "signed_zero": True, "mixed_t0_no_draw": True,
            "invalid_intermediate_safe_continuation": True,
            "nan_inf_all_masked_refused": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != UUID:
        raise RuntimeError("PP4 assigned solo UUID required")
    fd = int(os.environ["PP4_LOCK_FD"])
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-") != UUID.removeprefix("GPU-"):
        raise RuntimeError("wrong physical GPU")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(4104)
    rows: list[dict[str, Any]] = []
    root = Path(__file__).resolve().parents[2]
    result: dict[str, Any] = {"schema": "pp4.fused-qualification.v1", "complete": False,
                              "classification": "MEASURED synthetic solo", "uuid": UUID,
                              "torch": torch.__version__, "rows": rows,
                              "source_sha256": digest(Path(__file__)),
                              "fused_sha256": digest(root / "worker/sampled_fused.py")}
    started = time.monotonic()
    model = model_fixture()
    compile_peer_branches(model)
    result["peer_specializations_compile_only"] = True
    result["edges"] = edge_cases(model)
    for batch in range(1, 5):
        for depth in (4, 5):
            for mixed in (False, True):
                mask = torch.tensor([j % 2 == 0 if mixed else True for j in range(batch)], device=model.devs[0])
                row = qualify(model, batch, depth, mask)
                rows.append(row)
                result["maxrss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                result["gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
                args.out.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(row), flush=True)
    result["elapsed_s"] = time.monotonic() - started
    result["complete"] = True
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
