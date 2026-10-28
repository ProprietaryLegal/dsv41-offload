"""Private N1-only performance probes; no captured input receives optimizer updates."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import functools
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterator, TypedDict, cast

import torch
from torch import Tensor

from .loss import distillation_loss
from .model import Drafter, Stage, load
from .ops import Dense, PackedLinear
from .safety import assert_gpu_lease


def n1_inputs(root: Path, batch: int, device: str) -> dict[str, Tensor]:
    paths = sorted(root.glob("n1-*.pt"))
    if len(paths) != 4 or batch < 1:
        raise ValueError("require all four N1 synthetic fixtures and positive batch")
    windows = [torch.load(p, map_location="cpu", weights_only=False)["inputs"] for p in paths]
    return {k: torch.cat([windows[i % 4][k] for i in range(batch)], 0).to(device) for k in windows[0]}


class Targets(TypedDict):
    previous: Tensor
    top_ids: Tensor
    top_logp: Tensor
    tail_mass: Tensor
    labels: Tensor
    present: Tensor
    position_mask: Tensor


def synthetic_targets(model: Drafter, inp: dict[str, Tensor]) -> Targets:
    b = inp["anchor"].numel()
    device = inp["anchor"].device
    previous = (inp["anchor"][:, None] + torch.arange(5, device=device)) % model.c.vocab
    present = torch.ones(b, 5, device=device, dtype=torch.bool)
    return {"previous": previous, "top_ids": torch.arange(20, device=device)[None, None].expand(b, 5, -1),
            "top_logp": torch.full((b, 5, 20), .04, device=device).log(),
            "tail_mass": torch.full((b, 5), .2, device=device), "labels": previous,
            "present": present, "position_mask": present}


class PhaseTimes:
    """Nested synchronized exclusive wall times. Instrumentation overhead is explicit."""

    def __init__(self, synchronize: Callable[[], None]):
        self.synchronize = synchronize
        self.seconds: dict[str, float] = defaultdict(float)
        self.stack: list[list[Any]] = []

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        self.synchronize()
        start = time.perf_counter()
        frame: list[Any] = [name, 0.]
        self.stack.append(frame)
        try:
            yield
        finally:
            self.synchronize()
            elapsed = time.perf_counter() - start
            self.stack.pop()
            self.seconds[name] += elapsed - frame[1]
            if self.stack:
                self.stack[-1][1] += elapsed

    def wrap(self, fn: Callable[..., Any], name: str) -> Callable[..., Any]:
        @functools.wraps(fn)
        def timed(*args: Any, **kwargs: Any) -> Any:
            with self.phase(name):
                return fn(*args, **kwargs)
        return timed


@contextmanager
def instrument(times: PhaseTimes) -> Iterator[None]:
    changes = [(Drafter, "conditioning", "conditioning"), (Stage, "attention", "attention"),
               (Stage, "moe", "experts"), (Dense, "forward", "dense"),
               (PackedLinear, "forward", "experts"), (Drafter, "logit_chunk", "markov_head")]
    originals = []
    for cls, attr, name in changes:
        original = getattr(cls, attr)
        originals.append((cls, attr, original))
        setattr(cls, attr, times.wrap(original, name))
    try:
        yield
    finally:
        for cls, attr, original in originals:
            setattr(cls, attr, original)


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference-math", action="store_true")
    ap.add_argument("--fast-training", action="store_true")
    ap.add_argument("--operator-profile", action="store_true", help="copy operator evidence (adds profiler overhead)")
    a = ap.parse_args()
    assert_gpu_lease()
    torch.set_num_threads(8)
    model = load(a.ckpt, "cuda", reference_math=a.reference_math)
    if a.fast_training:
        model.enable_fast_training()
    model.train()
    inp = n1_inputs(a.fixtures, a.batch, model.activation_device)
    target = synthetic_targets(model, inp)
    parameters = [p for n, p in model.named_parameters() if "confidence_head" not in n]
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, foreach=False, fused=a.fast_training)
    rows = []
    for step in range(a.steps):
        times = PhaseTimes(torch.cuda.synchronize)
        optimizer.zero_grad(set_to_none=True)
        begin = time.perf_counter()
        with instrument(times), times.phase("other_forward"):
            h, _ = model.hidden_from(inp)
            with times.phase("loss"):
                loss, _ = distillation_loss(model, h, **target)
            with times.phase("backward_other"):
                cast(Callable[[], None], loss.backward)()
            with times.phase("optimizer"):
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
        seconds = time.perf_counter() - begin
        row = {"step": step, "seconds": seconds, "blocks_per_second": a.batch / seconds,
               "exclusive_seconds": dict(times.seconds), "loss": float(loss.detach())}
        rows.append(row)
        print(json.dumps(row), flush=True)
    copies = []
    if a.operator_profile:
        optimizer.zero_grad(set_to_none=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            h, _ = model.hidden_from(inp)
            loss, _ = distillation_loss(model, h, **target)
            cast(Callable[[], None], loss.backward)()
            optimizer.step()
            torch.cuda.synchronize()
        copies = [{"name": e.key, "calls": e.count, "self_cpu_seconds": e.self_cpu_time_total / 1e6,
                   "self_device_seconds": e.self_device_time_total / 1e6} for e in prof.key_averages()
                  if "copy" in e.key.lower() or "memcpy" in e.key.lower() or e.key == "aten::to"]
    report = {"batch": a.batch, "reference_math": a.reference_math, "fast_training": a.fast_training, "steps": rows,
              "copy_operators_nonadditive": copies, "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "note": "Synchronized phase wall times include overhead; backward recomputation appears in its phase. "
                      "Copy operator times overlap phase times and must not be added to them."}
    a.out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    a.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
