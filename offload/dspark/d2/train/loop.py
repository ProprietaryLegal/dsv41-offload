"""Boundary-checkpointed AdamW/cosine loop; deterministic exact resume."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import signal
import threading
import time
from typing import Any, Callable, Literal, cast

import torch

from .data import Block, Blocks, D2Blocks, SyntheticBlocks, collate, sha256
from .loss import confidence_loss, distillation_loss
from .model import Drafter, load
from .replay import apply_adapter, apply_trainables, read_checkpoint
from .synthetic import SyntheticSource, tiny_config
from .ops import TrainingCancelled, cancellation
from .fast_policy import BACKEND, numeric_sources
from .full_admission import full_run, pilot_recipe, require_full_pilot


@dataclass(frozen=True)
class Run:
    epochs: int = 3
    lr: float = 2e-4
    weight_decay: float = .01
    seed: int = 23
    chunk: int = 4096
    divergence: Literal["kl", "tvd", "ace"] = "kl"
    warmup_fraction: float = 0.
    stage: str = "distill"
    checkpoint_seconds: float = 300
    rank: int = 32
    batch_size: int = 1

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.chunk < 1 or not 0 < self.checkpoint_seconds <= 600 or self.rank not in (32, 64):
            raise ValueError("invalid epochs, checkpoint interval or LoRA rank")
        if self.stage not in ("distill", "confidence") or self.divergence not in ("kl", "tvd", "ace"):
            raise ValueError("unknown training stage or divergence")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("invalid warm-up fraction")


def lr_multiplier(run: Run, step: int, total: int) -> float:
    """Applied update index starts at zero; KL retains its exact cosine arithmetic."""
    if run.warmup_fraction:
        if step >= total:
            return 0.
        warmup = math.ceil(total * run.warmup_fraction)
        if step < warmup:
            return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(step-warmup,total-warmup)/(total-warmup)))
    return .5*(1+math.cos(math.pi*min(step,total)/total))


@dataclass(frozen=True)
class BatchPlan:
    """Optional pilot schedule. Defaults leave the qualified loop untouched."""
    batches: tuple[tuple[int, ...], ...]
    checkpoints: tuple[int, ...]
    identity: str


Observer = Callable[[Drafter, Block, torch.Tensor, torch.Tensor, dict[str, float]], None]


def atomic_save(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as f:
        os.fchmod(f.fileno(), 0o600)
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_boundary(path: Path, state: dict[str, Any]) -> None:
    atomic_save(path, state)
    archive = path.parent/(path.stem+".checkpoints")
    archive.mkdir(mode=0o700, exist_ok=True)
    version = archive/f"step-{state['step']:09d}.pt"
    if not version.exists():
        os.link(path, version)  # later atomic replacement of latest never mutates this inode
        fd = os.open(archive, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def snapshot(model: Drafter, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR,
             step: int, run: Run, data: Blocks, curve: list[dict[str, float]]) -> dict[str, Any]:
    return {**({"full_recipe": model.full_recipe} if hasattr(model, "full_recipe") else {}),
            "schema": "dsv41.d2.train-checkpoint.v1", "run": asdict(run), "dataset": data.fingerprint,
            "model_config": asdict(model.c), "acc": str(model.acc), "reference_math": model.reference_math,
            "fast_training": model.fast_training,
            "fast_backend": BACKEND if model.fast_training else None,
            "fast_numeric_sources": numeric_sources() if model.fast_training else None,
            "fast_agreement_sha256": getattr(model, "fast_agreement_sha256", None),
            "fast_pilot_sha256": getattr(model, "fast_pilot_sha256", None),
            "frozen_identity": model.frozen_identity,
            "trainable": {n: p.detach().cpu().clone() for n, p in model.named_parameters()},
            "valid_supervised_tokens": int(sum(r.get("valid_supervised_tokens", 0) for r in curve)),
            "optimizer_updates": sum("loss" in r for r in curve),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step,
            "torch_rng": torch.get_rng_state(), "random_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if any(p.is_cuda for p in model.parameters()) else [], "curve": curve}


def train(model: Drafter, data: Blocks, run: Run, output: Path, stop: threading.Event | None = None,
          resume: bool = False, max_steps: int | None = None, *, plan: BatchPlan | None = None,
          observer: Observer | None = None) -> dict[str, Any]:
    stop = stop or threading.Event()
    with cancellation(stop):
        return _train(model, data, run, output, stop, resume, max_steps, plan, observer)


def _train(model: Drafter, data: Blocks, run: Run, output: Path, stop: threading.Event,
           resume: bool, max_steps: int | None, plan: BatchPlan | None, observer: Observer | None) -> dict[str, Any]:
    run.validate()
    if not resume and (output.exists() or (output.parent/(output.stem+".checkpoints")).exists()):
        raise FileExistsError("checkpoint namespace exists; use --resume or a new output path")
    stop = stop or threading.Event()
    model.train()
    for name, p in model.named_parameters():
        p.requires_grad_("confidence_head" in name if run.stage == "confidence" else "confidence_head" not in name)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=run.lr, weight_decay=run.weight_decay, foreach=False,
                                  fused=True if model.fast_training else None)
    batches_per_epoch = math.ceil(len(data)/run.batch_size)
    total = run.epochs * batches_per_epoch
    if plan is not None:
        if (not plan.batches or any(not b or len(b) > run.batch_size or any(i < 0 or i >= len(data) for i in b)
                                    for b in plan.batches)
                or any(s < 1 or s > len(plan.batches) for s in plan.checkpoints)):
            raise ValueError("invalid pilot batch plan")
        total = len(plan.batches)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(run, step, total))
    random.seed(run.seed)
    torch.manual_seed(run.seed)
    device = next(model.parameters()).device
    step, curve = 0, []
    if resume:
        state, _ = read_checkpoint(output)
        if state["run"] != asdict(run) or state["dataset"] != data.fingerprint or state["model_config"] != asdict(model.c) or state["acc"] != str(model.acc) or state["reference_math"] != model.reference_math:
            raise ValueError("resume contract changed (run, data, shape or precision)")
        if state.get("full_recipe") != getattr(model, "full_recipe", None):
            raise ValueError("resume full recipe changed")
        if state["frozen_identity"] != model.frozen_identity:
            raise ValueError("resume frozen-weight identity changed")
        if state.get("batch_plan_identity") != (plan.identity if plan else None):
            raise ValueError("resume pilot batch plan changed")
        if state.get("fast_training", False) != model.fast_training:
            raise ValueError("resume training backend changed")
        if model.fast_training and (state.get("fast_backend") != BACKEND or state.get("fast_numeric_sources") != numeric_sources()):
            raise ValueError("resume fast numerical backend changed")
        if model.fast_training and state.get("fast_agreement_sha256") != getattr(model, "fast_agreement_sha256", None):
            raise ValueError("resume fast agreement identity changed")
        if model.fast_training and state.get("fast_pilot_sha256") != getattr(model, "fast_pilot_sha256", None):
            raise ValueError("resume fast pilot identity changed")
        apply_trainables(model, state["trainable"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["random_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, curve = state["step"], state["curve"]
    last_save = time.monotonic()
    begin = last_save
    orders: dict[int, list[int]] = {}
    while step < total and not stop.is_set():
        if max_steps is not None and step >= max_steps:
            break
        epoch, offset = divmod(step, batches_per_epoch)
        if epoch not in orders:
            orders = {epoch: torch.randperm(len(data), generator=torch.Generator().manual_seed(run.seed+epoch)).tolist()}
        selected = list(plan.batches[step]) if plan else orders[epoch][offset*run.batch_size:(offset+1)*run.batch_size]
        block = collate([data[i] for i in selected])
        inp = {k: t.to(device) for k, t in block["inputs"].items()}
        targets = {k: t.to(device) for k, t in block["targets"].items()}
        optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        rng = torch.get_rng_state(), random.getstate()
        cuda_rng = torch.cuda.get_rng_state_all() if any(p.is_cuda for p in model.parameters()) else []
        try:
            h, hx = model.hidden_from(inp)
            if run.stage == "confidence":
                loss, known = confidence_loss(model, h, hx, targets["previous"], targets["top_ids"],
                                              targets["top_logp"], targets["position_mask"], run.chunk)
            else:
                loss, _ = distillation_loss(model, h, chunk=run.chunk, divergence=run.divergence, **targets)
                known = 0
            if loss is not None:
                cast(Callable[[], None], loss.backward)()
            if stop.is_set():
                raise TrainingCancelled()
            if loss is not None:
                if plan is not None and not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite pilot loss")
                if not model.fast_training and not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters):
                    raise FloatingPointError("nonfinite trainable gradient")
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True,
                                               foreach=True if model.fast_training else None)
                # Informational diagnostics observe the actual pre-update batch;
                # they do not alter gradients, RNG or the pass rule.
                if observer is not None:
                    observer(model, block, h.detach(), hx.detach(), {"step": float(step+1), "loss": float(loss.detach())})
        except TrainingCancelled:
            optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(rng[0])
            random.setstate(rng[1])
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
            break
        if stop.is_set():
            optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(rng[0])
            random.setstate(rng[1])
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
            break
        applied_lr = float(optimizer.param_groups[0]["lr"])
        if loss is not None:
            optimizer.step()
            scheduler.step()
        step += 1
        seconds = time.monotonic()-started
        row = {"step": float(step), "epoch": float(epoch), "seconds": seconds,
               "lr": float(scheduler.get_last_lr()[0]), "applied_lr": applied_lr,
               "valid_supervised_tokens": float(targets["position_mask"].sum()), "confidence_labels": float(known)}
        if loss is not None:
            row["loss"] = float(loss.detach())
        else:
            row["skipped_unlabelled"] = 1.
        curve.append(row)
        print(json.dumps(row), flush=True)
        if time.monotonic()-last_save >= run.checkpoint_seconds or stop.is_set() or (plan and step in plan.checkpoints):
            state = snapshot(model, optimizer, scheduler, step, run, data, curve)
            if plan:
                state["batch_plan_identity"] = plan.identity
                state["blocks"] = sum(len(b) for b in plan.batches[:step])
            save_boundary(output, state)
            last_save = time.monotonic()
    state = snapshot(model, optimizer, scheduler, step, run, data, curve)
    if plan:
        state["batch_plan_identity"] = plan.identity
        state["blocks"] = sum(len(b) for b in plan.batches[:step])
    save_boundary(output, state)
    if run.stage == "confidence" and step == total and not any(r["confidence_labels"] > 0 for r in curve):
        raise ValueError("confidence stage completed no labelled updates; inspect top20 coverage")
    return {"step": step, "total_steps": total, "complete": step == total, "seconds": time.monotonic()-begin,
            "curve": curve}


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--ckpt")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--stage", choices=("distill", "confidence"), default="distill")
    ap.add_argument("--divergence", choices=("kl", "tvd"), default="kl")
    ap.add_argument("--rank", type=int, choices=(32, 64), default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--initial-adapter")
    precision = ap.add_mutually_exclusive_group()
    precision.add_argument("--reference-math", action="store_true")
    precision.add_argument("--fast-training", action="store_true")
    ap.add_argument("--fast-agreement", type=Path, help="passing fast/reference gates for this exact numerical backend")
    ap.add_argument("--pilot-receipt", type=Path, help="passing real Amendment-3 matched-pilot receipt")
    ap.add_argument("--parity-receipt", type=Path, help="required unadapted 50-round gate before production training")
    ap.add_argument("--g1-receipt", type=Path, help="D2-DATA g1.py score report for this real shard set")
    ap.add_argument("--pilot-source", type=Path, help="original immutable pilot repo for source-bound PASS verification")
    ap.add_argument("--k2-dev", type=Path)
    ap.add_argument("--k2-legal", type=Path)
    a = ap.parse_args()
    if (a.k2_dev is None) != (a.k2_legal is None):
        ap.error("provide both --k2-dev and --k2-legal")
    if a.device != "cpu":
        from .safety import assert_gpu_lease
        assert_gpu_lease()
    qualified_run: Run | None = None
    pilot_hash: str | None = None
    if a.synthetic:
        model = Drafter(SyntheticSource(tiny_config()), tiny_config(), device=a.device, rank=a.rank)
        if a.fast_training:
            model.enable_fast_training()
        data: Blocks = SyntheticBlocks(model.c)
    else:
        if not a.manifest or not a.ckpt:
            ap.error("production requires --manifest and --ckpt")
        if (not (a.reference_math or a.fast_training) or a.parity_receipt is None or a.g1_receipt is None
                or (a.fast_training and (a.pilot_receipt is None or a.fast_agreement is not None))):
            ap.error("production requires --reference-math or qualified --fast-training/--pilot-receipt, plus --parity-receipt and --g1-receipt; --fast-agreement cannot authorize production")
        from .pilot import validate_live_g1
        validate_live_g1(a.g1_receipt, a.manifest)
        if a.fast_training:
            if a.initial_adapter or (a.epochs, a.stage, a.divergence, a.batch_size, a.chunk, a.rank) != (3, "distill", "kl", 128, 8192, 32):
                raise ValueError("full FAST requires fresh initialization and fixed qualified settings; recipe derives from PASS")
            spec = json.loads(a.pilot_receipt.read_text())["plan"]
            pilot_hash = require_full_pilot(a.pilot_receipt, spec["frozen_identity"], a.manifest,
                a.parity_receipt, a.g1_receipt, a.batch_size, a.chunk, a.rank, a.pilot_source,
                {"dev": a.k2_dev, "heldout_legal": a.k2_legal} if a.k2_dev and a.k2_legal else None)
            recipe = pilot_recipe(a.pilot_receipt, pilot_hash)
            qualified_run = full_run(recipe)
        model = load(a.ckpt, a.device, a.rank, reference_math=a.reference_math)
        parity = json.loads(a.parity_receipt.read_text())
        if not parity["pass"] or parity["samples"] < 50 or not parity["reference_math"] or parity["frozen_identity"] != model.frozen_identity:
            raise ValueError("unadapted parity gate missing, failed, or wrong frozen weights")
        if a.fast_training:
            if model.frozen_identity != spec["frozen_identity"] or sha256(a.pilot_receipt) != pilot_hash:
                raise ValueError("full FAST base or admitted receipt changed")
            from .pilot import initialization
            if initialization(model) != spec["initialization_sha256"]:
                raise ValueError("full FAST fresh initialization differs from qualified pilot")
            setattr(model, "fast_pilot_sha256", pilot_hash)
            setattr(model, "full_recipe", recipe)
            model.enable_fast_training()
        data = D2Blocks(a.manifest, "train")
    if a.initial_adapter:
        apply_adapter(model, a.initial_adapter)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    result = train(model, data, qualified_run or Run(epochs=a.epochs, stage=a.stage, divergence=a.divergence, rank=a.rank,
                                   batch_size=a.batch_size, chunk=a.chunk),
                   a.output, stop, a.resume)
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}), flush=True)


if __name__ == "__main__":
    main()
