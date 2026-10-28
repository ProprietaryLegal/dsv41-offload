"""Amendment-3 matched pilot: immutable plan, three arms and exact block boundaries.

Captured rounds are evaluation-only. Only an explicitly synthetic manifest can
use rehearsal sizes; it can never qualify the production fast path.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
from typing import Literal, NotRequired, TypedDict, cast

import torch
from torch import Tensor

from ..live_reference import Reference
from .data import Block, D2Blocks, sha256
from .loop import BatchPlan, Run, train
from .model import Drafter, load
from .replay import read_checkpoint
from .pilot_units import Units, TokenCounts, work_units, require_matched
from .pilot_sample import Sample, sample_plan, validate_sample, DEV_POPULATION

ARMS = ("REF", "FAST", "FAST-B")
CHECKPOINTS = (2500, 5000, 10000, 20000)
SEED = 23
Recipe = Literal["kl", "tvd", "ace", "lr"]
RECIPES: tuple[Recipe, ...] = ("kl", "tvd", "ace", "lr")


class Spec(TypedDict):
    schema: str
    synthetic: bool
    recipe: NotRequired[Recipe]
    manifest_sha256: str
    parity_sha256: str
    g1_sha256: str
    source_sha256: dict[str, str]
    frozen_identity: str
    initialization_sha256: str
    run: dict[str, object]
    checkpoints: list[int]
    order_seed: int
    order_seed_b: int
    population_sha256: str
    arm_orders: dict[str, str]
    sample: NotRequired[Sample]
    matched_units: dict[str, dict[str, Units]]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def sources() -> dict[str, str]:
    root = Path(__file__).parent
    files = list(root.glob("*.py")) + [root/"run_pilot_locked.sh", root.parent.parent/"native/formats.py",
                                     root.parent.parent/"native/official.py", root.parent/"g1.py", root.parent/"live_reference.py"]
    return {str(p.relative_to(root.parent.parent)): sha256(p) for p in sorted(files)}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with path.open("x") as f:
        os.fchmod(f.fileno(), 0o600)
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def read_spec(path: Path, *, sampled: bool = False) -> Spec:
    spec = cast(Spec, json.loads(path.read_text()))
    validate_spec(spec, sampled=sampled)
    return spec


def validate_spec(spec: Spec, *, sampled: bool = False) -> None:
    if spec["schema"] != "dsv41.d2.pilot-plan.v1" or spec["source_sha256"] != sources():
        raise ValueError("pilot source identity mismatch")
    if spec["run"] != dict(asdict(pilot_run(spec.get("recipe", "kl")))) or spec["order_seed"] != SEED or spec["order_seed_b"] != SEED+1:
        raise ValueError("pilot matched settings mismatch")
    points = spec["checkpoints"]
    if (len(points) != 4 or points != sorted(set(points)) or points[0] <= 0
            or (not spec["synthetic"] and tuple(points) != CHECKPOINTS)):
        raise ValueError("pilot checkpoint schedule mismatch")
    if set(spec["arm_orders"]) != set(ARMS) or spec["arm_orders"]["REF"] != spec["arm_orders"]["FAST"]:
        raise ValueError("pilot arm/order mismatch")
    require_matched(spec["matched_units"])
    if "sample" in spec:
        validate_sample(spec["sample"], spec["manifest_sha256"])
        if not spec["synthetic"] and set(spec["sample"]["keys"]) != set(DEV_POPULATION):
            raise ValueError("real pilot requires amendment-7 dev population")
    elif sampled:
        raise ValueError("pilot requires its once-only pre-training sample")


def pilot_run(recipe: Recipe = "kl") -> Run:
    if recipe not in RECIPES:
        raise ValueError("unknown pilot recipe: "+recipe)
    return Run(epochs=1, seed=SEED, chunk=8192, rank=32, batch_size=128,
               divergence=recipe if recipe in ("tvd", "ace") else "kl",
               lr=5e-5 if recipe == "lr" else 2e-4, warmup_fraction=.1 if recipe == "lr" else 0.)


def schedule(length: int, checkpoints: list[int], arm: str, recipe: Recipe = "kl") -> tuple[BatchPlan, list[int], str]:
    if arm not in ARMS or length < checkpoints[-1]:
        raise ValueError("pilot requires all arms and enough unique training blocks")
    # Same first N shuffled blocks for every arm. FAST-B only reorders this
    # population, using its independent data-order seed; initialization is fixed.
    population = torch.randperm(length, generator=torch.Generator().manual_seed(SEED)).tolist()[:checkpoints[-1]]
    order = population
    if arm == "FAST-B":
        order = []
        generator = torch.Generator().manual_seed(SEED+1)
        start = 0
        for boundary in checkpoints:
            segment = population[start:boundary]
            permutation = torch.randperm(len(segment), generator=generator).tolist()
            order.extend(segment[i] for i in permutation)
            start = boundary
    batches, stops = [], []
    start = 0
    for boundary in checkpoints:
        for lo in range(start, boundary, 128):
            batches.append(tuple(order[lo:min(lo+128, boundary)]))
        stops.append(len(batches))
        start = boundary
    identity = digest({"order": order, "boundaries": checkpoints, "run": asdict(pilot_run(recipe))})
    return BatchPlan(tuple(batches), tuple(stops), identity), population, digest(order)


def matched_units(data: TokenCounts, points: list[int], recipe: Recipe = "kl") -> dict[str, dict[str, Units]]:
    profiles = {}
    for arm in ARMS:
        batch, _, _ = schedule(len(data), points, arm, recipe)
        profiles[arm] = {str(n): work_units(data, batch, pilot_run(recipe), step)
                         for n, step in zip(points, batch.checkpoints)}
    require_matched(profiles)
    return profiles


def verify_units(spec: Spec, data: TokenCounts) -> None:
    if spec["matched_units"] != matched_units(data, spec["checkpoints"], spec.get("recipe", "kl")):
        raise ValueError("pilot matched-unit contract differs from actual dataset")


def initialization(model: Drafter) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
        if name.endswith(".b") and bool(p.detach().count_nonzero()):
            raise ValueError("pilot requires zero-init LoRA B")
    return h.hexdigest()


def manifest_is_synthetic(manifest: Path) -> bool:
    """Recognize all dataset writers' synthetic markers; any positive wins."""
    metadata = cast(dict[str, object], json.loads(manifest.read_text()))
    checks = metadata.get("checks", {})
    generation = metadata.get("generation", {})
    return (metadata.get("synthetic") is True or metadata.get("taps_source") == "synthetic"
            or (isinstance(checks, dict) and (checks.get("synthetic") is True or checks.get("taps_source") == "synthetic"))
            or (isinstance(generation, dict) and generation.get("synthetic") is True))


def validate_inputs(spec: Spec, manifest: Path, parity: Path, g1: Path) -> None:
    for path, expected in ((manifest, spec["manifest_sha256"]), (parity, spec["parity_sha256"]), (g1, spec["g1_sha256"])):
        if sha256(path) != expected:
            raise ValueError("pilot manifest/parity/G1 receipt mismatch")
    is_synthetic = manifest_is_synthetic(manifest)
    if is_synthetic != spec["synthetic"]:
        raise ValueError("pilot synthetic label mismatch")
    p = json.loads(parity.read_text())
    if p.get("pass") is not True or p.get("samples", 0) < 50 or p.get("reference_math") is not True or p.get("frozen_identity") != spec["frozen_identity"]:
        raise ValueError("pilot unadapted reference parity missing or mismatched")
    gate = json.loads(g1.read_text())
    if not spec["synthetic"]:
        validate_live_g1(g1, manifest)


def validate_live_g1(path: Path, manifest: Path) -> Reference:
    from ..live_reference import verify_reference
    gate = json.loads(path.read_text())
    binding = gate.get("g1", {})
    if (manifest_is_synthetic(path) or gate.get("split") != "dev" or binding.get("split") != "dev"
            or binding.get("mode") != "live-reference" or binding.get("amendment") != "7b"
            or binding.get("statistic") != "position0_acceptance" or binding.get("minimum_live_fraction") != .5
            or binding.get("tripwire_passed") is not True
            or binding.get("pass") is not True or binding.get("binding") is not True
            or binding.get("status") != "PASS" or gate.get("manifest_sha256") != sha256(manifest)):
        raise ValueError("genuine amendment-7b binding dev G1(b) PASS required; legacy acap refused")
    if not gate.get("live_reference_path") or not binding.get("live_reference_sha256"):
        raise ValueError("missing live reference receipt/hash")
    live = verify_reference(Path(gate["live_reference_path"]))
    if live["status"] != "VALID" or binding["live_reference_sha256"] != live["sha256"]:
        raise ValueError("live reference receipt hash/status mismatch")
    from ..g1 import g1_verdict
    if binding != g1_verdict(gate["position0_match_rate"], "dev", live_reference=live):
        raise ValueError("G1 verdict differs from live position-0 score")
    return live


def prepare(a: argparse.Namespace) -> None:
    from .safety import assert_gpu_lease
    assert_gpu_lease()
    data = D2Blocks(a.manifest, "train")
    synthetic = manifest_is_synthetic(a.manifest)
    if a.rehearsal != synthetic:
        raise ValueError("rehearsal flag and manifest synthetic label must match")
    points = [25, 50, 100, 200] if synthetic else list(CHECKPOINTS)
    torch.manual_seed(SEED)
    model = load(a.ckpt, "cuda", rank=32, reference_math=True)
    recipe: Recipe = a.recipe
    plans = {arm: schedule(len(data), points, arm, recipe) for arm in ARMS}
    spec: Spec = {"schema": "dsv41.d2.pilot-plan.v1", "synthetic": synthetic, "recipe": recipe, "matched_units": matched_units(data, points, recipe), "manifest_sha256": sha256(a.manifest),
                  "parity_sha256": sha256(a.parity_receipt), "g1_sha256": sha256(a.g1_receipt), "source_sha256": sources(),
                  "frozen_identity": model.frozen_identity, "initialization_sha256": initialization(model),
                  "run": dict(asdict(pilot_run(recipe))), "checkpoints": points, "order_seed": SEED, "order_seed_b": SEED+1,
                  "population_sha256": digest(plans["REF"][1]), "arm_orders": {arm: plans[arm][2] for arm in ARMS}}
    validate_inputs(spec, a.manifest, a.parity_receipt, a.g1_receipt)
    write_json(a.plan, spec)


class Diagnostics:
    """Private per-step drafts for matched-arm agreement; no gating thresholds.

    REF/FAST comparisons at equal steps measure the evolving matched training
    trajectories, not fixed-weight arithmetic parity. FAST-B has a different
    order, so its per-step agreement is to the committed training targets.
    """
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, mode=0o700, exist_ok=True)

    @torch.no_grad()
    def __call__(self, model: Drafter, block: Block, h: Tensor, hx: Tensor, row: dict[str, float]) -> None:
        inp = {k: v.to(model.activation_device) for k, v in block["inputs"].items()}
        picks, _ = model.greedy(h, hx, inp["anchor"], chunk=8192)
        truth = block["targets"]["labels"].cpu()
        mask = block["targets"]["position_mask"].cpu()
        picks = picks.cpu()
        matches = (picks == truth) & mask
        # Prefix counts, excluding unobserved target positions.
        accepted = matches.long().cumprod(-1).sum(-1).float().mean()
        payload = {**row, "drafts": picks.tolist(), "target_position1_agreement": float(matches[:, 0].float().mean()),
                   "target_exact_blocks": float((matches | ~mask).all(-1).float().mean()),
                   "mean_accepted_prefix": float(accepted), "informational_only": True}
        path = self.root/f"step-{int(row['step']):09d}.json"
        if path.exists():
            if json.loads(path.read_text()) != payload:
                raise ValueError("resumed per-step diagnostics changed")
        else:
            write_json(path, payload)


def arm(a: argparse.Namespace) -> None:
    from .safety import assert_gpu_lease
    assert_gpu_lease()
    spec = read_spec(a.plan, sampled=True)
    validate_inputs(spec, a.manifest, a.parity_receipt, a.g1_receipt)
    data = D2Blocks(a.manifest, "train")
    verify_units(spec, data)
    plan, population, order = schedule(len(data), spec["checkpoints"], a.arm, spec.get("recipe", "kl"))
    if digest(population) != spec["population_sha256"] or order != spec["arm_orders"][a.arm]:
        raise ValueError("pilot data order or population changed")
    torch.manual_seed(SEED)
    model = load(a.ckpt, "cuda", rank=32, reference_math=a.arm == "REF")
    if model.frozen_identity != spec["frozen_identity"] or initialization(model) != spec["initialization_sha256"]:
        raise ValueError("pilot arm frozen weights or initialization mismatch")
    if a.arm != "REF":
        model.enable_fast_training()
    # Bind resume and checkpoints to the complete immutable pilot contract.
    plan = BatchPlan(plan.batches, plan.checkpoints, digest({"plan": spec, "arm": a.arm, "schedule": plan.identity}))
    root = a.output/a.arm
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    result = train(model, data, pilot_run(spec.get("recipe", "kl")), root/"latest.pt", stop, a.resume, plan=plan, observer=Diagnostics(root/"steps"))
    if not result["complete"]:
        raise SystemExit("pilot checkpointed before completion; resume this same plan")
    checkpoints = {}
    for blocks, step in zip(spec["checkpoints"], plan.checkpoints):
        p = root/f"latest.checkpoints/step-{step:09d}.pt"
        state, checkpoint_sha = read_checkpoint(p)
        if state["blocks"] != blocks or state["batch_plan_identity"] != plan.identity:
            raise ValueError("pilot archived boundary mismatch")
        checkpoints[str(blocks)] = {"path": str(p.resolve()), "sha256": checkpoint_sha, "step": step}
    write_json(root/"arm.json", {"schema": "dsv41.d2.pilot-arm.v1", "arm": a.arm, "recipe": spec.get("recipe", "kl"), "plan_sha256": sha256(a.plan),
                                "batch_plan_identity": plan.identity, "checkpoints": checkpoints,
                                "units": spec["matched_units"][a.arm], "step_count": result["step"], "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                "complete": True, "seconds": result["seconds"], "synthetic": spec["synthetic"]})


def assert_same_sample(plan: Path, previous_plan: Path) -> None:
    """Compare the complete pure draw to pilot 1 without changing either plan.

    The first plan's old source map is deliberately not validated against this
    new snapshot. Its file hash is admitted by pilot_stage.check_inputs instead.
    """
    previous = cast(Spec, json.loads(previous_plan.read_text()))
    current = read_spec(plan, sampled=True)
    validate_sample(previous["sample"], previous["manifest_sha256"])
    if current["sample"] != previous["sample"]:
        raise ValueError("second pilot sample differs from first pilot keys/hash/inputs")


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample", help="freeze the Amendment-4 held-out sample on CPU, before training")
    for name in ("plan", "manifest", "k2-dev", "k2-legal"):
        sample.add_argument("--"+name, required=True, type=Path)
    sample.add_argument("--previous-plan", type=Path)
    for cmd in ("prepare", "arm"):
        p = sub.add_parser(cmd)
        for n in ("manifest", "parity-receipt", "g1-receipt", "plan"):
            p.add_argument("--"+n, type=Path, required=True)
        p.add_argument("--ckpt", required=True)
        if cmd == "prepare":
            p.add_argument("--recipe", choices=RECIPES, default="kl")
            p.add_argument("--rehearsal", action="store_true")
        else:
            p.add_argument("--arm", choices=ARMS, required=True)
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(8)
    if a.command == "sample":
        sample_plan(a.plan, a.manifest, {"dev": a.k2_dev, "heldout_legal": a.k2_legal})
        if a.previous_plan is not None:
            assert_same_sample(a.plan, a.previous_plan)
    elif a.command == "prepare":
        prepare(a)
    else:
        arm(a)


if __name__ == "__main__":
    main()
