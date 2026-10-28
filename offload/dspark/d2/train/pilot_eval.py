"""Reference-only pilot evaluation using G2's top20 KL and fixed K2 trajectory.

This is the pre-registered matched pilot, not dev checkpoint selection or the
final G2 test attempt. Outputs contain aggregates and identities, never rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import statistics
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import torch

from .data import D2Blocks, sha256
from .evaluate import bucket_kl, match_length
from .loss import distillation_loss
from .model import Drafter, load
from .pilot import ARMS, SEED, Recipe, Spec, digest, initialization, read_spec, schedule, write_json, verify_units
from .perf import Targets
from .replay import apply_trainables, read_checkpoint
from .pilot_statistics import DEV_SPLITS, RequestTotal, split_roles
from .pilot_sample import SPLITS as SPLITS, RoundKey, Sample, complete_rounds, draw_sample, key_hash, population_splits

EVALUATOR = "G2-reference-top20-tail-fixed-K2-amend6ab-v2"
PART_SCHEMA = "dsv41.d2.pilot-evaluation-part.v1"


class Metrics(TypedDict):
    requests: int
    rounds: int
    population_sha256: str
    kl_by_position: list[float]
    loss: float
    accepted_length: float
    kl_median_by_position: list[float]
    request_totals: list[RequestTotal]


class Checkpoint(TypedDict):
    path: str
    sha256: str
    step: int


class Arm(TypedDict):
    schema: str
    arm: str
    plan_sha256: str
    batch_plan_identity: str
    checkpoints: dict[str, Checkpoint]
    step_count: int
    peak_reserved_bytes: int
    complete: bool
    seconds: float
    synthetic: bool
    recipe: NotRequired[Recipe]


class Evaluation(TypedDict):
    schema: str
    plan_sha256: str
    synthetic: bool
    evaluator: str
    k2_sha256: dict[str, str]
    baseline: dict[str, Metrics]
    arms: dict[str, dict[str, dict[str, Metrics]]]
    checkpoints: dict[str, dict[str, Checkpoint]]
    step_agreement: dict[str, object]
    sample_sha256: str
    split: NotRequired[str]
    split_roles: NotRequired[dict[str, str]]


def read_arm(root: Path, name: str, spec: Spec, plan: Path | str, length: int) -> Arm:
    arm = cast(Arm, json.loads((root/name/"arm.json").read_text()))
    batch, _, _ = schedule(length, spec["checkpoints"], name, spec.get("recipe", "kl"))
    identity = digest({"plan": spec, "arm": name, "schedule": batch.identity})
    if (arm["schema"] != "dsv41.d2.pilot-arm.v1" or arm["arm"] != name or not arm["complete"]
            or arm["plan_sha256"] != (sha256(plan) if isinstance(plan, Path) else plan) or arm["synthetic"] != spec["synthetic"]
            or arm["batch_plan_identity"] != identity or arm["step_count"] != len(batch.batches)
            or arm.get("recipe", None if "recipe" in spec else "kl") != spec.get("recipe", "kl")
            or arm.get("units") != spec["matched_units"][name]
            or set(arm["checkpoints"]) != {str(n) for n in spec["checkpoints"]}):
        raise ValueError("pilot arm receipt mismatch or missing checkpoint")
    for n, step in zip(spec["checkpoints"], batch.checkpoints):
        record = arm["checkpoints"][str(n)]
        state, actual = read_checkpoint(Path(record["path"]))
        if (actual != record["sha256"] or record["step"] != step or state["step"] != step or state["blocks"] != n
                or state.get("valid_supervised_tokens") != spec["matched_units"][name][str(n)]["valid_supervised_tokens"]
                or state.get("optimizer_updates") != step
                or [row.get("applied_lr") for row in state["curve"]] != spec["matched_units"][name][str(n)]["learning_rates"]
                or state.get("batch_plan_identity") != identity or state["run"] != spec["run"]
                or state["frozen_identity"] != spec["frozen_identity"] or state["reference_math"] != (name == "REF")
                or state.get("fast_training", False) != (name != "REF")
                or any(not torch.isfinite(p).all() for p in state["trainable"].values())
                or any(not all(torch.isfinite(torch.tensor(v)) for v in row.values()) for row in state["curve"])):
            raise ValueError("pilot checkpoint identity/settings/nonfinite mismatch")
    return arm


def validate_sample_metrics(sample: Sample, split: str, metrics: Metrics) -> None:
    keys = sample["keys"][split]
    if (metrics["rounds"] != len(keys) or metrics["requests"] != len({k["request"] for k in keys})
            or metrics["population_sha256"] != key_hash(keys)
            or {r["request_sha256"]: r["rounds"] for r in metrics["request_totals"]}
               != {digest(request): count for request, count in Counter(k["request"] for k in keys).items()}):
        raise ValueError("pilot scored keys differ from the frozen sample")


def validate_sample_evaluation(spec: Spec, evaluation: Evaluation, splits: tuple[str, ...]) -> None:
    sample = spec["sample"]
    if (evaluation.get("sample_sha256") != sample["sha256"] or evaluation["k2_sha256"] != sample["k2_sha256"]
            or set(evaluation["baseline"]) != set(splits) or set(evaluation["arms"]) != set(ARMS)
            or set(evaluation["checkpoints"]) != set(ARMS)):
        raise ValueError("pilot evaluation sample hash/K2/split mismatch")
    if evaluation.get("split_roles") != split_roles(sample["keys"]):
        raise ValueError("pilot selection splits must be labelled dev; reserved test is frozen-candidate only")
    for split in splits:
        validate_sample_metrics(sample, split, evaluation["baseline"][split])
    for arm in ARMS:
        if (set(evaluation["arms"][arm]) != {str(n) for n in spec["checkpoints"]}
                or set(evaluation["checkpoints"][arm]) != {str(n) for n in spec["checkpoints"]}):
            raise ValueError("missing sampled pilot checkpoint")
        for metrics in evaluation["arms"][arm].values():
            if set(metrics) != set(splits):
                raise ValueError("sampled pilot split coverage mismatch")
            for split in splits:
                validate_sample_metrics(sample, split, metrics[split])


@torch.no_grad()
def score(model: Drafter, data: D2Blocks, k2: Path, keys: list[RoundKey], chunk: int = 8192,
          synthetic_windows: dict[str, dict[str, str]] | None = None) -> Metrics:
    if not model.reference_math or model.fast_training:
        raise ValueError("pilot evaluation requires qualified reference math")
    expected = {(k["request"], k["P"]) for k in keys}
    selected = [(index, key) for index, key in complete_rounds(data, k2) if (key["request"], key["P"]) in expected]
    if not keys or len(expected) != len(keys) or [key for _, key in selected] != keys:
        raise ValueError("pilot scored keys differ from the frozen sample")
    total_kl = torch.zeros(5, dtype=torch.float64)
    total_loss, total_accepted = 0., 0.
    actual: list[RoundKey] = []
    kl_rows: list[list[float]] = []
    request_totals: dict[str, RequestTotal] = {}
    for index, key in selected:
        request, pos = key["request"], key["P"]
        block = data[index]
        if block["request"] != request or int(block["inputs"]["P"][0]) != pos:
            raise ValueError("pilot scored keys differ from the frozen sample")
        targets = cast(Targets, {k: t.to(model.activation_device) for k, t in block["targets"].items()})
        if not bool(targets["position_mask"].all()):
            raise ValueError("sampled pilot round has incomplete targets")
        inp = {k: t.to(model.activation_device) for k, t in block["inputs"].items()}
        if synthetic_windows is not None:
            entry = synthetic_windows[request]
            exact = data.verified(entry["file"], entry["sha256"])
            original = cast(dict[str, torch.Tensor], torch.load(exact, map_location="cpu", weights_only=False))
            if int(original["P"][0]) != pos or int(original["anchor"][0]) != int(inp["anchor"][0]):
                raise ValueError("synthetic capture-window alignment mismatch")
            inp = {k: t.to(model.activation_device) for k, t in original.items()}
        h, hx = model.hidden_from(inp)
        picks, _ = model.greedy(h, hx, inp["anchor"], chunk=chunk)
        candidate_n = match_length(picks[0].tolist(), targets["labels"][0].tolist())
        if candidate_n is None:
            raise ValueError("indeterminate pilot replay round")
        logits = torch.cat([model.logit_chunk(h, targets["previous"], lo, min(lo+chunk, model.c.vocab))
                            for lo in range(0, model.c.vocab, chunk)], -1)[0]
        kl = bucket_kl(logits, targets["top_ids"][0], targets["top_logp"][0], targets["tail_mass"][0]).cpu()
        loss, _ = distillation_loss(model, h, chunk=chunk, **targets)
        if not torch.isfinite(kl).all() or not torch.isfinite(loss):
            raise FloatingPointError("nonfinite pilot evaluation")
        if request not in data.request_families:
            raise ValueError("pilot request missing source family")
        aggregate = request_totals.setdefault(request, {"request_sha256": digest(request),
            "family": data.request_families[request], "task_label": data.request_tasks[request], "rounds": 0, "accepted_sum": 0.})
        aggregate["rounds"] += 1
        aggregate["accepted_sum"] += candidate_n
        kl_rows.append(kl.tolist())
        total_kl += kl
        total_loss += float(loss)
        total_accepted += candidate_n
        actual.append(key)
    if actual != keys:
        raise ValueError("pilot scored keys differ from the frozen sample")
    rounds = len(actual)
    return {"requests": len({k["request"] for k in actual}), "rounds": rounds, "population_sha256": key_hash(actual),
            "kl_by_position": (total_kl/rounds).tolist(), "loss": total_loss/rounds,
            "accepted_length": total_accepted/rounds,
            "kl_median_by_position": [statistics.median(row[i] for row in kl_rows) for i in range(5)],
            "request_totals": [request_totals[r] for r in sorted(request_totals)]}


def step_agreement(root: Path, steps: int) -> dict[str, object]:
    result: dict[str, object] = {"informational_only": True, "interpretation": "REF/FAST evolving matched trajectories; FAST-B target agreement"}
    rows = []
    for step in range(1, steps+1):
        arms = [json.loads((root/a/f"steps/step-{step:09d}.json").read_text()) for a in ARMS]
        a, b = [torch.tensor(r["drafts"]) for r in arms[:2]]
        if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 5:
            raise ValueError("matched per-step diagnostic shape mismatch")
        rows.append({"step": step, "position1": float((a[:, 0] == b[:, 0]).float().mean()),
                     "exact_blocks": float((a == b).all(-1).float().mean()),
                     "loss_FAST_minus_REF": arms[1]["loss"]-arms[0]["loss"],
                     "target_agreement": {arm: {k: v for k, v in record.items() if k != "drafts"}
                                          for arm, record in zip(ARMS, arms)}})
    result["steps"] = rows
    return result


def verify_sample_draw(spec: Spec, manifest: Path, k2: dict[str, Path]) -> dict[str, D2Blocks]:
    """Recompute the registered draw without modifying the frozen plan.

    Verify both populations even for a single-split evaluation job, before any
    checkpoint/model load. A self-consistently rehashed alternative is refused.
    """
    if sha256(manifest) != spec["manifest_sha256"]:
        raise ValueError("pilot evaluation manifest mismatch")
    SPLITS = population_splits(spec["sample"]["keys"])
    hashes = {s: sha256(k2[s]) for s in SPLITS}
    if hashes != spec["sample"]["k2_sha256"]:
        raise ValueError("pilot K2 sources differ from frozen sample")
    data = {split: D2Blocks(manifest, split) for split in SPLITS}
    populations = {s: [key for _, key in complete_rounds(data[s], k2[s])] for s in SPLITS}
    expected = draw_sample(spec["manifest_sha256"], populations, hashes)
    if expected != spec["sample"]:
        raise ValueError("pilot frozen sample differs from seed-20261024 population draw")
    return data


def evaluate(a: argparse.Namespace) -> None:
    from .safety import assert_gpu_lease
    assert_gpu_lease()
    spec = read_spec(a.plan, sampled=True)
    sample = spec["sample"]
    SPLITS = population_splits(sample["keys"])
    splits = SPLITS if a.split == "all" else (a.split,)
    k2 = {SPLITS[0]: a.k2_dev if SPLITS[0] == "dev" else a.k2_acap, "heldout_legal": a.k2_legal}
    data = verify_sample_draw(spec, a.manifest, k2)
    train_data = D2Blocks(a.manifest, "train")
    verify_units(spec, train_data)
    arms = {name: read_arm(a.root, name, spec, a.plan, len(train_data)) for name in ARMS}
    torch.manual_seed(SEED)
    model = load(a.ckpt, "cuda", rank=32, reference_math=True).eval()
    if model.frozen_identity != spec["frozen_identity"] or initialization(model) != spec["initialization_sha256"]:
        raise ValueError("pilot evaluator source/init mismatch")
    extras = json.loads(a.manifest.read_text()).get("synthetic_eval_windows")
    if extras is not None and not spec["synthetic"]:
        raise ValueError("original capture-window sidecars are synthetic rehearsal only")
    result: Evaluation = {"schema": "dsv41.d2.pilot-evaluation.v1" if a.split == "all" else PART_SCHEMA, "plan_sha256": sha256(a.plan),
                          "synthetic": spec["synthetic"], "evaluator": EVALUATOR, "split_roles": split_roles(spec["sample"]["keys"]), "sample_sha256": sample["sha256"],
                          "k2_sha256": {s: sha256(p) for s, p in k2.items()},
                          "baseline": {s: score(model, data[s], k2[s], sample["keys"][s], synthetic_windows=extras) for s in splits}, "arms": {},
                          "checkpoints": {name: arm["checkpoints"] for name, arm in arms.items()},
                          "step_agreement": step_agreement(a.root, arms["REF"]["step_count"])}
    if a.split != "all":
        result["split"] = a.split
    for name in ARMS:
        result["arms"][name] = {}
        for n in spec["checkpoints"]:
            record = arms[name]["checkpoints"][str(n)]
            state, actual = read_checkpoint(Path(record["path"]))
            if actual != record["sha256"]:
                raise ValueError("checkpoint changed during evaluation")
            apply_trainables(model, state["trainable"])
            result["arms"][name][str(n)] = {s: score(model, data[s], k2[s], sample["keys"][s], synthetic_windows=extras) for s in splits}
            print(json.dumps({"arm": name, "blocks": n, "metrics": result["arms"][name][str(n)]}), flush=True)
    validate_sample_evaluation(spec, result, splits)
    write_json(a.out, result)


def merge_parts(plan: Path, paths: list[Path], out: Path) -> None:
    """CPU-only join of the two independently leased reference split jobs."""
    from .pilot_verdict import finite
    spec = read_spec(plan, sampled=True)
    SPLITS = population_splits(spec["sample"]["keys"])
    parts: dict[str, Evaluation] = {}
    for path in paths:
        part = cast(Evaluation, json.loads(path.read_text()))
        split = part.get("split")
        if (part["schema"] != PART_SCHEMA or split not in SPLITS or split in parts
                or part["plan_sha256"] != sha256(plan) or part["synthetic"] != spec["synthetic"]
                or part["evaluator"] != EVALUATOR):
            raise ValueError("pilot evaluation part identity/split mismatch")
        assert split is not None
        finite(part)
        validate_sample_evaluation(spec, part, (split,))
        parts[split] = part
    if set(parts) != set(SPLITS):
        raise ValueError("pilot merge requires exactly both held-out split parts")
    first, second = [parts[s] for s in SPLITS]
    if first["checkpoints"] != second["checkpoints"] or first["step_agreement"] != second["step_agreement"]:
        raise ValueError("pilot evaluation parts disagree on checkpoints/step evidence")
    result: Evaluation = {"schema": "dsv41.d2.pilot-evaluation.v1", "plan_sha256": sha256(plan),
                          "synthetic": spec["synthetic"], "evaluator": EVALUATOR, "split_roles": split_roles(spec["sample"]["keys"]),
                          "sample_sha256": spec["sample"]["sha256"], "k2_sha256": spec["sample"]["k2_sha256"],
                          "baseline": {s: parts[s]["baseline"][s] for s in SPLITS},
                          "arms": {a: {str(n): {s: parts[s]["arms"][a][str(n)][s] for s in SPLITS}
                                       for n in spec["checkpoints"]} for a in ARMS},
                          "checkpoints": first["checkpoints"], "step_agreement": first["step_agreement"]}
    validate_sample_evaluation(spec, result, SPLITS)
    write_json(out, result)


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    for n in ("plan", "out"):
        p.add_argument("--"+n, required=True, type=Path)
    for n in ("manifest", "root", "k2-acap", "k2-dev", "k2-legal"):
        p.add_argument("--"+n, type=Path)
    p.add_argument("--ckpt")
    p.add_argument("--split", choices=(*SPLITS, "dev", "all"), default="all")
    p.add_argument("--merge-parts", nargs=2, type=Path, help="CPU-only merge of dev and legal split outputs")
    a = p.parse_args()
    torch.set_num_threads(8)
    if a.merge_parts:
        merge_parts(a.plan, a.merge_parts, a.out)
    else:
        if any(getattr(a, n) is None for n in ("manifest", "root", "k2_legal", "ckpt")):
            p.error("evaluation requires --manifest, --root, --k2-dev, --k2-legal and --ckpt")
        if a.k2_dev is None and a.k2_acap is None:
            p.error("evaluation requires --k2-dev (historical rehearsal: --k2-acap)")
        evaluate(a)



if __name__ == "__main__":
    main()
