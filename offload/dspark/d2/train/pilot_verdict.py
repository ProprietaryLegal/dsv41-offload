"""Amendment 6: useful acceptance gain and paired family non-inferiority."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
from typing import TypedDict, cast

from .data import D2Blocks, sha256
from .pilot import ARMS, Spec, digest, read_spec, validate_inputs, write_json, verify_units
from .pilot_statistics import CLUSTER_KEY, FEW_CLUSTERS, DEV_SPLITS, Interval, paired_interval, split_roles
from .pilot_eval import EVALUATOR, SPLITS, Evaluation, Metrics, read_arm, validate_sample_evaluation
from .pilot_sample import validate_sample

RULE_VERSION = "amendment-6ab-source-prompt-bootstrap-v2"
RULE = "Amendments 6/6a/6b: dev E[A] excluding bonus; source_docs components or empty-source prompt_sha256 clusters, then requests; bootstrap 2000 seed 20261024, 90% CI; task labels are strata; REF/FAST gain >=0.05 with lower>0; FAST-REF lower>=-0.03; FAST baseline deficit<=0.01; median per-position KL deficit<=0.01; definitive failures => FAIL; extra seed only if every failing check straddles; REF no useful gain => INCONCLUSIVE"


def uncertainty(evaluation: Evaluation, points: list[int]) -> dict[str, Interval]:
    DEV_SPLITS = split_roles(evaluation["baseline"])
    SPLITS = tuple(DEV_SPLITS)
    result: dict[str, Interval] = {}
    for point in points:
        for split in SPLITS:
            base = evaluation["baseline"][split]["request_totals"]
            ref = evaluation["arms"]["REF"][str(point)][split]["request_totals"]
            fast = evaluation["arms"]["FAST"][str(point)][split]["request_totals"]
            prefix = f"{point}/{DEV_SPLITS[split]}"
            result[prefix+"/REF_gain"] = paired_interval(ref, base)
            result[prefix+"/FAST_gain"] = paired_interval(fast, base)
            result[prefix+"/FAST_minus_REF"] = paired_interval(fast, ref)
    return result


def status(checks: dict[str, bool]) -> str:
    if any(not value for key, value in checks.items() if key.endswith("REF_useful_gain")):
        return "INCONCLUSIVE"
    return "PASS" if all(checks.values()) else "FAIL"


def population_report(evaluation: Evaluation) -> dict[str, object]:
    """Report task strata separately from the sampled source/prompt clusters."""
    def strata(metrics: Metrics) -> dict[str, dict[str, float | int]]:
        groups: dict[str, list[tuple[int, float]]] = {}
        for row in metrics["request_totals"]:
            groups.setdefault(row["task_label"], []).append((row["rounds"], row["accepted_sum"]))
        return {task: {"requests": len(rows), "rounds": sum(n for n, _ in rows),
                       "accepted_length": math.fsum(a for _, a in rows)/sum(n for n, _ in rows)}
                for task, rows in sorted(groups.items())}
    DEV_SPLITS = split_roles(evaluation["baseline"])
    counts = {DEV_SPLITS[s]: len({r["family"] for r in m["request_totals"]})
              for s, m in evaluation["baseline"].items()}
    return {"cluster_key": CLUSTER_KEY, "clusters_per_split": counts,
            "caveats": {s: FEW_CLUSTERS for s, n in counts.items() if n < 5},
            "task_strata": {"baseline": {DEV_SPLITS[s]: strata(m) for s, m in evaluation["baseline"].items()},
                            "arms": {a: {p: {DEV_SPLITS[s]: strata(m) for s, m in splits.items()}
                                         for p, splits in points.items()} for a, points in evaluation["arms"].items()}}}



class Receipt(TypedDict):
    schema: str
    passed: bool
    verdict: str
    intervals: dict[str, Interval]
    extra_seed_eligible: bool
    synthetic: bool
    rule: str
    rule_version: str
    plan: Spec
    plan_sha256: str
    evaluation: Evaluation
    evaluation_sha256: str
    checks: dict[str, bool]
    evidence_sha256: str
    sample_sha256: str
    population: dict[str, object]


def finite(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)
    elif isinstance(value, (float, Decimal)) and not math.isfinite(value):
        raise ValueError("nonfinite pilot evidence")


def number(value: float) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("nonfinite pilot metric")
    return result


def validate_metrics(m: Metrics, baseline: Metrics) -> None:
    if (m["rounds"] <= 0 or m["requests"] <= 0 or len(m["kl_by_position"]) != 5
            or m["rounds"] != baseline["rounds"] or m["requests"] != baseline["requests"]
            or m["population_sha256"] != baseline["population_sha256"]
            or not Decimal(0) <= number(m["accepted_length"]) <= Decimal(5)):
        raise ValueError("pilot held-out population or metric mismatch")
    finite(m)
    rows = m["request_totals"]
    if (len(m["kl_median_by_position"]) != 5 or len(rows) != m["requests"]
            or sum(r["rounds"] for r in rows) != m["rounds"]
            or abs(math.fsum(r["accepted_sum"] for r in rows)/m["rounds"]-m["accepted_length"]) > 1e-12):
        raise ValueError("pilot request aggregates do not reproduce E[A]")
    if (len({r["request_sha256"] for r in rows}) != len(rows)
            or any(not r["family"] or not r["task_label"] or type(r["rounds"]) is not int or r["rounds"] < 1
                   or not 0 <= r["accepted_sum"] <= 5*r["rounds"] for r in rows)):
        raise ValueError("invalid pilot request aggregates/family coverage")


def outcome(spec: Spec, evaluation: Evaluation) -> dict[str, bool]:
    finite(evaluation)
    DEV_SPLITS = split_roles(evaluation["baseline"])
    SPLITS = tuple(DEV_SPLITS)
    if (evaluation["schema"] != "dsv41.d2.pilot-evaluation.v1" or evaluation["synthetic"] != spec["synthetic"]
            or evaluation["evaluator"] != EVALUATOR
            or set(evaluation["baseline"]) != set(SPLITS) or set(evaluation["arms"]) != set(ARMS)
            or set(evaluation["checkpoints"]) != set(ARMS) or set(evaluation["k2_sha256"]) != set(SPLITS)):
        raise ValueError("missing/mismatched pilot arms, splits or evaluator")
    if "sample" not in spec:
        raise ValueError("pilot verdict requires its once-only pre-training sample")
    validate_sample(spec["sample"], spec["manifest_sha256"])
    validate_sample_evaluation(spec, evaluation, SPLITS)
    points = {str(n) for n in spec["checkpoints"]}
    for name in ARMS:
        if set(evaluation["arms"][name]) != points or set(evaluation["checkpoints"][name]) != points:
            raise ValueError("missing pilot checkpoint")
        for n in points:
            if set(evaluation["arms"][name][n]) != set(SPLITS):
                raise ValueError("missing pilot held-out split")
            for split in SPLITS:
                validate_metrics(evaluation["baseline"][split], evaluation["baseline"][split])
                validate_metrics(evaluation["arms"][name][n][split], evaluation["baseline"][split])
    checks = {}
    intervals = uncertainty(evaluation, spec["checkpoints"][-2:])
    for point in spec["checkpoints"][-2:]:
        for split in SPLITS:
            prefix = f"{point}/{DEV_SPLITS[split]}"
            base = evaluation["baseline"][split]
            ref, fast = [evaluation["arms"][name][str(point)][split] for name in ("REF", "FAST")]
            for arm in ("REF", "FAST"):
                ci = intervals[prefix+f"/{arm}_gain"]
                checks[prefix+f"/{arm}_useful_gain"] = number(ci["difference"]) >= Decimal(".05") and ci["lower"] > 0
            checks[prefix+"/non_inferiority"] = number(intervals[prefix+"/FAST_minus_REF"]["lower"]) >= Decimal("-.03")
            checks[prefix+"/no_baseline_regression"] = number(fast["accepted_length"])-number(base["accepted_length"]) >= Decimal("-.01")
            checks[prefix+"/KL_median"] = all(number(f)-number(r) <= Decimal(".01") for f, r in
                zip(fast["kl_median_by_position"], ref["kl_median_by_position"]))

    return checks


def decision(spec: Spec, evaluation: Evaluation, checks: dict[str, bool]) -> tuple[str, bool]:
    intervals = uncertainty(evaluation, spec["checkpoints"][-2:])
    crossings: list[bool] = []
    definitive_failure = False
    for key, passed in checks.items():
        if passed:
            continue
        crossing = False
        if key.endswith("/non_inferiority"):
            ci = intervals[key.rsplit("/", 1)[0]+"/FAST_minus_REF"]
            crossing = ci["lower"] < -.03 < ci["upper"]
        elif key.endswith("_useful_gain"):
            ci = intervals[key.removesuffix("_useful_gain")+"_gain"]
            crossing = any(ci["lower"] < bound < ci["upper"] for bound in (0., .05))
        crossings.append(crossing)
        # Amendment 6's explicit exception: if REF cannot demonstrate useful
        # adaptation in this cell, missing useful gain there is INCONCLUSIVE.
        # It cannot excuse NI, KL, baseline, or another cell's definitive failure.
        unproved_adaptation = key.endswith("_useful_gain") and not checks[
            key.rsplit("/", 1)[0]+"/REF_useful_gain"]
        definitive_failure |= not crossing and not unproved_adaptation
    if definitive_failure:
        return "FAIL", False
    if crossings and all(crossings):
        return "INCONCLUSIVE", True
    return status(checks), False


def build(plan: Path, evaluation_path: Path, root: Path, manifest: Path, parity: Path, g1: Path) -> Receipt:
    spec = read_spec(plan, sampled=True)
    validate_inputs(spec, manifest, parity, g1)
    evaluation = cast(Evaluation, json.loads(evaluation_path.read_text()))
    if evaluation["plan_sha256"] != sha256(plan):
        raise ValueError("pilot evaluation receipt has a mismatched plan")
    data = D2Blocks(manifest, "train")
    verify_units(spec, data)
    for name in ARMS:
        arm = read_arm(root, name, spec, plan, len(data))
        if evaluation["checkpoints"].get(name) != arm["checkpoints"]:
            raise ValueError("evaluated checkpoint differs from arm receipt")
    checks = outcome(spec, evaluation)
    verdict, another_seed = decision(spec, evaluation, checks)
    evidence = {"plan": spec, "evaluation": evaluation, "checks": checks, "rule_version": RULE_VERSION, "sample_sha256": spec["sample"]["sha256"]}
    return {"schema": "dsv41.d2.pilot-receipt.v1", "passed": verdict == "PASS", "verdict": verdict,
            "intervals": uncertainty(evaluation, spec["checkpoints"][-2:]),
            "extra_seed_eligible": another_seed, "synthetic": spec["synthetic"],
            "population": population_report(evaluation),
            "rule": RULE, "rule_version": RULE_VERSION, "plan": spec, "plan_sha256": sha256(plan), "evaluation": evaluation,
            "evaluation_sha256": sha256(evaluation_path), "checks": checks, "evidence_sha256": digest(evidence),
            "sample_sha256": spec["sample"]["sha256"]}


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    for n in ("plan", "evaluation", "root", "manifest", "parity-receipt", "g1-receipt", "out"):
        p.add_argument("--"+n, required=True, type=Path)
    a = p.parse_args()
    result = build(a.plan, a.evaluation, a.root, a.manifest, a.parity_receipt, a.g1_receipt)
    write_json(a.out, result)
    print(json.dumps({"passed": result["passed"], "synthetic": result["synthetic"], "checks": result["checks"]}), flush=True)


if __name__ == "__main__":
    main()
