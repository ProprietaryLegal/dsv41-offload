"""Fail-closed qualification policy for the separate fast training backend."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .data import sha256

BACKEND = "d2-packed-tensorcore-v2"
NUMERIC_SOURCES = ("model.py", "ops.py", "fast_gpu.py", "loss.py", "fast_loss.py")


def numeric_sources() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: sha256(root / name) for name in NUMERIC_SOURCES}


def tensor_groups(names: set[str]) -> dict[str, list[str]]:
    """LoRA A+B is one adapter group; every plain trainable is its own group."""
    result: dict[str, list[str]] = {}
    for name in sorted(names):
        group = name[:-2] + ".lora" if name.endswith((".a", ".b")) else name
        result.setdefault(group, []).append(name)
    return result


def require_agreement(path: Path, frozen_identity: str, tensor_names: set[str],
                      batch_size: int = 1, chunk: int = 4096, rank: int = 32) -> str:
    from .agreement import population_gates
    report = json.loads(path.read_text())
    if (report.get("schema") != "dsv41.d2.fast-agreement.v1" or report.get("pass") is not True
            or report.get("backend") != BACKEND or report.get("frozen_identity") != frozen_identity):
        raise ValueError("fast training agreement is missing, failed, or for a different backend/base")
    if report.get("batch_size") != batch_size or report.get("vocab_chunk") != chunk:
        raise ValueError("fast training batch/chunk shape was not qualified by this receipt")
    if report.get("lora_rank") != rank:
        raise ValueError("fast training LoRA rank was not qualified by this receipt")
    sources: Mapping[str, str] = report["source_sha256"]
    if any(sources.get(name) != digest for name, digest in numeric_sources().items()):
        raise ValueError("fast training numerical source changed after agreement")
    for population, minimum in (("n1", 4), ("captures", 50)):
        result = report["populations"][population]
        if (len(result["rows"]) < minimum or set(result["gradient_groups"]) != set(tensor_groups(tensor_names))
                or set(result.get("gradient_tensors", {})) != tensor_names):
            raise ValueError("fast agreement sample or trainable tensor coverage is incomplete")
        if not population_gates(result["rows"], result["gradient_groups"])["pass"]:
            raise ValueError("fast training fixed agreement thresholds failed")
        if any(float(g["cosine"]) < .999 or g.get("missing_gradient", False) or g.get("both_zero", False)
               for g in result["gradient_groups"].values()):
            raise ValueError("fast training gradient direction failed")
        if any(g.get("missing_gradient", False) or (not g.get("both_zero", False) and float(g["cosine"]) < .999)
               for g in result["gradient_tensors"].values()):
            raise ValueError("fast training individual nonzero gradient direction failed")
    return sha256(path)


def require_pilot(path: Path, frozen_identity: str, manifest: Path, parity: Path, g1: Path,
                  batch_size: int, chunk: int, rank: int) -> str:
    """Production accepts only a real outcome-qualified matched pilot.

    A failed old agreement report or synthetic rehearsal is never permission.
    Checkpoints and arm receipts must still be present and byte-identical.
    """
    from typing import cast
    from .data import D2Blocks
    from .pilot import ARMS, digest, validate_inputs, validate_spec, verify_units
    from .pilot_eval import read_arm
    from .pilot_verdict import RULE, RULE_VERSION, Receipt, outcome, decision, uncertainty, population_report
    receipt = cast(Receipt, json.loads(path.read_text()))
    if (receipt.get("schema") != "dsv41.d2.pilot-receipt.v1" or receipt.get("passed") is not True
            or receipt.get("synthetic") is not False or receipt.get("rule") != RULE
            or receipt.get("rule_version") != RULE_VERSION or receipt.get("verdict") != "PASS"):
        raise ValueError("production fast training requires a PASSING real pilot receipt; agreement/synthetic/old joint rule refused")
    spec = receipt["plan"]
    validate_spec(spec, sampled=True)
    if receipt.get("sample_sha256") != spec["sample"]["sha256"]:
        raise ValueError("production pilot receipt missing/mismatched sample hash")
    validate_inputs(spec, manifest, parity, g1)
    if spec["synthetic"] or spec["frozen_identity"] != frozen_identity or (batch_size, chunk, rank) != (128, 8192, 32):
        raise ValueError("production fast pilot base, synthetic label or matched settings mismatch")
    evaluation = receipt["evaluation"]
    checks = outcome(spec, evaluation)
    verdict, another_seed = decision(spec, evaluation, checks)
    def document_digest(value: object) -> str:
        import hashlib
        return hashlib.sha256((json.dumps(value, indent=2, allow_nan=False)+"\n").encode()).hexdigest()
    if (verdict != "PASS" or receipt["checks"] != checks
            or receipt["intervals"] != uncertainty(evaluation, spec["checkpoints"][-2:])
            or receipt.get("population") != population_report(evaluation)
            or receipt["extra_seed_eligible"] != another_seed
            or receipt["plan_sha256"] != document_digest(spec)
            or receipt["evaluation_sha256"] != document_digest(evaluation)
            or evaluation["plan_sha256"] != receipt["plan_sha256"]
            or receipt["evidence_sha256"] != digest({"plan": spec, "evaluation": evaluation, "checks": checks,
                                                    "rule_version": RULE_VERSION, "sample_sha256": spec["sample"]["sha256"]})):
        raise ValueError("pilot receipt thresholds or evidence identity mismatch")
    train_data = D2Blocks(manifest, "train")
    verify_units(spec, train_data)
    length = len(train_data)
    root = Path(evaluation["checkpoints"]["REF"]["20000"]["path"]).parents[2]
    for name in ARMS:
        arm = read_arm(root, name, spec, receipt["plan_sha256"], length)
        if arm["checkpoints"] != evaluation["checkpoints"][name]:
            raise ValueError("pilot checkpoint receipt changed")
    return sha256(path)
