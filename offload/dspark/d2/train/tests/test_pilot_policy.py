from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import cast

import pytest
import torch

from offload.dspark.d2.train import data, fast_policy, pilot, pilot_eval, pilot_verdict
from offload.dspark.d2.train.pilot import Spec
from offload.dspark.d2.train.tests.test_pilot_verdict import evidence as historical_evidence
from offload.dspark.d2.train.tests.test_g1_live import gate_fixture
from offload.dspark.d2.train.pilot_sample import draw_sample


def evidence():
    # Preserve the original numerical verdict cases; change only split keys.
    spec, ev = historical_evidence()
    def rename(value):
        if isinstance(value, dict):
            return {('dev' if k == 'acap' else k): rename(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rename(v) for v in value]
        return 'dev' if value == 'dev_acap' else value
    spec, ev = rename(spec), rename(ev)
    spec['sample']['sha256'] = pilot.digest(spec['sample']['keys'])
    ev['sample_sha256'] = spec['sample']['sha256']
    return spec, ev


class Population:
    def supervised_tokens(self, index: int) -> int:
        return 5

    def __len__(self) -> int:
        return 23000


def receipt_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Path]:
    manifest, parity, g1 = [tmp_path/n for n in ("MANIFEST.json", "parity.json", "g1.json")]
    pilot.write_json(manifest, {"synthetic": False})
    pilot.write_json(parity, {"pass": True, "samples": 50, "reference_math": True, "frozen_identity": "base"})
    g1 = gate_fixture(tmp_path, manifest)
    evidence_spec, ev = evidence()
    sample = draw_sample(data.sha256(manifest), evidence_spec["sample"]["keys"], ev["k2_sha256"])
    ev["sample_sha256"] = sample["sha256"]
    plans = {arm: pilot.schedule(23000, list(pilot.CHECKPOINTS), arm) for arm in pilot.ARMS}
    spec: Spec = {"schema": "dsv41.d2.pilot-plan.v1", "synthetic": False,
                  "matched_units": pilot.matched_units(Population(), list(pilot.CHECKPOINTS)), "manifest_sha256": data.sha256(manifest),
                  "parity_sha256": data.sha256(parity), "g1_sha256": data.sha256(g1), "source_sha256": pilot.sources(),
                  "frozen_identity": "base", "initialization_sha256": "init", "run": dict(asdict(pilot.pilot_run())),
                  "checkpoints": list(pilot.CHECKPOINTS), "order_seed": 23, "order_seed_b": 24,
                  "population_sha256": pilot.digest(plans["REF"][1]), "arm_orders": {a: plans[a][2] for a in pilot.ARMS},
                  "sample": sample}
    plan_path = tmp_path/"plan.json"
    pilot.write_json(plan_path, spec)
    ev["plan_sha256"] = data.sha256(plan_path)
    for arm in pilot.ARMS:
        schedule = plans[arm][0]
        identity = pilot.digest({"plan": spec, "arm": arm, "schedule": schedule.identity})
        root = tmp_path/arm
        (root/"latest.checkpoints").mkdir(parents=True)
        checkpoints = {}
        for point, step in zip(pilot.CHECKPOINTS, schedule.checkpoints):
            path = root/f"latest.checkpoints/step-{step:09d}.pt"
            torch.save({"step": step, "blocks": point, "batch_plan_identity": identity, "run": spec["run"],
                        "frozen_identity": "base", "reference_math": arm == "REF", "fast_training": arm != "REF",
                        "valid_supervised_tokens": point*5, "optimizer_updates": step,
                        "trainable": {}, "curve": [{"loss": 1., "applied_lr": lr} for lr in spec["matched_units"][arm][str(point)]["learning_rates"]]}, path)
            checkpoints[str(point)] = {"path": str(path), "sha256": data.sha256(path), "step": step}
        ev["checkpoints"][arm] = cast(dict[str, pilot_eval.Checkpoint], checkpoints)
        pilot.write_json(root/"arm.json", {"schema": "dsv41.d2.pilot-arm.v1", "arm": arm, "complete": True,
                     "synthetic": False, "plan_sha256": data.sha256(plan_path), "batch_plan_identity": identity,
                     "units": spec["matched_units"][arm], "step_count": len(schedule.batches), "checkpoints": checkpoints})
    evaluation_path = tmp_path/"evaluation.json"
    pilot.write_json(evaluation_path, ev)
    monkeypatch.setattr(data, "D2Blocks", lambda *_: Population())
    monkeypatch.setattr(pilot_verdict, "D2Blocks", lambda *_: Population())
    receipt = pilot_verdict.build(plan_path, evaluation_path, tmp_path, manifest, parity, g1)
    receipt_path = tmp_path/"receipt.json"
    pilot.write_json(receipt_path, receipt)
    return receipt_path, manifest, parity, g1


def test_production_recomputes_real_pilot_and_all_checkpoint_identities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt, manifest, parity, g1 = receipt_fixture(tmp_path, monkeypatch)
    assert json.loads(receipt.read_text())["rule_version"] == pilot_verdict.RULE_VERSION
    assert fast_policy.require_pilot(receipt, "base", manifest, parity, g1, 128, 8192, 32) == data.sha256(receipt)
    checkpoint = tmp_path/"FAST-B/latest.checkpoints/step-000000159.pt"
    state = torch.load(checkpoint, weights_only=False)
    state["blocks"] = 19999
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="checkpoint identity/settings/nonfinite mismatch"):
        fast_policy.require_pilot(receipt, "base", manifest, parity, g1, 128, 8192, 32)


def test_population_receipt_reports_clusters_and_task_strata_and_refuses_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt, manifest, parity, g1 = receipt_fixture(tmp_path, monkeypatch)
    raw = json.loads(receipt.read_text())
    population = raw["population"]
    assert "union-find source_docs" in population["cluster_key"]
    assert population["clusters_per_split"] == {"dev": 4, "dev_legal": 4}
    assert all("cannot establish between-family uncertainty" in c for c in population["caveats"].values())
    strata = population["task_strata"]["baseline"]["dev_legal"]
    assert strata == {task: {"requests": 10, "rounds": 50, "accepted_length": 1.} for task in ("task-0", "task-1")}
    population["clusters_per_split"]["dev_legal"] = 24
    receipt.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="thresholds or evidence identity"):
        fast_policy.require_pilot(receipt, "base", manifest, parity, g1, 128, 8192, 32)


@pytest.mark.parametrize("defect", ["missing_version", "old_version", "old_amendment6_version", "old_rule"])
def test_production_refuses_old_joint_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    receipt, manifest, parity, g1 = receipt_fixture(tmp_path, monkeypatch)
    raw = json.loads(receipt.read_text())
    if defect == "missing_version":
        del raw["rule_version"]
    elif defect == "old_version":
        raw["rule_version"] = "amendment-3-joint-v1"
    elif defect == "old_amendment6_version":
        raw["rule_version"] = "amendment-6-family-bootstrap-v1"
    else:
        raw["rule"] = "Amendment 3: all metrics retain 90% OR all metrics pass spread"
    receipt.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="old joint rule"):
        fast_policy.require_pilot(receipt, "base", manifest, parity, g1, 128, 8192, 32)


@pytest.mark.parametrize("defect", ["synthetic", "source", "manifest", "base", "batch", "chunk", "rank", "checks", "plan", "metric"])
def test_production_refuses_mismatched_pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    receipt, manifest, parity, g1 = receipt_fixture(tmp_path, monkeypatch)
    raw = json.loads(receipt.read_text())
    base, batch, chunk, rank = "base", 128, 8192, 32
    if defect == "synthetic":
        raw["synthetic"] = True
    elif defect == "source":
        raw["plan"]["source_sha256"]["d2/train/model.py"] = "foreign"
    elif defect == "manifest":
        manifest.write_text("{}")
    elif defect == "base":
        base = "foreign"
    elif defect == "batch":
        batch = 256
    elif defect == "chunk":
        chunk = 4096
    elif defect == "rank":
        rank = 64
    elif defect == "checks":
        raw["checks"] = {}
    elif defect == "plan":
        raw["evaluation"]["plan_sha256"] = "foreign"
    elif defect == "metric":
        raw["evaluation"]["arms"]["FAST"]["20000"]["dev"]["accepted_length"] = 0.
    receipt.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        fast_policy.require_pilot(receipt, base, manifest, parity, g1, batch, chunk, rank)


def test_failed_agreement_cannot_substitute_for_pilot(tmp_path: Path) -> None:
    receipt = tmp_path/"agreement.json"
    pilot.write_json(receipt, {"schema": "dsv41.d2.fast-agreement.v1", "pass": False})
    with pytest.raises(ValueError, match="PASSING real pilot"):
        fast_policy.require_pilot(receipt, "base", tmp_path, tmp_path, tmp_path, 128, 8192, 32)


@pytest.mark.parametrize("defect", ["missing_receipt_hash", "wrong_receipt_hash", "missing_plan_sample", "missing_evaluation_hash", "changed_keys"])
def test_production_requires_sample_hash_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    receipt, manifest, parity, g1 = receipt_fixture(tmp_path, monkeypatch)
    raw = json.loads(receipt.read_text())
    if defect == "missing_receipt_hash":
        del raw["sample_sha256"]
    elif defect == "wrong_receipt_hash":
        raw["sample_sha256"] = "foreign"
    elif defect == "missing_plan_sample":
        del raw["plan"]["sample"]
    elif defect == "missing_evaluation_hash":
        del raw["evaluation"]["sample_sha256"]
    else:
        raw["plan"]["sample"]["keys"]["dev"][0]["P"] = 999
    receipt.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="sample"):
        fast_policy.require_pilot(receipt, "base", manifest, parity, g1, 128, 8192, 32)
