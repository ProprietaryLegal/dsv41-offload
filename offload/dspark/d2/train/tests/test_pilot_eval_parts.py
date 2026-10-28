from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.dspark.d2.train import pilot, pilot_eval
from offload.dspark.d2.train.data import sha256
from offload.dspark.d2.train.pilot_eval import Checkpoint, Evaluation, Metrics
from offload.dspark.d2.train.pilot_statistics import LIVE_DEV_SPLITS as DEV_SPLITS
from offload.dspark.d2.train.pilot_sample import DEV_POPULATION as SPLITS, key_hash, sample_plan
from offload.dspark.d2.train.tests.test_pilot_sample import sample_fixture


def part_fixture(root: Path) -> tuple[Path, list[Path]]:
    plan, manifest, k2 = sample_fixture(root)
    sample_plan(plan, manifest, k2)
    spec = pilot.read_spec(plan, sampled=True)
    sample = spec["sample"]
    paths = []
    checkpoints: dict[str, dict[str, Checkpoint]] = {a: {str(n): {"path": "fixture.pt", "sha256": "checkpoint", "step": i+1}
                       for i, n in enumerate(spec["checkpoints"])} for a in pilot.ARMS}
    for split in SPLITS:
        def metrics() -> Metrics:
            return {"requests": 1, "rounds": 2, "population_sha256": key_hash(sample["keys"][split]),
                    "kl_by_position": [.1]*5, "loss": .1, "accepted_length": 1.,
                    "kl_median_by_position": [.1]*5, "request_totals": [{"request_sha256": pilot.digest(split), "rounds": 2, "family": "fixture", "task_label": "task", "accepted_sum": 2.}]}
        part: Evaluation = {"schema": pilot_eval.PART_SCHEMA, "split": split, "plan_sha256": sha256(plan),
                            "sample_sha256": sample["sha256"], "synthetic": False, "evaluator": pilot_eval.EVALUATOR, "split_roles": DEV_SPLITS,
                            "k2_sha256": sample["k2_sha256"], "baseline": {split: metrics()},
                            "arms": {a: {str(n): {split: metrics()} for n in spec["checkpoints"]} for a in pilot.ARMS},
                            "checkpoints": checkpoints, "step_agreement": {"informational_only": True}}
        path = root/(split+".part.json")
        pilot.write_json(path, part)
        paths.append(path)
    return plan, paths


def test_two_split_jobs_merge_to_one_sample_bound_aggregate(tmp_path: Path) -> None:
    plan, parts = part_fixture(tmp_path)
    output = tmp_path/"evaluation.json"
    pilot_eval.merge_parts(plan, list(reversed(parts)), output)
    result = json.loads(output.read_text())
    spec = pilot.read_spec(plan, sampled=True)
    assert result["schema"] == "dsv41.d2.pilot-evaluation.v1" and "split" not in result
    assert result["sample_sha256"] == spec["sample"]["sha256"]
    assert set(result["baseline"]) == set(SPLITS)
    assert all(set(splits) == set(SPLITS) for points in result["arms"].values() for splits in points.values())
    assert "keys" not in result  # key lists remain private to the plan
    with pytest.raises(FileExistsError):
        pilot_eval.merge_parts(plan, parts, output)


@pytest.mark.parametrize("defect", ["sample_hash", "population", "rounds", "plan", "K2", "checkpoint", "missing_checkpoint", "duplicate_split", "nan"])
def test_foreign_or_mismatched_split_pass_is_refused(tmp_path: Path, defect: str) -> None:
    plan, parts = part_fixture(tmp_path)
    raw = json.loads(parts[1].read_text())
    if defect == "sample_hash":
        raw["sample_sha256"] = "foreign"
    elif defect == "population":
        raw["arms"]["FAST"]["10000"]["heldout_legal"]["population_sha256"] = "foreign"
    elif defect == "rounds":
        raw["baseline"]["heldout_legal"]["rounds"] = 1
    elif defect == "plan":
        raw["plan_sha256"] = "foreign"
    elif defect == "K2":
        raw["k2_sha256"]["dev"] = "foreign"
    elif defect == "checkpoint":
        raw["checkpoints"]["REF"]["2500"]["sha256"] = "foreign"
    elif defect == "missing_checkpoint":
        del raw["arms"]["FAST-B"]["5000"]
    elif defect == "duplicate_split":
        raw["split"] = "dev"
    elif defect == "nan":
        raw["baseline"]["heldout_legal"]["loss"] = float("nan")
    parts[1].write_text(json.dumps(raw))
    out = tmp_path/"evaluation.json"
    with pytest.raises(ValueError):
        pilot_eval.merge_parts(plan, parts, out)
    assert not out.exists()
