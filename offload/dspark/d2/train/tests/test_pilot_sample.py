from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import random

import pytest

from offload.dspark.d2.train import pilot, pilot_sample, safety
from offload.dspark.d2.train.data import D2Blocks, sha256
from offload.dspark.d2.train.pilot import Spec
from offload.dspark.d2.train.tests.test_pilot_policy import Population
from offload.dspark.d2.train.pilot_sample import RoundKey, DEV_POPULATION as SPLITS, complete_rounds, draw_sample, sample_plan
from offload.dspark.d2.train.tests.test_data import write_shard


def sample_fixture(root: Path) -> tuple[Path, Path, dict[str, Path]]:
    root.mkdir(exist_ok=True)
    splits = {}
    paths = {}
    for split in SPLITS:
        manifest = write_shard(root, split=split, L=20)
        side = root/(split+".json")
        meta = json.loads(side.read_text())
        meta["sequences"][0]["seq_id"] = split
        meta["sequences"][0]["source_docs"] = [split]
        side.write_text(json.dumps(meta))
        entry = json.loads(manifest.read_text())["splits"][split]
        entry["files"][0]["sidecar_sha256"] = sha256(side)
        splits[split] = entry
        path = root/(split+".k2.jsonl")
        path.write_text(json.dumps({"seq_id": split, "drafts": {str(pos): list(range(pos+2, pos+7)) for pos in range(3, 18)}})+"\n")
        paths[split] = path
    manifest = root/"MANIFEST.json"
    manifest.write_text(json.dumps({"schema": "d2.manifest.v1", "splits": splits}))
    plans = {a: pilot.schedule(23000, list(pilot.CHECKPOINTS), a) for a in pilot.ARMS}
    spec: Spec = {"schema": "dsv41.d2.pilot-plan.v1", "synthetic": False,
                  "matched_units": pilot.matched_units(Population(), list(pilot.CHECKPOINTS)), "manifest_sha256": sha256(manifest),
                  "parity_sha256": "parity", "g1_sha256": "g1", "source_sha256": pilot.sources(),
                  "frozen_identity": "base", "initialization_sha256": "init", "run": dict(asdict(pilot.pilot_run())),
                  "checkpoints": list(pilot.CHECKPOINTS), "order_seed": 23, "order_seed_b": 24,
                  "population_sha256": pilot.digest(plans["REF"][1]), "arm_orders": {a: plans[a][2] for a in pilot.ARMS}}
    plan = root/"plan.json"
    pilot.write_json(plan, spec)
    return plan, manifest, paths


def test_fixed_seed_uniform_sample_caps_each_split_and_preserves_source_order() -> None:
    populations: dict[str, list[RoundKey]] = {
        "dev": [{"request": "a", "P": p} for p in range(2500)],
        "heldout_legal": [{"request": "l", "P": p} for p in range(37)]}
    sample = draw_sample("manifest", populations, dict.fromkeys(SPLITS, "k2"))
    assert sample == draw_sample("manifest", populations, dict.fromkeys(SPLITS, "k2"))
    expected = sorted(random.Random(20261024).sample(range(2500), 1000))
    assert [k["P"] for k in sample["keys"]["dev"]] == expected
    assert sample["keys"]["heldout_legal"] == populations["heldout_legal"]
    assert sample["sha256"] == pilot.digest(sample["keys"])
    assert sample["population_rounds"] == {"dev": 2500, "heldout_legal": 37}


def test_complete_population_is_fixed_k2_trajectory_excluding_partial_tails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, manifest, paths = sample_fixture(tmp_path)
    monkeypatch.setattr(D2Blocks, "__getitem__", lambda *_: pytest.fail("sampling must not materialize taps/model inputs"))
    for split in SPLITS:
        rounds = complete_rounds(D2Blocks(manifest, split), paths[split])
        assert rounds == [(0, {"request": split, "P": 3}), (6, {"request": split, "P": 9})]
        # P15 is on K2's trajectory but lacks a full five-token target block.
        assert all(key["P"] != 15 for _, key in rounds)


def test_sample_is_persisted_once_and_redraw_refused_before_rng(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan, manifest, paths = sample_fixture(tmp_path)
    sample_plan(plan, manifest, paths)
    after = plan.read_bytes()
    spec = pilot.read_spec(plan, sampled=True)
    assert spec["sample"]["seed"] == 20261024
    assert spec["sample"]["k2_sha256"] == {s: sha256(p) for s, p in paths.items()}
    assert spec["sample"]["keys"] == {s: [{"request": s, "P": 3}, {"request": s, "P": 9}] for s in SPLITS}
    assert plan.stat().st_mode & 0o777 == 0o600
    monkeypatch.setattr(pilot_sample, "draw_sample", lambda *_: pytest.fail("redraw reached RNG"))
    with pytest.raises(FileExistsError, match="refusing to redraw"):
        sample_plan(plan, manifest, paths)
    assert plan.read_bytes() == after


def test_sample_lock_is_nonblocking_and_missing_sample_prevents_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan, manifest, paths = sample_fixture(tmp_path)
    before = plan.read_bytes()
    with Path(str(plan)+".sample.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            sample_plan(plan, manifest, paths)
    assert plan.read_bytes() == before
    monkeypatch.setattr(safety, "assert_gpu_lease", lambda: None)
    monkeypatch.setattr(pilot, "load", lambda *a, **k: pytest.fail("training reached model load before sampling"))
    with pytest.raises(ValueError, match="pre-training sample"):
        pilot.arm(argparse.Namespace(plan=plan))


def test_foreign_manifest_and_failed_draw_leave_plan_unsampled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan, manifest, paths = sample_fixture(tmp_path)
    before = plan.read_bytes()
    foreign = tmp_path/"foreign.json"
    foreign.write_text("{}")
    with pytest.raises(ValueError, match="manifest differs"):
        sample_plan(plan, foreign, paths)
    assert plan.read_bytes() == before
    monkeypatch.setattr(pilot_sample, "draw_sample", lambda *_: (_ for _ in ()).throw(ValueError("draw failed")))
    with pytest.raises(ValueError, match="draw failed"):
        sample_plan(plan, manifest, paths)
    assert plan.read_bytes() == before


def test_sample_hash_seed_and_duplicate_keys_are_checked() -> None:
    populations: dict[str, list[RoundKey]] = {s: [{"request": s, "P": 1}, {"request": s, "P": 2}] for s in SPLITS}
    valid = draw_sample("manifest", populations, dict.fromkeys(SPLITS, "k2"))
    for defect in ("hash", "seed", "duplicate"):
        sample = json.loads(json.dumps(valid))
        if defect == "hash":
            sample["sha256"] = "foreign"
        elif defect == "seed":
            sample["seed"] = 23
        else:
            sample["keys"]["dev"][1] = sample["keys"]["dev"][0]
        with pytest.raises(ValueError):
            pilot_sample.validate_sample(sample, "manifest")
