from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import pytest
import torch

from offload.dspark.d2.train import pilot, pilot_eval, safety
from offload.dspark.d2.train.pilot import digest
from offload.dspark.d2.train.data import Block, D2Blocks, SyntheticBlocks, sha256
from offload.dspark.d2.train.pilot_sample import DEV_POPULATION as SPLITS, RoundKey, draw_sample, sample_plan, validate_sample
from offload.dspark.d2.train.evaluate import bucket_kl, match_length
from offload.dspark.d2.train.tests.test_loop import model
from offload.dspark.d2.train.tests.test_pilot_sample import sample_fixture


def test_evaluator_recomputes_actual_complete_round_populations(tmp_path: Path) -> None:
    plan, manifest, paths = sample_fixture(tmp_path)
    sample_plan(plan, manifest, paths)
    before = plan.read_bytes()
    data = pilot_eval.verify_sample_draw(pilot.read_spec(plan, sampled=True), manifest, paths)
    assert set(data) == set(SPLITS)
    assert plan.read_bytes() == before


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("defect", ["alternative_keys", "population_count", "population_hash"])
def test_evaluator_refuses_rehashed_sample_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str, defect: str,
) -> None:
    plan, manifest, paths = sample_fixture(tmp_path)
    populations: dict[str, list[RoundKey]] = {
        s: [{"request": s, "P": p} for p in range(1100)] for s in SPLITS}
    # Large CPU-only population fixture, without allocating tapes/model inputs.
    def rounds(data: D2Blocks, path: Path) -> list[tuple[int, RoundKey]]:
        return list(enumerate(populations[path.name.split(".")[0]]))
    monkeypatch.setattr(pilot_eval, "complete_rounds", rounds)
    spec = pilot.read_spec(plan)
    spec["sample"] = draw_sample(sha256(manifest), populations, {s: sha256(paths[s]) for s in SPLITS})
    pilot_eval.verify_sample_draw(spec, manifest, paths)
    sample = spec["sample"]
    if defect == "alternative_keys":
        keys = sample["keys"][split]
        chosen = {k["P"] for k in keys}
        # Still 1000 valid unique keys in source order, with a matching hash.
        replacement = next(k for k in populations[split] if k["P"] not in chosen)
        keys[:] = sorted(keys[1:]+[replacement], key=lambda k: k["P"])
        sample["sha256"] = digest(sample["keys"])
    elif defect == "population_count":
        sample["population_rounds"][split] += 1
    else:
        sample["population_sha256"][split] = "self-consistent-but-foreign"
    validate_sample(sample, spec["manifest_sha256"])
    plan.write_text(json.dumps(spec))
    before = plan.read_bytes()
    monkeypatch.setattr(safety, "assert_gpu_lease", lambda: None)
    monkeypatch.setattr(pilot_eval, "load", lambda *a, **k: pytest.fail("bad sample reached model load"))
    # Even an dev-only job must reject a corrupted heldout_legal draw.
    with pytest.raises(ValueError, match="seed-20261024 population draw"):
        pilot_eval.evaluate(argparse.Namespace(plan=plan, manifest=manifest, k2_dev=paths["dev"],
                                              k2_legal=paths["heldout_legal"], split="dev"))
    assert plan.read_bytes() == before


class TinyHeldout(SyntheticBlocks):
    request_families = {"one": "fixture"}
    request_tasks = {"one": "task"}

    def __getitem__(self, index: int) -> Block:
        block = super().__getitem__(index)
        block["request"] = "one"
        return block


def test_reference_pilot_scorer_uses_g2_kl_and_fixed_k2_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(2)
    m = model().eval()
    m.reference_math = True  # tiny CPU fixture; production load enforces the real reference path
    data = TinyHeldout(m.c, 8)
    keys: list[RoundKey] = [{"request": "one", "P": 12}, {"request": "one", "P": 18}]
    monkeypatch.setattr(pilot_eval, "complete_rounds", lambda *_: [(0, keys[0]), (6, keys[1])])
    result = pilot_eval.score(m, cast(D2Blocks, data), tmp_path/"k2", keys, chunk=32)
    expected_kl, expected_accepted = [], []
    for i in (0, 6):
        block = data[i]
        with torch.no_grad():
            h, hx = m.hidden_from(block["inputs"])
            targets = block["targets"]
            logits = m.logit_chunk(h, targets["previous"], 0, m.c.vocab)[0]
            expected_kl.append(bucket_kl(logits, targets["top_ids"][0], targets["top_logp"][0], targets["tail_mass"][0]))
            picks, _ = m.greedy(h, hx, block["inputs"]["anchor"])
            accepted = match_length(picks[0].tolist(), targets["labels"][0].tolist())
            assert accepted is not None
            expected_accepted.append(accepted)
    assert result["rounds"] == 2 and result["requests"] == 1
    assert result["population_sha256"] == digest([("one", 12), ("one", 18)])
    torch.testing.assert_close(torch.tensor(result["kl_by_position"], dtype=torch.float64), torch.stack(expected_kl).mean(0))
    assert result["accepted_length"] == sum(expected_accepted)/2
    assert result["request_totals"][0]["family"] == "fixture"
    assert result["request_totals"][0]["task_label"] == "task"
    assert all(p.grad is None for p in m.parameters())
    assert set(result) == {"requests", "rounds", "population_sha256", "kl_by_position", "loss", "accepted_length", "kl_median_by_position", "request_totals"}
    m.reference_math = False
    with pytest.raises(ValueError, match="reference math"):
        pilot_eval.score(m, cast(D2Blocks, data), tmp_path/"k2", keys)


def test_only_selected_rounds_are_scored_on_all_thirteen_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(2)
    m = model().eval()
    m.reference_math = True
    data = TinyHeldout(m.c, 8)
    first: RoundKey = {"request": "one", "P": 12}
    second: RoundKey = {"request": "one", "P": 18}
    monkeypatch.setattr(pilot_eval, "complete_rounds", lambda *_: [(0, first), (6, second)])
    calls: list[int] = []
    hook = m.main_proj.register_forward_hook(lambda *_: calls.append(1))
    try:
        for _ in range(13):
            metrics = pilot_eval.score(m, cast(D2Blocks, data), tmp_path/"k2", [second], chunk=32)
            assert metrics["rounds"] == 1
            assert metrics["population_sha256"] == digest([("one", 18)])
    finally:
        hook.remove()
    assert len(calls) == 13  # one selected model forward per pass, never the excluded round


@pytest.mark.parametrize("defect", ["foreign", "duplicate", "reordered", "partial", "changed_input"])
def test_scored_key_mismatch_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    m = model().eval()
    m.reference_math = True
    data = TinyHeldout(m.c, 8)
    first: RoundKey = {"request": "one", "P": 12}
    second: RoundKey = {"request": "one", "P": 18}
    monkeypatch.setattr(pilot_eval, "complete_rounds", lambda *_: [(0, first), (6, second)])
    selected = [first, second]
    if defect == "foreign":
        selected = [{"request": "one", "P": 13}]
    elif defect == "duplicate":
        selected = [first, first]
    elif defect == "reordered":
        selected = [second, first]
    else:
        original = TinyHeldout.__getitem__
        def altered(self: TinyHeldout, index: int) -> Block:
            block = original(self, index)
            if defect == "partial":
                block["targets"]["position_mask"][0, -1] = False
            else:
                block["inputs"]["P"][0] += 1
            return block
        monkeypatch.setattr(TinyHeldout, "__getitem__", altered)
    hook = m.main_proj.register_forward_hook(lambda *_: pytest.fail("bad population reached model forward"))
    try:
        with pytest.raises(ValueError, match="scored keys differ|incomplete targets"):
            pilot_eval.score(m, cast(D2Blocks, data), tmp_path/"k2", selected, chunk=32)
    finally:
        hook.remove()
