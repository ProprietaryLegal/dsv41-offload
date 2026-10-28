from __future__ import annotations

import copy
from pathlib import Path

import pytest

from offload.dspark.d2.train import pilot
from offload.dspark.d2.train.data import D2Blocks
from offload.dspark.d2.train.pilot_units import estimate, require_matched
from offload.dspark.d2.train.tests.test_data import write_shard


class MaskedPopulation:
    def __len__(self) -> int:
        return 23000

    def supervised_tokens(self, index: int) -> int:
        return index % 5+1


def test_matched_valid_tokens_at_every_boundary_even_with_partial_tails() -> None:
    units = pilot.matched_units(MaskedPopulation(), list(pilot.CHECKPOINTS))
    require_matched(units)
    assert [u["optimizer_updates"] for u in units["REF"].values()] == [20, 40, 80, 159]
    for n, expected in units["REF"].items():
        assert expected["blocks"] == int(n)
        assert expected["valid_supervised_tokens"] < 5*int(n)
        assert expected["accumulation"] == 1
        assert len(expected["learning_rates"]) == expected["optimizer_updates"]
    altered = copy.deepcopy(units)
    altered["FAST-B"]["10000"]["valid_supervised_tokens"] += 1
    with pytest.raises(ValueError, match="units differ"):
        require_matched(altered)


def test_metadata_counts_match_real_position_masks_without_tap_cache(tmp_path: Path) -> None:
    manifest = write_shard(tmp_path, split="train", L=20)
    data = D2Blocks(manifest, "train")
    assert [data.supervised_tokens(i) for i in range(len(data))] == [int(data[i]["targets"]["position_mask"].sum()) for i in range(len(data))]


def test_complete_costs_include_all_three_arms_evaluation_replay_and_worker() -> None:
    unit = pilot.matched_units(MaskedPopulation(), list(pilot.CHECKPOINTS))["REF"]["20000"]
    value = estimate(unit, {"REF": 2.7, "FAST": 190., "FAST-B": 190.}, 2000, .505, 1800, 3600)
    assert value["training_blocks"] == 60000 and value["optimizer_updates"] == 477
    assert value["evaluation_gpu_hours"] == pytest.approx(3.64722222222)
    assert value["replay_gpu_hours"] == .5 and value["serving_gpu_hours"] == 1.
    assert value["total_gpu_hours"] == pytest.approx(7.26331578947)
    with pytest.raises(ValueError, match="including replay"):
        estimate(unit, {"REF": 2.7, "FAST": 190., "FAST-B": 190.}, 2000, .505, 0, 3600)


def test_reported_learning_rates_equal_actual_optimizer_applied_rates(tmp_path: Path) -> None:
    import torch
    from offload.dspark.d2.train.data import SyntheticBlocks
    from offload.dspark.d2.train.loop import BatchPlan, Run, train
    from offload.dspark.d2.train.pilot_units import work_units
    from offload.dspark.d2.train.tests.test_loop import model
    m = model()
    data = SyntheticBlocks(m.c, 7)
    run = Run(epochs=1, batch_size=1, chunk=32)
    schedule = BatchPlan(tuple((i,) for i in range(7)), (7,), "lr-audit")
    class Complete:
        def __len__(self) -> int:
            return 7
        def supervised_tokens(self, index: int) -> int:
            return 5
    expected = work_units(Complete(), schedule, run)
    train(m, data, run, tmp_path/"step.pt", plan=schedule)
    state = torch.load(tmp_path/"step.pt", weights_only=False)
    assert [r["applied_lr"] for r in state["curve"]] == expected["learning_rates"]
    assert state["valid_supervised_tokens"] == expected["valid_supervised_tokens"]
    assert state["optimizer_updates"] == expected["optimizer_updates"]
