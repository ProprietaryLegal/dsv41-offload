"""S24 item 2: matched work units across REF / FAST / FAST-B (CPU, synthetic metadata only)."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import cast

import pytest
import torch

from offload.dspark.d2.train import pilot
from offload.dspark.d2.train.loop import BatchPlan
from offload.dspark.d2.train.pilot_units import require_matched, work_units


class Tails:
    """Irregular masked tails: a block's supervised count depends on a hash of its index (1..5)."""
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def supervised_tokens(self, index: int) -> int:
        if not 0 <= index < self.n:
            raise IndexError(index)
        return 5 if (index*2654435761) % 7 else (index*40503) % 5+1


@pytest.mark.parametrize("length", [20000, 23017, 2_400_000])
def test_s24_fast_b_interval_populations_and_units_equal(length: int) -> None:
    points = list(pilot.CHECKPOINTS)
    plans = {arm: pilot.schedule(length, points, arm) for arm in pilot.ARMS}
    ref, fast, fastb = (plans[a][0] for a in pilot.ARMS)
    assert ref.batches == fast.batches
    assert ref.batches != fastb.batches                      # FAST-B really reorders
    assert ref.checkpoints == fast.checkpoints == fastb.checkpoints == (20, 40, 80, 159)
    def seen(plan: BatchPlan, upto: int) -> list[int]:
        return sorted(i for b in plan.batches[:upto] for i in b)
    for step in ref.checkpoints:
        assert seen(ref, step) == seen(fastb, step)          # identical checkpoint populations
    for plan in (ref, fastb):
        assert all(1 <= len(b) <= 128 for b in plan.batches)
        flat = [i for b in plan.batches for i in b]
        assert len(flat) == len(set(flat)) == 20000
    data = Tails(length)
    units = pilot.matched_units(data, points)
    require_matched(units)
    for n, step in zip(points, ref.checkpoints):
        exact = sum(data.supervised_tokens(i) for b in ref.batches[:step] for i in b)
        u = units["FAST-B"][str(n)]
        assert (u["blocks"], u["valid_supervised_tokens"], u["optimizer_updates"]) == (n, exact, step)
        assert u["accumulation"] == 1 and u["batch_ceiling"] == 128
    # An unmatched FAST-B (the pre-R3 whole-population permutation) would have broken token equality at 2.5K.
    old = torch.randperm(20000, generator=torch.Generator().manual_seed(24)).tolist()
    population = plans["REF"][1]
    old_first = sorted(population[i] for i in old[:2500])
    assert old_first != seen(ref, 20)


def test_s24_learning_rates_equal_a_real_adamw_lambdalr_at_159_updates() -> None:
    """Independent: drive torch's own AdamW + LambdaLR for the real 159-update cosine and compare exactly."""
    plan, _, _ = pilot.schedule(23017, list(pilot.CHECKPOINTS), "REF")
    run = pilot.pilot_run()
    units = work_units(Tails(23017), plan, run)
    p = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([p], lr=run.lr, weight_decay=run.weight_decay)
    total = len(plan.batches)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: .5*(1+math.cos(math.pi*min(s, total)/total)))
    applied = []
    for _ in range(total):
        applied.append(float(opt.param_groups[0]["lr"]))
        p.grad = torch.ones(3)
        opt.step()
        sched.step()
    assert applied == units["learning_rates"]
    assert applied[0] == run.lr and len(applied) == 159 and applied[-1] > 0


def test_s24_unit_tamper_refused_and_dataset_recheck() -> None:
    points = list(pilot.CHECKPOINTS)
    units = pilot.matched_units(Tails(23017), points)
    for field, delta in (("valid_supervised_tokens", 1), ("optimizer_updates", 1), ("blocks", -1)):
        bad = copy.deepcopy(units)
        bad["FAST"]["10000"][field] += delta  # type: ignore[literal-required]
        with pytest.raises(ValueError, match="units differ"):
            require_matched(bad)
    bad = copy.deepcopy(units)
    bad["FAST-B"]["20000"]["learning_rates"][-1] *= 1.0000001
    with pytest.raises(ValueError, match="units differ"):
        require_matched(bad)
    # All arms consistently altered pass require_matched but fail verify_units against the real metadata.
    consistent = copy.deepcopy(units)
    for arm in pilot.ARMS:
        consistent[arm]["20000"]["valid_supervised_tokens"] -= 1
    require_matched(consistent)
    spec = {"matched_units": consistent, "checkpoints": points}
    with pytest.raises(ValueError, match="differs from actual dataset"):
        pilot.verify_units(cast(pilot.Spec, spec), Tails(23017))
    pilot.verify_units(cast(pilot.Spec, {"matched_units": units, "checkpoints": points}), Tails(23017))


def test_s24_resumed_training_keeps_applied_lr_and_counters(tmp_path: Path) -> None:
    """Interrupt after 3 of 7 updates, resume, and compare counters with the plan's units."""
    from offload.dspark.d2.train.data import SyntheticBlocks
    from offload.dspark.d2.train.loop import BatchPlan, Run, train
    from offload.dspark.d2.train.tests.test_loop import model

    class Complete:
        def __len__(self) -> int:
            return 7

        def supervised_tokens(self, index: int) -> int:
            return 5
    run = Run(epochs=1, batch_size=2, chunk=32)
    plan = BatchPlan(((0, 1), (2,), (3, 4), (5,), (6, 0), (1, 2), (3,)), (2, 7), "resume-audit")
    expected = work_units(Complete(), plan, run)
    m = model()
    data = SyntheticBlocks(m.c, 7)
    train(m, data, run, tmp_path/"r.pt", plan=plan, max_steps=3)
    m2 = model()
    train(m2, data, run, tmp_path/"r.pt", plan=plan, resume=True)
    state = torch.load(tmp_path/"r.pt", weights_only=False)
    assert [r["applied_lr"] for r in state["curve"]] == expected["learning_rates"]
    assert state["optimizer_updates"] == 7 == expected["optimizer_updates"]
    assert state["valid_supervised_tokens"] == expected["valid_supervised_tokens"] == 55
