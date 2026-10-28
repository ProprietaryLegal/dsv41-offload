from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from offload.dspark.d2.train import pilot
from offload.dspark.d2.train.data import SyntheticBlocks
from offload.dspark.d2.train.loop import BatchPlan, Run, train
from offload.dspark.d2.train.tests.test_loop import model


def test_matched_population_order_and_exact_checkpoints() -> None:
    plans = {a: pilot.schedule(23000, list(pilot.CHECKPOINTS), a) for a in pilot.ARMS}
    ref, fast, other = [plans[a] for a in pilot.ARMS]
    assert ref == fast
    assert ref[1] == other[1]
    assert sorted(i for b in ref[0].batches for i in b) == sorted(i for b in other[0].batches for i in b)
    assert ref[2] != other[2]
    for arm in plans.values():
        assert [sum(map(len, arm[0].batches[:s])) for s in arm[0].checkpoints] == list(pilot.CHECKPOINTS)
        assert max(map(len, arm[0].batches)) == 128
        assert len(set(i for b in arm[0].batches for i in b)) == 20000
    with pytest.raises(ValueError, match="enough unique"):
        pilot.schedule(19999, list(pilot.CHECKPOINTS), "REF")
    assert asdict(pilot.pilot_run()) == asdict(Run(epochs=1, chunk=8192, batch_size=128))


def test_pilot_boundary_resume_preserves_schedule_and_observes_every_step(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    data = SyntheticBlocks(model().c, count=5)
    plan = BatchPlan(((4, 2), (0,), (1, 3)), (2, 3), "fixed")
    run = Run(epochs=1, batch_size=2, chunk=32)
    full, resumed = tmp_path/"full.pt", tmp_path/"resumed.pt"
    seen: list[int] = []
    train(model(), data, run, full, plan=plan, observer=lambda m, b, h, hx, r: seen.append(int(r["step"])))
    train(model(), data, run, resumed, plan=plan, max_steps=2)
    assert torch.load(resumed, weights_only=False)["blocks"] == 3
    train(model(), data, run, resumed, plan=plan, resume=True)
    a, b = [torch.load(p, weights_only=False) for p in (full, resumed)]
    assert seen == [1, 2, 3] and a["blocks"] == b["blocks"] == 5
    for n in a["trainable"]:
        assert torch.equal(a["trainable"][n], b["trainable"][n])
    with pytest.raises(ValueError, match="batch plan changed"):
        train(model(), data, run, resumed, plan=replace(plan, identity="changed"), resume=True)
    assert (tmp_path/"full.checkpoints/step-000000002.pt").exists()


def test_pilot_refuses_nonfinite_loss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from offload.dspark.d2.train import loop
    m = model()
    monkeypatch.setattr(loop, "distillation_loss", lambda *a, **k: (next(m.parameters()).sum()*float("nan"), {}))
    with pytest.raises(FloatingPointError, match="nonfinite pilot loss"):
        train(m, SyntheticBlocks(m.c, 1), Run(), tmp_path/"bad.pt", plan=BatchPlan(((0,),), (1,), "bad"))


def test_pilot_second_solo_owns_its_own_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fcntl
    from offload.dspark.d2.train.safety import SOLO_B, assert_gpu_lease
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("D2_PILOT_LEASE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", SOLO_B)
    path = tmp_path/"dsv41-offload/locks/gpu-bc62df2a.lock"
    path.parent.mkdir(parents=True)
    with path.open("w") as f:
        monkeypatch.setenv("D2_LOCK_FD", str(f.fileno()))
        with pytest.raises(RuntimeError, match="not held"):
            assert_gpu_lease()
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_gpu_lease()
        monkeypatch.delenv("D2_PILOT_LEASE")
        with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
            assert_gpu_lease()
