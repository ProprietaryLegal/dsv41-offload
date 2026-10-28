from __future__ import annotations

from pathlib import Path
import math
import random
import signal
import threading

import torch
import pytest

from offload.dspark.d2.train.data import Block, SyntheticBlocks
from offload.dspark.d2.train.data import sha256
from offload.dspark.d2.train.loop import Run, train
from offload.dspark.d2.train import loop
from offload.dspark.d2.train.model import Drafter
from offload.dspark.d2.train.synthetic import SyntheticSource, tiny_config


def model() -> Drafter:
    torch.manual_seed(31)
    c = tiny_config()
    return Drafter(SyntheticSource(c), c)


@pytest.mark.parametrize("batch_size", (1, 2))
def test_exact_resume_optimizer_rng_schedule_data_order(tmp_path: Path, batch_size: int) -> None:
    torch.set_num_threads(2)
    data = SyntheticBlocks(tiny_config(), count=3)
    run = Run(epochs=2, chunk=32, batch_size=batch_size)
    a, b = model(), model()
    pa, pb = tmp_path/"full.pt", tmp_path/"resume.pt"
    train(a, data, run, pa)
    train(b, data, run, pb, max_steps=2)
    archive = tmp_path/"resume.checkpoints/step-000000002.pt"
    preserved = sha256(archive)
    resumed = model()
    train(resumed, data, run, pb, resume=True)
    assert sha256(archive) == preserved
    assert torch.load(archive, weights_only=False)["step"] == 2
    sa = torch.load(pa, weights_only=False)
    sb = torch.load(pb, weights_only=False)
    assert sa["step"] == sb["step"] == 2*math.ceil(3/batch_size)
    for n in sa["trainable"]:
        assert torch.equal(sa["trainable"][n], sb["trainable"][n]), n
    for i, oa in sa["optimizer"]["state"].items():
        for n, t in oa.items():
            assert torch.equal(t, sb["optimizer"]["state"][i][n])
    assert sa["scheduler"] == sb["scheduler"]
    assert [r["loss"] for r in sa["curve"]] == [r["loss"] for r in sb["curve"]]
    assert torch.equal(sa["torch_rng"], sb["torch_rng"])
    assert pa.stat().st_mode & 0o777 == 0o600


def test_stop_writes_boundary_and_changed_contract_refused(tmp_path: Path) -> None:
    stop = threading.Event()
    stop.set()
    p = tmp_path/"term.pt"
    data = SyntheticBlocks(tiny_config(), count=2)
    out = train(model(), data, Run(epochs=1), p, stop)
    assert out["step"] == 0 and p.is_file()
    with pytest.raises(ValueError, match="resume contract changed"):
        train(model(), data, Run(epochs=2), p, resume=True)


def test_mid_forward_cancellation_discards_partial_step(tmp_path: Path) -> None:
    stop = threading.Event()
    m = model()
    original = {n: p.detach().clone() for n, p in m.named_parameters()}
    handle = m.main_proj.register_forward_hook(lambda *_: stop.set())
    path = tmp_path/"cancelled.pt"
    result = train(m, SyntheticBlocks(m.c, count=2), Run(epochs=1), path, stop)
    handle.remove()
    assert result["step"] == 0
    state = torch.load(path, weights_only=False)
    for n, p in state["trainable"].items():
        assert torch.equal(p, original[n])
    # The cancelled scope must not poison subsequent independent inference.
    stop.clear()
    m.hidden_from(SyntheticBlocks(m.c)[0]["inputs"])


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("cancel_inside_greedy", [True, False])
def test_reference_observer_signal_saves_and_resumes_boundary(
    tmp_path: Path, signum: int, cancel_inside_greedy: bool,
) -> None:
    def reference() -> Drafter:
        torch.manual_seed(31)
        c = tiny_config()
        return Drafter(SyntheticSource(c), c, acc=torch.float64, reference_math=True)

    stop = threading.Event()
    seen: list[int] = []
    def observe(m: Drafter, block: Block, h: torch.Tensor, hx: torch.Tensor, metrics: dict[str, float]) -> None:
        seen.append(int(metrics["step"]))
        if metrics["step"] == 2:
            # Both RNG streams must roll back with the uncommitted step.
            torch.rand(3)
            random.random()
            signal.raise_signal(signum)
            if cancel_inside_greedy:
                m.greedy(h, hx, block["inputs"]["anchor"], chunk=32)

    data = SyntheticBlocks(tiny_config(), count=3)
    run = Run(epochs=1, chunk=32)
    full, boundary, interrupted = [tmp_path/n for n in ("full.pt", "boundary.pt", "interrupted.pt")]
    train(reference(), data, run, full)
    train(reference(), data, run, boundary, max_steps=1)
    m = reference()
    old = signal.signal(signum, lambda *_: stop.set())
    try:
        result = train(m, data, run, interrupted, stop, observer=observe)
    finally:
        signal.signal(signum, old)
    assert seen == [1, 2]
    assert result["step"] == 1 and result["complete"] is False
    assert all(p.grad is None for p in m.parameters())
    saved = torch.load(interrupted, weights_only=False)
    expected = torch.load(boundary, weights_only=False)
    assert saved["step"] == expected["step"] == 1
    assert torch.equal(saved["torch_rng"], expected["torch_rng"])
    assert saved["random_rng"] == expected["random_rng"]
    for n, p in saved["trainable"].items():
        assert torch.equal(p, expected["trainable"][n]), n
    train(reference(), data, run, interrupted, resume=True)
    saved, expected = [torch.load(p, weights_only=False) for p in (interrupted, full)]
    for n, p in saved["trainable"].items():
        assert torch.equal(p, expected["trainable"][n]), n
    for i, state in saved["optimizer"]["state"].items():
        for n, t in state.items():
            assert torch.equal(t, expected["optimizer"]["state"][i][n])
    assert saved["scheduler"] == expected["scheduler"]
    assert [r["loss"] for r in saved["curve"]] == [r["loss"] for r in expected["curve"]]


def test_confidence_stage_refuses_zero_labelled_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loop, "confidence_loss", lambda *a, **k: (None, 0))
    output = tmp_path/"confidence.pt"
    with pytest.raises(ValueError, match="no labelled updates"):
        train(model(), SyntheticBlocks(tiny_config(), count=1), Run(epochs=1, stage="confidence"), output)
    assert output.is_file()  # retain the failed stage's inspection/checkpoint evidence


def test_initial_adapter_requires_matching_frozen_source(tmp_path: Path) -> None:
    from offload.dspark.d2.train.replay import apply_adapter
    m = model()
    p = tmp_path/"adapter.pt"
    state = {"schema": "dsv41.d2.train-checkpoint.v1", "frozen_identity": m.frozen_identity,
             "trainable": {n: v.detach().clone() for n, v in m.named_parameters()}}
    torch.save(state, p)
    assert apply_adapter(model(), str(p)) == sha256(p)
    state["frozen_identity"] = "different-source"
    torch.save(state, p)
    with pytest.raises(ValueError, match="frozen source"):
        apply_adapter(model(), str(p))


def test_reference_resume_refuses_a_fast_backend_checkpoint(tmp_path: Path) -> None:
    stop = threading.Event()
    stop.set()
    p = tmp_path / "reference.pt"
    data = SyntheticBlocks(tiny_config(), count=1)
    run = Run(epochs=1)
    train(model(), data, run, p, stop)
    state = torch.load(p, weights_only=False)
    assert state["fast_training"] is False and state["fast_backend"] is None
    state["fast_training"] = True
    loop.atomic_save(p, state)
    with pytest.raises(ValueError, match="backend changed"):
        train(model(), data, run, p, resume=True)
