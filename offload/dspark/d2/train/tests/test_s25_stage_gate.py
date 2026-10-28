"""S25 item 3: the TRAIN LOOP (not the test) gates parameters by stage. OBJ1's confidence tests set
requires_grad themselves; this runs loop.train for one real optimizer step per stage and compares weights."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from offload.dspark.d2.train.data import SyntheticBlocks
from offload.dspark.d2.train.loop import Run, train
from offload.dspark.d2.train.model import Drafter
from offload.dspark.d2.train.synthetic import SyntheticSource, tiny_config


def _weights(m: Drafter) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in m.named_parameters()}


@pytest.mark.parametrize("stage", ("distill", "confidence"))
def test_loop_stage_gate_moves_only_the_intended_parameters(tmp_path: Path, stage: str) -> None:
    torch.manual_seed(31)
    torch.set_num_threads(2)
    c = tiny_config()
    m = Drafter(SyntheticSource(c), c)
    for _, p in m.named_parameters():  # deliberately WRONG starting gate; the loop must reset it
        p.requires_grad_(stage == "distill")
    before = _weights(m)
    train(m, SyntheticBlocks(c, count=8), Run(epochs=1, chunk=32, batch_size=8, stage=stage, lr=1e-2), tmp_path/"o.pt")
    after = _weights(m)
    moved = {n for n in before if not torch.equal(before[n], after[n])}
    confidence = {n for n in before if "confidence_head" in n}
    markov = {n for n in before if "markov_head" in n}
    assert len(confidence) == 1 and len(markov) == 2
    if stage == "confidence":
        assert moved == confidence
    else:
        assert not (moved & confidence)
        assert markov <= moved  # both Markov matrices are adapted by distillation
