"""S24 item 5 helper: tiny reference/default training run; dumps trainables, optimizer state, scheduler and curve.

Run with PYTHONPATH pointing at the tree under test:  python ref_train_dump.py OUT.pt CASE
CASE = default | reference | b2 | epochs2 | plan
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from offload.dspark.d2.train.data import SyntheticBlocks
from offload.dspark.d2.train.loop import BatchPlan, Run, train
from offload.dspark.d2.train.model import Drafter
from offload.dspark.d2.train.synthetic import SyntheticSource, tiny_config


def main() -> None:
    out, case = Path(sys.argv[1]), sys.argv[2]
    torch.set_num_threads(2)
    torch.manual_seed(31)
    c = tiny_config()
    m = Drafter(SyntheticSource(c), c, reference_math=case == "reference",
                acc=torch.float64 if case == "reference" else torch.float32)
    data = SyntheticBlocks(c, 5)
    run = Run(epochs=2 if case == "epochs2" else 1, batch_size=2 if case in ("b2", "plan") else 1, chunk=32)
    plan = BatchPlan(((0, 3), (1,), (4, 2)), (1, 3), "s24-plan") if case == "plan" else None
    result = train(m, data, run, out.with_suffix(".ckpt"), plan=plan)
    state = torch.load(out.with_suffix(".ckpt"), weights_only=False)
    curve = [{k: v for k, v in row.items() if k not in ("seconds",)} for row in state["curve"]]
    torch.save({"trainable": state["trainable"], "optimizer": state["optimizer"], "scheduler": state["scheduler"],
                "curve": curve, "step": result["step"], "keys": sorted(state)}, out)


if __name__ == "__main__":
    main()
