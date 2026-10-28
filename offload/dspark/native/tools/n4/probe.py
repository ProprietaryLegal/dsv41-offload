"""Private same-input attention capture; no inference arithmetic is replaced."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import torch

from ....worker.v1_dump import V1Dump
from ... import tp2_native as native
from ..n3.replay import main


def install() -> None:
    directory = Path(os.environ["N4_PROBE_DIR"])
    directory.mkdir(parents=True, exist_ok=False)
    module = cast(Any, native)
    attention = module.attention_into
    emit = V1Dump.emit
    active = False
    call = 0

    def traced(q: torch.Tensor, ring: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor,
               lo: torch.Tensor, last: torch.Tensor, block: torch.Tensor, sink: torch.Tensor,
               scale: float, out: torch.Tensor) -> torch.Tensor:
        nonlocal call
        result = cast(torch.Tensor, attention(q, ring, positions, slots, lo, last, block, sink, scale, out))
        if active:
            tensors = dict(q=q, ring=ring, positions=positions, slots=slots, lo=lo, last=last,
                           block=block, sink=sink, out=result)
            payload = {name: t.detach().cpu().clone() for name, t in tensors.items()}
            torch.save(dict(tensors=payload, scale=scale), directory / f"attention-{call:03}.pt")
            call += 1
        return result

    def emitting(self: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal active
        active = True
        try:
            emit(self, *args, **kwargs)
        finally:
            active = False

    module.attention_into = traced
    V1Dump.emit = emitting  # type: ignore[method-assign]


if __name__ == "__main__":
    install()
    main()
