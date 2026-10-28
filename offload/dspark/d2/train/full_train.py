"""Seed full-run initialization before the existing boundary-checkpointed loop."""
from __future__ import annotations

import torch

from .loop import main as loop_main
from .pilot import SEED


def main() -> None:
    # loop.train seeds order/RNG after model construction. Seed construction too,
    # matching pilot.prepare/arm's native base plus shared zero-B initialization.
    torch.manual_seed(SEED)
    loop_main()


if __name__ == '__main__':
    main()
