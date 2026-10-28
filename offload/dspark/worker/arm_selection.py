"""Explicit TP2 arm policy; no torch/native imports on the default worker path."""
from __future__ import annotations

SERVING_ARMS = ("k2-tp2", "native-packed", "native-expanded", "native-hybrid-kx", "native-hybrid-kx2")


def selected_arm(value: str | None = None) -> str:
    import os
    arm = os.environ.get("DSPARK_WORKER_ARM", "k2-tp2") if value is None else value
    if arm not in SERVING_ARMS:
        raise ValueError(f"DSPARK_WORKER_ARM={arm!r}: expected one of {SERVING_ARMS}; no fallback")
    return arm
