"""Explicit PP3 experiment switch. Importing it never loads Torch/CUDA."""
from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Mapping

SWITCH = "DSPARK_SAMPLED_TOP32"


def enabled(env: Mapping[str, str] | None = None) -> bool:
    value = (os.environ if env is None else env).get(SWITCH, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{SWITCH} must be 0 or 1")
    return value == "1"


def request_key(session: int, slot: int, generation: int, seed: int) -> int:
    """Own request generation plus user seed. Round/position are Philox counters.
    Retries must reuse these counters; session renewal creates a new RNG owner.
    Return signed int64 bits for Torch's wire/GPU ABI.
    """
    if not (0 <= session < 2**32 and 0 <= slot < 4 and 0 <= generation < 2**32
            and -(2**63) <= seed < 2**63):
        raise ValueError("invalid sampled RNG owner")
    payload = struct.pack("<IIIq", session, slot, generation, seed)
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8, person=b"PP3-draft-v1").digest(),
                          "little", signed=True)
