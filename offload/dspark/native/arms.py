"""Torch-free, canonical native drafter arm configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypedDict

BATCH_INVARIANT_ENV = "DSPARK_NATIVE_BATCH_INVARIANT"


def batch_invariant_from_env() -> int:
    """NB1 batch-invariant dispatch level for native kx2, from one env switch: "0"/unset = off (default); "1" =
    every draft-body shape (B x block rows, B = 1..4) uses the dispatch B1 uses; "2" = also window insertion in
    fixed 8-row projection sub-batches (the bucket-8 arithmetic for every flush size). Anything else refuses."""
    value = os.environ.get(BATCH_INVARIANT_ENV, "0")
    if value not in ("0", "1", "2"):
        raise ValueError(f"{BATCH_INVARIANT_ENV}={value!r}: expected 0, 1 or 2")
    return int(value)


@dataclass(frozen=True)
class ArmConfig:
    name: str
    experts: str = "k2"
    dense: str = "k5"
    kv_qdq: str = "engine"
    wo_a_qdq: bool = False
    combine: str = "fused"
    head: str = "k6"
    expanded_stages: tuple[int, ...] = ()
    expert_kernel: str = "n1"         # packed experts: "n1" (ops.ExpertBank Triton pair kernels) | "kx-regfed"
    markov: str = "factorized"        # head bf16: "factorized" (exact fp32 bias computed per position) | "hot"
    markov_rows: int = 4096           #   (+ exact fp32 rows cached for the serve's most frequent previous tokens)
    dense_kernel: str = "n1"         # "kx-selective": measured FP8 shapes, BF16 otherwise
    dense_profile: str = "initial"  # N2 tuning control: "initial" | "measured-small"

    def check(self) -> None:
        if self.experts not in ("k2", "packed", "expanded") or self.dense not in ("k5", "fp8", "bf16"):
            raise ValueError(f"{self.name}: experts {self.experts} / dense {self.dense}")
        if self.kv_qdq not in ("official", "engine") or self.combine not in ("official", "fused"):
            raise ValueError(f"{self.name}: kv_qdq {self.kv_qdq} / combine {self.combine}")
        if self.expert_kernel not in ("n1", "kx-regfed", "kx-fused"):
            raise ValueError(f"{self.name}: expert kernel {self.expert_kernel}")
        if self.dense_kernel not in ("n1", "kx-selective") or (self.dense_kernel == "kx-selective" and self.dense != "bf16"):
            raise ValueError(f"{self.name}: dense kernel {self.dense_kernel} requires BF16 baseline")
        if self.dense_profile not in ("initial", "measured-small"):
            raise ValueError(f"{self.name}: dense profile {self.dense_profile}")
        if self.head not in ("k6", "bf16") or self.markov not in ("factorized", "hot"):
            raise ValueError(f"{self.name}: head {self.head} / markov {self.markov}")
        if self.combine == "official" and self.experts == "k2":
            raise ValueError(f"{self.name}: the K2 coop kernel sums its experts itself (combine must be fused)")
        if self.dense == "k5" and (self.kv_qdq != "engine" or self.wo_a_qdq):
            raise ValueError(f"{self.name}: K5 dense runs E's attention (engine KV record, no wo_a QDQ)")

    @property
    def native(self) -> bool:
        return self.experts != "k2" or self.dense != "k5" or self.head != "k6"

    @property
    def kx2_dispatch(self) -> bool:
        """The measured KX dispatch the batch-invariant mode pins (FP8 dense by M, fused experts by stage/rows)."""
        return (self.dense_kernel == "kx-selective" and self.dense_profile == "measured-small"
                and self.expert_kernel == "kx-fused")

    def stage_expert_kind(self, i: int) -> str:
        if self.experts == "packed" and i in self.expanded_stages:
            return "expanded"
        return self.experts


class OfficialPolicy(TypedDict):
    kv_qdq: str
    wo_a_qdq: bool
    combine: str


OFFICIAL: OfficialPolicy = dict(kv_qdq="official", wo_a_qdq=True, combine="official")
ARMS = {
    "k2-tp2": ArmConfig("k2-tp2"),                                                      # the deployed control
    "native-experts-packed": ArmConfig("native-experts-packed", experts="packed", combine="official"),
    "native-experts-expanded": ArmConfig("native-experts-expanded", experts="expanded", combine="official"),
    "native-packed": ArmConfig("native-packed", experts="packed", dense="fp8", **OFFICIAL),
    "native-expanded": ArmConfig("native-expanded", experts="expanded", dense="bf16", **OFFICIAL),
    "native-hybrid": ArmConfig("native-hybrid", experts="packed", dense="bf16", **OFFICIAL),
    "native-hybrid-s0x": ArmConfig("native-hybrid-s0x", experts="packed", dense="bf16", expanded_stages=(0,),
                                   **OFFICIAL),
    # NDM §6 second step: the draft-only official BF16 head (fp32 logits) with the exact fp32 Markov bias
    "native-hybrid-head": ArmConfig("native-hybrid-head", experts="packed", dense="bf16", head="bf16", **OFFICIAL),
    "native-expanded-head": ArmConfig("native-expanded-head", experts="expanded", dense="bf16", head="bf16",
                                      **OFFICIAL),
    "native-hybrid-head-hot": ArmConfig("native-hybrid-head-hot", experts="packed", dense="bf16", head="bf16",
                                        markov="hot", **OFFICIAL),
    "k2-head": ArmConfig("k2-head", head="bf16"),            # the head change alone on the control
    # KX-MXFP4's register-fed kernel for the packed experts (same math; their schedule, MMA and reuse)
    "native-hybrid-kx": ArmConfig("native-hybrid-kx", experts="packed", dense="bf16", expert_kernel="kx-regfed",
                                  **OFFICIAL),
    "native-packed-kx": ArmConfig("native-packed-kx", experts="packed", dense="fp8", expert_kernel="kx-regfed",
                                  **OFFICIAL),
    "native-hybrid-kx2": ArmConfig("native-hybrid-kx2", experts="packed", dense="bf16", expert_kernel="kx-fused",
                                   dense_kernel="kx-selective", dense_profile="measured-small", **OFFICIAL),
}
