"""NB1 batch-invariant switch: default off, refuses unknown values, pins B1's dispatch when on (CPU only)."""
from __future__ import annotations

from typing import cast

import pytest
import torch

from offload.dspark.kernels.fp8dense.api import DenseFP8
from offload.dspark.kernels.fusedmoe import select_variant
from offload.dspark.native import identity
from offload.dspark.native.arms import ARMS, BATCH_INVARIANT_ENV, batch_invariant_from_env
from offload.dspark.native.kx import KXSelectiveDense
from offload.dspark.native.ops import NativeDense
from offload.dspark.native.tp2_native import NativeTPDrafter


def test_switch_defaults_off_and_refuses_unknown_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(BATCH_INVARIANT_ENV, raising=False)
    assert batch_invariant_from_env() == 0
    for value in ("0", "1", "2"):
        monkeypatch.setenv(BATCH_INVARIANT_ENV, value)
        assert batch_invariant_from_env() == int(value)
    for bad in ("", "true", "3", " 1", "on"):
        monkeypatch.setenv(BATCH_INVARIANT_ENV, bad)
        with pytest.raises(ValueError):
            batch_invariant_from_env()


def test_only_kx2_has_the_pinned_dispatch() -> None:
    assert [name for name, cfg in ARMS.items() if cfg.kx2_dispatch] == ["native-hybrid-kx2"]


def _dense(n: int, k: int, groups: int = 1) -> tuple[NativeDense, DenseFP8]:
    baseline = NativeDense("probe", "bf16", torch.zeros(groups * n, k, dtype=torch.bfloat16), None)
    codes = torch.zeros(groups, n, k, dtype=torch.uint8)
    scales = torch.zeros(groups * n // 32, k // 32, dtype=torch.uint8)
    return baseline, DenseFP8(codes, scales, groups, n, k, None, True, "register")  # type: ignore[arg-type]


@pytest.mark.parametrize("n,k,groups", [(5120, 4096, 1), (5120, 1152, 1), (1024, 4096, 4), (1152, 5120, 1),
                                        (16384, 1280, 1), (512, 5120, 1)])
def test_fixed_splits_keep_b1_reduction_at_every_draft_shape(n: int, k: int, groups: int) -> None:
    baseline, candidate = _dense(n, k, groups)
    default = KXSelectiveDense(baseline, candidate, (5, 10, 15))
    b1_splits = candidate.configuration(5)[0]
    pinned = KXSelectiveDense(baseline, candidate, (5, 10, 15, 20), splits=b1_splits)
    assert pinned.configs[5] == default.configs[5]                      # B1 unchanged
    assert {m: c[0] for m, c in pinned.configs.items()} == dict.fromkeys((5, 10, 15, 20), b1_splits)
    assert {m: c[1] for m, c in pinned.configs.items()} == {m: candidate.configuration(m)[1] for m in (5, 10, 15, 20)}
    for m, (splits, _) in pinned.configs.items():
        scratch = pinned.scratch[m]
        assert (scratch is None) == (splits == 1)
        assert scratch is None or tuple(scratch.shape) == (splits, m, groups, n)


def test_measured_profile_is_what_the_switch_replaces() -> None:
    _, wo_b = _dense(5120, 4096)
    assert [wo_b.configuration(m)[0] for m in (5, 10, 15)] == [8, 4, 4]      # N6/NB1: split-K 8 -> 4
    assert select_variant(1, 10) == 256 and select_variant(1, 5) == 131072   # B1's expert variant everywhere


def test_identity_recipe_names_the_switch_only_when_on(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = {"files": {"model.safetensors": "AB"}, "markov_identity": "m"}
    monkeypatch.delenv(BATCH_INVARIANT_ENV, raising=False)
    off = identity.compute("native-hybrid-kx2", checkpoint)
    assert "batch_invariant_dispatch" not in off["recipe"]
    monkeypatch.setenv(BATCH_INVARIANT_ENV, "1")
    on = identity.compute("native-hybrid-kx2", checkpoint)
    assert on["recipe"]["batch_invariant_dispatch"].startswith("nb1-draft-v1")
    assert on["drafter_identity"] != off["drafter_identity"]
    monkeypatch.setenv(BATCH_INVARIANT_ENV, "2")
    both = identity.compute("native-hybrid-kx2", checkpoint)
    assert both["recipe"]["batch_invariant_dispatch"].endswith("8-row sub-batches")
    assert len({off["drafter_identity"], on["drafter_identity"], both["drafter_identity"]}) == 3
    with pytest.raises(ValueError):
        identity.compute("native-hybrid-kx", checkpoint)


class _Rows:
    """Stand-in drafter for the level-2 sub-batch helper (no CUDA, no checkpoint)."""
    insert_rows = 8


def test_fixed_rows_runs_every_row_through_8_row_calls_and_copies_out() -> None:
    shapes: list[int] = []
    buffer = torch.empty(8, 3)

    def linear(x: torch.Tensor) -> torch.Tensor:          # like KX dense: a persistent per-M output buffer
        shapes.append(x.shape[0])
        buffer.copy_(x[:, :3] * 2)
        return buffer
    x = torch.arange(13 * 4, dtype=torch.float32).view(13, 4)
    out = NativeTPDrafter._fixed_rows(cast(NativeTPDrafter, _Rows()), linear, x)
    assert shapes == [8, 8] and torch.equal(out, x[:, :3] * 2)
