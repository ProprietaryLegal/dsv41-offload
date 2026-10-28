"""R-PP4: #61161's multi-round check on PP4's sampled verifier (image kernel + PP3 q ABI), CPU interpreter only.

Run (Controller, CPU only), next to test_verifier_rng.py:
  TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 -m pytest -q -p no:cacheprovider \
      --basetemp=$SCRATCH/pytest offload/audit/verifier_rng/test_pp4_sampled_rng.py
Same alpha (1e-4, Wilson-Hilferty) and seeded draws as AUD1's tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kernel_loader
import multiround as mr
import pp4_sampled as ps

WORKERS = int(os.environ.get("AUD1_WORKERS", "16"))
ALPHA = 1e-4

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "",
    reason="needs TRITON_INTERPRET=1 and CUDA_VISIBLE_DEVICES= (CPU interpreter)",
)


def _check(case: ps.SampledCase, res: mr.Result) -> tuple[float, float]:
    stat, dof = mr.chi2_joint(case.toy, res)
    return stat, mr.chi2_threshold(dof, ALPHA)


@pytest.fixture(scope="module")
def kernel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = ps.write_pp4_kernel(tmp_path_factory.mktemp("pp4-kernel"))
    text = path.read_text()
    # PP4 leaves the acceptance draw of standard verification untouched: one Philox uniform per (seed, position),
    # read only while verifying, and it refuses block verification whenever sampled q is on.
    assert text.count("u = tl_rand32(seed, pos, includes_zero=False)") == 1
    assert "if use_block_verification or synthetic_conditional_rates is not None or watermark:" in text
    assert "spec_step_seed" not in text
    return path


def test_pp4_sampled_q_unbiased_context_windows_dead_rows(kernel: Path) -> None:
    res = ps.simulate(ps.PP4_CASE, 3000, seed=17, kernel=kernel, workers=WORKERS)
    stat, bound = _check(ps.PP4_CASE, res)
    assert res.accepted > res.trials // 2, "must exercise accepted sampled drafts"
    assert res.reread_keys == 0
    assert stat < bound, (stat, bound, mr.summary(ps.PP4_CASE.toy, res))


def test_pp4_sampled_q_unbiased_fixed5(kernel: Path) -> None:
    res = ps.simulate(ps.PP4_FIXED5, 3000, seed=19, kernel=kernel, workers=WORKERS)
    stat, bound = _check(ps.PP4_FIXED5, res)
    assert res.accepted > res.trials
    assert res.reread_keys == 0
    assert stat < bound, (stat, bound, mr.summary(ps.PP4_FIXED5.toy, res))


def test_negative_control_wrong_q_is_detected(kernel: Path) -> None:
    """The statistic has power on this path: drafts drawn from a law other than the q given to the verifier bias it."""
    res = ps.simulate(ps.MISMATCH, 3000, seed=19, kernel=kernel, workers=WORKERS)
    stat, bound = _check(ps.MISMATCH, res)
    assert stat > bound, (stat, bound, mr.summary(ps.MISMATCH.toy, res))


def test_pp4_kernel_refuses_block_verification(kernel: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ps.SAMPLED_ENV, "1")
    module = kernel_loader.load(kernel)
    q = torch.full((1, 2, 2), 0.5, dtype=torch.float32)
    with pytest.raises(ValueError, match="standard verification only"):
        module.rejection_sample(
            torch.zeros(3, 2, dtype=torch.float32), q, torch.tensor([0, 1, 0], dtype=torch.int64),
            torch.tensor([0, 3], dtype=torch.int32), torch.tensor([5, 6, 7], dtype=torch.int64),
            torch.zeros(1, dtype=torch.int32), torch.zeros(3, dtype=torch.int32),
            torch.tensor([0, 1, 2], dtype=torch.int32), torch.ones(1, dtype=torch.float32),
            torch.tensor([123], dtype=torch.int64), 2, use_block_verification=True)
