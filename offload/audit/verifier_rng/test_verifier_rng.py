"""AUD1 item 1: production's verifier vs vLLM #61161, on the image's real kernel (CPU interpreter).

Run (Controller, CPU only):
  cd offload/audit/verifier_rng
  TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 -m pytest -q -p no:cacheprovider \
      --basetemp=$SCRATCH/pytest test_verifier_rng.py
AUD1_WORKERS (default 16) sets the CPU processes per simulation. All draws are seeded, so the
statistics are reproducible; the chi-square bound has alpha 1e-4 (Wilson-Hilferty).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kernel_loader
import multiround as mr

WORKERS = int(os.environ.get("AUD1_WORKERS", "16"))
ALPHA = 1e-4

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "",
    reason="needs TRITON_INTERPRET=1 and CUDA_VISIBLE_DEVICES= (CPU interpreter)",
)


def _check(toy: mr.Toy, res: mr.Result) -> tuple[float, float]:
    stat, dof = mr.chi2_joint(toy, res)
    return stat, mr.chi2_threshold(dof, ALPHA)


def test_image_kernel_is_pre_61161(tmp_path: Path) -> None:
    """The image's _rejection_kernel keys every acceptance uniform by (request seed, position)."""
    src = (kernel_loader.DEFAULT_SRC / "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py")
    text = src.read_text()
    assert text.count("u = tl_rand32(seed, pos, includes_zero=False)") == 1
    assert "spec_step_seed" not in text
    variant = mr.write_pr61161_variant(tmp_path)
    assert "spec_step_seed(seed, i) if USE_BLOCK_VERIFICATION else seed" in variant.read_text()


def test_production_standard_one_hot_unbiased_pr_case() -> None:
    """#61161's own deterministic case on production's path: standard, one-hot q, T=1."""
    res = mr.simulate(mr.PR_CASE, "standard", 2000, seed=11, workers=WORKERS)
    stat, bound = _check(mr.PR_CASE, res)
    assert res.accepted > res.trials, "must exercise accepted drafts"
    assert res.reread_keys == 0
    assert stat < bound, (stat, bound, mr.summary(mr.PR_CASE, res))


def test_production_standard_one_hot_unbiased_context_windows_dead_rows() -> None:
    """Context-dependent target, deterministic drafter, windows 1..5, depth dead rows (-1)."""
    res = mr.simulate(mr.PROD_CASE, "standard", 3000, seed=13, workers=WORKERS)
    stat, bound = _check(mr.PROD_CASE, res)
    assert res.accepted > res.trials // 2
    assert res.reread_keys == 0
    assert stat < bound, (stat, bound, mr.summary(mr.PROD_CASE, res))


def test_negative_control_block_verification_is_biased_before_fix() -> None:
    """The same kernel in block mode reuses uniforms across rounds and fails the same check.

    This is the bug #61161 fixes; it shows the statistic detects it. Production does not use
    block verification (rejection_sample_method defaults to "standard"; serve.sh SPEC omits it).
    """
    res = mr.simulate(mr.PR_CASE, "block", 2000, seed=11, workers=WORKERS)
    stat, bound = _check(mr.PR_CASE, res)
    assert res.reread_keys > 0
    assert stat > bound, (stat, bound, mr.summary(mr.PR_CASE, res))


def test_positive_control_block_verification_with_61161_change(tmp_path: Path) -> None:
    variant = mr.write_pr61161_variant(tmp_path)
    res = mr.simulate(mr.PR_CASE, "block", 2000, seed=11, workers=WORKERS, variant=variant)
    stat, bound = _check(mr.PR_CASE, res)
    assert res.reread_keys == 0
    assert stat < bound, (stat, bound, mr.summary(mr.PR_CASE, res))


def test_loader_refuses_gpu_or_compiled_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(kernel_loader.LoaderError):
        kernel_loader.load()
