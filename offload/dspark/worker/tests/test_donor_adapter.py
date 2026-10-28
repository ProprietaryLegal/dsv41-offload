"""Explicit unsupported shapes must never be counted as donor-kernel tests."""

import subprocess

import pytest

from worker.tools.donor_adapter import trellis_dispatch, verify_checkout


def test_head_needs_explicit_wide_override():
    with pytest.raises(ValueError, match="exceeds donor default"):
        trellis_dispatch(6, 5120, 129280)
    plan = trellis_dispatch(6, 5120, 129280, allow_wide=True)
    assert plan.wide_override and plan.backend == "trellis_marlin_dense"


def test_our_k2_experts_are_rejected():
    for moe in (False, True):
        with pytest.raises(ValueError, match="does not implement K2"):
            trellis_dispatch(2, 5120, 1536, moe=moe)
    assert trellis_dispatch(5, 5120, 512).bits == 5
    with pytest.raises(ValueError, match="128-aligned"):
        trellis_dispatch(5, 5121, 512)


def test_pin_and_tracked_source_changes_are_verified(tmp_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True)
    git("init", "-q")
    source = tmp_path / "kernel.cu"
    source.write_text("original\n")
    git("add", "kernel.cu")
    git("-c", "user.name=W1 test", "-c", "user.email=w1@example.invalid", "commit", "-qm", "fixture")
    sha = git("rev-parse", "HEAD").stdout.strip()
    verify_checkout(tmp_path, sha)
    with pytest.raises(RuntimeError, match="revision mismatch"):
        verify_checkout(tmp_path, "0" * 40)
    source.write_text("altered\n")
    with pytest.raises(subprocess.CalledProcessError):
        verify_checkout(tmp_path, sha)
