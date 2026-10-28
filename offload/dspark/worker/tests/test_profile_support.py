"""CPU regressions: paired residuals and fail-closed GPU ownership."""

import pytest

from worker.tools.profile_support import GPU_UUID, expert_counts, require_gpu_window, summarize


def test_paired_residual_is_reduced_after_subtraction():
    samples = [{"wall_us": a, "gpu_us": b} for a, b in [(100, 90), (200, 10), (300, 290)]]
    result = summarize(samples)
    assert result["wall_minus_gpu_us"]["p50"] == 10
    assert result["wall_us"]["p50"] - result["gpu_us"]["p50"] == 110


def test_gate_and_uuid_are_both_required(tmp_path, monkeypatch):
    marker = tmp_path / "W1_GPU_OK"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    with pytest.raises(RuntimeError, match="window closed"):
        require_gpu_window(marker)
    marker.touch()
    require_gpu_window(marker)
    marker.unlink()
    with pytest.raises(RuntimeError, match="window closed"):
        require_gpu_window(marker)
    marker.touch()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        require_gpu_window(marker)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        require_gpu_window(marker)


def test_counts_include_unpopulated_experts():
    assert expert_counts([0, 3, 3], 4) == [1, 0, 0, 2]
    with pytest.raises(ValueError, match="outside"):
        expert_counts([4], 4)
    with pytest.raises(ValueError, match="empty"):
        summarize([])
