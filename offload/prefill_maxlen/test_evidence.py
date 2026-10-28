from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.prefill_maxlen.evidence import Json, allocation, cold_samples, collect


def row(rep: int, ttft: float, cached: int = 0) -> dict[str, Json]:
    return {"type": "request", "concurrency": 1, "mixed": False, "rep": rep,
            "prompt_tokens": 32000, "ttft_s": ttft, "error": None,
            "cache_state": "cold", "usage": {"prompt_tokens_details": {"cached_tokens": cached}}}


def test_reconstruct_and_exclude_lazy_load_symmetrically(tmp_path: Path) -> None:
    for arm, times in (("R1", (90.0, 10.0)), ("R5", (95.0, 15.0))):
        for ctx, suffix in ((32000, "c32"), (150000, "c150")):
            samples = [row(rep, time) for rep, time in enumerate(times)]
            for sample in samples:
                sample["prompt_tokens"] = ctx
            (tmp_path / f"{arm}-{suffix}.jsonl").write_text(
                "\n".join(json.dumps(sample) for sample in samples))
    result = collect(tmp_path)
    observations = result["observations"]
    assert isinstance(observations, list)
    first, second = observations
    assert isinstance(first, dict) and isinstance(second, dict)
    assert first["r1_ttft_s"] == 10.0
    assert first["r5_ttft_s"] == 15.0
    assert second["r1_ttft_s"] == 50.0
    assert first["delta_s_per_4096_equivalent"] == 0.64


@pytest.mark.parametrize("defect", ["cache", "error", "tokens", "negative", "missing_rep", "nan"])
def test_refuse_bad_evidence(tmp_path: Path, defect: str) -> None:
    samples = [row(0, 10), row(1, 11)]
    if defect == "cache":
        samples[0] = row(0, 10, cached=128)
    elif defect == "error":
        samples[0]["error"] = "failure"
    elif defect == "tokens":
        samples[0]["prompt_tokens"] = 31000
    elif defect == "negative":
        samples[0]["ttft_s"] = -1
    elif defect == "nan":
        samples[0]["ttft_s"] = float("nan")
    else:
        samples.pop()
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(json.dumps(sample) for sample in samples))
    with pytest.raises(ValueError):
        cold_samples(path, 32000)


def test_storage_is_not_timing() -> None:
    assert allocation(524288)["indexer_fp8_factor4_bytes"] == 264 * 1024**2
    assert allocation(327680)["indexer_fp8_factor4_bytes"] == 165 * 1024**2
    with pytest.raises(ValueError):
        allocation(0)
