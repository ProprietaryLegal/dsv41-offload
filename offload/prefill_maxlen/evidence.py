"""CPU-only numeric reconstruction of the PF1 cap regression; no network access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import cast

Json = None | bool | int | float | str | list["Json"] | dict[str, "Json"]


def record(value: Json) -> dict[str, Json]:
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def number(value: Json) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite number, got {value!r}")
    return result


def cold_samples(path: Path, ctx: int) -> list[dict[str, Json]]:
    result: list[dict[str, Json]] = []
    for line in path.read_text().splitlines():
        row = record(cast(Json, json.loads(line)))
        if row.get("type") != "request" or row.get("concurrency") != 1 or row.get("mixed"):
            continue
        if row.get("error") is not None or row.get("cache_state") != "cold":
            raise ValueError(f"{path}: failed/non-cold B1 request")
        if row.get("prompt_tokens") != ctx:
            raise ValueError(f"{path}: wrong prompt length")
        usage = record(row["usage"])
        details = record(usage["prompt_tokens_details"])
        if details.get("cached_tokens") != 0:
            raise ValueError(f"{path}: nonzero prefix cache hit")
        ttft = number(row["ttft_s"])
        if ttft <= 0:
            raise ValueError(f"{path}: nonpositive TTFT")
        result.append({"rep": row["rep"], "ttft_s": ttft})
    if len(result) < 2:
        raise ValueError(f"{path}: need at least two cold B1 requests")
    return result


def allocation(cap: int) -> dict[str, Json]:
    """ESTIMATED storage arithmetic, not observed timings or aggregate RSS."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    return {
        "cap": cap,
        "indexer_fp8_factor4_bytes": 4 * cap * 132,
        "indexer_fp8_factor1_bytes": cap * 132,
        "rope_two_fp32_64_column_caches_bytes": 2 * (cap + 2048) * 64 * 4,
        "expanded_indexer_block_table_per_ratio1_builder_bytes": 4096 * ((cap + 63) // 64) * 4,
        "request_token_ids_4_seats_int32_bytes": 4 * cap * 4,
        "hypothetical_one_extra_factor4_sweep_at_225GBps_ms": 4 * cap * 132 / 225e9 * 1000,
    }


def collect(root: Path) -> dict[str, Json]:
    rows: list[Json] = []
    hashes: dict[str, Json] = {}
    for ctx, suffix in ((32000, "c32"), (150000, "c150")):
        arms: dict[str, list[dict[str, Json]]] = {}
        for arm in ("R1", "R5"):
            path = root / f"{arm}-{suffix}.jsonl"
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            arms[arm] = cold_samples(path, ctx)
        # Same warmup treatment as S18: R1 first 32K is lazy-load affected.
        # Retain it in samples, but compare rep >= 1 at 32K in both arms.
        medians = {
            arm: statistics.median(
                number(sample["ttft_s"]) for sample in samples
                if ctx != 32000 or number(sample["rep"]) >= 1
            )
            for arm, samples in arms.items()
        }
        delta = medians["R5"] - medians["R1"]
        rows.append({
            "label": "MEASURED", "prompt_tokens": ctx,
            "samples": {arm: cast(Json, samples) for arm, samples in arms.items()},
            "r1_ttft_s": medians["R1"], "r5_ttft_s": medians["R5"],
            "delta_s": delta, "delta_ms_per_token": delta / ctx * 1000,
            "delta_s_per_4096_equivalent": delta / ctx * 4096,
            "regression_percent": delta / medians["R1"] * 100,
            "linear_cap_interpolation_393216_ttft_s_ESTIMATED": medians["R1"] + delta / 3,
        })
    return {"source_sha256": hashes, "observations": rows,
            "allocation_ESTIMATED": [allocation(cap) for cap in (327680, 393216, 524288)],
            "causal_status": "UNPROVEN: TTFT does not identify a kernel or a linear law"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = collect(args.results)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
