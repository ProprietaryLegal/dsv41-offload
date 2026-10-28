"""Transparent CPU arithmetic, not a GB10 profile or performance prediction.

Shape provenance: offload/spark_leads/metadata/official-1017.json $.config.text_config
(H5120, I2304, 40 target layers, 384 experts, top6); TP2 from inventory.
2.75 bpw is a checkpoint-wide approximation, not measured routed-byte weighting.
273 GB/s is NVIDIA's published hardware ceiling; 225 is a donor profile claim.
Expert selections are uniform/independent in this toy model. Real reuse, run
splitting, signs/scales, K distribution, overlapping dense traffic, and activation
costs invalidate treating the resulting bytes/bandwidth as an end-to-end bound.
"""
from __future__ import annotations

import json


def expected_distinct(rows: int, experts: int = 384, top_k: int = 6) -> float:
    if rows < 0 or experts <= 0 or not 0 < top_k <= experts:
        raise ValueError("Require rows >= 0 and 0 < top_k <= experts")
    return experts * (1 - (1 - top_k / experts) ** rows)


def expert_bytes_per_rank(hidden: int = 5120, intermediate: int = 2304,
                          tp: int = 2, bpw: float = 2.75) -> float:
    if min(hidden, intermediate, tp, bpw) <= 0:
        raise ValueError("All model dimensions and bpw must be positive")
    return 3 * hidden * intermediate / tp * bpw / 8


def collective_payload(rows: int, hidden: int = 5120) -> int:
    if rows < 0 or hidden <= 0:
        raise ValueError("Require rows >= 0 and hidden > 0")
    return rows * hidden * 2  # bf16, one reduction


def prefill(chunk: int, replay_rows: int = 128) -> dict[str, float | int | str]:
    """V4.1 trim-cut path: first 20 full layers; last 20 suffix-only layers.

    Layer 20 also writes full-batch KV, without its attention-output/MoE reductions.
    This simplified count omits that KV projection and extra Engram collectives.
    """
    if chunk <= 0 or not 0 < replay_rows <= 128:
        raise ValueError("Require positive chunk and 1 <= replay_rows <= 128")
    suffix = min(chunk, replay_rows)
    layer_rows = 20 * chunk + 20 * suffix
    reduce_bytes = layer_rows * 2 * 5120 * 2
    return {
        "status": "ESTIMATED; arithmetic only, no kernel/overlap measurement",
        "chunk_tokens": chunk,
        "replay_rows": suffix,
        "layer_rows": layer_rows,
        "routed_mlp_TFLOP_per_rank": layer_rows * (2 * 3 * 5120 * 1152 * 6) / 1e12,
        "reduce_bytes_per_rank": reduce_bytes,
        "reduce_wire_ms_200Gbps": reduce_bytes / 25e6,
        "engram_bytes_per_rank_2layers": chunk * 12 * 264 * 2,
        "engram_wire_ms_40Gbps": chunk * 12 * 264 * 2 / 5e6,
        "chunk_equivalent_ms_at_2150tps": chunk / 2150 * 1000,
    }


def main() -> None:
    expert = expert_bytes_per_rank()
    result = {
        "status": "ESTIMATED; uniform independent routing, no scale/activation/duplicate-run bytes",
        "packed_expert_bytes_per_rank": expert,
        "decode_scenarios": [{"live_rows": r, "distinct_experts_per_layer": expected_distinct(r),
                              "routed_GB_per_rank_40layers": expected_distinct(r) * expert * 40 / 1e9,
                              "stream_ms_at_273GBs": expected_distinct(r) * expert * 40 / 273e6,
                              "stream_ms_at_225GBs": expected_distinct(r) * expert * 40 / 225e6}
                             for r in (1, 2, 3, 4, 6, 12, 20)],
        "collectives": [{"scheduled_rows": r, "bytes_per_reduce": collective_payload(r),
                         "wire_ms_80reduces_at_200Gbps": collective_payload(r) * 80 / 25e6,
                         "logits_bytes_per_rank": r * 64640 * 2}
                        for r in (1, 6, 20, 24, 4096)],
        "note": "80 target reductions plus Engram/logit gathers; wire-only ignores host staging and rank skew.",
        "prefill_scenarios": [prefill(c) for c in (1024, 2048, 4096, 8192)],
        "replay_skip4096": prefill(4096, 1),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
