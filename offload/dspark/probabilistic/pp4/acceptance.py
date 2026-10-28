"""Emitted sparse-q scoring and source-paired comparison to frozen PP2.

No logits or generated text are retained. Prefix survival uses the reverse
target-trajectory estimator, never products of independent row overlaps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from . import acceptance_frozen as frozen

ARM = "dyadic24-k32-t0.7"
MASS = 1 << 24


def score_sparse(ids: Tensor, q: Tensor, greedy: Tensor, top_ids: Tensor,
                 top_lp: Tensor, residual: Tensor, labels: Tensor, tok_lp: Tensor,
                 *, vocab: int, numeric: Any) -> Tensor:
    """Return [sampled/greedy, PP2 metric, position] from ACTUAL emitted q.

    Preserve PP2's exact captured-target normalization/rounding model, but
    consume dyadic probabilities directly without temperature/renormalization.
    """
    if (ids.shape != q.shape or ids.ndim != 2 or ids.shape[1] != 32
            or q.dtype != torch.float32 or vocab <= 32
            or ids.dtype not in (torch.int32, torch.int64)
            or greedy.shape != labels.shape or labels.shape != ids.shape[:1]
            or top_ids.shape != top_lp.shape or top_ids.shape[0] != ids.shape[0]
            or top_ids.shape[1] >= vocab):
        raise ValueError("sparse-q/capture ABI mismatch")
    if bool(((ids < 0) | (ids >= vocab)).any() or (ids[:, 1:] <= ids[:, :-1]).any()):
        raise ValueError("support must be sorted, unique and inside vocabulary")
    counts = q.double() * MASS
    if (not bool(torch.isfinite(q).all()) or bool((counts < 0).any())
            or not torch.equal(counts, counts.round()) or not bool((counts.sum(-1) == MASS).all())):
        raise ValueError("q is not an exact normalized dyadic24 distribution")
    if bool(((top_ids < 0) | (top_ids >= vocab)).any()):
        raise ValueError("capture IDs outside vocabulary")
    if bool(((labels < 0) | (labels >= vocab) | (greedy < 0) | (greedy >= vocab)).any()):
        raise ValueError("label/greedy IDs outside vocabulary")
    if bool((top_ids.sort(-1).values[:, 1:] == top_ids.sort(-1).values[:, :-1]).any()):
        raise ValueError("capture support is not unique")
    p, r, z, pad = numeric.captured_target(top_lp, residual)
    if not bool(torch.isfinite(tok_lp).all()):
        raise ValueError("observed logprob is nonfinite")
    hit = top_ids == labels[:, None]
    lp = top_lp.gather(-1, hit.long().argmax(-1)[:, None]).squeeze(-1)
    if bool(((lp - tok_lp).abs()[hit.any(-1)] > 1e-5).any()):
        raise ValueError("observed probability differs from captured bucket")
    observed_p = tok_lp.double().exp() / z
    if bool((observed_p <= 0).any()):
        raise ValueError("observed target probability must be positive")
    greedy_hit = top_ids == greedy[:, None]
    known = (p * greedy_hit).sum(-1)
    gp_lo = (known - pad).clamp(0, 1)
    gp_hi = (torch.where(greedy_hit.any(-1), known, torch.minimum(r, p.min(-1).values)) + pad).clamp(0, 1)
    result: list[Tensor] = []
    for support, probabilities in ((ids, q.double()), (greedy[:, None], torch.ones_like(greedy[:, None], dtype=torch.float64))):
        known_q = ((top_ids[:, :, None] == support[:, None, :]) * probabilities[:, None, :]).sum(-1)
        overlap = torch.minimum(p, known_q).sum(-1)
        # q is zero outside sparse support. If target top buckets plus support
        # cover V, compute min over U explicitly; no dense-vocabulary tensor.
        in_known = (support[:, :, None] == top_ids[:, None, :]).any(-1)
        unknown_count = vocab - top_ids.shape[1]
        has_zero_tail = unknown_count > (~in_known).sum(-1)
        tail_min = probabilities.masked_fill(in_known, float("inf")).min(-1).values
        tail_min = torch.where(has_zero_tail, 0., tail_min)
        lower = overlap + torch.minimum(r, tail_min)
        upper = overlap + torch.minimum(r, (1 - known_q.sum(-1)).clamp(0, 1))
        observed_q = ((support == labels[:, None]) * probabilities).sum(-1)
        survival = (observed_q / observed_p).clamp(max=1).cumprod(-1)
        result.append(torch.stack(((lower - pad).clamp(0, 1), (upper + pad).clamp(0, 1), survival, gp_lo, gp_hi)))
    output = torch.stack(result)
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError("nonfinite sparse score")
    return output


def summarize(receipt: dict[str, Any], frozen_dir: Path) -> dict[str, Any]:
    baseline = frozen.value(frozen_dir, "heldout-replay.json")
    previous = frozen.value(frozen_dir, "heldout-summary.json")
    if (receipt.get("schema") != "pp4.acceptance-replay.v1" or not receipt.get("complete")
            or receipt.get("smoke_limit") or receipt.get("arms") != [ARM, "greedy"]
            or receipt.get("semantic") != "dyadic24-floor-first-v1"
            or receipt.get("weights_tensor_payload_verified") is not True
            or receipt.get("manifest_sha256") != baseline["manifest_sha256"]
            or receipt.get("selection_sha256") != baseline["selection_sha256"]
            or receipt.get("verified_shards") != baseline["verified_shards"]
            or receipt.get("rounds") != baseline["rounds"]
            or receipt.get("parity_rounds") != baseline["parity_rounds"]
            or set(receipt["groups"]) != set(baseline["groups"])):
        raise ValueError("complete frozen PP2 population/provenance required")
    drift_groups: dict[str, list[float]] = {}
    frozen_comparison_groups: dict[str, Any] = {}
    for key, group in receipt["groups"].items():
        old = baseline["groups"][key]
        for field in ("request_sha256", "source_cluster_sha256", "family", "context", "rounds"):
            if group[field] != old[field]:
                raise ValueError("frozen source/anchor grouping differs: " + field)
        if group["sum"][-1][2] != old["sum"][-1][2]:
            drift_groups[key] = [float(current - previous) for current, previous
                                 in zip(group["sum"][-1][2], old["sum"][-1][2], strict=True)]
        frozen_comparison_groups[key] = {**group, "sum": [group["sum"][0], group["sum"][-1], old["sum"][-1]]}
    # The exact frozen bootstrap preserves view order, source connectivity,
    # 2,000 paired resamples, seed4102 and unavailable one-source intervals.
    summarizer = frozen.module(frozen_dir, "summarize")
    result = cast(dict[str, Any], summarizer.summary(receipt))
    against_frozen = summarizer.summary({**receipt, "arms": [ARM, "current-greedy", "greedy"],
                                         "groups": frozen_comparison_groups})
    comparisons: dict[str, Any] = {}
    for name, view in result["views"].items():
        now, greedy = view["arms"][ARM], view["arms"]["greedy"]
        old = previous["views"][name]["arms"]["k32-t0.7"]
        original_greedy = previous["views"][name]["arms"]["greedy"]
        historical = against_frozen["views"][name]["arms"]
        interval = old["paired_source_cluster_CI95_delta"]
        comparisons[name] = {
            "sampled_retained_tokens_per_round": 1 + now["EA_depth5"],
            "greedy_retained_tokens_per_round": 1 + greedy["EA_depth5"],
            "delta_retained_tokens_per_round": now["delta_EA_greedy"],
            "CI95_delta": now["paired_source_cluster_CI95_delta"],
            "PP2_delta": old["delta_EA_greedy"], "PP2_CI95_delta": interval,
            "change_from_PP2": now["delta_EA_greedy"] - old["delta_EA_greedy"],
            "PP2_greedy_retained_tokens_per_round": 1 + original_greedy["EA_depth5"],
            "greedy_change_from_PP2": historical["current-greedy"]["delta_EA_greedy"],
            "CI95_greedy_change_from_PP2": historical["current-greedy"]["paired_source_cluster_CI95_delta"],
            "sampled_delta_frozen_greedy": historical[ARM]["delta_EA_greedy"],
            "CI95_sampled_delta_frozen_greedy": historical[ARM]["paired_source_cluster_CI95_delta"],
            "point_within_PP2_interval": interval[0] <= now["delta_EA_greedy"] <= interval[1] if interval else None,
        }
    result.update(schema="pp4.acceptance-summary.v1", classification="MEASURED emitted GPU q on frozen heldout anchors",
                  frozen_commit=frozen.COMMIT, frozen_sha256=frozen.HASHES, comparisons=comparisons,
                  greedy_baseline_drift={"changed_groups": len(drift_groups),
                                         "group_prefix_total_differences": drift_groups,
                                         "status": "AGGREGATES_CHANGED" if drift_groups else "AGGREGATES_IDENTICAL",
                                         "scope": "source/context aggregate totals, not per-anchor identity; same inputs; CPython3.13/Torch2.12 versus PP2 CPython3.12/Torch2.10"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = args.receipt.read_bytes()
    result = summarize(json.loads(data), args.frozen_dir)
    result["input_sha256"] = hashlib.sha256(data).hexdigest()
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
