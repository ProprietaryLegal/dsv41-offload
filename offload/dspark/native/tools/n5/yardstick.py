"""Additional S20 yardstick, separate from unchanged V1. CPU-only, aggregates only.

The retained control runs official GPU sparse_attn at s0, with the official CPU
pipeline elsewhere. It is NOT the requested complete official-GPU pipeline.
Window codes are reconstructed under a common power-of-two scale per official
32-value block, and accepted only after exact dequantized-value reconstruction.
Original quantizer scales and historical main_x rows were not saved by the worker.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import torch

from .strict_summary import ARMS, KEYS, mapping, number, record, sha


def dump(path: Path) -> dict[str, object]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"expected dump object: {path}")
    return cast(dict[str, object], value)


def tensor(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("missing tensor in private dump")
    return value


def relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape or not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise ValueError("non-finite or shape-mismatched tensors")
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def upper_median(values: list[float]) -> float:
    if not values:
        raise ValueError("empty distribution")
    return sorted(values)[len(values) // 2]  # V1's q(..., .5), not mean of central two


def code_steps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact signed E4M3FN code distances, merging the two encodings of zero."""
    if a.shape != b.shape or a.shape[-1] % 32:
        raise ValueError("FP8 comparison requires equal shapes and 32-value blocks")
    x, y = a.float().reshape(-1, 32), b.float().reshape(-1, 32)
    if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
        raise ValueError("nonfinite window KV")
    peak = torch.maximum(x.abs().amax(-1), y.abs().amax(-1))
    exponent = torch.ceil(torch.log2(peak.clamp_min(1e-30) / 448))
    scale = torch.pow(2.0, exponent)[:, None]
    qx, qy = (x / scale).to(torch.float8_e4m3fn), (y / scale).to(torch.float8_e4m3fn)
    if not torch.equal(qx.float() * scale, x) or not torch.equal(qy.float() * scale, y):
        raise ValueError("KV values do not reconstruct exactly under a common FP8 scale")
    cx, cy = qx.view(torch.uint8).int(), qy.view(torch.uint8).int()
    ox = torch.where(cx < 128, cx + 127, 255 - cx)
    oy = torch.where(cy < 128, cy + 127, 255 - cy)
    return (ox - oy).abs().reshape(a.shape)


def bf16_steps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError("main_x comparison requires equal shapes")
    # Worker dumps widen BF16 storage to FP32; prove narrowing loses no bits.
    if not torch.equal(a.to(torch.bfloat16).float(), a.float()) or not torch.equal(b.to(torch.bfloat16).float(), b.float()):
        raise ValueError("main_x dump contains values not exactly representable in BF16")
    a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    x, y = a.view(torch.int16).int() & 65535, b.view(torch.int16).int() & 65535
    return (torch.where(x < 32768, x + 32767, 65535 - x) -
            torch.where(y < 32768, y + 32767, 65535 - y)).abs()


def tensor_fields(d: dict[str, object]) -> dict[str, torch.Tensor]:
    out = {name: tensor(d[name]) for name in ("kv_window", "main_x_P", "base_logits", "conf", "head_hidden", "head_x")}
    for stage in range(3):
        st = mapping(d[f"s{stage}"])
        out.update({f"s{stage}.{name}": tensor(st[name]) for name in ("attn_out", "ffn_in", "ffn_out", "h_out")})
        out[f"s{stage}.route.idx"] = tensor(mapping(st["route"])["idx"])
        out[f"s{stage}.route.margin"] = tensor(mapping(st["route"])["margin"])
    return out


def analyze(pair: Path, n4: Path, mirrors: Path, strict: dict[str, object]) -> dict[str, object]:
    control_errors: list[float] = []
    rows: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    totals: dict[str, list[float]] = {arm: [] for arm in ARMS}
    clause: dict[str, bool] = dict.fromkeys(ARMS, True)
    for key in KEYS:
        mirror = dump(mirrors / f"{key}.pt")
        control_path = n4 / "official-component-control" / f"{key}.pt"
        control = dump(control_path)
        hashes[str(control_path)] = sha(control_path)
        control_error = relerr(tensor(mapping(control["s0"])["attn_out"]),
                               tensor(mapping(mirror["s0"])["attn_out"]))
        control_errors.append(control_error)
        for arm in ARMS:
            actual = dump(pair / f"{arm}-capture/{key}.pt")
            solo = dump(n4 / f"pilot-v2-{arm}/{key}.pt")
            pa, so = tensor_fields(actual), tensor_fields(solo)
            changed_fields = [name for name in pa if not torch.equal(pa[name], so[name])]
            error = relerr(pa["s0.attn_out"], tensor(mapping(mirror["s0"])["attn_out"]))
            totals[arm].append(error)
            pos = int(number(actual["P"]))
            positions = list(range(max(0, pos - 127), pos + 1))
            mx_steps = bf16_steps(pa["main_x_P"], tensor(control["main_x_P"]))
            stage_rows: list[dict[str, object]] = []
            for stage in range(3):
                refpath = n4 / "reference-attention12-v1" / f"{key}-s{stage}.pt"
                reference = tensor(dump(refpath)["kv"])[0, [p % 128 for p in positions]]
                window = pa["kv_window"][:, stage]
                if window.shape != reference.shape:
                    raise ValueError(f"{arm}/{key}/s{stage}: window mismatch")
                steps = code_steps(window, reference)
                different = window != reference
                bad = (steps > 2).any(-1)
                bad_pos = [positions[i] for i in bad.nonzero().flatten().tolist()]
                changed = steps[different]
                hist = {str(int(v)): int((changed == v).sum()) for v in changed.unique().tolist()}
                clause[arm] = clause[arm] and not bad_pos
                stage_rows.append({"stage": stage, "unequal": int(different.sum()),
                    "nope_unequal": int(different[:, :-64].sum()), "rope_unequal": int(different[:, -64:].sum()),
                    "positions": [positions[i] for i in different.any(-1).nonzero().flatten().tolist()],
                    "code_distance_counts": hist, "max_code_distance": int(steps.max()),
                    "positions_beyond_2_steps": bad_pos, "relative_norm": relerr(window, reference),
                    "max_abs_delta": float((window - reference).abs().max()),
                    "max_row_relative_norm": float(((window.double() - reference.double()).norm(dim=-1) /
                                                      reference.double().norm(dim=-1).clamp_min(1e-30)).max()),
                    "nonzero_sign_flips": int(((window * reference < 0) & different).sum())})
            rows.append({"arm": arm, "key": key, "P": pos, "s0_error": error,
                "official_attention_control_s0_error": control_error,
                "pair_solo_changed_tensor_fields": changed_fields,
                "pair_solo_same_drafts": actual["drafts"] == solo["drafts"],
                "pair_solo_same_window_metadata": actual["window"] == solo["window"],
                "main_x_P_unequal": int((mx_steps != 0).sum()), "main_x_P_max_bf16_steps": int(mx_steps.max()),
                "main_x_P_relative_norm": relerr(pa["main_x_P"], tensor(control["main_x_P"])),
                "window": stage_rows})
    control_median = upper_median(control_errors)
    strict_runs = strict["runs"]
    if not isinstance(strict_runs, list):
        raise ValueError("missing strict V1 results")
    runs = []
    for arm in ARMS:
        sr = next(mapping(r) for r in strict_runs if mapping(r)["arm"] == arm)
        pair_gates = mapping(mapping(sr["pair"])["gates"])
        median = upper_median(totals[arm])
        ratio = median / control_median
        gates = {"complete_scope": pair_gates["complete_requested_scope"],
                 "zero_v1_state_defects": pair_gates["zero_state_defects"],
                 "zero_v1_unexplained": pair_gates["zero_unexplained"],
                 "v1_kl_within_3x_band": pair_gates["kl_median_within_3x_summation_band"],
                 "s0_within_3x_available_official_attention_control": ratio <= 3,
                 "all_window_changes_within_1_or_2_fp8_steps": clause[arm],
                 "historical_main_x_origins_proved_1_bf16_step": False,
                 "complete_official_gpu_pipeline_control_available": False}
        runs.append({"arm": arm, "s0_median": median, "control_median": control_median,
                     "s0_over_available_official_control": ratio, "gates": gates,
                     "verdict": "QUALIFIED FOR GB10 A/B" if all(gates.values()) else "NOT QUALIFIED FOR GB10 A/B"})
    return {"scope": "same 12 keys; additional yardstick, never rewrites strict V1",
        "control": "official GPU sparse_attn stage0 only; official CPU main_proj/wkv/wq/wo/act_quant elsewhere",
        "full_official_gpu_control_available": False, "median_convention": "V1 upper order statistic",
        "official_control_3x_threshold": 3 * control_median,
        "fp8_scale_method": "common ue8m0 scale per 32 values; exact reconstruction required; original scales not dumped",
        "historical_main_x_limit": "main_x_P only is dumped; no direct native historical main_x for all differing KV rows",
        "control_sha256": hashes, "runs": runs, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", type=Path, required=True)
    ap.add_argument("--n4", type=Path, required=True)
    ap.add_argument("--mirrors", type=Path, required=True)
    ap.add_argument("--strict", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    result = analyze(a.pair, a.n4, a.mirrors, record(a.strict))
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["runs"], indent=2))


if __name__ == "__main__":
    main()
