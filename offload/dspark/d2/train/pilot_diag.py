"""D2-DIAG: reference-path diagnostic of the matched pilot (DIAG.md §Hypotheses).

Scores the pilot's frozen sample (plan.json keys, never a new draw) with the same
model load and adapter application as pilot_eval, for six weight variants, and
reports per-position aggregates. Shared variants must reproduce the receipt's
E[A] exactly or the run stops. Outputs contain aggregates and identities only,
never rows or token ids. Runs only through run_pilot_locked.sh; never trains.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Callable, cast

import torch
from torch import Tensor

from .data import D2Blocks, sha256
from .evaluate import bucket_kl, match_length
from .model import Drafter, load
from .pilot import SEED, Spec, digest, initialization, sources, validate_spec, verify_units, write_json
from .pilot_eval import EVALUATOR, Checkpoint, Evaluation, Metrics, read_arm, validate_sample_evaluation, verify_sample_draw
from .pilot_sample import DEV_POPULATION, RoundKey, complete_rounds, key_hash, population_splits
from .pilot_statistics import RequestTotal
from .replay import apply_trainables, read_checkpoint

SCHEMA = "dsv41.d2.pilot-diagnostic.v1"
PART_SCHEMA = "dsv41.d2.pilot-diagnostic-part.v1"
TOOL = "D2-DIAG-pilot-diagnostic-v1"
# The only module the diagnostic may add next to the sources the pilot plan hashed.
ALLOWED_NEW_SOURCES = frozenset({"d2/train/pilot_diag.py"})
TEMPERATURES = (.6, .8, .9, 1., 1.1, 1.25, 1.5, 2.)
NEAR_TIE = 1e-3  # logit gap below which a greedy/teacher-forced argmax miss is arithmetic, not conditioning
VARIANTS = ("V0", "R2.5", "R20", "F20", "L20", "H20")
# Variant -> (arm, checkpoint blocks) whose receipt metrics it must reproduce; V0 is the receipt baseline.
SHARED: dict[str, tuple[str, str] | None] = {"V0": None, "R2.5": ("REF", "2500"), "R20": ("REF", "20000"),
                                             "F20": ("FAST", "20000")}
POSITION_FIELDS = ("y_match", "t_match", "pp_lower", "pp_upper", "q1", "entropy", "gap", "kl", "kl_top", "kl_tail",
                   "survive")


class DiagnosticError(RuntimeError):
    """A built-in sanity check failed: the run is not a diagnosis of the pilot."""


def component(name: str) -> str:
    """Trainable tensor groups for the LoRA-only / head-only ablations; unknown names refuse."""
    if "markov_head" in name:
        return "markov"
    if "confidence_head" in name:
        return "confidence"
    if (name.endswith(".a") or name.endswith(".b")) and (".dense." in name or ".wo_a." in name):
        return "lora"
    if name.endswith("norm__weight"):
        return "norm"
    raise DiagnosticError(f"unclassified trainable tensor {name}")


def mix(base: dict[str, Tensor], donor: dict[str, Tensor], groups: frozenset[str]) -> dict[str, Tensor]:
    if set(base) != set(donor):
        raise DiagnosticError("ablation states have different tensor names")
    return {n: (donor if component(n) in groups else base)[n] for n in base}


def variant_states(initial: dict[str, Tensor], ref2500: dict[str, Tensor], ref20000: dict[str, Tensor],
                   fast20000: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    markov = frozenset({"markov"})
    return {"V0": initial, "R2.5": ref2500, "R20": ref20000, "F20": fast20000,
            "L20": mix(ref20000, initial, markov), "H20": mix(initial, ref20000, markov)}


def composition() -> dict[str, str]:
    return {"V0": "loaded initial state (zero-init LoRA B)", "R2.5": "REF 2,500 blocks", "R20": "REF 20,000 blocks",
            "F20": "FAST 20,000 blocks", "L20": "REF 20,000 with markov_head embed+head at the loaded initial value",
            "H20": "loaded initial state with only markov_head embed+head from REF 20,000"}


@dataclass(frozen=True)
class RoundStats:
    """One scored round. Picks are kept in memory for paired comparisons and never written."""
    request: str
    family: str
    accepted: int
    greedy: Tensor
    argmax: Tensor
    prefix: Tensor
    rb_lower: float
    rb_upper: float
    fields: dict[str, Tensor]
    teacher: dict[str, Tensor]
    temperature_kl: Tensor
    identity_checked: int
    near_ties: int


def teacher_probability(ids: Tensor, p: Tensor, tail: Tensor, token: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """p_teacher(token) per position: exact inside the top-20; outside it 0 (lower) or min(tail, p_20) (upper)."""
    match = ids == token[:, None]
    inside = match.any(-1)
    lower = (p * match).sum(-1)
    upper = torch.where(inside, lower, torch.minimum(tail, p[:, -1]))
    return lower, upper, ~inside


def round_stats(logits: Tensor, kl: Tensor, ids: Tensor, logp: Tensor, tail: Tensor, labels: Tensor, greedy: Tensor,
                accepted: int, request: str, family: str,
                temperatures: tuple[float, ...] = TEMPERATURES) -> RoundStats:
    """All per-position statistics of one round from its teacher-forced logits [5, V]; float64."""
    z = logits.double()
    ids, labels, greedy = ids.long(), labels.long(), greedy.long()
    log_z = z.logsumexp(-1)
    logq = z - log_z[:, None]
    best = z.topk(2, -1)
    argmax = z.argmax(-1)
    p = logp.double().exp().masked_fill(ids < 0, 0)
    tail = tail.double()
    top = z.gather(-1, ids) - log_z[:, None]
    complement = z.scatter(-1, ids, -torch.inf).logsumexp(-1) - log_z
    kl_top = (torch.xlogy(p, p) - p * top.masked_fill(p == 0, 0)).sum(-1)
    kl_tail = torch.xlogy(tail, tail) - tail * complement.masked_fill(tail == 0, 0)
    if not torch.allclose(kl_top + kl_tail, kl.double(), rtol=1e-12, atol=1e-12):
        raise DiagnosticError("KL parts do not sum to the evaluator's bucket KL")
    pp_lower, pp_upper, _ = teacher_probability(ids, p, tail, argmax)
    pg_lower, pg_upper, _ = teacher_probability(ids, p, tail, greedy)
    hits = greedy == labels
    prefix = torch.cat((torch.ones(1, dtype=torch.float64), hits.double().cumprod(0)[:-1]))
    if int(hits.double().cumprod(0).sum()) != accepted:
        raise DiagnosticError("greedy prefix differs from match_length")
    checked = prefix.bool()
    near = 0
    for i in torch.nonzero(checked & (greedy != argmax)).flatten().tolist():
        if float(z[i, argmax[i]] - z[i, greedy[i]]) > NEAR_TIE:
            raise DiagnosticError("own-draft pick differs from the teacher-forced argmax on a prefix-matched position")
        near += 1
    temperature_kl = torch.stack([kl.double() if t == 1. else bucket_kl(z / t, ids, logp, tail) for t in temperatures])
    fields = {"y_match": (argmax == labels).double(), "t_match": (argmax == ids[:, 0]).double(),
              "pp_lower": pp_lower, "pp_upper": pp_upper, "q1": (best.values[:, 0] - log_z).exp(),
              "entropy": -(logq.exp() * logq).sum(-1), "gap": best.values[:, 0] - best.values[:, 1],
              "kl": kl.double(), "kl_top": kl_top, "kl_tail": kl_tail,
              "survive": torch.tensor([float(accepted > i) for i in range(5)], dtype=torch.float64),
              "pick_outside": (~(ids == argmax[:, None]).any(-1)).double(),
              "greedy_outside": (~(ids == greedy[:, None]).any(-1)).double()}
    teacher = {"p1": p[:, 0], "entropy": -(torch.xlogy(p, p).sum(-1) + torch.xlogy(tail, tail)),
               "coverage": p.sum(-1), "hit": (ids[:, 0] == labels).double()}
    return RoundStats(request, family, accepted, greedy, argmax, prefix, float((prefix * pg_lower).sum()),
                      float((prefix * pg_upper).sum()), fields, teacher, temperature_kl, int(checked.sum()), near)


def selected_rounds(data: D2Blocks, k2: Path, keys: list[RoundKey]) -> list[tuple[int, RoundKey]]:
    """pilot_eval.score's population rule, evaluated once per split."""
    expected = {(k["request"], k["P"]) for k in keys}
    selected = [(index, key) for index, key in complete_rounds(data, k2) if (key["request"], key["P"]) in expected]
    if not keys or len(expected) != len(keys) or [key for _, key in selected] != keys:
        raise ValueError("pilot scored keys differ from the frozen sample")
    return selected


@torch.no_grad()
def score_variant(model: Drafter, data: D2Blocks, selected: list[tuple[int, RoundKey]], keys: list[RoundKey],
                  chunk: int = 8192) -> tuple[Metrics, list[RoundStats]]:
    """pilot_eval.score's E[A]/KL path (minus the loss), plus the private per-round statistics."""
    if not model.reference_math or model.fast_training:
        raise ValueError("pilot evaluation requires qualified reference math")
    total_kl = torch.zeros(5, dtype=torch.float64)
    total_accepted = 0.
    actual: list[RoundKey] = []
    kl_rows: list[list[float]] = []
    request_totals: dict[str, RequestTotal] = {}
    rounds: list[RoundStats] = []
    for index, key in selected:
        request, pos = key["request"], key["P"]
        block = data[index]
        if block["request"] != request or int(block["inputs"]["P"][0]) != pos:
            raise ValueError("pilot scored keys differ from the frozen sample")
        targets = {k: t.to(model.activation_device) for k, t in block["targets"].items()}
        if not bool(targets["position_mask"].all()):
            raise ValueError("sampled pilot round has incomplete targets")
        inp = {k: t.to(model.activation_device) for k, t in block["inputs"].items()}
        h, hx = model.hidden_from(inp)
        picks, _ = model.greedy(h, hx, inp["anchor"], chunk=chunk)
        candidate_n = match_length(picks[0].tolist(), targets["labels"][0].tolist())
        if candidate_n is None:
            raise ValueError("indeterminate pilot replay round")
        logits = torch.cat([model.logit_chunk(h, targets["previous"], lo, min(lo+chunk, model.c.vocab))
                            for lo in range(0, model.c.vocab, chunk)], -1)[0]
        kl = bucket_kl(logits, targets["top_ids"][0], targets["top_logp"][0], targets["tail_mass"][0]).cpu()
        if not torch.isfinite(kl).all():
            raise FloatingPointError("nonfinite pilot evaluation")
        if request not in data.request_families:
            raise ValueError("pilot request missing source family")
        family = data.request_families[request]
        aggregate = request_totals.setdefault(request, {"request_sha256": digest(request), "family": family,
                                                        "task_label": data.request_tasks[request], "rounds": 0,
                                                        "accepted_sum": 0.})
        aggregate["rounds"] += 1
        aggregate["accepted_sum"] += candidate_n
        kl_rows.append(kl.tolist())
        total_kl += kl
        total_accepted += candidate_n
        actual.append(key)
        rounds.append(round_stats(logits.cpu(), kl, targets["top_ids"][0].cpu(), targets["top_logp"][0].cpu(),
                                  targets["tail_mass"][0].cpu(), targets["labels"][0].cpu(), picks[0].cpu(),
                                  candidate_n, request, family))
    if actual != keys:
        raise ValueError("pilot scored keys differ from the frozen sample")
    n = len(actual)
    metrics: Metrics = {"requests": len({k["request"] for k in actual}), "rounds": n,
                        "population_sha256": key_hash(actual), "kl_by_position": (total_kl/n).tolist(),
                        "loss": math.nan,  # not recomputed; never written or compared
                        "accepted_length": total_accepted/n,
                        "kl_median_by_position": [statistics.median(row[i] for row in kl_rows) for i in range(5)],
                        "request_totals": [request_totals[r] for r in sorted(request_totals)]}
    return metrics, rounds


def check_reproduction(name: str, metrics: Metrics, expected: Metrics) -> dict[str, Any]:
    """E[A] and the scored population must equal the receipt exactly; KL is recorded and grossly bounded."""
    mine, theirs = cast(dict[str, Any], metrics), cast(dict[str, Any], expected)
    exact = {key: mine[key] == theirs[key]
             for key in ("accepted_length", "rounds", "requests", "population_sha256", "request_totals")}
    if not all(exact.values()):
        raise DiagnosticError(f"{name}: E[A]/population differs from the pilot receipt {sorted(k for k, v in exact.items() if not v)}; "
                              f"diagnostic {metrics['accepted_length']!r} vs receipt {expected['accepted_length']!r}")
    mean_diff = max(abs(a-b) for a, b in zip(metrics["kl_by_position"], expected["kl_by_position"]))
    median_diff = max(abs(a-b) for a, b in zip(metrics["kl_median_by_position"], expected["kl_median_by_position"]))
    if mean_diff > 1e-6 or median_diff > 1e-6:
        raise DiagnosticError(f"{name}: KL differs from the pilot receipt by {max(mean_diff, median_diff)}")
    return {"accepted_length": metrics["accepted_length"], "receipt_accepted_length": expected["accepted_length"],
            "exact": exact, "kl_mean_max_abs_diff": mean_diff, "kl_median_max_abs_diff": median_diff,
            "kl_bit_identical": metrics["kl_by_position"] == expected["kl_by_position"]
                                and metrics["kl_median_by_position"] == expected["kl_median_by_position"]}


def cluster_mean(values: Tensor, clusters: list[str]) -> dict[str, float | None]:
    """Mean with a cluster-robust (source-family) standard error; None when fewer than two clusters."""
    n = values.numel()
    if n == 0:
        return {"mean": None, "se": None}
    mean = float(values.double().mean())
    sums: dict[str, float] = {}
    counts: Counter[str] = Counter()
    for v, c in zip(values.double().tolist(), clusters):
        sums[c] = sums.get(c, 0.) + v
        counts[c] += 1
    groups = len(sums)
    if groups < 2:
        return {"mean": mean, "se": None}
    variance = groups/(groups-1) * math.fsum((sums[c]-counts[c]*mean)**2 for c in sums)
    return {"mean": mean, "se": math.sqrt(variance)/n}


def paired_ratio(a0: Tensor, b0: Tensor, a1: Tensor, b1: Tensor, clusters: list[str]) -> dict[str, float | None]:
    """Δ of ratios Σa1/Σb1 − Σa0/Σb0 with a linearised cluster-robust SE."""
    if float(b0.sum()) == 0 or float(b1.sum()) == 0:
        return {"mean": None, "se": None}
    r0, r1 = float(a0.sum()/b0.sum()), float(a1.sum()/b1.sum())
    n = a0.numel()
    u = (a1 - r1*b1)/float(b1.mean()) - (a0 - r0*b0)/float(b0.mean())
    se = cluster_mean(u, clusters)["se"] if n else None
    return {"mean": r1-r0, "se": se}


def stack(rounds: list[RoundStats], field: str) -> Tensor:
    return torch.stack([r.fields[field] for r in rounds])


def column_median(x: Tensor) -> list[float]:
    """statistics.median per column, the convention of pilot_eval's kl_median_by_position."""
    return [float(statistics.median(column)) for column in x.T.tolist()]


def summary(x: Tensor) -> dict[str, Any]:
    return {"mean": x.mean(0).tolist(), "median": column_median(x), "pooled_mean": float(x.mean())}


def median(x: Tensor) -> float:
    return float(statistics.median(x.tolist()))


def variant_metrics(rounds: list[RoundStats]) -> dict[str, Any]:
    prefix = torch.stack([r.prefix for r in rounds])
    survive = stack(rounds, "survive")
    temps = torch.stack([r.temperature_kl for r in rounds])  # [n, T, 5]
    mean_t = temps.mean(0)
    return {"EA": math.fsum(r.accepted for r in rounds)/len(rounds),
            "EA_RB_lower": math.fsum(r.rb_lower for r in rounds)/len(rounds),
            "EA_RB_upper": math.fsum(r.rb_upper for r in rounds)/len(rounds),
            "reach": prefix.mean(0).tolist(),
            "acc_given_reach": (survive.sum(0)/prefix.sum(0).clamp_min(1)).tolist(),
            "positions": {f: summary(stack(rounds, f)) for f in (*POSITION_FIELDS, "pick_outside", "greedy_outside")},
            "temperature": {"tau": list(TEMPERATURES), "kl_mean": mean_t.tolist(), "kl_median": [column_median(t) for t in temps.transpose(0, 1)],
                            "kl_mean_pooled": mean_t.mean(-1).tolist(),
                            "tau_star": [TEMPERATURES[int(i)] for i in mean_t.argmin(0)],
                            "tau_star_pooled": TEMPERATURES[int(mean_t.mean(-1).argmin())]},
            "identity": {"checked": sum(r.identity_checked for r in rounds), "near_ties": sum(r.near_ties for r in rounds)}}


def teacher_metrics(rounds: list[RoundStats]) -> dict[str, Any]:
    clusters = [r.family for r in rounds]
    teacher = {f: torch.stack([r.teacher[f] for r in rounds]) for f in ("p1", "entropy", "coverage", "hit")}
    gap = teacher["p1"] - teacher["hit"]
    return {**{f: summary(x) for f, x in teacher.items()},
            "calibration_gap": [cluster_mean(gap[:, i], clusters) for i in range(5)]}


def compare(base: list[RoundStats], other: list[RoundStats], base_metrics: dict[str, Any],
            other_metrics: dict[str, Any]) -> dict[str, Any]:
    """Paired variant − V0 statistics on identical rounds."""
    if [(r.request, r.family) for r in base] != [(r.request, r.family) for r in other]:
        raise DiagnosticError("paired comparison on different rounds")
    clusters = [r.family for r in base]
    out: dict[str, Any] = {}
    scalars: dict[str, Callable[[RoundStats], float]] = {
        "EA": lambda r: float(r.accepted), "EA_RB_lower": lambda r: r.rb_lower, "EA_RB_upper": lambda r: r.rb_upper}
    for name, get in scalars.items():
        out[name] = cluster_mean(torch.tensor([get(r) for r in other], dtype=torch.float64)
                                 - torch.tensor([get(r) for r in base], dtype=torch.float64), clusters)
    deltas = {f: stack(other, f) - stack(base, f) for f in POSITION_FIELDS}
    out["per_position"] = {f: [cluster_mean(d[:, i], clusters) for i in range(5)] for f, d in deltas.items()}
    out["pooled"] = {f: cluster_mean(d.mean(-1), clusters) for f, d in deltas.items()}
    out["pooled_1_4"] = {f: cluster_mean(d[:, 1:].mean(-1), clusters) for f, d in deltas.items()}
    p0, p1 = torch.stack([r.prefix for r in base]), torch.stack([r.prefix for r in other])
    s0, s1 = stack(base, "survive"), stack(other, "survive")
    out["acc_given_reach"] = [paired_ratio(s0[:, i], p0[:, i], s1[:, i], p1[:, i], clusters) for i in range(5)]
    kl0, kl1 = stack(base, "kl"), stack(other, "kl")
    y0, y1 = stack(base, "y_match").bool(), stack(other, "y_match").bool()
    out["kl_improved_fraction"] = (kl1 < kl0).double().mean(0).tolist()
    out["kl_worsened_fraction"] = (kl1 > kl0).double().mean(0).tolist()
    out["match_up_fraction"] = (~y0 & y1).double().mean(0).tolist()
    out["match_down_fraction"] = (y0 & ~y1).double().mean(0).tolist()
    a0 = torch.tensor([r.accepted for r in base])
    a1 = torch.tensor([r.accepted for r in other])
    out["accepted_rose_fraction"] = float((a1 > a0).double().mean())
    out["accepted_fell_fraction"] = float((a1 < a0).double().mean())
    changed = torch.stack([r.argmax for r in other]) != torch.stack([r.argmax for r in base])
    gap0 = stack(base, "gap")
    pp = deltas["pp_lower"]
    out["argmax_changed"] = []
    for i in range(5):
        rows = changed[:, i]
        members = [c for c, keep in zip(clusters, rows.tolist()) if keep]
        out["argmax_changed"].append({
            "rounds": int(rows.sum()), "delta_pick_probability": cluster_mean(pp[rows, i], members),
            "baseline_median_gap_changed": median(gap0[rows, i]) if bool(rows.any()) else None,
            "baseline_median_gap_all": median(gap0[:, i])})
    right = stack(base, "t_match").bool()
    for label, mask in (("kl_change_baseline_right", right), ("kl_change_baseline_wrong", ~right)):
        out[label] = []
        for i in range(5):
            rows = mask[:, i]
            members = [c for c, keep in zip(clusters, rows.tolist()) if keep]
            out[label].append({"rounds": int(rows.sum()), **cluster_mean(deltas["kl"][rows, i], members)})
    out["median_shift_pooled"] = {f: sum(other_metrics["positions"][f]["median"][i] - base_metrics["positions"][f]["median"][i]
                                         for i in range(5))/5 for f in ("kl", "kl_top", "kl_tail")}
    base_t = base_metrics["temperature"]
    at_one = base_t["kl_mean_pooled"][TEMPERATURES.index(1.)]
    best = min(base_t["kl_mean_pooled"])
    reduction = at_one - other_metrics["temperature"]["kl_mean_pooled"][TEMPERATURES.index(1.)]
    out["temperature_recovery"] = (at_one - best)/reduction if reduction > 0 else None
    return out


def diagnose_split(model: Drafter, data: D2Blocks, selected: list[tuple[int, RoundKey]], keys: list[RoundKey],
                   states: dict[str, dict[str, Tensor]], expected: dict[str, Metrics], chunk: int = 8192,
                   log: Callable[[str], None] = print) -> dict[str, Any]:
    """Score every variant in VARIANTS order; a shared variant that misses its receipt value stops the run."""
    if tuple(states) != VARIANTS or set(expected) != {v for v in VARIANTS if v in SHARED}:
        raise DiagnosticError("diagnostic variants or receipt values incomplete")
    per_round: dict[str, list[RoundStats]] = {}
    variants: dict[str, Any] = {}
    reproduction: dict[str, Any] = {}
    teacher: dict[str, Any] | None = None
    for name in VARIANTS:
        apply_trainables(model, states[name])
        metrics, rounds = score_variant(model, data, selected, keys, chunk)
        if name in expected:
            reproduction[name] = check_reproduction(name, metrics, expected[name])
        current = teacher_metrics(rounds)
        if teacher is None:
            teacher = current
        elif current != teacher:
            raise DiagnosticError("teacher statistics changed between passes")
        per_round[name] = rounds
        variants[name] = {"composition": composition()[name], "metrics": variant_metrics(rounds),
                          "request_totals": metrics["request_totals"]}
        if name != "V0":
            variants[name]["vs_V0"] = compare(per_round["V0"], rounds, variants["V0"]["metrics"], variants[name]["metrics"])
        log(json.dumps({"variant": name, "EA": metrics["accepted_length"],
                        "EA_RB_lower": variants[name]["metrics"]["EA_RB_lower"],
                        "reproduced": name in expected}))
    assert teacher is not None
    families = {r.family for r in per_round["V0"]}
    result = {"rounds": len(keys), "requests": len({k["request"] for k in keys}), "families": len(families),
              "cluster_key": "data.request_families (source-document components; prompt identity when source-less)",
              "teacher": teacher, "variants": variants, "reproduction": reproduction}
    result["readings"] = readings(result)
    return result


# ------------------------------------------------------------------ mechanical reading of DIAG.md §Hypotheses


def negative(x: dict[str, Any]) -> bool:
    return x["mean"] is not None and x["se"] is not None and x["mean"] < 0 and -x["mean"] >= 2*x["se"]


def positive(x: dict[str, Any]) -> bool:
    return x["mean"] is not None and x["se"] is not None and x["mean"] > 0 and x["mean"] >= 2*x["se"]


def verdict(confirmed: bool, ruled_out: bool, conditions: dict[str, Any]) -> dict[str, Any]:
    if confirmed and ruled_out:
        raise DiagnosticError("reading both confirmed and ruled out")
    return {"verdict": "CONFIRMED" if confirmed else "RULED OUT" if ruled_out else "UNRESOLVED", "conditions": conditions}


def readings(result: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered rules applied to one split, verbatim; the coordinator confirms the reading."""
    v = result["variants"]
    vs = {name: v[name]["vs_V0"] for name in VARIANTS if name != "V0"}
    r20 = vs["R20"]
    out: dict[str, Any] = {}
    ea, rb_lo, rb_hi = r20["EA"], r20["EA_RB_lower"], r20["EA_RB_upper"]
    h0_confirmed = (not negative(rb_lo)) and abs(rb_lo["mean"]) < .5*abs(ea["mean"])
    out["H0"] = verdict(h0_confirmed, negative(rb_lo) and negative(rb_hi),
                        {"EA": ea, "EA_RB_lower": rb_lo, "EA_RB_upper": rb_hi})

    def damage(name: str) -> dict[str, Any]:
        return cast(dict[str, Any], vs[name]["per_position"]["pp_lower"][0] if h0_confirmed else vs[name]["EA_RB_lower"])

    pooled = r20["pooled"]
    right, wrong = r20["kl_change_baseline_right"][0], r20["kl_change_baseline_wrong"][0]
    h1 = {"entropy_up": positive(pooled["entropy"]), "top1_down": negative(pooled["q1"]),
          "pick_down": r20["per_position"]["pp_lower"][0]["mean"] < 0
                       and (negative(r20["per_position"]["pp_lower"][0]) or negative(rb_lo)),
          "mean_kl_down": pooled["kl"]["mean"] < 0,
          "right_kl_up": right["mean"] is not None and right["mean"] > 0,
          "wrong_kl_down": wrong["mean"] is not None and wrong["mean"] < 0,
          "tail_carrier": r20["median_shift_pooled"]["kl_tail"] > r20["median_shift_pooled"]["kl_top"]}
    out["H1"] = verdict(all(h1[k] for k in h1 if k != "tail_carrier"),
                        pooled["entropy"]["mean"] <= 0 or pooled["q1"]["mean"] >= 0, h1)
    gaps = result["teacher"]["calibration_gap"]
    resolved = [negative(g) or positive(g) for g in gaps]
    out["H2"] = verdict(any(resolved), not any(resolved),
                        {"resolved_positions": resolved, "calibration_gap": gaps,
                         "note": "split reading; the merge requires both splits for RULED OUT"})
    d20 = damage("R20")
    readable = negative(d20)
    d = {name: damage(name)["mean"] for name in ("R2.5", "L20", "H20")}
    full = d20["mean"]
    head = readable and d["H20"] <= 2/3*full and d["L20"] >= full/3
    trunk = readable and d["L20"] <= 2/3*full and d["H20"] >= full/3
    both = d["H20"] <= full/3 and d["L20"] <= full/3
    neither = d["H20"] > full/3 and d["L20"] > full/3
    out["H3"] = verdict(head or trunk, readable and not (head or trunk) and (both or neither),
                        {"readable": readable, "D_R20": d20, "D_H20": d["H20"], "D_L20": d["L20"],
                         "head": head, "trunk": trunk, "both_reproduce": both, "neither_reproduces": neither,
                         "measure": "pp_lower[0]" if h0_confirmed else "EA_RB_lower"})
    pos = r20["per_position"]
    later = any(negative(x) for x in r20["acc_given_reach"][1:]) or negative(r20["pooled_1_4"]["pp_lower"])
    out["H4"] = verdict(not negative(pos["y_match"][0]) and not negative(pos["pp_lower"][0]) and later,
                        negative(pos["pp_lower"][0]) and abs(pos["pp_lower"][0]["mean"]) >= abs(r20["pooled_1_4"]["pp_lower"]["mean"]),
                        {"y_match_0": pos["y_match"][0], "pp_0": pos["pp_lower"][0], "pp_1_4": r20["pooled_1_4"]["pp_lower"],
                         "acc_given_reach": r20["acc_given_reach"],
                         "identity": {name: v[name]["metrics"]["identity"] for name in VARIANTS}})
    early = vs["R2.5"]
    flips = early["argmax_changed"][0]["delta_pick_probability"]
    same_sign = (math.copysign(1, early["pooled"]["q1"]["mean"]) == math.copysign(1, pooled["q1"]["mean"])
                 and math.copysign(1, early["pooled"]["entropy"]["mean"]) == math.copysign(1, pooled["entropy"]["mean"]))
    h5 = readable and d["R2.5"] <= 2/3*full and flips["mean"] is not None and flips["mean"] <= 0 and same_sign
    out["H5"] = verdict(h5, readable and (d["R2.5"] >= full/3 or positive(flips)),
                        {"readable": readable, "D_R2.5": d["R2.5"], "D_R20": full, "early_flip_delta_p0": flips,
                         "same_sign_q1_entropy": same_sign})
    t0 = v["V0"]["metrics"]["temperature"]
    t20 = v["R20"]["metrics"]["temperature"]
    recovery = r20["temperature_recovery"]
    h6 = (t0["tau_star"][0] != 1. and t0["tau_star_pooled"] != 1. and recovery is not None and recovery >= .5
          and abs(t20["tau_star_pooled"]-1) < abs(t0["tau_star_pooled"]-1))
    out["H6"] = verdict(h6, t0["tau_star"][0] == 1. or t0["tau_star_pooled"] == 1. or (recovery is not None and recovery < .25),
                        {"tau_star_V0_position0": t0["tau_star"][0], "tau_star_V0_pooled": t0["tau_star_pooled"],
                         "tau_star_R20_pooled": t20["tau_star_pooled"], "recovery": recovery})
    return out


def merge_readings(parts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Cross-split readings: H2 on both splits and H7 (dev vs heldout_legal)."""
    dev, legal = (parts[s]["variants"]["R20"]["vs_V0"]["EA_RB_lower"] for s in DEV_POPULATION)
    h2 = [parts[s]["readings"]["H2"]["verdict"] for s in DEV_POPULATION]
    out = {"H2": verdict("CONFIRMED" in h2, all(x == "RULED OUT" for x in h2), {"splits": dict(zip(DEV_POPULATION, h2))})}
    resolved = None
    if dev["se"] is not None and legal["se"] is not None:
        resolved = dev["mean"] - legal["mean"] > 2*math.sqrt(dev["se"]**2 + legal["se"]**2)
    out["H7"] = verdict(dev["mean"] >= 0 and negative(legal) and bool(resolved),
                        negative(dev) or (dev["mean"] <= 0 and not resolved),
                        {"dev": dev, "heldout_legal": legal, "difference_resolved": resolved})
    return out


# ------------------------------------------------------------------ inputs, identities and CLI


def validate_pilot(plan: Path, receipt_path: Path, evaluation_path: Path) -> tuple[Spec, Evaluation, dict[str, Any]]:
    """Bind the plan, receipt and evaluation together; the receipt-hashed modules must be byte-unchanged."""
    receipt = cast(dict[str, Any], json.loads(receipt_path.read_text()))
    evaluation = cast(Evaluation, json.loads(evaluation_path.read_text()))
    plan_hash = sha256(plan)
    if (receipt.get("schema") != "dsv41.d2.pilot-receipt.v1" or receipt.get("plan_sha256") != plan_hash
            or receipt.get("evaluation_sha256") != sha256(evaluation_path) or receipt.get("evaluation") != evaluation):
        raise DiagnosticError("receipt, evaluation and plan are not the same pilot")
    spec = cast(Spec, json.loads(plan.read_text()))
    hashed = spec.get("source_sha256")
    current = sources()
    if not isinstance(hashed, dict) or any(current.get(name) != value for name, value in hashed.items()):
        raise DiagnosticError("a module the pilot receipt hashes has changed")
    if not set(current) - set(hashed) <= ALLOWED_NEW_SOURCES:
        raise DiagnosticError(f"unexpected new source modules {sorted(set(current) - set(hashed) - ALLOWED_NEW_SOURCES)}")
    validate_spec(cast(Spec, {**spec, "source_sha256": current}), sampled=True)
    if receipt.get("plan") != spec or receipt.get("synthetic") != spec["synthetic"] or receipt.get("sample_sha256") != spec["sample"]["sha256"]:
        raise DiagnosticError("receipt does not bind this plan/sample")
    if (evaluation.get("plan_sha256") != plan_hash or evaluation.get("evaluator") != EVALUATOR
            or evaluation.get("synthetic") != spec["synthetic"]):
        raise DiagnosticError("evaluation is not this plan's reference pilot evaluation")
    splits = population_splits(spec["sample"]["keys"])
    if splits != DEV_POPULATION:
        raise DiagnosticError("diagnostic requires the amendment-7 dev population")
    validate_sample_evaluation(spec, evaluation, splits)
    return spec, evaluation, receipt


def expected_metrics(evaluation: Evaluation, split: str) -> dict[str, Metrics]:
    out: dict[str, Metrics] = {}
    for name, source in SHARED.items():
        out[name] = evaluation["baseline"][split] if source is None else evaluation["arms"][source[0]][source[1]][split]
    return out


def checkpoint_state(record: Checkpoint) -> dict[str, Tensor]:
    state, actual = read_checkpoint(Path(record["path"]))
    if actual != record["sha256"]:
        raise ValueError("checkpoint changed during evaluation")
    return cast(dict[str, Tensor], state["trainable"])


def run(a: argparse.Namespace) -> None:
    from .safety import assert_gpu_lease
    assert_gpu_lease()
    spec, evaluation, _ = validate_pilot(a.plan, a.receipt, a.evaluation)
    sample = spec["sample"]
    k2 = {"dev": a.k2_dev, "heldout_legal": a.k2_legal}
    data = verify_sample_draw(spec, a.manifest, k2)
    train_data = D2Blocks(a.manifest, "train")
    verify_units(spec, train_data)
    arms = {name: read_arm(a.root, name, spec, a.plan, len(train_data)) for name in ("REF", "FAST")}
    for name, arm in arms.items():
        if arm["checkpoints"] != evaluation["checkpoints"][name]:
            raise DiagnosticError("arm checkpoints differ from the receipt's evaluated checkpoints")
    keys = sample["keys"][a.split]
    selected = selected_rounds(data[a.split], k2[a.split], keys)
    torch.manual_seed(SEED)
    model = load(a.ckpt, "cuda", rank=32, reference_math=True).eval()
    if model.frozen_identity != spec["frozen_identity"] or initialization(model) != spec["initialization_sha256"]:
        raise ValueError("pilot evaluator source/init mismatch")
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    for name in initial:
        component(name)
    states = variant_states(initial, checkpoint_state(arms["REF"]["checkpoints"]["2500"]),
                            checkpoint_state(arms["REF"]["checkpoints"]["20000"]),
                            checkpoint_state(arms["FAST"]["checkpoints"]["20000"]))
    result = diagnose_split(model, data[a.split], selected, keys, states, expected_metrics(evaluation, a.split),
                            log=lambda line: print(line, flush=True))
    identity = {"schema": PART_SCHEMA, "tool": TOOL, "split": a.split, "split_role": evaluation.get("split_roles", {}).get(a.split),
                "classification": "MEASURED", "synthetic": spec["synthetic"], "plan_sha256": sha256(a.plan),
                "receipt_sha256": sha256(a.receipt), "evaluation_sha256": sha256(a.evaluation),
                "sample_sha256": sample["sha256"], "k2_sha256": sample["k2_sha256"], "manifest_sha256": spec["manifest_sha256"],
                "checkpoints": {f"{arm}/{n}": arms[arm]["checkpoints"][n] for arm, n in (("REF", "2500"), ("REF", "20000"), ("FAST", "20000"))},
                "source": {"git": os.environ.get("D2_GIT_SHA", "uncommitted-snapshot"), "pilot_diag_sha256": sha256(Path(__file__))},
                "temperatures": list(TEMPERATURES), "near_tie_logit_gap": NEAR_TIE}
    write_json(a.out, {**identity, **result})


def merge(paths: list[Path], out: Path) -> None:
    """CPU-only join of the two split parts plus the cross-split readings."""
    parts: dict[str, dict[str, Any]] = {}
    for path in paths:
        part = cast(dict[str, Any], json.loads(path.read_text()))
        split = part.get("split")
        if part.get("schema") != PART_SCHEMA or part.get("tool") != TOOL or split not in DEV_POPULATION or split in parts:
            raise DiagnosticError("diagnostic part identity/split mismatch")
        parts[cast(str, split)] = part
    if set(parts) != set(DEV_POPULATION):
        raise DiagnosticError("merge requires exactly the dev and heldout_legal parts")
    shared = ("plan_sha256", "receipt_sha256", "evaluation_sha256", "sample_sha256", "checkpoints", "source", "synthetic")
    first, second = (parts[s] for s in DEV_POPULATION)
    if any(first[k] != second[k] for k in shared):
        raise DiagnosticError("diagnostic parts disagree on pilot identity or source")
    write_json(out, {"schema": SCHEMA, "tool": TOOL, **{k: first[k] for k in shared}, "splits": parts,
                     "cross_split_readings": merge_readings(parts)})


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", required=True, type=Path)
    for n in ("plan", "receipt", "evaluation", "manifest", "root", "k2-dev", "k2-legal"):
        p.add_argument("--"+n, type=Path)
    p.add_argument("--ckpt")
    p.add_argument("--split", choices=DEV_POPULATION)
    p.add_argument("--merge-parts", nargs=2, type=Path, help="CPU-only merge of the dev and heldout_legal parts")
    a = p.parse_args()
    torch.set_num_threads(8)
    if a.merge_parts:
        merge(a.merge_parts, a.out)
        return
    if any(getattr(a, n) is None for n in ("plan", "receipt", "evaluation", "manifest", "root", "k2_dev", "k2_legal",
                                           "ckpt", "split")):
        p.error("diagnosis requires --plan --receipt --evaluation --manifest --root --k2-dev --k2-legal --ckpt --split")
    run(a)


if __name__ == "__main__":
    main()
