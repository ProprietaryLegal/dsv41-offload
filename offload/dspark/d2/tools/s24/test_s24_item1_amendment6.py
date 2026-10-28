"""S24 item 1: Amendment 6 decision rule, probed against an independent exact oracle.

The oracle is written from the amendment text (documents/plans/2026-10-23-expert-continuation.md §B), not from
pilot_verdict.py: point estimates are exact Fractions over integer accepted-prefix counts, the paired two-stage
(family, then request) bootstrap is re-implemented, and percentiles use exact rational linear interpolation.
Synthetic aggregates only; no real data.
"""
from __future__ import annotations

import copy
from decimal import Decimal
from fractions import Fraction
import math
import random
from typing import cast

import pytest

from offload.dspark.d2.train.pilot import ARMS, Spec, digest
from offload.dspark.d2.train.pilot_eval import EVALUATOR, Evaluation, Metrics, SPLITS
from offload.dspark.d2.train.pilot_sample import RoundKey, draw_sample, key_hash
from offload.dspark.d2.train.pilot_statistics import DEV_SPLITS, paired_interval
from offload.dspark.d2.train.pilot_verdict import decision, outcome

POINTS = [2500, 5000, 10000, 20000]


# ---------------------------------------------------------------- synthetic evidence ---------------------------------
def build(rng: random.Random, n_req: dict[str, int], families: dict[str, int]) -> tuple[Spec, Evaluation, dict[str, list[tuple[str, str, int]]]]:
    """Population of complete rounds per split; requests carry families; sample frozen like Amendment 4."""
    population: dict[str, list[RoundKey]] = {}
    fam_of: dict[str, str] = {}
    for s in SPLITS:
        keys: list[RoundKey] = []
        for r in range(n_req[s]):
            req = f"{s}-req{r}"
            fam_of[req] = f"{s}-fam{r % families[s]}" if families[s] > 1 else s
            keys += [{"request": req, "P": 3+6*j} for j in range(rng.randint(1, 12))]
        population[s] = keys
    sample = draw_sample("manifest", population, dict.fromkeys(SPLITS, "k2"))
    spec = cast(Spec, {"synthetic": False, "manifest_sha256": "manifest", "sample": sample, "checkpoints": POINTS})
    layout: dict[str, list[tuple[str, str, int]]] = {}
    for s in SPLITS:
        counts: dict[str, int] = {}
        for k in sample["keys"][s]:
            counts[k["request"]] = counts.get(k["request"], 0)+1
        layout[s] = [(req, fam_of[req], counts[req]) for req in sorted(counts)]
    ev = cast(Evaluation, {"schema": "dsv41.d2.pilot-evaluation.v1", "synthetic": False, "evaluator": EVALUATOR,
                           "split_roles": DEV_SPLITS, "k2_sha256": sample["k2_sha256"], "sample_sha256": sample["sha256"],
                           "baseline": {}, "arms": {a: {str(n): {} for n in POINTS} for a in ARMS},
                           "checkpoints": {a: {str(n): {} for n in POINTS} for a in ARMS}})
    return spec, ev, layout


def metrics(spec: Spec, split: str, layout: list[tuple[str, str, int]], accepted: list[int], kl_median: list[float]) -> Metrics:
    """accepted[i] = integer sum of accepted draft-prefix tokens (0..5 per round) of request i, as the scorer sums."""
    rounds = sum(n for _, _, n in layout)
    return {"requests": len(layout), "rounds": rounds, "population_sha256": key_hash(spec["sample"]["keys"][split]),
            "kl_by_position": list(kl_median), "loss": 1., "accepted_length": float(sum(accepted))/rounds,
            "kl_median_by_position": list(kl_median),
            "request_totals": [{"request_sha256": digest(req), "family": fam, "task_label": "task", "rounds": n, "accepted_sum": float(acc)}
                               for (req, fam, n), acc in zip(layout, accepted)]}


# ---------------------------------------------------------------- independent oracle ---------------------------------
def exact_percentile(xs: list[Fraction], q: int) -> Fraction:
    """Linear interpolation between order statistics (numpy 'linear' / Hyndman-Fan 7), exact."""
    xs = sorted(xs)
    h = Fraction(q, 100)*(len(xs)-1)
    lo = h.numerator//h.denominator
    return xs[lo] if lo+1 >= len(xs) else xs[lo]+(h-lo)*(xs[lo+1]-xs[lo])


def oracle_interval(left: Metrics, right: Metrics) -> tuple[Fraction, Fraction, Fraction]:
    a = {r["request_sha256"]: r for r in left["request_totals"]}
    b = {r["request_sha256"]: r for r in right["request_totals"]}
    by_family: dict[str, list[str]] = {}
    for key in sorted(a):
        by_family.setdefault(a[key]["family"], []).append(key)
    diff = {k: Fraction(int(a[k]["accepted_sum"]))-Fraction(int(b[k]["accepted_sum"])) for k in a}
    def est(keys: list[str]) -> Fraction:
        return sum((diff[k] for k in keys), Fraction(0))/sum(a[k]["rounds"] for k in keys)
    rng = random.Random(20261024)
    names = sorted(by_family)
    draws = []
    for _ in range(2000):
        chosen: list[str] = []
        for fam in rng.choices(names, k=len(names)):
            chosen += rng.choices(by_family[fam], k=len(by_family[fam]))
        draws.append(est(chosen))
    return est(sorted(a)), exact_percentile(draws, 5), exact_percentile(draws, 95)


def oracle(ev: Evaluation) -> tuple[dict[str, bool], dict[str, bool]]:
    """Independent amendment 6/6b checks and per-failed-check straddles, in exact fractions."""
    checks: dict[str, bool] = {}
    straddles: dict[str, bool] = {}
    for point in ("10000", "20000"):
        for split in SPLITS:
            p = f"{point}/{DEV_SPLITS[split]}"
            base = ev["baseline"][split]
            ref, fast = ev["arms"]["REF"][point][split], ev["arms"]["FAST"][point][split]
            for arm, m in (("REF", ref), ("FAST", fast)):
                d, lo, hi = oracle_interval(m, base)
                checks[f"{p}/{arm}_useful_gain"] = d >= Fraction(1, 20) and lo > 0
                if not checks[f"{p}/{arm}_useful_gain"]:
                    straddles[f"{p}/{arm}_useful_gain"] = lo < 0 < hi or lo < Fraction(1, 20) < hi
            d, lo, hi = oracle_interval(fast, ref)
            checks[f"{p}/non_inferiority"] = lo >= Fraction(-3, 100)
            if not checks[f"{p}/non_inferiority"]:
                straddles[f"{p}/non_inferiority"] = lo < Fraction(-3, 100) < hi
            fa = Fraction(int(sum(r["accepted_sum"] for r in fast["request_totals"])), fast["rounds"])
            ba = Fraction(int(sum(r["accepted_sum"] for r in base["request_totals"])), base["rounds"])
            checks[f"{p}/no_baseline_regression"] = fa-ba >= Fraction(-1, 100)
            checks[f"{p}/KL_median"] = all(Decimal(repr(f))-Decimal(repr(r)) <= Decimal("0.01")
                                           for f, r in zip(fast["kl_median_by_position"], ref["kl_median_by_position"]))
    return checks, straddles


def oracle_verdict(checks: dict[str, bool], straddles: dict[str, bool]) -> tuple[str, bool]:
    failed = [key for key, passed in checks.items() if not passed]
    # Hard regressions and definitive NI always fail. A definitive FAST gain
    # failure also fails wherever REF has demonstrated adaptation. Missing REF
    # gain (and FAST gain in that same cell) retains amendment 6's exception.
    for key in failed:
        if key.endswith(("/KL_median", "/no_baseline_regression")):
            return "FAIL", False
        if key.endswith("/non_inferiority") and not straddles[key]:
            return "FAIL", False
        if key.endswith("/FAST_useful_gain") and checks[key.replace("/FAST_", "/REF_")] and not straddles[key]:
            return "FAIL", False
    if failed and all(straddles.get(key, False) for key in failed):
        return "INCONCLUSIVE", True
    if any(key.endswith("/REF_useful_gain") for key in failed):
        return "INCONCLUSIVE", False
    return ("FAIL" if failed else "PASS"), False


# ---------------------------------------------------------------- scenario generator ---------------------------------
def scenario(seed: int, n_req: dict[str, int], families: dict[str, int]) -> tuple[Spec, Evaluation]:
    """One regime per scenario (so PASS is reachable), with occasional per-cell perturbations near the bounds."""
    rng = random.Random(seed)
    spec, ev, layout = build(rng, n_req, families)
    g_ref = rng.choice([.2, .3]) if rng.random() < .75 else rng.choice([0., .03, .05, .06])
    d_fast = rng.choice([0., .01, -.01, -.02, -.03, -.05, -.1])
    noise = rng.choice([0., .05, .15, .4, 1.])
    kl_shift = rng.choice([0., .005, .01, .011, .03])
    for s in SPLITS:
        base_rate = rng.uniform(1.2, 2.2)
        base = [min(5*n, max(0, round(base_rate*n+rng.gauss(0, .6*n**.5)))) for _, _, n in layout[s]]
        ev["baseline"][s] = metrics(spec, s, layout[s], base, [1.]*5)
        for point in map(str, POINTS):
            perturb = rng.random() < .25
            gr = g_ref+(rng.choice([-.2, -.15, .05]) if perturb else 0.)
            gf = gr+d_fast+(rng.choice([-.05, .03]) if perturb else 0.)
            kl_ref = [round(rng.uniform(.5, 1.), 3) for _ in range(5)]
            kl_fast = [round(k+kl_shift, 3) for k in kl_ref]
            for arm, gain, kl in (("REF", gr, kl_ref), ("FAST", gf, kl_fast), ("FAST-B", gf, kl_fast)):
                acc = [min(5*n, max(0, b+round(gain*n+rng.gauss(0, noise*n**.5))))
                       for b, (_, _, n) in zip(base, layout[s])]
                ev["arms"][arm][point][s] = metrics(spec, s, layout[s], acc, kl)
    return spec, ev


# ---------------------------------------------------------------- probes ---------------------------------------------
def test_s24_bootstrap_matches_independent_exact_oracle() -> None:
    """Point estimate exact; 5th/95th percentile within float rounding of the exact rational percentile."""
    spec, ev = scenario(7, {"acap": 120, "heldout_legal": 90}, {"acap": 1, "heldout_legal": 6})
    for left, right in (("REF", None), ("FAST", None), ("FAST", "REF")):
        for s in SPLITS:
            lm = ev["arms"][left]["10000"][s]
            rm = ev["baseline"][s] if right is None else ev["arms"][right]["10000"][s]
            ci = paired_interval(lm["request_totals"], rm["request_totals"])
            d, lo, hi = oracle_interval(lm, rm)
            assert Fraction(Decimal(repr(ci["difference"]))) == d or abs(Fraction(ci["difference"])-d) < Fraction(1, 10**15)
            assert abs(Fraction(ci["lower"])-lo) < Fraction(1, 10**14) and abs(Fraction(ci["upper"])-hi) < Fraction(1, 10**14)
            assert ci["resamples"] == 2000 and ci["seed"] == 20261024
            assert ci["families"] == (1 if s == "acap" else 6)


@pytest.mark.parametrize("seed", range(150))
def test_s24_verdict_differential_fuzz(seed: int) -> None:
    """Checks equal the oracle exactly; PASS iff the oracle PASSes; INCONCLUSIVE only from REF-no-gain or a straddle."""
    spec, ev = scenario(1000+seed, {"acap": 40, "heldout_legal": 30}, {"acap": 1, "heldout_legal": 4})
    checks = outcome(spec, ev)
    want, straddles = oracle(ev)
    assert checks == want
    verdict, extra = decision(spec, ev, checks)
    assert (verdict, extra) == oracle_verdict(want, straddles)
    assert not (verdict == "PASS" and extra)



def uniform_shift_case(per_round_hundredths: dict[str, int]) -> tuple[Spec, Evaluation]:
    """Every request has rounds divisible by 100 so a per-round shift of k/100 is an integer per request."""
    population: dict[str, list[RoundKey]] = {s: [{"request": f"{s}-r{r}", "P": 3+6*j} for r in range(10) for j in range(100)]
                                             for s in SPLITS}
    sample = draw_sample("manifest", population, dict.fromkeys(SPLITS, "k2"))
    spec = cast(Spec, {"synthetic": False, "manifest_sha256": "manifest", "sample": sample, "checkpoints": POINTS})
    ev = cast(Evaluation, {"schema": "dsv41.d2.pilot-evaluation.v1", "synthetic": False, "evaluator": EVALUATOR,
                           "split_roles": DEV_SPLITS, "k2_sha256": sample["k2_sha256"], "sample_sha256": sample["sha256"],
                           "baseline": {}, "arms": {a: {str(n): {} for n in POINTS} for a in ARMS},
                           "checkpoints": {a: {str(n): {} for n in POINTS} for a in ARMS}})
    for s in SPLITS:
        counts: dict[str, int] = {}
        for k in sample["keys"][s]:
            counts[k["request"]] = counts.get(k["request"], 0)+1
        layout = [(req, f"fam{i % 3}", counts[req]) for i, req in enumerate(sorted(counts))]
        assert sum(n for _, _, n in layout) == 1000 and all(n == 100 for _, _, n in layout)
        base = [180]*len(layout)
        ev["baseline"][s] = metrics(spec, s, layout, base, [1.]*5)
        for point in map(str, POINTS):
            for arm in ARMS:
                shift = per_round_hundredths[arm if arm != "FAST-B" else "FAST"]
                ev["arms"][arm][point][s] = metrics(spec, s, layout, [b+shift for b in base], [.8]*5)
    return spec, ev


def test_s24_exact_boundaries_no_epsilon() -> None:
    # REF +0.05 exactly, FAST +0.02 exactly: useful-gain boundary passes for REF; FAST gain 0.02 < 0.05 fails.
    spec, ev = uniform_shift_case({"REF": 5, "FAST": 5})
    checks = outcome(spec, ev)
    assert all(checks.values()), "gain exactly 0.05 with FAST == REF must PASS"
    assert decision(spec, ev, checks) == ("PASS", False)
    spec, ev = uniform_shift_case({"REF": 8, "FAST": 5})      # FAST - REF exactly -0.03 -> non-inferiority holds
    checks = outcome(spec, ev)
    assert checks["10000/dev_acap/non_inferiority"] and checks["20000/dev_legal/non_inferiority"]
    spec, ev = uniform_shift_case({"REF": 9, "FAST": 5})      # -0.04 -> fails, collapsed interval, no straddle
    checks = outcome(spec, ev)
    assert not checks["10000/dev_acap/non_inferiority"]
    assert decision(spec, ev, checks) == ("FAIL", False)
    spec, ev = uniform_shift_case({"REF": 6, "FAST": 4})      # FAST gain 0.04: point below 0.05 -> FAIL
    checks = outcome(spec, ev)
    assert not checks["10000/dev_acap/FAST_useful_gain"] and decision(spec, ev, checks) == ("FAIL", False)
    spec, ev = uniform_shift_case({"REF": 4, "FAST": 4})      # REF gain 0.04 -> INCONCLUSIVE, no extra seed
    checks = outcome(spec, ev)
    assert decision(spec, ev, checks) == ("INCONCLUSIVE", False)
    # Baseline regression exactly -0.01 passes, -0.02 fails (REF must still gain so verdict is not INCONCLUSIVE).
    spec, ev = uniform_shift_case({"REF": 5, "FAST": -1})
    assert outcome(spec, ev)["10000/dev_acap/no_baseline_regression"]
    spec, ev = uniform_shift_case({"REF": 5, "FAST": -2})
    assert not outcome(spec, ev)["10000/dev_acap/no_baseline_regression"]


def test_s24_kl_median_deficit_exactly_001_passes() -> None:
    spec, ev = uniform_shift_case({"REF": 5, "FAST": 5})
    for point in ("10000", "20000"):
        for s in SPLITS:
            ev["arms"]["FAST"][point][s]["kl_median_by_position"] = [.81, .81, .81, .81, .81]
    assert all(outcome(spec, ev).values())
    ev["arms"]["FAST"]["20000"]["heldout_legal"]["kl_median_by_position"][4] = math.nextafter(.81, 1.)
    assert not outcome(spec, ev)["20000/dev_legal/KL_median"]


def test_s24_early_checkpoints_do_not_gate() -> None:
    spec, ev = uniform_shift_case({"REF": 5, "FAST": 5})
    for s in SPLITS:
        for point in ("2500", "5000"):
            ev["arms"]["FAST"][point][s]["accepted_length"] = 0.
            for r in ev["arms"]["FAST"][point][s]["request_totals"]:
                r["accepted_sum"] = 0.
    assert decision(spec, ev, outcome(spec, ev)) == ("PASS", False)


def test_s24_definitive_failure_plus_unrelated_straddle_stays_fail() -> None:
    """Amendment 6b: definitive inferiority remains FAIL when an unrelated failing CI straddles."""
    spec, ev = uniform_shift_case({"REF": 10, "FAST": 10})
    # 10K/dev_acap: FAST - REF = -0.05 on every request -> collapsed interval entirely below -0.03 (definitive).
    for r in ev["arms"]["FAST"]["10000"]["acap"]["request_totals"]:
        r["accepted_sum"] -= 5
    ev["arms"]["FAST"]["10000"]["acap"]["accepted_length"] = sum(
        r["accepted_sum"] for r in ev["arms"]["FAST"]["10000"]["acap"]["request_totals"])/1000
    checks = outcome(spec, ev)
    assert not checks["10000/dev_acap/non_inferiority"]
    assert decision(spec, ev, checks) == ("FAIL", False)
    # Now make 20K/dev_legal's FAST - REF straddle -0.03 (mean unchanged at 0, spread across requests).
    legal = ev["arms"]["FAST"]["20000"]["heldout_legal"]["request_totals"]
    for i, r in enumerate(legal):
        r["accepted_sum"] += 40 if i % 2 == 0 else -40
    checks = outcome(spec, ev)
    verdict, extra = decision(spec, ev, checks)
    assert verdict != "PASS"
    assert (verdict, extra) == ("FAIL", False)


def test_s24_realistic_size_runtime_and_determinism() -> None:
    """1,000 sampled rounds per split (the Amendment-4 cap); the verdict is deterministic across calls."""
    import time
    spec, ev = scenario(99, {"acap": 400, "heldout_legal": 300}, {"acap": 1, "heldout_legal": 8})
    assert ev["baseline"]["acap"]["rounds"] == 1000 and ev["baseline"]["heldout_legal"]["rounds"] == 1000
    t = time.perf_counter()
    first = outcome(spec, ev)
    elapsed = time.perf_counter()-t
    second = outcome(copy.deepcopy(spec), copy.deepcopy(ev))
    assert first == second and first == oracle(ev)[0]
    assert elapsed < 600
