from __future__ import annotations

import copy
from typing import cast

import pytest

from offload.dspark.d2.train.pilot import ARMS, Spec, digest
from offload.dspark.d2.train.pilot_eval import EVALUATOR, Evaluation, Metrics, SPLITS
from offload.dspark.d2.train.pilot_statistics import DEV_SPLITS, paired_interval
from offload.dspark.d2.train.pilot_verdict import status
from offload.dspark.d2.train.pilot_verdict import outcome
from offload.dspark.d2.train.pilot_sample import RoundKey, draw_sample, key_hash


def evidence() -> tuple[Spec, Evaluation]:
    population: dict[str, list[RoundKey]] = {s: [{"request": f"{s}-r{i//5}", "P": 3+6*(i%5)} for i in range(100)] for s in SPLITS}
    sample = draw_sample("manifest", population, dict.fromkeys(SPLITS, "digest"))
    spec = cast(Spec, {"synthetic": False, "manifest_sha256": "manifest", "sample": sample,
                       "checkpoints": [2500, 5000, 10000, 20000]})
    def m(split: str, kl: float, accepted: float) -> Metrics:
        return {"requests": 20, "rounds": 100, "population_sha256": key_hash(sample["keys"][split]), "kl_by_position": [kl]*5,
                "accepted_length": accepted, "loss": kl, "kl_median_by_position": [kl]*5,
                "request_totals": [{"request_sha256": digest(f"{split}-r{i}"), "family": f"family-{i//5}", "task_label": f"task-{i%2}",
                                    "rounds": 5, "accepted_sum": accepted*5} for i in range(20)]}
    ev = cast(Evaluation, {"schema": "dsv41.d2.pilot-evaluation.v1", "synthetic": False,
                          "evaluator": EVALUATOR, "split_roles": DEV_SPLITS, "k2_sha256": sample["k2_sha256"],
                          "sample_sha256": sample["sha256"],
                          "baseline": {s: m(s, 1., 1.) for s in SPLITS},
                          "arms": {a: {str(n): {s: m(s, .8 if a != "FAST" else .805, 1.2 if a != "FAST" else 1.18)
                                               for s in SPLITS} for n in spec["checkpoints"]} for a in ARMS},
                          "checkpoints": {a: {str(n): {} for n in spec["checkpoints"]} for a in ARMS}})
    return spec, ev


def set_acceptance(metrics: Metrics, value: float) -> None:
    metrics["accepted_length"] = value
    for row in metrics["request_totals"]:
        row["accepted_sum"] = value*row["rounds"]


def test_useful_gain_and_non_inferiority() -> None:
    spec, ev = evidence()
    assert status(outcome(spec, ev)) == "PASS"
    set_acceptance(ev["arms"]["FAST"]["10000"]["acap"], 1.16)
    assert not outcome(spec, ev)["10000/dev_acap/non_inferiority"]
    assert status(outcome(spec, ev)) == "FAIL"


def test_ref_without_useful_gain_is_inconclusive() -> None:
    spec, ev = evidence()
    set_acceptance(ev["arms"]["REF"]["20000"]["heldout_legal"], 1.)
    assert status(outcome(spec, ev)) == "INCONCLUSIVE"


def test_absolute_noninferiority_boundary() -> None:
    spec, ev = evidence()
    set_acceptance(ev["arms"]["FAST"]["10000"]["acap"], 1.17)
    assert outcome(spec, ev)["10000/dev_acap/non_inferiority"]
    set_acceptance(ev["arms"]["FAST"]["10000"]["acap"], 1.169)
    assert not outcome(spec, ev)["10000/dev_acap/non_inferiority"]


def test_fast_b_spread_cannot_excuse_degradation() -> None:
    spec, ev = evidence()
    set_acceptance(ev["arms"]["FAST-B"]["10000"]["acap"], 4.)
    set_acceptance(ev["arms"]["FAST"]["10000"]["acap"], 1.1)
    assert status(outcome(spec, ev)) == "FAIL"


@pytest.mark.parametrize("position", range(5))
def test_median_kl_absolute_deficit(position: int) -> None:
    spec, ev = evidence()
    ev["arms"]["FAST"]["10000"]["acap"]["kl_median_by_position"][position] = .811
    assert not outcome(spec, ev)["10000/dev_acap/KL_median"]


def test_bootstrap_pairs_requests_and_families_and_is_repeatable() -> None:
    _, ev = evidence()
    a = ev["baseline"]["acap"]["request_totals"]
    b = copy.deepcopy(a)
    for r in a:
        r["accepted_sum"] += .5 if r["family"] in ("family-0", "family-1") else 1.5
    ci = paired_interval(a, b)
    assert ci == paired_interval(a, b)
    assert ci["difference"] == .2 and ci["lower"] < .2 < ci["upper"]
    assert ci["resamples"] == 2000 and ci["families"] == 4
    assert paired_interval(a, a)["lower"] == paired_interval(a, a)["upper"] == 0
    b[0]["rounds"] += 1
    with pytest.raises(ValueError, match="unpaired"):
        paired_interval(a, b)


@pytest.mark.parametrize("defect", ["arm", "checkpoint", "split", "population", "NaN", "inf", "baselineNaN", "evaluator", "old_evaluator"])
def test_incomplete_or_nonfinite_mismatched_evidence_refused(defect: str) -> None:
    spec, ev = evidence()
    if defect == "arm":
        del ev["arms"]["FAST-B"]
    elif defect == "checkpoint":
        del ev["arms"]["FAST"]["2500"]
    elif defect == "split":
        del ev["arms"]["REF"]["5000"]["acap"]
    elif defect == "population":
        ev["arms"]["FAST"]["10000"]["acap"]["population_sha256"] = "foreign"
    elif defect == "evaluator":
        ev["evaluator"] = "fast"
    elif defect == "old_evaluator":
        ev["evaluator"] = "G2-reference-top20-tail-fixed-K2-amend6-v1"
    elif defect == "baselineNaN":
        ev["baseline"]["acap"]["loss"] = float("nan")
    else:
        ev["arms"]["FAST"]["2500"]["acap"]["loss"] = float("nan" if defect == "NaN" else "inf")
    with pytest.raises(ValueError):
        outcome(spec, ev)


@pytest.mark.parametrize("defect", ["missing_plan_sample", "missing_evaluation_hash", "foreign_sample", "foreign_K2", "same_foreign_population_everywhere"])
def test_outcome_requires_the_pre_registered_sample(defect: str) -> None:
    spec, ev = evidence()
    if defect == "missing_plan_sample":
        del spec["sample"]
    elif defect == "missing_evaluation_hash":
        ev = cast(Evaluation, {k: v for k, v in ev.items() if k != "sample_sha256"})
    elif defect == "foreign_sample":
        ev["sample_sha256"] = "foreign"
    elif defect == "foreign_K2":
        ev["k2_sha256"] = {"acap": "foreign", "heldout_legal": "digest"}
    else:
        for metrics in ev["baseline"].values():
            metrics["population_sha256"] = "same-foreign-population"
        for points in ev["arms"].values():
            for splits in points.values():
                for metrics in splits.values():
                    metrics["population_sha256"] = "same-foreign-population"
    with pytest.raises(ValueError, match="sample|scored keys"):
        outcome(spec, ev)


def test_single_recorded_family_reports_request_uncertainty_without_inventing_families() -> None:
    _, ev = evidence()
    a = ev["baseline"]["acap"]["request_totals"]
    b = copy.deepcopy(a)
    for i, (left, right) in enumerate(zip(a, b)):
        left["family"] = right["family"] = "acap"
        left["accepted_sum"] += .5 if i < 10 else 1.5
    ci = paired_interval(a, b)
    assert ci["families"] == 1 and ci["requests"] == 20
    assert ci["lower"] < ci["difference"] < ci["upper"]


def test_straddled_noninferiority_is_inconclusive_and_only_then_extra_seed_eligible() -> None:
    from offload.dspark.d2.train.pilot_verdict import decision
    spec, ev = evidence()
    assert decision(spec, ev, outcome(spec, ev)) == ("PASS", False)
    fast = ev["arms"]["FAST"]["10000"]["acap"]
    for i, row in enumerate(fast["request_totals"]):
        row["accepted_sum"] = 6.4 if i < 10 else 5.4
    # Mean is unchanged; clustered variation straddles the NI bound.
    assert decision(spec, ev, outcome(spec, ev)) == ("INCONCLUSIVE", True)
    fast["kl_median_by_position"][0] = 2.
    assert decision(spec, ev, outcome(spec, ev)) == ("FAIL", False)


@pytest.mark.parametrize("failure", ["KL", "baseline", "other_NI", "other_FAST_gain"])
def test_ref_no_gain_cannot_excuse_definitive_regressions(failure: str) -> None:
    from offload.dspark.d2.train.pilot_verdict import decision
    spec, ev = evidence()
    set_acceptance(ev["arms"]["REF"]["20000"]["heldout_legal"], 1.)
    assert decision(spec, ev, outcome(spec, ev)) == ("INCONCLUSIVE", False)
    if failure == "KL":
        ev["arms"]["FAST"]["20000"]["heldout_legal"]["kl_median_by_position"][0] = 2.
    elif failure == "baseline":
        # NI passes (-0.02); the baseline point failure must still dominate.
        set_acceptance(ev["arms"]["FAST"]["20000"]["heldout_legal"], .98)
    elif failure == "other_NI":
        set_acceptance(ev["arms"]["FAST"]["10000"]["acap"], 1.1)
    else:
        set_acceptance(ev["arms"]["REF"]["10000"]["acap"], 1.06)
        set_acceptance(ev["arms"]["FAST"]["10000"]["acap"], 1.04)
    assert decision(spec, ev, outcome(spec, ev)) == ("FAIL", False)
