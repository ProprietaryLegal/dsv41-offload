from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from offload.dspark.d2.train import pilot, pilot_diag, pilot_eval
from offload.dspark.d2.train.data import Block, D2Blocks, SyntheticBlocks, sha256
from offload.dspark.d2.train.evaluate import bucket_kl
from offload.dspark.d2.train.model import Drafter
from offload.dspark.d2.train.pilot_diag import DiagnosticError
from offload.dspark.d2.train.pilot_eval import Metrics
from offload.dspark.d2.train.pilot_sample import RoundKey, key_hash, sample_plan
from offload.dspark.d2.train.pilot_statistics import split_roles
from offload.dspark.d2.train.replay import apply_trainables
from offload.dspark.d2.train.tests.test_loop import model
from offload.dspark.d2.train.tests.test_pilot_sample import sample_fixture


# ------------------------------------------------------------------ per-round metrics against a dense oracle


def teacher_rows(seed: int, vocab: int = 40, exclude: dict[int, int] | None = None) -> tuple[Tensor, Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    full = (torch.randn(5, vocab, generator=g, dtype=torch.float64) * 2).softmax(-1)
    for position, token in (exclude or {}).items():
        full[position, token] = 0.  # keep this token out of the top-20 (an outside pick)
    full = full / full.sum(-1, keepdim=True)
    p, ids = full.topk(20, -1)
    return ids, p.log().float(), (1 - p.sum(-1)).float()


def dense_kl(z: Tensor, ids: Tensor, logp: Tensor, tail: Tensor) -> list[float]:
    out = []
    for i in range(5):
        q = [math.exp(v) for v in z[i].double().tolist()]
        total = math.fsum(q)
        q = [v/total for v in q]
        members = ids[i].tolist()
        p = [math.exp(v) for v in logp[i].double().tolist()]
        qt = math.fsum(v for j, v in enumerate(q) if j not in members)
        t = float(tail[i])
        value = math.fsum(pi*(math.log(pi)-math.log(q[j])) for pi, j in zip(p, members) if pi > 0)
        out.append(value + (t*(math.log(t)-math.log(qt)) if t > 0 else 0.))
    return out


def test_round_statistics_match_a_dense_oracle() -> None:
    g = torch.Generator().manual_seed(5)
    logits = torch.randn(5, 40, generator=g) * 3
    picks = logits.argmax(-1)
    ids, logp, tail = teacher_rows(7, exclude={4: int(picks[4])})
    labels = picks.clone()
    labels[2] = (picks[2] + 1) % 40  # accepted prefix of length 2
    kl = bucket_kl(logits, ids, logp, tail)
    s = pilot_diag.round_stats(logits, kl, ids, logp, tail, labels, picks, 2, "req", "fam")
    p = logp.double().exp()
    for i in range(5):
        q = logits[i].double().softmax(-1)
        top2 = logits[i].double().topk(2).values
        assert s.fields["q1"][i].item() == pytest.approx(float(q.max()), rel=1e-12)
        assert s.fields["entropy"][i].item() == pytest.approx(float(-(q*q.log()).sum()), rel=1e-10)
        assert s.fields["gap"][i].item() == pytest.approx(float(top2[0]-top2[1]), rel=1e-12)
        inside = int(picks[i]) in ids[i].tolist()
        expected_lower = float(p[i, ids[i].tolist().index(int(picks[i]))]) if inside else 0.
        expected_upper = expected_lower if inside else min(float(tail[i]), float(p[i, -1]))
        assert s.fields["pp_lower"][i].item() == pytest.approx(expected_lower, abs=1e-15)
        assert s.fields["pp_upper"][i].item() == pytest.approx(expected_upper, abs=1e-15)
        assert s.fields["pick_outside"][i].item() == float(not inside)
        assert s.fields["y_match"][i].item() == float(picks[i] == labels[i])
        assert s.fields["t_match"][i].item() == float(picks[i] == ids[i, 0])
        assert s.fields["survive"][i].item() == float(i < 2)
        assert s.teacher["p1"][i].item() == pytest.approx(float(p[i, 0]))
        assert s.teacher["coverage"][i].item() == pytest.approx(float(p[i].sum()))
        t = float(tail[i])
        assert s.teacher["entropy"][i].item() == pytest.approx(float(-(p[i]*p[i].log()).sum()) - (t*math.log(t) if t > 0 else 0.))
        assert s.teacher["hit"][i].item() == float(ids[i, 0] == labels[i])
    assert s.fields["pick_outside"][4].item() == 1.
    torch.testing.assert_close(s.fields["kl"], torch.tensor(dense_kl(logits, ids, logp, tail), dtype=torch.float64))
    torch.testing.assert_close(s.fields["kl_top"] + s.fields["kl_tail"], kl.double(), rtol=0, atol=0)
    for row, tau in zip(s.temperature_kl, pilot_diag.TEMPERATURES):
        torch.testing.assert_close(row, torch.tensor(dense_kl(logits.double()/tau, ids, logp, tail), dtype=torch.float64))
    # Rao-Blackwellised E[A]: prefix 1, 1, 1, 0, 0 over the teacher probabilities of the greedy picks.
    expected_rb = sum(s.fields["pp_lower"][i].item() for i in range(3))
    assert s.rb_lower == pytest.approx(expected_rb)
    assert s.prefix.tolist() == [1., 1., 1., 0., 0.]
    assert s.identity_checked == 3 and s.near_ties == 0


def test_identity_check_counts_arithmetic_near_ties_and_refuses_conditioning_misses() -> None:
    logits = torch.zeros(5, 40)
    logits[:, 7] = 5.
    ids, logp, tail = teacher_rows(3)
    greedy = torch.full((5,), 7)
    logits[3, 9] = 5. - 1e-5
    logits[3, 7] = 5.  # teacher-forced argmax 7, greedy picked 9 within the arithmetic tolerance
    greedy[3] = 9
    kl = bucket_kl(logits, ids, logp, tail)
    s = pilot_diag.round_stats(logits, kl, ids, logp, tail, greedy.clone(), greedy, 5, "r", "f")
    assert s.near_ties == 1 and s.identity_checked == 5
    logits[3, 9] = 4.9
    kl = bucket_kl(logits, ids, logp, tail)
    with pytest.raises(DiagnosticError, match="own-draft pick differs"):
        pilot_diag.round_stats(logits, kl, ids, logp, tail, greedy.clone(), greedy, 5, "r", "f")


def test_round_statistics_refuse_inconsistent_kl_or_prefix() -> None:
    logits = torch.randn(5, 40, generator=torch.Generator().manual_seed(1))
    ids, logp, tail = teacher_rows(2)
    picks = logits.argmax(-1)
    kl = bucket_kl(logits, ids, logp, tail)
    with pytest.raises(DiagnosticError, match="KL parts"):
        pilot_diag.round_stats(logits, kl + 1e-6, ids, logp, tail, picks, picks, 5, "r", "f")
    with pytest.raises(DiagnosticError, match="match_length"):
        pilot_diag.round_stats(logits, kl, ids, logp, tail, picks, picks, 4, "r", "f")


def test_cluster_mean_and_paired_ratio_formulas() -> None:
    values = torch.tensor([1., 2., 3., 6.])
    clusters = ["a", "a", "b", "c"]
    out = pilot_diag.cluster_mean(values, clusters)
    mean = 3.
    sums = {"a": (3., 2), "b": (3., 1), "c": (6., 1)}
    variance = 3/2 * sum((s - n*mean)**2 for s, n in sums.values())
    assert out["mean"] == mean and out["se"] == pytest.approx(math.sqrt(variance)/4)
    assert pilot_diag.cluster_mean(values, ["a"]*4)["se"] is None
    assert pilot_diag.cluster_mean(torch.tensor([]), []) == {"mean": None, "se": None}
    a0, b0 = torch.tensor([1., 0., 1., 1.]), torch.ones(4)
    a1, b1 = torch.tensor([1., 1., 1., 1.]), torch.tensor([1., 1., 1., 1.])
    ratio = pilot_diag.paired_ratio(a0, b0, a1, b1, clusters)
    assert ratio["mean"] == pytest.approx(.25) and ratio["se"] is not None
    assert pilot_diag.paired_ratio(a0, torch.zeros(4), a1, b1, clusters)["mean"] is None


# ------------------------------------------------------------------ tiny drafter, real scoring path


class FamilyBlocks(SyntheticBlocks):
    request_families = {"r0": "fa", "r1": "fb", "r2": "fc"}
    request_tasks = {"r0": "t", "r1": "t", "r2": "u"}
    labels: dict[int, Tensor] = {}

    def __getitem__(self, index: int) -> Block:
        block = super().__getitem__(index)
        block["request"] = f"r{index % 3}"
        if index in self.labels:
            labels = self.labels[index][None].clone()
            block["targets"]["labels"] = labels
            block["targets"]["previous"] = torch.cat((block["inputs"]["anchor"][:, None], labels[:, :-1]), -1)
        return block


def fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[Drafter, FamilyBlocks, list[tuple[int, RoundKey]], list[RoundKey]]:
    torch.set_num_threads(2)
    m = model().eval()
    m.reference_math = True  # tiny CPU fixture; production load enforces the real reference path
    data = FamilyBlocks(m.c, 9)
    FamilyBlocks.labels = {}
    # Make some rounds accept: labels equal to the initial drafter's own greedy picks.
    for index in (0, 3, 4, 7):
        block = data[index]
        with torch.no_grad():
            h, hx = m.hidden_from(block["inputs"])
            picks, _ = m.greedy(h, hx, block["inputs"]["anchor"], chunk=32)
        labels = picks[0].clone()
        if index == 4:
            labels[2] = (labels[2] + 1) % m.c.vocab
        FamilyBlocks.labels[index] = labels
    selected: list[tuple[int, RoundKey]] = [(i, {"request": f"r{i % 3}", "P": 12+i}) for i in range(9)]
    keys = [k for _, k in selected]
    monkeypatch.setattr(pilot_eval, "complete_rounds", lambda *_: selected)
    monkeypatch.setattr(pilot_diag, "complete_rounds", lambda *_: selected)
    return m, data, selected, keys


def perturbed(initial: dict[str, Tensor], seed: int, scale: float) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, value in initial.items():
        group = pilot_diag.component(name)
        if group in ("lora", "markov") and not name.endswith(".a"):
            out[name] = value + scale*torch.randn(value.shape, generator=g)
        elif group == "norm":
            out[name] = value * (1 + scale*torch.randn(value.shape, generator=g))
        else:
            out[name] = value.clone()
    return out


def states_and_receipt(m: Drafter, data: FamilyBlocks, keys: list[RoundKey], tmp_path: Path
                       ) -> tuple[dict[str, dict[str, Tensor]], dict[str, Metrics]]:
    initial = {n: p.detach().clone() for n, p in m.named_parameters()}
    states = pilot_diag.variant_states(initial, perturbed(initial, 1, .05), perturbed(initial, 2, .2), perturbed(initial, 3, .2))
    expected = {}
    for name in ("V0", "R2.5", "R20", "F20"):
        apply_trainables(m, states[name])
        expected[name] = pilot_eval.score(m, cast(D2Blocks, data), tmp_path/"k2", keys, chunk=32)
    return states, expected


def test_score_variant_reproduces_pilot_eval_exactly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m, data, selected, keys = fixture(monkeypatch)
    states, expected = states_and_receipt(m, data, keys, tmp_path)
    for name in ("V0", "R20"):
        apply_trainables(m, states[name])
        metrics, rounds = pilot_diag.score_variant(m, cast(D2Blocks, data), selected, keys, chunk=32)
        assert {k: v for k, v in metrics.items() if k != "loss"} == {k: v for k, v in expected[name].items() if k != "loss"}
        assert [r.accepted for r in rounds] == [int(x) for x in [r.accepted for r in rounds]]
        assert sum(r.accepted for r in rounds)/len(rounds) == expected[name]["accepted_length"]
    assert expected["V0"]["accepted_length"] > 0  # the fixture really accepts drafts


def test_diagnose_split_scores_all_variants_and_ablations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m, data, selected, keys = fixture(monkeypatch)
    states, expected = states_and_receipt(m, data, keys, tmp_path)
    lines: list[str] = []
    result = pilot_diag.diagnose_split(m, cast(D2Blocks, data), selected, keys, states, expected, chunk=32, log=lines.append)
    assert [json.loads(line)["variant"] for line in lines] == list(pilot_diag.VARIANTS)
    assert set(result["reproduction"]) == {"V0", "R2.5", "R20", "F20"}
    assert all(all(r["exact"].values()) and r["kl_bit_identical"] for r in result["reproduction"].values())
    assert result["families"] == 3 and result["rounds"] == 9
    for name in pilot_diag.VARIANTS:
        metrics = result["variants"][name]["metrics"]
        assert len(metrics["positions"]["kl"]["mean"]) == 5
        assert metrics["identity"]["checked"] >= 9
        assert metrics["EA_RB_lower"] <= metrics["EA_RB_upper"]
    assert result["variants"]["V0"]["metrics"]["EA"] == expected["V0"]["accepted_length"]
    vs = result["variants"]["R20"]["vs_V0"]
    assert vs["EA"]["mean"] == pytest.approx(expected["R20"]["accepted_length"] - expected["V0"]["accepted_length"])
    assert vs["EA"]["se"] is not None
    assert set(result["readings"]) == {"H0", "H1", "H2", "H3", "H4", "H5", "H6"}
    # Ablations: L20 keeps the trained trunk with the initial Markov head; H20 the reverse.
    for name, value in states["L20"].items():
        donor = states["V0"] if pilot_diag.component(name) == "markov" else states["R20"]
        assert torch.equal(value, donor[name])
    for name, value in states["H20"].items():
        donor = states["R20"] if pilot_diag.component(name) == "markov" else states["V0"]
        assert torch.equal(value, donor[name])
    json.dumps(result, allow_nan=False)
    assert_no_token_lists(result)


def assert_no_token_lists(value: object) -> None:
    """Aggregates only: picks/labels never leave memory, so no list of integers appears anywhere."""
    if isinstance(value, dict):
        for item in value.values():
            assert_no_token_lists(item)
    elif isinstance(value, list):
        assert not any(type(item) is int for item in value)
        for item in value:
            assert_no_token_lists(item)


@pytest.mark.parametrize("variant", ["V0", "R20"])
def test_receipt_mismatch_stops_the_run_at_that_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str) -> None:
    m, data, selected, keys = fixture(monkeypatch)
    states, expected = states_and_receipt(m, data, keys, tmp_path)
    expected[variant] = {**expected[variant], "accepted_length": expected[variant]["accepted_length"] + 1e-12}
    lines: list[str] = []
    with pytest.raises(DiagnosticError, match=f"{variant}: E\\[A\\]"):
        pilot_diag.diagnose_split(m, cast(D2Blocks, data), selected, keys, states, expected, chunk=32, log=lines.append)
    order = list(pilot_diag.VARIANTS)
    assert [json.loads(line)["variant"] for line in lines] == order[:order.index(variant)]


def test_request_totals_and_kl_are_part_of_the_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m, data, selected, keys = fixture(monkeypatch)
    states, expected = states_and_receipt(m, data, keys, tmp_path)
    totals = [cast(dict[str, Any], dict(t)) for t in expected["F20"]["request_totals"]]
    totals[0]["accepted_sum"] += 1
    broken = {**expected, "F20": cast(Metrics, {**expected["F20"], "request_totals": totals})}
    with pytest.raises(DiagnosticError, match="request_totals"):
        pilot_diag.diagnose_split(m, cast(D2Blocks, data), selected, keys, states, broken, chunk=32, log=lambda _: None)
    shifted = {**expected, "R2.5": cast(Metrics, {**expected["R2.5"], "kl_by_position": [v+1e-3 for v in expected["R2.5"]["kl_by_position"]]})}
    with pytest.raises(DiagnosticError, match="KL differs"):
        pilot_diag.diagnose_split(m, cast(D2Blocks, data), selected, keys, states, shifted, chunk=32, log=lambda _: None)
    with pytest.raises(DiagnosticError, match="incomplete"):
        pilot_diag.diagnose_split(m, cast(D2Blocks, data), selected, keys, states,
                                  {k: v for k, v in expected.items() if k != "F20"}, chunk=32)


def test_components_cover_the_drafter_and_unknown_tensors_refuse() -> None:
    groups = Counter(pilot_diag.component(n) for n, _ in model().named_parameters())
    assert set(groups) == {"lora", "norm", "markov", "confidence"} and groups["markov"] == 2
    with pytest.raises(DiagnosticError, match="unclassified"):
        pilot_diag.component("t.mtp__0__main_proj__weight")
    with pytest.raises(DiagnosticError, match="different tensor names"):
        pilot_diag.mix({"a.markov_head": torch.zeros(1)}, {}, frozenset({"markov"}))


# ------------------------------------------------------------------ plan / receipt / source binding


def metrics_for(keys: list[RoundKey]) -> Metrics:
    counts = Counter(k["request"] for k in keys)
    return {"requests": len(counts), "rounds": len(keys), "population_sha256": key_hash(keys),
            "kl_by_position": [0.]*5, "loss": 0., "accepted_length": 0., "kl_median_by_position": [0.]*5,
            "request_totals": [{"request_sha256": pilot.digest(r), "family": "f", "task_label": "t", "rounds": c,
                                "accepted_sum": 0.} for r, c in sorted(counts.items())]}


def pilot_files(tmp_path: Path, edit: Any = None) -> tuple[Path, Path, Path]:
    plan, manifest, paths = sample_fixture(tmp_path)
    sample_plan(plan, manifest, paths)
    spec = json.loads(plan.read_text())
    if edit is not None:
        edit(spec)
        plan.unlink()
        pilot.write_json(plan, spec)
    keys = spec["sample"]["keys"]
    evaluation = {"schema": "dsv41.d2.pilot-evaluation.v1", "plan_sha256": sha256(plan), "synthetic": False,
                  "evaluator": pilot_eval.EVALUATOR, "split_roles": split_roles(keys), "sample_sha256": spec["sample"]["sha256"],
                  "k2_sha256": spec["sample"]["k2_sha256"], "baseline": {s: metrics_for(keys[s]) for s in keys},
                  "arms": {a: {str(n): {s: metrics_for(keys[s]) for s in keys} for n in pilot.CHECKPOINTS} for a in pilot.ARMS},
                  "checkpoints": {a: {str(n): {"path": "x", "sha256": "y", "step": 1} for n in pilot.CHECKPOINTS} for a in pilot.ARMS},
                  "step_agreement": {}}
    evaluation_path = tmp_path/"evaluation.json"
    pilot.write_json(evaluation_path, evaluation)
    receipt = tmp_path/"receipt.json"
    pilot.write_json(receipt, {"schema": "dsv41.d2.pilot-receipt.v1", "plan_sha256": sha256(plan),
                               "evaluation_sha256": sha256(evaluation_path), "evaluation": evaluation, "plan": spec,
                               "synthetic": False, "sample_sha256": spec["sample"]["sha256"]})
    return plan, receipt, evaluation_path


def test_pilot_binding_allows_only_the_diagnostic_as_a_new_module(tmp_path: Path) -> None:
    spec, evaluation, _ = pilot_diag.validate_pilot(*pilot_files(tmp_path/"same"))
    assert evaluation["evaluator"] == pilot_eval.EVALUATOR and "sample" in spec
    # The real pilot plan predates pilot_diag.py: it is the one permitted addition.
    pilot_diag.validate_pilot(*pilot_files(tmp_path/"new", lambda s: s["source_sha256"].pop("d2/train/pilot_diag.py")))
    with pytest.raises(DiagnosticError, match="unexpected new source"):
        pilot_diag.validate_pilot(*pilot_files(tmp_path/"extra", lambda s: s["source_sha256"].pop("d2/train/loss.py")))
    with pytest.raises(DiagnosticError, match="has changed"):
        pilot_diag.validate_pilot(*pilot_files(tmp_path/"changed", lambda s: s["source_sha256"].update({"d2/train/loss.py": "0"*64})))


@pytest.mark.parametrize("defect", ["evaluation_hash", "embedded_evaluation", "plan_hash", "receipt_plan", "evaluator"])
def test_pilot_binding_refuses_foreign_receipts(tmp_path: Path, defect: str) -> None:
    plan, receipt, evaluation = pilot_files(tmp_path)
    r = json.loads(receipt.read_text())
    e = json.loads(evaluation.read_text())
    if defect == "evaluation_hash":
        r["evaluation_sha256"] = "0"*64
    elif defect == "embedded_evaluation":
        r["evaluation"]["baseline"]["dev"]["accepted_length"] = 1.
    elif defect == "plan_hash":
        r["plan_sha256"] = "0"*64
    elif defect == "receipt_plan":
        r["plan"]["order_seed"] = 99
    else:
        e["evaluator"] = "other"
        evaluation.unlink()
        pilot.write_json(evaluation, e)
        r["evaluation"], r["evaluation_sha256"] = e, sha256(evaluation)
    receipt.unlink()
    pilot.write_json(receipt, r)
    with pytest.raises(DiagnosticError):
        pilot_diag.validate_pilot(plan, receipt, evaluation)


def test_run_refuses_before_any_model_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    from offload.dspark.d2.train import safety
    plan, receipt, evaluation = pilot_files(tmp_path, lambda s: s["source_sha256"].update({"d2/train/loss.py": "0"*64}))
    monkeypatch.setattr(safety, "assert_gpu_lease", lambda: None)
    monkeypatch.setattr(pilot_diag, "load", lambda *a, **k: pytest.fail("refused inputs reached model load"))
    with pytest.raises(DiagnosticError):
        pilot_diag.run(argparse.Namespace(plan=plan, receipt=receipt, evaluation=evaluation, manifest=tmp_path/"m",
                                          root=tmp_path, k2_dev=tmp_path, k2_legal=tmp_path, ckpt="x", split="dev",
                                          out=tmp_path/"out.json"))


# ------------------------------------------------------------------ mechanical readings of DIAG.md §Hypotheses


def cm(mean: float, se: float = .01) -> dict[str, float]:
    return {"mean": mean, "se": se}


def fake_vs(rb: float = -.05, rb_hi: float = -.05, ea: float = -.06, pp0: float = -.03, pp14: float = -.01,
            y0: float = -.02, entropy: float = .1, q1: float = -.05, kl: float = -.1, right: float = .05,
            wrong: float = -.3, flips: float = -.01, recovery: float | None = .6) -> dict[str, Any]:
    per = {f: [cm(0.) for _ in range(5)] for f in pilot_diag.POSITION_FIELDS}
    per["pp_lower"][0], per["y_match"][0] = cm(pp0), cm(y0)
    pooled = {f: cm(0.) for f in pilot_diag.POSITION_FIELDS}
    pooled.update(entropy=cm(entropy), q1=cm(q1), kl=cm(kl))
    later = {f: cm(0.) for f in pilot_diag.POSITION_FIELDS}
    later["pp_lower"] = cm(pp14)
    return {"EA": cm(ea), "EA_RB_lower": cm(rb), "EA_RB_upper": cm(rb_hi), "per_position": per, "pooled": pooled,
            "pooled_1_4": later, "acc_given_reach": [cm(0.)]*5,
            "argmax_changed": [{"rounds": 3, "delta_pick_probability": cm(flips)}]*5,
            "kl_change_baseline_right": [{"rounds": 5, **cm(right)}]*5,
            "kl_change_baseline_wrong": [{"rounds": 5, **cm(wrong)}]*5,
            "median_shift_pooled": {"kl": .02, "kl_top": .01, "kl_tail": .02}, "temperature_recovery": recovery}


def fake_result(tau0: float = 1.25, early: dict[str, Any] | None = None, **variants: dict[str, Any]) -> dict[str, Any]:
    vs = {"R2.5": early or fake_vs(rb=-.045), "R20": fake_vs(), "F20": fake_vs(), "L20": fake_vs(rb=-.005),
          "H20": fake_vs(rb=-.045)}
    vs.update(variants)
    temperature = {"V0": {"tau_star": [tau0]*5, "tau_star_pooled": tau0}, "R20": {"tau_star": [1.]*5, "tau_star_pooled": 1.}}
    v: dict[str, Any] = {name: {"metrics": {"temperature": temperature.get(name, temperature["R20"]),
                                            "identity": {"checked": 1, "near_ties": 0}}} for name in pilot_diag.VARIANTS}
    for name, value in vs.items():
        v[name]["vs_V0"] = value
    return {"variants": v, "teacher": {"calibration_gap": [cm(.001)]*5}}


def verdicts(result: dict[str, Any]) -> dict[str, str]:
    return {h: r["verdict"] for h, r in pilot_diag.readings(result).items()}


def test_readings_apply_the_preregistered_rules() -> None:
    base = verdicts(fake_result())
    assert base == {"H0": "RULED OUT", "H1": "CONFIRMED", "H2": "RULED OUT", "H3": "CONFIRMED", "H4": "RULED OUT",
                    "H5": "CONFIRMED", "H6": "CONFIRMED"}
    assert pilot_diag.readings(fake_result())["H3"]["conditions"]["head"] is True
    trunk = pilot_diag.readings(fake_result(L20=fake_vs(rb=-.045), H20=fake_vs(rb=-.005)))["H3"]
    assert trunk["verdict"] == "CONFIRMED" and trunk["conditions"]["trunk"] is True
    assert verdicts(fake_result(L20=fake_vs(rb=-.03), H20=fake_vs(rb=-.03)))["H3"] == "RULED OUT"  # both reproduce
    assert verdicts(fake_result(L20=fake_vs(rb=-.001), H20=fake_vs(rb=-.001)))["H3"] == "RULED OUT"  # interaction
    early = verdicts(fake_result(early=fake_vs(rb=-.01)))
    assert early["H5"] == "RULED OUT"
    assert verdicts(fake_result(early=fake_vs(rb=-.045, flips=.05)))["H5"] == "RULED OUT"
    assert verdicts(fake_result(R20=fake_vs(entropy=-.01)))["H1"] == "RULED OUT"
    assert verdicts(fake_result(R20=fake_vs(right=-.01)))["H1"] == "UNRESOLVED"
    # Δpp_0 must itself be negative; a resolved EA_RB fall only supplies the resolution.
    assert verdicts(fake_result(R20=fake_vs(pp0=.001)))["H1"] == "UNRESOLVED"
    assert verdicts(fake_result(R20=fake_vs(pp0=-.001)))["H1"] == "CONFIRMED"
    h4 = verdicts(fake_result(R20=fake_vs(pp0=-.005, y0=0., pp14=-.05)))
    assert h4["H4"] == "CONFIRMED"
    assert verdicts(fake_result(tau0=1.))["H6"] == "RULED OUT"
    position0 = fake_result()
    position0["variants"]["V0"]["metrics"]["temperature"]["tau_star"] = [1.] + [1.25]*4
    assert verdicts(position0)["H6"] == "RULED OUT"
    assert verdicts(fake_result(R20=fake_vs(recovery=.1)))["H6"] == "RULED OUT"
    assert verdicts(fake_result(R20=fake_vs(recovery=.4)))["H6"] == "UNRESOLVED"


def test_noise_reading_switches_the_damage_measure_and_gates_h3_h5() -> None:
    noise = pilot_diag.readings(fake_result(R20=fake_vs(rb=-.01, rb_hi=-.01, pp0=-.03)))
    assert noise["H0"]["verdict"] == "CONFIRMED"
    assert noise["H3"]["conditions"]["measure"] == "pp_lower[0]"
    unreadable = pilot_diag.readings(fake_result(R20=fake_vs(rb=-.01, rb_hi=-.01, pp0=-.005)))
    assert unreadable["H3"]["verdict"] == "UNRESOLVED" and unreadable["H5"]["verdict"] == "UNRESOLVED"
    gaps = fake_result()
    gaps["teacher"]["calibration_gap"] = [cm(.001)]*4 + [cm(.05)]
    assert verdicts(gaps)["H2"] == "CONFIRMED"


def part(split: str, rb: float, h2: str = "RULED OUT") -> dict[str, Any]:
    result = fake_result(R20=fake_vs(rb=rb))
    result["readings"] = {"H2": {"verdict": h2}}
    return {"schema": pilot_diag.PART_SCHEMA, "tool": pilot_diag.TOOL, "split": split, "plan_sha256": "p",
            "receipt_sha256": "r", "evaluation_sha256": "e", "sample_sha256": "s", "checkpoints": {}, "source": {},
            "synthetic": False, **result}


def test_merge_reads_h2_and_h7_across_splits(tmp_path: Path) -> None:
    def merged(dev: dict[str, Any], legal: dict[str, Any]) -> dict[str, Any]:
        paths = []
        for p in (dev, legal):
            path = tmp_path/f"{p['split']}-{len(list(tmp_path.iterdir()))}.json"
            path.write_text(json.dumps(p))
            paths.append(path)
        out = tmp_path/f"merged-{len(list(tmp_path.iterdir()))}.json"
        pilot_diag.merge(paths, out)
        return cast(dict[str, Any], json.loads(out.read_text())["cross_split_readings"])
    assert merged(part("dev", .02), part("heldout_legal", -.05)) == {
        "H2": {"verdict": "RULED OUT", "conditions": {"splits": {"dev": "RULED OUT", "heldout_legal": "RULED OUT"}}},
        "H7": {"verdict": "CONFIRMED", "conditions": {"dev": cm(.02), "heldout_legal": cm(-.05), "difference_resolved": True}}}
    assert merged(part("dev", -.05), part("heldout_legal", -.05, "CONFIRMED")) == {
        "H2": {"verdict": "CONFIRMED", "conditions": {"splits": {"dev": "RULED OUT", "heldout_legal": "CONFIRMED"}}},
        "H7": {"verdict": "RULED OUT", "conditions": {"dev": cm(-.05), "heldout_legal": cm(-.05), "difference_resolved": False}}}
    other = part("heldout_legal", -.05)
    other["checkpoints"] = {"REF/20000": "different"}
    with pytest.raises(DiagnosticError, match="disagree"):
        merged(part("dev", .02), other)
    with pytest.raises(DiagnosticError, match="split mismatch"):
        merged(part("dev", .02), part("dev", .02))
