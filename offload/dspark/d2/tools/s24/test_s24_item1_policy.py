"""S24 item 1: production FAST cannot be authorized by FAIL / INCONCLUSIVE / synthetic / old-rule receipts,
even when every digest is consistently recomputed after forging the verdict fields."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable, cast

import pytest

from offload.dspark.d2.train import fast_policy, pilot, pilot_verdict
from offload.dspark.d2.train.pilot import Spec
from offload.dspark.d2.train.pilot_eval import Evaluation
from offload.dspark.d2.train.pilot_verdict import Receipt
from offload.dspark.d2.train.tests import test_pilot_policy as lane
from offload.dspark.d2.train.tests.test_pilot_verdict import set_acceptance

amendment7_evidence = cast(Callable[[], tuple[Spec, Evaluation]], lane.evidence)


def forged(raw: Receipt, *, verdict: str = "PASS") -> Receipt:
    raw = copy.deepcopy(raw)
    raw["passed"], raw["verdict"], raw["extra_seed_eligible"] = True, verdict, False
    raw["checks"] = {k: True for k in raw["checks"]}
    raw["evidence_sha256"] = pilot.digest({"plan": raw["plan"], "evaluation": raw["evaluation"], "checks": raw["checks"],
                                           "rule_version": pilot_verdict.RULE_VERSION, "sample_sha256": raw["plan"]["sample"]["sha256"]})
    return raw


def make(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: Callable[[Evaluation], None]) -> tuple[Path, Path, Path, Path]:
    def altered() -> tuple[Spec, Evaluation]:
        spec, ev = amendment7_evidence()
        change(ev)
        return spec, ev
    monkeypatch.setattr(lane, "evidence", altered)
    return lane.receipt_fixture(tmp_path, monkeypatch)


def ref_no_gain(ev: Evaluation) -> None:
    set_acceptance(ev["arms"]["REF"]["20000"]["heldout_legal"], 1.)


def fast_inferior(ev: Evaluation) -> None:
    set_acceptance(ev["arms"]["FAST"]["10000"]["dev"], 1.1)


@pytest.mark.parametrize("change,expected", [(ref_no_gain, "INCONCLUSIVE"), (fast_inferior, "FAIL")])
def test_s24_non_pass_receipts_refused_even_when_forged_consistently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                     change: Callable[[Evaluation], None], expected: str) -> None:
    receipt, manifest, parity, g1 = make(tmp_path, monkeypatch, change)
    raw = json.loads(receipt.read_text())
    assert raw["verdict"] == expected and raw["passed"] is False
    with pytest.raises(ValueError, match="PASSING real pilot"):
        fast_policy.require_pilot(receipt, "base", manifest, parity, g1, 128, 8192, 32)
    receipt.write_text(json.dumps(forged(raw)))
    with pytest.raises(ValueError, match="thresholds or evidence identity"):
        fast_policy.require_pilot(receipt, "base", manifest, parity, g1, 128, 8192, 32)


def test_s24_pass_receipt_accepted_then_old_or_synthetic_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt, manifest, parity, g1 = lane.receipt_fixture(tmp_path, monkeypatch)
    raw = json.loads(receipt.read_text())
    assert raw["verdict"] == "PASS"
    fast_policy.require_pilot(receipt, "base", manifest, parity, g1, 128, 8192, 32)
    for key, value in (("rule_version", "amendment-5-per-metric-v1"), ("synthetic", True), ("verdict", "INCONCLUSIVE"),
                       ("schema", "dsv41.d2.fast-agreement.v1")):
        bad = dict(raw)
        bad[key] = value
        receipt.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            fast_policy.require_pilot(receipt, "base", manifest, parity, g1, 128, 8192, 32)
    # Interval tamper: a widened interval with every other field intact is refused (intervals are recomputed).
    bad = dict(raw)
    bad["intervals"] = {k: {**v, "lower": v["lower"]+1.} for k, v in raw["intervals"].items()}
    receipt.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="thresholds or evidence identity"):
        fast_policy.require_pilot(receipt, "base", manifest, parity, g1, 128, 8192, 32)
    # Dev naming is required in the bound evaluation.
    bad = json.loads(json.dumps(raw))
    bad["evaluation"]["split_roles"] = {"acap": "heldout_acap", "heldout_legal": "test_legal"}
    receipt.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        fast_policy.require_pilot(receipt, "base", manifest, parity, g1, 128, 8192, 32)
    assert set(raw["intervals"]) == {f"{p}/{s}/{k}" for p in (10000, 20000) for s in ("dev", "dev_legal")
                                     for k in ("REF_gain", "FAST_gain", "FAST_minus_REF")}
