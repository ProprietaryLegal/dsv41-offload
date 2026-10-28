"""Lane S16's FIX probes (offload/s16-check 63c613b, 29a2084: d2/s16/test_s16_equiv.py, test_s16_dataset.py), turned
into passing tests of the fixes F9-a, F9-b, F9-c, F10-a and F10-b (S16.report.md section 2), plus the dataset-build
confirmations S16 ran with an unbound gate file, re-run here with a gate bound to the replay (F10-a)."""

from __future__ import annotations

import json
import math
import types
from fractions import Fraction

import pytest

from d2 import equiv as E

SAME = [1, 2, 3, 4, 5]
POS2 = [1, 9, 3, 4, 5]          # differs from SAME at position 2 only
POS1 = [7, 2, 3, 4, 5]          # differs at position 1


def rec(j, D, R, decode_rows=128, kl=(1e-5,) * 5, key=None):
    """S16's synthetic draft record (d2/s16/test_s16_equiv.py)."""
    r = {"key": key or f"r{j:06d}_b0", "req": "acap0", "P": j, "realized": D, "engine": D,
         "D": {"drafts": D, "margin": [1.0] * 5}, "R": {"drafts": R, "margin": [1.0] * 5},
         "decode_rows_in_window": decode_rows, "main_x_P_rel": 0.0}
    if kl is not None:
        r["kl_D_R"] = list(kl)
    return r


def report(tmp_path, shards: dict[str, list[dict]]) -> dict:
    work = tmp_path / "w"
    work.mkdir(exist_ok=True)
    for name, recs in shards.items():
        with open(work / name, "w") as fh:
            for x in recs:
                fh.write(json.dumps(x) + "\n")
    out = tmp_path / "gate.json"
    E.cmd_report(types.SimpleNamespace(work=str(work), taps="", out=str(out)))
    return json.load(open(out))


# ---------------------------------------------------------------------------------------------------------- F9-a
def test_F9a_nan_kl_does_not_meet_the_bf16_noise_rule(tmp_path):
    """S16's probe: KL [1e-5, nan, nan, nan, nan] on 100 blocks that all disagree must not pass. Refused."""
    recs = [rec(j, SAME, POS2, kl=(1e-5, math.nan, math.nan, math.nan, math.nan)) for j in range(100)]
    with pytest.raises(SystemExit, match="not 5 finite values"):
        report(tmp_path, {"drafts-0of1.jsonl": recs})


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_F9a_non_finite_control_kl_refused_too(tmp_path, bad):
    recs = [rec(j, SAME, POS2) for j in range(10)]
    for r in recs:
        r["N"] = {"drafts": SAME, "margin": [1.0] * 5}
        r["kl_D_N"] = [1e-3, 1e-3, bad, 1e-3, 1e-3]
        r["control"] = {"sigma": E.CONTROL_SIGMA, "seed": E.CONTROL_SEED}
    with pytest.raises(SystemExit, match="kl_D_N"):
        report(tmp_path, {"drafts-0of1.jsonl": recs})


def test_F9a_drafts_side_check_and_nan_safe_max():
    assert E.finite_kl([0.0, 1e-3, 2.0, 0.5, 0.1], "x") == [0.0, 1e-3, 2.0, 0.5, 0.1]
    for bad in ([math.nan] * 5, [1.0, 1.0, 1.0, 1.0, math.inf], [1.0] * 4):
        with pytest.raises(SystemExit, match="refused"):
            E.finite_kl(bad, "blk")
    assert max(1e-3, math.nan) == 1e-3                       # what the plain max would have done
    with pytest.raises(SystemExit, match="non-finite"):
        E.finite_max(1e-3, math.nan)
    assert E.finite_max(1e-3, 2e-3) == 2e-3


# ---------------------------------------------------------------------------------------------------------- F9-b
def test_F9b_report_refuses_missing_shards(tmp_path):
    """S16's probe: only drafts-0of3.jsonl present (shards 1 and 2 dead or never started) -> refused."""
    recs = [rec(j, SAME, SAME) for j in range(60)]
    with pytest.raises(SystemExit, match="incomplete shard set: 1 of 3 shards present, missing k = \\[1, 2\\]"):
        report(tmp_path, {"drafts-0of3.jsonl": recs})


def test_F9b_complete_set_passes_and_is_recorded(tmp_path):
    recs = [rec(j, SAME, SAME) for j in range(60)]
    rep = report(tmp_path, {f"drafts-{k}of3.jsonl": recs[k::3] for k in range(3)})
    assert rep["shards"] == {"n": 3, "found": 3, "files": [f"drafts-{k}of3.jsonl" for k in range(3)]}
    assert rep["blocks"] == 60 and rep["gate"]["verdict"] == "pass"


@pytest.mark.parametrize("names,msg", [
    (["drafts-0of3.jsonl", "drafts-1of3.jsonl", "drafts-2of3.jsonl", "drafts-0of2.jsonl"], "different splits"),
    (["drafts-0of2.jsonl", "drafts-00of2.jsonl", "drafts-1of2.jsonl"], "appears twice"),
    (["drafts-1of2.jsonl", "drafts-2of2.jsonl"], "missing k = \\[0\\]"),
    (["drafts-0of1.jsonl", "drafts-old.jsonl"], "not a drafts-<k>of<n>"),
])
def test_F9b_inconsistent_sets_refused(tmp_path, names, msg):
    recs = [rec(j, SAME, SAME) for j in range(40)]
    shards = {name: recs[i::len(names)] for i, name in enumerate(names)}
    with pytest.raises(SystemExit, match=msg):
        report(tmp_path, shards)


def test_F9b_duplicate_keys_still_reported_first(tmp_path):
    """S16's confirmation (two runs with different --shard in one work dir): still refused as duplicate keys."""
    a = [rec(j, SAME, SAME) for j in range(10)]
    with pytest.raises(SystemExit, match="duplicate"):
        report(tmp_path, {"drafts-0of1.jsonl": a, "drafts-0of3.jsonl": a[:3]})


# ---------------------------------------------------------------------------------------------------------- F9-c
def test_F9c_gate_threshold_is_not_met_by_rounding(tmp_path):
    """S16's probe: 395 of 399 equal (98.997 %) displayed as 0.99 must fail; 396/400 (exactly 99 %) passes."""
    recs = [rec(j, SAME, SAME if j < 395 else POS1, kl=(1.0,) * 5) for j in range(399)]
    rep = report(tmp_path, {"drafts-0of1.jsonl": recs})
    assert rep["block_equal_frac"] == 0.99 and rep["gate"]["block_equal_met"] is False
    assert rep["gate"]["verdict"] == "fail"
    recs = [rec(j, SAME, SAME if j < 396 else POS1, kl=(1.0,) * 5) for j in range(400)]
    rep = report(tmp_path, {"drafts-0of1.jsonl": recs})
    assert rep["gate"]["blocks_equal"] == 396 and rep["gate"]["verdict"] == "pass"


def test_F9c_every_rounding_case_up_to_600_fails():
    """Every population S16 found where k/n < 0.99 <= round(k/n, 4) (n <= 600) fails the exact comparison."""
    hits = [(n, k) for n in range(1, 601) for k in range(n + 1) if k / n < 0.99 <= round(k / n, 4)]
    assert (399, 395) in hits
    assert all(not (Fraction(k, n) >= E.GATE_EXACT) for n, k in hits)


def _ctl(recs, ctl_pos1_miss):
    for j, r in enumerate(recs):
        r["N"] = {"drafts": [6, 2, 3, 4, 5] if j < ctl_pos1_miss else SAME, "margin": [1.0] * 5}
        r["kl_D_N"] = [1e-2] * 5
        r["control"] = {"sigma": E.CONTROL_SIGMA, "seed": E.CONTROL_SEED}
    return recs


def test_F9c_pos1_checks_are_exact():
    """Amendment 2's threshold control pos1 - 0.05, compared as exact fractions."""
    # control pos1 = 0.96 -> threshold exactly 0.91; R pos1 = 91/100 is met, 90/100 is not
    r = E.bf16_noise_rule(_ctl([rec(j, SAME, POS1 if j < 9 else POS2, kl=(1e-2,) * 5) for j in range(100)], 4))
    assert r["pos1_threshold"] == 0.91 and r["pos1_equal_frac"] == 0.91 and r["checks"]["pos1"] is True
    r = E.bf16_noise_rule(_ctl([rec(j, SAME, POS1 if j < 10 else POS2, kl=(1e-2,) * 5) for j in range(100)], 4))
    assert r["checks"]["pos1"] is False
    # control pos1 = 54/100: in floats 0.54 - 0.05 = 0.49000000000000005 > 0.49, so R at 49/100 would be refused;
    # exactly, the threshold is 49/100 and it is met
    assert 0.54 - 0.05 > 0.49
    r = E.bf16_noise_rule(_ctl([rec(j, SAME, POS1 if j < 51 else POS2, kl=(1e-2,) * 5) for j in range(100)], 46))
    assert r["pos1_equal_frac"] == 0.49 and r["checks"]["pos1"] is True
    # control at 100 %: threshold 0.95; 949/1000 fails, 950/1000 passes
    r = E.bf16_noise_rule(_ctl([rec(j, SAME, POS1 if j < 51 else POS2, kl=(1e-2,) * 5) for j in range(1000)], 0))
    assert r["checks"]["pos1"] is False
    r = E.bf16_noise_rule(_ctl([rec(j, SAME, POS1 if j < 50 else POS2, kl=(1e-2,) * 5) for j in range(1000)], 0))
    assert r["checks"]["pos1"] is True
    # a pos1 fraction displayed as the threshold is still compared exactly: 37999/40000 shows 0.95, is under 0.95
    r = E.bf16_noise_rule(_ctl([rec(j, SAME, POS1 if j < 2001 else POS2, kl=(1e-2,) * 5) for j in range(40000)], 0))
    assert r["pos1_equal_frac"] == 0.95 and r["pos1_threshold"] == 0.95 and r["checks"]["pos1"] is False


# --------------------------------------------------------------------------------------------------------- F10-a
def _ns(replay, out, **kw):
    import argparse
    a = argparse.Namespace(replay=str(replay), out=str(out), prompt_manifest=str(replay / "prompt-manifest.json"), taps_source="prefill_replay",
                           splits="", allow_skips=False, equiv_gate="", override_equiv_gate="", override_plumbing="")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.mark.parametrize("body", [{"gate": {"pass": True}},
                                  {"gate": {"pass": True, "verdict": "fail", "block_equal_frac": 0.5}}],
                         ids=["minimal", "inconsistent"])
def test_F10a_gate_file_must_be_an_equiv_report(tmp_path, body):
    """S16's probe: a two-key hand-made file, or one whose verdict says fail, is refused, with or without override."""
    from d2 import dataset as D
    from d2.tests.test_dataset import _fake_replay, _recs
    out = _fake_replay(tmp_path, _recs())
    g = tmp_path / "hand.json"
    g.write_text(json.dumps(body))
    with pytest.raises(SystemExit, match="not a valid equivalence report"):
        D.cmd_build(_ns(out, tmp_path / "shards", equiv_gate=str(g)))
    with pytest.raises(SystemExit, match="not a valid equivalence report"):
        D.cmd_build(_ns(out, tmp_path / "shards", equiv_gate=str(g), override_equiv_gate="x"))
    assert not (tmp_path / "shards").exists() and not (tmp_path / "shards.partial").exists()


def test_F10a_gate_of_another_replay_or_unbound_is_refused(tmp_path):
    from d2 import dataset as D
    from d2.tests.test_dataset import _bind_acap, _fake_replay, _gate, _recs
    out = _fake_replay(tmp_path, _recs())
    _bind_acap(out)                                     # this replay's acap0 DONE.json
    other = {"acap0": {"sha256_taps": "0" * 64, "sha256_ids": "1" * 64}}
    with pytest.raises(SystemExit, match="sha256_taps of this replay differs"):
        D.cmd_build(_ns(out, tmp_path / "s1", equiv_gate=_gate(tmp_path, True, out, binding=other)))
    # a real report whose draft records carry no replay binding (pre-F10-a drafts, or S16's synthetic records)
    work = tmp_path / "unbound"
    work.mkdir()
    with open(work / "drafts-0of1.jsonl", "w") as fh:
        for j in range(20):
            fh.write(json.dumps(rec(j, SAME, SAME)) + "\n")
    E.cmd_report(types.SimpleNamespace(work=str(work), taps="", out=str(work / "gate.json")))
    assert json.load(open(work / "gate.json"))["replay_binding"] is None
    with pytest.raises(SystemExit, match="no replay_binding"):
        D.cmd_build(_ns(out, tmp_path / "s2", equiv_gate=str(work / "gate.json")))


def test_F10a_every_acap_sequence_of_the_plan_must_be_bound(tmp_path):
    from d2 import dataset as D
    from d2.tests.test_dataset import _fake_replay, _gate, _recs
    out = _fake_replay(tmp_path, _recs())
    gate = _gate(tmp_path, True, out)
    with open(out / "plan.jsonl", "a") as fh:            # the replay also planned acap1, which the gate never saw
        fh.write(json.dumps({"seq_id": "acap1", "split": "acap", "family": "acap", "kind": "acap",
                             "request_id": "d2r.acap1.0.10"}) + "\n")
    with pytest.raises(SystemExit, match="not in the gate's binding: \\['acap1'\\]"):
        D.cmd_build(_ns(out, tmp_path / "s", equiv_gate=gate))


def test_F10a_bound_gate_builds_and_is_recorded(tmp_path):
    from d2 import dataset as D
    from d2.tests.test_dataset import _fake_replay, _gate, _recs
    out = _fake_replay(tmp_path, _recs())
    gate = _gate(tmp_path, True, out)
    D.cmd_build(_ns(out, tmp_path / "s", equiv_gate=gate))
    g = json.load(open(tmp_path / "s" / "MANIFEST.json"))["checks"]["equivalence_gate"]
    assert g["verdict"] == "pass" and g["replay_binding"] == json.load(open(gate))["replay_binding"]
    # a failed real gate of this replay: refused; the override covers a fail verdict (recorded)
    bad = _gate(tmp_path, False, out)
    with pytest.raises(SystemExit, match="did not pass"):
        D.cmd_build(_ns(out, tmp_path / "s2", equiv_gate=bad))
    D.cmd_build(_ns(out, tmp_path / "s3", equiv_gate=bad, override_equiv_gate="coordinator: test"))
    g = json.load(open(tmp_path / "s3" / "MANIFEST.json"))["checks"]["equivalence_gate"]
    assert g["verdict"] == "fail" and g["override"] == "coordinator: test"


def test_F10a_report_refuses_records_of_two_replays(tmp_path):
    a = [rec(j, SAME, SAME) for j in range(10)]
    for j, r in enumerate(a):
        r["replay"] = {"acap0": {"sha256_taps": ("a" if j < 5 else "b") * 64, "sha256_ids": "c" * 64}}
    with pytest.raises(SystemExit, match="different replays"):
        report(tmp_path, {"drafts-0of1.jsonl": a})
    del a[7]["replay"]
    for r in a:
        r["replay"] = r.get("replay") and {"acap0": {"sha256_taps": "a" * 64, "sha256_ids": "c" * 64}}
    with pytest.raises(SystemExit, match="without a replay binding"):
        report(tmp_path, {"drafts-0of1.jsonl": a})


def test_F10a_replay_binding_reads_the_acap_DONE_hashes(tmp_path):
    for i in range(2):
        d = tmp_path / "taps" / f"acap{i}"
        d.mkdir(parents=True)
        json.dump({"sha256_taps": f"{i}" * 64, "sha256_ids": f"{i + 5}" * 64}, open(d / "DONE.json", "w"))
    assert E.replay_binding(str(tmp_path), 2) == {"acap0": {"sha256_taps": "0" * 64, "sha256_ids": "5" * 64},
                                                  "acap1": {"sha256_taps": "1" * 64, "sha256_ids": "6" * 64}}
    json.dump({"from": 0}, open(tmp_path / "taps" / "acap1" / "DONE.json", "w"))
    with pytest.raises(SystemExit, match="no sha256_taps"):
        E.replay_binding(str(tmp_path), 2)


# --------------------------------------------------------------------------------------------------------- F10-b
@pytest.mark.parametrize("label", ["decode_capture", "synthetic"])
def test_F10b_taps_source_label_does_not_bypass_the_gate(tmp_path, label):
    """S16's probe: a prefill-replay tree (plan request ids d2r.*) built with --taps-source decode_capture skipped the
    gate and was published as decode taps. The label must match the tree: refused, with or without a gate."""
    from d2 import dataset as D
    from d2 import replay as RP
    from d2.tests.test_dataset import _fake_replay, _gate, _recs
    out = _fake_replay(tmp_path, _recs())
    assert all(e["request_id"].startswith("d2r.") for e in RP.read_plan(str(out)))
    with pytest.raises(SystemExit, match="holds prefill_replay taps"):
        D.cmd_build(_ns(out, tmp_path / "shards", taps_source=label))
    with pytest.raises(SystemExit, match="holds prefill_replay taps"):
        D.cmd_build(_ns(out, tmp_path / "shards", taps_source=label, equiv_gate=_gate(tmp_path, True, out)))
    assert not (tmp_path / "shards").exists() and not (tmp_path / "shards.partial").exists()


def test_F10b_tree_that_is_not_a_replay_plan_is_refused(tmp_path):
    from d2 import dataset as D
    from d2.tests.test_dataset import _fake_replay, _recs
    out = _fake_replay(tmp_path, _recs())
    lines = [json.loads(x) for x in open(out / "plan.jsonl")]
    lines[1]["request_id"] = "d2c.d2p0002.0.300"                 # e.g. a future capture-generation tag
    with open(out / "plan.jsonl", "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in lines)
    assert D.tree_taps_source.__doc__ and "d2r." in D.tree_taps_source.__doc__
    for label in ("prefill_replay", "decode_capture"):
        with pytest.raises(SystemExit, match="cannot tell which taps this tree holds \\(2 of 3"):
            D.cmd_build(_ns(out, tmp_path / "shards", taps_source=label, override_equiv_gate="x"))
