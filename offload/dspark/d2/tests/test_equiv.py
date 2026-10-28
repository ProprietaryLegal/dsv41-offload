"""Equivalence tool pieces that run without captures: prefill-file validation and the gate report. The NativeRef
draft path was checked end to end on Controller (identical taps -> 7/7 blocks equal; 2 % tap noise -> 5/7, gate FAIL)."""

import json
import os
import types

import numpy as np
import pytest
import torch

from d2 import equiv as E


def _replay(tmp_path, L, toks, frm=0, missing=0):
    d = tmp_path / "taps" / "acap0"
    d.mkdir(parents=True)
    t = torch.randn(L, E.ROW).to(torch.bfloat16)
    t.view(torch.int16).numpy().tofile(d / "taps.bin")
    np.asarray(toks, dtype="<i4").tofile(d / "ids.bin")
    (d / "DONE.json").write_text(json.dumps({"from": frm, "end": L, "rows_missing": missing, "zero_rows": 0}))
    return t


def test_load_prefill_round_trip_and_refusals(tmp_path):
    toks = list(range(10, 50))
    t = _replay(tmp_path, 40, toks)
    got = E.load_prefill(str(tmp_path), "acap0", 40, toks)
    assert torch.equal(got, t)
    with pytest.raises(RuntimeError, match="row ids differ"):
        E.load_prefill(str(tmp_path), "acap0", 40, toks[:-1] + [7])


def test_load_prefill_refuses_partial_replay(tmp_path):
    toks = list(range(10, 50))
    _replay(tmp_path, 40, toks, frm=5)
    with pytest.raises(RuntimeError, match="not a full"):
        E.load_prefill(str(tmp_path), "acap0", 40, toks)


def _rec(key, d, r, realized, engine, decode_rows=5):
    return {"key": key, "req": "acap0", "P": 1, "realized": realized, "engine": engine,
            "D": {"drafts": d, "margin": [0.5] * 5}, "R": {"drafts": r, "margin": [0.1] * 5},
            "decode_rows_in_window": decode_rows, "main_x_P_rel": 0.001}


def test_report_gate(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    real = [1, 2, 3, 4, 5]
    recs = [_rec(f"r{i:06d}_b0", real, real, real, real) for i in range(199)]
    recs.append(_rec("r999999_b0", [1, 2, 9, 9, 9], [1, 2, 3, 9, 9], real, [1, 9, 9, 9, 9]))
    with open(work / "drafts-0of1.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    out = tmp_path / "gate.json"
    E.cmd_report(types.SimpleNamespace(work=str(work), taps="", out=str(out)))
    rep = json.load(open(out))
    assert rep["blocks"] == 200 and rep["block_equal_frac"] == 0.995 and rep["gate"]["pass"]
    assert rep["first_mismatch_position"] == {"2": 1} and rep["mismatch_min_margin_quantiles"]["0.5"] == 0.1
    assert rep["committed_trajectory_proxy"]["engine"]["mean_accepted"] == round((199 * 5 + 1) / 200, 4)
    with open(work / "drafts-0of1.jsonl", "a") as fh:
        for i in range(3):
            fh.write(json.dumps(_rec(f"x{i}", [9] * 5, real, real, real)) + "\n")
    E.cmd_report(types.SimpleNamespace(work=str(work), taps="", out=str(out)))
    assert not json.load(open(out))["gate"]["pass"]


def test_match_len():
    assert E.match_len([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 5
    assert E.match_len([1, 2, 0, 4, 5], [1, 2, 3, 4, 5]) == 2
    assert E.match_len([1, 2, 3], [1, None, 3]) is None
