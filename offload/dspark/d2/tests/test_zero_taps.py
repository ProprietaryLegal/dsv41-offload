"""Zero taps rows (CONTRACT.md "tap validity"): eager prefill steps run layers >= 20 only on each request's last
min(q, 128) rows and leave the trimmed rows' taps zero. The engine keeps every row of a request with prompt_logprobs
(nvidia/model_state.py _keeps_rows_np), so replay rows are all computed; these tests check that a zero row, if one
ever appears, is counted by the overlay and refused by replay copy, equiv and the dataset rules."""

from __future__ import annotations

import hashlib
import json
import os
import types

import numpy as np
import pytest
import torch

from d2 import equiv as E
from d2 import replay as RP
from d2 import shards as S
from d2.tests.test_replay import _FakeServe, _gen_state
from d2.tests.test_replay_overlay import _call, _load, _round


def test_overlay_counts_zero_rows_and_records_them_once(tmp_path):
    ns = _load(tmp_path, env=True)
    spec = ns["DSparkSpeculator"]()
    rid = "cmpl-d2r.pZ.0.300-0-aa"
    for lo, n in ((0, 200), (200, 100)):
        ib, aux = _round([(rid, 5, lo, n)])
        if lo == 0:
            for t in aux:                              # a trimmed step: only the last 128 rows computed
                t[:200 - 128] = 0
        _call(spec, ib, aux)
    done = json.load(open(tmp_path / "pZ" / "DONE.json"))
    assert done["rows_missing"] == 0 and done["zero_rows"] == 72 and done["first_zero_row"] == 0
    errs = [json.loads(x) for x in open(tmp_path / "ERRORS.jsonl")]
    assert len(errs) == 1 and errs[0]["kind"] == "zero_taps" and errs[0]["position"] == 0


def test_overlay_one_zero_layer_counts(tmp_path):
    ns = _load(tmp_path, env=True)
    spec = ns["DSparkSpeculator"]()
    ib, aux = _round([("cmpl-d2r.pY.0.50-0-aa", 6, 0, 50)])
    aux[2][17] = 0                                     # layer 39 of position 17 only
    _call(spec, ib, aux)
    done = json.load(open(tmp_path / "pY" / "DONE.json"))
    assert done["zero_rows"] == 1 and done["first_zero_row"] == 17


def test_overlay_clean_replay_has_zero_rows_0(tmp_path):
    ns = _load(tmp_path, env=True)
    spec = ns["DSparkSpeculator"]()
    ib, aux = _round([("cmpl-d2r.pX.0.40-0-aa", 7, 0, 40)])
    _call(spec, ib, aux)
    assert json.load(open(tmp_path / "pX" / "DONE.json"))["zero_rows"] == 0
    assert not (tmp_path / "ERRORS.jsonl").exists()


def test_check_record_refuses_zero_rows():
    toks = [0] + [100 + i for i in range(1, 300)]
    g = 200
    n = 300 - g
    taps = torch.randn(300 - S.r0_of(g), S.TAPW).to(torch.bfloat16)
    lp = torch.full((n, S.TOPK), -10.0)
    lp[:, 0] = -0.01
    ids = torch.stack([torch.tensor(toks[g + j:g + j + 1] + [5000 + k for k in range(S.TOPK - 1)]) for j in range(n)])
    mass = lp.double().exp().sum(-1)
    rec = S.SeqRecord({"seq_id": "z"}, torch.tensor(toks, dtype=torch.int32), g, taps, ids.int(), lp,
                      lp[:, 0].clone(), torch.ones(n, dtype=torch.int32), (1 - mass).clamp(0, 1).float())
    assert S.check_record(rec) == []
    taps[5, 2 * 5120:] = 0
    err = S.check_record(rec)
    assert len(err) == 1 and "1 taps rows with a layer all zero" in err[0] and f"position {S.r0_of(g) + 5}" in err[0]


def _copy_with_done(tmp_path, **done_kw):
    seqs = tmp_path / "seqs"
    seqs.mkdir()
    prompt, gen = list(range(1, 300)), [100 + j for j in range(60)]
    _gen_state(seqs, "d2p0007", "train", prompt, gen)
    out = tmp_path / "replay"
    RP.cmd_plan(types.SimpleNamespace(seqs=str(seqs), acap="", out=str(out)))
    e = RP.read_plan(str(out))[0]
    remote = tmp_path / "spark" / "d2p0007"
    remote.mkdir(parents=True)
    rows = e["L"] - e["from"]
    taps = os.urandom(rows * RP.ROW_BYTES)
    ids = np.asarray((prompt + gen)[e["from"]:], dtype="<i4").tobytes()
    (remote / "taps.bin").write_bytes(taps)
    (remote / "ids.bin").write_bytes(ids)
    (remote / "ledger.jsonl").write_text('{"round": 0}\n')
    done = {"seq": "d2p0007", "from": e["from"], "end": e["L"], "rows": rows, "rows_missing": 0, "in_order": True,
            "sha256_taps": hashlib.sha256(taps).hexdigest(), "sha256_ids": hashlib.sha256(ids).hexdigest(),
            **done_kw}
    (remote / "DONE.json").write_text(json.dumps(done))
    srv = _FakeServe()
    try:
        RP.cmd_copy(types.SimpleNamespace(out=str(out), metrics=srv.url + "/metrics",
                                          remote_dir=str(tmp_path / "spark"), ssh="bash -c", rate=0, limit=0))
    finally:
        srv.httpd.shutdown()
    return out


def test_copy_refuses_zero_rows(tmp_path):
    with pytest.raises(SystemExit, match="zero_rows 3"):
        _copy_with_done(tmp_path, zero_rows=3, first_zero_row=240)
    assert not (tmp_path / "replay" / "taps" / "d2p0007" / "taps.bin").exists()


def test_copy_refuses_an_overlay_without_the_count(tmp_path):
    with pytest.raises(SystemExit, match="zero_rows None"):
        _copy_with_done(tmp_path)


def test_copy_accepts_zero_rows_0(tmp_path):
    out = _copy_with_done(tmp_path, zero_rows=0)
    assert (out / "taps" / "d2p0007" / "taps.bin").exists()


def test_equiv_load_prefill_refuses_zero_rows(tmp_path):
    L, toks = 40, list(range(10, 50))
    d = tmp_path / "taps" / "acap0"
    d.mkdir(parents=True)
    t = torch.randn(L, E.ROW).to(torch.bfloat16)
    t[7, :5120] = 0
    t.view(torch.int16).numpy().tofile(d / "taps.bin")
    np.asarray(toks, dtype="<i4").tofile(d / "ids.bin")
    (d / "DONE.json").write_text(json.dumps({"from": 0, "end": L, "rows_missing": 0, "zero_rows": 1}))
    with pytest.raises(RuntimeError, match="zero_rows 1"):
        E.load_prefill(str(tmp_path), "acap0", L, toks)
    (d / "DONE.json").write_text(json.dumps({"from": 0, "end": L, "rows_missing": 0, "zero_rows": 0}))
    with pytest.raises(RuntimeError, match="1 taps rows with a layer all zero"):
        E.load_prefill(str(tmp_path), "acap0", L, toks)


def test_engine_keeps_prompt_logprobs_rows():
    """The engine source this rests on (the image's model_state.py) still keeps every row of a prompt_logprobs request.
    Skipped where the extracted image source is absent."""
    p = "/opt/dsv41/data/src-image/vllm/models/deepseek_v41/nvidia/model_state.py"
    if not os.path.exists(p):
        pytest.skip("no extracted image source on this host")
    src = open(p).read()
    assert "params is not None and params.prompt_logprobs is not None" in src
    assert "self._keeps_rows_np[input_batch.idx_mapping_np[:num_reqs]],\n            query_lens," in src


def test_contract_names_the_prompt_logprobs_exception():
    """CONTRACT.md section 2 'tap validity' carries the exception the replay relies on, and its backstop."""
    text = open(os.path.join(os.path.dirname(__file__), "..", "..", "CONTRACT.md")).read()
    row = next(line for line in text.splitlines() if line.startswith("| tap validity |"))
    assert "prompt logprobs keeps every row" in row and "_keeps_rows_np" in row and "e0760ef" in row
    assert "| 1.8 note | 2026-10-23 |" in text
