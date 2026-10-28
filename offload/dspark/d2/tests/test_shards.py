"""Schema d2.shard.v1 round trip on synthetic sequences: writer, reader, blocks, integrity checks, V1 adapter."""

import json
import os

import pytest
import torch

from d2 import shards as S


def _write(tmp_path, split="heldout_legal", lens=((900, 600), (300, 100), (420, 410))):
    recs = [S.synth_record(f"s{i}", split, L, g, seed=i) for i, (L, g) in enumerate(lens)]
    os.makedirs(tmp_path / split, exist_ok=True)
    ent = S.write_shard(str(tmp_path / split / f"{split}-00000"), split, recs, "synthetic")
    return recs, ent


def test_round_trip_and_offsets(tmp_path):
    recs, ent = _write(tmp_path)
    sh = S.Shard(ent["file"], verify_sha256=True)
    assert sh.structure_errors() == []
    assert ent["n_rows"] == sum(r.L - S.r0_of(r.gen_start) for r in recs)
    assert ent["n_span"] == sum(r.L - r.gen_start for r in recs)
    for i, r in enumerate(recs):
        back = sh.record(i)
        assert torch.equal(back.tokens, r.tokens) and back.gen_start == r.gen_start
        assert torch.equal(back.taps, r.taps) and torch.equal(back.top_ids, r.top_ids)
        assert torch.equal(back.tok_rank, r.tok_rank) and torch.equal(back.resid, r.resid)
        assert S.check_record(back) == []
        s = sh.seq(i)
        assert s["r0"] == max(0, r.gen_start - S.W) and s["tokens_sha256"] == S.tokens_sha256(r.tokens)
    side = json.load(open(ent["sidecar"]))
    assert side["schema"] == S.SCHEMA and side["W"] == 128 and side["tap_layers"] == [37, 38, 39]


def test_block_semantics(tmp_path):
    recs, ent = _write(tmp_path)
    sh = S.Shard(ent["file"])
    r = recs[0]
    blocks = sh.blocks(0)
    assert blocks[0] == r.gen_start - 1 and blocks[-1] == r.L - 3
    b = sh.block(0, r.gen_start - 1)
    assert b["window_pos"] == list(range(r.gen_start - S.W, r.gen_start))
    assert torch.equal(b["window_taps"], r.taps[: S.W])            # r0 = gen_start - W: the first W rows
    assert b["anchor"] == int(r.tokens[r.gen_start])
    t0 = b["targets"][0]
    assert t0["pos"] == r.gen_start + 1 and t0["prev"] == b["anchor"]
    assert torch.equal(t0["top_ids"], r.top_ids[1])
    last = sh.block(0, r.L - 3)
    assert last["targets"][0]["pos"] == r.L - 1 and last["targets"][1:] == [None] * 4
    with pytest.raises(IndexError):
        sh.block(0, r.gen_start - 2)
    # short prompt: r0 = 0 and the first window is clipped at 0
    b1 = sh.block(1, recs[1].gen_start - 1)
    assert b1["window_pos"] == list(range(0, recs[1].gen_start))


def test_check_record_catches_corruption():
    r = S.synth_record("x", "train", 400, 300, seed=7)
    assert S.check_record(r) == []
    bad = S.synth_record("x", "train", 400, 300, seed=7)
    bad.top_lp[3] = bad.top_lp[3].flip(0)
    assert any("non-increasing" in e for e in S.check_record(bad))
    bad = S.synth_record("x", "train", 400, 300, seed=7)
    bad.taps[5, 9] = float("nan")
    assert any("non-finite taps" in e for e in S.check_record(bad))
    bad = S.synth_record("x", "train", 400, 300, seed=7)
    j = int((bad.tok_rank <= S.TOPK).nonzero()[0])
    bad.tok_rank[j] = S.TOPK + 1
    assert any("tok_rank" in e for e in S.check_record(bad))
    bad = S.synth_record("x", "train", 400, 300, seed=7)
    bad.resid[0] += 0.01
    assert any("resid" in e for e in S.check_record(bad))
    bad = S.synth_record("x", "train", 400, 300, seed=7)
    bad.taps = bad.taps[1:]
    assert any("taps" in e for e in S.check_record(bad))


def test_writer_refuses_mixed_splits(tmp_path):
    a = S.synth_record("a", "train", 300, 200, seed=1)
    b = S.synth_record("b", "dev", 300, 200, seed=2)
    with pytest.raises(ValueError):
        S.write_shard(str(tmp_path / "x"), "train", [a, b], "synthetic")


def test_capture_set_adapter(tmp_path):
    recs, ent = _write(tmp_path)
    cs = S.ShardCaptureSet([ent["file"]])
    assert cs.plumbing == []
    n_blocks = sum(r.L - 2 - (r.gen_start - 1) for r in recs)
    assert len(cs.samples) == n_blocks
    assert len({s.key for s in cs.samples}) == n_blocks             # keys unique, r*_b* form
    s = next(x for x in cs.samples if x.req == "s0" and x.P == recs[0].gen_start + 10)
    assert s.anchor == int(recs[0].tokens[s.P + 1])
    assert s.realized == [int(recs[0].tokens[t]) for t in range(s.P + 2, s.P + 7)]
    assert s.engine_drafts_placeholder and s.engine_drafts == [S.NOISE] * 5
    ai = cs.arm_inputs(s)
    assert [p for p, _, _ in ai["window"]] == cs.window(s) and len(ai["window"]) == S.W
    assert all(kind == "taps" for _, kind, _ in ai["window"])
    p0, _, t0 = ai["window"][0]
    assert torch.equal(t0, recs[0].taps[p0 - S.r0_of(recs[0].gen_start)])
    assert cs.rows["s0"][s.P].tok == int(recs[0].tokens[s.P]) and cs.window_status(s)["missing"] == []
    # supplied drafts replace the placeholders
    cs2 = S.ShardCaptureSet([ent["file"]], seq_ids={"s1"}, drafts={})
    assert {x.req for x in cs2.samples} == {"s1"}
    P1 = recs[1].gen_start + 3
    cs3 = S.ShardCaptureSet([ent["file"]], seq_ids={"s1"}, drafts_by_seq={"s1": {P1: [1, 2, 3, 4, 5]}})
    got = [x for x in cs3.samples if x.P == P1][0]
    assert got.engine_drafts == [1, 2, 3, 4, 5] and not got.engine_drafts_placeholder
