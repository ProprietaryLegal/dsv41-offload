"""dataset.py build/verify/synth on a synthetic replay tree, the G1 plumbing pass and the G1 scoring arithmetic."""

import argparse
import hashlib
import json
import os
import types

import numpy as np
import pytest
import torch

from d2 import dataset as D
from d2 import equiv as E
from d2 import g1 as G
from d2 import replay as RP
from d2 import shards as S


def _fake_replay(tmp_path, recs, corrupt_ids=False):
    seqs = tmp_path / "seqs"
    seqs.mkdir()
    for r in recs:
        toks = r.tokens.tolist()
        g = r.gen_start
        st = {"schema": "d2.genstate.v1", "pid": r.meta["seq_id"], "split": r.meta["split"], "family": r.meta["family"],
              "prompt_sha256": r.meta["prompt_sha256"], "status": "done", "prompt_ids": toks[:g], "gen_ids": toks[g:],
              "segments": [{"seg": 0, "start": 0, "n": len(toks) - g}],
              "finish_reason": "stop", "tokens_sha256": RP.tokens_sha256(toks)}
        json.dump(st, open(seqs / (r.meta["seq_id"] + ".json"), "w"))
    out = tmp_path / "replay"
    RP.cmd_plan(types.SimpleNamespace(seqs=str(seqs), acap="", out=str(out)))
    lp = out / "logprobs"
    lp.mkdir()
    for e, r in zip(sorted(RP.read_plan(str(out)), key=lambda e: e["seq_id"]), sorted(recs, key=lambda r: r.meta["seq_id"])):
        assert e["seq_id"] == r.meta["seq_id"] and e["from"] == r.r0
        np.savez(lp / (e["seq_id"] + ".npz"), top_ids=r.top_ids.numpy(), top_lp=r.top_lp.numpy(),
                 tok_lp=r.tok_lp.numpy(), tok_rank=r.tok_rank.numpy(), resid=r.resid.numpy())
        json.dump({"tokens_sha256": e["tokens_sha256"], "L": e["L"], "gen_start": e["gen_start"], "utc": "x",
                   "ties_at_edge": 0}, open(lp / (e["seq_id"] + ".json"), "w"))
        d = out / "taps" / e["seq_id"]
        d.mkdir(parents=True)
        tb = r.taps.view(torch.int16).numpy().tobytes()
        ids = np.asarray(r.tokens.tolist()[e["from"]:], dtype="<i4")
        if corrupt_ids:
            ids[3] += 1
        ib = ids.tobytes()
        (d / "taps.bin").write_bytes(tb)
        (d / "ids.bin").write_bytes(ib)
        json.dump({"from": e["from"], "end": e["L"], "rows": e["L"] - e["from"], "rows_missing": 0, "in_order": True,
                   "zero_rows": 0, "sha256_taps": hashlib.sha256(tb).hexdigest(), "sha256_ids": hashlib.sha256(ib).hexdigest()},
                  open(d / "DONE.json", "w"))
    from collections import Counter
    prompts = [{"pid": r.meta["seq_id"], "split": r.meta["split"], "family": r.meta["family"],
                "prompt_sha256": r.meta["prompt_sha256"], "source_docs": [r.meta["seq_id"]], "source": "synthetic"}
               for r in recs]
    counts = {sp: dict(Counter(p["family"] for p in prompts if p["split"] == sp))
              for sp in {p["split"] for p in prompts}}
    (out / "prompt-manifest.json").write_text(json.dumps({"schema": "d2.prompts.synthetic.v1", "prompts": prompts, "counts": counts, "n_prompts": len(prompts)}))
    return out


def _bind_acap(replay) -> dict:
    """An acap sequence's DONE.json in the fake replay tree (the plan has no acap entries; the gate binds to it)."""
    d = replay / "taps" / "acap0"
    if not (d / "DONE.json").exists():
        d.mkdir(parents=True)
        json.dump({"from": 0, "end": 10, "rows": 10, "rows_missing": 0, "in_order": True, "zero_rows": 0,
                   "sha256_taps": hashlib.sha256(b"acap0 taps").hexdigest(),
                   "sha256_ids": hashlib.sha256(b"acap0 ids").hexdigest()}, open(d / "DONE.json", "w"))
    done = json.load(open(d / "DONE.json"))
    return {"acap0": {"sha256_taps": done["sha256_taps"], "sha256_ids": done["sha256_ids"]}}


def _gate(tmp_path, ok: bool, replay, binding: dict | None = None) -> str:
    """A gate.json written by d2.equiv report itself, bound to the replay tree (S16 F10-a): 100 differing blocks, all
    equal (pass) or 5 position-1 mismatches with KL 1.0 (fail)."""
    work = tmp_path / f"gate-work-{ok}-{len(list(tmp_path.glob('gate-work-*')))}"
    work.mkdir()
    b = _bind_acap(replay) if binding is None else binding
    d = [1, 2, 3, 4, 5]
    with open(work / "drafts-0of1.jsonl", "w") as fh:
        for j in range(100):
            r = d if (ok or j >= 5) else [7, 2, 3, 4, 5]
            fh.write(json.dumps({"key": f"r{j:06d}_b0", "req": "acap0", "P": j, "realized": d, "engine": d,
                                 "D": {"drafts": d, "margin": [1.0] * 5}, "R": {"drafts": r, "margin": [1.0] * 5},
                                 "decode_rows_in_window": 128, "main_x_P_rel": 0.0, "kl_D_R": [1.0] * 5,
                                 "replay": b}) + "\n")
    out = work / "gate.json"
    E.cmd_report(types.SimpleNamespace(work=str(work), taps="", out=str(out)))
    assert json.load(open(out))["gate"]["pass"] is ok
    return str(out)


def _recs():
    return [S.synth_record("d2p0001", "train", 700, 500, seed=1, family="holding_summary"),
            S.synth_record("d2p0002", "train", 300, 100, seed=2, family="argument_section"),
            S.synth_record("d2p0003", "heldout_legal", 450, 300, seed=3, family="chronology")]


def test_build_verify_plumbing(tmp_path):
    recs = _recs()
    replay = _fake_replay(tmp_path, recs)
    out = tmp_path / "shards"
    D.cmd_build(argparse.Namespace(replay=str(replay), out=str(out), prompt_manifest=str(replay / "prompt-manifest.json"), taps_source="prefill_replay",
                                   splits="", allow_skips=False, equiv_gate=_gate(tmp_path, True, replay),
                                   override_equiv_gate="", override_plumbing=""))
    D.cmd_verify(argparse.Namespace(out=str(out)))
    G.cmd_plumbing(argparse.Namespace(shards=str(out), stride=7))
    man = json.load(open(out / "MANIFEST.json"))
    assert man["splits"]["train"]["n_seq"] == 2 and man["splits"]["heldout_legal"]["families"] == {"chronology": 1}
    assert not os.path.exists(str(out) + ".partial")
    sh = S.Shard(str(out / man["splits"]["train"]["files"][0]["file"]), verify_sha256=True)
    by = {sh.seq(i)["seq_id"]: sh.record(i) for i in range(len(sh))}
    for r in recs[:2]:
        b = by[r.meta["seq_id"]]
        assert torch.equal(b.taps, r.taps) and torch.equal(b.top_lp, r.top_lp) and torch.equal(b.tokens, r.tokens)
    assert oct(os.stat(out).st_mode & 0o777) == "0o700"


def test_build_refuses_miskeyed_rows(tmp_path):
    replay = _fake_replay(tmp_path, _recs(), corrupt_ids=True)
    out = tmp_path / "shards"
    with pytest.raises(SystemExit, match="integrity"):
        D.cmd_build(argparse.Namespace(replay=str(replay), out=str(out), prompt_manifest=str(replay / "prompt-manifest.json"),
                                       taps_source="prefill_replay", splits="", allow_skips=False,
                                       equiv_gate=_gate(tmp_path, True, replay), override_equiv_gate="",
                                       override_plumbing=""))
    assert not out.exists()


def test_synth_dataset_is_valid(tmp_path):
    out = tmp_path / "syn"
    D.cmd_synth(argparse.Namespace(out=str(out), scale=0.5))
    D.cmd_verify(argparse.Namespace(out=str(out)))
    man = json.load(open(out / "MANIFEST.json"))
    assert set(man["splits"]) == set(S.SPLITS)


def test_g1_score_round_simulation():
    tok = list(range(1000, 1030))
    seqs = {"a": {"tokens": tok, "gen_start": 10, "L": 30, "family": "chronology"}}
    perfect = {P: tok[P + 2: P + 7] + [0] * (5 - len(tok[P + 2: P + 7])) for P in range(9, 28)}
    rep = G.score(seqs, {"a": perfect}, "acap", mode="historical-acap")
    # perfect drafts: rounds at P = 9, 15, 21 accept 5 each; P = 27 has one target left -> 1 match then None (end)
    assert rep["rounds"] == 3 and rep["k2_accepted_per_round"] == 5.0
    wrong = {P: [0] * 5 for P in range(9, 28)}
    rep = G.score(seqs, {"a": wrong}, "acap", mode="historical-acap")
    assert rep["k2_accepted_per_round"] == 0.0 and rep["rounds"] == 19 and not rep["g1"]["pass"]   # P = 9..27
    assert rep["conditional_acceptance"][0] == 0.0


def test_g1_binds_on_acap_only():
    """Coordinator decision 2026-10-23: G1(b) is scored on the acap split; heldout_legal is information."""
    tok = list(range(1000, 1030))
    seqs = {"a": {"tokens": tok, "gen_start": 10, "L": 30, "family": "chronology"}}
    wrong = {P: [0] * 5 for P in range(9, 28)}
    acap = G.score(seqs, {"a": wrong}, "acap", mode="historical-acap")["g1"]
    assert acap["binding"] is True and acap["pass"] is False and acap["within_tolerance"] is False
    info = G.score(seqs, {"a": wrong}, "heldout_legal", mode="historical-acap")["g1"]
    assert info["binding"] is False and info["pass"] is None and info["within_tolerance"] is False
    assert info["status"] == "INFORMATION"
    assert G.g1_verdict(1.80, "acap", mode="historical-acap")["pass"] is True and G.g1_verdict(1.95, "acap", mode="historical-acap")["pass"] is False
    assert G.BINDING_SPLIT == "dev"

def test_builder_refuses_duplicate_even_with_allow_skips(tmp_path):
    from d2 import dataset as D
    from d2.tests.test_dataset import _fake_replay, _gate, _recs
    from d2.tests.test_s16_fixes import _ns
    replay = _fake_replay(tmp_path, _recs())
    p = replay / 'plan.jsonl'
    p.write_text(p.read_text() + p.read_text().splitlines()[0] + '\n')
    with pytest.raises(SystemExit, match='duplicate'):
        D.cmd_build(_ns(replay, tmp_path / 'shards', allow_skips=True,
                        equiv_gate=_gate(tmp_path, True, replay)))
    assert not (tmp_path / 'shards').exists()


def test_served_build_requires_sequence_identity_context(tmp_path):
    from d2.tests.test_s16_fixes import _ns
    replay = _fake_replay(tmp_path, _recs())
    path = replay / "prompt-manifest.json"
    pm = json.loads(path.read_text())
    pm["schema"] = "d2.prompts.v1"
    for p in pm["prompts"]:
        p["source"] = "public_fixture"
    path.write_text(json.dumps(pm))
    with pytest.raises(SystemExit, match="sequence-context"):
        D.cmd_build(_ns(replay, tmp_path / "shards", equiv_gate=_gate(tmp_path, True, replay)))
    assert not (tmp_path / "shards").exists()
