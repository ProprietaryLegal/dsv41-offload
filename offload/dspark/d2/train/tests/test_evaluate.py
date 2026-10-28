from __future__ import annotations

import pytest
import torch
import json
from pathlib import Path

from offload.dspark.d2.train.evaluate import Score, bucket_kl, clustered, g2, packaging_gate, select_dev
from offload.dspark.d2.train import evaluate
from offload.dspark.d2.train.tests.test_data import write_shard
from offload.dspark.d2.train.data import sha256


def test_clustered_bootstrap_resamples_whole_requests() -> None:
    rows: list[Score] = [{"request": "a", "candidate": 2., "k2": 1., "kl_candidate": [.1]*5, "kl_baseline": [.2]*5} for _ in range(100)]
    rows += [{"request": "b", "candidate": 0., "k2": 1., "kl_candidate": [.1]*5, "kl_baseline": [.2]*5}]
    s = clustered(rows, iterations=1000)
    assert s["requests"] == 2
    assert s["ci95_request_cluster"] == [-1., 1.]
    assert s["worst_request_gain"] == -1


def test_g2_all_conditions_and_dev_only_selection() -> None:
    passing = {"requests": 20, "gain": .2, "ci95_request_cluster": [.1, .3], "worst_request_gain": 0.,
               "kl_candidate": [.1]*5, "kl_baseline": [.2]*5}
    general, acap = {**passing, "requests": 10, "gain": -.02}, {**passing, "requests": 4}
    assert g2(passing, general, acap)["verdict"] == "PASS"
    assert g2(passing, {**general, "gain": -.031}, acap)["verdict"] == "FAIL"
    assert g2(passing, general, {**acap, "worst_request_gain": -.201})["verdict"] == "FAIL"
    assert g2({**passing, "kl_candidate": [.1, .1, .2, .1, .1]}, general, acap)["verdict"] == "FAIL"
    with pytest.raises(ValueError, match="dev-only"):
        select_dev([{"split": "heldout_legal"}])


def test_bucket_scorer_and_packaging_fail_closed() -> None:
    z = torch.tensor([[1., 2., 3., 4.]])
    lp = z.log_softmax(-1)
    ids = torch.tensor([[3, 2]])
    chosen = lp.gather(-1, ids)
    tail = 1-chosen.exp().sum(-1)
    torch.testing.assert_close(bucket_kl(z, ids, chosen, tail), torch.zeros(1, dtype=torch.float64), atol=2e-7, rtol=0)
    cl = {"classes": {"agree": 50}, "samples": 50, "representation": "native",
          "relerr_p99": {"s0.attn_out": .0011}, "arith_band_p99": {"s0.attn_out": .001}}
    assert packaging_gate(.2, .15, cl)["pass"]
    assert not packaging_gate(.2, .149, cl)["pass"]
    assert not packaging_gate(.2, .2, {**cl, "classes": {"agree_but_state_flag": 1}})["pass"]


def test_score_uses_fixed_k2_rounds_and_real_g1_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path/"data"
    root.mkdir()
    manifest = write_shard(root, split="dev", L=20)
    k2 = tmp_path/"dev.k2_drafts.jsonl"
    k2.write_text(json.dumps({"seq_id": "p1", "drafts": {str(P): [t if t < 20 else 0 for t in range(P+2, P+7)] for P in range(3, 18)}})+"\n")
    candidate, baseline = tmp_path/"candidate", tmp_path/"baseline"
    candidate.mkdir()
    baseline.mkdir()
    adapter = tmp_path/"adapter.pt"
    torch.save({"trainable": {}}, adapter)
    for P in (3, 9):  # K2 accepts five, so the next round moves by six
        drafts = list(range(P+2, P+6))+[0]
        base = torch.zeros(5, 129280)
        base[torch.arange(5), torch.tensor(drafts)] = 1
        common = {"P": P, "anchor": P+1, "sample": {"req": "p1", "engine_drafts_placeholder": True}}
        torch.save({**common, "drafts": drafts, "base_logits": base, "meta": {"adapter_sha256": sha256(adapter)}}, candidate/f"r{P:06d}_b0.pt")
        torch.save({**common, "drafts": [0]*5, "base_logits": torch.zeros_like(base)}, baseline/f"r{P:06d}_b0.pt")
    monkeypatch.setattr(evaluate, "markov", lambda *_: (torch.zeros(129280, 1), torch.zeros(129280, 1)))
    summary = evaluate.score_dumps(manifest, "dev", candidate, baseline, k2, adapter, tmp_path)
    assert summary["metrics"]["rounds"] == 2
    assert summary["metrics"]["candidate"] == 4
    assert summary["metrics"]["k2"] == 5
    assert summary["metrics"]["gain"] == -1
    assert summary["k2_jsonl_sha256"] is not None
    package_root = tmp_path/"merged"
    for path in candidate.glob("*.pt"):
        dump = torch.load(path, weights_only=False)
        dump["meta"]["package_identity"] = "merged-byte-identity"
        torch.save(dump, path)
    package_emb = torch.ones(129280, 1)
    package_head = torch.linspace(-.2, .2, 129280)[:, None]
    monkeypatch.setattr(evaluate, "package_markov", lambda *_: (package_emb, package_head,
                        {"drafter_identity": "merged-byte-identity"}))
    merged = evaluate.score_dumps(manifest, "dev", candidate, baseline, k2, adapter, tmp_path, package_root)
    assert merged["package_identity"] == "merged-byte-identity"
    assert merged["metrics"]["gain"] == summary["metrics"]["gain"]
    assert merged["metrics"]["kl_candidate"] != summary["metrics"]["kl_candidate"]
    with pytest.raises(ValueError, match="package identity"):
        evaluate.score_dumps(manifest, "dev", candidate, baseline, k2, adapter, tmp_path)
    torch.save({"trainable": {}, "different": True}, adapter)
    with pytest.raises(ValueError, match="checkpoint identity"):
        evaluate.score_dumps(manifest, "dev", candidate, baseline, k2, adapter, tmp_path)
