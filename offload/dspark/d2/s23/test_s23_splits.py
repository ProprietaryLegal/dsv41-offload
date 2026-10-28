"""S23 probes of D2-FIX1 item 2 (counts, split/identity preflight) on real and minimal data.

Real-data probes read the private prompt set / acap exports / live generation states READ-ONLY and write the replay
plan (ids and hashes only) under pytest's tmp dir; they skip where the private files are absent. No text is printed.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import pytest

from d2 import replay as R
from d2.split_audit import audit_plan, validate_manifest

PRIV = Path(os.environ.get('D2_PRIVATE_TEST_ROOT', '/opt/dsv41/data/d2'))
real = pytest.mark.skipif(not (PRIV / "prompts/manifest.json").exists() or not (PRIV / "acap").exists(),
                          reason="private D2 inputs absent")


@real
def test_real_counts_546_minus_8() -> None:
    m = json.loads((PRIV / "prompts/manifest.json").read_text())
    rows = [json.loads(x) for x in open(PRIV / "prompts/prompts.jsonl")]
    assert Counter(r["split"] for r in rows) == {"train": 472, "dev": 30, "heldout_legal": 24, "heldout_general": 12}
    assert len(m["excluded_pids"]) == 8 and not {r["pid"] for r in rows} & set(m["excluded_pids"])
    nums = {int(r["pid"][3:]) for r in rows}
    assert {f"d2p{i:04d}" for i in set(range(546)) - nums} == set(m["excluded_pids"])
    validate_manifest(m)


@real
def test_real_acap_plan_passes_the_split_audit(tmp_path: Path) -> None:
    """S23 FIX 1 regression: accept two distinct captures of one prompt in the real population."""
    out = tmp_path / "plan"
    out.mkdir(mode=0o700)
    R.cmd_plan(argparse.Namespace(seqs="", acap=str(PRIV / "acap"), out=str(out)))
    plan = R.read_plan(str(out))
    acap = [e for e in plan if e["kind"] == "acap"]
    shas = Counter(e["prompt_sha256"] for e in acap)
    assert len(acap) == 4 and max(shas.values()) == 2                       # two acap share one prompt
    assert len({e["tokens_sha256"] for e in acap}) == 4                       # four distinct sequences
    m = json.loads((PRIV / "prompts/manifest.json").read_text())
    audit_plan(plan, m)


def test_minimal_two_samples_of_one_acap_prompt_are_accepted() -> None:
    prompts = [{"pid": "p1", "split": "train", "family": "holding_summary", "prompt_sha256": "h1",
                "source_docs": ["u1"]}]
    m = {"prompts": prompts, "n_prompts": 1, "counts": {"train": {"holding_summary": 1}}}
    plan = [{"seq_id": f"acap{i}", "split": "acap", "family": "acap", "kind": "acap", "prompt_sha256": "hA",
             "tokens_sha256": f"t{i}"} for i in range(2)]
    assert audit_plan(plan, m)["by_split"] == {"acap": 2}


def test_overlap_attempts_are_refused() -> None:
    base = [{"pid": "p1", "split": "train", "family": "holding_summary", "prompt_sha256": "h1", "source_docs": ["u1"]},
            {"pid": "p2", "split": "dev", "family": "holding_summary", "prompt_sha256": "h2", "source_docs": ["u2"]}]

    def man(ps):
        c: dict = {}
        for p in ps:
            c.setdefault(p["split"], Counter())[p["family"]] += 1
        return {"prompts": ps, "n_prompts": len(ps), "counts": {k: dict(v) for k, v in c.items()}}
    # same text identity in two splits (counts kept consistent)
    dup = [dict(base[0]), dict(base[1], prompt_sha256="h1")]
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(man(dup))
    # a document chain p1:{u1,u3} train, p2:{u3,u2} dev: the shared u3 joins them
    chain = [dict(base[0], source_docs=["u1", "u3"]), dict(base[1], source_docs=["u3", "u2"])]
    with pytest.raises(ValueError, match="source family"):
        validate_manifest(man(chain))
    # an excluded ID still present
    with pytest.raises(ValueError, match="excluded"):
        validate_manifest({**man(base), "excluded_pids": {"p1": "near copy"}})
    # a train state relabelled dev in the plan
    plan = [{"seq_id": "p1", "split": "dev", "family": "holding_summary", "prompt_sha256": "h1", "kind": "gen"}]
    with pytest.raises(ValueError, match="differs"):
        audit_plan(plan, man(base))
    # NOT detectable here by design: a refiled near-copy under a different document identity (prompt-build screen)
    near = [dict(base[0], source_docs=["opinion-A"]), dict(base[1], source_docs=["opinion-A-refiled"])]
    validate_manifest(man(near))
