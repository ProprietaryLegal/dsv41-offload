"""S24 FIX-1: the Amendment-6 bootstrap clusters on the TASK label (`family`), not the SOURCE family.

Amendment 6 (expert-continuation §B): "Resampling unit = request and source family". In this repository a source family
is a connected component of `source_docs` (d2-fix1 split_audit.py; DATASET.md), distinct from the task label in the
`family` field. D2Blocks.request_families -> pilot_eval.score -> request_totals["family"] uses the task label.

Real-data reachability (MEASURED aggregate of the private prompt manifest, no text): heldout_legal (= dev_legal) has
24 prompts in 2 task labels (12/12) and 24 source families, none spanning a task label. The pilot therefore
bootstraps dev_legal over 2 clusters where the amendment's unit gives 24.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
import torch
from safetensors.torch import save_file

from offload.dspark.d2.train.data import D2Blocks, sha256
from offload.dspark.d2.train.pilot import ARMS, Spec, digest
from offload.dspark.d2.train.pilot_eval import EVALUATOR, Evaluation, Metrics, SPLITS
from offload.dspark.d2.train.pilot_sample import RoundKey, draw_sample, key_hash
from offload.dspark.d2.train.pilot_statistics import DEV_SPLITS
from offload.dspark.d2.train.pilot_verdict import decision, outcome

POINTS = [2500, 5000, 10000, 20000]


def shard(root: Path, name: str, split: str, family: str, docs: list[str], L: int = 12, gen: int = 4) -> dict[str, str]:
    path = root/f"{name}.safetensors"
    p = torch.full((L-gen, 20), .045)
    save_file({"seq_index": torch.tensor([[0, L, gen, 0, 0, L, 0, L-gen]]),
               "tokens": torch.arange(L, dtype=torch.int32), "taps": torch.zeros(L, 15360, dtype=torch.bfloat16),
               "row_pos": torch.arange(L, dtype=torch.int32),
               "top_ids": torch.arange(20, dtype=torch.int32).expand(L-gen, -1).contiguous(), "top_lp": p.log(),
               "resid": 1-p.sum(-1), "tok_lp": p[:, 0].log(), "tok_rank": torch.ones(L-gen, dtype=torch.int32)},
              path, metadata={"schema": "d2.shard.v1", "split": split})
    meta = {"schema": "d2.shard.v1", "split": split, "W": 128, "K": 5, "topk": 20, "tap_layers": [37, 38, 39],
            "hidden": 5120, "noise_token": 128799, "sha256": sha256(path),
            "sequences": [{"seq_id": name, "source_docs": docs, "family": family, "L": L, "gen_start": gen,
                           "tokens_sha256": hashlib.sha256(torch.arange(L, dtype=torch.int32).numpy().tobytes()).hexdigest()}]}
    side = root/f"{name}.json"
    side.write_text(json.dumps(meta))
    return {"file": path.name, "sha256": sha256(path), "sidecar": side.name, "sidecar_sha256": sha256(side)}


def test_s24_fix1_bootstrap_family_is_the_source_family(tmp_path: Path) -> None:
    """Four held-out requests: two task labels, four disjoint source families (the real dev_legal shape)."""
    files = [shard(tmp_path, f"r{i}", "heldout_legal", ("chronology", "motion_outline")[i % 2], [f"doc{i}"]) for i in range(4)]
    manifest = tmp_path/"MANIFEST.json"
    manifest.write_text(json.dumps({"schema": "d2.manifest.v1", "checks": {"plumbing_errors": 0},
                                    "splits": {"heldout_legal": {"files": files}}}))
    data = D2Blocks(manifest, "heldout_legal")
    # pilot_eval.score copies data.request_families[request] into request_totals[...]["family"].
    assert len(set(data.request_families.values())) == 4, data.request_families


def test_transitive_source_components_across_shards_ignore_tasks_and_order(tmp_path: Path) -> None:
    # The final shard bridges two components whose requests have different tasks.
    files = [shard(tmp_path, name, "heldout_legal", task, docs) for name, task, docs in
             [("a", "chronology", ["x"]), ("b", "motion_outline", ["y"]),
              ("c", "chronology", ["z"]), ("bridge", "motion_outline", ["x", "y"])]]
    manifest = tmp_path/"MANIFEST.json"
    raw = {"schema": "d2.manifest.v1", "splits": {"heldout_legal": {"files": files}}}
    manifest.write_text(json.dumps(raw))
    data = D2Blocks(manifest, "heldout_legal")
    clusters = data.request_families
    assert clusters["a"] == clusters["b"] == clusters["bridge"] != clusters["c"]
    assert data.request_tasks == {"a": "chronology", "b": "motion_outline", "c": "chronology", "bridge": "motion_outline"}
    files.reverse()
    manifest.write_text(json.dumps(raw))
    assert D2Blocks(manifest, "heldout_legal").request_families == clusters


def test_source_less_captures_cluster_by_prompt_and_require_identity(tmp_path: Path) -> None:
    files = [shard(tmp_path, f"acap{i}", "acap", "acap", []) for i in range(4)]
    for i, entry in enumerate(files):
        side = tmp_path/entry["sidecar"]
        meta = json.loads(side.read_text())
        meta["sequences"][0]["prompt_sha256"] = "same-prompt" if i < 2 else f"prompt-{i}"
        side.write_text(json.dumps(meta))
        entry["sidecar_sha256"] = sha256(side)
    manifest = tmp_path/"MANIFEST.json"
    raw = {"schema": "d2.manifest.v1", "splits": {"acap": {"files": files}}}
    manifest.write_text(json.dumps(raw))
    data = D2Blocks(manifest, "acap")
    assert data.request_families["acap0"] == data.request_families["acap1"]
    assert len(set(data.request_families.values())) == 3
    side = tmp_path/files[-1]["sidecar"]
    meta = json.loads(side.read_text())
    del meta["sequences"][0]["prompt_sha256"]
    side.write_text(json.dumps(meta))
    files[-1]["sidecar_sha256"] = sha256(side)
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="missing prompt identity"):
        D2Blocks(manifest, "acap")


def evaluation(task_clusters: bool) -> tuple[Spec, Evaluation]:
    """dev_acap: 10 requests, clean +0.1 gain. dev_legal: 24 requests x 40 rounds, 12 chronology (+0.15/round)
    and 12 motion_outline (-0.025/round); FAST == REF. Each legal request is its own source family."""
    population: dict[str, list[RoundKey]] = {}
    layout: dict[str, list[tuple[str, str, int]]] = {}
    for s, n in (("acap", 10), ("heldout_legal", 24)):
        population[s] = [{"request": f"{s}-{r}", "P": 3+6*j} for r in range(n) for j in range(40)]
        layout[s] = [(f"{s}-{r}", "acap" if s == "acap" else
                      (("chronology", "motion_outline")[r % 2] if task_clusters else f"source-{r}"), 40) for r in range(n)]
    sample = draw_sample("manifest", population, dict.fromkeys(SPLITS, "k2"))
    spec = cast(Spec, {"synthetic": False, "manifest_sha256": "manifest", "sample": sample, "checkpoints": POINTS})
    def metrics(s: str, accepted: list[int]) -> Metrics:
        rows = layout[s]
        rounds = sum(n for *_, n in rows)
        return {"requests": len(rows), "rounds": rounds, "population_sha256": key_hash(sample["keys"][s]),
                "kl_by_position": [.8]*5, "loss": 1., "accepted_length": sum(accepted)/rounds,
                "kl_median_by_position": [.8]*5,
                "request_totals": [{"request_sha256": digest(req), "family": fam, "task_label": "task", "rounds": n, "accepted_sum": float(a)}
                                   for (req, fam, n), a in zip(rows, accepted)]}
    base = {s: [72]*len(layout[s]) for s in SPLITS}
    gain = {"acap": [4]*10, "heldout_legal": [6 if r % 2 == 0 else -1 for r in range(24)]}
    ev = cast(Evaluation, {"schema": "dsv41.d2.pilot-evaluation.v1", "synthetic": False, "evaluator": EVALUATOR,
                           "split_roles": DEV_SPLITS, "k2_sha256": sample["k2_sha256"], "sample_sha256": sample["sha256"],
                           "baseline": {s: metrics(s, base[s]) for s in SPLITS},
                           "arms": {a: {str(p): {s: metrics(s, [b+g for b, g in zip(base[s], gain[s])]) for s in SPLITS}
                                        for p in POINTS} for a in ARMS},
                           "checkpoints": {a: {str(p): {} for p in POINTS} for a in ARMS}})
    return spec, ev


def test_s24_fix1_decision_impact_on_the_real_dev_legal_shape() -> None:
    """Same accepted counts; only the cluster labels differ. Amendment-6 unit (source family) PASSes; the pilot's
    unit (task label) reports INCONCLUSIVE ('adaptation not demonstrated') with an extra seed."""
    spec, ev = evaluation(task_clusters=False)
    checks = outcome(spec, ev)
    assert decision(spec, ev, checks) == ("PASS", False)
    spec, ev = evaluation(task_clusters=True)
    checks = outcome(spec, ev)
    assert not checks["10000/dev_legal/REF_useful_gain"] and not checks["20000/dev_legal/FAST_useful_gain"]
    assert decision(spec, ev, checks) == ("INCONCLUSIVE", True)
