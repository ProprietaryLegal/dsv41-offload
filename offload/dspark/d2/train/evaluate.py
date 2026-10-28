"""V1 paired replay scoring and dev-only selection/G2/packaging gates.

Scores are aggregate-only. Candidate logits use that candidate's Markov head;
using the stock Markov head after adapting it would invalidate both KL and
V1's own-draft self-consistency check. Never score NOISE placeholder baselines.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any, TypedDict, cast

import torch
from torch import Tensor

from ...native.official import OfficialShards
from .data import D2Blocks, sha256
from .loss import bucket_divergence
from .replay import read_checkpoint


class Score(TypedDict):
    request: str
    candidate: float
    k2: float
    kl_candidate: list[float | None]
    kl_baseline: list[float | None]


def match_length(drafts: list[int], truth: list[int | None]) -> int | None:
    for i, (d, t) in enumerate(zip(drafts, truth)):
        if t is None:
            return None
        if d != t:
            return i
    return len(drafts)


def clustered(rows: list[Score], iterations: int = 10000, seed: int = 23) -> dict[str, Any]:
    if not rows or iterations < 100:
        raise ValueError("empty scores or insufficient bootstrap iterations")
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by[r["request"]].append(r["candidate"]-r["k2"])
    rng = random.Random(seed)
    keys = sorted(by)
    boots = []
    for _ in range(iterations):
        samples = [by[rng.choice(keys)] for _ in keys]
        boots.append(sum(map(sum, samples))/sum(map(len, samples)))
    boots.sort()
    return {"rounds": len(rows), "requests": len(by), "candidate": mean(r["candidate"] for r in rows),
            "k2": mean(r["k2"] for r in rows), "gain": mean(r["candidate"]-r["k2"] for r in rows),
            "ci95_request_cluster": [boots[int(iterations*.025)], boots[int(iterations*.975)]],
            "worst_request_gain": min(mean(v) for v in by.values()),
            "kl_candidate": [mean(xs) if (xs := [cast(float, r["kl_candidate"][i]) for r in rows if r["kl_candidate"][i] is not None]) else None for i in range(5)],
            "kl_baseline": [mean(xs) if (xs := [cast(float, r["kl_baseline"][i]) for r in rows if r["kl_baseline"][i] is not None]) else None for i in range(5)]}


def g2(legal: dict[str, Any], general: dict[str, Any], acap: dict[str, Any]) -> dict[str, Any]:
    checks = {"legal_requests_ge20": legal["requests"] >= 20, "general_requests_ge10": general["requests"] >= 10,
              "acap_requests_eq4": acap["requests"] == 4, "legal_gain_ge015": legal["gain"] >= .15,
              "legal_ci_positive": legal["ci95_request_cluster"][0] > 0, "general_gain_ge_minus003": general["gain"] >= -.03,
              "every_request_ge_minus02": all(s["worst_request_gain"] >= -.2 for s in (legal, general, acap)),
              "kl_lower_every_position": all(a is not None and b is not None and a < b
                         for s in (legal, general, acap) for a, b in zip(s["kl_candidate"], s["kl_baseline"]))}
    return {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "legal": legal, "general": general, "acap": acap}


def select_dev(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries or any(s["split"] != "dev" for s in summaries):
        raise ValueError("checkpoint selection is dev-only")
    if len({s["dataset"] for s in summaries}) != 1:
        raise ValueError("dev dataset changed between candidates")
    if any(any(x is None for x in s["metrics"]["kl_candidate"]) for s in summaries):
        raise ValueError("missing dev KL position")
    return min(summaries, key=lambda s: sum(w*k for w, k in zip((1., .8, .65, .5, .4), s["metrics"]["kl_candidate"])))


def packaging_gate(before: float, after: float, classification: dict[str, Any], expected_identity: str | None = None) -> dict[str, Any]:
    classes = classification["classes"]
    # Evaluate the specified decimal gate without binary .20*.75 rounding
    # above the exact .15 boundary. No statistical tolerance is added.
    checks = {"gain_retained_ge75pct": before > 0 and Decimal(str(after)) >= Decimal(".75")*Decimal(str(before)),
              "zero_state_defects": classes.get("state_provenance_defect", 0) == 0 and classes.get("agree_but_state_flag", 0) == 0,
              "zero_unexplained": classes.get("unexplained", 0) == 0,
              "layer_excess_lt3x": bool(classification.get("relerr_p99"))
                         and set(classification["relerr_p99"]) == set(classification.get("arith_band_p99", {}))
                         and all(error < 3*classification["arith_band_p99"][name]
                                 or error == classification["arith_band_p99"][name] == 0
                                 for name, error in classification["relerr_p99"].items()),
              "native_representation": classification.get("representation") == "native",
              "nonempty_classification": classification["samples"] >= 50,
              "own_reference_identity": expected_identity is None or classification.get("candidate_reference_identity") == expected_identity}
    return {"pass": all(checks.values()), "checks": checks, "gain_before": before, "gain_after": after}


def bucket_kl(logits: Tensor, ids: Tensor, lp: Tensor, tail: Tensor) -> Tensor:
    z = logits.double()
    logz = z.logsumexp(-1)
    top = z.gather(-1, ids.long()) - logz[:, None]
    complement = z.scatter(-1, ids.long(), -torch.inf).logsumexp(-1)-logz
    return bucket_divergence(top, complement, ids, lp.double(), tail.double(), "kl")


def markov(adapter: dict[str, Any] | None, native: Path) -> tuple[Tensor, Tensor]:
    if adapter is not None:
        state = adapter["trainable"]
        prefix = "t.mtp__2__markov_head__"
        return state[prefix+"embed__weight"].float(), state[prefix+"head__weight"].float()
    return native_markov(native)


def native_markov(native: Path, manifest: Path | None = None) -> tuple[Tensor, Tensor]:
    source = OfficialShards(str(native)) if manifest is None else OfficialShards(str(native), str(manifest))
    names = ["mtp.2.markov_head.embed.weight", "mtp.2.markov_head.head.weight"]
    for n in names:
        source.verify(n)
    return source.get(names[0]).float(), source.get(names[1]).float()


def package_markov(root: Path, adapter_digest: str) -> tuple[Tensor, Tensor, dict[str, Any]]:
    from .reference import verify_package
    receipt = verify_package(root)
    if receipt["adapter_sha256"] != adapter_digest:
        raise ValueError("package source adapter differs from selected checkpoint")
    emb, head = native_markov(root, root/"NATIVE_MANIFEST.json")
    return emb, head, receipt


def score_dumps(manifest: Path, split: str, candidate: Path, baseline: Path, k2: Path,
                adapter: Path, native: Path, package_root: Path | None = None) -> dict[str, Any]:
    if split == "train":
        raise ValueError("evaluation must be held out")
    data = D2Blocks(manifest, split)
    state, adapter_digest = read_checkpoint(adapter)
    package = None
    if package_root is None:
        emb, head = markov(state, native)
    else:
        emb, head, package = package_markov(package_root, adapter_digest)
    be, bh = markov(None, native)
    by_seq = None
    if k2.is_file():
        from .replay import g1_drafts
        by_seq = g1_drafts(k2, data)
    rows: list[Score] = []
    # V1 dump names are keyed by absolute P and sequence index. ShardCaptureSet
    # enumerates selected split sequences in manifest order (same as D2Blocks).
    for seq_index, (path, seq, req) in enumerate(data.sequences):
        L, gen = seq[1:3]
        first_index = data.ends[seq_index-1] if seq_index else 0
        next_round = gen-1
        for offset, P in enumerate(range(gen-1, L-2)):
            if P != next_round:
                continue  # fixed K2 trajectory; candidate predictions never select the scored rounds
            block = data[first_index+offset]
            t = block["targets"]
            mask = t["position_mask"][0]
            if not bool(mask.all()):
                break
            key = f"r{P:06d}_b{data.sequence_b[seq_index]}.pt"
            a, b = [torch.load(d/key, map_location="cpu", weights_only=False) for d in (candidate, baseline)]
            if a.get("meta", {}).get("adapter_sha256") != adapter_digest:
                raise ValueError("candidate dump checkpoint identity differs from scored adapter")
            if a.get("meta", {}).get("package_identity") != (package["drafter_identity"] if package else None):
                raise ValueError("candidate dump package identity differs from scored arm")
            if b.get("meta", {}).get("adapter_sha256") is not None:
                raise ValueError("native baseline must be unadapted")
            k = None if by_seq is not None else torch.load(k2/key, map_location="cpu", weights_only=False)
            if k is not None and "k2" not in str(k.get("meta", {}).get("arm", "")).lower():
                raise ValueError("explicit K2 baseline provenance required")
            if k is not None and k.get("sample", {}).get("engine_drafts_placeholder", False):
                raise ValueError("placeholder K2 baseline")
            for dump in ([a, b, k] if k is not None else [a, b]):
                if dump["P"] != P or dump["anchor"] != int(block["inputs"]["anchor"][0]) or dump["sample"]["req"] != req:
                    raise ValueError("mismatched trajectory or placeholder K2 baseline")
            truth = [int(x) if bool(v) else None for x, v in zip(t["labels"][0], mask)]
            if by_seq is not None:
                k2_ids = by_seq[req][P]
            else:
                assert k is not None
                k2_ids = k["drafts"]
            cand_n, k2_n = match_length(a["drafts"], truth), match_length(k2_ids, truth)
            if cand_n is None or k2_n is None:
                raise ValueError("complete target block has an indeterminate draft score")
            if len(a["drafts"]) != 5 or len(k2_ids) != 5:
                raise ValueError("candidate or K2 does not contain five drafts")
            next_round = P+k2_n+1
            previous = t["previous"][0]
            aq = a["base_logits"] + emb[previous] @ head.T
            bq = b["base_logits"] + be[previous] @ bh.T
            kl_a = bucket_kl(aq[mask], t["top_ids"][0, mask], t["top_logp"][0, mask], t["tail_mass"][0, mask]).tolist()
            kl_b = bucket_kl(bq[mask], t["top_ids"][0, mask], t["top_logp"][0, mask], t["tail_mass"][0, mask]).tolist()
            rows.append({"request": req, "candidate": cand_n, "k2": k2_n,
                         "kl_candidate": kl_a+[None]*(5-len(kl_a)), "kl_baseline": kl_b+[None]*(5-len(kl_b))})
    return {"split": split, "dataset": data.fingerprint, "checkpoint": str(adapter), "checkpoint_sha256": adapter_digest,
            "metrics": clustered(rows), "proxy": "V1 committed-trajectory paired vs explicit K2",
            "k2_source": str(k2), "k2_jsonl_sha256": sha256(k2) if by_seq is not None else None,
            "target_manifest_sha256": sha256(manifest), "package_identity": package["drafter_identity"] if package else None}


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    sel = commands.add_parser("select-dev")
    sel.add_argument("summaries", nargs="+", type=Path)
    sel.add_argument("--out", required=True, type=Path)
    verdict = commands.add_parser("g2")
    for n in ("legal", "general", "acap", "selection", "out"):
        verdict.add_argument("--"+n, required=True, type=Path)
    score = commands.add_parser("score")
    for n in ("manifest", "candidate", "baseline", "k2", "adapter", "native", "out"):
        score.add_argument("--"+n, required=True, type=Path)
    score.add_argument("--split", choices=("dev", "heldout_legal", "heldout_general", "acap"), required=True)
    score.add_argument("--package-root", type=Path, help="score the merged native arm with its actual stored Markov head")
    score.add_argument("--selection", type=Path, help="immutable dev selection; required before test scoring")
    package = commands.add_parser("packaging-gate")
    package.add_argument("--gain-before", type=float, required=True)
    package.add_argument("--gain-after", type=float, required=True)
    for name in ("classification", "package-root", "out"):
        package.add_argument("--"+name, type=Path, required=True)
    a = ap.parse_args()
    if a.command == "select-dev":
        result = select_dev([json.loads(p.read_text()) for p in a.summaries])
    elif a.command == "g2":
        source = [json.loads(p.read_text()) for p in (a.legal, a.general, a.acap)]
        if [s["split"] for s in source] != ["heldout_legal", "heldout_general", "acap"]:
            raise ValueError("wrong held-out split labels")
        chosen = json.loads(a.selection.read_text())["checkpoint_sha256"]
        if any(s["checkpoint_sha256"] != chosen for s in source):
            raise ValueError("test checkpoint differs from dev selection")
        result = g2(*(s["metrics"] for s in source))
        result["checkpoint_sha256"] = chosen
    elif a.command == "packaging-gate":
        from .reference import verify_package
        receipt = verify_package(a.package_root)
        result = packaging_gate(a.gain_before, a.gain_after, json.loads(a.classification.read_text()), receipt["drafter_identity"])
    else:
        if a.split != "dev":
            if a.selection is None:
                raise ValueError("test scoring requires a prior dev-only selection")
            chosen = json.loads(a.selection.read_text())
            if chosen["split"] != "dev" or chosen["checkpoint_sha256"] != sha256(a.adapter):
                raise ValueError("test checkpoint differs from dev-only selection")
            if a.package_root:
                from .reference import verify_package
                receipt = verify_package(a.package_root)
                if receipt["adapter_sha256"] != chosen["checkpoint_sha256"]:
                    raise ValueError("package source differs from dev-selected checkpoint")
                ledger_name = f"D2-PACKAGE-ONCE-{receipt['drafter_identity']}-{a.split}.json"
            else:
                ledger_name = f"D2-TEST-ONCE-{a.split}.json"
            ledger = a.manifest.parent/ledger_name
            with ledger.open("x") as f:
                json.dump({"checkpoint_sha256": chosen["checkpoint_sha256"], "output": str(a.out),
                           "state": "test-scoring-started; do not retune or rerun"}, f)
        result = score_dumps(a.manifest, a.split, a.candidate, a.baseline, a.k2, a.adapter, a.native, a.package_root)
    # Exclusive creation enforces one reported test result and immutable selection.
    with a.out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
