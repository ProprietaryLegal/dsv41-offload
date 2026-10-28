"""D2 G1 plumbing and amendment-7 K2 committed-trajectory proxy.

Default binding: dev position-0 match rate >= half the hash-bound live
second-pass position-0 acceptance (amendment 7b gross-error tripwire). Historical acap/1.82 mode is explicit and cannot admit
pilot training. See G1-LIVE.md for coordinator-only execution instructions.
"""

from __future__ import annotations

import argparse
import importlib
import collections
import json
import os
import random
import sys
import shlex
import time
from pathlib import Path
from decimal import Decimal
from typing import Any, Callable, TypedDict

from .live_reference import build_reference, verify_reference, file_hash, Reference

import torch

from . import shards as S

K2_REF = 1.82
TOL = 0.1
BINDING_SPLIT = "dev"          # Amendment 7; other splits are information
# cards this lane may use (each under its own flock ~/dsv41-offload/locks/gpu-<8 hex>.lock, <= 14 min per hold);
# Historical G1 allocation; coordinator must recheck availability. Never c4e7a371 or 240876ad.
GPU_ALLOWED = ("GPU-bc62df2a-59cf-9828-7d43-558d1355bab7", "GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc")
ROOT = "/opt/dsv41/x86host-data/dspark"


# ------------------------------------------------------------------------------------------------------- plumbing
def cmd_plumbing(a: argparse.Namespace) -> None:
    from .dataset import cmd_verify
    cmd_verify(argparse.Namespace(out=a.shards))
    with open(os.path.join(a.shards, "MANIFEST.json")) as fh:
        man = json.load(fh)
    rep: dict[str, Any] = {}
    for split, sp in man["splits"].items():
        cs = S.ShardCaptureSet([os.path.join(a.shards, f["file"]) for f in sp["files"]], stride=a.stride)
        rep[split] = {"sequences": len(cs.rows), "samples": len(cs.samples), "plumbing_errors": len(cs.plumbing),
                      "first": cs.plumbing[:5]}
    rep["pass"] = all(v["plumbing_errors"] == 0 for v in rep.values() if isinstance(v, dict))
    print(json.dumps(rep, indent=1))
    if not rep["pass"]:
        raise SystemExit(1)


# ------------------------------------------------------------------------------------------------------ K2 drafts
class SeqSource:
    """What the K2 runner needs of one sequence: tokens, the first position with taps, a row reader, the blocks."""

    def __init__(self, seq_id: str, tokens: torch.Tensor, first_row: int, rows: Callable[[int, int], torch.Tensor], blocks: list[int],
                 anchors: dict[int, int] | None = None):
        self.seq_id, self.tokens, self.first_row, self.rows, self.blocks = seq_id, tokens, first_row, rows, blocks
        self.anchors = anchors or {}          # explicit anchor tokens (captures: the last anchor is past the rows)

    def anchor(self, P: int) -> int:
        a = self.anchors.get(P)
        return int(self.tokens[P + 1]) if a is None else a


def check_gpu() -> torch.device:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis not in GPU_ALLOWED:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be exactly one of {GPU_ALLOWED} (got {vis!r})")
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"{torch.cuda.device_count()} visible devices, want 1")
    uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    if uuid and uuid not in vis:
        raise SystemExit(f"visible device uuid {uuid} is not {vis}")
    return torch.device("cuda", 0)


def load_k2(root: str, dev: torch.device, slots: int) -> Any:
    dspark = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if dspark not in sys.path:
        sys.path.insert(0, dspark)
    Checkpoint = importlib.import_module("worker.checkpoint").Checkpoint
    DSparkDrafter = importlib.import_module("worker.model").DSparkDrafter
    ck = Checkpoint(root)
    m = DSparkDrafter(ck, dev, slots=slots)
    if slots * 5 > m.max_rows:
        raise SystemExit(f"{slots} slots x 5 drafts = {slots * 5} rows exceed the drafter's {m.max_rows}-row scratch")
    m.capture(batch_sizes=tuple(range(1, slots + 1)), ks=(5,), insert_rows=(8,), n_outs=(5,), margins=False)
    return m


def run_k2(m: Any, seqs: list[SeqSource], slots: int, deadline: float,
           on_done: Callable[[str, dict[int, list[int]]], None]) -> int:
    """Draft every block of every sequence, `slots` sequences in lock step; on_done(seq_id, {P: drafts}) per finished
    sequence. Returns the number of sequences finished before the deadline (unfinished ones are not reported)."""
    dev = m.dev
    W = m.s.window
    queue = list(seqs)
    active: dict[int, dict[str, Any]] = {}
    finished = 0
    while queue or active:
        if time.time() > deadline:
            break
        for sl in range(slots):
            if sl not in active and queue:
                s = queue.pop(0)
                m.reset_slots(torch.tensor([sl], device=dev))
                active[sl] = {"s": s, "i": 0, "next": s.first_row, "out": {}}
        if not active:
            break
        ins_slot, ins_pos, ins_taps = [], [], []
        order = sorted(active)
        for sl in order:
            st = active[sl]
            P = st["s"].blocks[st["i"]]
            lo = max(st["next"], P - W + 1, st["s"].first_row)
            if lo <= P:
                ins_slot.append(torch.full((P + 1 - lo,), sl, dtype=torch.int64))
                ins_pos.append(torch.arange(lo, P + 1, dtype=torch.int64))
                ins_taps.append(st["s"].rows(lo, P + 1))
                st["next"] = P + 1
        if ins_slot:
            taps = torch.cat(ins_taps).to(dev)
            m.insert(torch.cat(ins_slot).to(dev), torch.cat(ins_pos).to(dev), m.main_x(taps))
        Ps = [active[sl]["s"].blocks[active[sl]["i"]] for sl in order]
        anchors = [active[sl]["s"].anchor(P) for sl, P in zip(order, Ps)]
        ids, _ = m.draft_graphed(torch.tensor(order, device=dev), torch.tensor(anchors, device=dev),
                                 torch.tensor(Ps, device=dev), 5)
        ids = ids[: len(order)].cpu().tolist()
        for sl, P, d in zip(order, Ps, ids):
            st = active[sl]
            st["out"][P] = [int(x) for x in d]
            st["i"] += 1
            if st["i"] == len(st["s"].blocks):
                on_done(st["s"].seq_id, st["out"])
                finished += 1
                del active[sl]
    return finished


def cmd_k2(a: argparse.Namespace) -> None:
    dev = check_gpu()
    with open(os.path.join(a.shards, "MANIFEST.json")) as fh:
        man = json.load(fh)
    os.makedirs(a.work, mode=0o700, exist_ok=True)
    out_path = os.path.join(a.work, f"{a.split}.k2_drafts.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            done = {json.loads(line)["seq_id"] for line in fh}
    seqs = []
    for f in man["splits"][a.split]["files"]:
        sh = S.Shard(os.path.join(a.shards, f["file"]))
        for i in range(len(sh)):
            s = sh.seq(i)
            if s["seq_id"] in done:
                continue
            seqs.append(SeqSource(s["seq_id"], sh.tokens(i), s["r0"], _rows_of(sh, i), list(sh.blocks(i))))
    print(f"{len(seqs)} sequences to draft ({len(done)} done) in {a.split}", flush=True)
    if not seqs:
        return
    seqs.sort(key=lambda s: -len(s.blocks))
    t0 = time.time()
    deadline = t0 + a.deadline_s
    m = load_k2(a.root, dev, a.slots)
    print(f"K2 drafter loaded + graphs in {time.time() - t0:.0f} s", flush=True)
    fh = open(out_path, "a")
    os.chmod(out_path, 0o600)

    def on_done(sid: str, out: dict[int, list[int]]) -> None:
        fh.write(json.dumps({"seq_id": sid, "drafts": {str(p): d for p, d in out.items()}}) + "\n")
        fh.flush()
    n = run_k2(m, seqs, a.slots, deadline, on_done)
    print(f"{n} of {len(seqs)} sequences drafted in {time.time() - t0:.0f} s"
          + ("" if n == len(seqs) else " (deadline: run again in the next hold to continue)"), flush=True)


def _rows_of(sh: S.Shard, i: int) -> Callable[[int, int], torch.Tensor]:
    return lambda p0, p1: sh.taps(i, p0, p1)


# ----------------------------------------------------------------------------------------------------------- score
def match_len(drafts: list[int], truth: list[int | None]) -> int | None:
    n = 0
    for d, t in zip(drafts, truth):
        if t is None:
            return None
        if d != t:
            return n
        n += 1
    return n


def cluster_ci(vals_by_seq: dict[str, list[int]], iters: int = 2000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    keys = list(vals_by_seq)
    boots = []
    for _ in range(iters):
        pick = [vals_by_seq[keys[rng.randrange(len(keys))]] for _ in keys]
        tot = sum(len(p) for p in pick)
        boots.append(sum(sum(p) for p in pick) / max(tot, 1))
    boots.sort()
    return [round(boots[int(0.025 * iters)], 4), round(boots[int(0.975 * iters)], 4)]


def bucket(P: int) -> str:
    return "<2K" if P < 2000 else "2-8K" if P < 8000 else "8-16K" if P < 16000 else ">=16K"


class Sequence(TypedDict):
    tokens: list[int]
    gen_start: int
    L: int
    family: str


def score(seqs: dict[str, Sequence], drafts: dict[str, dict[int, list[int]]], split: str = BINDING_SPLIT, *, live_reference: Reference | None = None,
          mode: str = "live-reference") -> dict[str, Any]:
    """seqs: seq_id -> {tokens, gen_start, L, family}; drafts: seq_id -> {P: [5]}. G1(b) binds only on BINDING_SPLIT."""
    rounds_by_seq, allpos_by_seq = {}, {}
    by_fam, by_bucket = collections.defaultdict(list), collections.defaultdict(list)
    cond_n, cond_hit = [0] * 5, [0] * 5
    for sid, s in seqs.items():
        d = drafts.get(sid)
        if d is None:
            raise ValueError(f"{sid}: no drafts")
        tok, g, L = s["tokens"], s["gen_start"], s["L"]

        def realized(P: int) -> list[int | None]:
            return [int(tok[t]) if t < L else None for t in range(P + 2, P + 7)]
        allpos = [x for x in (match_len(d[P], realized(P)) for P in range(g - 1, L - 2)) if x is not None]
        rounds = []
        P = g - 1
        while P <= L - 3:
            ml = match_len(d[P], realized(P))
            if ml is None:
                break
            rounds.append(ml)
            for i in range(5):
                if ml >= i:
                    cond_n[i] += 1
                    cond_hit[i] += int(ml >= i + 1)
            by_fam[s["family"]].append(ml)
            by_bucket[bucket(P)].append(ml)
            P += ml + 1
        rounds_by_seq[sid] = rounds
        allpos_by_seq[sid] = allpos
    rr = [x for v in rounds_by_seq.values() for x in v]
    aa = [x for v in allpos_by_seq.values() for x in v]
    if not rr or not aa:
        raise ValueError("no scoreable committed-trajectory rounds/positions")
    mean = sum(rr) / len(rr)
    return {"sequences": len(seqs), "rounds": len(rr), "k2_accepted_per_round": round(mean, 4),
            "ci95_request_cluster": cluster_ci(rounds_by_seq),
            "k2_accepted_all_positions": round(sum(aa) / len(aa), 4), "positions": len(aa),
            "position0_match_rate": sum(x > 0 for x in rr) / len(rr),
            "per_position_match_rate": [sum(x > i for x in rr) / len(rr) for i in range(5)],
            "live_comparison": live_comparison(live_reference, split) if live_reference else None,
            "conditional_acceptance": [round(h / n, 4) if n else None for h, n in zip(cond_hit, cond_n)],
            "by_family": {k: {"rounds": len(v), "mean": round(sum(v) / len(v), 4)} for k, v in sorted(by_fam.items())},
            "by_context": {k: {"rounds": len(v), "mean": round(sum(v) / len(v), 4)} for k, v in sorted(by_bucket.items())},
            "per_sequence_mean_range": [round(min(sum(v) / len(v) for v in rounds_by_seq.values() if v), 4),
                                        round(max(sum(v) / len(v) for v in rounds_by_seq.values() if v), 4)],
            "g1": g1_verdict(sum(x > 0 for x in rr) / len(rr) if mode == "live-reference" else mean,
                             split, live_reference=live_reference, mode=mode)}


def g1_verdict(value: float, split: str, *, live_reference: Reference | None = None,
               mode: str = "live-reference") -> dict[str, object]:
    if mode == "historical-acap":
        reference, binding_split = K2_REF, "acap"
        valid = True
    elif mode == "live-reference":
        if live_reference is None:
            raise ValueError("amendment-7 live-reference receipt required")
        reference, binding_split = live_reference["position0_acceptance"], BINDING_SPLIT
        valid = live_reference["status"] == "VALID"
    else:
        raise ValueError("unknown G1 mode")
    binding = split == binding_split
    if mode == "live-reference":
        threshold = Decimal(str(reference)) * Decimal("0.5")
        passed = Decimal(str(value)) >= threshold
        rule: dict[str, object] = {"minimum_live_fraction": 0.5, "minimum_proxy_rate": float(threshold),
                                   "tripwire_passed": passed}
    else:
        passed = abs(Decimal(str(value))-Decimal(str(reference))) <= Decimal(str(TOL))
        rule = {"tolerance": TOL, "within_tolerance": passed}
    return {"mode": mode, "amendment": "7b" if mode == "live-reference" else None,
            "statistic": "position0_acceptance" if mode == "live-reference" else "accepted_per_round",
            "reference": reference, "split": split, "binding": binding, **rule,
            "pass": passed if binding and valid else None,
            "status": ("PASS" if passed else "FAIL") if binding and valid else "INCONCLUSIVE" if not valid else "INFORMATION",
            "live_reference_sha256": live_reference["sha256"] if live_reference else None}


def live_comparison(live: Reference, split: str) -> dict[str, object]:
    return {"position0_acceptance": live["position0_acceptance"],
            "accepted_per_round": live["accepted_per_round"],
            "per_position_acceptance": live["per_position_acceptance"],
            "foreign_share": live["foreign_share"], "status": live["status"],
            "biases": {
                "depth_0_censoring": "Live conf policy can verify zero drafts; proxy position 0 is uncensored.",
                "window": "Proxy fixed window 5 and live adaptive windows select different round starts.",
                "split_coverage": f"Proxy covers {split}; live counters cover the whole second pass.",
                "foreign_share": "Live counters include foreign traffic; VALID requires foreign share <= 2%.",
            }}


def coordinator_procedure(shards: Path, work: Path, live: Path, gpu: str = GPU_ALLOWED[0]) -> dict[str, object]:
    """Render only. No filesystem reads, CUDA, subprocesses or host access."""
    if gpu not in GPU_ALLOWED:
        raise ValueError("G1 GPU must be in GPU_ALLOWED")
    lock = f"$HOME/dsv41-offload/locks/gpu-{gpu[4:12]}.lock"
    python = "$HOME/dsv41-offload/o2/venv/bin/python"
    setup = "set -euo pipefail\numask 077\nsource ~/.config/ndm/paths.sh\nexport PYTHONPATH=\"$PWD/offload/dspark:$PWD\"\n"
    holds: dict[str, str] = {}
    scores: dict[str, str] = {}
    for split in (BINDING_SPLIT, "heldout_legal", "heldout_general"):
        args = shlex.join(["-m", "d2.g1", "k2", "--shards", str(shards), "--split", split,
                           "--work", str(work), "--deadline-s", "780"])
        holds[split] = setup + (f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} flock -n \"{lock}\" "
                               f"timeout --signal=TERM --kill-after=20 820 \"{python}\" {args}")
        scores[split] = setup + "CUDA_VISIBLE_DEVICES='' \""+python+'" '+shlex.join([
            "-m", "d2.g1", "score", "--shards", str(shards), "--split", split,
            "--work", str(work), "--live-reference", str(live), "--out", str(work/f"{split}.json")])
    return {"dry": True, "gpu": gpu, "lock": lock, "deadline_s": 780, "hold_limit_s": 840, "term_timeout_s": 820,
            "holds": holds, "scores": scores, "binding_split": BINDING_SPLIT,
            "outputs": [str(work/f"{s}.{ext}") for s in holds for ext in ("k2_drafts.jsonl", "json")]}


def cmd_procedure(a: argparse.Namespace) -> None:
    print(json.dumps(coordinator_procedure(a.shards, a.work, a.live_reference, a.gpu), indent=2))


def cmd_live_reference(a: argparse.Namespace) -> None:
    receipt = build_reference(a.snapshots, a.rates, a.paused_at, a.rollback_at)
    with open(a.out, "x") as fh:
        json.dump(receipt, fh, indent=2, allow_nan=False)
    print(json.dumps(receipt, indent=2))


def cmd_score(a: argparse.Namespace) -> None:
    with open(os.path.join(a.shards, "MANIFEST.json")) as fh:
        man = json.load(fh)
    seqs: dict[str, Sequence] = {}
    for f in man["splits"][a.split]["files"]:
        sh = S.Shard(os.path.join(a.shards, f["file"]))
        for i in range(len(sh)):
            s = sh.seq(i)
            seqs[s["seq_id"]] = {"tokens": sh.tokens(i).tolist(), "gen_start": s["gen_start"], "L": s["L"],
                                 "family": s["family"]}
    drafts = {}
    with open(os.path.join(a.work, f"{a.split}.k2_drafts.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            drafts[r["seq_id"]] = {int(p): d for p, d in r["drafts"].items()}
    live = verify_reference(a.live_reference) if a.live_reference else None
    rep = {"split": a.split, **score(seqs, drafts, a.split, live_reference=live, mode=a.mode),
           "manifest_sha256": file_hash(Path(a.shards)/"MANIFEST.json"),
           "k2_sha256": file_hash(Path(a.work)/f"{a.split}.k2_drafts.jsonl"),
           "live_reference_path": str(a.live_reference.resolve()) if a.live_reference else None}
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))


# ------------------------------------------------------------------------------------------- pre-check on captures
def cmd_k2_captures(a: argparse.Namespace) -> None:
    """Worker drafts vs the engine's drafts on window-1 blocks whose 128-row window is all decode taps (no prefill rows
    are needed): validates the K2 runner on real taps before any D2 data exists."""
    dev = check_gpu()
    if a.v1 not in sys.path:
        sys.path.insert(0, a.v1)
    CaptureSet = importlib.import_module("verify.captures").CaptureSet
    cs = CaptureSet(a.captures)
    if cs.plumbing:
        raise SystemExit(f"capture plumbing errors: {cs.plumbing[:3]}")
    reqs = sorted(r for r in cs.rows if "warmup" not in r)
    seqs, eng = [], {}
    for r in reqs:
        rows = cs.rows[r]
        g = min(s.P for s in cs.samples if s.req == r) + 1
        L = max(rows) + 1
        # every window row must carry decode taps (rounds of > 64 tokens, e.g. a decode batched with another
        # request's prefill chunk, saved no taps)
        smp = [s for s in cs.samples if s.req == r and s.P - 127 >= g
               and not (st := cs.window_status(s))["missing"] and not st["no_taps"]]
        if not smp:
            continue
        for s in smp:
            eng[(r, s.P)] = (s.engine_drafts, s.realized)
        tok = torch.tensor([rows[p].tok for p in range(L)])
        seqs.append(SeqSource(r, tok, g, (lambda rows: lambda p0, p1: torch.stack([rows[p].taps for p in
                                                                                   range(p0, p1)]))(rows),
                              sorted({s.P for s in smp}), anchors={s.P: s.anchor for s in smp}))
    t0 = time.time()
    m = load_k2(a.root, dev, a.slots)
    got = {}
    n = run_k2(m, seqs, a.slots, t0 + a.deadline_s, lambda sid, out: got.update({(sid, p): d for p, d in out.items()}))
    same, pos, lw, le = 0, [0] * 5, [], []
    for k, (ed, real) in eng.items():
        w = got[k]
        same += int(w == ed)
        for i in range(5):
            pos[i] += int(w[i] == ed[i])
        mw, me = match_len(w, real), match_len(ed, real)
        if mw is not None and me is not None:
            lw.append(mw)
            le.append(me)
    nb = len(eng)
    rep = {"requests": n, "blocks": nb, "worker_vs_engine_block_exact": round(same / nb, 4),
           "worker_vs_engine_pos_equal": [round(x / nb, 4) for x in pos],
           "proxy_worker": round(sum(lw) / len(lw), 4), "proxy_engine": round(sum(le) / len(le), 4),
           "scored": len(lw), "seconds": round(time.time() - t0, 1)}
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plumbing")
    p.add_argument("--shards", required=True)
    p.add_argument("--stride", type=int, default=1)
    k = sp.add_parser("k2")
    k.add_argument("--shards", required=True)
    k.add_argument("--split", default=BINDING_SPLIT)
    k.add_argument("--work", required=True)
    k.add_argument("--root", default=ROOT)
    k.add_argument("--slots", type=int, default=6, help="sequences drafted together (x 5 rows <= the 32-row scratch)")
    k.add_argument("--deadline-s", type=float, default=780.0)
    s = sp.add_parser("score")
    s.add_argument("--shards", required=True)
    s.add_argument("--split", default=BINDING_SPLIT, help=f"G1(b) binds on {BINDING_SPLIT}; others are information")
    s.add_argument("--work", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--mode", choices=("live-reference", "historical-acap"), default="live-reference")
    s.add_argument("--live-reference", type=Path)
    r = sp.add_parser("live-reference", help="CPU only: bind paused, pre-rollback snapshots")
    r.add_argument("--snapshots", nargs="+", required=True, type=Path)
    r.add_argument("--rates", nargs="+", required=True, type=Path)
    r.add_argument("--paused-at", required=True)
    r.add_argument("--rollback-at", required=True)
    r.add_argument("--out", required=True)
    c = sp.add_parser("k2-captures")
    c.add_argument("--captures", required=True)
    c.add_argument("--v1", required=True)
    c.add_argument("--root", default=ROOT)
    c.add_argument("--slots", type=int, default=4)
    c.add_argument("--deadline-s", type=float, default=600.0)
    c.add_argument("--out", required=True)
    p = sp.add_parser("procedure", help="dry-only coordinator command renderer; no host access")
    p.add_argument("--shards", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--live-reference", required=True, type=Path)
    p.add_argument("--gpu", choices=GPU_ALLOWED, default=GPU_ALLOWED[0])
    a = ap.parse_args()
    {"plumbing": cmd_plumbing, "k2": cmd_k2, "score": cmd_score, "live-reference": cmd_live_reference, "procedure": cmd_procedure, "k2-captures": cmd_k2_captures}[a.cmd](a)


if __name__ == "__main__":
    main()
