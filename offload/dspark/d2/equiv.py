"""D2 equivalence check: prefill taps (replay window, configs/A-replay.env) vs the decode taps captured in window 1
(A-capture-20261022), on the 4 A-capture requests. Gate: on the blocks whose two arms differ (>= 1 decode row in the
window), the V1 FP32 reference drafts the same 5 tokens from both on >= 99 % of them; below that, the BF16-noise rule
calibrated to a same-run noise control (coordinator amendment 2026-10-23 14:59Z, cmd_report); otherwise train on
decode-captured taps (d2/FALLBACK.md).

  acap-export (Controller: captures + D1 Phase-0 request files)
      python3 -m d2.equiv acap-export --v1 /opt/dsv41/repos/.worktrees/dsv41-v1-verify/offload/dspark \
          --captures /opt/dsv41/data/v1/captures/A-capture-20261022 --d1-exact /opt/dsv41/data/v1/d1-exact \
          --out /opt/dsv41/data/d2/acap
  taps    (x86host, CPU, minutes): per-row relative error of each tap layer, prefill vs decode, where window 1 has decode
          taps; token ids of the replayed rows checked against the captures
      python3 -m d2.equiv taps --v1 <v1 dspark> --captures <x86host captures> --acap <acap dir> --replay <replay dir> --out T.json
  drafts  (x86host, CPU, ~25-40 s per block per process for the three arms with V1's ref mode; --shard i/n to split)
      python3 -m d2.equiv drafts --v1 <v1 dspark> --captures ... --acap ... --replay ... --ckpt <native-1017> \
          --official <native-1017/inference> --work <dir> [--shard 0/3] [--threads 8] --every 1
          (amendment 2: every block; ~30 s per block per process at 8 threads on Controller for the three arms, so the
          542 blocks take about 90 min in 3 shards)
  report  python3 -m d2.equiv report --work <dir> --taps T.json --out <gate.json> (aggregates only; may be committed)

Three arms per captured block (V1 CaptureSet samples, the window rows max(0, P-127)..P):
  D  decode taps where window 1 captured them (decode rounds), prefill taps elsewhere (V1 used the engine main_x there);
  R  prefill taps for every row;
  N  the noise control: D with Gaussian relative noise sigma = 2^-8 on every decode-captured row (x * (1 + sigma * z),
     cast back to BF16; z from a generator seeded by 20261023, the request and the position, so the control is the
     same whatever the shard, order or --every).
R and N differ from D only where window 1 has decode taps, which is the comparison asked for. Drafts are V1 NativeRef
(ref mode: FP32 activations, FP64 accumulation) greedy own-prefix drafts through V1 run_ref.prepare_window.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
from fractions import Fraction
import math
import os
import re
import sys
import time

import numpy as np
import torch

ROW = 3 * 5120
GATE = 0.99
GATE_EXACT = Fraction(99, 100)   # S16 F9-c: exact comparison; rounding is for display only
# noise control (coordinator amendment 2026-10-23 14:59Z, documents/plans/2026-10-23-d2-plan.md)
CONTROL_SIGMA = 2.0 ** -8
CONTROL_SEED = 20261023


def v1_import(v1: str):
    if v1 not in sys.path:
        sys.path.insert(0, v1)
    from verify import captures, run_ref
    return captures, run_ref


def acap_requests(cs) -> list[str]:
    """The 4 live A-capture requests in D1's order (exact_ceiling build: sorted, warmups excluded)."""
    return sorted(r for r in cs.rows if "warmup" not in r)


def cmd_acap_export(a) -> None:
    C, _ = v1_import(a.v1)
    cs = C.CaptureSet(a.captures)
    if cs.plumbing:
        raise RuntimeError(f"capture plumbing errors: {cs.plumbing[:3]}")
    from .shards import tokens_sha256
    os.makedirs(a.out, mode=0o700, exist_ok=True)
    os.chmod(a.out, 0o700)
    for i, r in enumerate(acap_requests(cs)):
        rows = cs.rows[r]
        n = max(rows) + 1
        if sorted(rows) != list(range(n)):
            raise RuntimeError(f"acap{i}: committed positions are not contiguous 0..{n - 1}")
        toks = [rows[p].tok for p in range(n)]
        with open(os.path.join(a.d1_exact, f"req{i}.request.json")) as fh:
            d1 = json.load(fh)["prompt"]
        if d1 != toks:
            raise RuntimeError(f"acap{i}: tokens differ from D1 Phase-0 req{i}")
        g = min(s.P for s in cs.samples if s.req == r) + 1
        rec = {"schema": "d2.acap.v1", "seq_id": f"acap{i}", "capture_req": r, "tokens": toks, "gen_start": g,
               "prompt_sha256": tokens_sha256(toks[:g]), "tokens_sha256": tokens_sha256(toks),
               "decode_tap_rows": sum(1 for p in range(n) if rows[p].taps is not None)}
        path = os.path.join(a.out, f"acap{i}.json")
        with open(path, "w") as fh:
            json.dump(rec, fh)
        os.chmod(path, 0o600)
        print(f"acap{i}: L={n} gen_start={g} decode-tap rows={rec['decode_tap_rows']} -> {path}")


def load_prefill(replay: str, seq: str, L: int, toks: list[int]) -> torch.Tensor:
    """Replay overlay rows 0..L-1 of an acap sequence (bf16 [L, 15360]); ids.bin must equal the captured tokens."""
    d = os.path.join(replay, "taps", seq)
    with open(os.path.join(d, "DONE.json")) as fh:
        done = json.load(fh)
    if (done["from"], done["end"], done["rows_missing"]) != (0, L, 0):
        raise RuntimeError(f"{seq}: DONE.json {done} is not a full 0..{L} replay")
    if done.get("zero_rows") != 0:
        raise RuntimeError(f"{seq}: DONE.json zero_rows {done.get('zero_rows')}: zero taps rows (trimmed eager prefill?)")
    ids = np.fromfile(os.path.join(d, "ids.bin"), dtype="<i4")
    if ids.tolist() != toks:
        raise RuntimeError(f"{seq}: replayed row ids differ from the captured tokens")
    raw = np.fromfile(os.path.join(d, "taps.bin"), dtype="<i2")
    if raw.size != L * ROW:
        raise RuntimeError(f"{seq}: taps.bin holds {raw.size} values, want {L * ROW}")
    t = torch.from_numpy(raw).view(torch.bfloat16).view(L, ROW)
    zero = (t.view(L, 3, ROW // 3) == 0).all(-1).any(-1)
    if bool(zero.any()):
        raise RuntimeError(f"{seq}: {int(zero.sum())} taps rows with a layer all zero (first {int(zero.nonzero()[0])})")
    return t


GATE_SCHEMA = "d2.equiv.gate.v1"


def replay_binding(replay: str, n: int) -> dict:
    """S16 F10-a: the acap sequences' taps/ids sha256 as their DONE.json in <replay>/taps records them (replay copy
    keeps those fields). Every draft record carries this, the report writes it into gate.json, and dataset build checks
    it against the replay tree it builds from."""
    out = {}
    for i in range(n):
        with open(os.path.join(replay, "taps", f"acap{i}", "DONE.json")) as fh:
            done = json.load(fh)
        h = {k: done.get(k) for k in ("sha256_taps", "sha256_ids")}
        if not all(isinstance(v, str) and len(v) == 64 for v in h.values()):
            raise SystemExit(f"acap{i}: DONE.json has no sha256_taps / sha256_ids; the gate could not be bound to it")
        out[f"acap{i}"] = h
    return out


def load_all(a):
    C, R = v1_import(a.v1)
    cs = C.CaptureSet(a.captures)
    if cs.plumbing:
        raise RuntimeError(f"capture plumbing errors: {cs.plumbing[:3]}")
    reqs = acap_requests(cs)
    pre = {}
    for i, r in enumerate(reqs):
        with open(os.path.join(a.acap, f"acap{i}.json")) as fh:
            ex = json.load(fh)
        if ex["capture_req"] != r:
            raise RuntimeError(f"acap{i}: export names {ex['capture_req']}, captures order gives {r}")
        n = max(cs.rows[r]) + 1
        pre[r] = load_prefill(a.replay, f"acap{i}", n, ex["tokens"])
    return C, R, cs, reqs, pre


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def q(xs: list[float], p: float) -> float | None:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def cmd_taps(a) -> None:
    _, _, cs, reqs, pre = load_all(a)
    per = {L: [] for L in (37, 38, 39)}
    per_req = {}
    exact = 0
    n = 0
    for i, r in enumerate(reqs):
        rr = {L: [] for L in (37, 38, 39)}
        for p, row in cs.rows[r].items():
            if row.taps is None:
                continue
            d = row.taps.view(3, 5120)
            f = pre[r][p].view(3, 5120)
            n += 1
            exact += int(torch.equal(d, f))
            for k, L in enumerate((37, 38, 39)):
                e = rel(f[k], d[k])
                per[L].append(e)
                rr[L].append(e)
        per_req[f"acap{i}"] = {str(L): {"rows": len(v), "median": q(v, 0.5), "p99": q(v, 0.99), "max": max(v)}
                               for L, v in rr.items() if v}
    rep = {"rows_compared": n, "rows_bit_identical": exact,
           "relerr_prefill_vs_decode": {str(L): {"median": q(v, 0.5), "p90": q(v, 0.9), "p99": q(v, 0.99),
                                                 "max": max(v) if v else None} for L, v in per.items()},
           "per_request": per_req}
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))


@dataclasses.dataclass
class ArmRow:
    tok: int
    round: int
    mx: torch.Tensor | None
    taps: torch.Tensor | None


class ArmSet:
    """V1 CaptureSet's window interface over patched rows (one arm)."""

    def __init__(self, cs, rows: dict):
        self.cs, self.rows = cs, rows

    def window(self, s) -> list[int]:
        return self.cs.window(s)

    def window_status(self, s) -> dict:
        rows = self.rows[s.req]
        pos = self.window(s)
        missing = [p for p in pos if p not in rows or rows[p].round > s.round]
        return {"n": len(pos), "missing": missing, "no_taps": [p for p in pos if p in rows and rows[p].taps is None]}


def control_noise(taps: torch.Tensor, req_index: int, pos: int, sigma: float = CONTROL_SIGMA,
                  seed: int = CONTROL_SEED) -> torch.Tensor:
    """The control arm's row: taps * (1 + sigma * z), z ~ N(0, 1) elementwise in FP32, cast back to BF16. The generator
    is seeded per (seed, request, position), so a row's noise does not depend on which blocks are drafted or in what
    order."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + req_index * 10_000_019 + pos)
    z = torch.randn(taps.numel(), generator=g, dtype=torch.float32).view(taps.shape)
    return (taps.float() * (1 + sigma * z)).to(torch.bfloat16)


def arm_rows(cs, pre: dict, arm: str, reqs: list[str] | None = None) -> dict:
    """D: decode taps where captured, prefill elsewhere; R: prefill everywhere; N: D plus control noise on the
    decode-captured rows (reqs gives each request's index for the control seed)."""
    if arm not in ("D", "R", "N"):
        raise ValueError(f"unknown arm {arm}")
    if arm == "N" and reqs is None:
        raise ValueError("the control arm needs the request order (reqs) for its seed")
    out = {}
    for r, rows in cs.rows.items():
        if r not in pre:
            continue
        out[r] = {}
        for p, row in rows.items():
            if row.taps is None or arm == "R":
                t = pre[r][p].clone()
            elif arm == "D":
                t = row.taps
            else:
                t = control_noise(row.taps, reqs.index(r), p)
            out[r][p] = ArmRow(row.tok, row.round, row.mx, t)   # mx: only V1's engine-vs-native diagnostic reads it
    return out


def cmd_drafts(a) -> None:
    C, R, cs, reqs, pre = load_all(a)
    torch.set_num_threads(a.threads)
    from verify.native_ref import NativeRef
    samples = sorted((s for s in cs.samples if s.req in pre), key=lambda s: (s.req, s.P, s.round))[:: a.every]
    k, n = (int(x) for x in a.shard.split("/"))
    samples = [s for j, s in enumerate(samples) if j % n == k]
    os.makedirs(a.work, mode=0o700, exist_ok=True)
    os.chmod(a.work, 0o700)
    out_path = os.path.join(a.work, f"drafts-{k}of{n}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            done = {json.loads(line)["key"] for line in fh}
    todo = [s for s in samples if s.key not in done]
    print(f"{len(samples)} blocks in shard {k}/{n}, {len(todo)} to go", flush=True)
    if not todo:
        return
    max_pos = max(s.P for s in todo) + 16
    ref = NativeRef(a.ckpt, a.official, mode="ref", max_seq_len=max_pos + 64)
    from verify.analyze import final_logits, markov_tables
    m_emb, m_head = markov_tables(a.ckpt)
    arms = {name: ArmSet(cs, arm_rows(cs, pre, name, reqs)) for name in ("D", "R", "N")}
    binding = replay_binding(a.replay, len(reqs))
    caches = {name: R.WindowCaches() for name in arms}
    fh = open(out_path, "a")
    os.chmod(out_path, 0o600)
    t0 = time.time()
    for j, s in enumerate(todo):
        rec = {"key": s.key, "req": f"acap{reqs.index(s.req)}", "P": s.P, "realized": s.realized,
               "engine": s.engine_drafts, "control": {"sigma": CONTROL_SIGMA, "seed": CONTROL_SEED},
               "replay": binding}
        base = {}
        for name, aset in arms.items():                     # D first: its own prefix is the common Markov input
            win, srcs, mx_P, _ = R.prepare_window(ref, aset, s, caches[name])
            res = ref.draft(s.P, s.anchor, mx_P, keep_full_logits=True)
            top = res["variants"]["own"]["top8_vals"]
            rec[name] = {"drafts": res["drafts"], "margin": [round(float(top[i, 0] - top[i, 1]), 5)
                                                              for i in range(top.shape[0])]}
            base[name] = res["base_logits"]
            if name == "D":
                prev_D = res["variants"]["own"]["prev"]
        # per-position KL(D || R) and KL(D || N) of the final draft distributions, all under D's Markov prefix (the
        # calibrated BF16-noise rule compares their medians)
        lp = {n: torch.log_softmax(final_logits(base[n], prev_D, m_emb, m_head).double(), -1) for n in base}
        for other in ("R", "N"):
            rec[f"kl_D_{other}"] = finite_kl([float(v) for v in (lp["D"].exp() * (lp["D"] - lp[other])).sum(-1)],
                                             f"{s.key} kl_D_{other}")
        rec["decode_rows_in_window"] = sum(1 for p in cs.window(s) if cs.rows[s.req][p].taps is not None)
        rec["main_x_P_rel"] = rel(ref.main_x(arms["R"].rows[s.req][s.P].taps),
                                  ref.main_x(arms["D"].rows[s.req][s.P].taps))
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        if j % 10 == 0:
            print(f"{j + 1}/{len(todo)} {time.time() - t0:.0f} s", flush=True)
    print(f"done {len(todo)} blocks in {time.time() - t0:.0f} s", flush=True)


def finite_kl(kl: list[float], where: str) -> list[float]:
    """S16 F9-a: a non-finite KL is not a measurement. Refused where it is computed (cmd_drafts writes no record) and
    where it is read (cmd_report)."""
    if len(kl) != 5 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in kl):
        raise SystemExit(f"{where}: KL {kl} is not 5 finite values; refused (a NaN or inf KL is not a measurement)")
    return kl


def finite_max(*xs: float) -> float:
    """max() that refuses a non-finite argument (Python's max skips a NaN that is not first; S16 F9-a)."""
    if not all(math.isfinite(x) for x in xs):
        raise SystemExit(f"non-finite value in {xs}; refused")
    return max(xs)


def match_len(drafts: list[int], truth: list[int | None]) -> int | None:
    n = 0
    for d, t in zip(drafts, truth):
        if t is None:
            return None
        if d != t:
            return n
        n += 1
    return n


BINS = ((0, 0), (1, 15), (16, 63), (64, 127), (128, 128))
# calibrated BF16-noise rule (coordinator amendment 2026-10-23 14:59Z; amendment 2, 16:30Z: pos1 is a gross-error
# backstop; both set before any replay data)
POS1_SLACK = 0.05       # pos1(D, R) >= control pos1 - POS1_SLACK (absolute), compared as exact fractions
POS1_SLACK_EXACT = Fraction(5, 100)
KL_FLOOR = 1e-3         # per position: median KL(D || R) <= max(KL_FLOOR, KL_CONTROL_FACTOR x control median)
KL_CONTROL_FACTOR = 2.0
KL_CEILING = 0.05       # per position: median KL(D || R) <= KL_CEILING whatever the control (hard ceiling)


def bf16_noise_rule(differ: list[dict]) -> dict:
    """The calibrated rule over the inputs-differ blocks: met when pos1 agreement D vs R >= control pos1 - 0.05
    (amendment 2: a gross-error backstop; first-token flips at near-ties measure the drafter, not the taps), and at
    every position the median KL(D || R) <= max(1e-3, 2 x the control's median KL(D || N)) and <= 0.05 nats (the KL
    checks are the gate). Not met (with the reason) when the KL or the control was not recorded on every block."""
    n = len(differ)
    pos1_n = sum(1 for r in differ if r["D"]["drafts"][0] == r["R"]["drafts"][0])
    pos1 = pos1_n / n
    has_kl = all("kl_D_R" in r for r in differ)
    has_ctl = all("N" in r and "kl_D_N" in r for r in differ)
    out = {"pos1_equal_frac": round(pos1, 4), "kl_recorded": has_kl, "control_recorded": has_ctl,
           "kl_floor": KL_FLOOR, "kl_control_factor": KL_CONTROL_FACTOR, "kl_ceiling": KL_CEILING,
           "kl_median_per_position": [q([r["kl_D_R"][i] for r in differ], 0.5) for i in range(5)] if has_kl else None,
           "pos1_slack": POS1_SLACK, "control": None, "pos1_threshold": None, "kl_bound_per_position": None,
           "checks": None, "met": False}
    if not (has_kl and has_ctl):
        out["reason"] = "per-position KL(D || R) not recorded" if not has_kl else "noise control not recorded"
        return out
    ctl_cfg = {json.dumps(r.get("control"), sort_keys=True) for r in differ}
    if len(ctl_cfg) != 1:
        raise SystemExit(f"blocks were drafted with different noise controls: {sorted(ctl_cfg)}")
    c_pos1_n = sum(1 for r in differ if r["D"]["drafts"][0] == r["N"]["drafts"][0])
    c_pos1 = c_pos1_n / n
    c_kl = [q([r["kl_D_N"][i] for r in differ], 0.5) for i in range(5)]
    thr = Fraction(c_pos1_n, n) - POS1_SLACK_EXACT                     # exact (S16 F9-c); amendment 2
    bound = [finite_max(KL_FLOOR, KL_CONTROL_FACTOR * c) for c in c_kl]
    kl = out["kl_median_per_position"]
    checks = {"pos1": Fraction(pos1_n, n) >= thr,
              "kl_within_control_bound": all(k <= b for k, b in zip(kl, bound)),
              "kl_within_ceiling": all(k <= KL_CEILING for k in kl)}
    out.update({"control": {**json.loads(ctl_cfg.pop()), "blocks": n, "pos1_equal_frac": round(c_pos1, 4),
                            "pos_equal_frac": [round(sum(1 for r in differ if r["D"]["drafts"][i] == r["N"]["drafts"][i])
                                                     / n, 4) for i in range(5)],
                            "kl_median_per_position": c_kl},
                "pos1_threshold": round(float(thr), 4), "kl_bound_per_position": bound, "checks": checks,
                "met": all(checks.values())})
    return out


SHARD_FILE = re.compile(r"drafts-(\d+)of(\d+)\.jsonl")


def shard_set(files: list[str]) -> dict:
    """S16 F9-b: the report gates only on a complete shard set: every drafts-<k>of<n>.jsonl for k = 0..n-1, one n, no
    k twice (e.g. drafts-0of3 and drafts-00of3), no other drafts-*.jsonl name."""
    parsed = []
    for f in files:
        m = SHARD_FILE.fullmatch(f)
        if m is None:
            raise SystemExit(f"{f}: not a drafts-<k>of<n>.jsonl shard file; refused")
        parsed.append((int(m.group(1)), int(m.group(2))))
    ns = {n for _, n in parsed}
    if len(ns) != 1:
        raise SystemExit(f"shard files from different splits (n = {sorted(ns)}): {files}; refused")
    n = ns.pop()
    ks = sorted(k for k, _ in parsed)
    if len(ks) != len(set(ks)):
        raise SystemExit(f"a shard index appears twice: {files}; refused")
    if ks != list(range(n)):
        missing = sorted(set(range(n)) - set(ks))
        raise SystemExit(f"incomplete shard set: {len(ks)} of {n} shards present, missing k = {missing} "
                         f"(or k outside 0..{n - 1}: {sorted(set(ks) - set(range(n)))}); refused")
    return {"n": n, "found": len(ks), "files": files}


def cmd_report(a) -> None:
    """Gate over the blocks whose two arms get different inputs (>= 1 decode row in the window, S14 F9); blocks with
    identical inputs agree by construction and are reported apart. Below 99 % equal blocks, the BF16-noise rule
    calibrated to the same-run noise control (bf16_noise_rule; coordinator amendment 2026-10-23 14:59Z) decides
    pass_bf16_noise; otherwise training waits for the decode-capture fallback (d2/FALLBACK.md option A). Mean accepted
    length of D, R, N and the engine against the realized tokens is recorded, not gating."""
    recs = []
    files = [f for f in sorted(os.listdir(a.work)) if f.startswith("drafts-") and f.endswith(".jsonl")]
    for f in files:
        with open(os.path.join(a.work, f)) as fh:
            recs += [json.loads(line) for line in fh]
    if not recs:
        raise SystemExit(f"no draft records under {a.work}")
    if len({r["key"] for r in recs}) != len(recs):
        raise SystemExit("duplicate block keys across shards")
    shards = shard_set(files)
    bindings = {json.dumps(r.get("replay"), sort_keys=True) for r in recs}
    if len(bindings) != 1:
        raise SystemExit(f"draft records from {len(bindings)} different replays (or some without a replay binding); "
                         "refused")
    binding = json.loads(bindings.pop())
    for r in recs:                                    # S16 F9-a: NaN / inf KL never reaches a rule
        for k in ("kl_D_R", "kl_D_N"):
            if k in r:
                finite_kl(r[k], f"{r['key']} {k}")
    differ = [r for r in recs if r["decode_rows_in_window"] > 0]
    same_in = [r for r in recs if r["decode_rows_in_window"] == 0]
    if not differ:
        raise SystemExit("no block has a decode row in its window: nothing was compared")

    def eq(r):
        return r["D"]["drafts"] == r["R"]["drafts"]

    def frac(rs):
        return round(sum(1 for r in rs if eq(r)) / len(rs), 4) if rs else None
    n = len(differ)
    pos = [sum(1 for r in differ if r["D"]["drafts"][i] == r["R"]["drafts"][i]) / n for i in range(5)]
    first_mm = collections.Counter()
    mm_margins = []
    for r in differ:
        dd, rr = r["D"]["drafts"], r["R"]["drafts"]
        i0 = next((i for i in range(5) if dd[i] != rr[i]), None)
        if i0 is not None:
            first_mm[i0] += 1
            mm_margins.append(min(r["D"]["margin"][i0], r["R"]["margin"][i0]))
    acc = {}
    for arm in ("D", "R", "N", "engine"):
        if arm == "N" and not all("N" in r for r in differ):
            continue
        ls = [match_len(r[arm]["drafts"] if arm != "engine" else r["engine"], r["realized"]) for r in differ]
        ls = [x for x in ls if x is not None]
        acc[arm] = {"n": len(ls), "mean_accepted": round(sum(ls) / len(ls), 4) if ls else None}
    f_diff = frac(differ)
    rule2 = bf16_noise_rule(differ)
    equal_n = sum(1 for r in differ if eq(r))
    gate_met = Fraction(equal_n, n) >= GATE_EXACT                                # unrounded (S16 F9-c)
    verdict = "pass" if gate_met else ("pass_bf16_noise" if rule2["met"] else "fail")
    rep = {"schema": GATE_SCHEMA, "replay_binding": binding,          # S16 F10-a: dataset build checks both
           "blocks": len(recs), "blocks_inputs_differ": n, "blocks_inputs_identical": len(same_in),
           "block_equal_frac_inputs_identical": frac(same_in),
           "decode_rows_in_window_bins": {f"{lo}-{hi}": {"blocks": len(b), "block_equal_frac": frac(b)}
                                          for lo, hi in BINS
                                          for b in [[r for r in recs if lo <= r["decode_rows_in_window"] <= hi]]},
           "block_equal_frac": f_diff,                 # = the gate population (inputs differ), as below
           "block_equal_frac_with_decode_rows": f_diff, "block_equal_frac_all_blocks": frac(recs),
           "block_equal_frac_all_decode_windows": frac([r for r in recs if r["decode_rows_in_window"] >= 128]),
           "pos_equal_frac": [round(x, 4) for x in pos], "first_mismatch_position": dict(sorted(first_mm.items())),
           "mismatch_min_margin_quantiles": {p: q(mm_margins, p) for p in (0.5, 0.9, 1.0)} if mm_margins else None,
           "main_x_P_rel_prefill_vs_decode": {"median": q([r["main_x_P_rel"] for r in differ], 0.5),
                                              "p99": q([r["main_x_P_rel"] for r in differ], 0.99)},
           "committed_trajectory_proxy": acc,         # recorded, not gating
           "shards": shards,
           "gate": {"population": "blocks whose 128-row window holds >= 1 decode-captured row (the arms differ)",
                    "threshold": GATE, "block_equal_frac": f_diff, "blocks_equal": equal_n,
                    "block_equal_met": gate_met, "rule2_bf16_noise": rule2, "verdict": verdict,
                    "pass": verdict != "fail",
                    "if_fail": "training waits for the decode-capture fallback (d2/FALLBACK.md)"}}
    if a.taps:
        with open(a.taps) as fh:
            rep["taps"] = json.load(fh)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("acap-export")
    x.add_argument("--v1", required=True)
    x.add_argument("--captures", required=True)
    x.add_argument("--d1-exact", required=True)
    x.add_argument("--out", required=True)
    for name in ("taps", "drafts"):
        p = sp.add_parser(name)
        p.add_argument("--v1", required=True)
        p.add_argument("--captures", required=True)
        p.add_argument("--acap", required=True)
        p.add_argument("--replay", required=True)
        if name == "taps":
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--ckpt", required=True)
            p.add_argument("--official", required=True)
            p.add_argument("--work", required=True)
            p.add_argument("--shard", default="0/1")
            p.add_argument("--threads", type=int, default=12)
            p.add_argument("--every", type=int, default=1, help="every N-th block (1 = all)")
    r = sp.add_parser("report")
    r.add_argument("--work", required=True)
    r.add_argument("--taps", default="")
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"acap-export": cmd_acap_export, "taps": cmd_taps, "drafts": cmd_drafts, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
