"""D2 replay window: one prefill per finished sequence on the replay serve (configs/A-replay.env, :8001) returns the
target's top-20 at every position (prompt_logprobs 20) while the replay overlay saves the taps rows on sparkhead.
Runs on x86host; proprietary arrays go to /opt/dsv41/x86host-data/d2/replay/ (mode 700), never JSON on Controller.

  plan   python3 -m d2.replay plan --seqs /opt/dsv41/x86host-data/d2/seqs --acap /opt/dsv41/x86host-data/d2/acap \
             --out /opt/dsv41/x86host-data/d2/replay
  run    python3 -m d2.replay run --out /opt/dsv41/x86host-data/d2/replay --server http://HEAD_IP:8001
  copy   python3 -m d2.replay copy --out /opt/dsv41/x86host-data/d2/replay --metrics http://HEAD_IP:8001/metrics
  check  python3 -m d2.replay check --out /opt/dsv41/x86host-data/d2/replay
  estimate python3 -m d2.replay estimate --seqs /opt/dsv41/x86host-data/d2/seqs

Request per sequence: token-id prompt (prompt + generated), max_tokens 1, prompt_logprobs 20, return_token_ids,
priority 5, a fresh cache_salt, request_id d2r.<seq>.<from>.<L> with from = max(0, gen_start - 128) (the overlay's
tag; acap sequences use from = 0 so the equivalence check sees every row). prompt_logprobs disables prefix-cache
reads, so every row is computed and saved.

Outputs per sequence: logprobs/<seq>.npz (top_ids int32 [n,20], top_lp f32, tok_lp f32, tok_rank i32, resid f32 over
positions gen_start..L-1, DATASET.md section 6.1 conventions) + logprobs/<seq>.json (request metadata, timing, checks);
taps/<seq>/{taps.bin,ids.bin,DONE.json,ledger.jsonl} after `copy` (O_DIRECT reads on the Spark over the direct link,
copy_to_x86host.copy: rate cap, idle guard, Spark MemAvailable guard, stream + re-read sha256 against DONE.json).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

import numpy as np

from .window_checks import response_policy

W = 128
TOPK = 20
ROW_BYTES = 3 * 5120 * 2
MODEL = "DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2"
MODEL_ROOT = "/srv/dsv41/legal-v2a"
REMOTE_DIR = "/home/rank0user/dsv41-offload-run/d2-replay"
ORDER = {"acap": 0, "heldout_legal": 1, "heldout_general": 2, "dev": 3, "train": 4}


def utc(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if t is None else t))


def tokens_sha256(ids) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


def private_dir(path: str) -> str:
    os.makedirs(path, mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def atomic_json(path: str, obj) -> None:
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh)
    os.chmod(path + ".tmp", 0o600)
    os.replace(path + ".tmp", path)


def load_seq(path: str) -> dict:
    """A generation state (d2.genstate.v1) or an acap export (d2.acap.v1) -> {seq_id, tokens, gen_start, ...}."""
    with open(path) as fh:
        s = json.load(fh)
    if s.get("schema") == "d2.acap.v1":
        return {"seq_id": s["seq_id"], "split": "acap", "family": "acap", "kind": "acap", "tokens": s["tokens"],
                "gen_start": s["gen_start"], "finish_reason": "capture", "segments": 1,
                "prompt_sha256": s["prompt_sha256"], "src": path}
    if s.get("schema") != "d2.genstate.v1":
        raise ValueError(f"{path}: unknown schema {s.get('schema')!r}")
    if s["status"] not in ("done", "partial", "stopped"):
        return {}
    toks = s["prompt_ids"] + s["gen_ids"]
    if s["status"] == "done" and s.get("tokens_sha256") != tokens_sha256(toks):
        raise ValueError(f"{path}: tokens_sha256 does not match the stored ids")
    return {"seq_id": s["pid"], "split": s["split"], "family": s["family"], "kind": "gen", "tokens": toks,
            "gen_start": len(s["prompt_ids"]), "finish_reason": s.get("finish_reason") or "aborted",
            "segments": len(s["segments"]), "prompt_sha256": s["prompt_sha256"], "src": path}


# ----------------------------------------------------------------------------------------------------------- plan
def cmd_plan(a) -> None:
    out = private_dir(a.out)
    entries = []
    srcs = []
    if a.seqs:
        srcs += [os.path.join(a.seqs, f) for f in sorted(os.listdir(a.seqs)) if f.endswith(".json")]
    if a.acap:
        srcs += [os.path.join(a.acap, f) for f in sorted(os.listdir(a.acap)) if f.endswith(".json")]
    skipped = 0
    for p in srcs:
        s = load_seq(p)
        if not s:
            skipped += 1
            continue
        L, g = len(s["tokens"]), s["gen_start"]
        if not 1 <= g <= L - 2:
            skipped += 1                      # fewer than two generated tokens: no training block
            continue
        frm = 0 if s["kind"] == "acap" else max(0, g - W)
        entries.append({"seq_id": s["seq_id"], "split": s["split"], "family": s["family"], "kind": s["kind"],
                        "src": p, "L": L, "gen_start": g, "from": frm, "tokens_sha256": tokens_sha256(s["tokens"]),
                        "prompt_sha256": s["prompt_sha256"], "finish_reason": s["finish_reason"],
                        "segments": s["segments"], "request_id": f"d2r.{s['seq_id']}.{frm}.{L}"})
    entries.sort(key=lambda e: (ORDER[e["split"]], e["seq_id"]))
    if len({e["seq_id"] for e in entries}) != len(entries):
        raise RuntimeError("duplicate seq_id in the plan")
    with open(os.path.join(out, "plan.jsonl"), "w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    os.chmod(os.path.join(out, "plan.jsonl"), 0o600)
    est = estimate(entries)
    print(json.dumps({"entries": len(entries), "skipped": skipped, **est}, indent=1))


def estimate(entries: list[dict], prefill_tok_s: float = 2200.0, replay_tok_s: float = 1060.0) -> dict:
    """Window cost: every token is prefilled once; rows from..L-1 are written; Phase 0 measured ~1,060 tok/s end to end
    for one request at a time (8.3K-token requests incl. logprob serialisation and transfer)."""
    tok = sum(e["L"] for e in entries)
    rows = sum(e["L"] - e["from"] for e in entries)
    span = sum(e["L"] - e["gen_start"] for e in entries)
    return {"prefill_tokens": tok, "span_positions": span, "tap_rows": rows,
            "taps_GB": round(rows * ROW_BYTES / 1e9, 2), "top20_MB": round(span * (TOPK * 8 + 12) / 1e6, 1),
            "prompt_logprobs_json_GB": round(tok * 1430 / 1e9, 2),
            "minutes_at_pure_prefill_2200": round(tok / prefill_tok_s / 60, 1),
            "minutes_at_phase0_rate_1060_one_in_flight": round(tok / replay_tok_s / 60, 1),
            "copy_minutes_at_1GBps": round(rows * ROW_BYTES / 1e9 / 60, 1)}


def read_plan(out: str) -> list[dict]:
    with open(os.path.join(out, "plan.jsonl")) as fh:
        return [json.loads(line) for line in fh]


# ------------------------------------------------------------------------------------------------------------ run
def parse_logprobs(resp: dict, tokens: list[int], gen_start: int) -> tuple[dict[str, np.ndarray], dict]:
    """prompt_logprobs -> DATASET.md arrays over positions gen_start..L-1, with every check of section 8 rule 4/5."""
    response_policy(resp)                  # refuse HTTP-200 errors/refusals and cache hits before writing arrays
    ch = resp["choices"][0]
    echo = ch.get("prompt_token_ids")
    if echo is None:
        raise ValueError("response carries no prompt_token_ids (return_token_ids ignored?)")
    if [int(x) for x in echo] != tokens:
        raise ValueError("the server prefilled different token ids than were sent")
    pl = ch.get("prompt_logprobs")
    if pl is None or len(pl) != len(tokens):
        raise ValueError(f"prompt_logprobs has {None if pl is None else len(pl)} entries for {len(tokens)} tokens")
    n = len(tokens) - gen_start
    top_ids = np.zeros((n, TOPK), dtype=np.int32)
    top_lp = np.zeros((n, TOPK), dtype=np.float32)
    tok_lp = np.zeros(n, dtype=np.float32)
    tok_rank = np.zeros(n, dtype=np.int32)
    ties = 0
    for j in range(n):
        p = gen_start + j
        e = pl[p]
        t = tokens[p]
        if e is None:
            raise ValueError(f"position {p}: no prompt logprobs")
        items = sorted(((int(k), float(v["logprob"]), int(v["rank"])) for k, v in e.items()),
                       key=lambda x: (-x[1], x[2], x[0]))
        mine = [x for x in items if x[0] == t]
        if not mine:
            raise ValueError(f"position {p}: the committed token's own logprob is missing")
        if len(items) == TOPK + 1 and mine[0][2] > TOPK:
            top = [x for x in items if x[0] != t]
        else:
            if len(items) not in (TOPK, TOPK + 1):
                raise ValueError(f"position {p}: {len(items)} logprob entries")
            if len(items) == TOPK + 1:
                ties += 1                    # the token ranks <= 20 but 21 entries came back: a tie at the edge
            top = items[:TOPK]
        ids = [x[0] for x in top]
        top_ids[j] = ids
        top_lp[j] = [x[1] for x in top]
        tok_lp[j] = mine[0][1]
        tok_rank[j] = ids.index(t) + 1 if t in ids else max(TOPK + 1, mine[0][2])
    mass = np.exp(top_lp.astype(np.float64)).sum(-1)
    if not np.isfinite(top_lp).all() or (top_lp > 1e-6).any() or (mass > 1 + 1e-4).any():
        raise ValueError(f"top-20 logprobs out of range (max mass {mass.max():.6f})")
    resid = np.clip(1.0 - mass, 0.0, 1.0).astype(np.float32)
    return ({"top_ids": top_ids, "top_lp": top_lp, "tok_lp": tok_lp, "tok_rank": tok_rank, "resid": resid},
            {"ties_at_edge": ties, "mean_resid": float(resid.mean()) if n else None,
             "tok_in_top20": float((tok_rank <= TOPK).mean()) if n else None})


def post(server: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(server.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def replay_one(a, e: dict, lp_dir: str) -> dict:
    s = load_seq(e["src"])
    toks = s["tokens"]
    if tokens_sha256(toks) != e["tokens_sha256"] or len(toks) != e["L"]:
        raise RuntimeError(f"{e['seq_id']}: source ids changed since the plan")
    body = {"model": a.model, "prompt": toks, "max_tokens": 1, "temperature": 1.0, "top_p": 1.0,
            "prompt_logprobs": TOPK, "return_token_ids": True, "priority": 5, "request_id": e["request_id"],
            "cache_salt": f"d2r-{e['seq_id']}-{secrets.token_hex(16)}"}
    t0 = time.time()
    resp = post(a.server, body, a.timeout)
    t1 = time.time()
    arrays, chk = parse_logprobs(resp, toks, e["gen_start"])
    t2 = time.time()
    np.savez(os.path.join(lp_dir, e["seq_id"] + ".npz.tmp.npz"), **arrays)
    os.chmod(os.path.join(lp_dir, e["seq_id"] + ".npz.tmp.npz"), 0o600)
    os.replace(os.path.join(lp_dir, e["seq_id"] + ".npz.tmp.npz"), os.path.join(lp_dir, e["seq_id"] + ".npz"))
    usage = resp.get("usage") or {}
    meta = {"seq_id": e["seq_id"], "request_id": e["request_id"], "L": e["L"], "gen_start": e["gen_start"],
            "from": e["from"], "tokens_sha256": e["tokens_sha256"], "server": a.server, "model": resp.get("model"),
            "created": resp.get("created"), "usage": usage, "seconds_request": round(t1 - t0, 3),
            "seconds_parse": round(t2 - t1, 3), "utc": utc(t0), **chk}
    atomic_json(os.path.join(lp_dir, e["seq_id"] + ".json"), meta)
    return meta


def check_server(a) -> None:
    u = urllib.parse.urlparse(a.server)
    if u.port == 8000 and not a.allow_production:
        raise SystemExit("refusing :8000: the replay runs on the replay serve (:8001, configs/A-replay.env)")
    with urllib.request.urlopen(a.server.rstrip("/") + "/v1/models", timeout=10) as r:
        m = json.loads(r.read())["data"][0]
    if m["id"] != a.model or m.get("root") != MODEL_ROOT:
        raise SystemExit(f"{a.server} serves {m['id']} from {m.get('root')}, want {a.model} from {MODEL_ROOT}")


def remote_done(a, seq: str) -> dict | None:
    """DONE.json of one sequence on sparkhead (a small file: plain ssh cat), or None if absent."""
    r = subprocess.run(remote_ssh(a) + [f"cat {a.remote_dir}/{seq}/DONE.json 2>/dev/null || true"],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"ssh to the Spark failed: {r.stderr.strip()[:300]}")
    return json.loads(r.stdout) if r.stdout.strip() else None


def remote_ssh(a) -> list[str]:
    import copy_to_x86host
    return list(a.ssh.split()) if a.ssh else list(copy_to_x86host.SSH)


def cmd_run(a) -> None:
    check_server(a)
    out = private_dir(a.out)
    lp_dir = private_dir(os.path.join(out, "logprobs"))
    plan = read_plan(out)
    todo = [e for e in plan if not os.path.exists(os.path.join(lp_dir, e["seq_id"] + ".npz"))]
    if a.limit:
        todo = todo[: a.limit]
    log = open(os.path.join(out, "run.jsonl"), "a")
    os.chmod(log.name, 0o600)
    print(f"{utc()} replay: {len(todo)} of {len(plan)} sequences to go, {a.inflight} in flight, {a.server}", flush=True)
    lock = threading.Lock()
    t0 = time.time()
    done_tok = 0
    first_checked = not a.check_remote

    def work(e):
        return replay_one(a, e, lp_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.inflight) as ex:
        it = iter(todo)
        futs = {}
        for e in it:
            futs[ex.submit(work, e)] = e
            if len(futs) >= a.inflight or not first_checked:
                break
        while futs:
            done, _ = concurrent.futures.wait(futs, return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                e = futs.pop(f)
                meta = f.result()                   # an error stops the run loudly; finished sequences stay saved
                with lock:
                    done_tok += e["L"]
                    el = time.time() - t0
                    rec = {"t": utc(), "seq_id": e["seq_id"], "L": e["L"], "s": meta["seconds_request"],
                           "tok_s_req": round(e["L"] / max(meta["seconds_request"], 1e-9)),
                           "tok_s_run": round(done_tok / max(el, 1e-9)), "ties": meta["ties_at_edge"]}
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(json.dumps(rec), flush=True)
                if not first_checked:
                    dn = remote_done(a, e["seq_id"])
                    if dn is None or dn.get("rows_missing") != 0 or dn.get("rows") != e["L"] - e["from"] \
                            or dn.get("zero_rows") != 0:
                        raise SystemExit(f"{e['seq_id']}: the replay overlay did not save its rows (DONE.json {dn}); "
                                         "is the serve running configs/A-replay.env?")
                    first_checked = True
                    print(f"{utc()} overlay check passed on {e['seq_id']}: {dn['rows']} rows", flush=True)
            for e in it:
                futs[ex.submit(work, e)] = e
                if len(futs) >= a.inflight:
                    break
    print(f"{utc()} replay finished: {len(todo)} sequences, {done_tok} tokens in {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------------------------------------- copy
def cmd_copy(a) -> None:
    import copy_to_x86host as C
    C.METRICS = a.metrics
    if a.ssh:
        C.SSH = a.ssh.split()
    if a.rate:
        C.RATE = a.rate
    out = private_dir(a.out)
    tdir = private_dir(os.path.join(out, "taps"))
    plan = read_plan(out)
    receipts = open(os.path.join(out, "copy_receipts.jsonl"), "a")
    os.chmod(receipts.name, 0o600)
    errs = subprocess.run(remote_ssh(a) + [f"cat {a.remote_dir}/ERRORS.jsonl 2>/dev/null || true"],
                          capture_output=True, text=True, timeout=30)
    if errs.returncode != 0:
        raise RuntimeError(f"ssh to the Spark failed: {errs.stderr.strip()[:300]}")
    planned = {e["seq_id"] for e in plan}
    dupes: set[str] = set()
    if errs.stdout.strip():
        recs = [json.loads(x) for x in errs.stdout.splitlines() if x.strip()]
        # S14 F11: a sequence replayed again after its DONE.json keeps its first complete replay; that is recorded,
        # not an error. Anything else the overlay recorded for a planned sequence stops the copy.
        dupes = {r.get("seq") for r in recs if r.get("kind") == "already_complete"}
        bad = {r.get("seq") for r in recs if r.get("kind", "error") != "already_complete"}
        if bad & planned:
            raise SystemExit(f"the overlay recorded errors for planned sequences: {sorted(bad & planned)[:10]}")
    n = 0
    for e in plan:
        if a.limit and n >= a.limit:
            break
        d = private_dir(os.path.join(tdir, e["seq_id"]))
        final = os.path.join(d, "DONE.json")
        if os.path.exists(final) and os.path.exists(os.path.join(d, "taps.bin")) and \
                os.path.exists(os.path.join(d, "ids.bin")):
            continue
        dn = remote_done(a, e["seq_id"])
        if dn is None:
            raise SystemExit(f"{e['seq_id']}: no DONE.json on the Spark (not replayed, or the overlay failed)")
        rows = e["L"] - e["from"]
        if (dn["from"], dn["end"], dn["rows"], dn["rows_missing"]) != (e["from"], e["L"], rows, 0):
            raise SystemExit(f"{e['seq_id']}: DONE.json {dn} disagrees with the plan ({e['from']}..{e['L']})")
        if dn.get("zero_rows") != 0:
            raise SystemExit(f"{e['seq_id']}: DONE.json zero_rows {dn.get('zero_rows')} (first at position "
                             f"{dn.get('first_zero_row')}): trimmed eager-prefill taps, or an overlay without the "
                             "zero-row count; not copied")
        led = subprocess.run(remote_ssh(a) + [f"cat {a.remote_dir}/{e['seq_id']}/ledger.jsonl"], capture_output=True,
                             text=True, timeout=60, check=True).stdout
        src = f"{a.remote_dir}/{e['seq_id']}"
        r1 = C.copy(src + "/taps.bin", C.Path(os.path.join(d, "taps.bin")), dn["sha256_taps"], rows * ROW_BYTES,
                    resume=bool(dn["in_order"]))
        r2 = C.copy(src + "/ids.bin", C.Path(os.path.join(d, "ids.bin")), dn["sha256_ids"], rows * 4,
                    resume=bool(dn["in_order"]))
        if not dn["in_order"]:
            # rows arrived out of order or twice: no stream hash; the copy's own sha256 is the identity from here on
            dn["sha256_taps"], dn["sha256_ids"] = r1["sha256"], r2["sha256"]
            dn["sha256_from_copy"] = True
        with open(os.path.join(d, "ledger.jsonl"), "w") as fh:
            fh.write(led)
        os.chmod(os.path.join(d, "ledger.jsonl"), 0o600)
        if e["seq_id"] in dupes:
            dn["duplicate_replay"] = True                 # replayed twice; the first complete replay was kept
        atomic_json(final, dn)
        receipts.write(json.dumps({"seq_id": e["seq_id"], "taps": r1, "ids": r2,
                                   "duplicate_replay": e["seq_id"] in dupes}) + "\n")
        receipts.flush()
        n += 1
    print(f"{utc()} copied {n} sequences", flush=True)


# ---------------------------------------------------------------------------------------------------------- check
def cmd_check(a) -> None:
    plan = read_plan(a.out)
    rep = {"planned": len(plan), "logprobs": 0, "taps": 0, "missing_logprobs": [], "missing_taps": [], "bad": []}
    for e in plan:
        lp = os.path.join(a.out, "logprobs", e["seq_id"] + ".npz")
        td = os.path.join(a.out, "taps", e["seq_id"])
        if os.path.exists(lp):
            rep["logprobs"] += 1
            z = np.load(lp)
            if z["top_ids"].shape != (e["L"] - e["gen_start"], TOPK):
                rep["bad"].append(f"{e['seq_id']}: top_ids {z['top_ids'].shape}")
        else:
            rep["missing_logprobs"].append(e["seq_id"])
        if os.path.exists(os.path.join(td, "DONE.json")):
            rep["taps"] += 1
            rows = e["L"] - e["from"]
            if os.path.getsize(os.path.join(td, "taps.bin")) != rows * ROW_BYTES:
                rep["bad"].append(f"{e['seq_id']}: taps.bin size")
            ids = np.fromfile(os.path.join(td, "ids.bin"), dtype="<i4")
            toks = load_seq(e["src"])["tokens"]
            if ids.tolist() != toks[e["from"]: e["L"]]:
                rep["bad"].append(f"{e['seq_id']}: overlay row ids differ from the sequence")
        else:
            rep["missing_taps"].append(e["seq_id"])
    rep["missing_logprobs"] = rep["missing_logprobs"][:20]
    rep["missing_taps"] = rep["missing_taps"][:20]
    print(json.dumps(rep, indent=1))
    if rep["bad"]:
        raise SystemExit(1)


def cmd_estimate(a) -> None:
    entries = []
    for f in sorted(os.listdir(a.seqs)):
        if f.endswith(".json"):
            s = load_seq(os.path.join(a.seqs, f))
            if s:
                L, g = len(s["tokens"]), s["gen_start"]
                entries.append({"L": L, "gen_start": g, "from": max(0, g - W)})
    print(json.dumps({"sequences": len(entries), **estimate(entries)}, indent=1))


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # offload/dspark: copy_to_x86host
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--seqs", default="")
    p.add_argument("--acap", default="")
    p.add_argument("--out", required=True)
    r = sp.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--server", required=True)
    r.add_argument("--model", default=MODEL)
    r.add_argument("--inflight", type=int, default=2)
    r.add_argument("--timeout", type=float, default=1800.0)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--allow-production", action="store_true")
    r.add_argument("--check-remote", action="store_true", help="after the first sequence, check its DONE.json on the "
                   "Spark (fails fast when the serve is not running the replay overlay)")
    r.add_argument("--remote-dir", default=REMOTE_DIR)
    r.add_argument("--ssh", default="", help="ssh command to the Spark (default: copy_to_x86host's direct-link SSH)")
    c = sp.add_parser("copy")
    c.add_argument("--out", required=True)
    c.add_argument("--metrics", required=True, help="the serve whose idleness guards the copy (:8001 in the window)")
    c.add_argument("--remote-dir", default=REMOTE_DIR)
    c.add_argument("--ssh", default="")
    c.add_argument("--rate", type=int, default=0, help="bytes/s cap (default copy_to_x86host's 1e9)")
    c.add_argument("--limit", type=int, default=0)
    k = sp.add_parser("check")
    k.add_argument("--out", required=True)
    e = sp.add_parser("estimate")
    e.add_argument("--seqs", required=True)
    a = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "copy": cmd_copy, "check": cmd_check, "estimate": cmd_estimate}[a.cmd](a)


if __name__ == "__main__":
    main()
