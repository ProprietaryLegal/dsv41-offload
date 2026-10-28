#!/usr/bin/env python3
"""Single-stream benchmark for the DSv4.1 TP2 serve.

Per run: actual prompt/completion tokens (server usage), TTFT, prefill tok/s (prompt_tokens / TTFT),
decode tok/s excluding prefill ((completion_tokens - 1) / (t_last - t_first)), DSpark acceptance from /metrics
deltas, min MemAvailable and max GPU temp / clock on both nodes (sampled over ssh every 2 s).
Cold prefill: every request gets a fresh cache_salt and a random nonce at the start of the prompt.

  ./bench.py --ctx 0 8000 32000 --workload legal --reps 3 --max-tokens 1500 --out logs/bench.jsonl
"""
import argparse, json, os, random, re, statistics, subprocess, threading, time, urllib.request, uuid

URL = "http://localhost:8000"
MODEL = "DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2"
NODES = ["spark", "sparkworker"]

# Workload prompts and the synthetic-record vocabulary are not shipped with the public source (prompt-bearing text
# stays outside the PLI Labs data boundary). DSV41_BENCH_SPEC_FILE names a JSON file with keys:
#   workloads   {name: instruction} for the names in WORKLOAD_NAMES
#   first, last, places, verbs   [str]   synthetic-record vocabulary
#   claims      [str], each may use {n} and {m}
#   paragraph   str using {i} {d} {place} {a} {verb} {claim} {b} {dcn}
WORKLOAD_NAMES = ("prose", "legal", "summary", "reason", "synth", "extract")
_SPEC = None


def spec():
    global _SPEC
    if _SPEC is None:
        path = os.environ.get("DSV41_BENCH_SPEC_FILE")
        if not path:
            raise SystemExit("set DSV41_BENCH_SPEC_FILE (JSON workload spec); prompts are not shipped")
        with open(path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        missing = [k for k in ("workloads", "first", "last", "places", "verbs", "claims", "paragraph") if k not in loaded]
        if missing:
            raise SystemExit(f"DSV41_BENCH_SPEC_FILE {path} lacks keys {missing}")
        _SPEC = loaded
    return _SPEC


def record_paragraph(rng, i):
    s = spec()
    d = f"20{rng.randint(22, 25)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    a, b = f"{rng.choice(s['first'])} {rng.choice(s['last'])}", f"{rng.choice(s['first'])} {rng.choice(s['last'])}"
    c = rng.choice(s["claims"]).format(n=rng.randint(1, 9), m=rng.randint(200, 25000))
    return s["paragraph"].format(i=i, d=d, place=rng.choice(s["places"]), a=a, verb=rng.choice(s["verbs"]), claim=c,
                                 b=b, dcn=rng.randint(100000, 999999))


def tok_count(text):
    req = urllib.request.Request(URL + "/tokenize", data=json.dumps({"model": MODEL, "prompt": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["count"]


_DOCS = None
def build_real(target_tokens, seed):
    """Concatenate public CourtListener opinions + RECAP filings (data/*.jsonl) to target_tokens (exact via /tokenize)."""
    global _DOCS
    if _DOCS is None:
        _DOCS = [json.loads(l) for f in ("data/opinions_cl.jsonl", "data/filings_recap.jsonl") for l in open(f)]
    docs = _DOCS[:]; random.Random(seed).shuffle(docs)
    parts, chars = [], 0
    ratio = 3.3                                    # chars/token first guess; corrected below
    for d in docs:
        parts.append(f"=== {d.get('title', '')} ({d.get('court', '')}, {d.get('date', '')}) ===\n{d['text']}")
        chars += len(parts[-1])
        if chars / ratio >= target_tokens * 1.02:
            n = tok_count("\n\n".join(parts)); ratio = chars / n
            if n >= target_tokens:
                break
    text = "\n\n".join(parts)
    while tok_count(text) > target_tokens:            # trim to the target
        text = text[: int(len(text) - (tok_count(text) - target_tokens) * ratio) - 50]
    return text


def build_record(target_tokens, seed):
    rng = random.Random(seed)
    paras, i, approx = [], 1, 0
    while approx < target_tokens:
        p = record_paragraph(rng, i)
        paras.append(p); approx += len(p) / 3.6; i += 1   # ~3.6 chars/token for this text (calibrated below)
    return "\n".join(paras)


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.stop = threading.Event()
        self.minavail = {n: 1e9 for n in NODES}; self.maxtemp = {n: 0 for n in NODES}
        self.clk = {n: [] for n in NODES}; self.util = {n: [] for n in NODES}

    def run(self):
        while not self.stop.is_set():
            for n in NODES:
                try:
                    o = subprocess.run(["ssh", "-o", "BatchMode=yes", n,
                                        "awk '/MemAvailable/{print $2}' /proc/meminfo; nvidia-smi --query-gpu=temperature.gpu,clocks.gr,utilization.gpu --format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=10).stdout.split("\n")
                    self.minavail[n] = min(self.minavail[n], int(o[0]) / 1048576)
                    t, c, u = [x.strip() for x in o[1].split(",")]
                    self.maxtemp[n] = max(self.maxtemp[n], int(t)); self.clk[n].append(int(c)); self.util[n].append(int(u))
                except Exception:
                    pass
            self.stop.wait(2)


def metrics():
    try:
        txt = urllib.request.urlopen(URL + "/metrics", timeout=5).read().decode()
    except Exception:
        return {}
    out = {}
    for k in ("spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total",
              "spec_decode_num_drafts_total"):
        vals = re.findall(rf"^vllm:{k}(?:{{[^}}]*}})? ([0-9.e+]+)$", txt, re.M)
        out[k] = sum(float(v) for v in vals) if vals else None
    return out


def run_one(prompt, max_tokens, thinking, temperature, top_p, seed=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": False},
            "chat_template_kwargs": {"thinking": thinking}, "cache_salt": uuid.uuid4().hex}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = t_last = None; usage = None; text = []; reasoning = []; finish = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece:
                    now = time.time(); t_first = t_first or now; t_last = now
                    (reasoning if (d.get("reasoning_content") or d.get("reasoning")) else text).append(piece)
                finish = ch.get("finish_reason") or finish
    pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
    ttft = t_first - t0
    dec = (ct - 1) / (t_last - t_first) if t_last and t_last > t_first else None
    return {"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": round(ttft, 2),
            "prefill_tps": round(pt / ttft, 1), "decode_tps": round(dec, 2) if dec else None,
            "total_s": round(time.time() - t0, 1), "finish": finish,
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "text": "".join(text), "reasoning": "".join(reasoning)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[0])
    ap.add_argument("--workload", nargs="+", default=["legal"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--out", default="logs/bench.jsonl")
    ap.add_argument("--keep-text", action="store_true")
    ap.add_argument("--filler", choices=["synthetic", "real"], default="synthetic")
    a = ap.parse_args()
    for ctx in a.ctx:
        for wl in a.workload:
            rows = []
            for rep in range(a.reps):
                nonce = f"Session {uuid.uuid4().hex[:12]}.\n"
                if a.filler == "real" and ctx:
                    rec = build_real(ctx, seed=1000 * ctx + rep)
                    prompt = nonce + f"MATERIALS (public court documents):\n{rec}\n\nEND OF MATERIALS.\n\n" + spec()["workloads"][wl]
                else:
                    rec = build_record(ctx, seed=1000 * ctx + rep) if ctx else ""
                    prompt = nonce + (f"CASE RECORD (fictional):\n{rec}\n\nEND OF RECORD.\n\n" if rec else "") + spec()["workloads"][wl]
                m0 = metrics(); s = Sampler(); s.start()
                try:
                    r = run_one(prompt, a.max_tokens, a.thinking, a.temperature, a.top_p)
                    err = None
                except Exception as e:
                    r, err = {}, repr(e)
                s.stop.set(); s.join(); m1 = metrics()
                acc = None
                if m0.get("spec_decode_num_accepted_tokens_total") is not None and m1.get("spec_decode_num_drafts_total"):
                    da = m1["spec_decode_num_accepted_tokens_total"] - m0["spec_decode_num_accepted_tokens_total"]
                    dd = m1["spec_decode_num_draft_tokens_total"] - m0["spec_decode_num_draft_tokens_total"]
                    dn = m1["spec_decode_num_drafts_total"] - m0["spec_decode_num_drafts_total"]
                    acc = {"accept_rate": round(da / dd, 3) if dd else None,
                           "tokens_per_round": round(1 + da / dn, 2) if dn else None}
                row = {"ts": time.strftime("%FT%T"), "target_ctx": ctx, "workload": wl, "rep": rep, "filler": a.filler, "filler": a.filler,
                       "thinking": a.thinking, "temperature": a.temperature, "top_p": a.top_p, "error": err,
                       **{k: v for k, v in r.items() if k not in ("text", "reasoning")}, "spec": acc,
                       "min_avail_gib": {n: round(v, 2) for n, v in s.minavail.items()},
                       "max_temp_c": s.maxtemp,
                       "gpu_clock_med": {n: (statistics.median(v) if v else None) for n, v in s.clk.items()},
                       "gpu_util_med": {n: (statistics.median(v) if v else None) for n, v in s.util.items()},
                       "tail": r.get("text", "")[-300:]}
                if a.keep_text:
                    row["text"], row["reasoning"] = r.get("text"), r.get("reasoning")
                rows.append(row)
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps({k: row[k] for k in ("target_ctx", "workload", "rep", "prompt_tokens", "completion_tokens",
                                                       "ttft_s", "prefill_tps", "decode_tps", "finish", "cached_tokens",
                                                       "spec", "min_avail_gib", "max_temp_c", "error") if k in row}), flush=True)
            d = [x["decode_tps"] for x in rows if x.get("decode_tps")]
            if d:
                print(f"## ctx {ctx} {wl}: decode median {statistics.median(d):.1f} tok/s, range {min(d):.1f}-{max(d):.1f}, n={len(d)}", flush=True)


if __name__ == "__main__":
    main()
