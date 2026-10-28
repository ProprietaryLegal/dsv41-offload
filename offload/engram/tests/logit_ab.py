#!/usr/bin/env python3
"""Fixed-prefix output comparison of two serve configurations (A = local Engram, B = remote Engram), for the
coordinator's maintenance window. Greedy, thinking off, a fresh cache_salt per request (cold prefill: every prompt
position goes through the Engram lookups), top-20 logprobs of every generated token and, when the serve accepts it,
prompt_logprobs (top-5 at every prompt position).

    logit_ab.py run --url http://HEAD_IP:8001 --label A1 --out logs/ab_A1.jsonl [--data opinions_cl.jsonl]
    logit_ab.py compare logs/ab_A1.jsonl logs/ab_A2.jsonl      # noise floor (A vs A)
    logit_ab.py compare logs/ab_A1.jsonl logs/ab_B1.jsonl      # B vs A

Prompts: a short fixed legal question, and ~8K / ~32K-token prefixes built from public CourtListener opinions
(data/opinions_cl.jsonl, the bench's filler) - fixed text, so A and B see identical token ids.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

MODEL = "DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2"


def post(url, body, timeout=3600):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _prompt_spec():
    """Prompt text is not shipped with the public source. DSV41_LOGIT_AB_PROMPTS_FILE names a JSON file with keys
    "short" (a short question) and "summarize" (the instruction placed before the long documents)."""
    path = os.environ.get("DSV41_LOGIT_AB_PROMPTS_FILE")
    if not path:
        raise SystemExit("set DSV41_LOGIT_AB_PROMPTS_FILE (JSON with keys short, summarize); prompts are not shipped")
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    missing = [k for k in ("short", "summarize") if k not in spec]
    if missing:
        raise SystemExit(f"DSV41_LOGIT_AB_PROMPTS_FILE {path} lacks keys {missing}")
    return spec


def prompts(data_path, sizes):
    spec = _prompt_spec()
    out = [("short", spec["short"])]
    if data_path:
        docs = [json.loads(l)["text"] for l in open(data_path)]
        for chars in sizes:
            text, i = "", 0
            while len(text) < chars:
                text += docs[i % len(docs)] + "\n\n"
                i += 1
            out.append((f"{chars // 4000}k", spec["summarize"] + "\n\n" + text[:chars]))
    return out


def run(a):
    rows = []
    for name, text in prompts(a.data, a.sizes):
        body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": a.max_tokens,
                "temperature": 0.0, "top_p": 1.0, "logprobs": True, "top_logprobs": 20, "priority": 10,
                "chat_template_kwargs": {"thinking": False}, "cache_salt": uuid.uuid4().hex,
                "prompt_logprobs": 5}
        t0 = time.time()
        try:
            r = post(a.url, body)
            plp = True
        except urllib.error.HTTPError as e:  # prompt_logprobs may be refused (e.g. with speculative decoding)
            msg = e.read().decode()[:300]
            body.pop("prompt_logprobs")
            body["cache_salt"] = uuid.uuid4().hex
            r = post(a.url, body)
            plp = f"refused: {msg}"
        ch = r["choices"][0]
        toks = [{"token": t["token"], "logprob": t["logprob"],
                 "top": {x["token"]: x["logprob"] for x in t.get("top_logprobs", [])}}
                for t in (ch.get("logprobs") or {}).get("content", [])]
        row = {"label": a.label, "prompt": name, "secs": round(time.time() - t0, 2), "usage": r.get("usage"),
               "text": ch["message"].get("content"), "tokens": toks, "prompt_logprobs": plp,
               "prompt_lp": r.get("prompt_logprobs") if plp is True else None}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("label", "prompt", "secs", "usage")}), flush=True)
    with open(a.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def compare(a):
    A = {r["prompt"]: r for r in map(json.loads, open(a.a))}
    B = {r["prompt"]: r for r in map(json.loads, open(a.b))}
    worst = 0.0
    same_all = True
    for name in A:
        if name not in B:
            print(f"{name}: missing in {a.b}")
            same_all = False
            continue
        ta, tb = A[name]["tokens"], B[name]["tokens"]
        n = min(len(ta), len(tb))
        first_diff = next((i for i in range(n) if ta[i]["token"] != tb[i]["token"]), None)
        same = first_diff is None and len(ta) == len(tb)
        diffs = []
        for i in range(n if first_diff is None else first_diff):
            common = set(ta[i]["top"]) & set(tb[i]["top"])
            diffs += [abs(ta[i]["top"][k] - tb[i]["top"][k]) for k in common]
        mx = max(diffs) if diffs else 0.0
        worst = max(worst, mx)
        p_diff = None
        pa, pb = A[name].get("prompt_lp"), B[name].get("prompt_lp")
        if pa and pb:
            p_diff = 0.0
            for x, y in zip(pa, pb):
                if not x or not y:
                    continue
                for k in set(x) & set(y):
                    p_diff = max(p_diff, abs(x[k]["logprob"] - y[k]["logprob"]))
        same_all &= same
        print(f"{name}: generated tokens {'identical' if same else f'differ from position {first_diff}'} "
              f"({len(ta)} vs {len(tb)}); max |dlogprob| over top-20 = {mx:.3g}"
              + (f"; prompt top-5 max |dlogprob| = {p_diff:.3g}" if p_diff is not None else "; no prompt logprobs"))
    print(f"OVERALL: tokens {'identical' if same_all else 'DIFFER'}; worst |dlogprob| {worst:.3g}")
    return 0 if same_all else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--data", default="data/opinions_cl.jsonl")
    r.add_argument("--sizes", type=int, nargs="*", default=[32000, 128000])  # chars (~8K, ~32K tokens)
    r.add_argument("--max-tokens", type=int, default=64)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    sys.exit(run(a) if a.cmd == "run" else compare(a))
