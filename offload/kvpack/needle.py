#!/usr/bin/env python3
"""K1 needle / cross-request corruption check for KV-layout changes (0057/0060) and long max-model-len.

A haystack of public CourtListener material (the bench corpus) of an exact token length holds N "vault code" needles
at evenly spread depths; the model (T=0, thinking off, priority 10) must list every code. --concurrent C sends C
different haystacks (different seeds, so different codes) at once: a KV page shared or overwritten across requests shows
up as a missed code or, more tellingly, a FOREIGN code (one that belongs to another request in the same wave).
  python3 offload/kvpack/needle.py --url http://HEAD_IP:8001 --ctx 250000 --needles 6 --out offload/results/K1/needle.jsonl
  python3 offload/kvpack/needle.py --url ... --ctx 8000 30000 50000 100000 --concurrent 4 --out ...   # corruption test
  python3 offload/kvpack/needle.py --dry-run --ctx 8000                                                  # offline: build + score
Exit status 1 when any request recalls fewer than --min-recall of its needles or returns a foreign code.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

NAMES = ["Abbeville", "Beaufort", "Calhoun", "Darlington", "Edgefield", "Fairfield", "Georgetown", "Horry", "Jasper",
         "Kershaw", "Lancaster", "Marion", "Newberry", "Oconee", "Pickens", "Richland", "Saluda", "Union"]


def load_docs(data_dir: Path) -> list[dict]:
    docs = []
    for name in ("opinions_cl.jsonl", "filings_recap.jsonl"):
        with (data_dir / name).open() as f:
            docs.extend(json.loads(line) for line in f)
    return docs


class Builder:
    def __init__(self, url: str | None, model: str | None, data_dir: Path):
        self.url, self.model, self.docs = url, model, load_docs(data_dir)

    def count(self, text: str) -> int:
        if self.url is None:
            return int(len(text) / 3.3)                         # dry run: the bench's first-guess ratio
        req = urllib.request.Request(self.url + "/tokenize", data=json.dumps({"model": self.model, "prompt": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return int(json.load(r)["count"])

    def build(self, ctx: int, n: int, seed: int) -> tuple[str, dict[str, str]]:
        rng = random.Random(seed)
        names = rng.sample(NAMES, n)
        codes = {name: f"{rng.randrange(10**5, 10**6)}" for name in names}
        question = ("\n\nEND OF MATERIALS.\n\nSomewhere in the materials above are sentences giving the vault code for "
                    f"{n} vaults. List the vault code for each of these vaults: {', '.join(names)}. Answer only with "
                    "lines of the form `Name: code`, one per vault, nothing else.")
        budget = ctx - self.count(question) - 64 * n
        docs = self.docs[:]
        rng.shuffle(docs)
        parts, chars, ratio = [], 0, 3.3
        for d in docs * 4:                                    # 1M tokens may need the corpus more than once
            parts.append(f"=== {d.get('title', '')} ({d.get('court', '')}, {d.get('date', '')}) ===\n{d['text']}")
            chars += len(parts[-1])
            if chars / ratio >= budget * 1.02:
                got = self.count("\n\n".join(parts))
                ratio = chars / got
                if got >= budget:
                    break
        hay = "\n\n".join(parts)
        while (got := self.count(hay)) > budget:
            hay = hay[: int(len(hay) - (got - budget) * ratio) - 50]
        # needles at paragraph boundaries, evenly spread from 5 % to 95 % depth
        out, last = [], 0
        for i, name in enumerate(names):
            pos = int(len(hay) * (0.05 + 0.9 * i / max(1, n - 1)))
            cut = hay.find("\n", pos)
            cut = len(hay) if cut < 0 else cut
            out.append(hay[last:cut])
            out.append(f"\nThe vault code for {name} is {codes[name]}.\n")
            last = cut
        out.append(hay[last:])
        return "".join(out) + question, codes


def score(text: str, codes: dict[str, str], foreign: set[str]) -> dict:
    found = {name: (m.group(1) if (m := re.search(rf"{name}[^\d\n]{{0,20}}(\d{{6}})", text)) else None)
             for name in codes}
    hits = sum(found[name] == code for name, code in codes.items())
    returned = set(re.findall(r"\b\d{6}\b", text))
    return {"recall": hits, "of": len(codes), "found": found, "foreign_codes": sorted(returned & foreign)}


def ask(url: str, model: str, prompt: str, max_tokens: int, salt: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_p": 1,
            "max_tokens": max_tokens, "priority": 10, "cache_salt": salt,
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=7200) as r:
        res = json.load(r)
    msg = res["choices"][0]["message"]
    return {"text": msg.get("content") or "", "elapsed_s": round(time.monotonic() - t0, 2), "usage": res.get("usage")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--ctx", type=int, nargs="+", required=True)
    ap.add_argument("--needles", type=int, default=6)
    ap.add_argument("--concurrent", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1022)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--min-recall", type=float, default=1.0)
    ap.add_argument("--data-dir", type=Path, default=Path("/opt/dsv41/data/bench-data"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not args.url:
        ap.error("--url is required unless --dry-run")
    url = None if args.dry_run else args.url.rstrip("/")
    model = args.model
    if url and not model:
        with urllib.request.urlopen(url + "/v1/models", timeout=30) as r:
            model = json.load(r)["data"][0]["id"]
    b = Builder(url, model, args.data_dir)
    rows, failed = [], False

    def counters() -> dict[str, float]:
        if url is None:
            return {}
        with urllib.request.urlopen(url + "/metrics", timeout=30) as r:
            out: dict[str, float] = {}
            for line in r.read().decode().splitlines():
                if line.startswith(("vllm:num_preemptions", "vllm:prefix_cache_queries", "vllm:prefix_cache_hits")):
                    name = line.split("{")[0].split(" ")[0]
                    out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            return out

    m0 = counters()
    for ctx in args.ctx:
        for rep in range(args.reps):
            seeds = [args.seed + 1000 * rep + 17 * i + ctx for i in range(args.concurrent)]
            built = [b.build(ctx, args.needles, s) for s in seeds]
            all_codes = [set(c.values()) for _, c in built]
            results: list[dict | None] = [None] * len(built)

            def run(i: int) -> None:
                prompt, codes = built[i]
                foreign = set().union(*(all_codes[:i] + all_codes[i + 1:])) if len(built) > 1 else set()
                if url is None:   # dry run: score a perfect answer and a cross-contaminated one
                    good = "\n".join(f"{k}: {v}" for k, v in codes.items())
                    results[i] = {"dry_run": True, "prompt_chars": len(prompt), **score(good, codes, foreign)}
                    return
                ans = ask(url, model, prompt, args.max_tokens, f"k1-needle-{seeds[i]}-{time.time_ns()}")
                results[i] = {**ans, **score(ans["text"], codes, foreign)}

            threads = [threading.Thread(target=run, args=(i,)) for i in range(len(built))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for i, r in enumerate(results):
                assert r is not None
                ok = r["recall"] >= args.min_recall * r["of"] and not r["foreign_codes"]
                failed |= not ok
                row = {"ctx": ctx, "rep": rep, "slot": i, "concurrent": args.concurrent, "seed": seeds[i], "ok": ok,
                       **{k: v for k, v in r.items() if k != "text"}, "answer": r.get("text", "")[:400],
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                rows.append(row)
                usage = r.get("usage") or {}
                print(f"ctx {ctx:>9,} rep {rep} slot {i}: {r['recall']}/{r['of']} "
                      f"{'OK' if ok else 'FAIL'} foreign={r['foreign_codes']} prompt_tokens={usage.get('prompt_tokens')} "
                      f"elapsed={r.get('elapsed_s')}", flush=True)
    m1 = counters()
    delta = {k: round(m1.get(k, 0.0) - m0.get(k, 0.0), 3) for k in sorted(set(m0) | set(m1))}
    if delta:
        print("metrics delta:", json.dumps(delta))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as f:
            for row in rows:
                f.write(json.dumps({**row, "metrics_delta": delta}) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
