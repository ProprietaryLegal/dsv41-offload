#!/usr/bin/env python3
"""K1 (S8 R-4a): a prefix-cache hit must give the same output as a cold run (T=0, thinking off, top-20 logprobs).

Packing changes what a prefix hit reads: hits align to the 384-token scheduler block, and the ratio-1 group's 192-token
blocks are found through 64-token hashes. For each length n:
  cold    = P with a fixed salt S, first time          (cached_tokens must be 0)
  hit     = P with the same salt S                      (cached_tokens must be floor((prompt_tokens - 1) / block) * block)
  control = P with a fresh salt                         (cold again, cached_tokens must be 0: the run-to-run floor)
--partial N:K sends a P2 that shares P1's first K tokens (different tail), cold with salt S after P1, plus a control.
--order ABA (pool-pressure runs) sends A, then --fillers other prompts (B, C, ...), then A again, one salt each: under
pressure A's blocks are evicted (A's second cached_tokens < full) or requests preempted; A's two outputs must agree.
Every fresh-salt control (and ABA's first A) is checked for cached_tokens 0 (S8R2 FIX-4): a control that hit the cache
would quietly widen the floor.
The verdict is PASS (cached as expected; outputs identical, or no further apart than cold vs control), FLAG (cached as
expected but outputs further apart than this run's floor: compare with the same row in R0, the v5 layout), or FAIL
(cached_tokens wrong, including a cold run or control that read the cache). Exit 1 on any FAIL.
  python3 offload/kvpack/prefix_identity.py --url http://HEAD_IP:8001 --block 384 --n 36016 100017 250000 \\
      --partial 250000:200000 --out offload/results/K1/prefix-R1.jsonl
  python3 offload/kvpack/prefix_identity.py --url ... --block 128 ...                 # R0 (v5 layout): the floor
  python3 offload/kvpack/prefix_identity.py --url ... --order ABA --n 300000 ...      # R1b pool pressure
  python3 offload/kvpack/prefix_identity.py --dry-run --n 8000                         # offline self-check
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from needle import load_docs  # noqa: E402

QUESTION = ("\n\nEND OF MATERIALS.\n\nIn two sentences, state the main holding of the first document in the "
            "materials above.")


class Client:
    def __init__(self, url: str | None, model: str | None, data_dir: Path):
        self.url, self.model, self.docs = url, model, load_docs(data_dir)

    def post(self, path: str, body: dict, timeout: float = 7200) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def count(self, text: str) -> int:
        if self.url is None:
            return int(len(text) / 3.3)
        return int(self.post("/tokenize", {"model": self.model, "prompt": text}, 900)["count"])

    def chat_tokens(self, content: str) -> list[int] | None:
        """Token ids of the chat-formatted prompt (as the server builds it), or None if /tokenize cannot do chat."""
        if self.url is None:
            return None
        try:
            r = self.post("/tokenize", {"model": self.model, "messages": [{"role": "user", "content": content}],
                                        "add_generation_prompt": True, "chat_template_kwargs": {"thinking": False}}, 900)
            return [int(t) for t in r["tokens"]]
        except Exception:  # noqa: BLE001 - optional refinement; the bounds check below still applies
            return None

    def text(self, tokens: int, seed: int) -> str:
        """Corpus text of exactly `tokens` tokens (bench corpus, shuffled by seed)."""
        rng = random.Random(seed)
        docs = self.docs[:]
        rng.shuffle(docs)
        parts, chars, ratio = [], 0, 3.3
        for d in docs * 4:
            parts.append(f"=== {d.get('title', '')} ({d.get('court', '')}, {d.get('date', '')}) ===\n{d['text']}")
            chars += len(parts[-1])
            if chars / ratio >= tokens * 1.02:
                got = self.count("\n\n".join(parts))
                ratio = chars / got
                if got >= tokens:
                    break
        t = "\n\n".join(parts)
        while (got := self.count(t)) > tokens:
            t = t[: int(len(t) - (got - tokens) * ratio) - 50]
        return t

    def ask(self, content: str, salt: str, max_tokens: int) -> dict:
        if self.url is None:
            raise RuntimeError("dry run sends nothing")
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0, "top_p": 1,
                "max_tokens": max_tokens, "priority": 10, "cache_salt": salt, "logprobs": True, "top_logprobs": 20,
                "chat_template_kwargs": {"thinking": False}}
        t0 = time.monotonic()
        r = self.post("/v1/chat/completions", body)
        choice = r["choices"][0]
        toks = [{"t": c["token"], "lp": c["logprob"], "top": {x["token"]: x["logprob"] for x in c.get("top_logprobs", [])}}
                for c in (choice.get("logprobs") or {}).get("content") or []]
        usage = r.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        return {"text": choice["message"].get("content") or "", "tokens": toks, "prompt_tokens": usage.get("prompt_tokens"),
                "cached": cached, "elapsed_s": round(time.monotonic() - t0, 2)}

    def metrics(self) -> dict[str, float]:
        if self.url is None:
            return {}
        with urllib.request.urlopen(self.url + "/metrics", timeout=30) as r:
            out: dict[str, float] = {}
            for line in r.read().decode().splitlines():
                if line.startswith(("vllm:num_preemptions", "vllm:prefix_cache_queries", "vllm:prefix_cache_hits",
                                    "vllm:kv_cache_usage_perc")) and not line.startswith("#"):
                    name = line.split("{")[0].split(" ")[0]
                    out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            return out


def compare(a: dict, b: dict) -> dict:
    """First generated position where a and b differ (None = identical), the largest chosen-token logprob gap over the
    common prefix, and the largest gap among tokens both top-20 lists contain."""
    ta, tb = a["tokens"], b["tokens"]
    first = None
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x["t"] != y["t"]:
            first = i
            break
    if first is None and len(ta) != len(tb):
        first = min(len(ta), len(tb))
    upto = len(ta) if first is None else first
    lp = max((abs(ta[i]["lp"] - tb[i]["lp"]) for i in range(upto)), default=0.0)
    top = max((abs(ta[i]["top"][k] - tb[i]["top"][k]) for i in range(upto) for k in ta[i]["top"].keys() & tb[i]["top"].keys()),
              default=0.0)
    return {"first_divergence": first, "max_chosen_lp_delta": round(lp, 6), "max_top20_lp_delta": round(top, 6),
            "text_identical": a["text"] == b["text"]}


def within_floor(d: dict, floor: dict) -> bool:
    if d["first_divergence"] is None:
        return True
    if floor["first_divergence"] is None:
        return False
    return d["first_divergence"] >= floor["first_divergence"] and d["max_top20_lp_delta"] <= max(floor["max_top20_lp_delta"], 1e-3) * 1.5


def expected_hit(prompt_tokens: int, block: int, common: int | None = None) -> int:
    limit = prompt_tokens - 1 if common is None else min(common, prompt_tokens - 1)
    return limit // block * block


def first_salt(*runs: dict) -> bool:
    """Each run was its salt's first request (cold, fresh-salt control, ABA's first A): cached_tokens must be 0."""
    return all(r["cached"] == 0 for r in runs)


def verdict(cached_ok: bool, d: dict, floor: dict) -> str:
    if not cached_ok:
        return "FAIL"
    return "PASS" if within_floor(d, floor) else "FLAG"


def self_test() -> None:
    tok = lambda seq: {"text": "".join(seq), "tokens": [{"t": s, "lp": -0.1, "top": {s: -0.1, "x": -2.0}} for s in seq]}  # noqa: E731
    a, b, c = tok("abcd"), tok("abcd"), tok("abxd")
    assert compare(a, b)["first_divergence"] is None and within_floor(compare(a, b), compare(a, c))
    assert compare(a, c)["first_divergence"] == 2
    assert not within_floor(compare(a, c), compare(a, b))                # diverges where the floor did not
    assert within_floor(compare(a, c), compare(a, tok("abyd")))           # same point as the floor
    assert expected_hit(36016, 384) == 35712 and expected_hit(36016, 128) == 35968
    assert expected_hit(250010, 384, common=200050) == 199680
    assert verdict(False, compare(a, b), compare(a, b)) == "FAIL"
    assert first_salt({"cached": 0}, {"cached": 0})
    assert not first_salt({"cached": 0}, {"cached": 384}) and not first_salt({"cached": None})   # a control that hit
    print("self-test: compare / floor / expected-hit / verdict / fresh-salt control OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--n", type=int, nargs="*", default=[])
    ap.add_argument("--partial", default=None, help="N:K, e.g. 250000:200000")
    ap.add_argument("--order", choices=["identity", "ABA"], default="identity")
    ap.add_argument("--block", type=int, default=384, help="hit granularity: 384 packed (0060), 128 v5 layout")
    ap.add_argument("--fillers", type=int, default=1, help="ABA: prompts sent between the two A requests")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2210)
    ap.add_argument("--data-dir", type=Path, default=Path("/opt/dsv41/data/bench-data"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    self_test()
    if args.dry_run:
        c = Client(None, None, args.data_dir)
        for n in args.n:
            t = c.text(n, args.seed + n)
            print(f"dry run: built {c.count(t + QUESTION):,} estimated tokens for n={n:,}")
        return
    if not args.url:
        ap.error("--url is required unless --dry-run")
    url = args.url.rstrip("/")
    model = args.model
    if not model:
        with urllib.request.urlopen(url + "/v1/models", timeout=30) as r:
            model = json.load(r)["data"][0]["id"]
    c = Client(url, model, args.data_dir)
    nonce = time.strftime("%Y%m%dT%H%M%S")
    rows, failed = [], False
    m0 = c.metrics()

    def emit(row: dict) -> None:
        nonlocal failed
        failed |= row["verdict"] == "FAIL"
        rows.append(row)
        print(f"{row['kind']:>9} n={row.get('n', '-')!s:>8}: {row['verdict']}  cached={row.get('cached')} "
              f"(expected {row.get('expected_cached')})  control cached={row.get('control_cached')}  "
              f"hit-vs-cold {row['vs_cold']}  floor {row['floor']}", flush=True)

    if args.order == "identity":
        for n in args.n:
            p = c.text(n - c.count(QUESTION), args.seed + n) + QUESTION
            salt = f"k1-prefix-{n}-{nonce}"
            cold = c.ask(p, salt, args.max_tokens)
            hit = c.ask(p, salt, args.max_tokens)
            ctl = c.ask(p, f"{salt}-control", args.max_tokens)
            exp = expected_hit(hit["prompt_tokens"], args.block)
            ok = first_salt(cold, ctl) and hit["cached"] == exp
            d, fl = compare(cold, hit), compare(cold, ctl)
            emit({"kind": "identity", "n": n, "prompt_tokens": hit["prompt_tokens"], "cold_cached": cold["cached"],
                  "cached": hit["cached"], "expected_cached": exp, "control_cached": ctl["cached"], "vs_cold": d,
                  "floor": fl, "verdict": verdict(ok, d, fl),
                  "elapsed": [cold["elapsed_s"], hit["elapsed_s"], ctl["elapsed_s"]], "texts": [cold["text"][:300], hit["text"][:300]]})
        if args.partial:
            N, K = (int(x) for x in args.partial.split(":"))
            prefix = c.text(K, args.seed + 7 * K)
            tail_n = N - K - c.count(QUESTION)
            p1 = prefix + "\n\n" + c.text(tail_n, args.seed + 11) + QUESTION
            p2 = prefix + "\n\n" + c.text(tail_n, args.seed + 13) + QUESTION
            salt = f"k1-partial-{N}-{K}-{nonce}"
            c.ask(p1, salt, args.max_tokens)                               # fills the cache with P1
            part = c.ask(p2, salt, args.max_tokens)                        # partial hit on the shared prefix
            ctl = c.ask(p2, f"{salt}-control", args.max_tokens)
            ctl2 = c.ask(p2, f"{salt}-control2", args.max_tokens)
            t1, t2 = c.chat_tokens(p1), c.chat_tokens(p2)
            common = None
            if t1 is not None and t2 is not None:
                common = next((i for i, (x, y) in enumerate(zip(t1, t2)) if x != y), min(len(t1), len(t2)))
            if common is not None:
                exp = expected_hit(part["prompt_tokens"], args.block, common)
                ok = part["cached"] == exp
            else:   # /tokenize without chat support: bounds only
                exp = None
                ok = part["cached"] is not None and part["cached"] % args.block == 0 and K - 2 * args.block <= part["cached"] <= K + args.block
            ok = ok and first_salt(ctl, ctl2)
            d, fl = compare(ctl, part), compare(ctl, ctl2)
            emit({"kind": "partial", "n": N, "shared": K, "common_tokens": common, "cached": part["cached"],
                  "expected_cached": exp, "control_cached": [ctl["cached"], ctl2["cached"]], "vs_cold": d, "floor": fl,
                  "verdict": verdict(ok, d, fl)})
    else:  # ABA under pool pressure
        for n in args.n:
            a = c.text(n - c.count(QUESTION), args.seed + n) + QUESTION
            sa = f"k1-aba-A-{n}-{nonce}"
            a1 = c.ask(a, sa, args.max_tokens)
            for k in range(args.fillers):
                filler = c.text(n - c.count(QUESTION), args.seed + n + 1 + k) + QUESTION
                c.ask(filler, f"k1-aba-F{k}-{n}-{nonce}", args.max_tokens)
            a2 = c.ask(a, sa, args.max_tokens)
            a3 = c.ask(a, f"{sa}-control", args.max_tokens)
            ok = first_salt(a1, a3) and a2["cached"] is not None and a2["cached"] % args.block == 0
            d, fl = compare(a1, a2), compare(a1, a3)
            emit({"kind": "ABA", "n": n, "fillers": args.fillers, "evicted": a2["cached"] < expected_hit(a2["prompt_tokens"], args.block),
                  "cached": a2["cached"], "expected_cached": f"<= {expected_hit(a2['prompt_tokens'], args.block)} (less = evicted)",
                  "first_cached": a1["cached"], "control_cached": a3["cached"], "vs_cold": d, "floor": fl,
                  "verdict": verdict(ok, d, fl)})
    m1 = c.metrics()
    delta = {k: round(m1.get(k, 0.0) - m0.get(k, 0.0), 3) for k in sorted(set(m0) | set(m1))}
    print("metrics delta:", json.dumps(delta))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as f:
            for row in rows:
                f.write(json.dumps({**row, "metrics_delta": delta, "block": args.block,
                                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
