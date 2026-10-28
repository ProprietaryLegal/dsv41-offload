#!/usr/bin/env python3
"""AUD1 item 2: does production reuse a resident ~150K-token document prefix across two questions?

Dry by default: builds the document from the public bench corpus, prints the plan (sizes, hashes, request shape), and
sends nothing. With --execute --base-url URL --out FILE it sends exactly two /v1/chat/completions requests, one after
the other: the same document with two different short questions, small max_tokens, one fresh shared cache_salt (so the
first request is cold whatever ran before, and the second may reuse the first). Before each request it reads /metrics
and refuses to send unless the server is idle (vllm:num_requests_running and _waiting both 0, COMMON rule 2). It reads
/metrics again after each request. Only GET /metrics and the two POSTs reach the server: no /tokenize, no /v1/models.

Recorded (numbers and hashes only, never prompt or generated text): prompt/cached/completion tokens from usage, client
TTFT and total time, /metrics deltas (prefix_cache_queries/hits, prompt_tokens_cached, external prefix cache,
preemptions, server TTFT sum/count, prefill KV-computed tokens), sha256 of the corpus file, the document, each prompt
and each generated text.

Verdict (see offload/audit/PREFIX-REUSE.md): RESIDENT_REUSE_WORKS when the second request's cached_tokens covers the
shared document to within one 384-token scheduler block plus the question allowance, the /metrics hit delta agrees,
and its TTFT is a small fraction of the first; RESIDENT_REUSE_FAILS when it caches less than half; otherwise PARTIAL.
CONTAMINATED when other traffic changed the counters, NOT_COLD when the first request already hit.

  python3 offload/audit/prefix_reuse_probe.py                                    # dry plan, no network
  python3 offload/audit/prefix_reuse_probe.py --execute --base-url http://HEAD_IP:8000 --out FILE.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DATA = Path("/opt/dsv41/data/bench-data/opinions_cl.jsonl")
# bench-data/SHA256SUMS (the same file as sparkhead's ~/dsv41-deploy/data/opinions_cl.jsonl)
CORPUS_SHA256 = "80461203f88a6b837eed8fdf9b7e77a40660acb36dfa9887e00a0ae37483b6a1"
SERVED = "DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2"
TARGET_TOKENS = 150_000
CHARS_PER_TOKEN = 3.5  # planning ratio only (bench.py's first guess is 3.3); usage reports the real count
MAX_CONTEXT = 327_680  # production cap (21bed31 PROD_MAX_LEN)
SCHED_BLOCK = 384  # K1 scheduler block: hits land on multiples of it (S8 N-4, measured S18)
REPLAY_WINDOW = 128  # SWA bounded replay recomputes the hit's last 128 tokens (scheduler._mark_prefix_replay)
QUESTION_ALLOWANCE = 96  # tokens of question + chat suffix + BPE boundary that are not shared
SEPARATOR = "\n\nEND OF MATERIALS.\n\n"
QUESTIONS = (
    "In two sentences, state the main holding of the first document in the materials above.",
    "Which court decided the last document in the materials above? Answer in one line.",
)
METRICS = (
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:prompt_tokens_cached",
    "vllm:prompt_tokens",
    "vllm:external_prefix_cache_queries",
    "vllm:external_prefix_cache_hits",
    "vllm:num_preemptions",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:request_prefill_kv_computed_tokens_sum",
    "vllm:request_prefill_kv_computed_tokens_count",
)
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")


class ProbeError(RuntimeError):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def build_document(path: Path, target_tokens: int, expected_sha: str | None = CORPUS_SHA256) -> str:
    """Deterministic document: corpus records in file order until the planned character budget is reached."""
    if expected_sha is not None:
        got = sha256_file(path)
        if got != expected_sha:
            raise ProbeError(f"{path}: sha256 {got} is not the pinned public corpus {expected_sha}")
    budget = int(target_tokens * CHARS_PER_TOKEN)
    parts: list[str] = []
    total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            part = f"=== {rec.get('title', '')} ({rec.get('court', '')}, {rec.get('date', '')}) ===\n{rec['text']}"
            parts.append(part)
            total += len(part) + 2
            if total >= budget:
                break
    if total < budget:
        raise ProbeError(f"{path}: corpus holds only {total} characters, need {budget}")
    doc = "\n\n".join(parts)
    cut = doc.rfind(" ", 0, budget)
    return doc[: cut if cut > budget // 2 else budget]


def request_body(document: str, question: str, salt: str, max_tokens: int, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": document + SEPARATOR + question}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
        "priority": 10,  # background class: yields its prefill if a real request arrives (patch 0031)
        "cache_salt": salt,
    }


def parse_metrics(text: str) -> dict[str, float]:
    """Sum every sample of each watched metric over its labels; counters may carry a _total suffix."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        name = name.removesuffix("_total")
        if name in METRICS or name in GAUGES:
            try:
                value = float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError) as e:
                raise ProbeError(f"unparsable metrics line for {name}") from e
            out[name] = out.get(name, 0.0) + value
    return out


@dataclass
class Client:
    base_url: str
    timeout_s: float

    def metrics(self) -> dict[str, float]:
        with urllib.request.urlopen(self.base_url + "/metrics", timeout=30) as r:
            return parse_metrics(r.read().decode("utf-8", "replace"))

    def chat_stream(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST one streaming chat request; return timings, usage and a hash of the generated text (not the text)."""
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        ttft: float | None = None
        usage: dict[str, Any] = {}
        gen = hashlib.sha256()
        chunks = 0
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = "".join(str(delta.get(k) or "") for k in ("reasoning_content", "reasoning", "content"))
                    if piece:
                        chunks += 1
                        gen.update(piece.encode("utf-8"))
                        if ttft is None:
                            ttft = time.monotonic() - t0
        total = time.monotonic() - t0
        details = usage.get("prompt_tokens_details") or {}
        return {
            "ttft_s": None if ttft is None else round(ttft, 3),
            "total_s": round(total, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "content_chunks": chunks,
            "generated_sha256": gen.hexdigest(),
        }


def _delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {k: round(after.get(k, 0.0) - before.get(k, 0.0), 6) for k in METRICS}


def _require_idle(m: dict[str, float], when: str) -> None:
    for g in GAUGES:
        if g not in m:
            raise ProbeError(f"{when}: /metrics lacks {g}; refusing to send")
        if m[g] != 0:
            raise ProbeError(f"{when}: server not idle ({g} = {m[g]:g}); nothing sent")


def verdict(a: dict[str, Any], b: dict[str, Any], da: dict[str, float], db: dict[str, float]) -> dict[str, Any]:
    pa, pb = a.get("prompt_tokens"), b.get("prompt_tokens")
    ca, cb = a.get("cached_tokens"), b.get("cached_tokens")
    if not all(isinstance(x, int) for x in (pa, pb, ca, cb)):
        return {"verdict": "INCOMPLETE", "reason": "usage lacks prompt_tokens or cached_tokens "
                "(server needs --enable-prompt-tokens-details)"}
    assert isinstance(pa, int) and isinstance(pb, int) and isinstance(ca, int) and isinstance(cb, int)
    shared_est = min(pa, pb) - QUESTION_ALLOWANCE
    expected_min = max(0, (shared_est // SCHED_BLOCK) * SCHED_BLOCK - SCHED_BLOCK)
    # queries count each admitted request's tokens once (kv_cache_manager.record_prefix_cache_stats); hits are recorded
    # at admission, before the SWA replay rewind, so they may differ from usage by at most the 128-token replay window
    hit_gap = [d["vllm:prefix_cache_hits"] - c for d, c in ((da, ca), (db, cb))]
    contaminated = (da["vllm:prefix_cache_queries"] != pa or db["vllm:prefix_cache_queries"] != pb
                    or any(abs(g) > REPLAY_WINDOW for g in hit_gap))
    ttft_a, ttft_b = a.get("ttft_s"), b.get("ttft_s")
    ttft_ratio = (round(ttft_b / ttft_a, 4) if isinstance(ttft_a, float) and isinstance(ttft_b, float)
                  and ttft_a > 0 else None)
    facts: dict[str, Any] = {
        "shared_prefix_estimate_tokens": shared_est,
        "expected_min_cached_tokens": expected_min,
        "second_cached_fraction_of_prompt": round(cb / pb, 4) if pb else None,
        "second_cached_aligned_to_384": cb % SCHED_BLOCK == 0,
        "ttft_ratio_second_over_first": ttft_ratio,
        "metrics_hits_minus_usage_cached": hit_gap,
        "metrics_agree_with_usage": not contaminated,
    }
    if contaminated:
        v, why = "CONTAMINATED", "other traffic or a mismatch changed the prefix-cache counters; rerun when idle"
    elif ca != 0:
        v, why = "NOT_COLD", "the first request already read the cache (salt collision or stale state)"
    elif cb >= expected_min and cb < pb and (ttft_ratio is not None and ttft_ratio <= 0.2):
        v, why = "RESIDENT_REUSE_WORKS", "second request reused the resident document prefix"
    elif cb < 0.5 * shared_est:
        v, why = "RESIDENT_REUSE_FAILS", "second request recomputed most of the shared document"
    else:
        v, why = "PARTIAL", "some reuse, short of the full shared prefix or without the TTFT drop"
    return {"verdict": v, "reason": why, **facts}


def plan(args: argparse.Namespace, document: str, salt: str) -> dict[str, Any]:
    bodies = [request_body(document, q, salt, args.max_tokens, args.model) for q in QUESTIONS]
    est = int(len(document) / CHARS_PER_TOKEN)
    if len(document) / 2.5 > MAX_CONTEXT * 0.92:
        raise ProbeError("document could exceed the production context cap at 2.5 chars/token; lower --target-tokens")
    return {
        "probe": "AUD1 prefix_reuse_probe v1",
        "corpus": str(args.data),
        "corpus_sha256": CORPUS_SHA256 if args.data_sha_check else "unchecked",
        "document_chars": len(document),
        "document_sha256": sha256_text(document),
        "document_tokens_estimate": est,
        "questions_sha256": [sha256_text(q) for q in QUESTIONS],
        "prompts_sha256": [sha256_text(b["messages"][0]["content"]) for b in bodies],
        "request_shape": {k: v for k, v in bodies[0].items() if k not in ("messages", "cache_salt")},
        "cache_salt_sha256": sha256_text(salt),
        "requests_to_send": 2,
        "other_requests": "GET /metrics before and after each request (4 reads)",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    document = build_document(args.data, args.target_tokens, CORPUS_SHA256 if args.data_sha_check else None)
    salt = "aud1-prefix-probe-" + secrets.token_hex(16)
    out: dict[str, Any] = {"plan": plan(args, document, salt), "executed": False}
    if not args.execute:
        return out
    client = Client(args.base_url.rstrip("/"), args.timeout_s)
    out["executed"] = True
    out["started_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results: list[dict[str, Any]] = []
    deltas: list[dict[str, float]] = []
    for i, q in enumerate(QUESTIONS):
        before = client.metrics()
        _require_idle(before, f"before request {i + 1}")
        try:
            res = client.chat_stream(request_body(document, q, salt, args.max_tokens, args.model))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            # a request was sent: keep what is known, never send the next one
            out.update(requests=results, metrics_delta=deltas, error=f"request {i + 1}: {type(e).__name__}: {e}",
                       result={"verdict": "INCOMPLETE", "reason": f"request {i + 1} failed"})
            return out
        after = client.metrics()
        results.append(res)
        deltas.append(_delta(after, before))
    out["requests"] = results
    out["metrics_delta"] = deltas
    out["result"] = verdict(results[0], results[1], deltas[0], deltas[1])
    out["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", action="store_true", help="send the two requests (default: dry plan only)")
    p.add_argument("--base-url", help="server root, e.g. http://HEAD_IP:8000 (required with --execute)")
    p.add_argument("--out", type=Path, help="JSON result file (required with --execute; never overwritten)")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--no-data-sha-check", dest="data_sha_check", action="store_false",
                   help="tests only: accept a corpus other than the pinned public file")
    p.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--model", default=SERVED)
    p.add_argument("--timeout-s", type=float, default=900.0)
    args = p.parse_args(argv)
    if args.execute and (not args.base_url or not args.out):
        p.error("--execute needs --base-url and --out")
    if args.out is not None and args.out.exists():
        p.error(f"{args.out} exists; choose a new file")
    if not 1 <= args.max_tokens <= 64:
        p.error("--max-tokens must be 1..64")
    try:
        out = run(args)
    except (ProbeError, urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"PREFIX PROBE STOP: {e}", file=sys.stderr)
        return 2
    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    if out.get("executed"):
        print(f"verdict: {out['result']['verdict']}", file=sys.stderr)
    if "error" in out:
        print(f"PREFIX PROBE STOP: {out['error']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
