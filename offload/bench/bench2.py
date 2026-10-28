#!/usr/bin/env python3
"""Extended DSv4.1 TP2 harness, copied from deployment bench.py.

Loads its workload text from DSV41_BENCH_SPEC_FILE (not shipped) and keeps the public
CourtListener/RECAP prompt builders.
Adds channel-aware SSE timing, exact occupied chat counts, run-wide speculative
counters, concurrent/mixed requests and bounded telemetry. Coordinator alone runs
benchmarks; --dry-run checks connectivity without inference or tokenization.
See README.md for accounting limits and reproducible commands.
"""
import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import TypeAlias, TypedDict, cast

from telemetry import Telemetry

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
Object: TypeAlias = dict[str, JSON]

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


class BenchSpec(TypedDict):
    workloads: dict[str, str]
    first: list[str]
    last: list[str]
    places: list[str]
    verbs: list[str]
    claims: list[str]
    paragraph: str


_SPEC: BenchSpec | None = None


def spec() -> BenchSpec:
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
        _SPEC = cast(BenchSpec, loaded)
    return _SPEC


def record_paragraph(rng: random.Random, i: int) -> str:
    s = spec()
    d = f"20{rng.randint(22, 25)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    a, b = f"{rng.choice(s['first'])} {rng.choice(s['last'])}", f"{rng.choice(s['first'])} {rng.choice(s['last'])}"
    c = rng.choice(s["claims"]).format(n=rng.randint(1, 9), m=rng.randint(200, 25000))
    return s["paragraph"].format(i=i, d=d, place=rng.choice(s["places"]), a=a, verb=rng.choice(s["verbs"]), claim=c,
                                 b=b, dcn=rng.randint(100000, 999999))


def tok_count(text: str) -> int:
    req = urllib.request.Request(URL + "/tokenize", data=json.dumps({"model": MODEL, "prompt": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return int(json.load(response)["count"])


_DOCS: list[dict[str, str]] | None = None
def build_real(target_tokens: int, seed: int) -> str:
    """Concatenate public CourtListener opinions + RECAP filings (data/*.jsonl) to target_tokens (exact via /tokenize)."""
    global _DOCS
    if _DOCS is None:
        _DOCS = []
        for filename in ("opinions_cl.jsonl", "filings_recap.jsonl"):
            with (DATA_DIR / filename).open() as source:
                _DOCS.extend(json.loads(line) for line in source)
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


def build_record(target_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    paras, i, approx = [], 1, 0.0
    while approx < target_tokens:
        p = record_paragraph(rng, i)
        paras.append(p); approx += len(p) / 3.6; i += 1   # ~3.6 chars/token for this text (calibrated below)
    return "\n".join(paras)


def post(path: str, body: Object, timeout: float = 3600) -> Object:
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return cast(Object, json.load(response))


def stream_request(body: Object, started: threading.Event | None = None) -> Object:
    """Count delivered tokens only. Channel tokenization is labelled, never guessed from chunks."""
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    begin = time.monotonic()
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    chunks: dict[str, list[str]] = {"reasoning": [], "answer": []}
    # Arrival time of every content-bearing SSE event: vLLM streams one event per engine step, so the gaps are a direct
    # client-side round timer (at concurrency 1 a step is one verify round plus its draft).
    event_times: list[float] = []
    usage: Object | None = None
    finish: JSON = None
    done = False
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError(f"stream error: {event['error']}")
            if event.get("usage"):
                usage = event["usage"]
            if any((c.get("delta", {}).get("reasoning_content") or c.get("delta", {}).get("reasoning")
                    or c.get("delta", {}).get("content")) for c in event.get("choices", [])):
                event_times.append(time.monotonic() - begin)
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                for kind, piece in (("reasoning", delta.get("reasoning_content") or delta.get("reasoning")),
                                    ("answer", delta.get("content"))):
                    if piece:
                        now = time.monotonic() - begin
                        first.setdefault(kind, now)
                        last[kind] = now
                        chunks[kind].append(piece)
                        if started is not None:
                            started.set()
                finish = choice.get("finish_reason") or finish
    elapsed = time.monotonic() - begin  # excludes post-hoc /tokenize calls
    if started is not None:
        started.clear()  # mixed prefill requires streams still generating, not retokenizing
    if not done or usage is None or finish is None:
        raise RuntimeError(f"incomplete stream: done={done}, usage={usage}, finish={finish}")
    result: Object = {"usage": usage, "finish": finish, "total_s": elapsed,
                      "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
                      "ttft_s": min(first.values()) if first else None,
                      "first_visible_s": first.get("answer"), "first_by_kind_s": cast(Object, first),
                      "last_by_kind_s": cast(Object, last)}
    details = cast(Object, usage.get("completion_tokens_details") or {})
    server_reasoning = details.get("reasoning_tokens")
    for kind, pieces in chunks.items():
        text = "".join(pieces)
        # Some servers report reasoning_tokens explicitly. Otherwise exact tokenization of
        # emitted text is useful, but is NOT the original generation token sequence.
        count = (int(cast(int, server_reasoning)) if kind == "reasoning" and server_reasoning is not None
                 else tok_count(text) if text else 0)
        source = "server_usage" if kind == "reasoning" and server_reasoning is not None else "retokenized_text"
        if kind == "answer" and server_reasoning is not None:
            count = int(cast(int, usage["completion_tokens"])) - int(cast(int, server_reasoning))
            source = "server_usage_difference_includes_control_tokens"
        span = last.get(kind, 0) - first.get(kind, 0)
        result[kind + "_tokens"] = count
        result[kind + "_token_source"] = source
        result[kind + "_decode_tps"] = (count - 1) / span if span > 0 and count > 1 else None
        result[kind + "_text"] = text
    gaps = sorted(1000.0 * (b - a) for a, b in zip(event_times, event_times[1:]))
    if gaps:
        def pct(q: float) -> float:
            return round(gaps[min(len(gaps) - 1, int(q * len(gaps)))], 3)
        result["step_gap_ms"] = {"n": len(gaps), "mean": round(sum(gaps) / len(gaps), 3), "p10": pct(0.10),
                                 "p50": pct(0.50), "p90": pct(0.90), "max": round(gaps[-1], 3)}
        result["step_events"] = len(event_times)
        result["step_times_ms"] = [round(1000.0 * t, 1) for t in event_times]
    all_span = max(last.values()) - min(first.values()) if first else 0
    result["decode_tps"] = (int(cast(int, usage["completion_tokens"])) - 1) / all_span if all_span > 0 else None
    return result

MetricMap: TypeAlias = dict[str, float]


def scrape(url: str | None = None) -> str:
    with urllib.request.urlopen(url or URL + "/metrics", timeout=10) as response:
        return str(response.read().decode())


def parse_metrics(text: str) -> MetricMap:
    values: MetricMap = {}
    for line in text.splitlines():
        match = re.fullmatch(r'([^#\s{]+)(\{.*\})?\s+([-+0-9.eE]+)(?:\s+\d+)?', line)
        if match:
            values[match[1] + (match[2] or "")] = float(match[3])
    return values


def metric_sum(values: MetricMap, name: str) -> float | None:
    found = [v for k, v in values.items() if k.split("{")[0] == name]
    return sum(found) if found else None


def require_idle(values: MetricMap) -> None:
    for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        count = metric_sum(values, name)
        if count is None or count != 0:
            raise RuntimeError(f"serve is not verified idle: {name}={count}")


def spec_delta(before: MetricMap, after: MetricMap) -> Object:
    keys = {k for k in before | after if k.startswith("vllm:spec_decode_") and k.split("{")[0].endswith("_total")}
    if any(k not in after or after[k] < before.get(k, 0) for k in keys):
        raise RuntimeError("speculative counters reset/disappeared during run; comparison invalid")
    # A per-position series may be lazily created on its first use.
    delta = {k: after[k] - before.get(k, 0) for k in keys}
    accepted = metric_sum(delta, "vllm:spec_decode_num_accepted_tokens_total")
    proposed = metric_sum(delta, "vllm:spec_decode_num_draft_tokens_total")
    rounds = metric_sum(delta, "vllm:spec_decode_num_drafts_total")
    positions: dict[str, float] = {}
    for key, value in delta.items():
        if key.split("{")[0] == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            match = re.search(r'(?:\{|,)\s*position="([^"]+)"', key)
            if match:
                positions[match[1]] = positions.get(match[1], 0) + value
    return {"proposed_tokens": proposed, "accepted_tokens": accepted, "rounds": rounds,
            "mean_accepted_per_round": accepted / rounds if rounds and accepted is not None else None,
            "acceptance_fraction": accepted / proposed if proposed and accepted is not None else None,
            "accepted_per_position": cast(Object, positions),
            "acceptance_per_position_per_round": {k: v / rounds if rounds else None for k, v in positions.items()},
            "raw_delta": cast(Object, delta), "scope": "server_global_run_window"}

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def models() -> Object:
    with urllib.request.urlopen(URL + "/v1/models", timeout=10) as response:
        return cast(Object, json.load(response))


def prompt_tokens(prompt: str, template: Object) -> int:
    result = post("/tokenize", {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                                "add_generation_prompt": True, "chat_template_kwargs": template})
    return int(cast(int, result["count"]))


def make_prompt(ctx: int, workload: str, filler: str, seed: int, template: Object) -> tuple[str, int]:
    """Measure complete occupied chat prompt, including template and legal instruction."""
    instruction = spec()["workloads"][workload]
    if not ctx:
        return instruction, prompt_tokens(instruction, template)
    # 4,096 tokens of slack: build_real's trim loop can undershoot its target by ~2K at 150K, and the search below only
    # shortens. The material is the same seeded document order either way, so shorter contexts keep identical prompts.
    rec = build_real(ctx + 4096, seed) if filler == "real" else build_record(ctx + 4096, seed)
    title = "MATERIALS (public court documents)" if filler == "real" else "CASE RECORD (fictional)"
    def wrap(length: int) -> str:
        return f"{title}:\n{rec[:length]}\n\nEND OF MATERIALS.\n\n{instruction}"
    lo, hi = 0, len(rec)
    best, count = wrap(0), prompt_tokens(wrap(0), template)
    if count > ctx:
        raise ValueError(f"requested context {ctx} is smaller than instruction/template ({count})")
    while lo <= hi:
        middle = (lo + hi) // 2
        candidate = wrap(middle)
        n = prompt_tokens(candidate, template)
        if n <= ctx:
            best, count, lo = candidate, n, middle + 1
        else:
            hi = middle - 1
    if ctx - count > 16:
        raise ValueError(f"corpus cannot fill requested context {ctx}; measured {count}; supply more real documents")
    return best, count


def request_body(prompt: str, args: argparse.Namespace, salt: str) -> Object:
    return {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens,
            "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
            "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": False},
            "chat_template_kwargs": {"thinking": args.thinking, "reasoning_effort": args.reasoning_effort},
            "priority": args.priority, "cache_salt": salt}


def settings(args: argparse.Namespace, model_info: Object) -> Object:
    return {"config_label": args.label, "url": URL, "model": MODEL, "models_response": model_info,
            "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
            "chat_template_kwargs": {"thinking": args.thinking, "reasoning_effort": args.reasoning_effort},
            "max_tokens": args.max_tokens, "reserved_output": args.max_tokens, "priority": args.priority,
            "max_model_len": args.max_model_len,
            "cache_mode": args.cache_mode, "filler": args.filler, "data_dir": str(DATA_DIR)}


def write_row(path: Path, row: Object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(json.dumps(row, allow_nan=False) + "\n")
        output.flush()


def run_group(args: argparse.Namespace, config: Object, ctx: int, workload: str, concurrency: int,
              rep: int, prompts: list[tuple[str, int]], salts: list[str], mixed: bool = False) -> Object:
    from concurrent.futures import ThreadPoolExecutor
    run_id = uuid.uuid4().hex
    before = parse_metrics(scrape())
    require_idle(before)
    count = 3 if mixed else concurrency
    starts = [threading.Event() for _ in range(count)]
    barrier = threading.Barrier(2 if mixed else count)
    begin = time.monotonic()
    common: Object = {"run_id": run_id, "config_label": args.label, "target_ctx": ctx, "workload": workload,
                      "concurrency": count, "mixed": mixed, "rep": rep, "settings": config,
                      "cache_state": "cold" if args.cache_mode == "cold" or rep == 0 else "warm_repeat",
                      "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    def request(index: int, synchronize: bool = True) -> Object:
        if synchronize:
            barrier.wait(timeout=30)
        offset = time.monotonic() - begin
        prompt, occupied = prompts[index]
        body = request_body(prompt, args, salts[index])
        try:
            result = stream_request(body, starts[index])
            error = None
        except Exception as exc:  # noqa: BLE001 -- request boundary records error and fails run
            result, error = {}, f"{type(exc).__name__}: {exc}"
        if not args.keep_text:
            result.pop("answer_text", None)
            result.pop("reasoning_text", None)
        return {**common, "type": "request", "request_index": index, "start_offset_s": offset,
                "occupied_prompt_tokens": occupied, "reserved_output": args.max_tokens,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "cache_salt": salts[index], "error": error, **result}
    sampler = Telemetry(args.spark_hosts, args.x86host_host, args.process_name, args.extra_metrics, args.x86host_nic)
    sampler.start()
    rows: list[Object] = []
    schedule_error: str | None = None
    try:
        with ThreadPoolExecutor(max_workers=count) as pool:
            futures = [pool.submit(request, i) for i in range(2 if mixed else count)]
            if mixed:
                deadline = time.monotonic() + args.mixed_timeout
                while not all(event.is_set() for event in starts[:2]):
                    if any(f.done() for f in futures) or time.monotonic() >= deadline:
                        schedule_error = "mixed prefill not submitted: two active decoding streams not observed"
                        break
                    time.sleep(0.02)
                if schedule_error is None:
                    time.sleep(args.mixed_delay)
                    if any(f.done() for f in futures) or not all(e.is_set() for e in starts[:2]):
                        schedule_error = "mixed prefill not submitted: a decoding stream already finished"
                    else:
                        futures.append(pool.submit(request, 2, False))
            for future in futures:
                row = future.result()
                rows.append(row)
                write_row(Path(args.out), row)
        client_elapsed = time.monotonic() - begin
        elapsed = max((float(cast(float, r["start_offset_s"])) + float(cast(float, r.get("total_s") or 0)) for r in rows), default=client_elapsed)
    finally:
        sampler.finish()
    # vLLM counters are periodically published; this delay is excluded from throughput.
    time.sleep(args.metrics_settle)
    metric_error: str | None = None
    after: MetricMap = {}
    try:
        after = parse_metrics(scrape())
        speculation = spec_delta(before, after)
    except Exception as exc:  # noqa: BLE001 -- request boundary records error and fails run
        speculation = {"error": str(exc)}
        metric_error = f"metrics accounting failed: {type(exc).__name__}: {exc}"
    delivered = sum(int(cast(int, r.get("completion_tokens") or 0)) for r in rows)
    error = schedule_error or metric_error or ("one or more requests failed" if any(r["error"] for r in rows) else None)
    summary: Object = {**common, "type": "run", "error": error, "elapsed_s": elapsed,
                       "delivered_tokens": delivered, "aggregate_tps": delivered / elapsed,
                       "client_elapsed_s": client_elapsed,
                       "spec": speculation, "telemetry": sampler.samples,
                       "telemetry_errors": sampler.errors,
                       "metrics_before": cast(Object, before), "metrics_after": cast(Object, after),
                       "metrics_settle_s": args.metrics_settle,
                       "request_count": len(rows)}
    write_row(Path(args.out), summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("telemetry", "metrics_before", "metrics_after")} ), flush=True)
    if error:
        raise RuntimeError(error)
    return summary


def main() -> None:
    global URL, MODEL, DATA_DIR
    ap = argparse.ArgumentParser(description="Coordinator benchmark harness; --dry-run never generates tokens")
    ap.add_argument("--url", default="http://HEAD_IP:8000")
    ap.add_argument("--model", help="served model id; default is first /v1/models id")
    ap.add_argument("--label", default="A")
    ap.add_argument("--ctx", type=int, nargs="+", default=[0])
    ap.add_argument("--workload", nargs="+", choices=sorted(WORKLOAD_NAMES), default=["legal"])
    ap.add_argument("--concurrency", type=int, nargs="+", choices=[1, 2, 3, 4], default=[1])
    ap.add_argument("--mixed", action="store_true", help="also run two decoders with a long-prefill arrival")
    ap.add_argument("--mixed-short-ctx", type=int, default=0)
    ap.add_argument("--mixed-delay", type=float, default=1)
    ap.add_argument("--mixed-timeout", type=float, default=600)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--max-model-len", type=int, default=327680, help="also capped by live /v1/models limit")
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--temperature", type=float, default=1)
    ap.add_argument("--top-p", type=float, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--priority", type=int, default=10)
    ap.add_argument("--cache-mode", choices=["cold", "warm"], default="cold")
    ap.add_argument("--filler", choices=["real", "synthetic"], default="real")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", default="offload/results/bench2.jsonl")
    ap.add_argument("--keep-text", action="store_true")
    ap.add_argument("--spark-hosts", nargs=2, default=["sparkhead", "sparkworker"])
    ap.add_argument("--x86host-host", default="x86host")
    ap.add_argument("--process-name", action="append", default=[], help="exact /proc comm or executable basename")
    ap.add_argument("--x86host-nic", nargs=2, default=["enp130s0f0np0", "enp130s0f1np1"])
    ap.add_argument("--extra-metrics", action="append", default=[])
    ap.add_argument("--metrics-settle", type=float, default=2, help="counter publication wait after each group")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.reps < 1 or args.max_tokens < 1 or min(args.ctx) < 0 or args.mixed_short_ctx < 0:
        ap.error("reps/output must be positive; context must be nonnegative")
    if min(args.mixed_delay, args.metrics_settle) < 0 or args.mixed_timeout <= 0:
        ap.error("timing options must be nonnegative; mixed-timeout must be positive")
    URL, DATA_DIR = args.url.rstrip("/"), args.data_dir.resolve()
    info = models()
    records = cast(list[Object], info["data"])
    if not records:
        raise RuntimeError("/v1/models returned no models")
    MODEL = args.model or str(records[0]["id"])
    if not any(record["id"] == MODEL for record in records):
        raise ValueError(f"model {MODEL!r} absent in /v1/models")
    selected = next(record for record in records if record["id"] == MODEL)
    args.max_model_len = min(args.max_model_len, int(cast(int, selected.get("max_model_len") or args.max_model_len)))
    initial = parse_metrics(scrape())
    config = settings(args, info)
    if args.dry_run:
        # Connectivity reads only: no /tokenize, prompt construction or chat completions.
        sampler = Telemetry(args.spark_hosts, args.x86host_host, args.process_name, args.extra_metrics, args.x86host_nic)
        sampler.sample()
        row: Object = {"type": "dry_run", "settings": config, "metrics": cast(Object, initial),
                       "telemetry": sampler.samples, "telemetry_errors": sampler.errors,
                       "generation_requests": 0}
        write_row(Path(args.out), row)
        print(json.dumps(row), flush=True)
        if sampler.errors:
            raise RuntimeError("dry-run telemetry failed; see recorded errors")
        return
    require_idle(initial)
    template: Object = {"thinking": args.thinking, "reasoning_effort": args.reasoning_effort}
    for ctx in args.ctx:
        for workload in args.workload:
            for mixed, concurrency in [(False, c) for c in args.concurrency] + ([(True, 3)] if args.mixed else []):
                sizes = [args.mixed_short_ctx, args.mixed_short_ctx, ctx] if mixed else [ctx] * concurrency
                # Same prompts across repetitions/variants. Cold changes only salt; warm primes on rep zero.
                prompts = [make_prompt(size, workload, args.filler, args.seed + i, template) for i, size in enumerate(sizes)]
                for _, occupied in prompts:
                    if occupied + args.max_tokens > args.max_model_len:
                        raise ValueError(f"occupied {occupied} + output {args.max_tokens} exceeds {args.max_model_len}")
                warm_salts = [uuid.uuid4().hex for _ in sizes]
                for rep in range(args.reps):
                    salts = warm_salts if args.cache_mode == "warm" else [uuid.uuid4().hex for _ in sizes]
                    run_group(args, config, ctx, workload, concurrency, rep, prompts, salts, mixed)


if __name__ == "__main__":
    main()
