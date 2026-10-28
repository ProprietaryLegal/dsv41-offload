"""Coordinator-only bounded profiling traffic; --execute required for any network.

Reuses the existing bench2 prompt construction, sampler settings and stream accounting.
No generated text is written. Profile all TP ranks via vLLM's existing API; the local_rank
field is not a global-rank filter. B4 decode capture begins after ALL streams emit text.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Protocol, cast
from urllib.parse import urlparse

from offload.gb10_budget.evidence import JSON, Object


class Bench(Protocol):
    URL: str
    MODEL: str
    DATA_DIR: Path

    def models(self) -> Object: ...
    def scrape(self, url: str | None = None) -> str: ...
    def parse_metrics(self, text: str) -> dict[str, float]: ...
    def require_idle(self, values: dict[str, float]) -> None: ...
    def make_prompt(self, ctx: int, workload: str, filler: str, seed: int,
                    template: Object) -> tuple[str, int]: ...
    def request_body(self, prompt: str, args: argparse.Namespace, salt: str) -> Object: ...
    def stream_request(self, body: Object, started: threading.Event | None = None) -> Object: ...


def load_bench() -> Bench:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
    return cast(Bench, importlib.import_module("bench2"))


def control(url: str, verb: str) -> None:
    req = urllib.request.Request(url.rstrip("/") + "/" + verb, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"{verb} returned HTTP {response.status}")


def run(bench: Bench, args: argparse.Namespace,
        command: Callable[[str, str], None] = control) -> Object:
    bench.URL, bench.DATA_DIR = args.url, args.data_dir
    bench.require_idle(bench.parse_metrics(bench.scrape()))
    info = bench.models()
    models = info.get("data")
    if not isinstance(models, list) or not models or not isinstance(models[0], dict):
        raise ValueError("No served model in /v1/models")
    bench.MODEL = str(models[0]["id"])
    cap = models[0].get("max_model_len")
    if not isinstance(cap, int):
        raise ValueError("Missing served max_model_len")
    template: Object = {"thinking": True, "reasoning_effort": "high"}
    workload = "synth" if args.ctx == 150000 else "summary"
    prompts = [bench.make_prompt(args.ctx, workload, "real", args.seed + i, template)
               for i in range(args.batch)]
    if any(n + args.max_tokens > cap for _, n in prompts):
        raise ValueError("Occupied prompt + generation exceeds live model cap")
    bench.require_idle(bench.parse_metrics(bench.scrape()))
    starts = [threading.Event() for _ in prompts]
    barrier = threading.Barrier(args.batch)
    armed = False
    mark = time.monotonic()

    def request(i: int) -> Object:
        body = bench.request_body(prompts[i][0], args, uuid.uuid4().hex)
        barrier.wait(timeout=30)
        result = bench.stream_request(body, starts[i])
        result.pop("reasoning_text", None)
        result.pop("answer_text", None)
        return result

    prefill_delay = float(getattr(args, "prefill_delay", 0))
    if args.phase == "prefill" and prefill_delay == 0:
        command(args.url, "start_profile")
        armed = True
    with ThreadPoolExecutor(max_workers=args.batch) as pool:
        futures = [pool.submit(request, i) for i in range(args.batch)]
        try:
            if args.phase == "prefill" and prefill_delay > 0:
                time.sleep(prefill_delay)
                for future in futures:
                    if future.done():
                        future.result()
                        raise RuntimeError("Stream ended before delayed prefill capture")
                if any(event.is_set() for event in starts):
                    raise RuntimeError("Decode already started; delayed prefill capture would be mislabeled")
                command(args.url, "start_profile")
                armed = True
            if args.phase == "decode":
                deadline = time.monotonic() + args.ttft_timeout
                while not all(event.is_set() for event in starts):
                    for future in futures:
                        if future.done():
                            future.result()  # Propagate a stream failure instead of timing out.
                            raise RuntimeError("Stream ended before all streams reached decode")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Not all streams reached decode before TTFT deadline")
                    time.sleep(0.02)
                command(args.url, "start_profile")
                armed = True
            if armed:
                time.sleep(args.seconds)
        finally:
            if armed:
                command(args.url, "stop_profile")
        results: list[JSON] = [future.result() for future in futures]
    return {"status": "MEASURED profile-affected client observations, not promotion timings",
            "phase": args.phase, "context": args.ctx, "batch": args.batch,
            "prefill_delay_s": prefill_delay, "workload": workload,
            "occupied_prompt_tokens": [n for _, n in prompts],
            "elapsed_s": time.monotonic() - mark, "requests": results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:18001")
    ap.add_argument("--data-dir", type=Path, default=Path("/opt/dsv41/data/bench-data"))
    ap.add_argument("--ctx", type=int, choices=(32000, 150000), default=32000)
    ap.add_argument("--batch", type=int, choices=(1, 4), default=1)
    ap.add_argument("--phase", choices=("none", "prefill", "decode"), default="decode")
    ap.add_argument("--seconds", type=float, default=3)
    ap.add_argument("--prefill-delay", type=float, default=0, help="Capture later chunks; fail if decode already started")
    ap.add_argument("--ttft-timeout", type=float, default=600)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 0 < args.seconds <= 30 or args.ttft_timeout <= 0 or args.max_tokens < 1:
        ap.error("Require seconds in (0,30], positive timeout and positive generation budget")
    if not 0 <= args.prefill_delay <= 60 or (args.prefill_delay and args.phase != "prefill"):
        ap.error("Prefill delay must be 0..60 seconds and only used with --phase prefill")
    if urlparse(args.url).hostname not in ("127.0.0.1", "localhost") or urlparse(args.url).port != 18001:
        ap.error("Use loopback SSH forwarding 18001 -> reserved experiment port 8001")
    if not args.execute:
        print("Preview only: no network. Add --execute in the coordinator's reserved window.")
        return
    if args.output.exists():
        ap.error("Output exists; use a fresh artifact path")
    for filename in ("opinions_cl.jsonl", "filings_recap.jsonl"):
        if not (args.data_dir / filename).is_file():
            ap.error(f"Missing existing benchmark corpus: {filename}")
    args.seed, args.priority, args.temperature, args.top_p = 42, 10, 1, 1
    args.thinking, args.reasoning_effort = True, "high"
    result = run(load_bench(), args)
    with args.output.open("x") as target:
        target.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote aggregate profile observations: {args.output}")


if __name__ == "__main__":
    main()
