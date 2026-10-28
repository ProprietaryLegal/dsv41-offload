"""Bounded, aggregate-only internal warm-up; importing this module sends nothing."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import re
from threading import Barrier
import time
from typing import Callable, cast
from urllib.request import Request, urlopen
import uuid

BUDGET = 32
SHAPES = (1, 2, 3, 4)
TIMEOUT = 60
# Total budget for the whole warm-up: it runs inside K1's 600 s start guard (last K1 start 432 s), whose timeout
# restores the pre-K1 configuration, so a slow warm-up must end as the existing start.sh WARN, never as a guard trip.
DEADLINE_S = 90.0
Reader = Callable[[str, dict[str, object] | None], str]


def obj(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return cast(dict[str, object], value)


def payload(model: str, salt: str, wave: int, index: int) -> dict[str, object]:
    # Distinct prefix per stream avoids warming only a cached suffix. Raw
    # completion bypasses chat/reasoning/tool parsers, with at most 32 tokens.
    prompt = f"Internal startup {salt} wave {wave} stream {index}.\n"
    prompt += "".join(
        f"Paragraph {i}. The system reads a numbered record and checks each entry against its stored value. "
        for i in range(240 if wave == 1 else 1)
    )
    return {"model": model, "prompt": prompt, "temperature": 0, "n": 1,
            "max_tokens": BUDGET, "min_tokens": BUDGET, "ignore_eos": True,
            "stop": [], "stop_token_ids": [], "stream": False, "priority": 20}


def summary(raw: str) -> dict[str, object]:
    data = obj(json.loads(raw))
    usage = obj(data.get("usage"))
    count = usage.get("completion_tokens")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("warm-up requires exactly one completion per request")
    choice = obj(choices[0])
    reason = choice.get("finish_reason")
    stop = choice.get("stop_reason")
    # Never print text, token IDs, arbitrary error bodies or unexpected strings.
    result: dict[str, object] = {
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": count,
        "cached_tokens": obj(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "finish_reason": reason if reason in ("length", "stop") else "unexpected",
        "stop_reason": stop if type(stop) is int or stop is None else "string-present",
        "reasoning_tokens": obj(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "parser": "raw-completion; no chat/reasoning/tool parser",
    }
    metrics = obj(data.get("metrics") or {})
    spec = obj(metrics.get("speculative_decoding") or {})
    # Installed fork may omit per-request metrics. Missing is UNKNOWN, never 0.
    result["speculative_metrics"] = {
        key: value for key, value in spec.items()
        if re.fullmatch(r"[a-z_]+", key) and type(value) in (int, float)
        and math.isfinite(cast(float, value))
    } or None
    if type(count) is not int or count != BUDGET or reason != "length" or stop is not None:
        raise ValueError("warm-up output contract failed: " + json.dumps(result, sort_keys=True))
    return result


def idle(raw: str) -> None:
    for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        values = []
        for line in raw.splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0].split("{")[0] == name:
                values.append(float(fields[1]))
        if not values or any(not math.isfinite(v) or v != 0 for v in values):
            raise ValueError(f"warm-up refused: {name} missing, invalid or nonzero")


def request_timeout(end: float) -> float:
    """Per-request socket timeout: never past the warm-up deadline."""
    left = end - time.monotonic()
    if left <= 0:
        raise ValueError(f"warm-up deadline {DEADLINE_S:.0f} s reached")
    return min(float(TIMEOUT), left)


def run(model: str, read: Reader, end: float | None = None) -> None:
    end = time.monotonic() + DEADLINE_S if end is None else end
    salt = uuid.uuid4().hex
    for size in SHAPES:
        if time.monotonic() >= end:
            raise ValueError(f"warm-up deadline {DEADLINE_S:.0f} s reached before the concurrency-{size} wave")
        idle(read("/metrics", None))
        gate = Barrier(size)

        def one(index: int) -> dict[str, object]:
            request = payload(model, salt, size, index)
            gate.wait(timeout=5)
            result = summary(read("/v1/completions", request))
            if size == 1:
                tokens = result["prompt_tokens"]
                if type(tokens) is not int or tokens <= 4096:
                    raise ValueError("warm-up did not exercise full+partial 4096-token prefill")
                cached = result["cached_tokens"]
                if cached is not None and cached != 0:
                    raise ValueError("warm-up long prefill unexpectedly reused cached tokens")
            return result

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=size) as pool:
            results = list(pool.map(one, range(size)))
        print(json.dumps({"warmup": "completed", "requested_concurrency": size,
                          "actual_graph_shapes": "requires engine/worker trace; not inferred",
                          "elapsed_s": round(time.monotonic() - start, 3),
                          "requests": results}, sort_keys=True), flush=True)
    if time.monotonic() > end:
        raise ValueError(f"warm-up finished after its {DEADLINE_S:.0f} s deadline")   # late success is a failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("invalid port")

    end = time.monotonic() + DEADLINE_S

    def read(path: str, body: dict[str, object] | None) -> str:
        request = Request(f"http://localhost:{args.port}{path}",
                          data=None if body is None else json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=request_timeout(end)) as response:
            return response.read(1 << 20).decode()

    try:
        run(args.model, read, end)
    except Exception as exc:
        # HTTP error strings contain status, not the response body/prompt.
        parser.exit(1, f"warm-up FAILED ({type(exc).__name__}): {exc}\n")


if __name__ == "__main__":
    main()
