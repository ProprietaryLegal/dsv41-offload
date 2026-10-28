#!/usr/bin/env python3
"""Run the L-SUITE against one served arm at T = 0 and keep pass/fail rows only. Dry run unless --execute.

  # plan only (no network): verifies the task file against the corpus and prints the ESTIMATED time
  python3 -m offload.lsuite.run_suite --bench-data /opt/dsv41/data/bench-data --arm X
  # one arm, for real (coordinator, inside a window for a candidate; production can be the control any time)
  python3 -m offload.lsuite.run_suite --bench-data /opt/dsv41/data/bench-data --arm X \\
      --url http://HEAD_IP:8001 --out /opt/dsv41/campaigns/DSV41-OFFLOAD/<window>/lsuite-X --execute

Requests run one at a time (B1: production kernels are not batch-invariant, so a lone request is the reproducible
shape). Idle gate (COMMON hard rule 2): before each task the runner reads the arm's /metrics and sends only when
vllm:num_requests_running + vllm:num_requests_waiting is 0, polling up to --idle-wait seconds, after which it stops
with a partial run; after each reply it reads them again and marks the row error "concurrent" when anything else is
running or waiting; a /metrics without both gauges fails loud. Each request carries a fresh cache salt (cold
prefill, no cross-task prefix reuse), thinking off, temperature 0, seed 0, neutral penalties and background
priority. The visible reply is scored in memory and discarded: `rows.jsonl` holds the outcome, a reason
code, the reply length and token counts, never text, plus two fingerprints compare.py checks: `request_sha256` (the
user message) and `body_sha256` (every other sent setting, the salt counted as its per-task policy). `run.json`
(written first) records `code_sha256` (corpus, scoring, tasks and runner sources: which scorer ran) and
`summary.json` (written last) holds counts. Exit 0 = every task scored, 1 = some request failed, a row was
concurrent or the arm stayed busy (the run is INVALID for compare.py), 2 = refused before any request.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from offload.lsuite.corpus import CORPUS_FILES, Corpus, CorpusError, sha256_text
from offload.lsuite.tasks import TASKS_FILE, Materialized, Suite, TaskError, as_list, as_obj

SERVED = "DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2"
DEFAULT_BENCH_DATA = Path(__file__).resolve().parents[2] / "data"   # sparkhead: ~/dsv41-deploy/data
# ESTIMATED time model, anchored on archive measurements (offload/results/COMPARISON.md: 150K cold TTFT 68.7-69.7 s,
# 32K 14.5-14.7 s -> ~2,170 prompt tok/s; F-METHOD section 3.1: short-b1 decode ~55 tok/s, taken as 50).
CHARS_PER_TOKEN = 3.5
PREFILL_TOKENS_PER_S = 2_170.0
DECODE_TOKENS_PER_S = 50.0
REQUEST_OVERHEAD_S = 0.5
EXPECTED_REPLY_TOKENS = {"quote": 60, "figure": 8, "citation": 45, "refusal": 6, "needle": 14}

Post = Callable[[str, dict[str, object], float], dict[str, object]]
Get = Callable[[str, float], dict[str, object]]
Metrics = Callable[[str, float], str]
LOAD_GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")


class RunError(Exception):
    """Refused before any request (bad output directory, unknown model, unverifiable task file)."""


def http_post(url: str, body: dict[str, object], timeout: float) -> dict[str, object]:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return as_obj(json.load(response), "response")


def http_get(url: str, timeout: float) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return as_obj(json.load(response), "response")


def http_metrics(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return str(response.read().decode("utf-8"))


def load_on(metrics: Metrics, url: str, timeout: float) -> int:
    """Requests running + waiting on the arm (vLLM /metrics). COMMON rule 2: nothing is sent while this is not 0."""
    total, seen = 0, set()
    for line in metrics(f"{url}/metrics", timeout).splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in LOAD_GAUGES:
            total += int(float(line.rsplit(" ", 1)[1]))
            seen.add(name)
    if seen != set(LOAD_GAUGES):
        raise RunError(f"{url}/metrics lacks {sorted(set(LOAD_GAUGES) - seen)}")
    return total


def estimate_seconds(item: Materialized) -> float:
    tokens = len(item.message) / CHARS_PER_TOKEN
    return tokens / PREFILL_TOKENS_PER_S + EXPECTED_REPLY_TOKENS[item.family] / DECODE_TOKENS_PER_S + REQUEST_OVERHEAD_S


def request_body(suite: Suite, item: Materialized, model: str, salt: str, priority: int) -> dict[str, object]:
    body: dict[str, object] = {"model": model, "messages": [{"role": "user", "content": item.message}],
                               "max_tokens": item.max_tokens, "cache_salt": salt}
    body.update(suite.request)
    if priority:
        body["priority"] = priority
    return body


@dataclass(frozen=True)
class Row:
    task: str
    family: str
    outcome: str            # pass | fail | error
    reason: str             # scorer reason code, or the error class
    span_chars: int         # length of the visible reply (the reply itself is never kept)
    reasoning_chars: int
    finish: str
    prompt_tokens: int
    completion_tokens: int
    request_sha256: str     # sha256 of the user message: both arms must have sent identical prompts
    body_sha256: str        # sha256 of the sent body minus messages/model, salt replaced by its policy
    elapsed_s: float
    others_after: int = 0   # other requests running/waiting when the reply arrived (> 0 = not a lone request)


def _usage(response: dict[str, object], key: str) -> int:
    usage = response.get("usage")
    value = usage.get(key) if isinstance(usage, dict) else None
    return value if isinstance(value, int) else -1


def body_fingerprint(body: dict[str, object], run_id: str, task_id: str) -> str:
    """Every sent setting except the prompt and model name; a per-task salt counts as the policy, any other salt as
    itself."""
    shape = {key: value for key, value in body.items() if key not in ("messages", "model")}
    salt = shape.get("cache_salt")
    shape["cache_salt"] = "lsuite-<run_id>-<task>" if salt == f"lsuite-{run_id}-{task_id}" else f"other:{salt}"
    return sha256_text(json.dumps(shape, sort_keys=True))


def code_sha256() -> str:
    """One hash over the sources that build prompts and score replies: two runs must have used the same scorer."""
    here = Path(__file__).resolve().parent
    return sha256_text("".join(sha256_text((here / name).read_text(encoding="utf-8"))
                               for name in ("corpus.py", "scoring.py", "tasks.py", "run_suite.py")))


def run_one(suite: Suite, item: Materialized, post: Post, url: str, model: str, run_id: str, priority: int,
            timeout: float) -> Row:
    digest = sha256_text(item.message)
    body = request_body(suite, item, model, f"lsuite-{run_id}-{item.task_id}", priority)
    shape = body_fingerprint(body, run_id, item.task_id)
    started = time.monotonic()
    try:
        response = post(f"{url}/v1/chat/completions", body, timeout)
    except urllib.error.HTTPError as error:
        return Row(item.task_id, item.family, "error", f"http-{error.code}", 0, 0, "", -1, -1, digest, shape,
                   round(time.monotonic() - started, 2))
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, TaskError) as error:
        return Row(item.task_id, item.family, "error", type(error).__name__, 0, 0, "", -1, -1, digest, shape,
                   round(time.monotonic() - started, 2))
    elapsed = round(time.monotonic() - started, 2)
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return Row(item.task_id, item.family, "error", "no-choices", 0, 0, "", _usage(response, "prompt_tokens"),
                   _usage(response, "completion_tokens"), digest, shape, elapsed)
    choice: dict[str, object] = choices[0]
    message = choice.get("message")
    reply = message.get("content") if isinstance(message, dict) else None
    reasoning = None
    if isinstance(message, dict):
        reasoning = message.get("reasoning_content") or message.get("reasoning")
    reply_text = reply if isinstance(reply, str) else ""
    finish = choice.get("finish_reason")
    score = item.score(reply_text)
    row = Row(item.task_id, item.family, "pass" if score.passed else "fail", score.reason, len(reply_text),
              len(reasoning) if isinstance(reasoning, str) else 0, finish if isinstance(finish, str) else "",
              _usage(response, "prompt_tokens"), _usage(response, "completion_tokens"), digest, shape, elapsed)
    return row


def git_head() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True,
                          text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def check_out_dir(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise RunError(f"--out {out} exists and is not empty; every run gets a new directory")
    repo = Path(__file__).resolve().parents[2]
    if out.resolve().is_relative_to(repo):
        raise RunError(f"--out {out} is inside the repository {repo}; window outputs live in the campaign directory")


def served_identity(get: Get, url: str, model: str, timeout: float) -> dict[str, object]:
    listing = get(f"{url}/v1/models", timeout)
    models = [as_obj(entry, "models.data[]") for entry in as_list(listing.get("data"), "models.data")]
    for entry in models:
        if entry.get("id") == model:
            return {"id": model, "root": entry.get("root"), "max_model_len": entry.get("max_model_len")}
    raise RunError(f"{url} does not serve {model!r} (serves {[entry.get('id') for entry in models]})")


def summarize(rows: list[Row], planned: int) -> dict[str, object]:
    families: dict[str, dict[str, int]] = {}
    for row in rows:
        counts = families.setdefault(row.family, {"pass": 0, "fail": 0, "error": 0})
        counts[row.outcome] += 1
    errors = sum(1 for row in rows if row.outcome == "error")
    return {"tasks": len(rows), "planned": planned, "complete": len(rows) == planned and errors == 0,
            "pass": sum(1 for row in rows if row.outcome == "pass"), "errors": errors, "families": families,
            "elapsed_s": round(sum(row.elapsed_s for row in rows), 1)}


def execute(suite: Suite, items: list[Materialized], args: argparse.Namespace, post: Post, get: Get,
            planned: int, metrics: Metrics) -> int:
    out: Path = args.out
    check_out_dir(out)
    url = args.url.rstrip("/")
    identity = served_identity(get, url, args.model, args.timeout)
    run_id = uuid.uuid4().hex[:12]
    out.mkdir(parents=True, exist_ok=True)
    header = {"suite": suite.suite, "run_id": run_id, "arm": args.arm, "url": url, "served": identity,
              "tasks_file": str(suite.path), "tasks_sha256": suite.sha256, "corpus": dict(CORPUS_FILES),
              "request": suite.request, "max_tokens": suite.max_tokens, "priority": args.priority,
              "selected": [item.task_id for item in items], "planned": planned, "repo_head": git_head(),
              "code_sha256": code_sha256(),
              "host": socket.gethostname(), "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "run.json").write_text(json.dumps(header, indent=1) + "\n", encoding="utf-8")
    rows: list[Row] = []
    stopped = False
    with (out / "rows.jsonl").open("w", encoding="utf-8") as sink:
        for item in items:
            waited = 0.0
            while load_on(metrics, url, args.timeout) != 0 and waited < args.idle_wait:
                time.sleep(args.idle_poll)
                waited += args.idle_poll
            if waited >= args.idle_wait and load_on(metrics, url, args.timeout) != 0:
                print(f"STOPPED: {url} stayed busy for {waited:.0f} s before {item.task_id}; run is partial",
                      flush=True)
                stopped = True
                break
            row = run_one(suite, item, post, url, args.model, run_id, args.priority, args.timeout)
            others = load_on(metrics, url, args.timeout)
            if others:
                row = replace(row, outcome="error", reason="concurrent", others_after=others)
            rows.append(row)
            sink.write(json.dumps({"suite": suite.suite, "run_id": run_id, "arm": args.arm} | vars(row)) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            print(f"{row.task:12} {row.outcome:5} {row.reason:17} reply {row.span_chars:5} chars "
                  f"prompt {row.prompt_tokens:7} tok {row.elapsed_s:7.1f} s", flush=True)
    summary = summarize(rows, planned) | {"run_id": run_id, "arm": args.arm, "tasks_sha256": suite.sha256,
                                          "stopped_busy": stopped,
                                          "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"{args.arm}: {summary['pass']}/{len(rows)} pass, {summary['errors']} error(s); rows in {out}")
    return 0 if summary["errors"] == 0 and not stopped else 1


def plan(items: list[Materialized], suite: Suite) -> None:
    total = 0.0
    print(f"suite {suite.suite} tasks {suite.path.name} sha256 {suite.sha256}")
    for item in items:
        seconds = estimate_seconds(item)
        total += seconds
        tokens = len(item.message) / CHARS_PER_TOKEN
        print(f"{item.task_id:12} {item.family:8} {len(item.message):8,} chars ~{tokens:9,.0f} tok  "
              f"max_tokens {item.max_tokens:4}  ~{seconds:6.1f} s  request {sha256_text(item.message)[:16]}")
    print(f"ESTIMATED {total / 60:.1f} min per arm at B1 ({len(items)} requests; prefill {PREFILL_TOKENS_PER_S:,.0f} "
          f"tok/s, decode {DECODE_TOKENS_PER_S:.0f} tok/s, {CHARS_PER_TOKEN} chars/token planning ratio). "
          "Dry run: nothing sent. Add --execute --url URL --out DIR to run.")


def main(argv: list[str] | None = None, post: Post = http_post, get: Get = http_get,
         metrics: Metrics = http_metrics) -> int:
    parser = argparse.ArgumentParser(description="Run the L-SUITE against one served arm (dry run by default).")
    parser.add_argument("--bench-data", type=Path, default=DEFAULT_BENCH_DATA)
    parser.add_argument("--tasks-file", type=Path, default=TASKS_FILE)
    parser.add_argument("--arm", required=True, help="arm label recorded in every row (e.g. C, X, C2)")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default=SERVED)
    parser.add_argument("--out", type=Path, help="new directory for run.json, rows.jsonl, summary.json")
    parser.add_argument("--priority", type=int, default=10, help="vLLM priority (larger = background); 0 omits it")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds per request")
    parser.add_argument("--tasks", help="comma-separated task ids (a smoke subset; compare.py refuses partial runs)")
    parser.add_argument("--execute", action="store_true", help="send requests (otherwise print the plan only)")
    parser.add_argument("--idle-wait", type=float, default=600.0, help="seconds to wait for an idle arm before a task")
    parser.add_argument("--idle-poll", type=float, default=5.0, help="seconds between /metrics reads while waiting")
    args = parser.parse_args(argv)
    try:
        suite = Suite.load(args.tasks_file)
        items = suite.materialize(Corpus.load(args.bench_data))
        planned = len(items)
        if args.tasks:
            wanted = [name.strip() for name in args.tasks.split(",") if name.strip()]
            unknown = sorted(set(wanted) - {item.task_id for item in items})
            if unknown:
                raise RunError(f"unknown task id(s): {unknown}")
            items = [item for item in items if item.task_id in wanted]
        if not args.execute:
            plan(items, suite)
            return 0
        if args.out is None:
            raise RunError("--execute needs --out")
        return execute(suite, items, args, post, get, planned, metrics)
    except (RunError, TaskError, CorpusError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
