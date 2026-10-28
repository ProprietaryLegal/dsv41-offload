"""Extract numeric benchmark evidence without copying prompts, text or tensors.

Usage: python -m offload.gb10_budget.evidence --results offload/results \
    --extra-results /path/to/coordinator/offload/results/K1 --output /campaign/evidence.json
Only allowlisted result directories are read. Every input is hashed; malformed JSON
fails with a source locator. Reported step p50s are per-request client SSE gaps, not
GPU timestamps. Acceptance belongs to a whole run, never an individual B4 request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import TypeAlias, cast

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
Object: TypeAlias = dict[str, JSON]
ALLOWED = ("A", "A-target-only", "B", "Bprod", "C", "D", "Dasync",
           "Dasync-si0", "E", "E0-eager", "E1-si0", "W4-sync", "W4-p1",
           "W4-E0", "W4-E1p", "W4-E1s", "W5", "W5-E1p", "W5-prod", "W5-drill", "K1")
REQUEST_FIELDS = ("target_ctx", "occupied_prompt_tokens", "concurrency", "mixed",
                  "rep", "prompt_tokens", "completion_tokens", "ttft_s", "total_s",
                  "decode_tps", "start_offset_s")


def obj(value: JSON) -> Object:
    return value if isinstance(value, dict) else {}


def number(value: JSON) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError("Non-finite evidence number")
        return float(value)
    return None


def numeric_tree(value: JSON) -> JSON:
    """Drop every string value, preserving numeric counter names and structure."""
    if isinstance(value, dict):
        return {k: numeric_tree(v) for k, v in value.items()
                if not isinstance(v, str) and v is not None}
    if isinstance(value, list):
        return [numeric_tree(v) for v in value if not isinstance(v, str)]
    if isinstance(value, (int, float, bool)):
        if not isinstance(value, bool):
            number(value)
        return value
    return None


def extract(path: Path, raw_text: str) -> tuple[list[Object], list[Object], list[Object]]:
    requests: list[Object] = []
    runs: list[Object] = []
    metrics: list[Object] = []
    for line_no, line in enumerate(raw_text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = obj(cast(JSON, json.loads(line)))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: malformed JSON") from exc
        locator: Object = {"source": str(path), "line": line_no, "status": "MEASURED",
                           "run_id": row.get("run_id")}
        if row.get("type") == "request":
            out = locator | {k: row[k] for k in REQUEST_FIELDS
                             if k in row and (number(row[k]) is not None or isinstance(row[k], bool))}
            out["cache_state"] = row.get("cache_state") if row.get("cache_state") in ("cold", "warm", "warm_repeat") else None
            out["error_present"] = bool(row.get("error"))
            out["step_gap_ms"] = numeric_tree(row.get("step_gap_ms"))
            out["cached_tokens"] = obj(obj(row.get("usage")).get("prompt_tokens_details")).get("cached_tokens")
            # Only B1 has a meaningful occupied/TTFT prefill proxy. It still includes startup/first-step costs.
            p, t = number(row.get("prompt_tokens")), number(row.get("ttft_s"))
            if row.get("concurrency") == 1 and p and t and not out.get("cached_tokens"):
                out["prefill_proxy_tps"] = p / t
                out["chunk4096_equivalent_ms"] = 4096 * t / p * 1000
            requests.append(out)
        elif row.get("type") == "run":
            spec = obj(row.get("spec"))
            out = locator | {k: row.get(k) for k in ("target_ctx", "concurrency", "mixed", "rep")}
            out["error_present"] = bool(row.get("error"))
            out["accepted_per_round"] = spec.get("mean_accepted_per_round")
            out["acceptance_per_position_per_round"] = numeric_tree(spec.get("acceptance_per_position_per_round"))
            out["aggregate_including_prefill_tps"] = row.get("aggregate_tps")
            out["rounds_request_sum"] = spec.get("rounds")
            # Counters at B>1 count request-rounds, NOT GPU engine iterations.
            runs.append(out)
            # Full per-server snapshots only numeric values and fixed prefixes; no model label strings.
            for name in ("metrics_before", "metrics_after"):
                snapshot = obj(row.get(name))
                metrics.append(locator | {"phase": name, "values": {
                    k.split("{")[0]: v for k, v in snapshot.items()
                    if k.startswith(("vllm:time_per_output_token_seconds_sum", "vllm:time_per_output_token_seconds_count"))}})
    return requests, runs, metrics


def analyze(roots: list[Path]) -> Object:
    files = sorted({p.resolve() for root in roots for p in root.rglob("*")
                    if p.is_file() and p.suffix in (".json", ".jsonl", ".txt", ".log")})
    manifest: list[JSON] = []
    requests: list[Object] = []
    runs: list[Object] = []
    stats: list[JSON] = []
    for path in files:
        data = path.read_bytes()
        manifest.append({"source": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        if path.suffix == ".jsonl":
            req, run, _ = extract(path, data.decode())
            requests.extend(req)
            runs.extend(run)
        elif path.suffix == ".json" and "stats" in path.name:
            stats.append({"source": str(path), "status": "MEASURED", "numeric": numeric_tree(cast(JSON, json.loads(data)))})
    groups: dict[tuple[str, int, int, str, bool], list[Object]] = defaultdict(list)
    for row in requests:
        key = (str(row["source"]), int(cast(int, row.get("target_ctx", 0))),
               int(cast(int, row.get("concurrency", 0))), str(row.get("cache_state")), bool(row.get("mixed")))
        groups[key].append(row)
    summaries: list[JSON] = []
    for (source, ctx, batch, cache, mixed), rows in sorted(groups.items()):
        summary: Object = {"source": source, "context": ctx, "batch": batch, "cache": cache,
                           "mixed": mixed, "n_requests": len(rows), "status": "MEASURED"}
        for name in ("ttft_s", "decode_tps", "total_s", "prefill_proxy_tps", "chunk4096_equivalent_ms"):
            values = [v for row in rows if (v := number(row.get(name))) is not None]
            summary[name] = {"mean": mean(values), "min": min(values), "max": max(values)} if values else None
        values = [v for row in rows if (v := number(obj(row.get("step_gap_ms")).get("p50"))) is not None]
        summary["request_step_p50_ms"] = {"mean": mean(values), "min": min(values), "max": max(values)} if values else None
        # Match B1 run counters by repetition; never attribute global acceptance to one concurrent stream.
        estimates = []
        if batch == 1 and not mixed:
            for row in rows:
                match = [r for r in runs if r["source"] == source and r.get("run_id") == row.get("run_id")]
                if len(match) == 1:
                    a, tps = number(match[0].get("accepted_per_round")), number(row.get("decode_tps"))
                    if a is not None and tps:
                        estimates.append(1000 * (1 + a) / tps)
        summary["acceptance_normalized_round_ms_ESTIMATED"] = mean(estimates) if estimates else None
        summaries.append(summary)
    return {"method": "Allowlisted aggregates only; SSE timings and wall-time proxies, not kernel profiles.",
            "inputs": manifest, "requests": list[JSON](requests), "runs": list[JSON](runs),
            "stats": stats, "summaries": summaries}


def markdown(result: Object) -> str:
    lines = ["# SP1 complete numeric evidence reconciliation", "",
             "All timing values: MEASURED saved-client observations; norm-ms: ESTIMATED acceptance-normalized B1 mean.",
             "Each row is one source/context/batch/cache/mixed group. p50 column is the mean of request medians.",
             "Do not pool medians as engine steps or count concurrent requests as independent repetitions.", "",
             "| Source | ctx | B | cache | mixed | n requests | TTFT s | SSE p50 ms | norm ms | decode tok/s |",
             "|---|---:|---:|---|---|---:|---:|---:|---:|---:|"]
    summaries = result["summaries"]
    assert isinstance(summaries, list)
    for raw in summaries:
        row = obj(raw)
        def avg(key: str) -> str:
            value = number(obj(row.get(key)).get("mean"))
            return f"{value:.3f}" if value is not None else "—"
        norm = number(row.get("acceptance_normalized_round_ms_ESTIMATED"))
        norm_cell = f"{norm:.3f}" if norm is not None else "—"
        lines.append(f"| {row['source']} | {row['context']} | {row['batch']} | {row['cache']} | "
                     f"{row['mixed']} | {row['n_requests']} | {avg('ttft_s')} | "
                     f"{avg('request_step_p50_ms')} | {norm_cell} | {avg('decode_tps')} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--extra-results", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.results.is_dir():
        parser.error(f"Missing result root: {args.results}")
    roots = [args.results / name for name in ALLOWED if (args.results / name).is_dir()]
    for root in args.extra_results:
        if not root.is_dir() or root.name not in ALLOWED:
            parser.error(f"Missing or non-allowlisted extra results: {root}")
        roots.append(root)
    result = analyze(roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".md").write_text(markdown(result))
    print(f"Wrote {args.output}: {len(cast(list[JSON], result['inputs']))} hashed files, "
          f"{len(cast(list[JSON], result['requests']))} request aggregates")


if __name__ == "__main__":
    main()
