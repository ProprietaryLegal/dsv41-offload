#!/usr/bin/env python3
"""Tabulate run means and repeat spread; failures never enter success averages."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import TypeAlias, cast

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
Object: TypeAlias = dict[str, JSON]


def spread(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values) if len(values) > 1 else 0:.2f}"


def table(paths: list[Path]) -> str:
    requests: dict[str, list[Object]] = defaultdict(list)
    groups: dict[tuple[str, ...], list[Object]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = cast(Object, json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            if row.get("type") not in ("run", "request"):
                continue
            run_id = str(row["run_id"])
            identity = (run_id, str(row.get("request_index", "summary")))
            if identity in seen:
                raise ValueError(f"duplicate run/request {identity}; do not include the same data twice")
            seen.add(identity)
            if row["type"] == "request":
                requests[run_id].append(row)
            else:
                config = cast(Object, row["settings"])
                # Keep unlike inference settings separate even if labels/context match.
                signature = json.dumps({k: config.get(k) for k in ("model", "temperature", "top_p", "seed", "chat_template_kwargs", "max_tokens", "priority", "filler")}, sort_keys=True)
                group = tuple(str(row.get(k)) for k in ("config_label", "target_ctx", "concurrency", "mixed", "workload", "cache_state")) + (signature,)
                groups[group].append(row)
    lines = ["Mean ± sample standard deviation across repeated run means; seconds or tokens/s as labelled.",
             "Warm priming runs are cold rows. Speculation is server-global within each run window.", "",
             "| Config | Context | C | Mixed | Workload | Cache | OK/total | Aggregate tok/s | Stream tok/s | Reason tok/s | Answer tok/s | TTFT s | Visible s | Task s | Accepted/round |",
             "|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    notes: list[str] = []
    for key, runs in sorted(groups.items()):
        good = [r for r in runs if not r.get("error")]
        def run_values(field: str, good: list[Object] = good) -> list[float]:
            return [float(cast(float, r[field])) for r in good if r.get(field) is not None]
        def request_means(field: str, good: list[Object] = good) -> list[float]:
            values: list[float] = []
            for run in good:
                rows = requests[str(run["run_id"])]
                if len(rows) != run["request_count"]:
                    raise ValueError(f"missing requests for {run['run_id']}")
                nums = [float(cast(float, r[field])) for r in rows if r.get(field) is not None and not r.get("error")]
                if nums:
                    values.append(statistics.mean(nums))
            return values
        accepted = [float(cast(float, cast(Object, r["spec"])["mean_accepted_per_round"])) for r in good
                    if cast(Object, r["spec"]).get("mean_accepted_per_round") is not None]
        cells = list(key[:6]) + [f"{len(good)}/{len(runs)}", spread(run_values("aggregate_tps"))]
        cells += [spread(request_means(f)) for f in ("decode_tps", "reasoning_decode_tps", "answer_decode_tps", "ttft_s", "first_visible_s", "total_s")]
        cells.append(spread(accepted))
        lines.append("| " + " | ".join(cells) + " |")
        notes.append(f"\nSettings for {key[0]}/{key[1]}/{key[2]}/{key[5]}: `{key[6]}`\n")
        for run in runs:
            if run.get("telemetry_errors"):
                notes.append(f"Telemetry errors {run['run_id']}: {json.dumps(run['telemetry_errors'])}\n")
            if run.get("error"):
                notes.append(f"Failure {run['run_id']}: {run['error']}\n")
    return "\n".join(lines + notes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = table(args.files)
    if args.out:
        args.out.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
