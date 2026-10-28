"""Aggregate phase-tagged 1s pair telemetry and already logged protocol timings.

Reads copied result files only. No Torch, CUDA, SSH, listener, or service calls.
SMI peaks are sampled board memory; neither exact allocation peaks nor weight
size. The synthetic loopback 40-case mix cannot calibrate GB10 depth costs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from .strict_summary import ARMS, mapping, number, record

CARDS = ("GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93", "GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc")
PHASES = ("load-and-prewarm", "serve", "done")


def phase_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("missing required phase/card telemetry")
    for row in rows:
        for field in ("time", "memory_mib", "power_w", "limit_w", "sm_mhz", "mem_mhz", "temp_c"):
            if not math.isfinite(number(row[field])):
                raise ValueError(f"nonfinite telemetry: {field}")
        if row["limit_w"] != 325:
            raise ValueError("unexpected pair power limit")
    times = [number(row["time"]) for row in rows]
    if times != sorted(times):
        raise ValueError("telemetry time regression")
    return {"samples": len(rows), "observed_span_s": times[-1] - times[0],
        "memory_peak_mib": max(number(r["memory_mib"]) for r in rows),
        "power_sample_mean_w": sum(number(r["power_w"]) for r in rows) / len(rows),
        "power_peak_w": max(number(r["power_w"]) for r in rows), "power_limit_w": 325,
        "sm_mhz_min": min(number(r["sm_mhz"]) for r in rows),
        "sm_mhz_max": max(number(r["sm_mhz"]) for r in rows),
        "temperature_peak_c": max(number(r["temp_c"]) for r in rows)}


def collect(pair: Path) -> dict[str, object]:
    arms: list[dict[str, object]] = []
    for arm in ARMS:
        modes: list[dict[str, object]] = []
        digests = []
        for mode in ("off", "on"):
            folder = pair / f"{arm}-{mode}"
            result = record(folder / "result.json")
            client = record(folder / "client.json")
            if result["client"] != client or result["arm"] != arm or result["mode"] != mode:
                raise ValueError(f"{arm}/{mode}: receipt mismatch")
            cases = mapping(client["cases"])
            if client["rounds"] != 2000 or len(cases) != 40 or set(cases.values()) != {50}:
                raise ValueError(f"{arm}/{mode}: incomplete 40-case/2000-round soak")
            expected_cases = {f"B{b}_n{n}_margin{m}" for b in range(1, 5) for n in range(1, 6) for m in range(2)}
            if set(cases) != expected_cases:
                raise ValueError("unexpected graph-case scope")
            stats = mapping(client["stats"])
            if stats["tp_comm_errors"] != 0 or stats["tp_comm_errors_in_a_row"] != 0 or stats["steps"] != 2000:
                raise ValueError("TP error or incomplete steps")
            rows = [mapping(json.loads(line)) for line in (folder / "memory-1s.jsonl").read_text().splitlines()]
            if len(rows) != result["samples"] or {r["uuid"] for r in rows} != set(CARDS):
                raise ValueError("telemetry cardinality/UUID mismatch")
            if {r["phase"] for r in rows} != set(PHASES):
                raise ValueError("unexpected/missing memory phase")
            phases: dict[str, dict[str, object]] = {}
            declared_peaks = mapping(result["memory_peak_mib_1s_observed"])
            for phase in PHASES:
                phases[phase] = {}
                for card in CARDS:
                    summary = phase_summary([r for r in rows if r["phase"] == phase and r["uuid"] == card])
                    if summary["memory_peak_mib"] != mapping(declared_peaks[phase])[card]:
                        raise ValueError("recomputed memory peak differs from receipt")
                    phases[phase][card] = summary
            for card in CARDS:
                stream = [r for r in rows if r["uuid"] == card]
                sequence = [PHASES.index(str(r["phase"])) for r in stream]
                if sequence != sorted(sequence):
                    raise ValueError("phase chronology regression")
            ready = re.findall(r"TP=2 worker ready in ([0-9.]+) s", (folder / "worker.log").read_text())
            if len(ready) != 1:
                raise ValueError("missing/ambiguous worker ready time")
            digests.append(client["digest"])
            modes.append({"mode": mode, "identity": result["identity"], "startup_to_status_s": result["startup_s"],
                "worker_factory_ready_s": float(ready[0]), "phases": phases,
                "rounds": 2000, "cases": 40, "replays_per_case": 50, "tp_comm_errors": 0,
                "deliberate_hello_identity_refusals": stats["identity_mismatch"],
                "deliberate_hello_config_refusals": stats["config_mismatch"],
                "loopback_client_mixed_round_latency_ms": client["latency_ms"],
                "worker_mixed_round_latency_ms": {"p50": number(stats["worker_us_p50"]) / 1000,
                                                   "p99": number(stats["worker_us_p99"]) / 1000},
                "reply_digest": client["digest"]})
        equality = record(pair / f"{arm}-on-off-equal.json")
        if equality["equal"] is not True or digests[0] != digests[1]:
            raise ValueError("on/off reply mismatch")
        capture: dict[str, object] = {}
        with (pair / f"{arm}-capture.smi.csv").open() as f:
            raw = list(csv.reader(f))
        for card in CARDS:
            selected = [r for r in raw if r[1].strip() == card]
            if not selected or any(float(r[4].strip().split()[0]) != 325 for r in selected):
                raise ValueError("capture telemetry missing card or changed power cap")
            capture[card] = {"samples": len(selected),
                "unphased_whole_capture_memory_peak_mib": max(float(r[2].strip().split()[0]) for r in selected),
                "unphased_whole_capture_power_peak_w": max(float(r[3].strip().split()[0]) for r in selected)}
        arms.append({"arm": arm, "on_off_replies_equal": True, "modes": modes, "capture_debug_unphased": capture})
    return {"source_run": pair.name, "cards_rank_order": list(CARDS),
        "method": "1s observed SMI board memory and power; load includes prewarm; serve isolated by sampler phase",
        "timing_scope": "single synthetic local TCP-stub soak, 40 equally sampled B1-4/n1-5/margin0-1 cases",
        "worker_timing_scope": "server perf_counter around prepare+draft, rounded microseconds; includes dump observers when on",
        "not_gb10_cost_calibration": True, "arms": arms}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    result = collect(a.pair)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
