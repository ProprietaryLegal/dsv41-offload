#!/usr/bin/env python3
"""PF2-LOWKV memory baselines (CPU only, read-only): per-arm MemAvailable from earlier windows' 1 s telemetry, and the
FW1 post-update sparkhead inventory when it exists.

Arm traffic intervals come from each window's requests-ledger.jsonl: every arm begins with exactly one 32K warm-up
(PF1 window_requests --warmup; SP1 cell 'warmup'), so the n-th warm-up opens the n-th arm and the arm's interval runs to its
last ledger event. Arm names are passed in run order. Numbers only.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, number, record

CAMPAIGN = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')
HOSTS = ('sparkhead', 'sparkworker-wired')
# Run order from LANES.md (2026-10-25/26) and each window's summary; 150K cold TTFT medians (s) MEASURED there.
PF1_WINDOWS: dict[str, tuple[tuple[str, float | None], ...]] = {
    'PF1-W1-20261025T234254Z': (('off-327', 70.38), ('off-524', 95.33), ('on-327', None)),
    'PF1-W2-20261026T010854Z': (('on-327', None), ('on-524', 91.48)),
    'PF1-CW-20261026T024340Z': (('cause-327', 70.37), ('cause-524', 69.20), ('cause-524-t0', 69.77)),
    'PF1-W3-20261026T032022Z': (('off-327', 68.96), ('off-524', 91.34)),
    'PF1-MATCH-20261026T045943Z': (('off-524', 70.18),),
}


@dataclass(frozen=True)
class ArmMemory:
    window: str
    arm: str
    position: int
    ttft_150k_s: float | None
    host: str
    samples: int
    min_gib: float
    p05_gib: float
    median_gib: float


def intervals(ledger: Path) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for line in ledger.read_text().splitlines():
        row = record(cast(Json, json.loads(line)))
        t = number(row['time'])
        if row.get('event') in ('send_intent', 'send') and (row.get('kind') == 'warmup' or row.get('cell') == 'warmup'):
            out.append([t, t])
        elif out:
            out[-1][1] = max(out[-1][1], t)
    return [(a, b) for a, b in out]


def memory(path: Path) -> list[tuple[float, int]]:
    rows = []
    for line in path.read_text().splitlines():
        row = record(cast(Json, json.loads(line)))
        if row.get('kind') == 'memory':
            rows.append((number(row['utc']), int(number(record(row['kb'])['MemAvailable']))))
    return rows


def window(directory: Path, arms: tuple[tuple[str, float | None], ...]) -> list[ArmMemory]:
    spans = intervals(directory / 'requests-ledger.jsonl')
    if len(spans) != len(arms):
        raise ValueError(f'{directory.name}: {len(spans)} warm-ups for {len(arms)} arms')
    out: list[ArmMemory] = []
    for host in HOSTS:
        samples = memory(directory / f'telemetry-{host}.jsonl')
        for position, ((name, ttft), (a, b)) in enumerate(zip(arms, spans), 1):
            xs = sorted(v for t, v in samples if a <= t <= b)
            if not xs:
                raise ValueError(f'{directory.name} {name}: no {host} memory samples in the arm interval')
            out.append(ArmMemory(directory.name, name, position, ttft, host, len(xs), xs[0] / 2**20,
                                 xs[len(xs) // 20] / 2**20, statistics.median(xs) / 2**20))
    return out


FW1_KERNEL = re.compile(r'^\d+\.\d+\.\d+-\d+-nvidia\S*$')
FW1_DRIVER = re.compile(r'^\d{3}\.\d+(?:\.\d+)?$')


def inventory(path: Path) -> dict[str, Json]:
    """FW1 fw1_sparkhead.sh inventory-<pre|post>.txt: uname -r, driver, MemTotal/MemAvailable/SwapTotal (kB)."""
    lines = [x.strip() for x in path.read_text().splitlines()]
    kernels = [x for x in lines if FW1_KERNEL.fullmatch(x)]
    drivers = [x for x in lines if FW1_DRIVER.fullmatch(x)]
    mem = {m[1]: int(m[2]) for x in lines if (m := re.fullmatch(r'(MemTotal|MemAvailable|SwapTotal):\s+(\d+) kB', x))}
    if len(kernels) != 1 or len(drivers) != 1 or set(mem) != {'MemTotal', 'MemAvailable', 'SwapTotal'}:
        raise ValueError(f'{path.name}: not an FW1 inventory (kernel {len(kernels)}, driver {len(drivers)}, meminfo {sorted(mem)})')
    return {'kernel': kernels[0], 'driver': drivers[0], **{k: v for k, v in mem.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--campaign', type=Path, default=CAMPAIGN)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args(argv)
    rows = [r for name, arms in PF1_WINDOWS.items() for r in window(args.campaign / name, arms)]
    fw1: dict[str, Json] = {}
    for stage in ('pre', 'post'):
        path = args.campaign / 'FW1' / f'inventory-{stage}.txt'
        fw1[stage] = inventory(path) if path.exists() else 'absent'
    if args.json:
        print(json.dumps({'arms': [asdict(r) for r in rows], 'fw1': fw1}, indent=1))
        return 0
    print('window                      arm           pos  150K TTFT  host        n     min   p05   median GiB')
    for r in sorted(rows, key=lambda r: (r.host, r.window, r.position)):
        ttft = f'{r.ttft_150k_s:6.2f} s' if r.ttft_150k_s is not None else '     -  '
        print(f'{r.window:<27} {r.arm:<13} {r.position:>3}  {ttft}  {r.host:<10} {r.samples:>5} '
              f'{r.min_gib:5.2f} {r.p05_gib:5.2f} {r.median_gib:6.2f}')
    print('FW1 inventory:', json.dumps(fw1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
