"""WIN2-R3 2x resolution filter (F-METHOD §3.2 rule 3): refuse to queue a measurement that cannot resolve its own claim.

Every pre-registration states `expected_gain` (a fraction) per decision cell. From the control archive, per definition
(cell, metric, max_tokens, clock_cap_mhz, commit and ONE prompt identity: DP-2, never pooled across prompts), take the within-window CV of one repeat (pooled over windows,
df-weighted), compute the 95 % half-width of the gain for the registered design, and refuse when
expected_gain < 2 x half-width:

| design | half-width |
|---|---|
| unpaired:AvB (A control, B candidate repeats) | t(0.975, A+B-2) x CV x sqrt(1/A + 1/B) |
| pairs:N (unseeded pairs) | t(0.975, N-1) x CV x sqrt(2) / sqrt(N) |
| seeded:N (seeded pairs, target-side lever) | t(0.975, N-1) x diff CV / sqrt(N); diff CV = the seeded A/A pilot's measured paired-difference CV (--diff-cv), else ESTIMATED as the definition's ms/round CV x sqrt(2) (F-METHOD §2) |

Example (F-METHOD, ESTIMATED): a +3 % decode lever passes only as seeded pairs on 32k-b1 (+-1.5 %), not on short-b1
(+-3.1 %) and never as 3 v 3 unpaired (+-16 %). Read only; exit 0 QUEUE, 2 REFUSE.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import cast

from offload.launch.windows import decision_v2 as d
from offload.prefill_maxlen.evidence import Json, number

ARCHIVE = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/control-archive.jsonl')
FACTOR = 2.0
DESIGN = re.compile(r'^(unpaired):([0-9]+)v([0-9]+)$|^(pairs|seeded):([0-9]+)$')


def pooled_cv(archive: Path, cell: str, metric: str, *, max_tokens: int, clock_cap_mhz: int, commit: str,
              prompt_sha256: str | None = None) -> dict[str, Json]:
    """Within-window CV of one repeat for one definition, pooled over windows with df weights (all control repeats of a
    window, so the between-start variation inside a C/X/C window is included). The definition's rows must share ONE
    prompt identity (DP-2); with several, `prompt_sha256` must pick one."""
    per_window: dict[str, list[float]] = {}
    prompts: set[str] = set()
    for row in d.archive_rows(archive):
        if (row.get('cell'), row.get('metric'), row.get('max_tokens'), row.get('clock_cap_mhz'), row.get('commit')) == \
                (cell, metric, max_tokens, clock_cap_mhz, commit):
            if prompt_sha256 is not None and row.get('prompt_sha256') != prompt_sha256:
                continue
            prompts.add(str(row.get('prompt_sha256')))
            per_window.setdefault(str(row['window']), []).append(number(row['value']))
    if len(prompts) > 1:
        raise ValueError(f'{cell}/{metric} max_tokens={max_tokens}: archive rows of {len(prompts)} prompts '
                         f'({", ".join(sorted(x[:12] for x in prompts))}); never pooled across prompt hashes, pass '
                         'prompt_sha256 (--prompt-sha256)')
    usable = {w: v for w, v in per_window.items() if len(v) >= 2}
    if not usable:
        raise ValueError(f'no archive history for {cell}/{metric} max_tokens={max_tokens} clock={clock_cap_mhz} '
                         f'commit={commit[:12]}; measure the definition before registering a design on it')
    num = sum((len(v) - 1) * (statistics.stdev(v) / statistics.fmean(v)) ** 2 for v in usable.values())
    df = sum(len(v) - 1 for v in usable.values())
    return {'cv': math.sqrt(num / df), 'df': df, 'windows': len(usable), 'rows': sum(len(v) for v in usable.values()),
            'prompt_sha256': next(iter(prompts))}


def half_width(design: str, cv: float, *, diff_cv: float | None = None, ms_cv: float | None = None) -> dict[str, Json]:
    m = DESIGN.fullmatch(design)
    if m is None:
        raise ValueError(f'design must be unpaired:AvB, pairs:N or seeded:N, got {design!r}')
    if m.group(1):
        a, b = int(m.group(2)), int(m.group(3))
        if a < 2 or b < 2:
            raise ValueError('unpaired designs need at least 2 repeats per arm')
        return {'half_width': d.t975(a + b - 2) * cv * math.sqrt(1 / a + 1 / b), 'basis': 'archive CV, pooled-variance t'}
    kind, n = m.group(4), int(m.group(5))
    if n < 2:
        raise ValueError('paired designs need at least 2 pairs')
    if kind == 'pairs':
        return {'half_width': d.t975(n - 1) * cv * math.sqrt(2) / math.sqrt(n), 'basis': 'archive CV x sqrt(2), paired t'}
    if n < d.MIN_PAIRS:
        raise ValueError(f'seeded designs need n >= {d.MIN_PAIRS} pairs (win2-paired-v1)')
    if diff_cv is not None:
        return {'half_width': d.t975(n - 1) * diff_cv / math.sqrt(n), 'basis': 'MEASURED seeded A/A paired-difference CV'}
    if ms_cv is None:
        raise ValueError('seeded designs need --diff-cv (seeded A/A pilot) or the definition\'s ms_per_round history')
    return {'half_width': d.t975(n - 1) * ms_cv * math.sqrt(2) / math.sqrt(n),
            'basis': 'ESTIMATED diff CV = ms/round CV x sqrt(2) (F-METHOD §2) until the seeded A/A pilot measures it'}


def check(cell: str, metric: str, design: str, expected_gain: float, *, max_tokens: int, clock_cap_mhz: int, commit: str,
          archive: Path = ARCHIVE, diff_cv: float | None = None, prompt_sha256: str | None = None) -> dict[str, Json]:
    if not (math.isfinite(expected_gain) and expected_gain > 0):
        raise ValueError('expected_gain must be a positive fraction (e.g. 0.03)')
    if diff_cv is not None and not (math.isfinite(diff_cv) and diff_cv > 0):
        raise ValueError('diff_cv must be a positive fraction')
    key: dict[str, Json] = {'max_tokens': max_tokens, 'clock_cap_mhz': clock_cap_mhz, 'commit': commit}
    seeded = design.startswith('seeded:')
    history: dict[str, Json] | None = None
    ms: dict[str, Json] | None = None
    if not (seeded and diff_cv is not None):
        found = pooled_cv(archive, cell, 'ms_per_round' if seeded else metric, max_tokens=max_tokens,
                          clock_cap_mhz=clock_cap_mhz, commit=commit, prompt_sha256=prompt_sha256)
        if seeded:
            ms = found
        else:
            history = found
    width = half_width(design, number(history['cv']) if history else 0.0, diff_cv=diff_cv,
                       ms_cv=number(ms['cv']) if ms else None)
    half = number(width['half_width'])
    verdict = 'QUEUE' if expected_gain >= FACTOR * half else 'REFUSE'
    return {'cell': cell, 'metric': metric, 'design': design, 'definition': cast(Json, key),
            'expected_gain': expected_gain, 'half_width95': half, 'required_gain': FACTOR * half,
            'basis': width['basis'], 'archive_cv': history, 'ms_per_round_cv': ms, 'diff_cv': diff_cv,
            'verdict': verdict,
            'reason': (f'expected {expected_gain:.2%} {">=" if verdict == "QUEUE" else "<"} {FACTOR:g} x half-width '
                       f'{half:.2%} for {design}')}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--metric', required=True)
    ap.add_argument('--design', required=True, help='unpaired:AvB | pairs:N | seeded:N')
    ap.add_argument('--expected-gain', type=float, required=True, help='registered expected gain, fraction (0.03 = 3 %%)')
    ap.add_argument('--max-tokens', type=int, required=True)
    ap.add_argument('--clock', type=int, required=True, help='clock_cap_mhz of the definition')
    ap.add_argument('--commit', required=True, help='40-hex production commit of the definition')
    ap.add_argument('--diff-cv', type=float, help='seeded designs: MEASURED paired-difference CV from the A/A pilot')
    ap.add_argument('--archive', type=Path, default=ARCHIVE)
    ap.add_argument('--prompt-sha256', help='the definition\'s prompt identity when the archive holds several (DP-2)')
    args = ap.parse_args(argv)
    try:
        result = check(args.cell, args.metric, args.design, args.expected_gain, max_tokens=args.max_tokens,
                       clock_cap_mhz=args.clock, commit=args.commit, archive=args.archive, diff_cv=args.diff_cv,
                       prompt_sha256=args.prompt_sha256)
    except ValueError as exc:
        print(f'REFUSE: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0 if result['verdict'] == 'QUEUE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
