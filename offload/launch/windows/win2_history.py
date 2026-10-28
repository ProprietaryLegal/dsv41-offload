"""WIN2 history: re-score finished windows under win2-v1 and backfill the control archive (read-only on window dirs).

Two window layouts exist: PC1/UP3/TV1-style `<start>-<cell>.json` with `repetitions` (per repeat: server_decode_ms,
spec_request_rounds, acceptance_per_round, median_decode_tps, median_ttft_s), and NBW-style flat files with one value
per start. Roles come from `summary.json` starts: 'control' or 'K' are controls; any other role is the candidate.
A cell needs >= 3 values per arm for a win2-v1 verdict; below that the honest answer is INSUFFICIENT_REPEATS, with the
descriptive gain and (n >= 2 per arm) a Welch interval shown for orientation only.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from offload.launch.windows import decision_v2 as d
from offload.prefill_maxlen.evidence import Json, number, record

CONTROL_ROLES = ('control', 'K')
# metric -> higher_is_better; ms_per_round is pooled server time per speculative round.
METRICS = {'ms_per_round': False, 'accepted_per_round': True, 'decode_tps': True, 'ttft_s': False}
CLOCK = re.compile(r'^(spark(?:worker)?|sparkhead|sparkworker) clock cap ([0-9]+) MHz', re.M)


def _read(path: Path) -> dict[str, Json]:
    return record(cast(Json, json.loads(path.read_text())))


def _rep_metrics(rep: dict[str, Json]) -> dict[str, float]:
    out: dict[str, float] = {}
    if 'server_decode_ms' in rep and 'spec_request_rounds' in rep and number(rep['spec_request_rounds']) > 0:
        out['ms_per_round'] = number(rep['server_decode_ms']) / number(rep['spec_request_rounds'])
    # NBW flat files carry the same quotient precomputed (round_time_scope: decode duration sum / request rounds).
    for key, metric in (('server_decode_ms_per_spec_request_round', 'ms_per_round'),
                        ('acceptance_per_round', 'accepted_per_round'), ('median_decode_tps', 'decode_tps'),
                        ('median_ttft_s', 'ttft_s')):
        if metric in out:
            continue
        value = rep.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
            out[metric] = float(value)
    return out


def _max_tokens(rep: dict[str, Json], where: Path) -> int:
    """The one max_tokens every stream of this repeat ran with (recorded in each stream's sampling block)."""
    streams = rep.get('streams')
    values = {record(record(s).get('sampling') or {}).get('max_tokens')
              for s in (streams if isinstance(streams, list) else [])}
    if len(values) != 1:
        raise ValueError(f'{where}: expected one recorded max_tokens across the streams, found {sorted(map(str, values))}')
    tokens = values.pop()
    if not (isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0):
        raise ValueError(f'{where}: recorded max_tokens {tokens!r} is not a positive integer')
    return tokens


def samples(window: Path) -> list[tuple[str, str, str, float, int]]:
    """(cell, 'control'|'candidate', metric, value, max_tokens) for every repeat of every start file of a window."""
    summary = _read(window / 'summary.json')
    starts = record(summary['starts'])
    out: list[tuple[str, str, str, float, int]] = []
    for start, data in starts.items():
        role = str(record(data).get('role'))
        arm = 'control' if role in CONTROL_ROLES else 'candidate'
        for path in sorted(window.glob(f'{start}-*.json')):
            cell = path.name[len(start) + 1:-len('.json')]
            if cell in ('greedy', 'identity'):
                continue
            body = _read(path)
            reps = body.get('repetitions')
            rows = [record(r) for r in reps] if isinstance(reps, list) and reps else [body]
            for rep in rows:
                tokens = _max_tokens(rep, path)
                out.extend((cell, arm, metric, value, tokens) for metric, value in _rep_metrics(rep).items())
    return out


def extract(window: Path) -> dict[str, dict[str, dict[str, list[float]]]]:
    """{cell: {'control'|'candidate': {metric: [values...]}}} from every start file of a finished window."""
    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for cell, arm, metric, value, _ in samples(window):
        out.setdefault(cell, {}).setdefault(arm, {}).setdefault(metric, []).append(value)
    return out


def _welch(control: list[float], candidate: list[float], higher: bool) -> list[float] | None:
    if len(control) < 2 or len(candidate) < 2:
        return None
    mk, mc = statistics.fmean(control), statistics.fmean(candidate)
    vk, vc = statistics.variance(control) / len(control), statistics.variance(candidate) / len(candidate)
    se = math.sqrt(vk + vc)
    df = (vk + vc) ** 2 / (vk ** 2 / (len(control) - 1) + vc ** 2 / (len(candidate) - 1)) if se > 0 else 1e9
    point = (mc - mk) / mk if higher else (mk - mc) / mk
    half = d.t975(df) * se / mk
    return [point - half, point + half]


def reanalyze(window: Path, decision_cells: Sequence[str]) -> dict[str, Json]:
    data = extract(window)
    cells: dict[str, Json] = {}
    verdicts: list[str] = []
    for cell, arms in sorted(data.items()):
        metrics: dict[str, Json] = {}
        for metric, higher in METRICS.items():
            control = arms.get('control', {}).get(metric, [])
            candidate = arms.get('candidate', {}).get(metric, [])
            if not control or not candidate:
                continue
            point = ((statistics.fmean(candidate) - statistics.fmean(control)) / statistics.fmean(control)
                     * (1 if higher else -1))
            row: dict[str, Json] = {'n_control': len(control), 'n_candidate': len(candidate), 'point_gain': point,
                                    'welch_interval95': cast(Json, _welch(control, candidate, higher))}
            if min(len(control), len(candidate)) >= d.MIN_REPEATS:
                row['verdict'] = d.decide_cell(d.Series(cell, metric, higher, tuple(control), tuple(candidate),
                                                        paired=False))['verdict']
            else:
                row['verdict'] = 'INSUFFICIENT_REPEATS'
            metrics[metric] = row
        primary = next((m for m in ('ms_per_round', 'decode_tps', 'ttft_s') if m in metrics), None)
        decides = cell in decision_cells
        cell_verdict = str(record(metrics[primary])['verdict']) if primary else 'NO_DATA'
        if decides:
            verdicts.append(cell_verdict)
        cells[cell] = {'decides': decides, 'primary_metric': primary, 'verdict': cell_verdict if decides else 'REPORTED',
                       'metrics': metrics}
    if any(v in ('INSUFFICIENT_REPEATS', 'NO_DATA') for v in verdicts) or len(verdicts) != len(decision_cells):
        status = 'INCONCLUSIVE (insufficient repeats for win2-v1)'
    elif 'LOSS' in verdicts:
        status = 'NOT A CANDIDATE'
    else:
        status = 'CANDIDATE' if 'WIN' in verdicts else 'INCONCLUSIVE'
    old = record(_read(window / 'summary.json').get('decision') or {}).get('status')
    return {'window': window.name, 'rule': d.RULE, 'decision_cells': cast(Json, list(decision_cells)),
            'historical_status': old, 'win2_status': status, 'cells': cells}


def clock_caps(window: Path) -> list[int]:
    caps: set[int] = set()
    for log in (window / 'cmd').glob('*start*.log') if (window / 'cmd').is_dir() else []:
        caps.update(int(m.group(2)) for m in CLOCK.finditer(log.read_text(errors='replace')))
    return sorted(caps)


def backfill(windows: Sequence[Path], archive: Path) -> int:
    """Append control rows; a (window, cell, metric) already archived is skipped, so a re-run adds only what is new."""
    archived = {(r.get('window'), r.get('cell'), r.get('metric')) for r in d.archive_rows(archive)}
    rows: list[dict[str, Json]] = []
    for window in windows:
        summary = _read(window / 'summary.json')
        if not isinstance(summary.get('starts'), dict) or not summary['starts']:
            continue                       # windows that stopped before any start have no control cells
        commit = summary.get('production_commit') or summary.get('experiment_commit')
        caps = clock_caps(window)
        date = window.name.rsplit('-W-', 1)[-1][:8] if '-W-' in window.name else None
        for cell, arm, metric, value, tokens in samples(window):
            if arm != 'control' or (window.name, cell, metric) in archived:
                continue
            rows.append({'window': window.name, 'cell': cell, 'metric': metric, 'value': value, 'max_tokens': tokens,
                         'commit': commit, 'clock_caps_mhz': cast(Json, caps), 'date': date,
                         'source': 'win2_history backfill'})
    if rows:
        d.archive_append(archive, rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('reanalyze')
    r.add_argument('--decision', action='append', required=True, help='decision cell (at most two)')
    r.add_argument('windows', nargs='+', type=Path)
    b = sub.add_parser('backfill')
    b.add_argument('--archive', type=Path, required=True)
    b.add_argument('windows', nargs='+', type=Path)
    args = parser.parse_args(argv)
    if args.cmd == 'reanalyze':
        for window in args.windows:
            print(json.dumps(reanalyze(window, args.decision), allow_nan=False))
    else:
        print(f'appended {backfill(args.windows, args.archive)} control rows to {args.archive}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
