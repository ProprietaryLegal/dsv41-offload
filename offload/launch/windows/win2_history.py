"""WIN2 history: re-score finished windows under win2-v1 and backfill the control archive (read-only on window dirs).

Two window layouts exist: PC1/UP3/TV1-style `<start>-<cell>.json` with `repetitions` (per repeat: server_decode_ms,
spec_request_rounds, acceptance_per_round, median_decode_tps, median_ttft_s), and NBW-style flat files with one value
per start. Roles come from `summary.json` starts: 'control', 'K' (NBW), 'off' (TV1, feature off) and 'H' (PF2, the
production KV reservation) are controls; every other role is a candidate arm, and each candidate role is judged
separately against the controls (a two-arm window such as R-UP3's dual ring + DEP1 is never pooled).
A cell needs >= 3 values per arm for a win2-v1 verdict; below that the honest answer is INSUFFICIENT_REPEATS, with the
descriptive gain and (n >= 2 per arm) a Welch interval shown for orientation only.

Archive provenance (WIN2-R2, S92 FIX 2): the production commit comes from the summary's `production_commit`, then
`production`, then the window's rollback.json (`production_commit` or `commit`), never from `experiment_commit`. The clock
cap is proved per start (`start_clock_cap`): from that start's own log `cmd/NNNN-<start>-start.log` (start_exp.sh prints
"<node> clock cap N MHz" only after `nvidia-smi -lgc 0,N` succeeded; both Sparks must print it and agree), or, when the
start has no start log, from scratch/CLOCK.log by the start's checkpoint times (clock_source 'clock-log'); a start whose
log lacks the line, whose Sparks disagree, or that straddles a CLOCK.log change is refused.

Prompt identity (WIN2-R3, DP-2): every repeat carries the identity of its streams' `prompt_sha256`
(decision_v2.prompt_identity); re-scoring passes it into every decision series (values of different prompts are never
pooled) and every archive row records it (`prompt_sha256`, `prompt_members`). A control repeat without stream hashes is
refused by backfill; re-scoring a cell without them refuses.
"""
from __future__ import annotations

import argparse
import calendar
import json
import math
import re
import statistics
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple, cast

from offload.launch.windows import decision_v2 as d
from offload.prefill_maxlen.evidence import Json, number, record

CONTROL_ROLES = ('control', 'K', 'off', 'H')
# metric -> higher_is_better; ms_per_round is pooled server time per speculative round.
METRICS = {'ms_per_round': False, 'accepted_per_round': True, 'decode_tps': True, 'ttft_s': False}
CLOCK = re.compile(r'^(spark(?:worker)?|sparkhead|sparkworker) clock cap ([0-9]+) MHz', re.M)
START_LOG = re.compile(r'^[0-9]{4}-(?P<start>.+)-start\.log$')
COMMIT = re.compile('[0-9a-f]{40}')
CLOCK_LOG = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/CLOCK.log')
CLOCK_LOG_LINE = re.compile(r'^(\S+) clock cap ([0-9]+) MHz (20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z)', re.M)
# dsv41.env's GPU_CLOCK_CAP default (2150) applied before CLOCK.log's first entry; CLOCK.log lines of one change (one per
# host) arrive within seconds of each other.
BASE_CLOCK_MHZ = 2150
CHANGE_GROUP_S = 60.0


class Sample(NamedTuple):
    """One measured value of one repeat: `samples()` keeps the first five fields for older callers."""
    cell: str
    arm: str            # 'control' or 'candidate'
    metric: str
    value: float
    max_tokens: int
    start: str
    repeat: int
    role: str
    prompt_sha256: str | None = None          # decision_v2.prompt_identity of the repeat's streams; None without hashes
    prompt_members: tuple[str, ...] = ()      # the streams' prompt_sha256, in stream order


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


def _prompts(rep: dict[str, Json]) -> tuple[str | None, tuple[str, ...]]:
    """(identity, members) of one repeat from its streams' prompt_sha256; (None, ()) when any stream lacks a hash."""
    streams = rep.get('streams')
    shas = [record(s).get('prompt_sha256') for s in streams] if isinstance(streams, list) else []
    if not shas or not all(isinstance(x, str) and d.SHA256.fullmatch(x) for x in shas):
        return None, ()
    members = tuple(str(x) for x in shas)
    return d.prompt_identity(members), members


def repeat_samples(window: Path) -> list[Sample]:
    """Every measured value of every repeat of every start file of a window, with its start, repeat index, role and
    prompt identity."""
    summary = _read(window / 'summary.json')
    starts = record(summary['starts'])
    out: list[Sample] = []
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
            for repeat, rep in enumerate(rows):
                tokens = _max_tokens(rep, path)
                prompt, members = _prompts(rep)
                out.extend(Sample(cell, arm, metric, value, tokens, start, repeat, role, prompt, members)
                           for metric, value in _rep_metrics(rep).items())
    return out


def samples(window: Path) -> list[tuple[str, str, str, float, int]]:
    """(cell, 'control'|'candidate', metric, value, max_tokens) for every repeat of every start file of a window."""
    return [(x.cell, x.arm, x.metric, x.value, x.max_tokens) for x in repeat_samples(window)]


def extract_arms(window: Path) -> dict[str, dict[str, dict[str, list[float]]]]:
    """{cell: {'control' | <candidate role>: {metric: [values...]}}}: candidate roles are kept apart."""
    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for x in repeat_samples(window):
        key = 'control' if x.arm == 'control' else x.role
        out.setdefault(x.cell, {}).setdefault(key, {}).setdefault(x.metric, []).append(x.value)
    return out


def extract_arm_prompts(window: Path) -> dict[str, dict[str, dict[str, list[str | None]]]]:
    """extract_arms' shape with the prompt identity of every value instead of the value (same order)."""
    out: dict[str, dict[str, dict[str, list[str | None]]]] = {}
    for x in repeat_samples(window):
        key = 'control' if x.arm == 'control' else x.role
        out.setdefault(x.cell, {}).setdefault(key, {}).setdefault(x.metric, []).append(x.prompt_sha256)
    return out


def candidate_roles(window: Path) -> list[str]:
    return sorted({x.role for x in repeat_samples(window) if x.arm == 'candidate'})


def extract(window: Path) -> dict[str, dict[str, dict[str, list[float]]]]:
    """{cell: {'control'|'candidate': {metric: [values...]}}} from every start file of a finished window with at most ONE
    candidate role; a window with several candidate arms refuses (use extract_arms: arms are never pooled)."""
    roles = candidate_roles(window)
    if len(roles) > 1:
        raise ValueError(f'{window.name}: candidate arms {roles} must not be pooled; use extract_arms')
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


GUARDS = ('accepted_per_round', 'decode_tps')


def _decide(cell: str, metrics: dict[str, dict[str, list[float]]], primary: str,
            prompts: dict[str, dict[str, list[str | None]]]) -> dict[str, Json] | None:
    """decide_window's cell rule (primary + guards, unpaired: history windows are C/X/C or K/I/K/I/K starts) when the
    primary has >= 3 repeats per arm; guards with fewer repeats are left out. None below three primary repeats. Every
    value carries its prompt identity; a missing or second identity refuses (decide_decision_cell, DP-2)."""
    def ids(arm: str, metric: str) -> tuple[str, ...]:
        found = prompts[arm][metric]
        if any(p is None for p in found):
            raise ValueError(f'{cell}/{metric} {arm}: a repeat without stream prompt hashes cannot be re-scored (DP-2)')
        return tuple(str(p) for p in found)

    def series(metric: str) -> d.Series:
        return d.Series(cell, metric, METRICS[metric], tuple(metrics['control'][metric]), tuple(metrics['candidate'][metric]),
                        paired=False, control_prompts=ids('control', metric), candidate_prompts=ids('candidate', metric))
    def enough(metric: str) -> bool:
        return all(len(metrics.get(arm, {}).get(metric, [])) >= d.MIN_REPEATS for arm in ('control', 'candidate'))
    if not enough(primary):
        return None
    return d.decide_decision_cell([series(primary), *(series(g) for g in GUARDS if g != primary and enough(g))])


def _reanalyze_arm(data: dict[str, dict[str, dict[str, list[float]]]], role: str | None,
                   decision_cells: Sequence[str],
                   prompt_data: dict[str, dict[str, dict[str, list[str | None]]]]) -> tuple[dict[str, Json], str]:
    cells: dict[str, Json] = {}
    verdicts: list[str] = []
    for cell, by_arm in sorted(data.items()):
        arms = {'control': by_arm.get('control', {}), 'candidate': by_arm.get(role, {}) if role is not None else {}}
        ids = prompt_data.get(cell, {})
        prompts = {'control': ids.get('control', {}), 'candidate': ids.get(role, {}) if role is not None else {}}
        metrics: dict[str, Json] = {}
        for metric, higher in METRICS.items():
            control = arms['control'].get(metric, [])
            candidate = arms['candidate'].get(metric, [])
            if not control or not candidate:
                continue
            point = ((statistics.fmean(candidate) - statistics.fmean(control)) / statistics.fmean(control)
                     * (1 if higher else -1))
            metrics[metric] = {'n_control': len(control), 'n_candidate': len(candidate), 'point_gain': point,
                               'welch_interval95': cast(Json, _welch(control, candidate, higher))}
        primary = next((m for m in ('ms_per_round', 'decode_tps', 'ttft_s') if m in metrics), None)
        decides = cell in decision_cells
        cell_verdict = 'NO_DATA'
        if primary is not None:
            decided = _decide(cell, arms, primary, prompts) if decides else None
            for metric, row in metrics.items():
                if decided is not None and metric in record(decided['metrics']):
                    record(row)['verdict'] = record(record(decided['metrics'])[metric])['verdict']
                    record(row)['role'] = 'primary' if metric == primary else 'guard'
                else:
                    record(row)['verdict'] = 'INSUFFICIENT_REPEATS' if decides else 'REPORTED'
            cell_verdict = str(decided['verdict']) if decided is not None else 'INSUFFICIENT_REPEATS'
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
    return cells, status


def reanalyze(window: Path, decision_cells: Sequence[str]) -> dict[str, Json]:
    """Re-score a finished window under win2-v1 (decide_window's cell rule), each candidate arm on its own. One
    candidate arm: 'arm', 'win2_status' and 'cells' at the top level, as before. Several: 'arms' {role: {...}} and a
    'win2_status' of 'PER ARM: <role>=<status>; ...'."""
    decision = list(decision_cells)
    if not decision or len(decision) > d.MAX_DECISION_CELLS or len(set(decision)) != len(decision):
        raise ValueError(f'win2-v1 needs 1..{d.MAX_DECISION_CELLS} distinct decision cells, got {decision!r}')
    data = extract_arms(window)
    prompt_data = extract_arm_prompts(window)
    roles = candidate_roles(window)
    old = record(_read(window / 'summary.json').get('decision') or {}).get('status')
    result: dict[str, Json] = {'window': window.name, 'rule': d.RULE, 'rule_revision': d.RULE_REVISION,
                               'decision_cells': cast(Json, decision), 'historical_status': old}
    if len(roles) <= 1:
        role = roles[0] if roles else None
        cells, status = _reanalyze_arm(data, role, decision, prompt_data)
        result.update({'arm': role, 'win2_status': status, 'cells': cells})
        return result
    arms: dict[str, Json] = {}
    for role in roles:
        cells, status = _reanalyze_arm(data, role, decision, prompt_data)
        arms[role] = {'win2_status': status, 'cells': cells}
    result['arms'] = arms
    result['win2_status'] = 'PER ARM: ' + '; '.join(f'{r}={record(arms[r])["win2_status"]}' for r in roles)
    return result


def _start_logs(window: Path) -> dict[str, list[Path]]:
    logs: dict[str, list[Path]] = {}
    cmd = window / 'cmd'
    for log in sorted(cmd.iterdir()) if cmd.is_dir() else []:
        found = START_LOG.fullmatch(log.name)
        if found is not None:
            logs.setdefault(found.group('start'), []).append(log)
    return logs


def clock_changes(clock_log: Path | None = None) -> list[tuple[float, float, int]]:
    """(first host time, last host time, cap) per coordinator clock change recorded in CLOCK.log, in time order."""
    path = clock_log if clock_log is not None else CLOCK_LOG
    changes: list[tuple[float, float, int]] = []
    for m in CLOCK_LOG_LINE.finditer(path.read_text(errors='replace')):
        when = calendar.timegm(time.strptime(m.group(3), '%Y-%m-%dT%H:%M:%SZ'))
        cap = int(m.group(2))
        if changes and changes[-1][2] == cap and when - changes[-1][0] <= CHANGE_GROUP_S:
            changes[-1] = (changes[-1][0], when, cap)
        elif changes and when < changes[-1][1]:
            raise ValueError(f'{path}: entries out of time order')
        else:
            changes.append((when, when, cap))
    return changes


def clock_log_cap(t0: float, t1: float, clock_log: Path | None = None) -> int:
    """The cap CLOCK.log says applied throughout [t0, t1]; refuses an interval that straddles a change."""
    if not (math.isfinite(t0) and math.isfinite(t1) and t0 <= t1):
        raise ValueError(f'invalid start interval {t0!r}..{t1!r}')
    cap = BASE_CLOCK_MHZ
    for first, last, value in clock_changes(clock_log):
        if last < t0:
            cap = value
        elif first <= t1 and value != cap:
            raise ValueError(f'start interval straddles the CLOCK.log change to {value} MHz at '
                             f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(first))}; clock cap unproven')
        elif first <= t1:
            continue
    return cap


def _start_interval(window: Path, start: str) -> tuple[float, float]:
    times: list[float] = []
    log = window / 'checkpoints.jsonl'
    for line in log.read_text().splitlines() if log.exists() else []:
        row = record(cast(Json, json.loads(line)))
        if start in (row.get('start'), row.get('arm')) and isinstance(row.get('time'), (int, float)):
            times.append(number(row['time']))
    if not times:
        raise ValueError(f'{window.name}/{start}: no start log and no checkpoint times; clock cap unproven')
    return min(times), max(times)


def start_clock_cap(window: Path, start: str, *, clock_log: Path | None = None) -> tuple[int, str]:
    """(cap MHz, clock_source) for one start (S92 FIX 2 item 6-7). 'start-log': that start's own start log, where both
    Sparks must print "<host> clock cap N MHz" with the same N. 'clock-log': the start has no start log, so its checkpoint
    interval is classified against CLOCK.log. Refuses a start log without the line or with disagreeing Sparks, and a
    start that straddles a CLOCK.log change."""
    logs = _start_logs(window).get(start, [])
    if logs:
        nodes: dict[str, set[int]] = {}
        for log in logs:
            for m in CLOCK.finditer(log.read_text(errors='replace')):
                nodes.setdefault(m.group(1), set()).add(int(m.group(2)))
        caps = set().union(*nodes.values()) if nodes else set()
        if len(nodes) < 2 or len(caps) != 1:
            raise ValueError(f'{window.name}/{start}: start log does not prove one clock cap on both Sparks '
                             f'(found {dict((k, sorted(v)) for k, v in sorted(nodes.items()))})')
        return caps.pop(), 'start-log'
    t0, t1 = _start_interval(window, start)
    return clock_log_cap(t0, t1, clock_log), 'clock-log'


def start_clock_caps(window: Path, start: str) -> list[int] | str:
    """Audit form: sorted MHz caps that start's own start log shows on BOTH nodes, else 'unknown' (never refuses)."""
    nodes: dict[str, set[int]] = {}
    for log in _start_logs(window).get(start, []):
        for m in CLOCK.finditer(log.read_text(errors='replace')):
            nodes.setdefault(m.group(1), set()).add(int(m.group(2)))
    if len(nodes) < 2:
        return d.UNKNOWN_CLOCK
    return sorted(set().union(*nodes.values()))


def clock_caps_mhz(window: Path, starts: Iterable[str] | None = None) -> list[int] | str:
    """Window-level AUDIT field (archive `clock_caps_mhz`): the union of the caps the given starts' logs show (default:
    every start that has a start log); 'unknown' if any of them proves none. Rows carry `start_clock_cap` of their own
    start as the required `clock_cap_mhz`."""
    names = list(starts) if starts is not None else list(_start_logs(window))
    caps: set[int] = set()
    for name in names:
        proved = start_clock_caps(window, name)
        if isinstance(proved, str):
            return d.UNKNOWN_CLOCK
        caps.update(proved)
    return sorted(caps) if caps else d.UNKNOWN_CLOCK


def clock_caps(window: Path) -> list[int]:
    """Older API (48bc7a7): the proven window caps, or [] when unproven. Archive rows need `clock_caps_mhz` /
    `start_clock_caps` ('unknown', never [])."""
    caps = clock_caps_mhz(window)
    return [] if isinstance(caps, str) else caps


def production_commit(window: Path) -> str:
    """The 40-hex PRODUCTION commit of a window: summary production_commit, then production, then rollback.json
    (production_commit or commit). Never experiment_commit (S92 FIX 2); refuses when none is recorded."""
    summary = _read(window / 'summary.json')
    found: list[Json] = [summary.get('production_commit'), summary.get('production')]
    rollback = window / 'rollback.json'
    if rollback.exists():
        data = _read(rollback)
        found += [data.get('production_commit'), data.get('commit')]
    for value in found:
        if value is None:
            continue
        if not (isinstance(value, str) and COMMIT.fullmatch(value)):
            raise ValueError(f'{window.name}: recorded production commit {value!r} is not 40-hex')
        return value
    raise ValueError(f'{window.name}: no production commit recorded (summary production_commit/production, rollback.json)')


def window_date(window: Path) -> str:
    """YYYYMMDD from the window directory name (<RUNNER>-...-W-YYYYMMDDTHHMMSSZ)."""
    found = re.search(r'-W-(20[0-9]{6})T[0-9]{6}Z$', window.name)
    if found is None:
        raise ValueError(f'{window.name}: no -W-YYYYMMDDTHHMMSSZ date in the window name')
    return found.group(1)


def backfill(windows: Sequence[Path], archive: Path) -> int:
    """Append control rows; a (window, cell, metric) already archived is skipped, so a re-run adds only what is new.
    A window named twice in one call is read once (S92: one call used to append it twice)."""
    archived = {(r.get('window'), r.get('cell'), r.get('metric')) for r in d.archive_rows(archive)}
    rows: list[dict[str, Json]] = []
    seen: set[str] = set()
    for window in windows:
        if window.name in seen:
            continue
        seen.add(window.name)
        summary = _read(window / 'summary.json')
        if not isinstance(summary.get('starts'), dict) or not summary['starts']:
            continue                       # windows that stopped before any start have no control cells
        found = [x for x in repeat_samples(window) if x.arm == 'control' and (window.name, x.cell, x.metric) not in archived]
        if not found:
            continue
        bare = sorted({f'{x.start}/{x.cell}#{x.repeat}' for x in found if x.prompt_sha256 is None})
        if bare:
            raise ValueError(f'{window.name}: control repeats without stream prompt hashes {bare}; every archive row must '
                             'carry its prompt identity (DP-2)')
        commit, date = production_commit(window), window_date(window)
        caps = {start: start_clock_cap(window, start) for start in sorted({x.start for x in found})}
        audit = cast(Json, clock_caps_mhz(window))
        rows.extend({'window': window.name, 'cell': x.cell, 'metric': x.metric, 'value': x.value, 'max_tokens': x.max_tokens,
                     'commit': commit, 'clock_cap_mhz': caps[x.start][0], 'clock_source': caps[x.start][1],
                     'clock_caps_mhz': audit, 'date': date, 'start': x.start, 'repeat': x.repeat,
                     'prompt_sha256': x.prompt_sha256, 'prompt_members': cast(Json, list(x.prompt_members)),
                     'source': 'win2_history backfill'} for x in found)
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
