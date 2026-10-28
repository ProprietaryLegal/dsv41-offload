"""CPU-only amendment-7 reference receipts. No endpoint or generator access.

Counters are cumulative since the 02:55Z engine restart, NOT snapshot deltas.
The coordinator supplies the pause/rollback times; filenames alone cannot prove
that a snapshot was taken after pause and before rollback.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from datetime import datetime, timezone
from typing import TypedDict

PREFIX = 'vllm:spec_decode_'
NAMES = ('num_drafts', 'num_draft_tokens', 'num_accepted_tokens')
RESTART = '2026-10-25T02:55:00Z'


class Snapshot(TypedDict):
    counters: dict[str, float]
    starts: dict[str, float]


class Reference(TypedDict):
    schema: str
    pass_id: str
    synthetic: bool
    restart: str
    paused_at: str
    rollback_at: str
    sources: dict[str, str]
    snapshots: list[str]
    rates: list[str]
    counters: dict[str, float]
    position0_acceptance: float
    per_position_acceptance: list[float]
    accepted_per_round: float
    mean_depth: float
    foreign_share: float
    generator_tokens: int
    status: str
    sha256: str


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp(value: str) -> float:
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('UTC/offset timestamp required')
    return dt.timestamp()


def file_time(path: Path, prefix: str, suffix: str) -> float:
    match = re.fullmatch(prefix+r'-(\d{8}T\d{6}Z)\.'+suffix, path.name)
    if not match:
        raise ValueError('timestamped snapshot filename required: '+str(path))
    return datetime.strptime(match[1], '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).timestamp()


def parse_metrics(path: Path) -> Snapshot:
    counters: dict[str, float] = {}
    starts: dict[str, float] = {}
    identity: tuple[str, str] | None = None
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        m = re.fullmatch(r'([\w:]+)(?:\{(.*)\})?\s+(\S+)(?:\s+\d+)?', line)
        if not m:
            raise ValueError('invalid Prometheus metric line in '+str(path))
        name, raw, val = m.groups()
        if not (name.startswith(PREFIX) or name == 'vllm:generation_tokens_total'):
            continue
        if not (name.endswith('_total') or name.endswith('_created')):
            continue
        labels = dict(re.findall(r'(\w+)="([^"\\]*)"', raw or ''))
        current = (labels.get('engine', ''), labels.get('model_name', ''))
        if not all(current) or (identity is not None and identity != current):
            raise ValueError('requires one consistent engine/model series')
        identity = current
        key = name
        if 'per_pos' in name:
            pos = labels.get('position')
            if pos not in ('0', '1', '2', '3', '4'):
                raise ValueError('invalid draft position')
            key += ':'+pos
        target = counters if name.endswith('_total') else starts
        value = float(val)
        if key in target or not math.isfinite(value) or value < 0:
            raise ValueError('duplicate/invalid counter: '+key)
        target[key] = value
    required = {PREFIX+n+'_total' for n in NAMES} | {'vllm:generation_tokens_total'}
    required |= {PREFIX+'num_accepted_tokens_per_pos_total:'+str(i) for i in range(5)}
    if set(counters) != required:
        raise ValueError('missing or unexpected speculative counters')
    if not {PREFIX+n+'_created' for n in NAMES} <= set(starts):
        raise ValueError('missing engine counter creation times')
    # The API process may predate the engine restart; engine counters identify it.
    if any(not timestamp(RESTART) <= v < timestamp(RESTART)+60 for v in starts.values()):
        raise ValueError('snapshot is not from the 02:55Z engine start (rollback restart forbidden)')
    return {'counters': counters, 'starts': starts}


def build_reference(snapshots: list[Path], rates: list[Path], paused_at: str, rollback_at: str) -> Reference:
    if len(snapshots) < 2 or not rates or len(set(snapshots)) != len(snapshots) or len(set(rates)) != len(rates):
        raise ValueError('at least two distinct snapshots and matching final generator rate required')
    snapshots = sorted((p.resolve() for p in snapshots), key=lambda p: file_time(p, 'metrics', 'prom'))
    rates = sorted((p.resolve() for p in rates), key=lambda p: file_time(p, 'rate', 'json'))
    if len(set(snapshots)) != len(snapshots) or len({file_time(p, 'metrics', 'prom') for p in snapshots}) != len(snapshots):
        raise ValueError('distinct chronologically ordered snapshots required')
    sources = {str(p): file_hash(p) for p in [*snapshots, *rates]}
    final_time = file_time(snapshots[-1], 'metrics', 'prom')
    if not timestamp(RESTART) < timestamp(paused_at) <= final_time < timestamp(rollback_at):
        raise ValueError('final snapshot must be after pause and before rollback restart')
    previous: Snapshot | None = None
    for path in snapshots:
        current = parse_metrics(path)
        if previous and (current['starts'] != previous['starts'] or any(
                v < previous['counters'][k] for k, v in current['counters'].items())):
            raise ValueError('engine start changed or counters reset')
        previous = current
    assert previous is not None
    c = previous['counters']
    total = c['vllm:generation_tokens_total']
    rounds, depth, accepted = [c[PREFIX+n+'_total'] for n in NAMES]
    positions = [c[PREFIX+'num_accepted_tokens_per_pos_total:'+str(i)] for i in range(5)]
    if (rounds <= 0 or total <= 0 or not rounds <= depth <= 5*rounds
            or sum(positions) != accepted or accepted > depth
            or positions[0] > rounds or positions != sorted(positions, reverse=True)):
        raise ValueError('inconsistent engine counters')
    gen = -1
    snapshot_times = {file_time(p, 'metrics', 'prom') for p in snapshots}
    for rate in rates:
        t = file_time(rate, 'rate', 'json')
        row = json.loads(rate.read_text())
        value = row['gen_tokens']
        if (t not in snapshot_times or not 0 <= t-timestamp(row['t']) <= 120
                or type(value) is not int or value < 0 or value < gen):
            raise ValueError('unmatched/stale/nonmonotonic generator rate row')
        gen = value
    if file_time(rates[-1], 'rate', 'json') != final_time or gen > total:
        raise ValueError('final rate missing or generator tokens exceed engine total')
    foreign = (total-gen)/total
    result: Reference = {'schema': 'd2.live-reference.v1', 'pass_id': 'd2pass2-r1', 'synthetic': False,
        'restart': RESTART, 'paused_at': paused_at, 'rollback_at': rollback_at, 'sources': sources,
        'snapshots': [str(p) for p in snapshots], 'rates': [str(p) for p in rates], 'counters': c,
        'position0_acceptance': positions[0]/rounds, 'per_position_acceptance': [p/rounds for p in positions],
        'accepted_per_round': accepted/rounds, 'mean_depth': depth/rounds, 'foreign_share': foreign,
        'generator_tokens': gen, 'status': 'INCONCLUSIVE' if foreign > .02 else 'VALID', 'sha256': ''}
    result['sha256'] = digest({k: v for k, v in result.items() if k != 'sha256'})
    if sources != {str(p): file_hash(p) for p in [*snapshots, *rates]}:
        raise ValueError('reference sources changed while reading')
    return result


def verify_reference(path: Path) -> Reference:
    receipt = json.loads(path.read_text())
    actual = build_reference([Path(p) for p in receipt['snapshots']], [Path(p) for p in receipt['rates']],
                             receipt['paused_at'], receipt['rollback_at'])
    if receipt != actual:
        raise ValueError('live reference receipt or snapshot hash changed')
    return actual
