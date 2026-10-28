"""Round6 budget: admission proves instrumentation; measurement prices no profiles."""
from __future__ import annotations

import math
from pathlib import Path
from typing import cast

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record

PATH = pf.ROOT/'offload/dspark/probabilistic/pp4/round6-budget.json'
MAX_HOLD_S, ABORT_S = 85*60, 73*60
ALL_CELLS = ('short-b1', '32k-b1', '32k-b4')
B1_CELLS = ALL_CELLS[:2]
RESTORED = 'RESTORED: K1 START VERIFIED, greedy 6/6 IDENTICAL, PRODUCTION HOLD OK, window closed'


def pair_reserve(admission_path: Path, expected: str) -> dict[str, Json]:
    """Use only the successful admission HOLD's own hash-bound physical step.

    There is no pointer, solo receipt, estimate override or old pair-only fallback
    on measurement admission. The600s bound remains planning/admission-only.
    """
    data = ck.read(admission_path)
    ck.require(data.get('schema') == 'pp4-live-admission-v1' and data.get('source_commit') == expected
               and data.get('verdict') == 'QUALIFIED' and not data.get('stop')
               and record(data['restore']).get('status') == RESTORED
               and record(data['restore']).get('restore_deadline_met') is True
               and record(data['post_close_watch']).get('status') == 'PASS', 'successful restored/watched admission HOLD required for measured pair duration')
    evidence = record(data['physical_pair_evidence'])
    admission_run = data.get('admission_run')
    ck.require(isinstance(admission_run, str) and bool(admission_run.strip())
               and evidence.get('run') == admission_run, 'physical pair must match parent admission run')
    path = Path(str(evidence['evidence_path']))
    ck.require(path.is_absolute() and path.parent == admission_path.resolve().parent
               and path.resolve().parent == admission_path.resolve().parent
               and path.name == 'physical-pair-qualification.json' and path.is_file() and pf.digest(path) == evidence['sha256'],
               'admission physical-pair primary evidence absent/changed')
    physical = ck.read(path)
    from offload.dspark.probabilistic.pp4.pair_contract import validate
    from offload.dspark.probabilistic.pp4.pair_qualify import SOURCES
    validate(physical, expected, cast(str, admission_run))
    ck.require(physical['source_sha256'] == {name: pf.digest(pf.ROOT/name) for name in SOURCES},
               'admission pair executed different sources')
    ck.require(record(data['artifact_sha256']).get(path.name) == evidence['sha256'],
               'physical pair is not an artifact of this admission HOLD')
    seconds = number(physical['seconds'])
    ck.require(math.isfinite(seconds) and seconds > 0, 'missing MEASURED admission pair duration')
    return {'seconds': math.ceil(max(180., seconds*1.5)), 'measured_driver_s': seconds,
            'classification': 'ESTIMATED reserve = max(180s, MEASURED admission pair driver +50%) rounded up',
            'source': str(path), 'sha256': pf.digest(path),
            'admission_receipt': str(admission_path), 'admission_receipt_sha256': pf.digest(admission_path)}


def estimate(*, pair: dict[str, Json] | None = None) -> dict[str, Json]:
    data = ck.read(PATH)
    ck.require(data.get('schema') == 'pp4-round6-budget-v1', 'registered budget schema differs')
    plan = record(data['cells'])
    ck.require(set(plan) == set(ALL_CELLS), 'registered budget cell set differs')
    prices: dict[str, float] = {}
    for name, value in plan.items():
        row = record(value)
        observations = ck.rows(row['observations'])
        scale = 4/3 if name == '32k-b4' else 1.
        ttft = max(number(r['max_ttft_s']) for r in observations)*scale
        span = max(number(r['max_decode_span_s']) for r in observations)*scale
        price = math.ceil((ttft+span*511/1499+7)/5)*5
        ck.require(row.get('allowance_per_group_s') == price and row.get('batch_scaling') == scale,
                   'budget must preserve cold TTFT and registered512-gap scaling')
        groups = 2 if name == '32k-b4' else 4
        ck.require(row.get('groups_per_start') == groups and row.get('profile_groups_per_start') == 0,
                   'Round6 retains headline groups and omits measurement profile companions')
        prices[name] = price*groups
    reserve: dict[str, Json] = {'seconds': 600, 'classification': 'ESTIMATED bound for dry planning/admission only; measured admission receipt still required'} if pair is None else pair
    pair_s = number(reserve['seconds'])
    ck.require(math.isfinite(pair_s) and pair_s > 0, 'invalid pair reserve')
    common = (number(data['worker_startup_s'])+max(number(x) for x in cast(list[Json], data['startup_observations_s']))
        +number(data['warm_per_start_s'])+number(data['transition_per_start_s']))

    def price_schedule(names: tuple[str, ...]) -> tuple[dict[str, Json], float]:
        block = common+sum(prices[name] for name in names)
        blocks: dict[str, Json] = {'s1-A': block+number(data['cuts_s']),
                                  's2-B': block+number(data['greedy_on_s']), 's3-A': block}
        total = number(data['setup_s'])+pair_s+sum(number(v) for v in blocks.values())+number(data['restore_s'])
        return blocks, total

    blocks, full_total = price_schedule(ALL_CELLS)
    names = ALL_CELLS if full_total <= MAX_HOLD_S else B1_CELLS
    blocks, total = price_schedule(names)
    return {'classification': data['classification'], 'registration': 'pp4-round6-budget-v1',
        'source_sha256': pf.digest(PATH), 'pair': reserve, 'block_budget_s': blocks,
        'cells': list(names), 'omitted_cells': [] if names == ALL_CELLS else ['32k-b4'],
        'all_cells_total_s': full_total, 'headline_per_start_s': sum(prices[name] for name in names),
        'profile_per_start_s': 0, 'full_round_p95_guard': 'omitted in measurement; instrumentation qualified in admission',
        'total_s': total, 'total_minutes': total/60, 'fits': total <= MAX_HOLD_S,
        'restore_s': data['restore_s'], 'abort_s': ABORT_S, 'restore_deadline_s': MAX_HOLD_S}


def admitted(admission_path: Path, expected: str) -> dict[str, Json]:
    result = estimate(pair=pair_reserve(admission_path, expected))
    ck.require(result['fits'] is True,
               f"registered measurement ESTIMATED {number(result['total_minutes']):.2f}min exceeds85min after B4 omission; refuses before host contact")
    return result
