"""CPU-only TV1 window reading. No promotion, host probes, model data or network."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import cast

from offload.gb10_budget import trace_budget as tb
from offload.launch.windows import tv1_window as tv
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record
from offload.tv1 import metrics, cells, greedy

THRESHOLDS = {'off_round_spread': .02, 'off_ttft_spread': .03, 'profiler_overhead_max': .10,
              'candidate_150k_round_gain': .02, 'compact_b4_round_gain': .02, 'compact_b4_client_rate_gain': .05,
              'other_round_regression_max': .02, 'client_rate_regression_max': .02,
              'ttft_regression_max': .03}


def client_cell(path: Path) -> dict[str, Json]:
    cell = ck.read(path)
    ck.require(cell.get('type') == 'tv1-repeated-cell-v3' and cell.get('profiled') is False,
               'TV1 repeated unprofiled control required')
    trials = ck.rows(cell['trials'])
    acceptance = metrics.pool(trials)
    expected_batch = 4 if str(cell['cell']).endswith('b4') else 1
    ck.require(len(trials) == 4//expected_batch and all(t.get('batch') == expected_batch for t in trials),
               'TV1 repeated cell must contain four requests at the requested batch')
    ttfts: list[float] = []
    prompts: list[Json] = []
    durations: list[float] = []
    tokens = 0.
    for index, trial in enumerate(trials):
        ck.require(trial.get('cell') == cell['cell'] and trial.get('profiled') is False
                   and trial.get('repetition') == index and trial.get('forced_output_tokens') == 512,
                   'TV1 control trial identity/protocol mismatch')
        streams = ck.rows(trial['streams'])
        ck.require(all(s.get('finish') == 'length' and record(s['usage']).get('completion_tokens') == 512 for s in streams),
                   'TV1 fixed output budget incomplete')
        ttft = [number(s['ttft_s']) for s in streams]
        starts = [number(s['submitted_rel_s']) for s in streams]
        ends = [a+number(s['total_s']) for a,s in zip(starts,streams,strict=True)]
        duration = max(ends)-min(a+b for a,b in zip(starts,ttft,strict=True))
        ck.require(math.isfinite(duration) and duration > 0 and all(math.isfinite(x) and x > 0 for x in ttft),
                   'TV1 invalid client durations')
        durations.append(duration)
        tokens += sum(number(record(s['usage'])['completion_tokens'])-1 for s in streams)
        ttfts.extend(ttft)
        prompts.extend(s['prompt_sha256'] for s in streams)
    return {'ttft_s': statistics.fmean(ttfts), 'client_aggregate_decode_tps': tokens/sum(durations),
            'client_decode_span_s': sum(durations), 'ttft_spread_s': max(ttfts)-min(ttfts),
            'server_decode_ms_per_spec_request_round': acceptance['server_decode_ms_per_spec_request_round'],
            'acceptance': acceptance, 'prompts': prompts}


def read_arm(root: Path, feature: str) -> dict[str, Json]:
    budget = tb.analyze_window(root)
    checked = ck.read(root / 'greedy.json')
    greedy.validate(checked)
    overhead = record(budget['overhead']).get('decode_overhead')
    ck.require(overhead is not None and number(overhead) <= THRESHOLDS['profiler_overhead_max'],
               'TV1 profiler overhead missing or above 10%')
    out: dict[str, Json] = {}
    captures = record(budget['captures'])
    for name in cells.controls(feature):
        capture = record(captures['prof-' + name + ':decode'])
        ck.require(capture.get('ranks_present') == [0, 1] and capture.get('aligned') is True,
                   'TV1 complete aligned two-rank decode capture required')
        periods: list[float] = []
        for r in record(capture['ranks']).values():
            rank = record(r)
            ck.require(rank.get('graph_visibility') not in ('invisible', 'partial'), 'TV1 graph kernels invisible')
            aggregate = record(rank['aggregate'])
            ck.require(number(aggregate['coverage']) >= .90, 'TV1 trace coverage <90%')
            steps = [record(x) for x in cast(list[Json], rank['steps']) if record(x).get('complete') is True]
            expected = 4 if name.endswith('b4') else 1
            ck.require(len(steps) >= 8 and all(s.get('phase') == 'decode' and s.get('generation_requests') == expected for s in steps),
                       'TV1 profile contains prefill or a surviving-request tail, not the requested batch')
            periods.append(number(aggregate['period_ms_median']))
        client = client_cell(root / 'cells' / ('ctl-' + name + '.json'))
        out[name] = {**client, 'profiled_round_ms': statistics.fmean(periods),
                     'rank_profiled_round_ms': cast(Json, periods)}
    return {'cells': out, 'profiler_overhead': overhead, 'trace_budget': budget, 'greedy': checked}


def compare(feature: str, measured: list[dict[str, Json]]) -> dict[str, Json]:
    ck.require(feature in ('candidate', 'compact') and len(measured) == 3, 'TV1 needs fixed OFF/ON/OFF')
    names = cells.controls(feature)
    ck.require(all(set(record(a['cells'])) == set(names) for a in measured),
               'TV1 exact target plus no-regression cell set required')
    failures: list[Json] = []
    drift: dict[str, Json] = {}
    for name in names:
        points = [record(record(a['cells'])[name]) for a in measured]
        ck.require(all(p['prompts'] == points[0]['prompts'] for p in points), 'TV1 prompt pairing hash mismatch')
        off = points[::2]
        for point in points:
            pooled = record(point['acceptance'])
            ck.require(number(pooled['requests']) >= 4 and number(pooled['rounds']) > 0,
                       'TV1 decision requires acceptance pooled over >=4 requests')
            ck.require(all(math.isfinite(number(point[key])) and number(point[key]) > 0 for key in
                       ('server_decode_ms_per_spec_request_round', 'ttft_s', 'client_aggregate_decode_tps')),
                       'TV1 decision metrics invalid')
        step = [number(p['server_decode_ms_per_spec_request_round']) for p in off]
        ttft = [number(p['ttft_s']) for p in off]
        step_spread = max(step)/min(step)-1
        ttft_spread = max(ttft)/min(ttft)-1
        drift[name] = {'off_round_spread': step_spread, 'off_ttft_spread': ttft_spread}
        if step_spread > THRESHOLDS['off_round_spread'] or ttft_spread > THRESHOLDS['off_ttft_spread']:
            failures.append(name + ': OFF drift exceeds pre-registered bound')
    pairs: list[Json] = []
    # The single ON must beat EACH bracketing OFF independently. An average
    # could hide a failed comparison; this reduced design has no replication claim.
    for control_index in (0, 2):
        index = 1
        row: dict[str, Json] = {}
        for name in names:
            center = record(record(measured[index]['cells'])[name])
            neighbors = [record(record(measured[control_index]['cells'])[name])]
            baseline = {key: statistics.fmean(number(p[key]) for p in neighbors)
                        for key in ('server_decode_ms_per_spec_request_round', 'ttft_s', 'client_aggregate_decode_tps')}
            round_gain = 1-number(center['server_decode_ms_per_spec_request_round'])/baseline['server_decode_ms_per_spec_request_round']
            client_gain = number(center['client_aggregate_decode_tps'])/baseline['client_aggregate_decode_tps']-1
            ttft_change = number(center['ttft_s'])/baseline['ttft_s']-1
            pooled = record(center['acceptance'])
            adjacent = [record(p['acceptance']) for p in neighbors]
            baseline_accepted = sum(number(p['accepted_tokens']) for p in adjacent)/sum(number(p['rounds']) for p in adjacent)
            accepted = number(pooled['accepted_tokens'])/number(pooled['rounds'])
            acceptance_rate_change = (1+accepted)/(1+baseline_accepted)/(1-round_gain)-1
            if acceptance_rate_change < -THRESHOLDS['client_rate_regression_max']:
                failures.append(f'ON{index} {name}: pooled acceptance-adjusted server rate regression')
            row[name] = {'server_decode_request_round_gain': round_gain, 'unprofiled_client_rate_gain': client_gain,
                         'unprofiled_ttft_change': ttft_change, 'acceptance': pooled,
                         'neighbor_pooled_accepted_per_round': baseline_accepted,
                         'pooled_acceptance_adjusted_server_rate_change': acceptance_rate_change}
            if ttft_change > THRESHOLDS['ttft_regression_max']:
                failures.append(f'ON{index} {name}: TTFT regression')
            if round_gain < -THRESHOLDS['other_round_regression_max'] or client_gain < -THRESHOLDS['client_rate_regression_max']:
                failures.append(f'ON{index} {name}: decode regression')
            if feature == 'candidate' and name == '150k-b1' and round_gain < THRESHOLDS['candidate_150k_round_gain']:
                failures.append(f'ON{index}: candidate 150K round gain below 2%')
            if feature == 'compact' and name == '32k-b4':
                if round_gain < THRESHOLDS['compact_b4_round_gain']:
                    failures.append(f'ON{index}: compact B4 server request-round gain below 2%')
                if client_gain < THRESHOLDS['compact_b4_client_rate_gain']:
                    failures.append(f'ON{index}: compact B4 client rate gain below 5%')
        pairs.append({'off_start_index': control_index, 'cells': row})
    return {'verdict': 'WINDOW_PASS_FOR_REVIEW' if not failures else 'HOLD', 'failures': failures,
            'classification': 'MEASURED C/arm/C: one ON against each OFF separately; >=4-request pooled acceptance; no independent ON replication or automatic promotion',
            'design': 'C/arm/C-v4',
            'round_time_scope': metrics.SCOPE,
            'thresholds': cast(Json, THRESHOLDS), 'off_drift': drift, 'paired': pairs}


def analyze(root: Path) -> dict[str, Json]:
    summary = ck.read(root / 'summary.json')
    ck.require(summary.get('registration_id') == greedy.REGISTRATION_ID and summary.get('greedy_rule') == greedy.RULE,
               'TV1 experimental greedy registration missing/different; historical strict runs retain their verdict')
    ck.require(summary.get('status') == 'COMPLETE' and 'stop' not in summary
               and str(record(summary['restore']).get('status', '')).startswith('RESTORED'),
               'TV1 incomplete window or unverified W8 restore')
    ck.require(record(summary.get('post_close_watch') or {}).get('status') == 'PASS',
               'TV1 post-close production watch missing/failed')
    ck.require(summary.get('analysis_sha256') == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'TV1 pre-registered analysis code differs from the window pin')
    artifacts = record(summary.get('artifact_sha256') or {})
    ck.require(bool(artifacts), 'TV1 artifact hashes missing')
    for name, expected in artifacts.items():
        path = (root / name).resolve()
        ck.require(path.is_relative_to(root.resolve()) and path.is_file(), 'TV1 unsafe/missing artifact')
        ck.require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'TV1 artifact hash drift: ' + name)
    feature = str(summary['feature'])
    ck.require(summary.get('design') == 'C/arm/C-v4'
               and list(record(summary['starts'])) == [a.name for a in tv.arms(feature)],
               'TV1 fixed C/arm/C starts missing, extra or reordered')
    if feature == 'candidate':
        for arm in tv.arms(feature):
            if arm.role != feature:
                continue
            for host in ('sparkhead', 'sparkworker-wired'):
                oracle = ck.read(root / 'arms' / arm.name / ('oracle-' + host + '.json'))
                ck.require(oracle.get('all_identical') is True and oracle.get('classification') == 'MEASURED SM121',
                           'TV1 installed SM121 oracle is not qualified')
    measured = [read_arm(root / 'arms' / a.name, feature) for a in tv.arms(feature)]
    result = compare(feature, measured)
    result['starts'] = cast(Json, measured)
    result.update({'registration_id': greedy.REGISTRATION_ID, 'greedy_rule': greedy.RULE})
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--window', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('TV1 analysis output must be new')
    result = analyze(a.window)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(result['verdict'])


if __name__ == '__main__':
    main()
