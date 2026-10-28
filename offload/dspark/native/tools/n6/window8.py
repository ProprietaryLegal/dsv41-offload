"""Recompute aggregate-only W8 acceptance and request-count-cap evidence.

No prompts, tokens, generated text or per-request hashes are copied to output.
The input JSONL files remain in the coordinator's completed result directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import TypedDict, cast


class Spec(TypedDict):
    rounds: float
    proposed_tokens: float
    accepted_tokens: float
    mean_accepted_per_round: float
    accepted_per_position: dict[str, float]


class Run(TypedDict):
    type: str
    concurrency: int
    mixed: bool
    rep: int
    spec: Spec
    aggregate_tps: float
    error: str | None


class Arm(TypedDict):
    runs: int
    rounds: float
    accepted: float
    proposed: float
    mean_run_acceptance: float
    sd_run_acceptance: float
    acceptance_per_round: float
    proposals_per_round: float
    accepted_fraction: float
    survival_per_position: list[float]
    replicate_acceptance: list[float]
    aggregate_tps: float


def summarize(runs: list[Run]) -> Arm:
    if len(runs) < 2 or any(r['error'] is not None for r in runs):
        raise ValueError('need at least two successful complete run windows')
    specs = [r['spec'] for r in runs]
    rounds = sum(s['rounds'] for s in specs)
    proposed = sum(s['proposed_tokens'] for s in specs)
    accepted = sum(s['accepted_tokens'] for s in specs)
    if min(rounds, proposed) <= 0:
        raise ValueError('empty acceptance denominator')
    for s in specs:
        positions = [s['accepted_per_position'].get(str(i), 0.0) for i in range(5)]
        if abs(sum(positions) - s['accepted_tokens']) > 1e-6:
            raise ValueError('accepted positions do not sum to total')
        if any(a < b for a, b in zip(positions, positions[1:])):
            raise ValueError('accepted positions are not prefix survival counts')
    rates = [s['accepted_tokens'] / s['rounds'] for s in specs]
    return dict(runs=len(runs), rounds=rounds, accepted=accepted, proposed=proposed,
                mean_run_acceptance=statistics.mean(rates), sd_run_acceptance=statistics.stdev(rates),
                acceptance_per_round=accepted/rounds, proposals_per_round=proposed/rounds,
                accepted_fraction=accepted/proposed,
                survival_per_position=[sum(s['accepted_per_position'].get(str(i), 0.0) for s in specs)/rounds
                                       for i in range(5)],
                replicate_acceptance=rates, aggregate_tps=statistics.mean(r['aggregate_tps'] for r in runs))


def collect(root: Path) -> dict[str, object]:
    sources = {}
    arms: dict[str, dict[str, Arm]] = {}
    counters: dict[str, dict[str, object]] = {}
    for arm in ('C1', 'X', 'C2'):
        path = root / f'{arm}-c32.jsonl'
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Read only the numeric run summary; ignore all request records.
        records = [cast(Run, json.loads(line)) for line in path.read_text().splitlines()
                   if json.loads(line).get('type') == 'run']
        arms[arm] = {}
        for b in (1, 2, 3):
            arms[arm][f'B{b}'] = summarize([r for r in records if r['concurrency'] == b and not r['mixed']])
        stats_path = root / f'stats-{arm}.json'
        sources[stats_path.name] = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        stats = json.loads(stats_path.read_text())
        counts = stats['counters']
        if stats['n_out_by_reqs'] != [0, 5, 5, 4, 4]:
            raise ValueError('unexpected W8 cap table')
        counters[arm] = dict(snapshot_rounds=stats['rounds'], cut_engine_rounds=counts['rounds_n_out_cut'],
                             cut_fraction=counts['rounds_n_out_cut']/stats['rounds'],
                             decode_rounds=counts['rounds_decode'], prefill_rounds=counts['rounds_prefill'],
                             usable_request_entries=counts['req_status_ok'],
                             rtt_p50_us=stats['latency']['rtt_us']['p50_us'],
                             n_out_by_reqs=stats['n_out_by_reqs'], n_out_reason=stats['n_out_reason'])
    deltas = {}
    for batch in ('B1', 'B2', 'B3'):
        c1, x, c2 = (arms[a][batch] for a in ('C1', 'X', 'C2'))
        control_pos = [(a+b)/2 for a, b in zip(c1['survival_per_position'], c2['survival_per_position'])]
        position_delta = [a-b for a, b in zip(x['survival_per_position'], control_pos)]
        control_mean = (c1['mean_run_acceptance'] + c2['mean_run_acceptance'])/2
        # Diagnostic cluster SE from the two run means per arm; NOT an iid-round
        # significance test or a qualified CI with only two temporally blocked runs.
        cluster_se = (x['sd_run_acceptance']**2/x['runs'] +
                      c1['sd_run_acceptance']**2/(4*c1['runs']) +
                      c2['sd_run_acceptance']**2/(4*c2['runs']))**0.5
        delta = x['mean_run_acceptance'] - control_mean
        deltas[batch] = dict(mean_run_acceptance_delta=delta,
                            control_return_drift=c2['mean_run_acceptance']-c1['mean_run_acceptance'],
                            diagnostic_cluster_se=cluster_se, diagnostic_delta_over_se=delta/cluster_se,
                            pooled_position_delta=position_delta,
                            pooled_position_delta_first_four=sum(position_delta[:4]),
                            pooled_acceptance_delta=sum(position_delta),
                            x_vs_control_proposals_per_round=x['proposals_per_round']-
                            (c1['proposals_per_round']+c2['proposals_per_round'])/2)
    return dict(label='MEASURED from completed Window 8 files; no new serving requests',
                sources_sha256=sources, arms=arms, deltas=deltas, cap_counters=counters,
                uncertainty='Two run windows per cell; cluster SE diagnostic only, no independence/CI claim',
                censoring='Position totals are already survival counts under cap/confidence policy. '
                          'No verified/reached denominator or raw confidence by batch is available.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(collect(args.root), indent=2)+'\n')


if __name__ == '__main__':
    main()
