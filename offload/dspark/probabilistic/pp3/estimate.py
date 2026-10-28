"""Conditional cost sensitivities, never a live speed forecast.

Known costs are solo SM86 proxies. Unknown K2/eager/peer/GB10/proxy work is
explicitly null; every displayed rate names an assumed extra cost. Fixed5
replay populations and historical adaptive windows stay separate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def wire_bytes(batch: int, depth: int) -> int:
    if batch not in range(1,5) or depth not in (4,5):
        raise ValueError('B1..B4 and fixed depth4/5 required')
    return 32 + 32 + 8*batch + 8*batch*depth + 16 + 4*batch + 8*batch*depth*32 + 4


def rate(retained: float, base_ms: float, delta: float, known_ms: float,
         unknown_ms: float) -> float:
    values = (retained, base_ms, delta, known_ms, unknown_ms)
    if not all(math.isfinite(x) for x in values) or retained <= 0 or base_ms <= 0 or min(known_ms,unknown_ms) < 0:
        raise ValueError('invalid cost sensitivity inputs')
    return (retained+delta)*1000/(base_ms+known_ms+unknown_ms)


def estimate(producer: dict[str, Any], gpu: dict[str, Any], pp2: dict[str, Any]) -> dict[str, Any]:
    if not producer['complete'] or not gpu['complete']:
        raise ValueError('incomplete GPU receipts cannot price a candidate')
    row = next(r for r in producer['rows'] if r['batch']==1 and r['depth']==5)
    parts: dict[str, float] = {
        'selector_addition_solo_synthetic_p95_ms': row['sampled']['p95_ms'],
        'sparse1280_D2H_solo_p95_ms': gpu['sparse1280_d2h']['p95_ms'],
        'sparse1280_H2D_solo_p95_ms': gpu['sparse1280_h2d']['p95_ms'],
        'full_dense_p_standard_verifier_solo_p95_ms': gpu['dense_p_verifier']['p95_ms'],
        'dense_q_expand_solo_p95_ms': gpu['dense_q_expansion']['p95_ms'],
        'two_serialized_framed_replies_40gbps_ms': 2*wire_bytes(1,5)*8/(pp2['wire_gbps']*1e6),
    }
    known = sum(parts.values())
    fixed: dict[str, Any] = {}
    for key, view in pp2['views'].items():
        retained = 1+view['greedy_EA']
        delta = view['delta_EA']
        interval = view['CI95_delta']
        base = pp2['fixed5_archived_price_ms']
        fixed[key] = {
            'retained_greedy': retained, 'retained_sampled': retained+delta,
            'archived_price_ms_not_live_mean': base,
            'break_even_total_added_ms': base*delta/retained,
            'break_even_total_added_ms_CI_lower': None if interval is None else base*interval[0]/retained,
            'remaining_unmeasured_budget_ms_CI_lower': None if interval is None else base*interval[0]/retained-known,
            'true_net_rate_change': None,
            'sensitivity': [{'assumed_unmeasured_ms': u,
                'conditional_tps': rate(retained,base,delta,known,u),
                'conditional_pct_change': (rate(retained,base,delta,known,u)/(retained*1000/base)-1)*100}
                for u in (0.,.1,.5,1.,2.,3.)],
        }
    historical = []
    for cell in pp2['w8_historical_adaptive']:
        name = cell['cell']
        # Only a conditional extrapolation: fixed5 replay gains do not describe
        # adaptive current serving. At150K there are no acceptance captures.
        view_key = 'context:<8K' if name.startswith('short') else 'context:16-32K' if name.startswith('32k') else 'all'
        view = pp2['views'][view_key]
        relative_gain = view['delta_EA']/(1+view['greedy_EA'])
        retained = 1+cell['historical_accepted_per_round']
        base = cell['historical_step_p50_ms']
        historical.append({'cell':name, 'historical_adaptive_baseline_tps':retained*1000/base,
            'acceptance_proxy':view_key, 'measured_acceptance_at150k':None if name.startswith('150k') else 'different fixed-depth replay population',
            'extra_assumption':'transplant relative fixed5 token-count gain to adaptive window; 150K extrapolates pooled <=47K anchors',
            'expected_live_tps':None,
            'sensitivity':[{'assumed_unmeasured_ms':u,
                'conditional_tps':rate(retained,base,retained*relative_gain,known,u),
                'conditional_pct_change':(rate(retained,base,retained*relative_gain,known,u)/(retained*1000/base)-1)*100}
                for u in (0.,.5,1.,2.,3.)]})
    return {'schema':'pp3.cost-sensitivity.v1','classification':'ESTIMATED conditional sensitivities from MEASURED proxies',
        'decision':'STOP: absolute selector target missed; full K2/pair/GB10 costs and long-context acceptance unmeasured',
        'single_launch_goal_met':False, 'selector_absolute_005ms_goal_met':row['sampled']['p95_ms']<=.05,
        'parts':parts,'known_proxy_sum_ms':known,'additional_unmeasured_ms':None,
        'unknown_components':['extra Markov materialization and duplicate greedy selector work',
            'eager graph/CPU launch gaps in sampled full K2 path','full-vocabulary worker peer copies and completion waits',
            'CPU CRC/q validation/proxy queue and registered pool overhead','full dense-q slot index-copy/TP synchronization',
            'GB10 H2D/expansion/rejection differences','policy-induced future occupancy and150K acceptance'],
        'fixed5_replay_sensitivities':fixed,'historical_adaptive_sensitivities':historical,
        'interpretation':'No unknown component is assigned zero in the decision. u=0 rows are illustrative only; no speedup is claimed.'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=Path(__file__).parent)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    files=('producer-final.json','gpu-qualification.json','pp2-frozen.json')
    result=estimate(*(json.loads((a.directory/name).read_text()) for name in files))
    a.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
