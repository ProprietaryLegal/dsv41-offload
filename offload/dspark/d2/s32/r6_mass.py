"""S32: aggregate top-20 mass counts of a real decode-smoke chat response (no ids, LPs or text printed).

Usage: python3 -I d2/s32/r6_mass.py <private>/smoke/response-1.json
"""
from __future__ import annotations

import json
import math
import sys


def content(o: object) -> list[dict] | None:
    if isinstance(o, dict):
        c = o.get('content')
        if isinstance(c, list) and c and isinstance(c[0], dict) and 'top_logprobs' in c[0]:
            return c
        o = list(o.values())
    if isinstance(o, list):
        for v in o:
            r = content(v)
            if r:
                return r
    return None


rows = content(json.load(open(sys.argv[1]))) or []
mass = [math.fsum(math.exp(e['logprob']) for e in r['top_logprobs']) for r in rows]
print(json.dumps({'schema': 's32.r6-mass.v1', 'classification': 'MEASURED (real R6 chat rows, aggregates only)',
                  'rows': len(rows), 'entries_per_row': sorted({len(r['top_logprobs']) for r in rows}),
                  'mass_gt_1': sum(m > 1 for m in mass), 'mass_gt_1_plus_1e-6': sum(m > 1 + 1e-6 for m in mass),
                  'top1_lp_exactly_0': sum(max(e['logprob'] for e in r['top_logprobs']) == 0.0 for r in rows),
                  'max_mass_minus_1': max(m - 1 for m in mass) if mass else None}))
