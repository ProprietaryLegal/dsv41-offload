"""Pinned OPS1/K1 candidate geometry and strict qualification receipts (stdlib only).

Sources: official-1017.json text_config; S8.report.md N-3 real allocator views;
K1 kv_arith.py STRIDE=138240, paired kernel divides manager stride by three.
All four consumers share layer20's ratio-one index cache, not ratio-two widths.
"""
from __future__ import annotations

from typing import TypedDict

LAYERS = (24, 28, 32, 36)
RAW_CONTEXTS = (1024, 8193, 32000, 150000, 327679, 327680)
LAYOUTS = ('single', 'native2', 'flat3', 'flat4', 'flat6', 'flat6-padded')
NUMERICS = ('random', 'tie', 'near-tie')
PAGE = 64
STRIDE = 138240 // 3
CACHE_OFFSET = 37440
MAX_RAW = 327680
HIGH_PAGE = 47000
EXCLUSIONS = ['packed-ragged-native: dense fallback', 'varlen-indices: dense fallback',
              'FP4/DCP/PCP: refused', 'native-next_n>2: refused on SM121']


class CaseSpec(TypedDict):
    id: str
    layer: int
    batch: int
    raw_context: int
    compress_ratio: int
    visible_width: int
    width: int
    rows: int
    next_n: int
    expansion: int
    layout: str
    numeric: str
    page: int
    stride: int
    cache_offset: int
    high_page: int
    candidate_capacity: int
    block_size: int
    heads: int
    dim: int
    topk: int
    backend: str


def case_matrix() -> list[CaseSpec]:
    out: list[CaseSpec] = []
    for li, layer in enumerate(LAYERS):
        ratio = 1  # Config compress_ratios[20:40]; shared source layer20.
        for batch in range(1, 5):
            for ci, raw in enumerate(RAW_CONTEXTS):
                layout = LAYOUTS[(ci + li + batch - 1) % len(LAYOUTS)]
                expansion = {'single': 1, 'native2': 2, 'flat3': 3,
                             'flat4': 4, 'flat6': 6, 'flat6-padded': 6}[layout]
                rows = batch * expansion + (1 if layout.endswith('padded') else 0)
                next_n = 2 if layout == 'native2' else 1
                for numeric in NUMERICS:
                    identity = f'L{layer}-B{batch}-R{raw}-{layout}-{numeric}'
                    out.append({'id': identity, 'layer': layer, 'batch': batch,
                                'raw_context': raw, 'compress_ratio': ratio,
                                'visible_width': raw // ratio, 'width': MAX_RAW // ratio,
                                'rows': rows, 'next_n': next_n, 'expansion': expansion,
                                'layout': layout, 'numeric': numeric, 'page': PAGE,
                                'stride': STRIDE, 'cache_offset': CACHE_OFFSET,
                                'high_page': HIGH_PAGE, 'candidate_capacity': 2048,
                                'block_size': 8, 'heads': 32, 'dim': 128,
                                'topk': 512, 'backend': 'persistent'})
    return out


def validate_oracle_receipt(receipt: object) -> None:
    """Reject old six-row receipts, substitutions, omissions, duplicates and false flags."""
    if not isinstance(receipt, dict) or receipt.get('type') != 'tv1-installed-oracle-v2':
        raise ValueError('TV1 candidate oracle receipt type mismatch')
    if (receipt.get('classification') != 'MEASURED SM121'
            or receipt.get('all_identical') is not True or receipt.get('capability') != [12, 1]
            or receipt.get('exclusions') != EXCLUSIONS):
        raise ValueError('TV1 candidate oracle qualification incomplete')
    expected = case_matrix()
    cases = receipt.get('cases')
    if not isinstance(cases, list) or len(cases) != len(expected):
        raise ValueError('TV1 candidate oracle matrix incomplete')
    for actual, spec in zip(cases, expected, strict=True):
        if (not isinstance(actual, dict) or set(actual) != set(spec) | {'eager_identical', 'replay_identical'}
                or any(type(actual.get(k)) is not type(v) or actual.get(k) != v for k, v in spec.items())
                or actual.get('eager_identical') is not True or actual.get('replay_identical') is not True):
            raise ValueError(f'TV1 candidate oracle case mismatch: {spec["id"]}')


def validate_engine_receipts(receipts: object) -> None:
    """Require real forwards at every consumer/B, plus long-context B4 before timing.

    Every per-rank list must include >=147000 occupied positions for B1/B4,
    and >=31360 for B2/B3 (the existing public prompt cuts allow 2% undershoot). B4 also requires a separate 32K bucket. Graph entries cannot substitute for checked eager forwards.
    """
    if not isinstance(receipts, list):
        raise ValueError('TV1 engine receipts must be a list')
    valid: set[tuple[int, int, str]] = set()
    for r in receipts:
        if not isinstance(r, dict) or r.get('type') != 'tv1-engine-oracle-v2':
            raise ValueError('TV1 engine oracle receipt type mismatch')
        for k in ('layer', 'batch', 'raw_context', 'rows', 'next_n', 'stride', 'cache_offset', 'width',
                  'candidate_capacity', 'page'):
            if type(r.get(k)) is not int:
                raise ValueError('TV1 engine oracle parameter type mismatch')
        if (r['layer'] not in LAYERS or r['batch'] not in range(1, 5)
                or r.get('identical') is not True or r.get('backend') != 'persistent'
                or r['stride'] != STRIDE or r['cache_offset'] % STRIDE != CACHE_OFFSET
                or r['width'] != MAX_RAW or r['next_n'] not in (1, 2)
                or r.get('candidate_capacity') != 2048 or r.get('page') != PAGE
                or r.get('layout') not in ('single', 'native2', 'flattened')
                or not 0 < r['rows'] <= 25 or not 0 < r['raw_context'] <= MAX_RAW):
            raise ValueError('TV1 engine oracle unqualified geometry or equality')
        layout = r['layout']
        if ((layout == 'single' and (r['rows'] != r['batch'] or r['next_n'] != 1))
                or (layout == 'native2' and (r['rows'] != r['batch'] * 2 or r['next_n'] != 2))
                or (layout == 'flattened' and (r['next_n'] != 1
                    or not r['batch'] <= r['rows'] <= r['batch'] * 6 + 1))):
            raise ValueError('TV1 engine oracle row ownership mismatch')
        if r['batch'] == 4 and 31360 <= r['raw_context'] < 140000:
            valid.add((r['layer'], 4, '32K'))
        threshold = 147000 if r['batch'] in (1, 4) else 31360
        if r['raw_context'] >= threshold:
            valid.add((r['layer'], r['batch'], 'occupied'))
    required = {(layer, batch, 'occupied') for layer in LAYERS for batch in range(1, 5)}
    required |= {(layer, 4, '32K') for layer in LAYERS}
    if required - valid:
        raise ValueError(f'TV1 actual metadata comparisons incomplete: {sorted(required - valid)}')
