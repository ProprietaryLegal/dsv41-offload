"""Render aggregate-only solo measurements; never reads token or fixture dumps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    arms = ('k2-tp2', 'native-hybrid-kx', 'native-hybrid-kx2')
    data = {(rank, arm): json.loads((args.results / f'bench_{arm}_solo_r{rank}.json').read_text())
            for rank in (0, 1) for arm in arms}
    print('|Rank|Arm|B1 us|B2 us|B3 us|B4 us|')
    print('|---:|---|---:|---:|---:|---:|')
    for (rank, arm), record in data.items():
        times = [record['draft'][f'B{batch}_k5_n5']['p50_us'] for batch in (1, 2, 3, 4)]
        print(f'|{rank}|{arm}|' + '|'.join(str(value) for value in times) + '|')
    print('\nKx2 minus K2 per-category kernel times (us; sorted largest gap first):')
    for rank in (0, 1):
        for batch in (1, 2, 3, 4):
            def categories(arm: str) -> dict[str, float]:
                cats = dict(data[rank, arm]['draft'][f'B{batch}_k5_n5']['breakdown']['device0']['us'])
                cats['dense'] = sum(cats.pop(key, 0) for key in ('native_bf16_dense', 'native_fp8_dense', 'dense_k5', 'wo_a'))
                cats['experts'] = cats.get('experts', 0) + cats.pop('native_experts', 0)
                return cats
            a, b = categories('native-hybrid-kx2'), categories('k2-tp2')
            gaps = {key: round(a.get(key, 0)-b.get(key, 0), 2) for key in a.keys() | b.keys()}
            print(rank, batch, sorted(gaps.items(), key=lambda item: -item[1]))


if __name__ == '__main__':
    main()
