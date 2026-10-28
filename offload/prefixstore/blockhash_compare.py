#!/usr/bin/env python3
"""MC1 byte-equality gate over the per-block hash dumps (VLLM_DSV41_OFFLOAD_HASH_DUMP, both ranks).

``restore``: every block of the restored run (``load`` lines) must equal the same chunk's in-memory bytes
(``store`` lines of the earlier run, same rank), joined on (offload key, group, sub-block). PASS iff at least one
block is compared and none differs, is missing from the in-memory dump, or is unlabeled.

``ranks``: the same event on rank 0 and rank 1 (independent rank identity, RP1 P1). Differences are reported with the
differing segment indices, so padding or inactive regions can be told apart from live payload; PASS iff equal.

    python3 offload/prefixstore/blockhash_compare.py restore --in-memory A_r0.jsonl --restored B_r0.jsonl
    python3 offload/prefixstore/blockhash_compare.py ranks --rank0 A_r0.jsonl --rank1 A_r1.jsonl --ev store
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

Key = tuple[str, int, int]


def load(path: Path, ev: str) -> tuple[dict[Key, dict[str, Any]], int]:
    """(key, group, sub) -> record for one event; repeated keys must agree (else the dump itself is unstable)."""
    out: dict[Key, dict[str, Any]] = {}
    unlabeled = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r['ev'] != ev:
            continue
        if r.get('key') is None:
            unlabeled += 1
            continue
        k: Key = (r['key'], int(r['g']), int(r['sub']))
        if k in out and out[k]['sha'] != r['sha']:
            raise ValueError(f'{path}: block {k} dumped twice with different bytes')
        out[k] = r
    return out, unlabeled


def segments(a: dict[str, Any], b: dict[str, Any]) -> list[int]:
    sa, sb = a.get('seg') or '', b.get('seg') or ''
    width = 16
    return [i // width for i in range(0, max(len(sa), len(sb)), width) if sa[i:i + width] != sb[i:i + width]]


def compare(reference: dict[Key, dict[str, Any]], candidate: dict[Key, dict[str, Any]],
            unlabeled: int, require_all: bool) -> dict[str, Any]:
    equal = differ = missing = 0
    seg_hist: Counter[int] = Counter()
    examples: list[dict[str, Any]] = []
    for k, rec in sorted(candidate.items()):
        ref = reference.get(k)
        if ref is None:
            missing += 1
            continue
        if ref['sha'] == rec['sha'] and ref['n'] == rec['n']:
            equal += 1
            continue
        differ += 1
        segs = segments(ref, rec)
        seg_hist.update(segs)
        if len(examples) < 5:
            examples.append({'key': k[0][:16], 'g': k[1], 'sub': k[2], 'lb': rec['lb'], 'segments': segs[:32]})
    extra = len(set(reference) - set(candidate)) if require_all else 0
    ok = equal > 0 and differ == 0 and missing == 0 and unlabeled == 0 and extra == 0
    return {'pass': ok, 'compared': equal + differ, 'equal': equal, 'differ': differ, 'missing': missing,
            'unlabeled': unlabeled, 'only_in_reference': extra, 'differing_segments': dict(seg_hist.most_common(64)),
            'examples': examples}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('restore')
    p.add_argument('--in-memory', type=Path, required=True)
    p.add_argument('--restored', type=Path, required=True)
    p.add_argument('--in-memory-ev', default='store')
    p.add_argument('--restored-ev', default='load')
    p = sub.add_parser('ranks')
    p.add_argument('--rank0', type=Path, required=True)
    p.add_argument('--rank1', type=Path, required=True)
    p.add_argument('--ev', default='store')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'restore':
        reference, u1 = load(args.in_memory, args.in_memory_ev)
        candidate, u2 = load(args.restored, args.restored_ev)
        result = compare(reference, candidate, u1 + u2, require_all=False)
    else:
        reference, u1 = load(args.rank0, args.ev)
        candidate, u2 = load(args.rank1, args.ev)
        result = compare(reference, candidate, u1 + u2, require_all=True)
    text = json.dumps(result, indent=1)
    if args.json:
        args.json.write_text(text + '\n')
    print(text)
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
