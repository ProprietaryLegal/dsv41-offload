"""WIN2-R3 distributional gate for candidates (F-METHOD §3.4): is the candidate's next-token distribution the same model?

Probe: 64 public-corpus prompts (32 short ~1K tokens, 24 at 32K, 8 at 150K; offsets registered below), each tokenized
once (/tokenize, raw text), then the top-20 next-token logprobs at 8 FIXED positions: the prompt's last 8 tokens are the
fixed continuation, and position k asks for the next token after ids[:L - 8 + k] (/v1/completions, max_tokens 1,
temperature 0, logprobs 20, token-id prompts, one cache salt per run so only the first probe of a prompt prefills). No
generation, so no generated text exists; only token ids and logprob vectors are recorded, on Controller (never git).

Compare (control || candidate) per position with greedy_kl.top20_kl (shared top-20 + OTHER; common mass >= 0.99 is
reported per position) and top-1 agreement. Gate against an A/A band registered BEFORE use from two production starts:
- kl_ceiling = 1.5 x the 99th percentile of the A/A MEAN KL (seeded bootstrap over the A/A positions); LOSS when the
  candidate's mean KL exceeds it (simulated no-effect false LOSS 0.0-0.2 %);
- top-1: F-METHOD's literal rule (LOSS when agreement < A/A minimum - 1) rejects a no-effect candidate 13-34 % of the
  time once the A/A itself shows >= 1 expected near-tie flip (one A/A run is a noisy minimum). The gate therefore uses an
  exact conditional test of the two flip counts instead: with D_c candidate and D_a A/A disagreements over k registered
  A/A comparisons, D_c | D_c + D_a ~ Binomial(D_c + D_a, 1 / (k + 1)) under equal flip rates; LOSS when the one-sided
  p < 0.01 (false LOSS <= 1 % by construction). The literal rule is still reported (`fmethod_top1_rule`).
- common mass: positions whose shared top-20 mass is < 0.99 (where the coarse KL is not trustworthy) are counted; LOSS
  when the candidate has significantly more of them than the A/A (same exact test). Greedy identity stays the restore
gate (W8), unchanged. COST (ESTIMATED from MEASURED TTFTs): one cold prefill per prompt per arm, 32 x ~0.5 s + 24 x ~15 s
+ 8 x ~69 s ~ 15.5 min per arm; `--subset short,32k` drops the 150K prompts (~6.5 min).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import greedy_kl
from offload.prefill_maxlen.evidence import Json, number, record

SCHEMA = 'dsv41-dist-gate-v1'
POSITIONS = 8
KL_FACTOR = 1.5
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261027
TOP1_ALPHA = 0.01
COORDINATOR = 'Deepseek/coordinator'


@dataclass(frozen=True)
class DistPrompt:
    key: str
    target: int
    offset: int


def _prompts() -> tuple[DistPrompt, ...]:
    out = [DistPrompt(f'dist-short-{i:02d}', 1024, 200_000 + 10_000 * i) for i in range(32)]
    out += [DistPrompt(f'dist-32k-{i:02d}', 32000, 1_000_000 + 120_000 * i) for i in range(24)]
    out += [DistPrompt(f'dist-150k-{i}', 150000, 4_200_000 + 560_000 * i) for i in range(8)]
    return tuple(out)


DIST_PROMPTS = _prompts()
SUBSETS = {'short': 1024, '32k': 32000, '150k': 150000}


def selected(subset: Sequence[str] | None) -> tuple[DistPrompt, ...]:
    if not subset:
        return DIST_PROMPTS
    unknown = set(subset) - set(SUBSETS)
    if unknown:
        raise ValueError(f'unknown prompt subsets {sorted(unknown)}; choose from {sorted(SUBSETS)}')
    targets = {SUBSETS[s] for s in subset}
    return tuple(p for p in DIST_PROMPTS if p.target in targets)


class DistApi(Protocol):
    def model(self) -> dict[str, Json]: ...
    def token_ids(self, text: str) -> list[int]: ...
    def top20(self, prefix: Sequence[int], salt: str) -> dict[str, Json]: ...


class Http(wc.Http):
    def token_ids(self, text: str) -> list[int]:
        r = self.post('/tokenize', {'model': self.model()['id'], 'prompt': text, 'add_special_tokens': True})
        ids = r.get('tokens')
        if not isinstance(ids, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in ids):
            raise ValueError('tokenize returned no token ids')
        return cast(list[int], ids)

    def top20(self, prefix: Sequence[int], salt: str) -> dict[str, Json]:
        r = self.post('/v1/completions', {'model': self.model()['id'], 'prompt': cast(Json, list(prefix)), 'max_tokens': 1,
                                          'temperature': 0, 'logprobs': 20, 'return_tokens_as_token_ids': True,
                                          'priority': 10, 'cache_salt': salt})
        choices = r.get('choices')
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError('one completion choice expected')
        tops = record(record(choices[0])['logprobs'])['top_logprobs']
        if not isinstance(tops, list) or len(tops) != 1:
            raise ValueError('one top-logprob position expected')
        return record(tops[0])


def cuts(api: DistApi, materials: str, hashes: Mapping[str, Json], prompts: Sequence[DistPrompt]) -> dict[str, Json]:
    """Numeric cuts (characters, tokens, sha256) of the registered prompts; built once, reused by every arm."""
    rows: dict[str, Json] = {}
    for p in prompts:
        chars, tokens = wc.cut(materials, wc.Prompt(p.key, p.target, p.offset), lambda t: len(api.token_ids(t)))
        text = wc.render(materials, p.offset, chars)
        rows[p.key] = {'target': p.target, 'offset': p.offset, 'chars': chars, 'tokens': tokens,
                       'sha256': hashlib.sha256(text.encode()).hexdigest()}
    return {'schema': SCHEMA, 'type': 'cuts', 'corpus_sha256': cast(Json, dict(hashes)), 'prompts': rows}


def probe(api: DistApi, materials: str, cut_rows: Mapping[str, Json], salt_prefix: str) -> dict[str, Json]:
    """Top-20 at the 8 fixed positions of every prompt in `cut_rows` on the server under test."""
    if not salt_prefix.strip():
        raise ValueError('a cache salt prefix per run is required')
    positions: list[Json] = []
    for key, value in record(cast(Json, dict(cut_rows))).items():
        row = record(value)
        text = wc.render(materials, int(number(row['offset'])), int(number(row['chars'])))
        if hashlib.sha256(text.encode()).hexdigest() != row.get('sha256'):
            raise ValueError(f'{key}: rebuilt prompt differs from its registered cut')
        ids = api.token_ids(text)
        if len(ids) <= POSITIONS:
            raise ValueError(f'{key}: prompt shorter than the fixed continuation')
        salt = f'{salt_prefix}:dist:{key}'
        for k in range(POSITIONS):
            at = len(ids) - POSITIONS + k
            positions.append({'prompt': key, 'position': at, 'k': k,
                              'prefix_sha256': hashlib.sha256(json.dumps(ids[:at]).encode()).hexdigest(),
                              'top20': api.top20(ids[:at], salt)})
    return {'schema': SCHEMA, 'type': 'probe', 'positions': positions, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ')}


def _top1(top: Mapping[str, Json]) -> str:
    return max(top, key=lambda k: number(top[k]))


def compare(control: Mapping[str, Json], candidate: Mapping[str, Json]) -> dict[str, Json]:
    a, b = [record(p) for p in cast(list[Json], control['positions'])], [record(p) for p in cast(list[Json], candidate['positions'])]
    if [(p['prompt'], p['position'], p['prefix_sha256']) for p in a] != [(p['prompt'], p['position'], p['prefix_sha256']) for p in b]:
        raise ValueError('control and candidate probed different prompts or positions')
    per: list[Json] = []
    kls: list[float] = []
    agree = low = 0
    for x, y in zip(a, b):
        r = greedy_kl.top20_kl(record(x['top20']), record(y['top20']))
        same = _top1(record(x['top20'])) == _top1(record(y['top20']))
        agree += same
        low += r['common_mass'] < greedy_kl.MIN_COMMON_MASS
        kls.append(r['kl_nats'])
        per.append({'prompt': x['prompt'], 'k': x['k'], 'kl_nats': r['kl_nats'], 'common_mass': r['common_mass'],
                    'top1_agree': same})
    return {'schema': SCHEMA, 'type': 'compare', 'positions': len(kls), 'mean_kl': statistics.fmean(kls),
            'max_kl': max(kls), 'top1_agree': agree, 'low_common_mass_positions': low, 'per_position': per}


def _p99(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)]


def register_band(aa: Sequence[Mapping[str, Json]], *, factor: float = KL_FACTOR) -> dict[str, Json]:
    """A/A band from compare() results of production vs production (two starts). Register it before any candidate use."""
    if not aa:
        raise ValueError('at least one A/A comparison is required')
    counts = {int(number(c['positions'])) for c in aa}
    if len(counts) != 1:
        raise ValueError('A/A comparisons cover different position sets')
    rng = random.Random(BOOTSTRAP_SEED)
    means: list[float] = []
    for c in aa:
        kls = [number(record(p)['kl_nats']) for p in cast(list[Json], c['per_position'])]
        means += [statistics.fmean(rng.choices(kls, k=len(kls))) for _ in range(BOOTSTRAP)]
    band: dict[str, Json] = {'schema': SCHEMA, 'type': 'band', 'positions': counts.pop(), 'factor': factor,
                             'aa_mean_kl_p99': _p99(means), 'kl_ceiling': factor * _p99(means),
                             'top1_min': min(int(number(c['top1_agree'])) for c in aa), 'aa_comparisons': len(aa),
                             'aa_top1_disagreements': sum(int(number(c['positions']) - number(c['top1_agree'])) for c in aa),
                             'aa_low_common_mass': sum(int(number(c['low_common_mass_positions'])) for c in aa),
                             'bootstrap': {'resamples': BOOTSTRAP, 'seed': BOOTSTRAP_SEED},
                             'registered_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    band['sha256'] = hashlib.sha256(json.dumps(band, sort_keys=True).encode()).hexdigest()
    return band


def gate(result: Mapping[str, Json], band: Mapping[str, Json]) -> dict[str, Json]:
    body = {k: v for k, v in band.items() if k != 'sha256'}
    if band.get('sha256') != hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest():
        raise ValueError('A/A band changed after registration')
    if result.get('positions') != band.get('positions'):
        raise ValueError('candidate comparison and A/A band cover different position sets')
    reasons = []
    if number(result['mean_kl']) > number(band['kl_ceiling']):
        reasons.append(f'mean KL {number(result["mean_kl"]):.3g} > ceiling {number(band["kl_ceiling"]):.3g}')
    flips = int(number(result['positions']) - number(result['top1_agree']))
    p = top1_p(flips, int(number(band['aa_top1_disagreements'])), int(number(band['aa_comparisons'])))
    if p < TOP1_ALPHA:
        reasons.append(f'top-1 disagreements {flips} vs A/A {band["aa_top1_disagreements"]} over '
                       f'{band["aa_comparisons"]} comparison(s): one-sided p {p:.4f} < {TOP1_ALPHA}')
    low = int(number(result['low_common_mass_positions']))
    p_low = top1_p(low, int(number(band['aa_low_common_mass'])), int(number(band['aa_comparisons'])))
    if p_low < TOP1_ALPHA:
        reasons.append(f'positions with top-20 common mass < {greedy_kl.MIN_COMMON_MASS} {low} vs A/A '
                       f'{band["aa_low_common_mass"]}: one-sided p {p_low:.4f} < {TOP1_ALPHA}')
    return {'verdict': 'LOSS' if reasons else 'PASS', 'reasons': cast(Json, reasons), 'band_sha256': band['sha256'],
            'mean_kl': result['mean_kl'], 'top1_agree': result['top1_agree'], 'top1_disagreements': flips,
            'top1_p': p, 'fmethod_top1_rule': number(result['top1_agree']) < number(band['top1_min']) - 1,
            'low_common_mass_positions': result['low_common_mass_positions']}


def top1_p(candidate_flips: int, aa_flips: int, aa_comparisons: int) -> float:
    """P(X >= candidate_flips), X ~ Binomial(candidate_flips + aa_flips, 1 / (aa_comparisons + 1)): the exact
    conditional test that the candidate's top-1 flip rate exceeds the A/A rate (Poisson counts, same positions)."""
    if candidate_flips < 0 or aa_flips < 0 or aa_comparisons < 1:
        raise ValueError('invalid flip counts')
    n, q = candidate_flips + aa_flips, 1 / (aa_comparisons + 1)
    return sum(math.comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(candidate_flips, n + 1))


def _read(path: Path) -> dict[str, Json]:
    return record(cast(Json, json.loads(path.read_text())))


def _write(path: Path, data: Mapping[str, Json]) -> None:
    with path.open('x') as handle:
        handle.write(json.dumps(data, allow_nan=False) + '\n')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('cuts', 'probe'):
        s = sub.add_parser(name)
        s.add_argument('--execute', action='store_true')
        s.add_argument('--url', default=wc.URL)
        s.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
        s.add_argument('--output', type=Path, required=True)
        s.add_argument('--subset', help='comma list of short,32k,150k (default: all 64 prompts)')
        if name == 'probe':
            s.add_argument('--cuts', type=Path, required=True)
            s.add_argument('--salt', required=True)
    c = sub.add_parser('compare')
    c.add_argument('control', type=Path)
    c.add_argument('candidate', type=Path)
    c.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('band')
    b.add_argument('aa', type=Path, nargs='+', help='compare() outputs of production vs production')
    b.add_argument('--output', type=Path, required=True)
    g = sub.add_parser('gate')
    g.add_argument('comparison', type=Path)
    g.add_argument('--band', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd in ('cuts', 'probe'):
        prompts = selected(args.subset.split(',') if args.subset else None)
        if not args.execute:
            print(json.dumps({'executed': False, 'prompts': len(prompts), 'positions': POSITIONS * len(prompts)}))
            return 0
        if os.environ.get('WINDOW_COORDINATOR') != COORDINATOR:
            print(f'STOP: probes are coordinator-only (WINDOW_COORDINATOR={COORDINATOR})', file=sys.stderr)
            return 2
        from offload.prefill_trace.window_requests import corpus
        materials, hashes = corpus(args.data_dir)
        api = Http(args.url)
        if args.cmd == 'cuts':
            _write(args.output, cuts(api, materials, hashes, prompts))
        else:
            registered = _read(args.cuts)
            if registered.get('corpus_sha256') != hashes:
                raise ValueError('public corpus changed since the cuts were built')
            rows = record(registered['prompts'])
            _write(args.output, probe(api, materials, {p.key: rows[p.key] for p in prompts}, args.salt))
        return 0
    if args.cmd == 'compare':
        _write(args.output, compare(_read(args.control), _read(args.candidate)))
        return 0
    if args.cmd == 'band':
        _write(args.output, register_band([_read(p) for p in args.aa]))
        return 0
    verdict = gate(_read(args.comparison), _read(args.band))
    print(json.dumps(verdict, allow_nan=False))
    return 0 if verdict['verdict'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
