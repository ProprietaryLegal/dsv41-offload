"""WIN2-R3: seeded paired cells in decision_v2 (rule win2-paired-v1): sign-flip test, n >= 6, A/A, profiles (CPU only)."""
from __future__ import annotations

import itertools
import random
from typing import cast

import pytest

from offload.launch.windows import decision_v2 as d
from offload.prefill_maxlen.evidence import Json, number, record

BASE = (50.0, 47.0, 53.0, 49.0, 51.5, 48.0)                 # seed luck: pairs differ, arms share it
PROMPT = 'a' * 64                                           # the one proved prompt identity of a cell (DP-2)
OTHER = 'b' * 64


def P(cell: str, metric: str, higher: bool, control: tuple[float, ...], candidate: tuple[float, ...], profile: str,
      aa: tuple[tuple[float, ...], tuple[float, ...]] | None = None, prompt: str | None = PROMPT) -> d.PairedSeries:
    return d.PairedSeries(cell, metric, higher, control, candidate, profile, aa, prompt)


def _x(gains: tuple[float, ...], base: tuple[float, ...] = BASE) -> tuple[float, ...]:
    return tuple(b * (1 + g) for b, g in zip(base, gains))


def _brute_p(diffs: list[float]) -> float:
    obs = abs(sum(diffs))
    hits = sum(1 for signs in itertools.product((1, -1), repeat=len(diffs))
               if abs(sum(s * v for s, v in zip(signs, diffs))) >= obs - 1e-12)
    return float(hits) / float(2 ** len(diffs))


@pytest.mark.parametrize('diffs', [[.01] * 6, [.01, .02, -.005, .03, .01, .02], [0.0] * 6, [1, -1, 2, -2, 3, -3.5],
                                   [.05, .04, .03, .02, .01, -.001, .002, .003]])
def test_sign_flip_matches_brute_force(diffs: list[float]) -> None:
    assert d.sign_flip_p(diffs) == pytest.approx(_brute_p(diffs))


def test_sign_flip_floor_by_n() -> None:
    assert d.sign_flip_p([.01] * 6) == pytest.approx(2 / 64)          # 0.031: every pair must agree at n = 6
    assert d.sign_flip_p([.01] * 3) == pytest.approx(2 / 8)           # 3 pairs can never reach 0.05
    assert d.sign_flip_p([.01, .01, .01, .01, .01, -.001]) > .05      # one disagreeing pair: no confirmation at n = 6
    assert d.sign_flip_p([.01] * 7 + [-.001]) < .05                   # n = 8 tolerates one small disagreement
    with pytest.raises(ValueError):
        d.sign_flip_p([.01] * 21)


def test_clear_paired_win_needs_all_three_conditions() -> None:
    win = d.decide_paired_cell(P('c', 'decode_tps', True, BASE, _x((.05, .045, .055, .05, .048, .052)), 'single'))
    assert win['verdict'] == 'WIN' and number(win['sign_flip_p']) < .05
    small = d.decide_paired_cell(P('c', 'decode_tps', True, BASE, _x((.012, .01, .011, .013, .01, .012)), 'single'))
    assert small['verdict'] == 'INCONCLUSIVE'                          # significant but below the 2 % effect
    one_off = d.decide_paired_cell(P('c', 'decode_tps', True, BASE, _x((.05, .05, .05, .05, .05, -.001)), 'single'))
    assert one_off['verdict'] == 'INCONCLUSIVE' and number(one_off['sign_flip_p']) > .05


def test_loss_is_any_significant_harm_and_guards_never_win() -> None:
    loss = d.decide_paired_cell(P('c', 'decode_tps', True, BASE, _x((-.012, -.01, -.011, -.013, -.01, -.012)), 'single'))
    assert loss['verdict'] == 'LOSS'
    ms = d.decide_paired_cell(P('c', 'ms_per_round', False, BASE, _x((.03, .031, .029, .03, .032, .03)), 'single'))
    assert ms['verdict'] == 'LOSS' and number(ms['point_gain']) < 0    # lower is better: 3 % slower rounds
    guard = d.decide_paired_cell(P('c', 'accepted_per_round', True, BASE, _x((.05,) * 6), 'single'), guard=True)
    assert guard['verdict'] == 'INCONCLUSIVE' and guard['role'] == 'guard'


@pytest.mark.parametrize('n', [3, 5])
def test_fewer_than_six_pairs_refused(n: int) -> None:
    with pytest.raises(ValueError, match='at least 6 complete pairs'):
        d.decide_paired_cell(P('c', 'decode_tps', True, BASE[:n], BASE[:n], 'single'))
    with pytest.raises(ValueError, match='at least 6'):
        d.decide_paired_cell(P('c', 'decode_tps', True, BASE, BASE[:5] + (1.0, 2.0), 'single'))


def test_aa_drift_and_stream_identity_make_the_cell_inconclusive() -> None:
    c1 = BASE
    drift = _x((.02, .021, .019, .02, .022, .02))                       # C2 2 % above C1 in every pair: start drift
    x = _x((.08,) * 6)
    s = P('c', 'decode_tps', True, tuple((a + b) / 2 for a, b in zip(c1, drift)), x, 'single', (c1, drift))
    r = d.decide_paired_cell(s)
    assert r['verdict'] == 'INCONCLUSIVE' and 'A/A drift' in str(r['inconclusive_reasons'])
    steady = _x((.001, -.001, .0005, -.0005, .0, .0002))
    ok = P('c', 'decode_tps', True, tuple((a + b) / 2 for a, b in zip(c1, steady)), x, 'single', (c1, steady))
    assert d.decide_paired_cell(ok)['verdict'] == 'WIN'
    assert d.decide_paired_cell(ok, expect_identical=True, streams_identical=False)['verdict'] == 'INCONCLUSIVE'
    assert d.decide_paired_cell(ok, expect_identical=True, streams_identical=True)['verdict'] == 'WIN'
    with pytest.raises(ValueError, match='stream identity'):
        d.decide_paired_cell(ok, expect_identical=True)


def test_window_reports_one_verdict_per_profile() -> None:
    single = P('short', 'decode_tps', True, BASE, _x((.05, .045, .055, .05, .048, .052)), 'single')
    multi = P('b3', 'decode_tps', True, BASE, _x((.001, -.002, .003, -.001, .0, .002)), 'multi')
    multi_guard = P('b3', 'ms_per_round', False, BASE, _x((.0,) * 6), 'multi')
    r = d.decide_paired_window([single, multi, multi_guard], ['short', 'b3'])
    assert r['profiles'] == {'multi': 'INCONCLUSIVE', 'single': 'CANDIDATE'}
    assert r['status'] == 'PER PROFILE: multi=INCONCLUSIVE; single=CANDIDATE' and r['rule'] == d.PAIRED_RULE
    assert d.decide_paired_window([single], ['short'])['status'] == 'CANDIDATE'
    cells = record(r['cells'])
    assert record(cells['b3'])['profile'] == 'multi' and record(record(cells['b3'])['metrics'])['ms_per_round']


def test_window_refusals() -> None:
    a = P('a', 'decode_tps', True, BASE, BASE, 'single')
    with pytest.raises(ValueError, match='per profile'):
        d.decide_paired_window([a, P('b', 'decode_tps', True, BASE, BASE, 'single'),
                                P('c', 'decode_tps', True, BASE, BASE, 'single')], ['a', 'b', 'c'])
    with pytest.raises(ValueError, match='same profile'):
        d.decide_paired_window([a, P('a', 'ms_per_round', False, BASE, BASE, 'multi')], ['a'])
    with pytest.raises(ValueError, match='profile must be'):
        d.decide_paired_cell(P('a', 'decode_tps', True, BASE, BASE, 'mixed'))
    with pytest.raises(ValueError, match='without measurements'):
        d.decide_paired_window([a], ['z'])


def test_paired_window_never_pairs_across_prompt_hashes() -> None:
    """DP-2: a decision cell needs ONE proved prompt identity; guards of another prompt, or none, refuse."""
    primary = P('a', 'decode_tps', True, BASE, _x((.05,) * 6), 'single')
    with pytest.raises(ValueError, match='never paired across prompt hashes'):
        d.decide_paired_window([primary, P('a', 'ms_per_round', False, BASE, BASE, 'single', prompt=OTHER)], ['a'])
    with pytest.raises(ValueError, match='never paired across prompt hashes'):
        d.decide_paired_window([P('a', 'decode_tps', True, BASE, BASE, 'single', prompt=None)], ['a'])
    with pytest.raises(ValueError, match='64-hex'):
        d.decide_paired_cell(P('a', 'decode_tps', True, BASE, BASE, 'single', prompt='abc'))
    r = d.decide_paired_window([primary], ['a'])
    assert record(record(r['cells'])['a'])['prompt_sha256'] == PROMPT


def _seeded(cell: str, profile: str, values: tuple[float, ...], seeds: int = 101_000,
            prompts: tuple[str, ...] | None = None) -> dict[str, Json]:
    return {'cell': cell, 'profile': profile,
            'pairs': cast(Json, [{'seeds': [seeds + 10 * i], 'decode_tps': v,
                                  'prompt_sha256': (prompts or (PROMPT,) * len(values))[i]} for i, v in enumerate(values)])}


def test_seeded_series_from_driver_results() -> None:
    s = d.seeded_series('decode_tps', True, _seeded('c', 'single', BASE), _seeded('c', 'single', _x((.05,) * 6)),
                        _seeded('c', 'single', BASE))
    assert s.control == BASE and s.aa == (BASE, BASE) and s.profile == 'single' and s.prompt_sha256 == PROMPT
    assert d.seeded_series('decode_tps', True, _seeded('c', 'single', BASE), _seeded('c', 'single', BASE)).aa is None
    with pytest.raises(ValueError, match='different cells'):
        d.seeded_series('decode_tps', True, _seeded('c', 'single', BASE), _seeded('d', 'single', BASE))
    with pytest.raises(ValueError, match='seeds'):
        d.seeded_series('decode_tps', True, _seeded('c', 'single', BASE), _seeded('c', 'single', BASE, seeds=7))
    # DP-2: one arm (or one pair) on another prompt, or results without prompt hashes, never pair
    with pytest.raises(ValueError, match='never pools or pairs across prompt hashes'):
        d.seeded_series('decode_tps', True, _seeded('c', 'single', BASE), _seeded('c', 'single', BASE, prompts=(OTHER,) * 6))
    with pytest.raises(ValueError, match='never pools or pairs across prompt hashes'):
        d.seeded_series('decode_tps', True, _seeded('c', 'single', BASE),
                        _seeded('c', 'single', BASE, prompts=(PROMPT,) * 5 + (OTHER,)))
    bare = {'cell': 'c', 'profile': 'single', 'pairs': [{'seeds': [101_000 + 10 * i], 'decode_tps': v}
                                                         for i, v in enumerate(BASE)]}
    with pytest.raises(ValueError, match='64-hex'):
        d.seeded_series('decode_tps', True, cast(dict[str, Json], bare), _seeded('c', 'single', BASE))
    with pytest.raises(ValueError, match='report-only'):
        d.seeded_series('decode_tps', True, {**_seeded('c', 'single', BASE), 'report_only': True}, _seeded('c', 'single', BASE))


def _null_rates(n: int, res_cv: float, trials: int) -> dict[str, float]:
    rng = random.Random(3)
    out = {'WIN': 0, 'LOSS': 0, 'INCONCLUSIVE': 0}
    for _ in range(trials):
        seeds = [rng.gauss(0, .07) for _ in range(n)]
        c = tuple(50 * (1 + b) * (1 + rng.gauss(0, res_cv)) for b in seeds)
        x = tuple(50 * (1 + b) * (1 + rng.gauss(0, res_cv)) for b in seeds)
        out[str(d.decide_paired_cell(P('c', 'decode_tps', True, c, x, 'single'))['verdict'])] += 1
    return {k: v / trials for k, v in out.items()}


@pytest.mark.parametrize('n', [6, 12])
def test_no_effect_rates_are_symmetric_and_small(n: int) -> None:
    """WIN2.md: no-effect false WIN and false LOSS each stay near or below 2.5 % (seed luck cancels in the pairs)."""
    rates = _null_rates(n, .02, 4000)
    assert rates['WIN'] < .03 and rates['LOSS'] < .04
