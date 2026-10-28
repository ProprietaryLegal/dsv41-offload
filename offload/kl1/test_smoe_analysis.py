"""Gate 2 CPU checks: explicit provenance, both hardware replicates, medians, drift, plateaus and prefetch bias."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest

from offload.kl1 import smoe_bench as sb
from offload.prefill_maxlen.evidence import Json, record

WINDOW = 'SP1-W-20261026T160012Z'
STACK = (sb.BENCH_KERNEL, sb.BENCH_DRIVER)


def doc(variant: str, rank: int, scale: float = 1, sample: str = 'variant') -> dict[str, Json]:
    results: list[Json] = [
        {'case': f'r4q1u{u}c0', 'rows': 4, 'requests': 1, 'distinct': u, 'bytes_per_layer': (11.1 + u * 6.12) * 1e6,
         'us_per_layer_median': scale * (100 + u * 20), 'out_sha256': f'same-{u}'} for u in (6, 12, 18, 24)]
    return {'type': 'kl1-smoe-bench', 'variant': variant, 'rank': rank, 'sample': sample,
            'window': 'KL1-W-20261026T200000Z', 'kernel': STACK[0], 'driver': STACK[1],
            'flags': next(v.flags for v in sb.VARIANTS if v.name == variant),
            'plan': 'sha256 test', 'banks': [0, 39], 'seed': 0, 'results': results}


def runs(rank: int, variant_scale: float = .9, drift: float = 0) -> list[dict[str, Json]]:
    return [doc('prod', rank, 1, 'first'), doc('pipe4', rank, variant_scale), doc('prod', rank, 1 + drift, 'last')]


def analyze(primary: list[dict[str, Json]] | None = None, replicate: list[dict[str, Json]] | None = None,
            steps: list[tuple[int, float]] | None = None) -> dict[str, Json]:
    return sb.analyze(primary if primary is not None else runs(0), steps if steps is not None else [(4, 340)],
                      window=WINDOW, rank=0, production_stack=STACK,
                      replicate=replicate if replicate is not None else runs(1))


def test_both_sparks_must_beat_three_percent_and_twice_drift() -> None:
    assert record(analyze()['gate2_variant_pass_both_sparks'])['pipe4'] is True
    assert record(analyze(replicate=runs(1, .98))['gate2_variant_pass_both_sparks'])['pipe4'] is False
    assert record(analyze(replicate=runs(1, .94, drift=.04))['gate2_variant_pass_both_sparks'])['pipe4'] is False
    assert record(analyze(replicate=[])['gate2_variant_pass_both_sparks'])['pipe4'] is False
    assert analyze(replicate=[])['gate2_status'] == 'INCOMPLETE'


def test_gate_uses_median_of_in_range_steps_and_reports_clamped_counts() -> None:
    out = analyze(steps=[(4, 300), (4, 340), (4, 380), (4, 1), (4, 9000)])
    assert out['in_range_steps'] == 3 and out['clamped_steps'] == 2
    inferred = [record(p) for p in cast(list[Json], out['production']) if record(p)['where'] == 'inside']
    assert out['median_isolated_gbps_in_range'] == sorted(float(cast(float, p['isolated_gbps'])) for p in inferred)[1]
    assert out['rate_label'] == "smoe isolated rate at production's inferred expert count"
    assert 'not independently measured production bandwidth' in str(out['rate_limit'])
    assert record(out['shared_prefetch_bias'])['estimated_experts_low'] == 1.7
    assert all(cast(float, p['distinct_prefetch_corrected_estimate']) == pytest.approx(cast(float, p['distinct']) + 1.7)
               for p in inferred)


def test_gate_median_is_not_changed_by_one_slow_outlier() -> None:
    both = [runs(0, .96), runs(1, .96)]
    for docs in both:
        record(cast(list[Json], docs[1]['results'])[0])['us_per_layer_median'] = 1000.0
    out = analyze(both[0], both[1], [(4, 220), (4, 340), (4, 460)])
    assert record(out['gate2_variant_pass_both_sparks'])['pipe4'] is True
    assert record(record(out['gate2_per_rank'])['0'])['pipe4'] is not None


@pytest.mark.parametrize('change', ['rank', 'window', 'driver', 'kernel', 'seed', 'banks', 'plan', 'duplicate', 'missing-prod-last'])
def test_refuse_pooled_or_incompatible_bench_data(change: str) -> None:
    primary = runs(0)
    if change == 'duplicate':
        primary.append(copy.deepcopy(primary[1]))
    elif change == 'missing-prod-last':
        primary.pop()
    else:
        primary[1][change] = cast(Json, {'rank': 1, 'window': 'another-window', 'driver': '580.159.03', 'kernel': 'old',
                              'seed': 42, 'banks': [6], 'plan': 'other'}[change])
    with pytest.raises(ValueError):
        analyze(primary)


def test_oracle_mismatch_blocks_gate_even_with_big_speedup() -> None:
    replica = runs(1, .5)
    record(cast(list[Json], replica[1]['results'])[0])['out_sha256'] = 'different'
    out = analyze(replicate=replica)
    assert record(record(out['oracle_mismatches'])['1'])['pipe4'] == ['r4q1u6c0']
    assert record(out['gate2_variant_pass_both_sparks'])['pipe4'] is False


def test_missing_case_cannot_pass_oracle_by_intersection() -> None:
    primary = runs(0)
    cast(list[Json], primary[1]['results']).pop()
    with pytest.raises(ValueError, match='cases/shapes/bytes'):
        analyze(primary)


def test_monotone_fit_and_plateaus_are_reported_without_crashing() -> None:
    curve = [(6.0, 200.0), (12.0, 200.0), (18.0, 190.0), (24.0, 400.0)]
    assert sb.monotone_curve(curve) == [(6, 200), (12, 200), (18, 200), (24, 400)]
    assert sb.invert(curve, 200) == (12, 'flat')
    assert sb.flat_stretches(curve) == [{'u_low': 6, 'u_high': 18, 'us': 200}]
    assert sb.invert(curve, 300) == (21, 'inside')
    assert sb.invert([(6, 200), (12, 200)], 200) == (9, 'flat')
    for bad in ([(6, 200), (6, 300)], [(6, float('nan')), (12, 200)]):
        with pytest.raises(ValueError):
            sb.invert(bad, 250)
    primary = runs(0)
    for d in (primary[0], primary[-1]):
        rs = cast(list[Json], d['results'])
        for r in rs[:3]:
            record(r)['us_per_layer_median'] = 200.0
    out = analyze(primary, steps=[(4, 200), (4, 300)])
    assert out['flat_steps'] == 1 and out['in_range_steps'] == 1
    assert record(out['flat_stretches'])['4'] == [{'u_low': 6, 'u_high': 18, 'us': 200}]


def bandwidth() -> dict[str, Json]:
    return {'windows': {WINDOW: {'prof-short-b1:decode': {'ranks': {
        '0': {'per_step': [{'rows': 4, 'smoe.span_ms': 13.6}]},
        '1': {'per_step': [{'rows': 6, 'smoe.span_ms': 99.0}]}}},
        'prof-32k-b4:decode': {'ranks': {'0': {'per_step': [{'rows': 20, 'smoe.span_ms': 88.0}]}}}},
        'SP1-W-20261026T144153Z': {'ignored': {}}}}


def test_selection_reads_only_selected_window_rank_and_saved_stack(tmp_path: Path) -> None:
    directory = tmp_path / WINDOW
    (directory / 'cmd').mkdir(parents=True)
    (directory / 'cmd/0028-sp1-327-memory-sparkhead.log').write_text('\n'.join(STACK) + '\n')
    (directory / 'cmd/0029-sp1-327-memory-sparkworker-wired.log').write_text('6.17.0-1026-nvidia\n580.159.03\n')
    steps, source = sb.production_selection(bandwidth(), WINDOW, 0, directory)
    assert steps == [(4, 340)] and source['rank'] == '0'
    with pytest.raises(ValueError, match='cross-stack'):
        sb.production_selection(bandwidth(), WINDOW, 1, directory)
    for selected in (WINDOW + ',SP1-W-20261026T144153Z', 'all'):
        with pytest.raises(ValueError, match='one SP1'):
            sb.production_selection(bandwidth(), selected, 0, directory)


def test_cli_requires_filters_and_writes_provenance(tmp_path: Path) -> None:
    directory = tmp_path / WINDOW
    (directory / 'cmd').mkdir(parents=True)
    (directory / 'cmd/0028-sp1-327-memory-sparkhead.log').write_text('\n'.join(STACK) + '\n')
    (directory / 'cmd/0029-sp1-327-memory-sparkworker-wired.log').write_text('6.17.0-1026-nvidia\n580.159.03\n')
    bw = tmp_path / 'bandwidth.json'
    bw.write_text(json.dumps(bandwidth()))
    files: list[str] = []
    replicas: list[str] = []
    for rank in (0, 1):
        for i, d in enumerate(runs(rank)):
            p = tmp_path / f'r{rank}-{i}.json'
            p.write_text(json.dumps(d))
            (files if rank == 0 else replicas).append(str(p))
    out = tmp_path / 'analysis.json'
    args = ['analyze', '--bench', *files, '--bandwidth', str(bw), '--out', str(out)]
    with pytest.raises(SystemExit):
        sb.main(args)
    assert not out.exists()
    assert sb.main([*args, '--window', WINDOW, '--rank', '0', '--replicate-bench', *replicas]) == 0
    result = json.loads(out.read_text())
    assert result['production_provenance']['kernel'] == STACK[0]
    assert result['gate2_variant_pass_both_sparks']['pipe4'] is True
    out.unlink()
    with pytest.raises(SystemExit, match='cross-stack'):
        sb.main([*args, '--window', WINDOW, '--rank', '1'])
    assert not out.exists()
