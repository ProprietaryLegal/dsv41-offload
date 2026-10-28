"""S36 independent probes of the coordinator's UP2 runner merges 5693e64 (0adfa65) and 564535f (e06e9b0).

Read-only on the recorded W13 SCREEN windows under the campaign directory; skipped when they are absent. Only hashes,
first-token probability comparisons and timings are read; no generated text exists in the artifacts or here.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import cast

import pytest

from offload.prefill_maxlen.evidence import Json, record
from offload.launch.windows import up2_checks as ck

CAMPAIGN = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')
RUNS = {'r1': 'UP2-screen-diagnostic-20261025T064342Z', 'r2': 'UP2-screen-diagnostic-20261025T075003Z',
        'r3': 'UP2-screen-diagnostic-20261025T120635Z'}
MEASURED = CAMPAIGN/'UP2-screen-300K-20261025T131508Z'
CONTEXTS = (512, 32000, 150000, 300000)
needs_runs = pytest.mark.skipif(not all((CAMPAIGN/r).is_dir() for r in RUNS.values()) or not MEASURED.is_dir(),
                                reason='W13 SCREEN window artifacts absent')


def row(run: str, arm: str, n: int, rep: int) -> dict[str, Json]:
    return ck.read(CAMPAIGN/RUNS[run]/f'diag-{arm}-num{n}-{rep}.json')


def text(r: dict[str, Json]) -> tuple[str, str]:
    return str(r['text_sha256']), str(r['finish'])


@needs_runs
def test_off_arm_alone_is_not_text_deterministic_at_any_context() -> None:
    """Replay OFF only (no UP2 code path): across the three windows' six OFF repeats, every context yields at least
    three distinct 64-token greedy texts, while all prompt hashes agree. The old gate's A/A half failed on OFF alone."""
    for n in CONTEXTS:
        rows = [row(r, 'off', n, rep) for r in RUNS for rep in (0, 1)]
        assert len({str(x['prompt_sha256']) for x in rows}) == 1
        assert len({text(x) for x in rows}) >= 3, n
        assert {str(x['finish']) for x in rows} == {'length'}


@needs_runs
def test_512_first_token_distribution_is_identical_in_all_nine_requests_while_text_varies() -> None:
    """At 512 the first-token top-20 probabilities are byte-identical across all 9 requests (both arms, three engine
    starts each), yet five distinct texts occur: the divergence is in decode, not in the prefill UP2 changes."""
    rows = [row(r, 'off', 512, rep) for r in RUNS for rep in (0, 1)] + [row(r, 'on', 512, 0) for r in RUNS]
    assert len({json.dumps(x['probabilities'], sort_keys=True) for x in rows}) == 1
    assert len({text(x) for x in rows}) == 5


@needs_runs
def test_on_texts_mostly_reappear_in_the_off_arm() -> None:
    """Of the 12 ON texts (4 contexts x 3 windows), 10 equal some OFF text from these windows. The two that do not
    (512 r3, 300K r2) sit at contexts where the six OFF repeats themselves give 4 and 6 distinct texts."""
    unmatched = []
    for n in CONTEXTS:
        off = {text(row(r, 'off', n, rep)) for r in RUNS for rep in (0, 1)}
        for r in RUNS:
            if text(row(r, 'on', n, 0)) not in off:
                unmatched.append((n, r))
    assert unmatched == [(512, 'r3'), (300000, 'r2')]
    assert len({text(row(r, 'off', 512, rep)) for r in RUNS for rep in (0, 1)}) == 4
    assert len({text(row(r, 'off', 300000, rep)) for r in RUNS for rep in (0, 1)}) == 6


@needs_runs
def test_hard_numerical_floor_recomputes_and_passes_in_every_window() -> None:
    """ck.numerical (the hard gate the change keeps) passes for all 12 matched triples; r3 values equal its checkpoint."""
    for r in RUNS:
        for n in CONTEXTS:
            ck.numerical(row(r, 'off', n, 0), row(r, 'off', n, 1), row(r, 'on', n, 0))
    checkpoint = json.loads((CAMPAIGN/RUNS['r3']/'checkpoints.jsonl').read_text().splitlines()[0])
    for recorded in cast(list[dict[str, Json]], checkpoint['numerical']):
        n = cast(int, recorded['context'])
        again = ck.numerical(row('r3', 'off', n, 0), row('r3', 'off', n, 1), row('r3', 'on', n, 0))
        assert {k: recorded[k] for k in again} == again
        assert recorded['text_aa_equal_INFORMATIONAL'] is False


def test_512_prompt_never_reaches_the_replay_skip_path() -> None:
    """UP2 skips the decoder replay on intermediate prefill chunks only. The runtime gate pins
    --max-num-batched-tokens 4096, so the 512 probe is one (final) chunk: its ON/OFF text never exercised UP2."""
    source = Path(ck.__file__).read_text()
    assert "('--max-num-batched-tokens','4096')" in source
    assert 512 < 4096


@needs_runs
def test_stage_b_300k_arithmetic() -> None:
    summary = ck.read(MEASURED/'summary.json')
    cells = {(arm, rep): ck.read(MEASURED/f'{arm}-cold300000-{rep}.json') for arm in ('off-a', 'on', 'off-b')
             for rep in range(3)}
    ratios, spreads = [], []
    for rep in range(3):
        x, y, z = (cast(float, cells[(arm, rep)]['ttft_s']) for arm in ('off-a', 'on', 'off-b'))
        assert len({str(cells[(arm, rep)]['prompt_sha256']) for arm in ('off-a', 'on', 'off-b')}) == 1
        assert len({str(cells[(arm, rep)]['cache_salt_sha256']) for arm in ('off-a', 'on', 'off-b')}) == 1
        ratios.append(y/((x+z)/2))
        spreads.append(abs(x-z)/((x+z)/2))
    stage_b = record(summary['stage_b'])
    assert stage_b['status'] == 'PASS'
    assert stage_b['gain'] == pytest.approx(1-statistics.median(ratios), abs=1e-12)
    assert stage_b['spread'] == pytest.approx(statistics.median(spreads), abs=1e-12)
    assert round(cast(float, stage_b['gain']), 4) == 0.0683 and round(cast(float, stage_b['spread']), 5) == 0.00105
    # Robust to the choice of spread statistic: even the largest paired OFF-A/OFF-B spread is below the gain.
    assert max(spreads) < cast(float, stage_b['gain'])
    assert summary['adoptable'] is False and summary['experiment_commit'] == '564535f13ffc9875a3c839741fbb64dd7eb3571d'
