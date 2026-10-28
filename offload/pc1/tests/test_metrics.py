from __future__ import annotations

import copy
from typing import cast

import pytest

from offload.pc1 import metrics, cells, window
from offload.pc1.tests.test_window import starts
from offload.prefill_maxlen.evidence import Json, record


@pytest.mark.parametrize('candidate,win', [(98., False), (97.999, True), (100., False), (101., False)])
def test_strict_two_percent_boundary(candidate: float, win: bool) -> None:
    row = metrics.compare([[100., 100.], [candidate, candidate], [100., 100.]], lower_is_better=True)
    assert row['win'] is win
    assert row['required_gain'] == 2


def test_paired_lower_bound_and_control_spread_dominate_point_win() -> None:
    noisy = metrics.compare([[100., 100.], [90., 105.], [100., 100.]], lower_is_better=True)
    assert noisy['point_gain'] == 2.5 and noisy['win'] is False
    assert noisy['paired_gain_interval95'] == [-5., 10.]
    assert noisy['no_loss_beyond_spread'] is False
    tied = metrics.compare([[100., 100.], [98., 98.], [104., 104.]], lower_is_better=True)
    assert tied['control_spread'] == 4 and tied['required_gain'] == 4 and tied['win'] is False
    assert metrics.compare([[100., 100.], [97.99, 97.99], [104., 104.]], lower_is_better=True)['win'] is True


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1., 0.])
def test_invalid_timing_refuses(value: float) -> None:
    with pytest.raises(ValueError):
        metrics.compare([[100., 100.], [value, 90.], [100., 100.]], lower_is_better=True)


def first_cell(rows: dict[str, Json], name: str) -> dict[str, Json]:
    return record(record(record(rows['s2-chunk2048'])['cells'])[name])


def first_stream(cell: dict[str, Json]) -> dict[str, Json]:
    return record(cast(list[Json], record(cast(list[Json], cell['repetitions'])[0])['streams'])[0])


@pytest.mark.parametrize('fault', ['prompt', 'cache', 'salt', 'output', 'seed', 'overlap', 'counter', 'reps', 'context', 'gaps'])
def test_invalid_evidence_cannot_drive_win(fault: str) -> None:
    rows = starts('chunk2048')
    cell = first_cell(rows, 'mixed150k-2decode' if fault == 'overlap' else '150k-b1')
    stream = first_stream(cell)
    repeat = record(cast(list[Json], cell['repetitions'])[0])
    if fault == 'prompt':
        stream['prompt_sha256'] = 'changed'
    elif fault == 'cache':
        record(record(stream['usage'])['prompt_tokens_details'])['cached_tokens'] = 1
    elif fault == 'salt':
        other = copy.deepcopy(first_cell(rows, '32k-b1'))
        stream['cache_salt_sha256'] = first_stream(other)['cache_salt_sha256']
    elif fault == 'output':
        record(stream['usage'])['completion_tokens'] = 511
    elif fault == 'seed':
        record(stream['sampling'])['seed'] = 99
    elif fault == 'overlap':
        repeat['mixed_three_active_overlap'] = False
    elif fault == 'counter':
        repeat['spec_request_rounds'] = 0
    elif fault == 'context':
        stream['prompt_tokens'] = record(stream['usage'])['prompt_tokens'] = 32000
    elif fault == 'gaps':
        stream['delivery_gap_samples'] = 510
    else:
        cell['repetitions'] = cast(list[Json], cell['repetitions'])[:1]
    with pytest.raises(ValueError):
        window.decision('chunk2048', rows)


@pytest.mark.parametrize('name', ['32k-b1', '150k-b1'])
def test_no_decision_cell_loss_is_hidden_by_the_other_cells_win(name: str) -> None:
    # win2-v1: a decision cell worse than its control spread in 2 of 3 repeats vetoes the window.
    rows = starts('chunk2048')
    for repetition in cast(list[Json], first_cell(rows, name)['repetitions'])[:2]:
        for stream in cast(list[Json], record(repetition)['streams']):
            record(stream)['ttft_s'] = 1.1
    assert window.decision('chunk2048', rows)['status'] == 'NOT A CANDIDATE'


def test_decode_win_without_ttft_win_is_not_candidate() -> None:
    # Decode/server metrics of the larger-chunk windows are reported only; a TTFT tie cannot become a CANDIDATE.
    rows = starts('chunk2048')
    for name in cells.plan_for('chunk2048').cells:
        cell = first_cell(rows, name)
        for index, repetition in enumerate(cast(list[Json], cell['repetitions'])):
            for stream in cast(list[Json], record(repetition)['streams']):
                record(stream)['ttft_s'] = 1. + index / 1000
    assert window.decision('chunk2048', rows)['status'] == 'INCONCLUSIVE'


def test_zero_delivery_gap_is_valid_and_cannot_create_ttft_win() -> None:
    row = metrics.compare([[0., 0.], [0., 0.], [0., 0.]], lower_is_better=True, allow_zero=True)
    assert row['no_loss_beyond_spread'] is True and row['win'] is False


def test_server_rate_pools_counters_instead_of_averaging_rates() -> None:
    row = metrics.compare([[1., 10.], [2., 20.], [1., 10.]], lower_is_better=False,
                          weights=[[100., 1.], [100., 1.], [100., 1.]])
    assert row['start_means'] == [110 / 101, 220 / 101, 110 / 101]
