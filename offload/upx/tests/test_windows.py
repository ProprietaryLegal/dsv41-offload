"""CPU-only: the R-UPX window designs meet WIN2 and the brief; verdict semantics; arms exist where they are named."""
from __future__ import annotations

import dataclasses
import json
import re
from typing import cast

import pytest

from offload.launch.windows import decision_v2 as dv
from offload.prefill_maxlen.evidence import Json, number, record
from offload.upx import band
from offload.upx import configs
from offload.upx import windows as w
from offload.upx.tests.test_build_0085b import git_show


@pytest.mark.parametrize('name', [*w.WINDOWS, w.SHARED.name])
def test_every_design_meets_win2(name: str) -> None:
    w.validate(w.WINDOWS.get(name, w.SHARED))


def test_estimated_holds_are_stable() -> None:
    assert {n: (w.hold_minutes(x), w.hold_minutes(x, last=True)) for n, x in w.WINDOWS.items()} == {
        'UPX-0064': (83, 106), 'UPX-PIN': (78, 100), 'UPX-0085B': (66, 89), 'UPX-0067': (58, 81)}
    assert (w.hold_minutes(w.SHARED), w.hold_minutes(w.SHARED, last=True)) == (135, 157)
    chained = sum(w.hold_minutes(w.WINDOWS[n]) for n in w.CHAIN[:-1]) + w.hold_minutes(w.WINDOWS[w.CHAIN[-1]], last=True)
    assert chained == 308


def test_0064_and_pin_decide_only_on_band_cells() -> None:
    for name in ('UPX-0064', 'UPX-PIN'):
        group = w.GROUPS[name]
        assert group.band_required and set(group.decision_cells) == set(band.BAND_CELLS)
    mixed = dataclasses.replace(w.GROUPS['UPX-0064'], decision_cells=('ctl-mixed150k-4user', 'ctl-32k-b4'))
    groups = {**w.GROUPS, 'UPX-0064': mixed}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(w, 'GROUPS', groups)
        with pytest.raises(ValueError, match='33-63-row steps'):
            w.validate(w.WINDOWS['UPX-0064'])


def test_validate_refuses_short_repeats_missing_gates_and_unbracketed_controls() -> None:
    window = w.WINDOWS['UPX-0064']
    thin = dataclasses.replace(window.starts[1], steps=(*window.starts[1].steps[:-2], *w.band_cells(2)))
    with pytest.raises(ValueError, match='< 3 per arm'):
        w.validate(dataclasses.replace(window, starts=(window.starts[0], thin, window.starts[2])))
    no_probe = dataclasses.replace(window.starts[1], steps=tuple(s for s in window.starts[1].steps if s.kind != 'band-probes'))
    with pytest.raises(ValueError, match=r"\['band-probes'\] missing"):
        w.validate(dataclasses.replace(window, starts=(window.starts[0], no_probe, window.starts[2])))
    no_kl = dataclasses.replace(window.starts[2], steps=tuple(s for s in window.starts[2].steps if s.kind != 'kl'))
    with pytest.raises(ValueError, match='next control: kl'):
        w.validate(dataclasses.replace(window, starts=(window.starts[0], window.starts[1], no_kl)))
    with pytest.raises(ValueError, match='bracket'):
        w.validate(dataclasses.replace(window, starts=(window.starts[1], window.starts[0], window.starts[2])))
    pin = w.WINDOWS['UPX-PIN']
    no_census = dataclasses.replace(pin.starts[1], steps=tuple(s for s in pin.starts[1].steps if s.kind != 'census'))
    with pytest.raises(ValueError, match="'census'"):
        w.validate(dataclasses.replace(pin, starts=(pin.starts[0], no_census, pin.starts[2])))


def test_0085b_pairs_give_three_paired_repeats_per_rank() -> None:
    window = w.WINDOWS['UPX-0085B']
    assert w.decision_repeats(window, w.GROUPS['UPX-0085B']) == {'mem-r0': {'control': 3, 'candidate': 3},
                                                                  'mem-r1': {'control': 3, 'candidate': 3}}
    assert w.GROUPS['UPX-0085B'].paired and not w.GROUPS['UPX-0064'].paired


def decision(group_name: str, control: list[float], candidate: list[float]) -> dict[str, Json]:
    group = w.GROUPS[group_name]
    rows = []
    for cell in group.decision_cells:
        for metric, higher in group.metrics[cell]:
            primary = metric == group.metrics[cell][0][0]
            base = control if primary else [3.0, 3.01, 2.99, 3.0][:len(control)]
            got = candidate if primary else base[:len(candidate)]
            rows.append(w.series(group, cell, metric, base, got))
    return dict(dv.decide_window(rows, group.decision_cells, groups=group.groups))


def gates(group_name: str, ok: bool = True) -> dict[str, bool]:
    group = w.GROUPS[group_name]
    return {g: ok for g in {*group.required_gates, *group.follow_gates}}


def test_speed_verdicts() -> None:
    faster = decision('UPX-0064', [80.0, 80.4, 79.8, 80.2], [76.0, 76.3, 75.9])
    assert faster['status'] == 'CANDIDATE'
    assert w.arm_verdict('UPX-0064', faster, gates('UPX-0064')) == 'CANDIDATE FOR PROMOTION'
    same = decision('UPX-0064', [80.0, 80.4, 79.8, 80.2], [80.1, 79.9, 80.2])
    assert w.arm_verdict('UPX-0064', same, gates('UPX-0064')) == 'NO GAIN SHOWN (stays off)'
    slower = decision('UPX-PIN', [80.0, 80.4, 79.8, 80.2], [84.0, 84.3, 83.9])
    assert w.arm_verdict('UPX-PIN', slower, gates('UPX-PIN')) == 'NOT A CANDIDATE'
    bad = {**gates('UPX-0064'), 'band-greedy': False}
    assert w.arm_verdict('UPX-0064', faster, bad) == 'STOP: band-greedy'
    with pytest.raises(ValueError, match='required'):
        w.arm_verdict('UPX-PIN', faster, gates('UPX-0064'))
    with pytest.raises(ValueError, match='win2-v1'):
        w.arm_verdict('UPX-0064', {**faster, 'rule': 'v1'}, gates('UPX-0064'))


def test_memory_verdict_is_paired_and_higher_is_better() -> None:
    off = [6_700_000.0, 6_710_000.0, 6_690_000.0]
    on = [7_250_000.0, 7_240_000.0, 7_260_000.0]
    result = decision('UPX-0085B', off, on)
    assert result['status'] == 'CANDIDATE'
    cells = result['cells']
    assert isinstance(cells, dict)
    assert w.arm_verdict('UPX-0085B', result, gates('UPX-0085B')) == 'CANDIDATE FOR PROMOTION'


def test_0085b_arm_is_the_configs_arm_and_controls_are_the_base() -> None:
    assert dict(w.ARMS['release-file'].env) == configs.ARMS['release-file-prod']
    assert dict(w.ARMS['C-upx'].env) == configs.ARMS['C-prod'] and w.ARMS['C-upx'].config == configs.CONFIG
    assert w.GROUPS['UPX-0085B'].control == 'C-upx'


def test_up3_arms_exist_at_the_pinned_commit() -> None:
    env = git_show(w.UP3_COMMIT, w.UP3_CONFIG)
    admitted = re.search(r'case "\$UP3_ARM" in ([^)]*)\)', env)
    assert admitted is not None
    names = set(admitted.group(1).split('|'))
    for arm in ('C-prod', 'split', 'cpu-pin', 'vor'):
        assert w.ARMS[arm].config == w.UP3_CONFIG and w.ARMS[arm].env['UP3_ARM'] in names
    assert 'split) EXP_DOCKER_ENV="$EXP_DOCKER_ENV VLLM_EXL3_MOE_SPLIT_MAX_TOKENS=63"' in env
    assert 'cpu-pin) EXP_DOCKER_ENV="$EXP_DOCKER_ENV EXL3_CPU_PIN=5-9,15-19"' in env
    assert 'vor) EXP_DOCKER_ENV="$EXP_DOCKER_ENV VLLM_DSV41_VOR_FUSED=1"' in env
    for arm in ('split', 'vor', 'cpu-pin'):
        manifest = json.loads(git_show(w.UP3_COMMIT, f'offload/up3/manifests/{arm}.json'))
        assert manifest['production'] == '596ba1cf4e3b11be29b41a57e7f69a5d8efa4799' and manifest['fuzz'] == 0
    split = json.loads(git_show(w.UP3_COMMIT, 'offload/up3/manifests/split.json'))
    adopt = json.loads((configs.ROOT / 'offload/upstream_adopt/MANIFEST.json').read_text())
    adopt_sha = {row['path']: row['sha256'] for row in adopt['files']}
    for rel, sha in split['files'].items():
        assert adopt_sha[rel] == sha, 'R-UP3 split bytes == UP1 0064 bytes (both = donor 0064 on v5)'


def test_plan_lists_every_window_with_its_estimates() -> None:
    plan = w.plan()
    assert list(plan) == [*w.CHAIN, 'UPX-W1']
    json.dumps(plan, allow_nan=False)


def test_0085b_ttft_guard_loses_only_with_its_whole_interval_below_zero() -> None:
    """WIN2-R2 rule revision 2 on the 0085b head cell (paired ON/OFF): a 2-of-3 slower next prefill with one large gain
    is no guard LOSS any more (interval spans zero); a consistent ~10 % slower next prefill still is."""
    group = w.GROUPS['UPX-0085B']
    off, on = [6_700_000.0, 6_710_000.0, 6_690_000.0], [7_250_000.0, 7_240_000.0, 7_260_000.0]

    def decide(ttft_on: list[float]) -> dict[str, Json]:
        rows = []
        for cell in group.decision_cells:
            for metric, _ in group.metrics[cell]:
                ttft = metric == 'next_ttft_s'
                rows.append(w.series(group, cell, metric, [10.0, 10.2, 9.8] if ttft else off, ttft_on if ttft else on))
        return dict(dv.decide_window(rows, group.decision_cells, groups=group.groups))

    noisy = decide([10.6, 9.0, 10.6])
    guard = record(record(record(record(noisy['cells'])['mem-r0'])['metrics'])['next_ttft_s'])
    assert guard['loss_repeat_test'] is True and guard['verdict'] == 'INCONCLUSIVE'
    assert noisy['rule_revision'] == 2 and noisy['status'] == 'CANDIDATE'
    assert w.arm_verdict('UPX-0085B', noisy, gates('UPX-0085B')) == 'CANDIDATE FOR PROMOTION'
    slower = decide([11.0, 11.2, 10.8])
    guard = record(record(record(record(slower['cells'])['mem-r0'])['metrics'])['next_ttft_s'])
    assert guard['verdict'] == 'LOSS' and number(cast(list[Json], guard['interval95'])[1]) < 0
    assert w.arm_verdict('UPX-0085B', slower, gates('UPX-0085B')) == 'NOT A CANDIDATE'
