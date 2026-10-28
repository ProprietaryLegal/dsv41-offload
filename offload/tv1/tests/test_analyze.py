from __future__ import annotations

import copy
from pathlib import Path
from typing import cast

import pytest

from offload.prefill_maxlen.evidence import Json, record
from offload.tv1 import analyze as a
from offload.tv1 import cells as c
from offload.tv1 import greedy as g


def points(feature: str) -> list[dict[str, Json]]:
    output: list[dict[str, Json]] = []
    for index in range(3):
        on = index == 1
        cells: dict[str, Json] = {}
        for name in c.controls(feature):
            affected = on and name == ('150k-b1' if feature == 'candidate' else '32k-b4')
            cells[name] = {'profiled_round_ms': 40., 'server_decode_ms_per_spec_request_round': 37. if affected else 40., 'ttft_s': 70.,
                           'client_aggregate_decode_tps': 60. if affected else 55.,
                           'prompts': ['a'], 'acceptance': {'acceptance_fraction': .6, 'requests':4, 'rounds':400, 'accepted_tokens':400}}
        output.append({'cells': cells})
    return output


@pytest.mark.parametrize('feature', ['candidate', 'compact'])
def test_pre_registered_paired_reading(feature: str) -> None:
    x = points(feature)
    assert a.compare(feature, x)['verdict'] == 'WINDOW_PASS_FOR_REVIEW'
    broken = copy.deepcopy(x)
    broken[2] = broken[1]
    assert a.compare(feature, broken)['verdict'] == 'HOLD'
    with pytest.raises(ValueError, match='OFF/ON'):
        a.compare(feature, x[:2])


def test_round_threshold_is_relative_to_off_time() -> None:
    x = points('candidate')
    for index in (1,):
        record(record(x[index]['cells'])['150k-b1'])['server_decode_ms_per_spec_request_round'] = 39.21
    # A 1.975% time decrease is below the 2% rule, even though the reciprocal
    # speedup exceeds 2%. All threshold percentages use the OFF denominator.
    assert a.compare('candidate', x)['verdict'] == 'HOLD'
    for index in (1,):
        record(record(x[index]['cells'])['150k-b1'])['server_decode_ms_per_spec_request_round'] = 39.19
        record(record(x[index]['cells'])['short-b1'])['server_decode_ms_per_spec_request_round'] = 40.81
    assert a.compare('candidate', x)['verdict'] == 'HOLD'
    for index in (1,):
        record(record(x[index]['cells'])['short-b1'])['server_decode_ms_per_spec_request_round'] = 40.79
    assert a.compare('candidate', x)['verdict'] == 'WINDOW_PASS_FOR_REVIEW'


@pytest.mark.parametrize('feature', ['candidate', 'compact'])
def test_reading_requires_complete_batch_profiles_and_acceptance(feature: str, tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    # Reader gates use numerical artifacts only; trace parsing itself is covered by
    # the real SP1 synthetic traces in the window rehearsal and upstream suite.
    import json
    from pathlib import Path
    from offload.prefill_maxlen.evidence import record
    from offload.gb10_budget import trace_budget as tb
    root = Path(str(tmp_path))
    (root / 'cells').mkdir()
    (root / 'greedy.json').write_text(json.dumps(g.registered({'checks':[
        {'i':i,'identical':True} for i in range(6)], 'identical':6, 'near_tie':0})))
    captures: dict[str, Json] = {}
    for name in c.controls(feature):
        batch = 4 if name.endswith('b4') else 1
        ranks: dict[str, Json] = {}
        for rank in ('0','1'):
            ranks[rank] = {'graph_visibility':'visible', 'aggregate':{'coverage':1.,'period_ms_median':40.},
                           'steps':[{'complete':True, 'phase':'decode', 'generation_requests':batch}]*8}
        captures['prof-'+name+':decode'] = {'ranks_present':[0,1], 'aligned':True, 'ranks':ranks}
        from offload.tv1.tests.test_metrics import trial
        trials: list[Json] = []
        for repetition in range(4//batch):
            row = trial(batch=batch, offset=1000*repetition)
            row.update({'cell':'ctl-'+name,'profiled':False,'forced_output_tokens':512,'repetition':repetition,
                'streams':[{'ttft_s':10.,'submitted_rel_s':0.,'total_s':30.,'finish':'length',
                            'usage':{'completion_tokens':512},'prompt_sha256':str(i)} for i in range(batch)]})
            trials.append(row)
        (root / 'cells' / ('ctl-'+name+'.json')).write_text(json.dumps({'type':'tv1-repeated-cell-v3',
            'cell':'ctl-'+name,'profiled':False,'trials':trials}))
    budget: dict[str, Json] = {'captures':captures, 'overhead':{'decode_overhead':.01}}
    monkeypatch.setattr(tb, 'analyze_window', lambda path: budget)
    assert set(record(a.read_arm(root, feature)['cells'])) == set(c.controls(feature))
    checked = g.registered({'checks':cast(list[Json], [
        {'i':i,'identical':True} for i in range(6)]), 'identical':5, 'near_tie':1})
    checks = cast(list[Json],checked['checks'])
    checks[3] = {'i':3,'identical':False,'near_tie':True,'token_index':2,'margin_nats':.5,'gap_to_top_nats':0.}
    (root/'greedy.json').write_text(json.dumps(checked))
    assert a.read_arm(root,feature)['greedy'] == checked
    record(checks[3])['near_tie'] = False
    (root/'greedy.json').write_text(json.dumps(checked))
    with pytest.raises(ValueError,match='not an NBW near-tie'):
        a.read_arm(root,feature)
    record(checks[3])['near_tie'] = True
    (root/'greedy.json').write_text(json.dumps(checked))
    broken = record(record(record(captures['prof-'+c.controls(feature)[1]+':decode'])['ranks'])['0'])
    broken['steps'] = [{'complete':True,'phase':'decode','generation_requests':(1 if feature == 'compact' else 4)}]*8
    with pytest.raises(ValueError, match='surviving-request tail'):
        a.read_arm(root, feature)


def test_profile_speed_cannot_substitute_for_server_request_round_gain() -> None:
    x = points('candidate')
    for index in (1,):
        center = record(record(x[index]['cells'])['150k-b1'])
        center['server_decode_ms_per_spec_request_round'] = 40
        center['profiled_round_ms'] = 1
    assert a.compare('candidate',x)['verdict'] == 'HOLD'


def test_compact_requires_both_server_round_gain_and_aggregate_tokens() -> None:
    x = points('compact')
    for index in (1,):
        record(record(x[index]['cells'])['32k-b4'])['server_decode_ms_per_spec_request_round'] = 40
    assert a.compare('compact',x)['verdict'] == 'HOLD'


def test_acceptance_loss_cannot_buy_a_faster_round() -> None:
    x = points('candidate')
    for index in (1,):
        record(record(record(x[index]['cells'])['150k-b1'])['acceptance'])['accepted_tokens'] = 0
    assert a.compare('candidate',x)['verdict'] == 'HOLD'


def test_fewer_than_four_acceptance_requests_refuses() -> None:
    x = points('candidate')
    record(record(record(x[1]['cells'])['150k-b1'])['acceptance'])['requests'] = 1
    with pytest.raises(ValueError, match='>=4 requests'):
        a.compare('candidate',x)


@pytest.mark.parametrize('feature', ['candidate', 'compact'])
def test_each_off_comparison_must_pass_even_when_the_mean_would_pass(feature: str) -> None:
    x = points(feature)
    target = c.controls(feature)[1]
    record(record(x[0]['cells'])[target])['server_decode_ms_per_spec_request_round'] = 40
    record(record(x[2]['cells'])[target])['server_decode_ms_per_spec_request_round'] = 40.7
    # OFF drift is 1.75%; ON gain from the mean is 2.85%, but only 2% or
    # better against BOTH controls qualifies. Here the first comparison is 1.975%.
    record(record(x[1]['cells'])[target])['server_decode_ms_per_spec_request_round'] = 39.21
    assert a.compare(feature, x)['verdict'] == 'HOLD'
    record(record(x[1]['cells'])[target])['server_decode_ms_per_spec_request_round'] = 39.19
    result = a.compare(feature, x)
    assert result['verdict'] == 'WINDOW_PASS_FOR_REVIEW'
    paired = result['paired']
    assert isinstance(paired, list)
    assert [record(p)['off_start_index'] for p in paired] == [0, 2]


def test_partial_or_unexpected_schedule_cannot_qualify() -> None:
    x = points('candidate')
    with pytest.raises(ValueError, match='fixed OFF/ON/OFF'):
        a.compare('candidate', x + x[:2])
    del record(x[2]['cells'])['short-b1']
    with pytest.raises(ValueError, match='exact target'):
        a.compare('candidate', x)


def test_compact_throughput_must_beat_each_control() -> None:
    x = points('compact')
    record(record(x[2]['cells'])['32k-b4'])['client_aggregate_decode_tps'] = 56
    record(record(x[1]['cells'])['32k-b4'])['client_aggregate_decode_tps'] = 58.7
    assert a.compare('compact', x)['verdict'] == 'HOLD'


@pytest.mark.parametrize('bad', ['missing', 'extra', 'reordered', 'design'])
def test_summary_requires_exact_ordered_three_starts(bad: str, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    import json
    from offload.launch.windows import tv1_window as tv
    names = [arm.name for arm in tv.arms('compact')]
    x = points('compact')
    artifact = tmp_path/'receipt.json'
    artifact.write_text('{}')
    summary: dict[str, Json] = {'status':'COMPLETE', 'feature':'compact', 'design':'C/arm/C-v4',
        'registration_id':g.REGISTRATION_ID, 'greedy_rule':g.RULE,
        'starts':{name:{} for name in names}, 'restore':{'status':'RESTORED'},
        'post_close_watch':{'status':'PASS'},
        'analysis_sha256':hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),
        'artifact_sha256':{'receipt.json':hashlib.sha256(artifact.read_bytes()).hexdigest()}}
    path = tmp_path/'summary.json'
    path.write_text(json.dumps(summary))
    monkeypatch.setattr(a, 'read_arm', lambda root, feature: x[names.index(root.name)])
    assert a.analyze(tmp_path)['verdict'] == 'WINDOW_PASS_FOR_REVIEW'
    if bad == 'design':
        summary['design'] = 'old-five-start'
    else:
        changed = names[:2] if bad == 'missing' else names+['s4-compact'] if bad == 'extra' else names[::-1]
        summary['starts'] = {name:{} for name in changed}
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='fixed C/arm/C starts'):
        a.analyze(tmp_path)


@pytest.mark.parametrize('registration', [None, 'historical-strict'])
def test_historical_summary_cannot_be_read_under_new_rule(registration: str | None, tmp_path: Path) -> None:
    import json
    (tmp_path/'summary.json').write_text(json.dumps({'registration_id':registration}))
    with pytest.raises(ValueError,match='historical strict runs retain their verdict'):
        a.analyze(tmp_path)
