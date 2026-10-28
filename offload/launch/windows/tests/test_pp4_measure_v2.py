"""CPU checks of R-PP4's WIN2 measurement: win2-v1 decisions, production-policy arm, carried qualification, chain mode.

All serving, host and GPU operations are fakes; the only real files read are the registered round6 budget, the frozen
production env at PROD_COMMIT and (when present) a real PP4 window's rollback.json.
"""
from __future__ import annotations

import copy
import json
import os
import shlex
import signal
import subprocess
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import chain
from offload.launch.windows import decision_v2 as d
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pp4_budget as budget
from offload.launch.windows import pp4_cells as cells
from offload.launch.windows import pp4_measure_v2 as m
from offload.launch.windows import pp4_prerequisites as prereq
from offload.launch.windows import pp4_window as w
from offload.launch.windows import scoped_remote as owner
from offload.launch.windows.tests import test_pf1_window as base
from offload.launch.windows.tests import test_pp4_window as prior
from offload.prefill_maxlen.evidence import Json, number, record

GATE: dict[str, Json] = {'status': 'PASS', 'receipt': '/fixture', 'pp4_kernel_sha256': 'a' * 64}
CLOCK_OK: dict[str, Json] = {'consistent': True, 'reason': None}
REAL_ROLLBACK = pf.CAMPAIGN / 'PP4-429-admission-W-20261027T133630Z/rollback.json'


def v2_cell(name: str, start: str, *, accepted: float = 200., seconds: float = 4., tps: float = 100.) -> dict[str, Json]:
    """Six registered waves; repeat r adds content variation shared by every start (paired by repeat)."""
    rows: list[Json] = []
    for wave in range(m.WAVES):
        r = wave // m.WAVES_PER_REPEAT
        rows.append({'wave': wave, 'repeat': r, 'family': cells.FAMILIES[wave % 2], 'batch': 1,
                     'spec_request_rounds': 100., 'accepted_tokens': float(round(accepted * (1 + .05 * r) + wave % 2)),
                     'server_decode_seconds': seconds * (1 + .01 * r), 'median_ttft_s': 1. + .1 * r,
                     'median_decode_tps': tps * (1 + .02 * r), 'completed_requests': 1,
                     'streams': [{'prompt_sha256': f'{name}-{m.ORDERS[name][wave]}-{cells.FAMILIES[wave % 2]}',
                                  'cache_salt_sha256': f'{start}-{name}-{wave}',
                                  'sampling': {'temperature': 1, 'top_p': 1, 'seed': 42 + wave, 'priority': 10,
                                               'max_tokens': 512, 'ignore_eos': True},
                                  'usage': {'completion_tokens': 512}, 'finish': 'length'}]})
    return {'cell': name, 'batch': 1, 'metric_registration': m.REGISTRATION, 'waves': rows}


GREEDY: dict[str, Json] = {'greedy_registration': cells.GREEDY_REGISTRATION, 'identical': 6, 'near_tie': 0,
                           'checks': [{'i': i, 'identical': True} for i in range(6)]}


def starts(mode: str, *, candidate: dict[str, float] | None = None) -> dict[str, Json]:
    out: dict[str, Json] = {}
    for name, role in m.MODES[mode]:
        values = candidate if role == 'B' and candidate is not None else {}
        block: dict[str, Json] = {'role': role, 'status': 'COMPLETE',
                                  'cells': {c: v2_cell(c, name, **values) for c in m.CELLS}}
        if name in m.GREEDY[mode]:
            block['greedy'] = copy.deepcopy(GREEDY)
        out[name] = block
    return out


WIN = {'accepted': 230., 'seconds': 4.1, 'tps': 108.}   # +15 % accepted, +2.5 % ms/round: rate +7 %


def test_v2_prompts_are_distinct_per_family_and_cuts_exist() -> None:
    for name in m.CELLS:
        for family in cells.FAMILIES:
            keys = [m.wave_prompts(name, w_)[0] for w_ in range(m.WAVES) if cells.FAMILIES[w_ % 2] == family]
            assert len(set(keys)) == m.REPEATS
        assert all(m.wave_prompts(name, w_)[0] in cells.PROMPTS for w_ in range(m.WAVES))
    with pytest.raises(ValueError):
        m.wave_prompts('32k-b4', 0)


def test_repeat_values_pool_exact_counters() -> None:
    rows = m.waves(v2_cell('short-b1', 's1-A'))
    reps = m.repeat_values(rows)
    assert [r['waves'] for r in reps] == [[0, 1], [2, 3], [4, 5]]
    first = reps[0]
    assert first['accepted_per_round'] == (200. + 201.) / 200
    assert first['ms_per_round'] == 1000 * 8. / 200
    assert first[m.PRIMARY] == pytest.approx((200 + 401) / 8.)


@pytest.mark.parametrize('change', ['registration', 'order', 'seed', 'duplicate-prompt', 'duplicate-salt', 'short-output'])
def test_wave_evidence_refusals(change: str) -> None:
    cell = v2_cell('32k-b1', 's1-A')
    rows = cast(list[dict[str, Json]], cell['waves'])
    stream = record(cast(list[Json], rows[3]['streams'])[0])
    if change == 'registration':
        cell['metric_registration'] = 'pp4-r6-512-aba-no-measurement-profiles-v1'
    elif change == 'order':
        rows[2]['family'], rows[3]['family'] = rows[3]['family'], rows[2]['family']
    elif change == 'seed':
        record(stream['sampling'])['seed'] = 42
    elif change == 'duplicate-prompt':
        stream['prompt_sha256'] = record(cast(list[Json], rows[1]['streams'])[0])['prompt_sha256']
    elif change == 'duplicate-salt':
        stream['cache_salt_sha256'] = record(cast(list[Json], rows[1]['streams'])[0])['cache_salt_sha256']
    else:
        record(stream['usage'])['completion_tokens'] = 511
    with pytest.raises(ValueError):
        m.waves(cell)


def test_controlled_win_is_decided_on_the_rate_not_on_ms_per_round() -> None:
    result = m.decide(starts('controlled', candidate=WIN), 'controlled', clock=CLOCK_OK, rng=GATE, archive=None)
    assert result['status'] == 'CANDIDATE' and result['rule'] == d.RULE == 'win2-v1' and result['min_gain'] == .02
    assert record(result['groups'])['single-user'] == 'CANDIDATE'
    for cell in m.CELLS:
        metrics = record(record(record(result['cells'])[cell])['metrics'])
        assert list(metrics) == [m.PRIMARY, 'accepted_per_round', 'decode_tps']
        assert record(metrics[m.PRIMARY])['verdict'] == 'WIN'
        reported = record(record(result['reported_metrics'])[cell])
        assert number(record(reported['ms_per_round'])['point_gain']) < 0  # the cost PP4 pays, visible and reported
    # What WIN2.md's ms/round-primary default would have said about the same evidence.
    assert record(result['checklist_literal'])['status'] != 'CANDIDATE'


def recompute(series_rows: Json) -> dict[str, Json]:
    series = [d.Series(str(r['cell']), str(r['metric']), bool(r['higher_is_better']),
                       tuple(number(v) for v in cast(list[Json], r['control'])),
                       tuple(number(v) for v in cast(list[Json], r['candidate'])), bool(r['paired']))
              for r in (record(x) for x in cast(list[Json], series_rows))]
    return d.decide_window(series, m.CELLS)


@pytest.mark.parametrize('candidate,clock', [(WIN, CLOCK_OK), ({'accepted': 150.}, CLOCK_OK),
                                             (WIN, {'consistent': False, 'reason': 'caps differ'})])
def test_decision_recomputes_from_its_recorded_series(candidate: dict[str, float], clock: dict[str, Json]) -> None:
    """The consumer contract (pp4_promote measurement_gate): rule/status/min_gain/cells recompute from decision_series."""
    result = m.decide(starts('production-policy', candidate=candidate), 'production-policy', clock=clock, rng=GATE, archive=None)
    again = recompute(result['decision_series'])
    same = all(json.dumps(again[k], sort_keys=True) == json.dumps(result[k], sort_keys=True)
               for k in ('rule', 'status', 'min_gain', 'decision_cells', 'cells'))
    assert same is bool(clock.get('consistent'))   # a clock-guarded verdict must NOT pass as recomputed evidence
    rows = [record(r) for r in cast(list[Json], result['decision_series'])]
    assert [(r['cell'], r['metric']) for r in rows] == [(c, k) for c in m.CELLS for k in (m.PRIMARY, 'accepted_per_round', 'decode_tps')]
    assert {(record(r)['cell'], record(r)['metric']) for r in cast(list[Json], result['reported_series'])} == {
        (c, k) for c in m.CELLS for k in ('ms_per_round', 'ttft_s')}


def test_controlled_loss_and_guard_loss() -> None:
    worse = m.decide(starts('controlled', candidate={'accepted': 150.}), 'controlled', clock=CLOCK_OK, rng=GATE, archive=None)
    assert worse['status'] == 'NOT A CANDIDATE'
    # Rate better through a much cheaper round, but client tok/s collapses: the guard alone makes it a LOSS.
    guard = m.decide(starts('controlled', candidate={'accepted': 200., 'seconds': 3.5, 'tps': 60.}), 'controlled',
                     clock=CLOCK_OK, rng=GATE, archive=None)
    assert guard['status'] == 'NOT A CANDIDATE'
    metrics = record(record(record(guard['cells'])['short-b1'])['metrics'])
    assert record(metrics[m.PRIMARY])['verdict'] == 'WIN' and record(metrics['decode_tps'])['verdict'] == 'LOSS'


def test_production_policy_mode_requires_greedy_on_first_p_and_b() -> None:
    rows = starts('production-policy', candidate=WIN)
    result = m.decide(rows, 'production-policy', clock=CLOCK_OK, rng=GATE, archive=None)
    assert result['status'] == 'CANDIDATE' and 'not the production binary' in str(result['comparison'])
    record(rows['s1-P']).pop('greedy')
    with pytest.raises(KeyError):
        m.decide(rows, 'production-policy', clock=CLOCK_OK, rng=GATE, archive=None)
    rows = starts('production-policy', candidate=WIN)
    record(cast(list[Json], record(record(rows['s1-P'])['greedy'])['checks'])[2])['identical'] = False
    with pytest.raises(ValueError, match='s1-P'):
        m.decide(rows, 'production-policy', clock=CLOCK_OK, rng=GATE, archive=None)


@pytest.mark.parametrize('change', ['clock', 'incomplete', 'missing-start', 'rng', 'inputs', 'role'])
def test_decision_refusals_and_inconclusive(change: str) -> None:
    rows = starts('controlled', candidate=WIN)
    clock: dict[str, Json] = dict(CLOCK_OK)
    rng: dict[str, Json] | None = GATE
    if change == 'clock':
        clock = {'consistent': False, 'reason': 'thermal watchdog dialled the clock back during the HOLD'}
    elif change == 'incomplete':
        record(rows['s3-A'])['status'] = 'STARTED'
    elif change == 'missing-start':
        rows.pop('s3-A')
    elif change == 'rng':
        rng = {'status': 'FAIL'}
    elif change == 'inputs':
        wave = record(cast(list[Json], record(record(record(rows['s2-B'])['cells'])['short-b1'])['waves'])[0])
        record(cast(list[Json], wave['streams'])[0])['prompt_sha256'] = 'other'
    else:
        record(rows['s2-B'])['role'] = 'A'
    if change in ('rng', 'inputs', 'role'):
        with pytest.raises(ValueError):
            m.decide(rows, 'controlled', clock=clock, rng=rng, archive=None)
    else:
        assert m.decide(rows, 'controlled', clock=clock, rng=rng, archive=None)['status'] == 'INCONCLUSIVE'


def test_clock_guard_and_thermal_log(tmp_path: Path) -> None:
    text = 'sparkhead clock cap 2300 MHz\nsparkworker clock cap 2300 MHz\nport 8001 restricted\n'
    caps = m.caps_from_start(text)
    assert caps == {'sparkhead': 2300, 'sparkworker': 2300}
    names = ['s1-A', 's2-B', 's3-A']
    assert m.clock_guard({n: caps for n in names}, names, [])['consistent'] is True
    assert m.clock_guard({n: {} for n in names}, names, [])['consistent'] is True  # stock clocks throughout
    assert m.clock_guard({**{n: caps for n in names}, 's3-A': {'sparkhead': 2150, 'sparkworker': 2150}}, names, [])['consistent'] is False
    assert m.clock_guard({'s1-A': caps, 's2-B': caps}, names, [])['consistent'] is False
    log = tmp_path / 'THERMAL.log'
    log.write_text('2026-10-27T20:00:00Z sparkhead 70, 2300, 100, Not Active, Not Active\n'
                   '2026-10-27T20:10:00Z DIAL BACK to 2150: sparkhead 91, ...\n')
    import datetime
    at = datetime.datetime(2026, 10, 27, 20, 10, tzinfo=datetime.timezone.utc).timestamp()
    assert m.dial_backs(log, at - 60, at + 60) == [{'utc': '2026-10-27T20:10:00Z', 'mhz': 2150}]
    assert m.dial_backs(log, at + 1, at + 60) == []
    assert m.clock_guard({n: caps for n in names}, names, m.dial_backs(log, at - 60, at + 60))['consistent'] is False


def test_budget_v2_fits_the_hold_and_names_every_start() -> None:
    pair: dict[str, Json] = {'seconds': 333, 'classification': 'fixture'}
    controlled = m.budget_v2('controlled', pair=pair)
    assert controlled['block_budget_s'] == {'s1-A': 1277., 's2-B': 1199., 's3-A': 1157.}
    assert controlled['total_s'] == 4806. and controlled['measurement_end_s'] == 4086. and controlled['fits'] is True
    production = m.budget_v2('production-policy', pair=pair)
    assert production['block_budget_s'] == {'s1-P': 1319., 's2-B': 1199., 's3-P': 1157.} and production['fits'] is True
    assert m.budget_v2('controlled')['fits'] is True                 # 600 s dry bound: 84.55 min
    assert m.budget_v2('production-policy')['fits'] is False         # 600 s dry bound: 85.25 min; refuses


def fake_git(changed: list[str], *, qualified: str = '1' * 40, runner_old: str | None = None) -> object:
    current = (pf.ROOT / m.RUNNER).read_text()
    def git(args: list[str]) -> str:
        if args[0] == 'merge-base':
            return qualified + '\n'
        if args[0] == 'diff':
            return '\n'.join(changed) + '\n'
        if args[0] == 'show':
            return runner_old if runner_old is not None and args[1].startswith(qualified) else current
        raise AssertionError(args)
    return git


@pytest.mark.parametrize('changed,ok', [
    ([m.RUNNER, 'offload/launch/windows/pp4_measure_v2.py', 'offload/gb10_budget/window_cells.py',
      'offload/launch/windows/tests/test_pp4_window.py', 'documents/plans/x.md'], True),
    (['offload/dspark/worker/server.py'], False), (['offload/launch/windows/PP4-sampled.env'], False),
    (['offload/dspark/spark/remote_dspark/speculator.py'], False), (['offload/launch/windows/pp4_admission.py'], False)])
def test_carried_qualification_allows_only_the_measurement_layer(monkeypatch: pytest.MonkeyPatch,
                                                                 changed: list[str], ok: bool) -> None:
    monkeypatch.setattr(prereq, 'qualification', lambda path, expected: {'verdict': 'QUALIFIED', 'source_commit': expected})
    monkeypatch.setattr(budget, 'pair_reserve', lambda path, expected: {'seconds': 333, 'classification': 'fixture'})
    git = fake_git(changed)
    if ok:
        data = m.carried(Path('/receipt.json'), '1' * 40, '2' * 40, 'controlled', git=git)  # type: ignore[arg-type]
        assert record(data['registered_budget'])['fits'] is True
        assert data['block_budget_s'] == record(data['registered_budget'])['block_budget_s']
        assert record(data['carried_qualification'])['qualified_commit'] == '1' * 40
    else:
        with pytest.raises(ValueError, match='new admission'):
            m.carried(Path('/receipt.json'), '1' * 40, '2' * 40, 'controlled', git=git)  # type: ignore[arg-type]


def test_carried_qualification_refuses_runner_unit_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prereq, 'qualification', lambda path, expected: {'verdict': 'QUALIFIED'})
    monkeypatch.setattr(budget, 'pair_reserve', lambda path, expected: {'seconds': 333, 'classification': 'fixture'})
    current = (pf.ROOT / m.RUNNER).read_text()
    old = current.replace("'timeout --signal=TERM --kill-after=15s --foreground 120 bash ~/dsv41-offload/prod_window.sh pair COORD\\n'",
                          "'bash ~/dsv41-offload/prod_window.sh pair COORD\\n'")
    assert old != current
    with pytest.raises(ValueError, match='worker_script'):
        m.carried(Path('/r.json'), '1' * 40, '2' * 40, 'controlled', git=fake_git([m.RUNNER], runner_old=old))  # type: ignore[arg-type]
    monkeypatch.setattr(budget, 'pair_reserve', lambda path, expected: {'seconds': 600, 'classification': 'fixture'})
    with pytest.raises(ValueError, match='exceeds'):
        m.carried(Path('/r.json'), '1' * 40, '2' * 40, 'production-policy', git=fake_git([]))  # type: ignore[arg-type]


def test_archive_rows_carry_max_tokens_and_only_finished_control_starts(tmp_path: Path) -> None:
    rows = starts('controlled', candidate=WIN)
    clock: dict[str, Json] = {'per_start': {n: {'sparkhead': 2300, 'sparkworker': 2300} for n in ('s1-A', 's2-B', 's3-A')}}
    archive = tmp_path / 'archive.jsonl'
    result = w.append_archive({'starts': rows}, 'controlled', window_name='PP4-v2-W-20261027T210000Z',
                              experiment_commit='2' * 40, clock=clock, path=archive)
    assert {k: result[k] for k in ('status', 'rows', 'path')} == {'status': 'APPENDED', 'rows': 60, 'path': str(archive)}
    lines = [json.loads(x) for x in archive.read_text().splitlines()]
    assert {x['start'] for x in lines} == {'s1-A', 's3-A'} and all(x['max_tokens'] == 512 for x in lines)
    assert all(x['clock_cap_mhz'] == 2300 and x['clock_source'] == 'start-log' for x in lines)
    assert result['archive_head_before'] == {'rows': 0, 'head_sha256': d.GENESIS}
    assert record(result['archive_head_after'])['rows'] == 60
    assert {x['cell'] for x in lines} == {'pp4rt-fixed5-greedy-short-b1', 'pp4rt-fixed5-greedy-32k-b1'}
    assert all(x['clock_caps_mhz'] == [2300] and x['date'] == '20261027' and x['commit'] == w.PROD_COMMIT for x in lines)
    where: dict[str, Json] = {'max_tokens': 512, 'clock_cap_mhz': 2300, 'commit': w.PROD_COMMIT}
    assert d.archive_spread(archive, 'pp4rt-fixed5-greedy-short-b1', m.PRIMARY, where=where) is None
    record(rows['s3-A'])['status'] = 'STARTED'
    again = w.append_archive({'starts': rows}, 'controlled', window_name='PP4-v2-W-20261028T010000Z',
                             experiment_commit='2' * 40, clock=clock, path=archive)
    assert again['rows'] == 30
    assert d.archive_spread(archive, 'pp4rt-fixed5-greedy-short-b1', m.PRIMARY, where=where) == 0.0
    stock: dict[str, Json] = {'per_start': {n: {} for n in ('s1-A', 's2-B', 's3-A')}}   # no cap printed: refuses loudly
    assert w.append_archive({'starts': rows}, 'controlled', window_name='PP4-v2-W-20261028T020000Z',
                            experiment_commit='2' * 40, clock=stock, path=archive)['status'] == 'FAILED'
    broken = tmp_path / 'broken.jsonl'
    broken.write_text('{"schema": "other"}\n')
    assert w.append_archive({'starts': rows}, 'controlled', window_name='x', experiment_commit='2' * 40, clock=clock,
                            path=broken)['status'] == 'FAILED'


# ---- runner integration: real window() with a fake backend


class FakeV2(prior.Fake):
    def __init__(self, fail: str = '', *, down: bool = False, chain_fail: bool = False,
                 candidate: dict[str, float] | None = None) -> None:
        super().__init__(fail)
        self.down, self.chain_fail = down, chain_fail
        self.candidate = WIN if candidate is None else candidate
        self.restored_with: bool | None = None
    def cell(self, arm: w.Arm, name: str) -> dict[str, Json]:
        self.events.append('cell-' + arm.name + '-' + name)
        if self.fail == 'cell':
            raise TimeoutError('CENSORED request')
        return v2_cell(name, arm.name, **(self.candidate if arm.role == 'B' else {}))
    def greedy(self, arm: w.Arm) -> dict[str, Json]:
        self.events.append('greedy-' + arm.name)
        return copy.deepcopy(GREEDY)
    def production_down(self) -> bool: return self.down
    def owns_hold(self) -> bool: return self.fail != 'hold'
    def chain_abort(self) -> dict[str, Json]:
        self.events.append('chain-abort')
        return {'status': 'RESTORED: chain ended by abort; watch PASS; marker cleared'}
    def measurement_context(self) -> dict[str, Json]: return {'clock': dict(CLOCK_OK), 'rng_gate': dict(GATE)}
    def chain_close(self) -> dict[str, Json]:
        self.events.append('chain-close')
        if self.chain_fail:
            raise ValueError('quiet proof failed')
        return {'status': w.CHAINED, 'restore_deadline_met': True}
    def restore(self, *, production_stopped: bool) -> dict[str, Json]:
        self.restored_with = production_stopped
        if not self.owns_hold() and production_stopped:   # Live.restore: no owned HOLD while production is down
            self.events.append('restore')
            raise ValueError('PP4 production may be stopped without owned HOLD')
        return super().restore(production_stopped=production_stopped)


@pytest.mark.parametrize('mode', list(m.MODES))
def test_v2_window_order_greedy_and_decision(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('WINDOW_CHAIN', raising=False)
    fake = FakeV2()
    arms = w.V2_ARMS[mode]
    result = w.window(fake, lambda row: None, selected=arms, measured_cells=m.CELLS, v2=mode)
    assert result['status'] == 'COMPLETE' and record(result['decision'])['status'] == 'CANDIDATE'
    assert 'decision_series' in result and 'decision_series' not in record(result['decision'])
    assert recompute(result['decision_series'])['status'] == 'CANDIDATE'
    assert [x for x in fake.events if x in [a.name for a in arms]] == [a.name for a in arms]
    assert [x for x in fake.events if x.startswith('greedy-')] == [f'greedy-{n}' for n in m.GREEDY[mode]]
    assert fake.events[-1] == 'restore' and 'chain' not in result
    assert result['chain_state_at_start'] == 'production serving'
    if mode == 'production-policy':
        assert [a.config for a in arms] == [w.PRODUCTION_CONFIG, w.CONFIG, w.PRODUCTION_CONFIG]


def test_v2_window_refuses_unregistered_cells_or_arms() -> None:
    fake = FakeV2()
    result = w.window(fake, lambda row: None, selected=w.V2_ARMS['controlled'], measured_cells=budget.B1_CELLS + ('32k-b4',),
                      v2='controlled')
    assert fake.events == [] and record(result['decision'])['status'] == 'REJECT'
    result = w.window(FakeV2(), lambda row: None, selected=w.ARMS, measured_cells=m.CELLS, v2='production-policy')
    assert record(result['decision'])['status'] == 'REJECT'


@pytest.mark.parametrize('env,fail,chain_fail,expect', [
    ({'WINDOW_CHAIN': '1'}, '', False, 'chained'),
    ({'WINDOW_CHAIN': '1', 'WINDOW_CHAIN_LAST': '1'}, '', False, 'restored'),
    ({'WINDOW_CHAIN': '1'}, 'cell', False, 'restored'),
    ({'WINDOW_CHAIN': '1'}, '', True, 'restored-after-chain-failure')])
def test_v2_chain_end(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], fail: str, chain_fail: bool, expect: str) -> None:
    for key in ('WINDOW_CHAIN', 'WINDOW_CHAIN_LAST'):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    fake = FakeV2(fail, chain_fail=chain_fail)
    result = w.window(fake, lambda row: None, selected=w.V2_ARMS['controlled'], measured_cells=m.CELLS, v2='controlled')
    if expect == 'chained':
        assert fake.events[-1] == 'chain-close' and 'restore' not in fake.events
        assert result['chain'] == 'CHAINED' and record(result['restore'])['status'] == w.CHAINED
        assert record(result['decision'])['status'] == 'CANDIDATE'
    else:
        assert fake.events[-1] == 'restore' and result.get('chain') != 'CHAINED'
        assert ('chain-close' in fake.events) == (expect == 'restored-after-chain-failure')
        if expect == 'restored-after-chain-failure':
            assert 'quiet proof failed' in str(result['chain_close_error'])
            assert record(result['decision'])['status'] == 'CANDIDATE'   # measurements stand; production restored
        if fail:
            assert record(result['decision'])['status'] == 'REJECT'


def test_admission_and_pair_only_never_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    fake = FakeV2()
    result = w.window(fake, lambda row: None, selected=())
    assert fake.events[-1] == 'restore' and 'chain-close' not in fake.events and result['status'] == 'PAIR_QUALIFIED'


def test_chained_start_failure_restores_with_production_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    fake = FakeV2('hold', down=True)
    result = w.window(fake, lambda row: None, selected=w.V2_ARMS['controlled'], measured_cells=m.CELLS, v2='controlled')
    assert fake.restored_with is True and result['chain_state_at_start'] == 'chained: production down'
    assert record(result['decision'])['status'] == 'REJECT'


# ---- runner integration: real Live methods over fake transports


def live(tmp_path: Path, plain: object | None = None, clock: base.Clock | None = None) -> w.Live:
    out = tmp_path / 'out'
    out.mkdir(exist_ok=True)
    sup = pf.Supervisor(out, clock=clock or base.Clock())
    return w.Live(out, sup, native=prior.WORKER, remote='experiment', expected=prior.WORKER.commit, data_dir=tmp_path,
                  tag='tag', admission={'verifier_sha256': {}}, plain=plain, clock=clock or base.Clock())  # type: ignore[arg-type]


@pytest.mark.parametrize('fail', ['', 'jit', 'serving', 'stash', 'commit-close', 'placeholder-up', 'placeholder-proof'])
def test_live_chain_close_order_and_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: str) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    monkeypatch.delenv('WINDOW_CHAIN_LAST', raising=False)
    monkeypatch.setattr(chain, 'MARKER', tmp_path / 'CHAINED')
    events: list[str] = []
    def plain(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        events.append(label)
        if label.startswith('owned-fence-'):
            return 0, owner.FENCED
        if label.startswith(('owned-absent-', 'owned-reap-')):
            return 0, owner.ABSENT
        if label == 'pair-close':
            return 0, 'PRODUCTION HOLD OK'
        if label.startswith('jit-wait-'):
            return (1, '') if fail == 'jit' else (0, 'PP4 EXPERIMENT ABSENT')
        if label == 'chain-production-down':
            return 0, 'SERVING' if fail == 'serving' else 'DOWN'
        if label.startswith('chain-placeholder-up-'):
            assert argv[-1] == w.PLACEHOLDER_UP
            return (1, '') if fail == 'placeholder-up' else (0, '')
        if label.startswith('chain-placeholder-proof-'):
            return 0, 'PLACEHOLDER=0' if fail == 'placeholder-proof' else 'PLACEHOLDER=1'
        if label == 'chain-hold-stash':
            assert '-restore-close' in argv[-1] and 'rm -f' in argv[-1]
            return (1, '') if fail == 'stash' else (0, '')
        if label == 'commit-close':
            return (1, '') if fail == label else (0, owner.CLOSED)
        if label == 'cancel-close':
            return 0, 'SCOPED CLOSE COMMIT FENCED'
        if label == 'retain-close':
            return 0, owner.RETAINED
        if label == 'experiment-stop':
            return 0, ''
        raise AssertionError(label)
    runner = live(tmp_path, plain)
    runner.hold_written = True
    runner.owner_tags = ['tag-hold', 'tag-s1-A-worker', 'tag-s1-A-engine']
    saved = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        if fail:
            with pytest.raises(ValueError):
                runner.chain_close()
            assert not chain.MARKER.exists()
            assert ('retain-close' in events) == (fail == 'commit-close')
            assert 'start' not in events and 'check' not in events
        else:
            result = runner.chain_close()
            assert result['status'] == w.CHAINED and result['restore_deadline_met'] is True
            assert events.index('pair-close') < events.index('experiment-stop') < events.index('jit-wait-sparkhead')
            assert events.index('chain-production-down') < events.index('chain-hold-stash') < events.index('commit-close')
            # Default-socket placeholders go up before the first fence takes PP4's scoped placeholder down.
            assert events.index('chain-placeholder-up-sparkworker-wired') < min(i for i, e in enumerate(events) if e.startswith('owned-fence-'))
            assert events.index('chain-placeholder-proof-sparkworker-wired') < events.index('chain-hold-stash')
            assert events[-1] == 'cancel-close' and not any(e in events for e in ('start', 'check', 'close'))
            marker = chain.read_marker()
            assert marker is not None and record(marker['quiet_proof'])['production'] == 'DOWN'
            assert marker['placeholders'] == chain.PLACEHOLDERS_KEPT
    finally:
        signal.signal(signal.SIGTERM, saved)
        runner.sup.close()


@pytest.mark.parametrize('state', ['production', 'chained', 'marker-without-chain', 'down-without-marker', 'not-quiet',
                                   'no-placeholders', 'snapshot-changed'])
def test_live_production_preflight_uses_the_chain_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    monkeypatch.setattr(chain, 'MARKER', tmp_path / 'CHAINED')
    previous = tmp_path / 'PREV-W'
    previous.mkdir()
    ranks = [base.prod_identity(0), base.prod_identity(1)]
    (previous / 'rollback.json').write_text(json.dumps({'production_commit': w.PROD_COMMIT, 'tag': w.PROD_TAG, 'ranks': ranks}))
    if state != 'down-without-marker' and state != 'production':
        chain.MARKER.write_text(json.dumps({'schema': chain.SCHEMA, 'window': str(previous),
                                            'production_snapshot': chain.snapshot_ref(previous / 'rollback.json')}))
    if state == 'snapshot-changed':
        (previous / 'rollback.json').write_text(json.dumps({'production_commit': w.PROD_COMMIT, 'tag': w.PROD_TAG,
                                                            'ranks': ranks, 'edited': True}))
    if state == 'marker-without-chain':
        monkeypatch.delenv('WINDOW_CHAIN', raising=False)
    else:
        monkeypatch.setenv('WINDOW_CHAIN', '1')
    serving = state == 'production'
    calls: list[str] = []
    def ssh(label: str, host: str, command: str, stdin: str | None = None, timeout: float = 120, inflight: int = 0) -> str:
        calls.append(label)
        if label == 'chain-state':
            assert command == w.CHAIN_PROBE
            return f'HOLD=0\nSERVING={int(serving)}\n'
        if label.startswith('chain-quiet-'):
            return '' if state == 'not-quiet' else 'PP4 CHAIN QUIET\n'
        if label.startswith('production-clock-'):
            assert '/etc/environment' in command
            return 'GPU_CLOCK_CAP=2300\n'
        if label.startswith('chain-placeholders-'):
            assert chain.PLACEHOLDER_PROBE in command
            return 'PLACEHOLDER=0\n' if state == 'no-placeholders' and host == 'sparkworker-wired' else 'PLACEHOLDER=1\n'
        return ''
    def as_json(label: str, host: str, command: str, stdin: str | None = None) -> dict[str, Json]:
        calls.append(label)
        return {'data': [{'max_model_len': 327680, 'root': '/srv/dsv41/legal-v2a'}]}
    runner = live(tmp_path)
    monkeypatch.setattr(runner, 'ssh', ssh)
    monkeypatch.setattr(runner, 'json', as_json)
    try:
        if state in ('marker-without-chain', 'down-without-marker', 'not-quiet', 'no-placeholders', 'snapshot-changed'):
            with pytest.raises(ValueError):
                runner.production_preflight()
        else:
            runner.production_preflight()
            assert runner.production_down() is (state == 'chained')
            assert runner.production_clock is not None and runner.production_clock['mhz'] == 2300
            if state == 'production':
                assert 'prod-models' in calls and runner.carried is None
                assert not any(c.startswith('chain-placeholders-') for c in calls)   # probed only with a marker
            else:
                assert 'prod-models' not in calls and calls[-2:] == ['chain-quiet-sparkhead', 'chain-quiet-sparkworker-wired']
                assert ['chain-placeholders-sparkhead', 'chain-placeholders-sparkworker-wired'] == [c for c in calls if 'placeholders' in c]
                runner.snapshot()
                saved = json.loads((runner.out / 'rollback.json').read_text())
                assert saved['carried_from'] == str(previous / 'rollback.json') and saved['ranks'] == ranks
                assert [r['id'] for r in runner.production] == ['prod0', 'prod1'] and runner.seen == {'prod0', 'prod1'}
    finally:
        runner.sup.close()


def test_chained_production_identity_checks(tmp_path: Path) -> None:
    good = {'production_commit': w.PROD_COMMIT[:7], 'tag': w.PROD_TAG, 'ranks': [base.prod_identity(0), base.prod_identity(1)]}
    path = tmp_path / 'rollback.json'
    path.write_text(json.dumps(good))
    assert w.chained_production(path)['tag'] == w.PROD_TAG
    for change in ({'tag': 'prod-k1-327-20261023'}, {'production_commit': '21bed31'},
                   {'ranks': [base.prod_identity(0)]}, {'ranks': [{'id': 'x'}, {'id': 'y'}]}):
        path.write_text(json.dumps({**good, **change}))
        with pytest.raises(ValueError):
            w.chained_production(path)


@pytest.mark.skipif(not REAL_ROLLBACK.is_file(), reason='real PP4 window rollback.json absent')
def test_real_pp4_window_rollback_is_a_valid_carried_identity() -> None:
    data = w.chained_production(REAL_ROLLBACK)
    assert data['production_commit'] == w.PROD_COMMIT and len(cast(list[Json], data['ranks'])) == 2


def test_runtime_production_policy_arm() -> None:
    prod = base.prod_identity()
    record(prod['env'])['VLLM_DSV41_ADAPTIVE_K'] = '1'
    spec = '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,2,5],[3,4,4]]}'
    prod['argv'] = str(prod['argv']) + ' --speculative-config ' + shlex.quote(spec)
    overlay: dict[str, Json] = {pf.DIST + 'vllm/v1/worker/gpu/spec_decode/remote_dspark/client.py': 'e' * 64}
    arm = w.V2_ARMS['production-policy'][0]
    current = copy.deepcopy(prod)
    current['argv'] = str(current['argv']).replace('--port 8000', '--port 8001')
    current['env'] = {**record(prod['env']), 'VLLM_PP4_ROUND_DIR': '/pp4-round', 'VLLM_DSPARK_SAMPLED_TOP32': '0',
                      **dict(zip(w.IDENTITY_KEYS, (prior.WORKER.identity, prior.WORKER.markov)))}
    current['mounted'] = {**record(prod['mounted']), **overlay}
    w.runtime(prod, current, arm, prior.WORKER, overlay)
    fixed = copy.deepcopy(current)
    fixed['argv'] = str(fixed['argv']).replace(shlex.quote(spec), shlex.quote(w.SPEC))
    with pytest.raises(ValueError, match='production speculative config'):
        w.runtime(prod, fixed, arm, prior.WORKER, overlay)
    policy = copy.deepcopy(current)
    record(policy['env']).update(w.COMMON_ENV)
    with pytest.raises(ValueError, match='environment drift'):
        w.runtime(prod, policy, arm, prior.WORKER, overlay)
    block = '{"method":"dspark","num_speculative_tokens":5,"rejection_sample_method":"block"}'
    for target in (prod, current):
        target['argv'] = str(target['argv']).replace(shlex.quote(spec), shlex.quote(block))
    with pytest.raises(ValueError, match='standard verification'):
        w.runtime(prod, current, arm, prior.WORKER, overlay)
    record(prod['env'])['VLLM_DSV41_ADAPTIVE_K'] = '0'
    with pytest.raises(ValueError, match='not adaptive'):
        w.runtime(prod, current, arm, prior.WORKER, overlay)


def render(tmp_path: Path, config: str, role: str, production_env: bytes) -> tuple[int, str, str]:
    home = tmp_path / 'home'
    (home / 'dsv41-deploy').mkdir(parents=True, exist_ok=True)
    (home / 'dsv41-deploy' / 'dsv41.env').write_bytes(production_env)
    for name in ('vllm/v1/worker/gpu/model_runner.py', 'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py',
                 'vllm/v1/worker/gpu/dsv41_pp4_round_trace.py'):
        file = tmp_path / prereq.OVERLAY_REL / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    (tmp_path / 'offload/launch/windows').mkdir(parents=True, exist_ok=True)
    for name in (w.CONFIG, w.PRODUCTION_CONFIG):
        (tmp_path / name).write_bytes((pf.ROOT / name).read_bytes())
    command = ('set -eu; source ' + shlex.quote(config) + '; printf "%s\\n%s\\n%s\\n%s\\n%s\\n" "$EXP_DOCKER_ENV" '
               '"$EXTRA_DOCKER_ARGS" "$OVERLAYS" "${SPEC_OVERRIDE-UNSET}" "$LABEL"')
    proc = subprocess.run(['bash', '-c', command], cwd=tmp_path, capture_output=True, text=True, check=False,
                          env={**os.environ, 'HOME': str(home), 'PP4_ROLE': role, 'PP4_NATIVE_IDENTITY': prior.WORKER.identity,
                               'PP4_NATIVE_MARKOV_IDENTITY': prior.WORKER.markov, 'PP4_TRACE_RUN': 'test-trace'})
    return proc.returncode, proc.stdout, proc.stderr


def test_production_policy_config_differs_from_greedy_a_only_by_policy(tmp_path: Path) -> None:
    production_env = subprocess.check_output(['git', 'show', w.PROD_COMMIT + ':dsv41.env'], cwd=pf.ROOT)
    rc, a, err = render(tmp_path, w.CONFIG, 'A', production_env)
    assert rc == 0, err
    rc, p, err = render(tmp_path, w.PRODUCTION_CONFIG, 'P', production_env)
    assert rc == 0, err
    a_lines, p_lines = a.splitlines(), p.splitlines()
    policy = {'VLLM_DSV41_ADAPTIVE_K=0', 'VLLM_DSPARK_DEPTH_POLICY=fixed:5', 'VLLM_DSPARK_DEPTH_FILE=',
              'VLLM_DSPARK_LOOKUP=', 'VLLM_DSPARK_REMOTE_WANT_MARGIN=0'}
    assert p_lines[0].split() == [t for t in a_lines[0].split() if t not in policy]
    assert policy <= set(a_lines[0].split()) and 'VLLM_DSPARK_SAMPLED_TOP32=0' in p_lines[0].split()
    assert p_lines[1:3] == a_lines[1:3]                                       # same mounts and overlays
    assert a_lines[3] == w.SPEC and p_lines[3] == 'UNSET' and p_lines[4] == 'PP4-P-production-policy'
    rc, _, err = render(tmp_path, w.PRODUCTION_CONFIG, 'A', production_env)
    assert rc != 0 and 'role P required' in err
    seeded = production_env.replace(b'OFFLOAD_DSPARK_ENV="', b'OFFLOAD_DSPARK_ENV="VLLM_DSPARK_LOOKUP=x ', 1)
    rc, _, err = render(tmp_path, w.PRODUCTION_CONFIG, 'P', seeded)
    assert rc != 0 and 'VLLM_DSPARK_LOOKUP' in err


def test_main_v2_refuses_before_any_remote_action(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                                                  tmp_path: Path) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError('remote/process action forbidden')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    assert w.main(['--measure-v2', 'controlled']) == 0
    assert 'PP4 v2 controlled' in capsys.readouterr().out
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    common = ['--execute', '--qualification', str(tmp_path / 'q.json'), '--expected-experiment-commit', '2' * 40]
    assert w.main([*common, '--measure-v2', 'controlled']) == 1
    assert '--rng-gate' in capsys.readouterr().err
    assert w.main([*common, '--rng-gate', str(tmp_path / 'g.json')]) == 1
    assert 'belong to --measure-v2' in capsys.readouterr().err
    assert w.main([*common, '--measure-v2', 'controlled', '--admission', '--rng-gate', str(tmp_path / 'g.json')]) != 0


def test_v2_greedy_uses_the_round15_near_tie_path_and_cells_use_v2(tmp_path: Path) -> None:
    """P and B greedy checks run pp4_cells' greedy (R15: first differing token in the common prefix; only a pure prefix
    refuses); only the measured cells switch to the v2 driver."""
    runner = live(tmp_path)
    try:
        runner.v2 = 'production-policy'
        assert runner.driver('greedy', tmp_path / 'g.json')[2] == 'offload.launch.windows.pp4_cells'
        assert runner.driver('cell', tmp_path / 'c.json')[2] == 'offload.launch.windows.pp4_measure_v2'
        assert runner.cell_timeout('32k-b1') == 35 * m.WAVES * 3 + 60
        runner.v2 = None
        assert runner.driver('cell', tmp_path / 'c.json')[2] == 'offload.launch.windows.pp4_cells'
    finally:
        runner.sup.close()
    import inspect
    assert 'len(ta) == len(tb)' not in inspect.getsource(cells.near_tie)


def test_production_cap_and_same_clock_guard() -> None:
    assert m.production_cap(['GPU_CLOCK_CAP=2300\n', 'GPU_CLOCK_CAP=2300\n']) == 2300
    assert m.production_cap(['', '']) == m.DEFAULT_CAP_MHZ == 2150
    assert m.production_cap(['GPU_CLOCK_CAP=\n', 'GPU_CLOCK_CAP=\n']) is None
    with pytest.raises(ValueError, match='disagree'):
        m.production_cap(['GPU_CLOCK_CAP=2300\n', ''])
    env = subprocess.check_output(['git', 'show', w.PROD_COMMIT + ':dsv41.env'], cwd=pf.ROOT, text=True)
    assert 'GPU_CLOCK_CAP=${GPU_CLOCK_CAP-2150}' in env
    names = ['s1-P', 's2-B', 's3-P']
    at: dict[str, Json] = {n: {'spark': 2300, 'sparkworker': 2300} for n in names}
    assert m.clock_guard(at, names, [], {'mhz': 2300})['consistent'] is True
    off = m.clock_guard(at, names, [], {'mhz': 2150})
    assert off['consistent'] is False and 'not production' in str(off['reason'])
    assert m.clock_guard({n: {} for n in names}, names, [], {'mhz': None})['consistent'] is True
    assert m.clock_guard({n: {} for n in names}, names, [], {'mhz': 2300})['consistent'] is False


def test_archive_spread_uses_only_same_commit_same_clock_rows(tmp_path: Path) -> None:
    archive = tmp_path / 'archive.jsonl'
    rows = starts('controlled', candidate=WIN)
    for window, mhz, scale in (('A-W-20261027T1', 2300, 1.0), ('B-W-20261027T2', 2300, 1.0), ('C-W-20261027T3', 2150, 1.3)):
        clock: dict[str, Json] = {'per_start': {n: {'spark': mhz, 'sparkworker': mhz} for n in ('s1-A', 's2-B', 's3-A')}}
        for start in ('s1-A', 's3-A'):
            record(rows[start])['cells'] = {c: v2_cell(c, start, seconds=4. * scale) for c in m.CELLS}
        assert w.append_archive({'starts': rows}, 'controlled', window_name=window, experiment_commit='2' * 40,
                                clock=clock, path=archive)['status'] == 'APPENDED'
    clock_now: dict[str, Json] = {**CLOCK_OK, 'per_start': {n: {'spark': 2300, 'sparkworker': 2300} for n in ('s1-A', 's2-B', 's3-A')}}
    result = m.decide(starts('controlled', candidate=WIN), 'controlled', clock=clock_now, rng=GATE, archive=archive,
                      production_commit=w.PROD_COMMIT)
    for cell in m.CELLS:
        assert record(record(result['archive'])[cell])['archive_day_to_day_spread'] == 0.0   # the 2150 MHz window is excluded


@pytest.mark.parametrize('case', ['chained-start-preflight-failure', 'chained-start-hold-never-written', 'no-marker'])
def test_v2_chain_abort_when_production_was_left_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    """WIN2-R2 FIX 1: a chained window that stops without owning a HOLD runs the shared chain abort, never NOT NEEDED."""
    monkeypatch.setattr(chain, 'MARKER', tmp_path / 'CHAINED')
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    monkeypatch.delenv('WINDOW_CHAIN_LAST', raising=False)
    if case != 'no-marker':
        chain.MARKER.write_text(json.dumps({'schema': chain.SCHEMA, 'window': str(tmp_path)}))
    fake = FakeV2('preflight' if case != 'chained-start-hold-never-written' else 'hold',
                  down=case == 'chained-start-hold-never-written')
    result = w.window(fake, lambda row: None, selected=w.V2_ARMS['controlled'], measured_cells=m.CELLS, v2='controlled')
    if case == 'no-marker':
        assert result['end_action'] == 'none' and 'chain-abort' not in fake.events
        assert record(result['restore'])['status'] == 'NOT NEEDED: preflight only'
    else:
        assert fake.events[-1] == 'chain-abort' and result['chain'] == 'ABORT'
        assert str(record(result['restore'])['status']).startswith('RESTORED: chain ended by abort')
        assert record(result['decision'])['status'] == 'REJECT'
        if case == 'chained-start-hold-never-written':
            assert 'own_restore' in result and fake.restored_with is True
