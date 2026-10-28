"""UP2-PROMOTE runner: every window path through a fake backend, plus the pure evidence/identity/regression gates."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import pf1_window as pf
from offload.prefill_maxlen.evidence import Json, record
from offload.promote.up2 import promote_window as w, switch
from offload.up3 import build as b, cells


class Fake:
    def __init__(self, fail: str = '', close_fail: int = 0, watch_fail: bool = False) -> None:
        self.fail, self.close_fail, self.watch_fail = fail, close_fail, watch_fail
        self.calls: list[str] = []

    def step(self, name: str) -> None:
        self.calls.append(name)
        if self.fail == name:
            raise ValueError('boom ' + name)

    def preflight(self) -> None: self.step('preflight')
    def open_hold(self) -> None: self.step('open_hold')

    def controls(self, label: str) -> dict[str, dict[str, Json]]:
        self.step('controls-' + label)
        return {name: cell(name) for name in w.DECISION_CELLS}

    def switch(self) -> None: self.step('switch')
    def gates(self) -> None: self.step('gates')
    def final(self) -> None: self.step('final')
    def accept(self) -> None: self.step('accept')

    def finish(self, passed: bool, touched: bool) -> None:
        self.calls.append(f'finish({passed},{touched})')
        if self.close_fail:
            self.close_fail -= 1
            raise ValueError('close failed')

    def post_close(self, passed: bool) -> None:
        self.calls.append(f'post_close({passed})')
        if self.watch_fail:
            raise ValueError('watch failed')

    def rollback_watch(self) -> None: self.calls.append('rollback_watch')


def cell(name: str, ttft: float = 1., ms: float = 40.) -> dict[str, Json]:
    """One UP3 B1 cell result with a 1 % repeat-to-repeat TTFT and ms/round spread (run-to-run variation)."""
    reps: list[Json] = []
    for k in range(cells.REPEATS[name]):
        wobble = 1 + .01 * (k % 2)
        reps.append({'repeat': k, 'streams': [{'ttft_s': ttft * wobble, 'decode_tps': 40., 'prompt_sha256': name}],
                     'spec_request_rounds': 1000, 'server_decode_ms': 1000 * ms * wobble, 'accepted_tokens': 3000})
    return {'cell': name, 'batch': 1, 'repetitions': reps}


def run(fake: Fake) -> dict[str, Json]:
    rows: list[dict[str, Json]] = []
    return w.window(fake, rows.append)


def test_pass_keeps_0075_and_watches_the_new_w8() -> None:
    fake = Fake()
    result = run(fake)
    assert str(result['verdict']).startswith('PASS') and result['close'] == 'PASS' and result['watch'] == 'PASS'
    assert fake.calls == ['preflight', 'open_hold', 'controls-control', 'switch', 'gates', 'controls-replay', 'final',
                          'accept', 'finish(True,True)', 'post_close(True)']
    assert record(result['G6'])['rule'] == 'win2-v1'


@pytest.mark.parametrize('step,touched', [('open_hold', False), ('controls-control', False), ('switch', True),
                                          ('gates', True), ('controls-replay', True), ('final', True), ('accept', True)])
def test_any_failure_restores_prior_production(step: str, touched: bool) -> None:
    fake = Fake(fail=step)
    result = run(fake)
    assert str(result['verdict']).startswith('FAIL') and 'boom' in str(result['stop'])
    assert f'finish(False,{touched})' in fake.calls and 'post_close(False)' in fake.calls


def test_preflight_failure_never_touches_hosts() -> None:
    fake = Fake(fail='preflight')
    result = run(fake)
    assert fake.calls == ['preflight'] and str(result['verdict']).startswith('FAIL') and 'close' not in result


def test_accepted_close_failure_rolls_back() -> None:
    fake = Fake(close_fail=1)
    result = run(fake)
    assert fake.calls[-3:] == ['finish(True,True)', 'finish(False,True)', 'post_close(False)']
    assert str(result['verdict']).startswith('FAIL') and 'rolled back' in str(result['stop'])


def test_double_close_failure_retains_hold() -> None:
    fake = Fake(close_fail=2)
    result = run(fake)
    assert 'HOLD retained' in str(result['rollback_stop']) and 'post_close(False)' not in fake.calls


def test_watch_failure_after_pass_restores_prior_production() -> None:
    fake = Fake(watch_fail=True)
    result = run(fake)
    assert fake.calls[-2:] == ['post_close(True)', 'rollback_watch'] and str(result['verdict']).startswith('FAIL')


def controls(ttft: float = 1., ms: float = 40.) -> dict[str, dict[str, Json]]:
    return {name: cell(name, ttft, ms) for name in w.DECISION_CELLS}


def test_g6_is_win2_v1_loss_not_a_win_requirement() -> None:
    same = w.regression(controls(), controls())
    assert same['rule'] == 'win2-v1' and same['status'] == 'INCONCLUSIVE'
    assert same['decision_cells'] == list(w.DECISION_CELLS)
    assert w.regression(controls(), controls(ttft=.9))['status'] == 'CANDIDATE'     # a 150K/32K TTFT win also passes
    with pytest.raises(ValueError, match='G6: a decision cell LOST'):
        w.regression(controls(), controls(ttft=1.05))                               # TTFT beyond its 1 % spread
    with pytest.raises(ValueError, match='G6: a decision cell LOST'):
        w.regression(controls(), controls(ms=42.))                                  # ms/round guard
    with pytest.raises(ValueError, match='control set incomplete'):
        w.regression(controls(), {'150k-b1': cell('150k-b1')})


def summary(**decision: Json) -> dict[str, Json]:
    replay = dict(b.verify('replay')['files'])
    return {'status': 'COMPLETE', 'restore': {'status': 'RESTORED: K1 START VERIFIED, greedy 6/6 IDENTICAL'},
            'decision': {'rule': 'win2-v1', 'status': 'INCONCLUSIVE', 'adopt': 'ADOPT CANDIDATE',
                         'registration': cells.REPLAY_REGISTRATION, 'replay_files': cast(Json, replay), **decision}}


def test_evidence_gate_requires_the_qualification_of_exactly_these_bytes() -> None:
    assert w.evidence_gate(summary())['package'] == w.PACKAGE_ID
    for bad in (summary(adopt='NOT A CANDIDATE'), summary(registration='nbw'), summary(rule='old'),
                summary(replay_files={'vllm/v1/core/sched/scheduler.py': '0' * 64}),
                {**summary(), 'status': 'INCOMPLETE'}, {**summary(), 'stop': 'x'},
                {**summary(), 'restore': {'status': 'FAILED'}}):
        with pytest.raises((ValueError, KeyError)):
            w.evidence_gate(cast(dict[str, Json], bad))


def test_new_references_require_three_identical_runs(tmp_path: Path) -> None:
    rows: list[dict[str, Json]] = [{'i': i, 'text': f't{i}'} for i in range(6)]
    paths = []
    for k in range(3):
        p = tmp_path / f'r{k}.jsonl'
        p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        paths.append(p)
    assert w.references_identical(paths) == rows
    paths[2].write_text(''.join(json.dumps({**r, 'text': str(r['text']) + ('x' if r['i'] == 3 else '')}) + '\n' for r in rows))
    with pytest.raises(ValueError, match='not reproducible'):
        w.references_identical(paths)


def identity(env: dict[str, str], mounted: dict[str, str], ident: str) -> dict[str, Json]:
    argv = "vllm serve /srv/dsv41/model --kv-cache-memory-bytes 6442450944 \\\n --port 8000"
    return {'id': ident, 'image': 'img', 'engine': 'e', 'files': {'a': 'b'}, 'engram_identity': {'x': 1},
            'env': cast(Json, dict(env)), 'mounted': cast(Json, dict(mounted)), 'argv': argv}


def test_identity_gate_allows_exactly_the_flag_and_the_package() -> None:
    remote = pf.DIST + switch.REMOTE + '/'
    before_env = {'VLLM_DSPARK_REMOTE': 'o1rdma://x', 'NCCL_X': '1'}
    before_mounted = {'/serve.sh': 's1', pf.DIST + 'vllm/models/x.py': 'k1', remote + 'round.py': 'old',
                      remote + 'wire.py': 'old'}
    after_mounted = {'/serve.sh': 's2', pf.DIST + 'vllm/models/x.py': 'k1',
                     **{(remote + p.removeprefix('remote_dspark/') if p.startswith('remote_dspark/') else pf.DIST + p): v
                        for p, v in w.PACKAGE_FILES.items()}}
    before = identity(before_env, before_mounted, 'c1')
    after = identity({**before_env, 'VLLM_DSV41_REPLAY_SKIP': '1'}, after_mounted, 'c2')
    w.identity_gate(before, after)
    for key, value in (('id', 'c1'), ('image', 'other'), ('env', {**before_env, 'VLLM_DSV41_REPLAY_SKIP': '0'}),
                       ('env', before_env), ('mounted', {**after_mounted, pf.DIST + 'vllm/models/x.py': 'drift'}),
                       ('mounted', {**after_mounted, remote + 'extra.py': 'x'}),
                       ('argv', str(after['argv']).replace('--port 8000', '--port 8001'))):
        bad = cast(dict[str, Json], {**after, key: value})
        with pytest.raises(ValueError):
            w.identity_gate(before, bad)


def test_live_refuses_pp4_and_chains_before_any_host_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / 'run'
    out.mkdir()
    for base in ('pp4', 'promo1-low'):          # S97 FIX 4: PROMO1-low too, until PROMO1 carries the 420be24 fold
        with pytest.raises(ValueError, match='only'):
            w.Live('T', out, '0' * 40, base, tmp_path / 'r.sh', '0' * 64, tmp_path / 's.json')
    with pytest.raises(SystemExit):
        w.main(['--execute', '--base', 'promo1-low'])
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    live = w.Live('T', out, '0' * 40, 'ops1', tmp_path / 'r.sh', '0' * 64, tmp_path / 's.json')
    with pytest.raises(ValueError, match='never runs chained'):
        live.preflight()
    assert not live.sup.procs and not (out / 'evidence.json').exists()


def test_package_constants_pin_the_new_w8_and_references() -> None:
    assert w.PACKAGE['arm'] == 'up3-replay' and w.PACKAGE_ID in w.NEW_REFERENCES.name
    assert f'ID={w.PACKAGE_ID}\n' in w.NEW_RESTORE.read_text()
    assert copy.deepcopy(w.PACKAGE_FILES) == switch.package('qualified')['files']


def test_readme_promotion_command_pins_the_full_w8_restore_sha() -> None:
    # S97 FIX 3: the documented command carried a 62-hex (truncated) sha and was refused before any action.
    import re
    readme = (w.HERE / 'README.md').read_text()
    pinned = re.findall(r'--rollback-restore-sha256 (\S+)', readme)
    assert pinned == [pf.RESTORE_SHA256] and re.fullmatch('[0-9a-f]{64}', pinned[0])


def test_budget_and_dry_plan_never_touch_hosts(capsys: pytest.CaptureFixture[str]) -> None:
    estimate = w.window_estimate()
    assert estimate['total_s'] <= w.RESTORE_S
    assert w.main([]) == 0
    text = capsys.readouterr().out
    assert w.PACKAGE_ID in text and 'PP4 refused' in text and 'ESTIMATED' in text
    assert w.main(['--execute']) == 2                   # required inputs absent: refused before any directory or host
