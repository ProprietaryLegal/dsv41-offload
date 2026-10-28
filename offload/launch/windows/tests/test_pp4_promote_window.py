"""PP4-PROMOTE runner: every window path through a fake backend, plus the pure evidence/identity/regression gates."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import time
from typing import cast

import pytest

from offload.launch.windows import decision_v2
from offload.launch.windows import pp4_promote_switch as switch
from offload.launch.windows import pp4_promote_window as w
from offload.launch.windows import pf1_window as pf
from offload.prefill_maxlen.evidence import Json, record


class Fake:
    def __init__(self, fail: str = '', close_fail: int = 0, watch_fail: bool = False, clock_fail: bool = False) -> None:
        self.fail, self.close_fail, self.watch_fail, self.clock_fail = fail, close_fail, watch_fail, clock_fail
        self.calls: list[str] = []

    def step(self, name: str) -> None:
        self.calls.append(name)
        if self.fail == name:
            raise ValueError('boom ' + name)

    def preflight(self) -> None: self.step('preflight')
    def open_hold(self) -> None: self.step('open_hold')

    def controls(self, label: str) -> dict[str, list[dict[str, Json]]]:
        self.step('controls-' + label)
        return {name: [{'ttft_s': 1.0, 'decode_tps': 50.0, 'prompt_sha256': name}] * w.REPEATS[name] for name in w.CONTROLS}

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

    def clock_cleanup(self) -> None:
        self.calls.append('clock_cleanup')
        if self.clock_fail:
            raise ValueError('sudo refused')


def run(fake: Fake) -> dict[str, Json]:
    rows: list[dict[str, Json]] = []
    return w.window(fake, rows.append)


def test_pass_keeps_pp4_and_watches_the_new_w8() -> None:
    fake = Fake()
    result = run(fake)
    assert str(result['verdict']).startswith('PASS') and result['close'] == 'PASS' and result['watch'] == 'PASS'
    assert fake.calls == ['preflight', 'open_hold', 'controls-control', 'switch', 'gates', 'controls-pp4', 'final',
                          'accept', 'finish(True,True)', 'post_close(True)', 'clock_cleanup']
    assert str(result['clock_cleanup']).startswith('PASS')


def test_clock_override_is_removed_only_after_a_passed_watch_and_its_failure_never_rolls_back() -> None:
    fake = Fake(clock_fail=True)
    result = run(fake)
    assert str(result['verdict']).startswith('PASS') and str(result['clock_cleanup']).startswith('FAILED')
    assert 'rollback_watch' not in fake.calls
    for failed in (Fake(fail='gates'), Fake(watch_fail=True), Fake(close_fail=1)):
        run(failed)
        assert 'clock_cleanup' not in failed.calls


@pytest.mark.parametrize('step,touched', [('open_hold', False), ('controls-control', False), ('switch', True),
                                          ('gates', True), ('controls-pp4', True), ('final', True), ('accept', True)])
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


def rows(ttft: list[float], tps: list[float], sha: str = 'p') -> list[dict[str, Json]]:
    return [{'ttft_s': a, 'decode_tps': b, 'prompt_sha256': sha} for a, b in zip(ttft, tps)]


def arms(tps: list[float], ttft: list[float] | None = None, long_tps: float = 30.0) -> dict[str, list[dict[str, Json]]]:
    """Every decision cell gets these three PP4 repeats; the reported 150K cell one repeat."""
    result = {name: rows(ttft or [1.0, 1.0, 1.0], tps) for name in w.DECISION_CELLS}
    result['ctl-150k-b1'] = rows([20.0], [long_tps])
    return result


CONTROL = arms([50.0, 50.5, 49.5])                 # 2 % own spread on decode tok/s, none on TTFT


def test_g6_is_win2_v1_decide_cell_loss_not_a_win_requirement() -> None:
    result = w.regression(CONTROL, arms([49.6, 49.4, 49.8]))          # worse, but inside the 2 % spread
    assert result['rule'] == decision_v2.RULE and result['status'] == 'INCONCLUSIVE'
    assert result['decision_cells'] == list(w.DECISION_CELLS)
    assert all(record(record(result['cells'])[name])['decides'] is True for name in w.DECISION_CELLS)
    assert w.regression(CONTROL, arms([52.0, 52.2, 51.9]))['status'] == 'CANDIDATE'          # a win also passes
    assert w.regression(CONTROL, arms([48.5, 50.4, 50.6]))['status'] == 'INCONCLUSIVE'       # 1 of 3 beyond spread


def test_g6_refuses_a_clear_loss_on_primary_or_guard() -> None:
    # Clear, consistent losses (every repeat ~6-10 % worse, interval far below zero) so the test holds under any
    # decision_v2 LOSS rule revision (S92 / WIN2-R2), not one threshold.
    with pytest.raises(ValueError, match='G6: PP4 LOSES'):
        w.regression(CONTROL, arms([47.0, 47.2, 47.1]))
    with pytest.raises(ValueError, match='G6: PP4 LOSES'):
        w.regression(CONTROL, arms([50.0, 50.0, 50.0], ttft=[1.10, 1.11, 1.10]))      # TTFT guard


def test_g6_reported_150k_never_decides_and_shapes_are_exact() -> None:
    result = w.regression(CONTROL, arms([50.0, 50.0, 50.0], long_tps=10.0))
    assert record(record(result['cells'])['ctl-150k-b1'])['verdict'] == 'REPORTED'
    short = {name: value[:2] for name, value in CONTROL.items()}
    with pytest.raises(ValueError, match='exactly 3 repeats'):
        w.regression(short, short)
    drift = copy.deepcopy(arms([50.0, 50.0, 50.0]))
    drift['ctl-32k-b1'][0]['prompt_sha256'] = 'other'
    with pytest.raises(ValueError, match='prompt drift'):
        w.regression(CONTROL, drift)
    with pytest.raises(ValueError, match='control set incomplete'):
        w.regression(CONTROL, {k: v for k, v in CONTROL.items() if k != 'ctl-150k-b1'})


def test_promotion_never_runs_chained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WINDOW_CHAIN', '1')
    out = tmp_path / 'run'
    out.mkdir()
    live = w.Live('T', out, '0' * 40, tmp_path / 'r.sh', '0' * 64, tmp_path / 'a.json', tmp_path / 'm.json')
    with pytest.raises(ValueError, match='never runs chained'):
        live.preflight()
    assert not live.sup.procs and not (out / 'evidence.json').exists()


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


def identity(spec: str, env: dict[str, str], mounted: dict[str, str], ident: str) -> dict[str, Json]:
    argv = f"vllm serve /srv/dsv41/model --speculative-config '{spec}' --kv-cache-memory-bytes 6442450944 \\\n --port 8000"
    return {'id': ident, 'image': 'img', 'engine': 'e', 'files': {'a': 'b'}, 'engram_identity': {'x': 1},
            'env': cast(Json, dict(env)), 'mounted': cast(Json, dict(mounted)), 'argv': argv}


def test_identity_gate_allows_exactly_the_registered_differences() -> None:
    prod_spec = '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,2,5],[3,4,4]]}'
    before_env = {'VLLM_DSV41_ADAPTIVE_K': '1', 'VLLM_DSPARK_REMOTE': 'o1rdma://x', 'NCCL_X': '1'}
    after_env = {k: v for k, v in before_env.items() if k not in switch.PP4_KEYS} | dict(switch.PP4_ENV)
    remote_old = {pf.DIST + switch.REMOTE + '/wire.py': 'old'}
    before_mounted = {'/serve.sh': 's1', pf.DIST + 'vllm/models/x.py': 'k1', **remote_old,
                      pf.DIST + switch.RUNNER: 'k1runner'}
    from offload.launch.windows import pp4_prerequisites as prereq
    after_mounted = {'/serve.sh': 's2', pf.DIST + 'vllm/models/x.py': 'k1',
                     **{k: str(v) for k, v in prereq.overlay_pins({'verifier_sha256': dict(switch.VERIFIER_SHA256)}).items()}}
    before = identity(prod_spec, before_env, before_mounted, 'c1')
    after = identity(switch.SPEC, after_env, after_mounted, 'c2')
    w.identity_gate(before, after)
    for key, value in (('id', 'c1'), ('image', 'other'), ('env', {**after_env, 'NCCL_X': '2'}),
                       ('mounted', {**after_mounted, pf.DIST + 'vllm/models/x.py': 'drift'}),
                       ('argv', str(after['argv']).replace('--port 8000', '--port 8001'))):
        bad = cast(dict[str, Json], {**after, key: value})
        with pytest.raises(ValueError):
            w.identity_gate(before, bad)


def test_worker_gate_requires_the_attested_sampled_unit() -> None:
    home = '/home/rank0user'
    release = f'{home}/dsv41-offload/prod/dspark-worker/{switch.PP4_COMMIT[:7]}/offload/dspark'
    runtime = f'{home}/dsv41-offload/prod/pp4-runtime/{switch.PP4_ID}'
    now = time.time()
    stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now))
    probe: dict[str, Json] = {'cmd': cast(Json, [switch.PYTHON, '-u', '-m', 'worker.server', '--tp', '2', '--root', '/opt/dsv41/x86host-data/dspark',
                                      '--ofl', '10.202.28.1:18620', '--max-model-len', '1048576', '']),
                              'cwd': release, 'sampled': '1', 'source': switch.PP4_COMMIT,
                              'pythonpath': f'{runtime}/stock:{runtime}/facc:{release}',
                              'status': {'tp': 2, 'tp_comm_errors': 0, 'time': stamp}}
    w.worker_gate(probe, now)
    for key, value in (('sampled', '0'), ('source', 'x'), ('cwd', '/elsewhere'), ('pythonpath', f'{release}'),
                       ('cmd', ['/home/operator/dsv41-offload/o2/venv/bin/python', '-m', 'worker.server'])):
        with pytest.raises((ValueError, IndexError)):
            w.worker_gate(cast(dict[str, Json], {**probe, key: value}), now)
    with pytest.raises(ValueError, match='stale'):
        w.worker_gate(probe, now + 60)


def test_budget_and_dry_plan_never_touch_hosts(capsys: pytest.CaptureFixture[str]) -> None:
    estimate = w.window_estimate()
    assert estimate['total_s'] <= w.RESTORE_S
    assert w.main([]) == 0
    assert 'DRY PLAN ONLY' in capsys.readouterr().out


def test_execute_refuses_missing_inputs_before_any_side_effect(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert w.main(['--execute', '--out', str(tmp_path / 'x')]) == 2
    assert not (tmp_path / 'x').exists()


@pytest.mark.parametrize('script', [w.x86host_apply_script(), w.x86host_rollback_script()])
def test_x86host_scripts_parse_and_pin_the_dropin(script: str) -> None:
    assert subprocess.run(['bash', '-n'], input=script, text=True).returncode == 0
    assert w.DROPIN_SHA256 in script and 'prod_window.sh check' in script.splitlines()[-1]


def etc_environment(tmp_path: Path, text: str, script: str) -> tuple[int, str, str]:
    """Run a generated /etc/environment script against a temporary copy (sudo is the only thing removed)."""
    path = tmp_path / 'environment'
    path.write_text(text)
    body = script.replace('sudo -n ', '').replace(w.ENVIRONMENT, str(path))
    done = subprocess.run(['bash', '-c', body], capture_output=True, text=True)
    return done.returncode, done.stdout, path.read_text()


BASE_ENV = 'PATH="/usr/local/sbin:/usr/bin"\n'


@pytest.mark.parametrize('before,after,state', [
    (BASE_ENV + 'GPU_CLOCK_CAP=2300\n', BASE_ENV, 'none: folded default 2300'),
    (BASE_ENV + 'GPU_CLOCK_CAP=2150\n', BASE_ENV + 'GPU_CLOCK_CAP=2150\n', 'GPU_CLOCK_CAP=2150'),   # dial-back stays
    (BASE_ENV, BASE_ENV, 'none: folded default 2300'),
])
def test_clock_cleanup_removes_only_the_exact_2300_override(tmp_path: Path, before: str, after: str, state: str) -> None:
    rc, out, text = etc_environment(tmp_path, before, w.clock_cleanup_script())
    assert rc == 0 and text == after and out.strip() == f'CLOCK OVERRIDE CLEARED {state}'


@pytest.mark.parametrize('before,after', [
    (BASE_ENV, BASE_ENV + 'GPU_CLOCK_CAP=2300\n'),
    (BASE_ENV + 'GPU_CLOCK_CAP=2300\n', BASE_ENV + 'GPU_CLOCK_CAP=2300\n'),
    (BASE_ENV + 'GPU_CLOCK_CAP=2150\n', BASE_ENV + 'GPU_CLOCK_CAP=2150\n'),                      # dial-back stays
])
def test_rollback_keeps_a_clock_override_before_the_prior_w8(tmp_path: Path, before: str, after: str) -> None:
    rc, out, text = etc_environment(tmp_path, before, w.clock_keep_script())
    assert rc == 0 and text == after and out.strip() == 'CLOCK OVERRIDE KEPT ' + after.splitlines()[-1]
