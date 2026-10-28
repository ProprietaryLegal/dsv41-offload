"""CPU-only PP4 rehearsals. All serving, host, unit and GPU operations are fakes."""
from __future__ import annotations

import ast
import copy
import importlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pp4_cells as cells
from offload.launch.windows import pp4_metrics as metric
from offload.launch.windows import pp4_prerequisites as prereq
from offload.launch.windows import pp4_window as w
from offload.launch.windows import scoped_remote as owner
from offload.launch.windows.tests import test_pf1_window as base
from offload.prefill_maxlen.evidence import Json, record

WORKER = w.Worker('/opt/dsv41/x86host-data/pp4/source/repo', 'a' * 40, 'b' * 64, 'c' * 64)


def block_cell(name: str, sampled: bool = False, start: str = 'fixture') -> dict[str, Json]:
    batch = cells.CELLS[name].batch
    waves: list[Json] = [{'wave': wave, 'family': cells.FAMILIES[wave % 2], 'batch': batch,
        'spec_request_rounds': 1000.0, 'accepted_tokens': 1200.0 if sampled else 1000.0,
        'server_decode_seconds': 20.0, 'median_ttft_s': 1.0, 'completed_requests': batch,
        'streams': [{'prompt_sha256': name+str(wave)+'-'+str(i),
            'cache_salt_sha256': start+'-'+name+str(wave)+'-'+str(i),
            'sampling': {'temperature': 1, 'top_p': 1, 'seed': 42+wave, 'priority': 10, 'max_tokens': 512, 'ignore_eos': True},
            'usage': {'completion_tokens': 512}, 'finish': 'length'} for i in range(batch)]}
        for wave in range(metric.group_count(name))]
    return {'cell': name, 'batch': batch, 'metric_registration': metric.REGISTRATION, 'waves': waves,
            'median_decode_tps': 110.0 if sampled else 100.0,
            'median_ttft_s': 1.0, 'p95_round_ms': 50.0,
            'complete_round_profile': {'by_family': {family: {'p95_round_ms': 50.0} for family in cells.FAMILIES}},
            'streams': [s for row in waves for s in cast(list[Json], record(row)['streams'])]}



def complete() -> dict[str, Json]:
    return {a.name: {'role': a.role, 'status': 'COMPLETE', 'greedy': {'identical': 6, 'near_tie': 0, 'checks': [{'i': i, 'identical': True} for i in range(6)]},
                    'cells': {name: block_cell(name, a.role == 'B', a.name) for name in cells.CELLS}} for a in w.ARMS}


def test_preregistered_rule_and_missing_ttft() -> None:
    rows = complete()
    result = w.decision(rows)
    assert result['status'] == 'CANDIDATE FOR REVIEW' and result['source_families'] == 2
    assert result['resamples'] == 2000 and result['bootstrap_seed'] == 4103
    record(record(record(rows[w.ARMS[1].name])['cells'])['32k-b4'])['median_ttft_s'] = None
    assert w.decision(rows)['status'] == 'INCONCLUSIVE'


@pytest.mark.parametrize('change', ['incomplete', 'different-input', 'ttft', 'server-cost', 'greedy', 'family-loss', 'low-gain'])
def test_rule_rejects_bad_evidence(change: str) -> None:
    rows = complete()
    if change == 'incomplete':
        rows.pop(w.ARMS[-1].name)
    for arm in (w.ARMS[1],):
        block = record(rows[arm.name])
        cell = record(record(block['cells'])['short-b1'])
        if change == 'different-input':
            record(cast(list[Json], record(cast(list[Json], cell['waves'])[0])['streams'])[0])['prompt_sha256'] = 'different'
        elif change == 'ttft': cell['median_ttft_s'] = 1.051
        elif change == 'server-cost':
            for wave in cast(list[Json], cell['waves']): record(wave)['server_decode_seconds'] = 21.01
        elif change == 'greedy': record(cast(list[Json], record(block['greedy'])['checks'])[5])['identical'] = False
        elif change == 'family-loss':
            for name in cells.CELLS:
                for wave in cast(list[Json], record(record(block['cells'])[name])['waves']):
                    if record(wave)['family'] == 'opinions':
                        record(wave)['accepted_tokens'] = 800.0
        elif change == 'low-gain':
            for name in cells.CELLS:
                for wave in cast(list[Json], record(record(block['cells'])[name])['waves']):
                    record(wave)['accepted_tokens'] = 1039.0
    if change in ('different-input', 'greedy'):
        with pytest.raises(ValueError): w.decision(rows)
    else:
        assert w.decision(rows)['status'] != 'CANDIDATE FOR REVIEW'


def test_runtime_declares_exact_common_differences() -> None:
    prod = base.prod_identity()
    prod['argv'] = str(prod['argv']) + " --speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"x\":1}'"
    overlay: dict[str, Json] = {pf.DIST + 'vllm/v1/worker/gpu/spec_decode/remote_dspark/client.py': 'e' * 64}
    for arm in w.ARMS:
        live = copy.deepcopy(prod)
        live['argv'] = str(live['argv']).replace('--port 8000', '--port 8001').replace(
            '{"method":"dspark","num_speculative_tokens":5,"x":1}', w.SPEC)
        live['env'] = {**record(prod['env']), **w.COMMON_ENV,
                       'VLLM_DSPARK_SAMPLED_TOP32': str(int(arm.role == 'B')),
                       **dict(zip(w.IDENTITY_KEYS, (WORKER.identity, WORKER.markov)))}
        live['mounted'] = {**record(prod['mounted']), **overlay}
        w.runtime(prod, live, arm, WORKER, overlay)
        record(live['env'])['NCCL_PROTO'] = 'drift'
        with pytest.raises(ValueError, match='environment drift'):
            w.runtime(prod, live, arm, WORKER, overlay)


class Fake:
    def __init__(self, fail: str = '') -> None:
        self.fail = fail
        self.events: list[str] = []

    def preflight(self) -> None:
        self.events.append('preflight')
        if self.fail == 'preflight': raise ValueError('missing qualification')
    def open_hold(self) -> float:
        self.events.append('hold')
        if self.fail == 'hold': raise ValueError('lost HOLD reply')
        return 1.0
    def snapshot(self) -> None: self.events.append('snapshot')
    def qualify_pair(self) -> dict[str, Json]:
        self.events.append('pair-qualification')
        if self.fail == 'pair-qualification': raise ValueError('physical peer refused')
        return {'fixture': True}
    def fits(self, arm: w.Arm) -> bool: return self.fail != 'time'
    def start(self, arm: w.Arm) -> None:
        self.events.append(arm.name)
        if self.fail == 'start': raise ValueError('lost pair-open reply')
    def build(self) -> None: self.events.append('freeze-inputs')
    def warm(self, arm: w.Arm) -> None: self.events.append('warm-' + arm.name)
    def greedy(self, arm: w.Arm) -> dict[str, Json]:
        self.events.append('greedy-' + arm.name)
        if self.fail == 'greedy': raise ValueError('greedy divergence')
        return {'identical': 6, 'near_tie': 0, 'checks': [{'i': i, 'identical': True} for i in range(6)]}
    def begin_measurement(self, arm: w.Arm) -> None: self.events.append('baseline-' + arm.name)
    def cell(self, arm: w.Arm, name: str) -> dict[str, Json]:
        self.events.append('cell-' + arm.name + '-' + name)
        if self.fail == 'cell': raise TimeoutError('CENSORED request')
        if self.fail == 'signal': raise KeyboardInterrupt('signal 15')
        return block_cell(name, arm.role == 'B', arm.name)
    def finish(self, arm: w.Arm) -> dict[str, Json]: return {'clean': True}
    # WIN2 chain/v2 hooks: production serving at start; a chained close is not expected outside WINDOW_CHAIN=1.
    def production_down(self) -> bool: return False
    def measurement_context(self) -> dict[str, Json]: return {'clock': {'consistent': True}, 'rng_gate': None}
    def chain_close(self) -> dict[str, Json]: raise AssertionError('chained close outside WINDOW_CHAIN=1')
    def chain_abort(self) -> dict[str, Json]: raise AssertionError('chain abort without a chain marker')
    def owns_hold(self) -> bool: return self.fail != 'hold'
    def restore(self, *, production_stopped: bool) -> dict[str, Json]:
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        self.events.append('restore')
        if self.fail == 'restore': raise ValueError('no absence proof')
        return {'status': 'RESTORED: K1 START VERIFIED, greedy 6/6 IDENTICAL, PRODUCTION HOLD OK, window closed',
                'restore_deadline_met': True, 'production_stopped': production_stopped}


def test_full_off_on_off_rehearsal() -> None:
    fake = Fake()
    result = w.window(fake, lambda row: None)
    assert result['status'] == 'COMPLETE' and record(result['decision'])['status'] == 'CANDIDATE FOR REVIEW'
    assert [x for x in fake.events if x in [a.name for a in w.ARMS]] == ['s1-A', 's2-B', 's3-A']
    for arm in (w.ARMS[1],):
        assert fake.events.index('greedy-' + arm.name) < next(i for i, x in enumerate(fake.events) if x.startswith('cell-' + arm.name))
    assert fake.events[-1] == 'restore'


@pytest.mark.parametrize('fail', ['hold', 'pair-qualification', 'start', 'time', 'greedy', 'cell', 'signal', 'restore'])
def test_every_failure_restores_and_cannot_promote(fail: str) -> None:
    fake = Fake(fail)
    result = w.window(fake, lambda row: None)
    assert fake.events[-1] == 'restore'
    assert record(result['decision'])['status'] == 'REJECT'
    assert str(record(result['restore'])['status']).startswith('FAILED' if fail == 'restore' else 'RESTORED:')


def test_no_hold_on_failed_local_gate() -> None:
    fake = Fake('preflight')
    result = w.window(fake, lambda row: None)
    assert fake.events == ['preflight'] and record(result['decision'])['status'] == 'REJECT'


@pytest.mark.parametrize('fail', ['', 'fence', 'reap', 'pair', 'start', 'greedy', 'deadline', 'jit', 'commit-close', 'cancel-close'])
def test_actual_live_restore_order(tmp_path: Path, fail: str) -> None:
    out = tmp_path / 'out'; out.mkdir()
    clock = base.Clock()
    sup = pf.Supervisor(out, clock=clock)
    events: list[str] = []
    live: w.Live
    def plain(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        events.append(label)
        assert timeout <= w.RESTORE_S
        if fail == 'deadline': clock.now += w.RESTORE_S + 1
        if label.startswith('owned-fence-'):
            return (1, '') if fail == 'fence' else (0, owner.FENCED)
        if label.startswith('owned-absent-'):
            return 0, owner.ABSENT
        if label.startswith('owned-reap-'):
            return (1, '') if fail == 'reap' else (0, owner.ABSENT)
        if label == 'pair-close': return (1, '') if fail == 'pair' else (0, 'PRODUCTION HOLD OK')
        if label.startswith('jit-wait-'): return (1, '') if fail == 'jit' else (0, 'PP4 EXPERIMENT ABSENT')
        if label == 'commit-close': return (1, '') if fail == label else (0, owner.CLOSED)
        if label == 'cancel-close': return (1, '') if fail == label else (0, 'SCOPED CLOSE COMMIT FENCED')
        if label == 'retain-close': return 0, owner.RETAINED
        if label == 'experiment-stop': return 0, ''
        text = base.RESTORE_OK[label]
        if fail == 'start' and label == 'start': text = 'invented success'
        if fail == 'greedy' and label == 'check': text = text.replace('5 IDENTICAL', '4 IDENTICAL')
        return 0, text
    live = w.Live(out, sup, native=WORKER, remote='experiment', expected=WORKER.commit, data_dir=tmp_path,
                  tag='tag', admission={'verifier_sha256': {}}, plain=plain, clock=clock)
    live.hold_written = True
    live.owner_tags = ['tag-hold', 'tag-s1-A-worker', 'tag-s1-A-engine']
    live.native_tags = ['tag-s1-A-worker']
    saved = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        if fail:
            with pytest.raises(ValueError): live.restore(production_stopped=True)
            if fail not in ('commit-close', 'cancel-close'):
                assert 'close' not in events
            else:
                assert 'retain-close' in events
        else:
            result = live.restore(production_stopped=True)
            assert str(result['status']).startswith('RESTORED')
            assert events.index('owned-reap-tag-s1-A-engine-sparkworker-wired') < events.index('pair-close') < events.index('start')
            assert [e for e in events if e in ('start', 'check', 'close')] == ['start', 'check', 'close']
            assert events[-2:] == ['commit-close', 'cancel-close']
            for phase in ('tag-hold', 'tag-s1-A-worker', 'tag-s1-A-engine', 'tag-pair-close-1',
                          'tag-restore-start-3', 'tag-restore-check-4', 'tag-restore-close'):
                for host in w.OWNER_HOSTS:
                    assert f'owned-fence-{phase}-{host}' in events
                    assert f'owned-reap-{phase}-{host}' in events
    finally:
        signal.signal(signal.SIGTERM, saved)
        sup.close()


def wait_file(path: Path) -> None:
    until = time.monotonic() + 5
    while not path.exists():
        assert time.monotonic() < until, f'owned fake never reached {path.name}'
        time.sleep(.01)


@pytest.mark.parametrize('pause', ['before-registration', 'pair-active'])
def test_exact_worker_late_pair_open_fenced(tmp_path: Path, pause: str) -> None:
    home = tmp_path / 'home'; home.mkdir()
    bin_dir = home / 'bin'; bin_dir.mkdir()
    (bin_dir / 'python3').symlink_to(sys.executable)
    (bin_dir / 'git').write_text('#!/bin/bash\nif [ "$1" = rev-parse ]; then echo ' + WORKER.commit + '; fi\n')
    (bin_dir / 'git').chmod(0o700)
    run = home / 'dsv41-offload'; run.mkdir()
    (run / 'prod_window.sh').write_text('#!/bin/bash\nset -eu\necho inactive > "$HOME/state"\n'
        'touch "$HOME/pair-active"\nwhile [ ! -e "$HOME/release" ]; do sleep .01; done\n'
        'touch "$HOME/PAIR_OK"\n')
    local_worker = w.Worker(str(home / 'source/repo'), WORKER.commit, WORKER.identity, WORKER.markov)
    worker_tree = Path(local_worker.tree) / 'offload/dspark/worker/run'; worker_tree.mkdir(parents=True)
    (worker_tree / 'gpu_locked.sh').write_text('#!/bin/sh\nexec sleep 30\n')
    (home / 'state').write_text('active')
    tag = 'pp4test-owned-cpu-pair'
    script = w.worker_script(local_worker, tag, True)
    launch = owner.command('launch', tag, ['bash', '-c', script], deadline=time.time()+30)
    if pause == 'before-registration':
        launch = 'touch "$HOME/pending"; while [ ! -e "$HOME/release" ]; do sleep .01; done\n' + launch
    env = {**os.environ, 'HOME': str(home), 'PATH': str(bin_dir) + ':' + os.environ['PATH'], 'CUDA_VISIBLE_DEVICES': ''}
    proc = subprocess.Popen(['bash', '-c', launch], env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_file(home / ('pending' if pause == 'before-registration' else 'pair-active'))
        for mode in ('fence', 'reap', 'absent'):
            result = subprocess.run(['bash', '-c', owner.command(mode, tag)], env=env, capture_output=True, text=True, timeout=30, check=False)
            assert result.returncode == 0, result.stderr
        (home / 'state').write_text('active')
        (home / 'PAIR_OK').unlink(missing_ok=True)
        (home / 'release').touch()
        proc.wait(timeout=5)
        assert (home / 'state').read_text() == 'active' and not (home / 'PAIR_OK').exists()
        retry = subprocess.run(['bash', '-c', owner.command('launch', tag, ['bash', '-c', script], deadline=time.time()+30)], env=env,
                               capture_output=True, text=True, timeout=5, check=False)
        assert retry.returncode == 75  # shared owner refuses persistent ABORT before launching
    finally:
        if proc.poll() is None: os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
        for mode in ('fence', 'reap', 'absent'):
            subprocess.run(['bash', '-c', owner.command(mode, tag)], env=env, capture_output=True, timeout=30, check=True)


def test_dry_and_missing_admission_make_no_calls(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def forbidden(*args: object, **kwargs: object) -> None: raise AssertionError('remote/process action forbidden')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    assert w.main([]) == 0
    assert capsys.readouterr().out.startswith('DRY RUN')
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    assert w.main(['--execute']) == 1
    assert 'qualification' in capsys.readouterr().err


def test_admission_requires_every_primary_evidence_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = tmp_path / 'evidence.json'; evidence.write_text('{}')
    data: dict[str, Json] = {'schema': 'pp4-live-admission-v1', 'verdict': 'QUALIFIED', 'source_commit': WORKER.commit,
        'depth': 5, 'worker_identity': WORKER.identity, 'markov_identity': WORKER.markov,
        'runtime_python': prereq.PYTHON, 'runtime_torch': '2.12.1+cu130',
        'runtime_artifact_sha256': {prereq.STOCK: 'd' * 64, prereq.FACC: 'e' * 64,
            'p2p/' + pf.digest(pf.ROOT / 'offload/dspark/worker/p2p/p2p.cu')[:16] + '/dspark_p2p.so': 'f' * 64},
        'gates': {g: {'pass': True, 'evidence_path': str(evidence), 'sha256': pf.digest(evidence)} for g in prereq.GATES},
        'verifier_sha256': {'vllm/v1/worker/gpu/model_runner.py': 'a' * 64,
                           'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py': 'b' * 64,
                           'vllm/v1/worker/gpu/dsv41_pp4_round_trace.py': 'c' * 64}}
    from offload.launch.windows import pp4_budget
    monkeypatch.setattr(pp4_budget, 'admitted', lambda path, expected: {'block_budget_s': {'s1-A': 100, 's2-B': 100, 's3-A': 100}})
    path = tmp_path / 'receipt.json'; path.write_text(json.dumps(data))
    assert prereq.qualification(path, WORKER.commit)['verdict'] == 'QUALIFIED'
    evidence.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='changed'): prereq.qualification(path, WORKER.commit)


def test_generated_scripts_compile_and_worker_is_production_k2() -> None:
    for sampled in (False, True):
        script = w.worker_script(WORKER, 'test-tag', sampled)
        subprocess.run(['bash', '-n', '-c', script], check=True)
        assert 'prod_window.sh pair COORD' in script and f'DSPARK_SAMPLED_TOP32={int(sampled)}' in script
        assert '--tp 2 --root /opt/dsv41/x86host-data/dspark --max-model-len 1048576' in script
        assert 'native-hybrid' not in script and 'DSPARK_WORKER_TP_EMULATE' in script
    ast.parse(w.native_probe_script(WORKER, 'test-tag'))
    subprocess.run(['bash', '-n', str(pf.ROOT / w.CONFIG)], check=True)


def test_decode_censor_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter([0., 121.])
    monkeypatch.setattr(time, 'monotonic', lambda: next(values))
    import threading
    first = threading.Event(); first.set()
    with pytest.raises(TimeoutError, match='CENSORED'):
        list(cells.decode_bound([b'a', b'b'], first))


def test_actual_config_arms_differ_only_sample_switch(tmp_path: Path) -> None:
    home = tmp_path / 'home'; prod = home / 'dsv41-deploy'; prod.mkdir(parents=True)
    # Exact production bytes: the former synthetic fixture invented a top-level
    # mount which production never imports (S52 FIX2).
    production_env = subprocess.check_output(['git', 'show', w.PROD_COMMIT + ':dsv41.env'], cwd=pf.ROOT)
    (prod / 'dsv41.env').write_bytes(production_env)
    for name in ('vllm/v1/worker/gpu/model_runner.py', 'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py', 'vllm/v1/worker/gpu/dsv41_pp4_round_trace.py'):
        file = tmp_path / prereq.OVERLAY_REL / name; file.parent.mkdir(parents=True, exist_ok=True); file.touch()
    config = pf.ROOT / w.CONFIG
    results = []
    for role in ('A', 'B'):
        command = 'set -eu; source ' + shlex.quote(str(config)) + '; printf "%s\\n%s\\n%s\\n%s\\n" "$EXP_DOCKER_ENV" "$EXTRA_DOCKER_ARGS" "$OVERLAYS" "$SPEC_OVERRIDE"'
        proc = subprocess.run(['bash', '-c', command], cwd=tmp_path,
            env={**os.environ, 'HOME': str(home), 'PP4_ROLE': role, 'PP4_NATIVE_IDENTITY': WORKER.identity,
                 'PP4_NATIVE_MARKOV_IDENTITY': WORKER.markov, 'PP4_TRACE_RUN': 'test-trace'}, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr
        results.append(proc.stdout)
    assert results[0].replace('VLLM_DSPARK_SAMPLED_TOP32=0', 'VLLM_DSPARK_SAMPLED_TOP32=1') == results[1]
    assert 'VLLM_ENGRAM_REMOTE_REQUIRED=0' in results[0] and 'VLLM_DSV41_ADAPTIVE_K=0' in results[0]
    nested = pf.DIST + 'vllm/v1/worker/gpu/spec_decode/remote_dspark'
    assert '/usr/local/lib/python3.12/dist-packages/remote_dspark' not in results[0]
    env_lines = results[0].splitlines()
    replacements = dict(pair.split('=', 1) for pair in env_lines[2].split())
    assert replacements['offload/dspark/spark/remote_dspark'] == nested
    mounts = shlex.split(env_lines[1])
    destinations = [mounts[i+1].split(':')[1] for i in range(0, len(mounts), 2)]
    assert nested not in destinations
    assert len(destinations) == len(set(destinations))
    # Preserve every other production mount, especially spec_decode/__init__.py.
    original = subprocess.check_output(['bash', '-c', '. "$HOME/dsv41-deploy/dsv41.env"; printf %s "$OFFLOAD_DOCKER_ARGS"'],
        env={**os.environ, 'HOME': str(home)}, text=True)
    originals = shlex.split(original)
    removed = {nested, pf.DIST+'vllm/v1/worker/gpu/model_runner.py',
               pf.DIST+'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py'}
    for i in range(0, len(originals), 2):
        mount = originals[i+1]
        if mount.split(':')[1] not in removed:
            assert mount in mounts
    assert pf.DIST+'vllm/v1/worker/gpu/spec_decode/__init__.py' in destinations
    assert len({*destinations, *replacements.values()}) == len(destinations)+len(replacements)
    pins = prereq.overlay_pins({'verifier_sha256': {}})
    assert pins and all(path.startswith(nested + '/') for path in pins)
    assert nested+'/speculator.py' in pins



def test_complete_round_summary_matches_both_ranks_and_actual_batch() -> None:
    ranks: dict[int, list[dict[str, Json]]] = {rank: [
        {'schema': 'pp4-complete-round-v1', 'cell': '32k-b4', 'rank': rank, 'round': i,
         'batch': 4, 'request_hash': 'same', 'failed': False, 'complete_wall_ms': float(i + rank + 1),
         'cuda_elapsed_ms': float(i + 1)} for i in range(20)] for rank in (0, 1)}
    result = w.round_summary(ranks, '32k-b4', 4)
    assert result['p95_round_ms'] == 20.0 and result['profiled_rounds'] == 20
    ranks[1].pop()
    with pytest.raises(ValueError, match='both ranks'): w.round_summary(ranks, '32k-b4', 4)
    ranks[0].pop(3); ranks[1].pop(3)
    with pytest.raises(ValueError, match='missing complete-round'): w.round_summary(ranks, '32k-b4', 4)


def test_profile_helper_completes_real_round_boundary_with_fake_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from offload.launch.windows import pp4_round_trace as trace
    class Event:
        def record(self) -> None: pass
        def synchronize(self) -> None: pass
        def elapsed_time(self, other: trace.Event) -> float: return 5.0
    monkeypatch.setenv('VLLM_PP4_ROUND_DIR', str(tmp_path))
    monkeypatch.setattr(importlib, 'import_module', lambda name: SimpleNamespace(cuda=SimpleNamespace(Event=lambda **kw: Event()), distributed=SimpleNamespace(get_rank=lambda: 0)))
    runner = SimpleNamespace(parallel_config=SimpleNamespace(rank=0), execute_model_state=object())
    scheduler = SimpleNamespace(num_scheduled_tokens={'id': 6}, scheduled_spec_decode_tokens={'id': [1] * 5})
    trace.begin(runner, scheduler, False)
    assert not (tmp_path / 'rank0.jsonl').exists(), 'disabled headline must not create CUDA events or traces'
    (tmp_path / 'ACTIVE').write_text('32k-b1')
    trace.begin(runner, scheduler, False)
    trace.finish(runner)
    row = json.loads((tmp_path / 'rank0.jsonl').read_text())
    assert row['cuda_elapsed_ms'] == 5 and row['batch'] == 1 and row['failed'] is False
    assert 'id' not in row and row['request_hash'] != 'id'


def test_round_overlay_preserves_probability_abi_and_wraps_both_methods() -> None:
    from offload.launch.windows.pp4_prepare import instrument
    text = 'REMOTE_DRAFT_PROBS_ABI = "pp3-dyadic24-standard-v1"\nclass Runner:\n    def execute_model(self): pass\n    def sample_tokens(self): pass\n'
    result = instrument(text)
    tree = ast.parse(result)
    assert 'REMOTE_DRAFT_PROBS_ABI = "pp3-dyadic24-standard-v1"' in result
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef))
    assert all(isinstance(f, ast.FunctionDef) and len(f.decorator_list) == 1 for f in cls.body)
    with pytest.raises(ValueError, match='anchors changed'): instrument('class Runner: pass\n')


@pytest.mark.parametrize('fail', [False, True])
def test_profile_collection_uses_complete_rounds_and_clears_markers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool) -> None:
    out = tmp_path / 'out'; out.mkdir()
    sup = pf.Supervisor(out)
    live = w.Live(out, sup, native=WORKER, remote='experiment', expected=WORKER.commit, data_dir=tmp_path,
                  tag='owned', admission={'verifier_sha256': {}})
    live.submit = time.time() + 3600; live.block_deadline = live.submit; sup.abort_at = live.submit
    events: list[str] = []
    monkeypatch.setattr(live, 'stats', lambda label: counter_stats())
    live.begin_measurement(w.ARMS[0])
    def call(label: str, argv: list[str], *, timeout: float = 120, inflight: int = 0) -> str:
        events.append(label)
        if '--output' not in argv:
            return owner.FENCED if label.startswith('owned-fence-') else owner.ABSENT
        if fail: raise TimeoutError('profile censored')
        Path(argv[argv.index('--output') + 1]).write_text('{}')
        return ''
    def ssh(label: str, host: str, command: str, stdin: str | None = None, *, timeout: float = 120) -> str:
        events.append(label)
        if '-round-trace-' in label:
            rank = pf.HOSTS.index(host)
            return ''.join(json.dumps({'schema': 'pp4-complete-round-v1', 'cell': 'opinions:short-b1',
                'rank': rank, 'round': i, 'batch': 1, 'request_hash': 'same', 'failed': False,
                'complete_wall_ms': 20.0, 'cuda_elapsed_ms': 18.0}) + '\n' for i in range(20))
        return ''
    def plain(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        events.append(label)
        return 0, owner.FENCED if label.startswith('owned-fence-') else owner.ABSENT
    monkeypatch.setattr(live, 'call', call); monkeypatch.setattr(live, 'ssh', ssh); monkeypatch.setattr(live, 'plain', plain)
    try:
        if fail:
            with pytest.raises(TimeoutError): live.profile_family(w.ARMS[0], 'short-b1', 'opinions', 0)
        else:
            result = live.profile_family(w.ARMS[0], 'short-b1', 'opinions', 0)
            assert result['p95_round_ms'] == 20.0
        assert [x for x in events if x.startswith(arm_prefix := w.ARMS[0].name + '-profile-off-opinions-')] == [arm_prefix+h for h in pf.HOSTS]
    finally:
        sup.close()


@pytest.mark.parametrize('rank', [0, 1])
def test_nested_package_mount_hash_drift_rejected_on_each_rank(rank: int) -> None:
    prod = base.prod_identity()
    prod['argv'] = str(prod['argv']) + " --speculative-config '{}'"
    overlay = prereq.overlay_pins({'verifier_sha256': {}})
    live = copy.deepcopy(prod)
    live['argv'] = str(live['argv']).replace('--port 8000', '--port 8001').replace("'{}'", shlex.quote(w.SPEC))
    live['env'] = {**record(prod['env']), **w.COMMON_ENV, 'VLLM_DSPARK_SAMPLED_TOP32': '0',
                   **dict(zip(w.IDENTITY_KEYS, (WORKER.identity, WORKER.markov)))}
    live['mounted'] = {**record(prod['mounted']), **overlay}
    w.runtime(prod, live, w.ARMS[0], WORKER, overlay)
    path = pf.DIST+'vllm/v1/worker/gpu/spec_decode/remote_dspark/speculator.py'
    record(live['mounted'])[path] = 'rank'+str(rank)+' drift'
    with pytest.raises(ValueError):
        w.runtime(prod, live, w.ARMS[0], WORKER, overlay)


def counter_stats(failed: int = 1, some: int = 2, ok: int = 100) -> dict[str, Json]:
    return {'counters': {'rounds_ok': ok, 'rounds_failed': failed, 'rounds_some_failed': some,
                         'fallback_internal_error': 1}, 'proxy_torch_loaded': False,
            'policy': 'capacity', 'async': True, 'async_proxy': {'mode': 'process'},
            'async_handoff': {'mlock': 'ok'}, 'async_watchdog': {'fires': 0, 'running': True}}


def test_baseline_after_warmup_and_greedy_before_headline() -> None:
    fake = Fake(); result = w.window(fake, lambda row: None)
    assert result['status'] == 'COMPLETE'
    for arm in w.ARMS:
        baseline = fake.events.index('baseline-'+arm.name)
        assert fake.events.index('warm-'+arm.name) < baseline
        if arm.role == 'B':
            assert fake.events.index('greedy-'+arm.name) < baseline
        assert baseline < next(i for i, event in enumerate(fake.events) if event.startswith('cell-'+arm.name))


@pytest.mark.parametrize('change', ['unchanged', 'headline', 'profile', 'reset', 'missing', 'late', 'stop-growth'])
def test_measurement_baseline_failure_growth_and_final_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              change: str) -> None:
    sup = pf.Supervisor(tmp_path)
    live = w.Live(tmp_path, sup, native=WORKER, remote='experiment', expected=WORKER.commit,
                  data_dir=tmp_path, tag='counter', admission={'verifier_sha256': {}})
    current = counter_stats()
    monkeypatch.setattr(live, 'stats', lambda label: copy.deepcopy(current))
    arm = w.ARMS[0]
    try:
        with pytest.raises(ValueError, match='baseline absent'):
            live.measured_stats(arm, 'before-baseline')
        live.begin_measurement(arm)
        with pytest.raises(ValueError, match='cannot be renewed'):
            live.begin_measurement(arm)
        cs = record(current['counters']); cs['rounds_ok'] = 110
        live.measured_stats(arm, 'headline-before')
        if change in ('headline', 'profile', 'late'): cs['rounds_failed'] = 2
        if change == 'reset': cs['rounds_ok'] = 1
        if change == 'missing': cs.pop('rounds_some_failed')
        if change == 'stop-growth': cs['fallback_internal_error'] = 2
        if change == 'unchanged':
            assert live.measured_stats(arm, 'headline-after')['counters'] == current['counters']
            assert live.measured_stats(arm, 'profile-after')['counters'] == current['counters']
            # Actual finish must accept the prior nonzero totals; later identity
            # work is faked only after the real final-counter gate.
            live.ids = ['id0', 'id1']; sup.arm_minima = {h: 1 for h in pf.HOSTS}
            monkeypatch.setattr(live, 'ssh', lambda label, host, command: live.ids[pf.HOSTS.index(host)])
            monkeypatch.setattr(live, 'json', lambda *args: {})
            monkeypatch.setattr(pf, 'logs', lambda *args: {})
            assert record(live.finish(arm)['proxy'])['counters'] == current['counters']
        elif change == 'late':
            with pytest.raises(ValueError, match='grew after measurement'):
                live.finish(arm)
        else:
            with pytest.raises(ValueError):
                live.measured_stats(arm, change+'-after')
    finally:
        sup.close()
