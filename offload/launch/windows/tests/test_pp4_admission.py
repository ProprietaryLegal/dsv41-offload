"""Qualification-only HOLD: each missing gate STOPs and uses actual window finally."""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pp4_admission as a
from offload.launch.windows import pp4_admission_requests as api
from offload.launch.windows import pp4_cells as cells
from offload.launch.windows import pp4_prerequisites as prereq
from offload.launch.windows import pp4_window as w
from offload.launch.windows.tests import test_pf1_window as base
from offload.launch.windows.tests import test_pp4_window as prior
from offload.prefill_maxlen.evidence import Json, number, record


class Fake(prior.Fake):
    def __init__(self, root: Path, fail: str = '') -> None:
        super().__init__()
        self.out = root
        self.expected = 'a'*40
        self.tag = 'fixture'
        self.failure = fail
        source = root/'seed.json'; source.write_text('{"fixture":"CPU only"}')
        self.admission: dict[str, Json] = {'source_commit': self.expected, 'gates': {
            name: {'pass': True, 'evidence_path': str(source.resolve()), 'sha256': pf.digest(source)} for name in a.SEED_GATES}}

    def qualify_pair(self) -> dict[str, Json]:
        from offload.dspark.probabilistic.pp4.pair_qualify import SOURCES
        from offload.launch.windows.tests.test_pp4_pair import receipt
        self.events.append('pair-qualification')
        physical = receipt()
        physical.update({'seconds': 120., 'source_sha256': {name: pf.digest(pf.ROOT/name) for name in SOURCES}})
        (self.out/'physical-pair-qualification.json').write_text(json.dumps(physical))
        return physical

    def stage(self, gate: str) -> dict[str, Json]:
        self.events.append(gate)
        if self.failure == gate: raise ValueError('CPU injected gate failure: '+gate)
        return {'complete': True, 'fixture': 'CPU ONLY, never live evidence'}

    def reconcile(self) -> dict[str, Json]: return self.stage('gb10_image_and_abi_reconciliation')
    def q_failures(self) -> dict[str, Json]: return self.stage('both_rank_q_failure_target_only')
    def api_gate(self, case: str) -> dict[str, Json]:
        return self.stage('grammar_penalty_stop_eos' if case == 'semantics' else 'tool_json_reasoning_parser')
    def lifecycle(self) -> dict[str, Json]: return self.stage('cancel_reset_reconnect_timeout_late_reply')
    def probes(self, during: object) -> dict[str, Json]:
        from collections.abc import Callable
        self.events.append('probes-B')
        cast(Callable[[], None], during)()
        return {'profiles': {name: {'by_family': {family: {
            'profiled_rounds': 0 if self.failure == 'complete_round_instrumentation' else 20,
            'p95_round_ms': 50.} for family in cells.FAMILIES}} for name in cells.CELLS}}
    def restore(self, *, production_stopped: bool) -> dict[str, Json]:
        super().restore(production_stopped=production_stopped)
        return {'status': 'RESTORED: K1 START VERIFIED, greedy 6/6 IDENTICAL, PRODUCTION HOLD OK, window closed',
                'restore_deadline_met': self.failure != 'deadline'}


def run(fake: Fake) -> tuple[a.Session, dict[str, Json]]:
    session = a.Session(fake, lambda row: None)
    result = w.window(fake, lambda row: None, selected=(), admission_run=session.run)
    result['post_close_watch'] = {'status': 'PASS'}
    session.finish(result)
    return session, result


def test_all_six_primary_gates_only_after_pair_and_never_headline(tmp_path: Path) -> None:
    fake = Fake(tmp_path)
    session, result = run(fake)
    assert result['status'] == 'ADMISSION_GATES_COMPLETE'
    assert session.data['verdict'] == 'QUALIFIED'
    assert set(record(session.data['gates'])) == set(prereq.GATES)
    assert fake.events.index('pair-qualification') < fake.events.index('probes-B')
    assert not any(event.startswith('cell-') for event in fake.events)  # only probe companions, no headline loop
    assert record(result['decision'])['status'] == 'NOT A CANDIDATE'
    for row in record(session.data['gates']).values(): a.checked_gate(record(row), 'fixture')


@pytest.mark.parametrize('gate', a.GATES)
def test_each_gate_failure_stops_and_strictly_restores(tmp_path: Path, gate: str) -> None:
    fake = Fake(tmp_path, gate)
    session, result = run(fake)
    assert 'stop' in result and session.data['verdict'] == 'NOT QUALIFIED'
    assert session.data['failing_gate'] == gate
    assert fake.events[-1] == 'restore'
    assert record(result['decision'])['status'] == 'REJECT'


@pytest.mark.parametrize('failure', ['restore', 'watch', 'deadline'])
def test_gates_cannot_qualify_without_restore_watch_and_deadline(tmp_path: Path, failure: str) -> None:
    fake = Fake(tmp_path, failure)
    session, result = run(fake)
    if failure == 'restore': result['restore'] = {'status': 'FAILED'}
    if failure == 'watch': result['post_close_watch'] = {'status': 'FAIL'}
    session.finish(result)
    assert session.data['verdict'] == 'NOT QUALIFIED'
    assert session.data['failing_gate'] == 'restore_and_post_close_watch'


def test_hash_changed_gate_never_qualifies(tmp_path: Path) -> None:
    session, result = run(Fake(tmp_path))
    path = Path(str(record(record(session.data['gates'])[a.GATES[0]])['evidence_path']))
    path.write_text('tampered')
    session.finish(result)
    assert session.data['verdict'] == 'NOT QUALIFIED'
    assert session.data['failing_gate'] == a.GATES[0]


def test_registered_probe_budget_fits_and_cannot_cut_reserves() -> None:
    assert a.estimate({})['total_s'] == sum(number(v) for v in a.DEFAULT_ESTIMATE.values()) <= 85*60
    for key in a.DEFAULT_ESTIMATE:
        with pytest.raises(ValueError): a.estimate({'admission_estimate': {**a.DEFAULT_ESTIMATE, key: 0}})
    with pytest.raises(ValueError): a.estimate({'admission_estimate': {**a.DEFAULT_ESTIMATE, 'probes_s': 4000}})
    with pytest.raises(ValueError): a.estimate({'admission_estimate': {**a.DEFAULT_ESTIMATE, 'engine_s': float('inf')}})


class Api:
    def __init__(self, bad: str = '') -> None:
        self.bad = bad
        self.requests: list[dict[str, Json]] = []
    def metrics(self) -> str: return 'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n'
    def post(self, path: str, data: dict[str, Json]) -> dict[str, Json]:
        self.requests.append(data)
        if path == '/tokenize': return {'tokens': [7]}
        if path == '/v1/completions':
            reason: Json = '!' if 'stop' in data else 8
            return {'choices': [{'finish_reason': 'length' if self.bad == str(reason) else 'stop',
                'text': '', 'stop_reason': reason}], 'usage': {'completion_tokens': 1}}
        message: dict[str, Json] = {'content': '{"answer":"PP4"}'}
        finish = 'stop'
        if 'tools' in data:
            message = {'tool_calls': [{'function': {'name': 'report', 'arguments': '{"answer":"PP4"}'}}]}
            if self.bad == 'tool': record(cast(list[Json], message['tool_calls'])[0])['function'] = {'name': 'other', 'arguments': '{}'}
            finish = 'tool_calls'
        elif record(data['chat_template_kwargs']).get('thinking') is True:
            message = {'content': '323', 'reasoning': '' if self.bad == 'reasoning' else 'public fixture'}
        elif self.bad in ('grammar', 'json', 'penalties'):
            if self.bad != 'penalties' or 'frequency_penalty' in data: message['content'] = '{}'
        return {'choices': [{'message': message, 'finish_reason': finish}], 'usage': {'completion_tokens': 10}}


def test_public_semantics_tools_requests_and_settings() -> None:
    http = Api()
    assert set(api.semantics(http, 8)) == {'grammar', 'penalties', 'stop', 'eos'}
    assert set(api.tools(http)) == {'tool', 'json', 'reasoning'}
    assert any(row.get('frequency_penalty') == .5 and row.get('repetition_penalty') == 1.1 for row in http.requests)
    assert all(row.get('temperature') == 1 and row.get('top_p') == 1 for row in http.requests if 'temperature' in row)


@pytest.mark.parametrize('bad', ['grammar', 'penalties', '!', '8', 'tool', 'json', 'reasoning'])
def test_api_semantics_never_fabricate_pass(bad: str) -> None:
    with pytest.raises(ValueError):
        if bad in ('tool', 'json', 'reasoning'): api.tools(Api(bad))
        else: api.semantics(Api(bad), 8)


def test_admission_flags_exclude_pair_only_and_dry_never_calls_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> object: raise AssertionError('remote call')
    monkeypatch.setattr(subprocess, 'check_output', forbidden)
    assert w.main(['--admission']) == 0
    with pytest.raises(SystemExit): w.main(['--admission', '--pair-only'])


@pytest.mark.parametrize('lost_reply', [False, True])
def test_admission85_deadline_binds_even_when_hold_ack_is_lost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lost_reply: bool) -> None:
    live = a.Live(tmp_path, pf.Supervisor(tmp_path), native=prior.WORKER, remote='experiment',
        expected=prior.WORKER.commit, data_dir=tmp_path, tag='owned-admission', admission={'verifier_sha256': {}})
    def opening(obj: w.Live) -> float:
        obj.hold_written = True
        obj.restore_by = 100+obj.restore_seconds
        obj.submit = 100+w.SUBMIT_S
        if lost_reply: raise TimeoutError('lost HOLD acknowledgement')
        return 100.
    monkeypatch.setattr(w.Live, 'open_hold', opening)
    if lost_reply:
        with pytest.raises(TimeoutError): live.open_hold()
    else: assert live.open_hold() == 100.
    assert live.restore_by == 100+85*60
    assert live.sup.abort_at == live.submit == 100+73*60


def test_real_probe_method_has_one_on_start_no_calibration_or_headlines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live = a.Live(tmp_path, pf.Supervisor(tmp_path), native=prior.WORKER, remote='experiment',
        expected=prior.WORKER.commit, data_dir=tmp_path, tag='owned-admission', admission={'verifier_sha256': {}})
    events: list[str] = []
    def start(arm: w.Arm) -> None:
        assert arm.admission and arm.role == 'B'
        events.append('start-B')
    monkeypatch.setattr(live, 'start', start)
    for method in ('build', 'warm', 'greedy', 'begin_measurement', 'finish'):
        def called(*args: object, key: str = method) -> None: events.append(key)
        monkeypatch.setattr(live, method, called)
    def forbidden(*args: object) -> None: raise AssertionError('headline cell forbidden')
    monkeypatch.setattr(live, 'cell', forbidden)
    monkeypatch.setattr(live, 'profile_cell', lambda arm, name: {'name': name})
    result = live.probes(lambda: events.append('fault-probes'))
    assert events == ['start-B', 'build', 'warm', 'greedy', 'fault-probes', 'begin_measurement', 'finish']
    assert set(record(result['profiles'])) == set(cells.CELLS)
    assert result['admission_only'] is True


def test_admission_runtime_requires_exact_run_env() -> None:
    prod = base.prod_identity()
    prod['argv'] = str(prod['argv']) + " --speculative-config '{}'"
    current = copy.deepcopy(prod)
    # Preserve quoting of the speculative JSON exactly as the production capture.
    import shlex
    argv = shlex.split(str(prod['argv']).replace('\\\n', ' ')); argv[argv.index('--port')+1] = '8001'
    argv[argv.index('--speculative-config')+1] = w.SPEC
    current['argv'] = shlex.join(argv)
    current['env'] = {**record(prod['env']), **w.COMMON_ENV, 'VLLM_DSPARK_SAMPLED_TOP32': '1',
        **dict(zip(w.IDENTITY_KEYS, (prior.WORKER.identity, prior.WORKER.markov))), 'VLLM_PP4_ADMISSION_RUN': 'owned-run'}
    arm = w.Arm('admission-B', 327680, 0, role='B', admission=True)
    w.runtime(prod, current, arm, prior.WORKER, {}, admission_run='owned-run')
    with pytest.raises(ValueError): w.runtime(prod, current, arm, prior.WORKER, {})
    with pytest.raises(ValueError): w.runtime(prod, current, arm, prior.WORKER, {}, admission_run='foreign-run')


def test_actual_admission_config_selects_only_new_overlay_paths(tmp_path: Path) -> None:
    import os
    import shlex
    home = tmp_path/'home'; prod = home/'dsv41-deploy'; prod.mkdir(parents=True)
    (prod/'dsv41.env').write_bytes(subprocess.check_output(['git', 'show', w.PROD_COMMIT+':dsv41.env'], cwd=pf.ROOT))
    root = tmp_path/'offload/launch/windows/pp4-admission-staged/verifier'
    for name in ('vllm/v1/worker/gpu/model_runner.py', 'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py',
                 'vllm/v1/worker/gpu/dsv41_pp4_round_trace.py', 'vllm/v1/worker/gpu/dsv41_pp4_admission.py'):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.touch()
    command = 'set -eu; source '+shlex.quote(str(pf.ROOT/w.CONFIG))+'; printf "%s\\n%s\\n" "$EXP_DOCKER_ENV" "$OVERLAYS"'
    proc = subprocess.run(['bash', '-c', command], cwd=tmp_path, env={**os.environ, 'HOME': str(home),
        'PP4_ADMISSION': '1', 'PP4_ROLE': 'B', 'PP4_NATIVE_IDENTITY': prior.WORKER.identity,
        'PP4_NATIVE_MARKOV_IDENTITY': prior.WORKER.markov, 'PP4_TRACE_RUN': 'owned-run'},
        capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert 'VLLM_PP4_ADMISSION_RUN=owned-run' in lines[0]
    mappings = dict(pair.split('=', 1) for pair in lines[1].split())
    assert len(mappings) == 5
    assert all(name.startswith('offload/launch/windows/pp4-admission-staged/verifier/') for name in mappings)
    assert mappings['offload/launch/windows/pp4-admission-staged/verifier/remote_dspark'] == pf.DIST+'vllm/v1/worker/gpu/spec_decode/remote_dspark'
    assert all('pp4-staged/' not in name for name in mappings)


def test_seed_requires_six_original_gates_and_exact_admission_manifest(tmp_path: Path) -> None:
    from offload.launch.windows import pp4_admission_prepare as prep
    from offload.launch.windows import up2_checks as ck
    ordinary = record(ck.read(pf.ROOT/'offload/dspark/probabilistic/pp4/verifier-k1-receipt.json')['output_sha256'])
    seed: dict[str, Json] = {'gates': {}, 'verifier_sha256': ordinary}
    def staging(path: Path, expected: str) -> dict[str, Json]: return copy.deepcopy(seed)
    source = tmp_path/'proof.json'; source.write_text('{}')
    gate: dict[str, Json] = {'pass': True, 'evidence_path': str(source.resolve()), 'sha256': pf.digest(source)}
    seed['gates'] = {name: gate for name in a.SEED_GATES}
    outputs: dict[str, Json] = {**ordinary, prep.HELPER: 'b'*64, 'remote_dspark/'+prep.SEAM: 'e'*64,
        **{'remote_dspark/'+p.name: 'c'*64 for p in (pf.ROOT/'offload/dspark/spark/remote_dspark').glob('*.py')}}
    seed['admission_verifier_sha256'] = outputs
    seed['admission_estimate'] = dict(a.DEFAULT_ESTIMATE)
    from unittest.mock import patch
    with patch.object(prereq, 'staging', staging):
        assert a.seed(source, 'a'*40)['admission_budget']
        outputs['unsafe;command'] = 'd'*64
        with pytest.raises(ValueError, match='exact'): a.seed(source, 'a'*40)
        outputs.pop('unsafe;command')
        seam = outputs.pop('remote_dspark/'+prep.SEAM)  # R15 manifests (no proxy-local seam) no longer seed
        with pytest.raises(ValueError, match='exact'): a.seed(source, 'a'*40)
        outputs['remote_dspark/'+prep.SEAM] = seam
        record(seed['gates'])[a.GATES[0]] = gate
        with pytest.raises(ValueError, match='six'): a.seed(source, 'a'*40)


@pytest.mark.parametrize('failure,watch_status,expected', [('', 'PASS', 0), ('both_rank_q_failure_target_only', 'PASS', 1),
    ('', 'FAILED', 2), ('restore', 'PASS', 2)])
def test_real_main_admission_receipt_requires_shared_restore_and_watch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        failure: str, watch_status: str, expected: int) -> None:
    import json
    import signal

    from offload.launch.windows import post_close_watch as watch
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    monkeypatch.setattr(pf, 'CAMPAIGN', tmp_path)
    monkeypatch.setattr(subprocess, 'check_output', lambda argv, **kwargs: prior.WORKER.commit if argv[1] == 'rev-parse' else '')
    monkeypatch.setattr(a, 'seed', lambda *args: {'worker_identity': prior.WORKER.identity, 'markov_identity': prior.WORKER.markov})
    created: list[Fake] = []
    class MainFake(Fake):
        def __init__(self, out: Path, *args: object, **kwargs: object) -> None:
            super().__init__(out, failure)
            created.append(self)
        def restore(self, *, production_stopped: bool) -> dict[str, Json]:
            if failure == 'restore':
                self.events.append('restore')
                raise ValueError('fixture failed strict restore')
            return super().restore(production_stopped=production_stopped)
    monkeypatch.setattr(a, 'Live', MainFake)
    monkeypatch.setattr(a, 'configure', lambda value: value)
    monkeypatch.setattr(watch, 'ProductionHealth', object)
    calls: list[str] = []
    def recovery(tag: str, path: Path) -> object:
        assert created[0].events[-1] == 'restore'
        calls.append('watch')
        return object()
    monkeypatch.setattr(watch, 'ScopedRecovery', recovery)
    monkeypatch.setattr(watch, 'watch', lambda *args, **kwargs: watch.Result(watch_status, 30, 0, 0))
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    out = tmp_path/'PP4-admission-main'
    try:
        code = w.main(['--execute', '--admission', '--qualification', str(tmp_path/'seed.json'),
            '--expected-experiment-commit', prior.WORKER.commit, '--worker-commit', prior.WORKER.commit,
            '--worker-tree', prior.WORKER.tree, '--worker-identity', prior.WORKER.identity,
            '--worker-markov-identity', prior.WORKER.markov, '--out', str(out)])
    finally:
        for sig, handler in saved.items(): signal.signal(sig, handler)
    assert code == expected
    receipt = json.loads((out/'admission-receipt.json').read_text())
    assert receipt['verdict'] == ('QUALIFIED' if not failure and watch_status == 'PASS' else 'NOT QUALIFIED')
    assert receipt['headline_measurements'] is False
    assert bool(calls) == (failure != 'restore')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['segment'] == 'admission' and summary['starts'] == {}
    assert summary['decision']['status'] != 'CANDIDATE'
