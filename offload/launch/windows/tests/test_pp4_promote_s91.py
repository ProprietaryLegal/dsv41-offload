"""S91 (FIX 6) reproductions of the PP4-PROMOTE package, kept as regressions. CPU only; no host contact."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import cast

import pytest

from offload.launch.windows import decision_v2
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import post_close_watch as watch
from offload.launch.windows import pp4_budget
from offload.launch.windows import pp4_promote_package as package
from offload.launch.windows import pp4_prerequisites as prereq
from offload.launch.windows import pp4_promote_switch as switch
from offload.launch.windows import pp4_promote_window as w
from offload.launch.windows import scoped_remote
from offload.launch.windows.tests import test_pp4_promote_switch as sw
from offload.launch.windows.tests.test_pp4_promote_switch import verified  # noqa: F401  (fixture)
from offload.prefill_maxlen.evidence import Json, record

IDENTITY = 'be2c1436fa9f9e66b398e1b1da2e140e6fddc82a11939e2ac4240a0ca79543bd'      # production 596ba1c dsv41.env
MARKOV = '0b6e71e638abd5abce2db433965bb5ff523cbbe92ed7ae54324a50ccaeffa19a'


def live(tmp_path: Path) -> w.Live:
    out = tmp_path / 'run'
    out.mkdir()
    for name in ('admission.json', 'measurement.json', 'restore.sh'):
        (tmp_path / name).write_text('{}\n')
    return w.Live('S91T', out, '0' * 40, tmp_path / 'restore.sh', '0' * 64,
                  tmp_path / 'admission.json', tmp_path / 'measurement.json')


W8_CHECK_OK = ('sleep 0.6\n' + ''.join(f"echo 'greedy K1-post vs restored: {i} IDENTICAL 100'\n" for i in range(6))
               + "echo 'PRODUCTION HOLD OK'\necho 'CHECK: DONE'\n")


@pytest.mark.parametrize('foreign', [0, 1])
def test_s91_1_final_w8_check_owns_one_request_and_still_sees_foreign_demand(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign: int) -> None:
    """S91 #1: the actual Live.final -> Live.command -> Supervisor.run -> StrictSupervisor.poll path, with the demand
    guard live. The W8 check's own greedy request is owned; one more request is foreign and still stops the window."""
    script = tmp_path / 'restore-pp4.sh'
    script.write_text(W8_CHECK_OK)
    monkeypatch.setattr(w, 'NEW_RESTORE', script)
    backend = live(tmp_path)
    (tmp_path / 'sup').mkdir()

    def demand() -> float:
        sup.demand_due = 0.0                 # observe at every poll instead of every 5 s
        child = sup.active
        return float(1 + foreign) if child is not None and child.poll() is None else 0.0

    sup = w.StrictSupervisor(tmp_path / 'sup', demand=demand)
    backend.sup = sup
    sup.mark_ready()
    try:
        if foreign:
            with pytest.raises(ValueError, match=r'foreign demand on the experiment \(2 > owned 1\)'):
                backend.final()
        else:
            backend.final()
    finally:
        sup.close()
    assert sup.active is None


# --- S91 #2: preflight accepted unbound / incomplete qualification and measurement data ---------------------------

def admission_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **change: Json) -> Path:
    """A QUALIFIED receipt in the real pp4_admission shape. pp4_budget.admitted (the physical-pair binding) is stubbed
    exactly as test_pp4_window's qualification test does; its own contract has its own tests."""
    run = tmp_path / 'admission-run'
    run.mkdir(exist_ok=True)
    gate = run / 'gate-evidence.json'
    gate.write_text('{"pass": true}\n')
    (run / 'admission-B-greedy.json').write_text('{"greedy": "pass"}\n')
    data: dict[str, Json] = {
        'schema': 'pp4-live-admission-v1', 'verdict': 'QUALIFIED', 'source_commit': switch.PP4_COMMIT, 'depth': 5,
        'worker_identity': IDENTITY, 'markov_identity': MARKOV, 'runtime_python': prereq.PYTHON,
        'runtime_torch': '2.12.1+cu130', 'runtime_artifact_sha256': cast(Json, dict(switch.SO_SHA256)),
        'gates': {g: {'pass': True, 'evidence_path': str(gate), 'sha256': pf.digest(gate)} for g in prereq.GATES},
        'verifier_sha256': cast(Json, dict(switch.VERIFIER_SHA256)),
        'artifact_sha256': {'admission-B-greedy.json': pf.digest(run / 'admission-B-greedy.json')}}
    data.update(change)
    monkeypatch.setattr(pp4_budget, 'admitted', lambda path, expected: {'block_budget_s': {'s1-A': 1, 's2-B': 1, 's3-A': 1}})
    monkeypatch.setattr(pp4_budget, 'pair_reserve', lambda path, expected: {'seconds': 333, 'classification': 'fixture'})
    path = run / 'admission-receipt.json'
    path.write_text(json.dumps(data))
    return path


R_PP4_HEAD = '441e0c3638e639f5387fe0814e6a972bac746336'   # offload/pp4-win2 441e0c3: merges R16 and carries 90f0278


def rpp4_git(args: list[str]) -> str:
    return subprocess.check_output(['git', *args], cwd=pf.ROOT, text=True)


def v2_rows(metrics: tuple[str, ...], rate: list[float] | None = None) -> list[dict[str, Json]]:
    result: list[dict[str, Json]] = []
    for cell in pp4_budget.B1_CELLS:
        for metric in metrics:
            candidate = rate if metric == 'pooled_rate_tps' and rate is not None else [10.0, 10.05, 9.95]
            result.append({'cell': cell, 'metric': metric, 'higher_is_better': w.ORIENTATION[metric],
                           'control': [10.0, 10.1, 9.9], 'candidate': cast(Json, candidate), 'paired': True})
    return result


def series_rows(rate: list[float] | None = None) -> list[dict[str, Json]]:
    return v2_rows(w.MEASUREMENT_METRICS, rate)


SAME_CLOCK: dict[str, Json] = {'consistent': True, 'reason': None}


def decided(rows: list[dict[str, Json]]) -> dict[str, Json]:
    """decide_window plus the same-clock fields R-PP4's pp4_measure_v2.decide always writes."""
    return {**decision_v2.decide_window([decision_v2.Series(str(r['cell']), str(r['metric']), bool(r['higher_is_better']),
        tuple(float(cast(float, v)) for v in cast(list[Json], r['control'])),
        tuple(float(cast(float, v)) for v in cast(list[Json], r['candidate'])), bool(r['paired'])) for r in rows],
        pp4_budget.B1_CELLS), 'clock': dict(SAME_CLOCK), 'reason': None}


def measurement_summary(tmp_path: Path, admission: Path, rows: list[dict[str, Json]] | None = None,
                        decision: dict[str, Json] | None = None, /, **change: Json) -> Path:
    """The R-PP4 v2 production-policy summary shape (pp4_window.main, measure_v2)."""
    run = tmp_path / 'measure-run'
    run.mkdir(exist_ok=True)
    (run / 'cells.json').write_text('{"cells": "recorded"}\n')
    rng = run / 'rng-gate.json'
    rng.write_text('{"status": "PASS"}\n')
    series = series_rows() if rows is None else rows
    data: dict[str, Json] = {
        'status': 'COMPLETE', 'measure_v2': 'production-policy', 'experiment_commit': R_PP4_HEAD,
        'runtime_commit': switch.PP4_COMMIT, 'qualification_sha256': pf.digest(admission),
        'carried_qualification': {'qualified_commit': switch.PP4_COMMIT, 'head': R_PP4_HEAD, 'changed_files': cast(
            Json, sorted(p for p in rpp4_git(['diff', '--name-only', switch.PP4_COMMIT, R_PP4_HEAD]).split() if p)),
            'rule': 'fixture'},
        'config_sha256': pf.digest(w.SAMPLED_CONFIG), 'production_policy_config_sha256': pf.digest(w.POLICY_CONFIG),
        'rng_gate': {'status': 'PASS', 'receipt': str(rng), 'receipt_sha256': pf.digest(rng), 'commit': R_PP4_HEAD,
                     'pp4_kernel_sha256': switch.VERIFIER_SHA256[switch.UTILS]},
        'decision': decided(series) if decision is None else decision, 'decision_series': cast(Json, series),
        'reported_series': cast(Json, v2_rows(w.REPORTED_METRICS)),
        'artifact_sha256': {'cells.json': pf.digest(run / 'cells.json')}}
    data.update(change)
    path = run / 'summary.json'
    path.write_text(json.dumps(data))
    return path


def s91_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """S91's synthetic reproduction: the three formerly checked fields right, everything else unbound or missing."""
    admission = tmp_path / 's91-admission.json'
    admission.write_text(json.dumps({'verdict': 'QUALIFIED', 'source_commit': switch.PP4_COMMIT,
        'verifier_sha256': dict(switch.VERIFIER_SHA256), 'schema': 'pp4-admission-unknown', 'gates': {},
        'runtime_artifact_sha256': {}}))
    measurement = tmp_path / 's91-measurement.json'
    measurement.write_text(json.dumps({'status': 'COMPLETE', 'experiment_commit': '0' * 40, 'decision': {
        'status': 'INCONCLUSIVE', 'cells': {'invented-b1': {'no_loss_beyond_spread': True, 'timing_guards': {}}}}}))
    return admission, measurement


def test_s91_2_reproduction_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    admission, measurement = s91_inputs(tmp_path)
    with pytest.raises(ValueError, match='schema'):
        w.evidence_gate(admission, measurement)
    good = admission_receipt(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='production-policy v2 measurement'):
        w.evidence_gate(good, measurement)
    # Labelled as the right mode and admission, it still carries no admitted source and no recomputable decision.
    unbound = json.loads(measurement.read_text()) | {'measure_v2': 'production-policy', 'qualification_sha256': pf.digest(good)}
    measurement.write_text(json.dumps(unbound))
    with pytest.raises(ValueError, match='does not carry the admitted source'):
        w.evidence_gate(good, measurement)


def test_s91_2_preflight_refuses_before_first_contact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    (campaign / 'D2-GEN.PAUSE').touch()
    (campaign / 'greedy-post.jsonl').write_text('{}\n')
    monkeypatch.setattr(w, 'CAMPAIGN', campaign)
    monkeypatch.setattr(w, 'K1_POST', campaign / 'greedy-post.jsonl')
    monkeypatch.setattr(w, 'NEW_REFERENCES', campaign / 'greedy-post-pp4.jsonl')
    monkeypatch.setenv('WINDOW_COORDINATOR', 'Deepseek/coordinator')
    admission, measurement = s91_inputs(tmp_path)
    restore = tmp_path / 'restore.sh'
    restore.write_text('# prior W8\n')
    out = tmp_path / 'run'
    out.mkdir()
    backend = w.Live('S91P', out, '1' * 40, restore, pf.digest(restore), admission, measurement)

    def contact(*args: object, **kwargs: object) -> str:
        raise AssertionError('first telemetry/host contact reached')
    monkeypatch.setattr(backend.sup, 'begin', contact)
    monkeypatch.setattr(backend, 'command', contact)
    with pytest.raises(ValueError, match='schema'):
        backend.preflight()
    assert not (out / 'evidence.json').exists()


def test_s91_2_bound_evidence_passes_without_a_win(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    admission = admission_receipt(tmp_path, monkeypatch)
    evidence = w.evidence_gate(admission, measurement_summary(tmp_path, admission))
    assert record(evidence['measurement'])['status'] == 'INCONCLUSIVE'
    assert record(evidence['admission'])['sha256'] == pf.digest(admission)
    rank: dict[str, Json] = {'env': {'VLLM_DSPARK_REMOTE_IDENTITY': IDENTITY, 'VLLM_DSPARK_REMOTE_MARKOV_IDENTITY': MARKOV}}
    w.bind_identities(evidence, [rank, rank])
    other: dict[str, Json] = {'env': {**record(rank['env']), 'VLLM_DSPARK_REMOTE_IDENTITY': 'f' * 64}}
    with pytest.raises(ValueError, match='worker_identity'):
        w.bind_identities(evidence, [rank, other])


ADMISSION_DEFECTS: list[tuple[str, dict[str, Json]]] = [
    ('NOT QUALIFIED', {'verdict': 'NOT QUALIFIED'}),
    ('other source', {'source_commit': '4292855' + '0' * 33}),
    ('verifier pin', {'verifier_sha256': {**switch.VERIFIER_SHA256, switch.RUNNER: 'a' * 64}}),
    ('runtime binary', {'runtime_artifact_sha256': {**switch.SO_SHA256, prereq.STOCK: 'd' * 64}}),
    ('no artifacts', {'artifact_sha256': {}}),
    ('missing artifact', {'artifact_sha256': {'absent.json': 'e' * 64}}),
    ('escaping artifact', {'artifact_sha256': {'../gate-evidence.json': 'e' * 64}}),
    ('gate missing', {'gates': {}}),
]


@pytest.mark.parametrize('name,change', ADMISSION_DEFECTS, ids=[n for n, _ in ADMISSION_DEFECTS])
def test_s91_2_each_admission_binding_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
                                              change: dict[str, Json]) -> None:
    admission = admission_receipt(tmp_path, monkeypatch, **change)
    with pytest.raises(ValueError):
        w.evidence_gate(admission, measurement_summary(tmp_path, admission))


def test_s91_2_changed_primary_gate_evidence_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    admission = admission_receipt(tmp_path, monkeypatch)
    measurement = measurement_summary(tmp_path, admission)
    (admission.parent / 'gate-evidence.json').write_text('{"pass": false}\n')
    with pytest.raises(ValueError, match='changed'):
        w.evidence_gate(admission, measurement)


def edited_status(tmp_path: Path, admission: Path) -> Path:
    rows = series_rows(rate=[9.4, 9.3, 9.45])        # every repeat ~6 % fewer tokens/s than OFF: LOSS under any revision
    path = measurement_summary(tmp_path, admission, rows)
    data = json.loads(path.read_text())
    assert data['decision']['status'] == 'NOT A CANDIDATE'
    data['decision']['status'] = 'INCONCLUSIVE'
    path.write_text(json.dumps(data))
    return path


MEASUREMENT_DEFECTS: list[tuple[str, dict[str, Json]]] = [
    ('incomplete', {'status': 'INCOMPLETE'}),
    ('stopped', {'stop': 'ValueError: boom'}),
    ('controlled mode', {'measure_v2': 'controlled'}),
    ('other admission', {'qualification_sha256': '9' * 64}),
    ('other runtime', {'runtime_commit': '941a3c3fbe6fec06a05e925a1e7d77569b0d12aa'}),
    ('other head', {'experiment_commit': '0' * 40}),
    ('uncarried head', {'experiment_commit': '6770b5b' + '0' * 33}),
    ('other config', {'config_sha256': '8' * 64}),
    ('other policy config', {'production_policy_config_sha256': '8' * 64}),
    ('rng not passed', {'rng_gate': {'status': 'FAIL'}}),
    ('no artifacts', {'artifact_sha256': {}}),
    ('v1 rule', {'decision': {'status': 'INCONCLUSIVE', 'cells': {}}}),
    ('no series', {'decision_series': []}),
    ('no reported series', {'reported_series': []}),
]


@pytest.mark.parametrize('name,change', MEASUREMENT_DEFECTS, ids=[n for n, _ in MEASUREMENT_DEFECTS])
def test_s91_2_each_measurement_binding_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
                                                change: dict[str, Json]) -> None:
    admission = admission_receipt(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        w.evidence_gate(admission, measurement_summary(tmp_path, admission, **change))


def test_s91_2_series_must_be_registered_oriented_and_recompute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    admission = admission_receipt(tmp_path, monkeypatch)
    without_guard = [r for r in series_rows() if not (r['cell'] == '32k-b1' and r['metric'] == 'decode_tps')]
    with pytest.raises(ValueError, match=r"32k-b1: decision series \['pooled_rate_tps', 'accepted_per_round'\] differ"):
        w.evidence_gate(admission, measurement_summary(tmp_path, admission, without_guard))
    ms_primary = v2_rows(('ms_per_round', 'accepted_per_round', 'decode_tps'))     # ms/round may not decide
    with pytest.raises(ValueError, match='decision series'):
        w.evidence_gate(admission, measurement_summary(tmp_path, admission, ms_primary))
    flipped = [{**r, 'higher_is_better': not r['higher_is_better']} if r['metric'] == 'pooled_rate_tps' else r
               for r in series_rows()]
    with pytest.raises(ValueError, match='orientation'):
        w.evidence_gate(admission, measurement_summary(tmp_path, admission, flipped))
    with pytest.raises(ValueError, match='reported series'):
        w.evidence_gate(admission, measurement_summary(tmp_path, admission, None, None,
                                                       reported_series=cast(Json, v2_rows(('ms_per_round',)))))
    with pytest.raises(ValueError, match='does not recompute'):
        w.evidence_gate(admission, edited_status(tmp_path, admission))
    with pytest.raises(ValueError, match='LOSES'):
        w.evidence_gate(admission, measurement_summary(tmp_path, admission, series_rows(rate=[9.4, 9.3, 9.45])))
    two: list[dict[str, Json]] = [{**r, 'control': [10.0, 10.1], 'candidate': [10.0, 10.0]} for r in series_rows()]
    with pytest.raises(ValueError, match='at least 3'):
        w.evidence_gate(admission, measurement_summary(tmp_path, admission, two, {
            'rule': decision_v2.RULE, 'min_gain': .02, 'decision_cells': list(pp4_budget.B1_CELLS), 'status': 'INCONCLUSIVE',
            'cells': {}, 'clock': dict(SAME_CLOCK), 'reason': None}))
    with pytest.raises(ValueError, match='decision cells'):
        w.evidence_gate(admission, measurement_summary(tmp_path, admission, None, {
            **decided(series_rows()), 'decision_cells': ['short-b1', 'invented-b1']}))
    changed = measurement_summary(tmp_path, admission)
    (changed.parent / 'cells.json').write_text('{"cells": "edited"}\n')
    with pytest.raises(ValueError, match='absent/changed'):
        w.evidence_gate(admission, changed)
    rng = measurement_summary(tmp_path, admission)
    (rng.parent / 'rng-gate.json').write_text('{"status": "edited"}\n')
    with pytest.raises(ValueError, match='verifier-RNG'):
        w.evidence_gate(admission, rng)


@pytest.mark.parametrize('candidate,clock,ok', [
    ({'accepted': 230., 'seconds': 4.1, 'tps': 108.}, {'consistent': True, 'reason': None}, True),   # R-PP4's WIN
    ({'accepted': 150.}, {'consistent': True, 'reason': None}, False),                                  # LOSS
    ({'accepted': 230., 'seconds': 4.1, 'tps': 108.}, {'consistent': False, 'reason': 'caps differ'}, False),  # clock guard
])
def test_rpp4_producer_summary_passes_this_gate_only_without_loss_or_clock_override(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate: dict[str, float], clock: dict[str, Json],
        ok: bool) -> None:
    """Cross-check with R-PP4's own producer (pp4_measure_v2.decide on its test fixtures), placed exactly as
    pp4_window.main writes it: series top-level, the rest under `decision`."""
    from offload.launch.windows import pp4_measure_v2 as mv2
    from offload.launch.windows.tests import test_pp4_measure_v2 as rpp4
    decided_v2 = mv2.decide(rpp4.starts('production-policy', candidate=candidate), 'production-policy', clock=clock,
                            rng=rpp4.GATE, archive=None)
    series, reported = decided_v2.pop('decision_series'), decided_v2.pop('reported_series')
    admission = admission_receipt(tmp_path, monkeypatch)
    path = measurement_summary(tmp_path, admission, cast(list[dict[str, Json]], series), decided_v2,
                               reported_series=reported)
    if ok:
        assert record(w.evidence_gate(admission, path)['measurement'])['status'] == 'CANDIDATE'
    else:
        with pytest.raises(ValueError, match='LOSES|does not recompute|clock guard'):
            w.evidence_gate(admission, path)


def test_measurement_contract_matches_the_rpp4_producer() -> None:
    from offload.launch.windows import pp4_measure_v2 as mv2
    assert tuple(mv2.CELLS) == tuple(w.MEASUREMENT_DECISION_CELLS) and mv2.PRIMARY == w.MEASUREMENT_METRICS[0]
    assert 'production-policy' in mv2.ARCHIVE_CELL and w.MEASUREMENT_MODE == 'production-policy'


# --- S91 #3: the abort deadline did not preserve the stated recovery reserve -------------------------------------

def test_s91_3_a_failure_at_the_abort_bound_keeps_the_whole_recovery_reserve() -> None:
    estimate = w.window_estimate()
    assert w.ABORT_S + w.RECOVERY_S + w.UNPRICED_SLACK_S + w.FINAL_CLOSE_RESERVE_S <= w.RESTORE_S
    assert estimate['failure_at_abort_hold_s'] <= w.RESTORE_S and estimate['forward_s'] <= w.ABORT_S
    assert w.SUBMIT_S + w.FINAL_W8_CHECK_S + w.ACCEPT_S <= w.ABORT_S
    bounds = w.deadlines(1000.0)
    assert bounds['scope'] - bounds['abort'] >= w.RECOVERY_S + w.UNPRICED_SLACK_S
    assert bounds['restore'] - bounds['scope'] == w.FINAL_CLOSE_RESERVE_S and bounds['submit'] < bounds['abort']


def test_s91_3_round1_bounds_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """S91's arithmetic: abort +65 with a 20.4-min reserve leaves 20 min to +85, before slack and the final close."""
    monkeypatch.setattr(w, 'ABORT_S', 65 * 60)
    with pytest.raises(ValueError, match='recovery reserve violated'):
        w.window_estimate()


def test_s91_3_forward_work_must_fit_before_the_abort_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(w, 'WORKER_SWAP_S', 20 * 60)
    with pytest.raises(ValueError, match='exceeds the abort bound'):
        w.window_estimate()
    monkeypatch.setattr(w, 'WORKER_SWAP_S', 10 * 60)
    monkeypatch.setattr(w, 'SUBMIT_S', w.ABORT_S - 60)
    with pytest.raises(ValueError, match='submission deadline'):
        w.window_estimate()


# --- S91 #4: hook activation and rollback did not preserve the default-mode behaviour -----------------------------

def production_tree(tmp_path: Path) -> tuple[Path, Path]:
    """596ba1c's serve.sh + K1 + kvpack with this package applied, an installed K1 record, and recording stubs for
    start.sh/stop.sh and a stage.sh that fails (S91's synthetic stage failure)."""
    tree = tmp_path / 'prod'
    tree.mkdir()
    archive = subprocess.run(['git', '-C', str(sw.ROOT), 'archive', sw.PRODUCTION, 'serve.sh', 'offload/promote/k1',
                              'offload/kvpack/stage_overlay.sh', 'offload/kvpack/MANIFEST.json', 'offload/kvpack/overlay'],
                             check=True, capture_output=True).stdout
    subprocess.run(['tar', '-x', '-C', str(tree)], input=archive, check=True)
    (tree / 'dsv41.env').write_text(sw.production_env(False, tmp_path))
    sw.fold(tree / 'dsv41.env')                       # 420be24 cherry-picked before the hooks
    package.apply(tree)
    calls = tmp_path / 'calls.log'
    calls.write_text('')
    (tree / 'start.sh').write_text(f'#!/bin/sh\necho start >> {calls}\n')
    (tree / 'stop.sh').write_text(f'#!/bin/sh\necho stop >> {calls}\n')
    (tree / 'offload/promote/k1/stage.sh').write_text('#!/bin/sh\necho synthetic stage failure >&2\nexit 7\n')
    manage = switch.k1_manager(tree)
    state = tree / 'logs/k1-promotion'
    state.mkdir(parents=True)
    for name in manage.FILES:
        shutil.copy2(tree / name, state / name)
    digest = {name: manage.sha(tree / name) for name in manage.FILES}
    (state / 'files.json').write_text(json.dumps({'profile': '327680', 'original': digest, 'promoted': digest}) + '\n')
    return tree, calls


def stubs(tmp_path: Path, calls: Path) -> dict[str, str]:
    """Local CPU-only stand-ins: ssh/docker answer nothing; the installed K1 manager records verify/restore."""
    bindir = tmp_path / 'bin'
    bindir.mkdir(exist_ok=True)
    for name, text in (('ssh', '#!/bin/sh\nexit 0\n'), ('docker', '#!/bin/sh\nexit 0\n'),
                       ('python3', f'#!/bin/sh\ncase "$1" in offload/promote/k1/manage.py) echo "manage $2" >> {calls}; '
                                   f'exit 0;; esac\nexec {sys.executable} "$@"\n')):
        (bindir / name).write_text(text)
        (bindir / name).chmod(0o755)
    home = tmp_path / 'home'
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PROD_', 'OFFLOAD_', 'KVPACK_'))}
    return {**env, 'PATH': f'{bindir}:{os.environ["PATH"]}', 'HOME': str(home)}


def start_guard(tree: Path, tmp_path: Path, calls: Path) -> tuple[int, str, list[str]]:
    calls.write_text('')
    done = subprocess.run(['bash', 'offload/promote/k1/start_checked.sh'], cwd=tree, env=stubs(tmp_path, calls),
                          capture_output=True, text=True, timeout=120)
    return done.returncode, done.stdout + done.stderr, calls.read_text().splitlines()


def test_s91_4_exact_rollback_disarms_the_delegation_and_keeps_the_ledger(
        tmp_path: Path, verified: list[tuple[Path, str | None]]) -> None:
    tree, calls = production_tree(tmp_path)
    rc, text, seen = start_guard(tree, tmp_path, calls)              # default mode: the original K1 fallback
    assert rc == 7 and 'K1 REFUSED; previous configuration restarted' in text
    assert seen == ['manage verify', 'stop', 'manage restore', 'start']
    original = (tree / 'dsv41.env').read_bytes()
    switch.apply(tree)
    rc, text, seen = start_guard(tree, tmp_path, calls)              # PP4 installed: delegated, never the pre-K1 backup
    assert rc == 7 and 'K1 PP4 REFUSED' in text and seen == ['manage verify', 'stop']
    (tree / 'dsv41.env').write_bytes(original)                        # transaction open, env write lost: still delegated
    rc, text, seen = start_guard(tree, tmp_path, calls)
    assert rc == 7 and 'K1 PP4 REFUSED' in text and 'manage restore' not in seen
    switch.restore(tree)                                              # exact rollback: S91's reproduction
    assert (tree / 'dsv41.env').read_bytes() == original and (tree / switch.BACKUP / 'transaction.json').exists()
    assert (tree / switch.BACKUP / switch.RESTORED).exists()
    rc, text, seen = start_guard(tree, tmp_path, calls)
    assert rc == 7 and 'K1 PP4 REFUSED' not in text and seen == ['manage verify', 'stop', 'manage restore', 'start']
    switch.restore(tree)                                              # a repeated restore stays exact and idempotent
    with pytest.raises(ValueError, match='restored'):
        switch.active_state(tree)


def serve(tree: Path, tmp_path: Path, calls: Path, **extra: str) -> tuple[int, str]:
    recipe = tmp_path / 'recipe'
    launcher = recipe / 'scripts/launch/dsv41_tp2.sh'
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text('#!/bin/bash\necho "LAUNCH SPEC=$SPEC"\n')
    env = {**stubs(tmp_path, calls), 'HEAD': 'head', 'WORKER': 'worker', 'RECIPE': str(recipe), **extra}
    done = subprocess.run(['bash', 'serve.sh'], cwd=tree, env=env, capture_output=True, text=True, timeout=60)
    return done.returncode, done.stdout + done.stderr


def test_s91_4_spec_override_needs_a_validated_active_switch(
        tmp_path: Path, verified: list[tuple[Path, str | None]]) -> None:
    tree, calls = production_tree(tmp_path)
    production = sw.show(sw.PRODUCTION, 'serve.sh').split("export SPEC='", 1)[1].split("'", 1)[0]
    rc, text = serve(tree, tmp_path, calls)
    assert rc == 0 and f'LAUNCH SPEC={production}' in text            # default mode unchanged
    rc, text = serve(tree, tmp_path, calls, PROD_SPEC_OVERRIDE=switch.SPEC)   # S91: inherited override only
    assert rc == 1 and 'LAUNCH' not in text and 'without an active PP4 switch' in text
    switch.apply(tree)
    rc, text = serve(tree, tmp_path, calls, PROD_SPEC_OVERRIDE=switch.SPEC)
    assert rc == 0 and f'LAUNCH SPEC={switch.SPEC}' in text
    rc, text = serve(tree, tmp_path, calls, PROD_SPEC_OVERRIDE='{"method":"dspark","num_speculative_tokens":3}')
    assert rc == 1 and 'LAUNCH' not in text
    switch.restore(tree)
    rc, text = serve(tree, tmp_path, calls, PROD_SPEC_OVERRIDE=switch.SPEC)
    assert rc == 1 and 'LAUNCH' not in text
    rc, text = serve(tree, tmp_path, calls)
    assert rc == 0 and f'LAUNCH SPEC={production}' in text


# --- S91 #5: the versioned W8 check did not verify the current PP4 configuration ----------------------------------

FAKE_SSH = '''import json, os, sys
args = sys.argv[1:]
while args and args[0].startswith('-'):
    args = args[2:] if args[0] == '-o' else args[1:]
host, command = args[0], ' '.join(args[1:])
scenario = json.load(open(os.environ['FAKE_SCENARIO']))
with open(os.environ['FAKE_CALLS'], 'a') as log:
    log.write(json.dumps([host, command]) + '\\n')
if host == 'x86host' and command == 'bash -s':
    with open(os.environ['FAKE_CALLS'] + '.worker', 'w') as probe:
        probe.write(sys.stdin.read())
    if scenario['worker']:
        print('PP4 WORKER SAMPLED')
    raise SystemExit(0 if scenario['worker'] else 1)
if host == 'x86host' and 'prod_window.sh check' in command:
    print('PRODUCTION HOLD OK')
elif host == 'x86host' and "echo 'PP4 WORKER SAMPLED'" in command:      # round-1 marker-only worker probe
    print('PP4 WORKER SAMPLED')
elif host == 'sparkhead' and 'k1-guard' in command:
    print('K1 START VERIFIED; complete the production-window probes before releasing HOLD')
    print('PP4 RUNNING VERIFIED')
elif host == 'sparkhead' and 'verify.sh' in command:
    if not scenario['config']:
        print('PP4 SWITCH REFUSED: no PP4 switch transaction', file=sys.stderr)
        raise SystemExit(1)
    print('PP4 ACTIVE ' + os.environ['FAKE_PP4_ID'])
    for node, cid in zip(('spark', 'sparkworker'), scenario['verified']):
        print(f'PP4 RUNNING VERIFIED {node} container {cid}')
    print('PP4 RUNNING VERIFIED')
    print('K1 runtime evidence: logs/k1-runtime.AbCdEfGh')
elif host == 'sparkhead' and 'docker inspect' in command:
    print('\\n'.join(scenario['after']))
else:
    print('unexpected fake ssh call', host, command, file=sys.stderr)
    raise SystemExit(9)
'''

FAKE_PY = '''import os, sys
if sys.argv[1] == '-':
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
if '--compare' in sys.argv:
    print('\\n'.join(f'{i} IDENTICAL 100' for i in range(6)))
else:
    open(sys.argv[-1], 'w').write('{}\\n')
'''


def w8_check(tmp_path: Path, **scenario: Json) -> tuple[int, str, list[list[str]], str]:
    """The ACTUAL restore-pp4.sh `check` in a sandbox: only its checkout/campaign/interpreter paths are relocated; ssh is a
    local transport fixture answering by host and command (no host contact)."""
    text = (pf.ROOT / 'offload/launch/windows/restore-pp4.sh').read_text()
    for old, new in (('cd /opt/dsv41/repos/dsv41-deploy\n', f'cd {tmp_path}\n'),
                     ('C=/opt/dsv41/campaigns/DSV41-OFFLOAD\n', f'C={tmp_path}/campaign\n'),
                     ('PY=/home/operator/miniconda3/bin/python3\n', f'PY={tmp_path}/bin/fakepy\n')):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    script = tmp_path / 'restore-pp4.sh'
    script.write_text(text)
    (tmp_path / 'campaign/K1-production').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'campaign/K1-production/greedy-post-pp4.jsonl').write_text('{"i": 0}\n')
    bindir = tmp_path / 'bin'
    bindir.mkdir(exist_ok=True)
    for name, code in (('ssh', FAKE_SSH), ('fakepy', FAKE_PY)):
        (bindir / name).write_text(f'#!{sys.executable}\n' + code)
        (bindir / name).chmod(0o755)
    ids = ['a' * 64, 'b' * 64]
    values: dict[str, Json] = {'config': True, 'worker': True, 'verified': cast(Json, ids), 'after': cast(Json, ids)}
    values.update(scenario)
    (tmp_path / 'scenario.json').write_text(json.dumps(values))
    calls = tmp_path / 'calls.jsonl'
    calls.write_text('')
    env = {**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}', 'FAKE_SCENARIO': str(tmp_path / 'scenario.json'),
           'FAKE_CALLS': str(calls), 'FAKE_PP4_ID': switch.PP4_ID}
    done = subprocess.run(['bash', str(script), 'check'], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=60)
    worker = Path(str(calls) + '.worker')
    return (done.returncode, done.stdout + done.stderr, [json.loads(x) for x in calls.read_text().splitlines()],
            worker.read_text() if worker.exists() else '')


def test_s91_5_historical_success_alone_no_longer_passes_check(tmp_path: Path) -> None:
    """S91's fixture: historical guard lines, six IDENTICAL rows and worker markers, but no current configuration."""
    rc, text, calls, _ = w8_check(tmp_path, config=False)
    assert rc != 0 and 'CHECK: DONE' not in text and 'current PP4 production configuration not verified' in text
    assert any('pp4_promote_switch.py active' in command and 'verify.sh' in command for _, command in calls)
    assert not any('k1-guard' in command for _, command in calls)


def test_s91_5_check_verifies_current_configuration_containers_and_worker(tmp_path: Path) -> None:
    rc, text, calls, worker = w8_check(tmp_path)
    assert rc == 0, text
    watch.restore_verified('check', rc, text)
    commands = [command for _, command in calls]
    assert commands.index(next(c for c in commands if 'verify.sh' in c)) < commands.index(
        next(c for c in commands if 'docker inspect' in c))
    for needle in (w.DROPIN_SHA256, 'DropInPaths', 'NeedDaemonReload', f'pp4-runtime/{switch.PP4_ID}/attest.sh',
                   '/proc/$pid/cmdline', '/proc/$pid/cwd', 'DSPARK_SAMPLED_TOP32=1',
                   f'DSPARK_SOURCE_COMMIT={switch.PP4_COMMIT}', 'PYTHONPATH=$rt/stock:$rt/facc:',
                   f'dspark-worker/{switch.PP4_COMMIT[:7]}/offload/dspark'):
        assert needle in worker, needle
    assert subprocess.run(['bash', '-n'], input=worker, text=True).returncode == 0


@pytest.mark.parametrize('scenario,reason', [
    ({'after': ['a' * 64, 'c' * 64]}, 'serving containers changed'),
    ({'verified': ['a' * 64]}, 'did not name both serving containers'),
    ({'worker': False}, 'PP4 sampled production worker not verified'),
])
def test_s91_5_container_drift_or_worker_failure_stops_check(tmp_path: Path, scenario: dict[str, Json],
                                                              reason: str) -> None:
    rc, text, _, _ = w8_check(tmp_path, **scenario)
    assert rc != 0 and reason in text and 'CHECK: DONE' not in text


# --- S91 #6: a failed post-close watch left rejected canonical PP4 references behind ------------------------------

W8_CHECK_TEXT = (''.join(f'greedy K1-post vs restored: {i} IDENTICAL 100\n' for i in range(6))
                 + 'PRODUCTION HOLD OK\nCHECK: DONE\n')


def transport(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Successful mocked transport boundaries at class level, so the rollback Live that rollback_watch builds uses them."""
    def invoke(self: w.Live, label: str, host: str, command: str, limit: float = 60) -> tuple[int, str]:
        calls.append(label)
        if label == 'commit-close':
            return 0, scoped_remote.CLOSED
        if label == 'cancel-close':
            return 0, 'SCOPED CLOSE COMMIT FENCED'
        return 0, 'PP4 QUIET' if label.startswith('quiet-') else 'HOLD OWNED'

    def owned(self: w.Live, phase: str, host: str, argv: list[str], limit: float, *,
              close_owner: str | None = None) -> tuple[int, str]:
        calls.append(phase)
        if phase.endswith('-restore-close'):
            return 0, 'CLOSE: window closed'
        if phase.startswith('rollback-clock-'):
            return 0, 'CLOCK OVERRIDE KEPT GPU_CLOCK_CAP=2300'
        return 0, {'rollback-x86host': 'PRODUCTION HOLD OK', 'restore-start': 'START: K1 START VERIFIED',
                   'restore-check': W8_CHECK_TEXT}.get(phase, '')

    def quiesce(self: w.Live, tag: str) -> None:
        calls.append('quiesce')
    monkeypatch.setattr(w.Live, 'invoke', invoke)
    monkeypatch.setattr(w.Live, 'owned', owned)
    monkeypatch.setattr(w.Live, 'quiesce', quiesce)


def published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[w.Live, Path, bytes]:
    references = tmp_path / 'K1-production/greedy-post-pp4.jsonl'
    references.parent.mkdir()
    monkeypatch.setattr(w, 'NEW_REFERENCES', references)
    backend = live(tmp_path)
    source = tmp_path / 'reference-0.jsonl'
    source.write_text(''.join(json.dumps({'i': i, 'text': f't{i}'}) + '\n' for i in range(6)))
    return backend, references, source.read_bytes()


def test_s91_6_watchdog_rollback_archives_only_this_runs_references(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """S91's reproduction: a PASSed promotion whose post-close watch fails is rolled back by rollback_watch through a
    NEW Live (run_out/watch-rollback). It now reconciles the canonical references this run published."""
    backend, references, data = published(tmp_path, monkeypatch)
    backend.publish_references(tmp_path / 'reference-0.jsonl', 6)
    assert references.read_bytes() == data and not list(references.parent.glob('.*staging'))
    calls: list[str] = []
    transport(monkeypatch, calls)
    backend.rollback_watch()
    evidence = json.loads((tmp_path / 'run/watch-rollback.json').read_text())
    assert str(evidence['status']).startswith('PASS')
    assert {'restore-start', 'restore-check', 'commit-close'} <= set(calls)
    # Coordinator clock rule: the prior W8 starts only after both Sparks keep a GPU_CLOCK_CAP override.
    assert max(calls.index('rollback-clock-' + host) for host in w.HOSTS) < calls.index('restore-start')
    assert not references.exists() and backend.reference_archive.read_bytes() == data
    assert json.loads((tmp_path / 'run/watch-rollback/rejected-references.json').read_text()) == {
        'references': str(backend.reference_archive)}


def test_s91_6_failed_promotion_reconciles_an_interrupted_publication(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    backend, references, data = published(tmp_path, monkeypatch)
    staging = references.with_name(f'.{references.name}.{backend.tag}.staging')
    backend.reference_receipt.write_text(json.dumps({'path': str(references), 'staging': str(staging),
        'sha256': hashlib.sha256(data).hexdigest(), 'rows': 6, 'run': backend.tag}))
    staging.write_bytes(data[:10])                                  # crashed mid-write, before the atomic link
    backend.opened = 1.0
    backend.deadline, backend.wall_deadline = backend.clock() + 3600, backend.wall() + 3600
    transport(monkeypatch, [])
    backend.finish(False, True)
    assert not references.exists() and not staging.exists()
    assert backend.reference_archive.with_suffix('.staging.jsonl').read_bytes() == data[:10]


@pytest.mark.parametrize('case', ['foreign', 'changed'])
def test_s91_6_foreign_or_changed_canonical_references_are_kept_and_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    backend, references, data = published(tmp_path, monkeypatch)
    if case == 'foreign':
        references.write_text('{"foreign": true}\n')
        with pytest.raises(FileExistsError):
            backend.publish_references(tmp_path / 'reference-0.jsonl', 6)    # exclusive: never overwrites
        backend.reference_receipt.unlink()
    else:
        backend.publish_references(tmp_path / 'reference-0.jsonl', 6)
        references.write_text('{"edited": true}\n')
    before = references.read_bytes()
    backend.opened = 1.0
    backend.deadline, backend.wall_deadline = backend.clock() + 3600, backend.wall() + 3600
    calls: list[str] = []
    transport(monkeypatch, calls)
    with pytest.raises(ValueError, match='foreign|changed'):
        backend.finish(False, True)
    assert references.read_bytes() == before and not backend.reference_archive.exists()
    assert 'restore-start' in calls and 'commit-close' not in calls       # prior production back, HOLD retained


def test_s91_2_carried_rule_is_reproven_from_git_not_trusted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A head that changed qualified sources (this branch's own 7483102 merges R16 and edits pp4_promote_window.py) cannot
    carry the admission even when the summary claims it; neither can a head without the qualified commit (9e84047);
    a recorded file list that disagrees with git is refused too."""
    admission = admission_receipt(tmp_path, monkeypatch)
    for head, reason in (('7483102db998683701d1cf421e3c3d12e53c46d6', 'new admission is required'),
                         ('9e84047147c58737f44a56c85e51c4175668ca37', 'not an ancestor')):
        claimed = measurement_summary(tmp_path, admission, None, None, experiment_commit=head, carried_qualification={
            'qualified_commit': switch.PP4_COMMIT, 'head': head, 'changed_files': [], 'rule': 'claimed'},
            rng_gate={'status': 'PASS'})
        with pytest.raises(ValueError, match=reason):
            w.evidence_gate(admission, claimed)
    listed = measurement_summary(tmp_path, admission, None, None, carried_qualification={
        'qualified_commit': switch.PP4_COMMIT, 'head': R_PP4_HEAD, 'changed_files': ['documents/x.md'], 'rule': 'claimed'})
    with pytest.raises(ValueError, match='another carried file list'):
        w.evidence_gate(admission, listed)


# --- S98 #1: a clock-guarded R-PP4 measurement passed whenever its WIN2 status was already INCONCLUSIVE ---------------

S98_GUARDED: list[dict[str, Json]] = [{'consistent': False, 'reason': 'caps differ: s1-P 2150 s2-B 2300'},
                                      {'consistent': False, 'reason': 'THERMAL-WATCH dial-back inside the HOLD'},
                                      {'consistent': False, 'reason': 'start not at production cap 2300'}]


def rpp4_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: dict[str, Json],
                 candidate: dict[str, float] | None) -> tuple[Path, Path, dict[str, Json]]:
    """S98's reproduction: R-PP4's real producer on its own fixtures, placed as pp4_window.main writes it."""
    from offload.launch.windows import pp4_measure_v2 as mv2
    from offload.launch.windows.tests import test_pp4_measure_v2 as rpp4
    decided_v2 = mv2.decide(rpp4.starts('production-policy', candidate=candidate), 'production-policy', clock=clock,
                            rng=rpp4.GATE, archive=None)
    series, reported = decided_v2.pop('decision_series'), decided_v2.pop('reported_series')
    admission = admission_receipt(tmp_path, monkeypatch)
    path = measurement_summary(tmp_path, admission, cast(list[dict[str, Json]], series), decided_v2,
                               reported_series=reported)
    return admission, path, decided_v2


@pytest.mark.parametrize('clock', S98_GUARDED, ids=['caps-differ', 'dial-back', 'not-production-cap'])
def test_s98_1_clock_guarded_inconclusive_measurement_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 clock: dict[str, Json]) -> None:
    admission, path, decided_v2 = rpp4_summary(tmp_path, monkeypatch, clock, None)   # candidate == control: no win
    assert decided_v2['status'] == 'INCONCLUSIVE' and decided_v2['win2_status'] == 'INCONCLUSIVE'
    assert str(decided_v2['reason']).startswith('clock guard')
    with pytest.raises(ValueError, match='clock guard fired'):
        w.evidence_gate(admission, path)


def test_s98_1_same_clock_inconclusive_still_passes_and_a_guarded_win_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    admission, path, decided_v2 = rpp4_summary(tmp_path, monkeypatch, dict(SAME_CLOCK), None)
    assert decided_v2['reason'] is None and 'win2_status' not in decided_v2
    assert record(w.evidence_gate(admission, path)['measurement'])['status'] == 'INCONCLUSIVE'
    from offload.launch.windows.tests import test_pp4_measure_v2 as rpp4
    admission, path, decided_v2 = rpp4_summary(tmp_path, monkeypatch, S98_GUARDED[0], rpp4.WIN)
    assert decided_v2['win2_status'] == 'CANDIDATE' and decided_v2['status'] == 'INCONCLUSIVE'
    with pytest.raises(ValueError, match='clock guard fired|does not recompute'):
        w.evidence_gate(admission, path)


# --- S98 #2: an accepted-close failure was rolled back after the abort bound without the recovery reserve -----------

class Clocked:
    """S98's fake backend on a HOLD-relative clock: acceptance ends just inside the abort bound, the accepted close fails
    after SUCCESS_CLOSE_S, and the rollback then needs the window's own RECOVERY_S."""

    def __init__(self) -> None:
        self.t = 0.0
        self.calls: list[tuple[str, float]] = []
        self.rollback_started = -1.0

    def preflight(self) -> None: ...
    def open_hold(self) -> None: ...

    def controls(self, label: str) -> dict[str, list[dict[str, Json]]]:
        row: dict[str, Json] = {'decode_tps': 50.0, 'ttft_s': 1.0, 'prompt_sha256': 'p'}
        return {name: [dict(row) for _ in range(w.REPEATS[name])] for name in w.CONTROLS}

    def switch(self) -> None: ...
    def gates(self) -> None: ...
    def final(self) -> None: ...

    def accept(self) -> None:
        self.t = w.ABORT_S - 1
        self.calls.append(('accept', self.t))

    def finish(self, passed: bool, touched: bool) -> None:
        self.calls.append((f'finish passed={passed}', self.t))
        if passed:
            self.t += w.SUCCESS_CLOSE_S
            raise RuntimeError('owned HOLD close unconfirmed')
        self.rollback_started = self.t
        self.t += w.RECOVERY_S

    def post_close(self, passed: bool) -> None: ...
    def rollback_watch(self) -> None: ...
    def clock_cleanup(self) -> None: ...


def test_s98_2_an_accepted_close_failure_still_keeps_the_whole_recovery_reserve() -> None:
    backend = Clocked()
    result = w.window(backend, lambda row: None)
    assert [c for c, _ in backend.calls] == ['accept', 'finish passed=True', 'finish passed=False']
    assert str(result['stop']).startswith('accepted-PP4 close failed')
    bounds = w.deadlines(0.0)
    assert backend.rollback_started > bounds['abort']                 # the late path S98 found
    assert bounds['scope'] - backend.rollback_started >= w.RECOVERY_S + w.UNPRICED_SLACK_S
    assert w.window_estimate()['late_close_rollback_hold_s'] <= w.RESTORE_S
    assert (w.ABORT_S, w.SUBMIT_S) == (59 * 60, 56 * 60)


def test_s98_2_round2_abort_bound_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """S98's arithmetic: 62 + 3 + 20.4 + 1 + 1.5 = 87.9 min > 85."""
    monkeypatch.setattr(w, 'ABORT_S', 62 * 60)
    monkeypatch.setattr(w, 'SUBMIT_S', 59 * 60)
    with pytest.raises(ValueError, match='accepted-close failure would start recovery without the whole reserve'):
        w.window_estimate()
