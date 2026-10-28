"""PP4-PROMOTE coordinator runner: promote sampled K2 drafting into production. Dry by default.

One owned HOLD: same-HOLD control on the LIVE production (596ba1c, or PROMO1's promoted commit) -> stop Spark serve ->
Spark env switch (pp4_promote_switch.py spark-apply, byte-exact snapshot) -> x86host TP2 unit drop-in through
prod_window.sh pair/close -> guarded production start (K1 START VERIFIED + PP4 RUNNING VERIFIED) -> PP4 greedy under
pp4-r14-near-tie-1p1 vs K1-post -> record NEW greedy references (three identical runs) -> same controls on PP4 -> no
decision cell LOSES under win2-v1 (decision_v2.decide_cell; NDM 2026-10-27 "Fairer win rule is essential"; a win is not
required) -> versioned W8 check (restore-pp4.sh) -> production commit + tag. Never chained: it leaves production serving.
Any failure, lost reply or STOP restores the exact prior production (Spark env + x86host unit) and its own W8.
No command runs at import or in the dry plan. Fake backends exercise every path (tests/test_pp4_promote_window.py).
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from typing import Protocol, cast
import uuid

from offload.gb10_budget import window_cells as cells
from offload.launch.windows import chain, decision_v2, hold_admission, production_health, scoped_remote, spark_quiet
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import post_close_watch as watch
from offload.launch.windows import pp4_budget as budget
from offload.launch.windows import pp4_measure_v2 as mv2
from offload.launch.windows import pp4_prerequisites as prereq
from offload.launch.windows import pp4_promote_package as package
from offload.launch.windows import pp4_promote_switch as switch
from offload.prefill_maxlen.evidence import Json, number, record

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = pf.CAMPAIGN
NEW_RESTORE = ROOT / 'offload/launch/windows/restore-pp4.sh'
K1_POST = CAMPAIGN / 'K1-production/greedy-post.jsonl'
NEW_REFERENCES = CAMPAIGN / 'K1-production/greedy-post-pp4.jsonl'
GREEDY_CHECK = CAMPAIGN / 'W5/greedy_check.py'
SAMPLED_CONFIG = ROOT / 'offload/launch/windows/PP4-sampled.env'
POLICY_CONFIG = ROOT / 'offload/launch/windows/PP4-production-policy.env'
# Measurement contract (R-PP4 v2, PP4-PROMOTE-GATE-V2.md; coordinator 2026-10-27): only the production-policy mode
# (production's adaptive policy vs PP4 fixed5 sampled on the same PP4 runtime) is promotion evidence. short + 32K B1
# decide; per cell the primary is pooled tokens per server decode second, then the accepted/round and client tok/s
# guards. ms/round (PP4 pays ~+0.31 ms/round by design to buy ~+6.9 % acceptance) and TTFT are reported only.
# Every series is recorded so the decision recomputes; the registered sets and orientation are fixed here, not trusted.
MEASUREMENT_MODE = 'production-policy'
MEASUREMENT_DECISION_CELLS = budget.B1_CELLS
MEASUREMENT_METRICS = ('pooled_rate_tps', 'accepted_per_round', 'decode_tps')   # decision_series per cell, primary first
REPORTED_METRICS = ('ms_per_round', 'ttft_s')                                    # reported_series: present, never decide
ORIENTATION = {'pooled_rate_tps': True, 'accepted_per_round': True, 'ms_per_round': False, 'ttft_s': False,
               'decode_tps': True}
IDENTITY_KEYS = {'VLLM_DSPARK_REMOTE_IDENTITY': 'worker_identity', 'VLLM_DSPARK_REMOTE_MARKOV_IDENTITY': 'markov_identity'}
# win2-v1: two DECISION cells, three repeats per arm (the rule's minimum); 150K is reported once per arm, never decides.
DECISION_CELLS = ('ctl-short-b1', 'ctl-32k-b1')
REPORTED_CELLS = ('ctl-150k-b1',)
CONTROLS = (*DECISION_CELLS, *REPORTED_CELLS)
REPEATS = {'ctl-short-b1': decision_v2.MIN_REPEATS, 'ctl-32k-b1': decision_v2.MIN_REPEATS, 'ctl-150k-b1': 1}
# First metric is the cell's primary series; the guard can only cause LOSS. Acceptance and ms/round are not compared:
# production's adaptive window and PP4's fixed5 draft different round shapes, so per-round metrics are not like for like.
METRICS = (('decode_tps', True), ('ttft_s', False))
REFERENCE_RUNS = 3
HOSTS = ('sparkhead', 'sparkworker-wired')
URL = production_health.PRODUCTION
TUNNEL = 'http://127.0.0.1:18001'
PAIR_OWNER = 'PP4PROMO'
HOOKED = package.HOOKS_PATHS
# MEASURED: PF2 DESIGN section 8 arm starts 6.5-6.8 min and W8 restore 7.4 min; PC1-chunk2048-W-20261027T112828Z W8
# check 20 s (cmd restore-0112..0114) and three PC1 starts 405-421 s. Everything else is an ESTIMATE, never evidence.
MEASURED_START_S, MEASURED_W8_S, MEASURED_W8_CHECK_S = 6.8 * 60, 7.4 * 60, 20
WORKER_SWAP_S = 10 * 60      # ESTIMATE: pair + drop-in + prod_window close ready bound 600 s (PP4 startup600s bound)
FINAL_W8_CHECK_S = 2 * 60    # ESTIMATE: measured 20 s check + fresh Spark/worker configuration verification
ACCEPT_S = 60                # ESTIMATE: one commit + tag on sparkhead
SUCCESS_CLOSE_S = 3 * 60     # ESTIMATE: accepted W8 check + close + owned HOLD close (after acceptance)
# Recovery from the abort bound (S91 #3): Spark stop/quiet/env restore, worker swap back (historical MEASURED 333 s
# worker start, pp4_admission), the prior W8 start/check/close (MEASURED 7.4 min), fencing + owned close.
RECOVERY_S = 3 * 60 + 6 * 60 + MEASURED_W8_S + 4 * 60
UNPRICED_SLACK_S = 60        # explicit margin for unpriced delay between the abort and recovery's first command
FINAL_CLOSE_RESERVE_S = 90   # owned scopes expire this long before HOLD+85 so the final close can still retain HOLD
RESTORE_S = 85 * 60
ABORT_S = 59 * 60            # forward work stops here; even an accepted close that fails after it keeps the reserve (S98)
SUBMIT_S = ABORT_S - FINAL_W8_CHECK_S - ACCEPT_S   # last cell/reference submission still leaves the final check
DROPIN_TEXT = switch.render_dropin()
DROPIN_SHA256 = hashlib.sha256(DROPIN_TEXT.encode()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def deadlines(opened: float) -> dict[str, float]:
    """Absolute HOLD-relative bounds used by open_hold (and recomputed by finish for an uncertain open)."""
    return {'submit': opened + SUBMIT_S, 'abort': opened + ABORT_S, 'scope': opened + RESTORE_S - FINAL_CLOSE_RESERVE_S,
            'restore': opened + RESTORE_S}


def window_estimate() -> dict[str, float]:
    """Admission guard; refuses before any host contact.

    Separate invariants (S91 #3, S98 #2), not one aggregate: forward work fits before the abort bound; a failure AT the
    abort bound, and an accepted close that fails after it, still have the whole recovery reserve, slack and final-close
    reserve before HOLD+85; the submission deadline leaves the final W8 check and acceptance before the abort bound."""
    inputs = {name: cells.CELLS[name].estimate_s for name in CONTROLS}
    require(all(math.isfinite(v) and v > 0 for v in inputs.values()), 'window estimate: cell costs must be positive')
    forward = {f'controls-{name}': 2 * REPEATS[name] * inputs[name] for name in CONTROLS}
    forward.update(stop_quiet_switch=4 * 60, worker_swap=WORKER_SWAP_S, guarded_start=MEASURED_START_S + 72,
                   pp4_greedy=4 * 60, new_references=REFERENCE_RUNS * 60, final_w8_check=FINAL_W8_CHECK_S,
                   accept=ACCEPT_S)
    forward_s = sum(forward.values())
    require(math.isfinite(forward_s) and forward_s <= ABORT_S,
            f'forward work ESTIMATED {forward_s / 60:g}min exceeds the abort bound {ABORT_S / 60:g}min; refused')
    worst = ABORT_S + RECOVERY_S + UNPRICED_SLACK_S + FINAL_CLOSE_RESERVE_S
    require(worst <= RESTORE_S, f'recovery reserve violated: abort {ABORT_S / 60:g} + recovery {RECOVERY_S / 60:g} + '
            f'slack {UNPRICED_SLACK_S / 60:g} + final close {FINAL_CLOSE_RESERVE_S / 60:g} min > {RESTORE_S / 60:g} min')
    require(SUBMIT_S + FINAL_W8_CHECK_S + ACCEPT_S <= ABORT_S, 'submission deadline leaves no final check/acceptance')
    require(ABORT_S + SUCCESS_CLOSE_S + FINAL_CLOSE_RESERVE_S <= RESTORE_S, 'accepted close does not fit the HOLD')
    # S98 #2: acceptance may end just inside the abort bound and its close (no abort bound) may then fail, which
    # window() rolls back; that recovery starts up to SUCCESS_CLOSE_S later and still needs the whole reserve.
    late = ABORT_S + SUCCESS_CLOSE_S + RECOVERY_S + UNPRICED_SLACK_S + FINAL_CLOSE_RESERVE_S
    require(late <= RESTORE_S, 'an accepted-close failure would start recovery without the whole reserve')
    return {**forward, 'forward_s': forward_s, 'success_close_s': SUCCESS_CLOSE_S, 'recovery_reserve_s': RECOVERY_S,
            'unpriced_slack_s': UNPRICED_SLACK_S, 'final_close_reserve_s': FINAL_CLOSE_RESERVE_S,
            'success_hold_s': forward_s + SUCCESS_CLOSE_S, 'failure_at_abort_hold_s': worst,
            'late_close_rollback_hold_s': late, 'total_s': max(worst, late)}


def canonical(value: Json) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def artifacts_intact(directory: Path, listed: Json, what: str) -> int:
    """Every primary artifact a receipt lists still exists beside it with the recorded digest."""
    hashes = record(listed)
    require(bool(hashes), f'{what}: primary artifact hashes absent')
    for name, digest in hashes.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, f'{what}: artifact {name!r} escapes its run')
        path = directory / relative
        require(path.is_file() and pf.digest(path) == digest, f'{what}: primary artifact {name} absent/changed')
    return len(hashes)


def admission_gate(path: Path) -> dict[str, Json]:
    """The existing PP4 qualification validation (schema, 12 gates with primary evidence hashes, runtime, physical pair
    via pp4_budget.admitted, identities), bound to exactly the bytes this switch promotes."""
    data = prereq.qualification(path, switch.PP4_COMMIT)
    require(record(data['verifier_sha256']) == switch.VERIFIER_SHA256,
            'admission verifier pins differ from the promoted ordinary verifier')
    require(record(data['runtime_artifact_sha256']) == switch.SO_SHA256,
            'admission runtime binaries differ from the attested PP4 runtime')
    artifacts = artifacts_intact(path.parent, data.get('artifact_sha256', {}), 'admission')
    return {'verdict': 'QUALIFIED', 'source_commit': switch.PP4_COMMIT, 'sha256': pf.digest(path), 'artifacts': artifacts,
            'worker_identity': data['worker_identity'], 'markov_identity': data['markov_identity']}


def measurement_series(value: Json) -> list[decision_v2.Series]:
    result: list[decision_v2.Series] = []
    for row in [record(r) for r in value] if isinstance(value, list) else []:
        cell, metric, higher, paired = row.get('cell'), row.get('metric'), row.get('higher_is_better'), row.get('paired')
        require(isinstance(cell, str) and isinstance(metric, str) and isinstance(higher, bool) and isinstance(paired, bool),
                'measurement series row malformed')
        require(metric in ORIENTATION and ORIENTATION[str(metric)] is higher, f'measurement {cell}/{metric}: orientation')
        result.append(decision_v2.Series(str(cell), str(metric), bool(higher),
                                         tuple(number(v) for v in cast(list[Json], row['control'])),
                                         tuple(number(v) for v in cast(list[Json], row['candidate'])), bool(paired)))
    require(bool(result), 'measurement records no series')
    keys = [(s.cell, s.metric) for s in result]
    require(len(keys) == len(set(keys)), 'measurement repeats a cell/metric series')
    return result


def measurement_gate(path: Path, admission: Path, admission_sha256: str) -> dict[str, Json]:
    """A COMPLETE production-policy v2 measurement that carries this admission's qualification to its own head, and whose
    recorded decision recomputes from its own series with no decision-cell LOSS (a win is not required)."""
    data = record(cast(Json, json.loads(path.read_text())))
    require(data.get('measure_v2') == MEASUREMENT_MODE, 'promotion evidence must be the production-policy v2 measurement')
    require(data.get('status') == 'COMPLETE' and not data.get('stop'), 'PP4 measurement not COMPLETE')
    require(data.get('qualification_sha256') == admission_sha256, 'PP4 measurement does not name this admission receipt')
    head = str(data.get('experiment_commit'))
    carried_record = record(data.get('carried_qualification') or {})
    require(data.get('runtime_commit') == switch.PP4_COMMIT and carried_record.get('qualified_commit') == switch.PP4_COMMIT
            and carried_record.get('head') == head, 'PP4 measurement does not carry the admitted source ' + switch.PP4_COMMIT)
    # Re-prove the carried rule now from git (allowlist + runner unit ASTs), never from the recorded file list.
    carried = mv2.carried(admission, switch.PP4_COMMIT, head, MEASUREMENT_MODE)
    require(record(carried['carried_qualification'])['changed_files'] == carried_record.get('changed_files'),
            'PP4 measurement records another carried file list than git shows')
    require(data.get('config_sha256') == pf.digest(SAMPLED_CONFIG), 'measured PP4 config differs from the promoted config')
    require(data.get('production_policy_config_sha256') == pf.digest(POLICY_CONFIG),
            'measured production-policy config differs from the registered one')
    rng = record(data.get('rng_gate') or {})
    receipt = Path(str(rng.get('receipt', '')))
    require(rng.get('status') == 'PASS' and rng.get('commit') == head
            and rng.get('pp4_kernel_sha256') == switch.VERIFIER_SHA256[switch.UTILS]
            and receipt.is_absolute() and receipt.is_file() and pf.digest(receipt) == rng.get('receipt_sha256'),
            '#61161 verifier-RNG gate receipt absent/changed or not for this kernel')
    artifacts = artifacts_intact(path.parent, data.get('artifact_sha256', {}), 'measurement')
    decision = record(data['decision'])
    require(decision.get('rule') == decision_v2.RULE, f'PP4 measurement not scored under {decision_v2.RULE}')
    # S98 #1: R-PP4's clock guard (caps differ, a start not at production's cap, a dial-back inside the HOLD) only
    # overwrites status with INCONCLUSIVE, which equals the recomputed status whenever the run had no win; read the guard.
    require(decision.get('reason') is None and 'win2_status' not in decision
            and record(decision.get('clock') or {}).get('consistent') is True,
            f"PP4 measurement clock guard fired ({decision.get('reason')}); not same-clock promotion evidence")
    cells_recorded = [str(c) for c in cast(list[Json], decision['decision_cells'])]
    require(sorted(cells_recorded) == sorted(MEASUREMENT_DECISION_CELLS),
            f'PP4 measurement decision cells {cells_recorded} differ from {list(MEASUREMENT_DECISION_CELLS)}')
    series = measurement_series(data.get('decision_series'))
    reported = measurement_series(data.get('reported_series'))
    for cell in MEASUREMENT_DECISION_CELLS:
        got = [s.metric for s in series if s.cell == cell]
        require(got == list(MEASUREMENT_METRICS), f'PP4 measurement {cell}: decision series {got} differ (primary first)')
        missing = sorted(set(REPORTED_METRICS) - {s.metric for s in reported if s.cell == cell})
        require(not missing, f'PP4 measurement {cell}: reported series {missing} absent')
    require(not {s.metric for s in reported} & set(MEASUREMENT_METRICS), 'a deciding metric is filed as reported-only')
    recomputed = decision_v2.decide_window(series, cells_recorded)
    for key in ('rule', 'status', 'min_gain', 'decision_cells', 'cells'):
        require(canonical(recomputed[key]) == canonical(decision[key]), f'PP4 measurement decision {key} does not recompute')
    lost = [c for c in MEASUREMENT_DECISION_CELLS if record(record(recomputed['cells'])[c])['verdict'] == 'LOSS']
    require(not lost and recomputed['status'] != 'NOT A CANDIDATE', f'PP4 measurement LOSES {decision_v2.RULE} cell(s) {lost}')
    return {'status': recomputed['status'], 'rule': decision_v2.RULE, 'sha256': pf.digest(path), 'artifacts': artifacts,
            'decision_cells': cast(Json, cells_recorded), 'mode': MEASUREMENT_MODE, 'experiment_commit': head,
            'carried_files': carried_record.get('changed_files')}


def evidence_gate(admission: Path, measurement: Path) -> dict[str, Json]:
    """Qualified hardware evidence only: the admitted source, its primary evidence, and a no-LOSS measurement of it.

    A win is not required (decision_v2: CANDIDATE or INCONCLUSIVE); any decision-cell LOSS refuses."""
    try:
        admitted = admission_gate(admission)
        measured = measurement_gate(measurement, admission, str(admitted['sha256']))
    except (KeyError, TypeError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        raise ValueError(f'PP4 evidence malformed or unprovable: {type(exc).__name__}: {exc}') from exc
    return {'admission': admitted, 'measurement': measured}


def bind_identities(evidence: dict[str, Json], identities: list[dict[str, Json]]) -> None:
    """The live production ranks already carry the admitted K2 checkpoint identities (the switch never changes them)."""
    admitted = record(evidence['admission'])
    require(len(identities) == len(HOSTS), 'both production ranks required')
    for identity in identities:
        env = record(identity['env'])
        for key, field in IDENTITY_KEYS.items():
            require(env.get(key) == admitted[field], f'production {key} differs from the admitted {field}')


def regression(control: dict[str, list[dict[str, Json]]], pp4: dict[str, list[dict[str, Json]]]) -> dict[str, Json]:
    """G6: PP4 must not LOSE any decision cell to the same-HOLD production control.

    LOSS is whatever decision_v2.decide_window/decide_cell rule this tree carries (primary series or its guard); no
    threshold is restated here. The arms run one after the other in the HOLD, so repeats are unpaired. WIN and
    INCONCLUSIVE both pass: the promotion needs no loss, not a win.
    """
    require(set(control) == set(pp4) == set(CONTROLS), 'G6: control set incomplete')
    series: list[decision_v2.Series] = []
    for name in CONTROLS:
        before, after = control[name], pp4[name]
        require(len(before) == len(after) == REPEATS[name], f'G6: {name} needs exactly {REPEATS[name]} repeats per arm')
        hashes = [r['prompt_sha256'] for r in before + after]
        require(all(h == hashes[0] for h in hashes), f'G6: {name} prompt drift')
        for key, higher_is_better in METRICS:
            series.append(decision_v2.Series(name, key, higher_is_better, tuple(number(r[key]) for r in before),
                                             tuple(number(r[key]) for r in after), paired=False))
    result = decision_v2.decide_window(series, DECISION_CELLS)
    cells_result = record(result['cells'])
    lost = [name for name in DECISION_CELLS if record(cells_result[name])['verdict'] == 'LOSS']
    require(not lost and result['status'] != 'NOT A CANDIDATE', f'G6: PP4 LOSES {decision_v2.RULE} decision cell(s) {lost}')
    return result


def references_identical(paths: list[Path]) -> list[dict[str, Json]]:
    runs = [[record(cast(Json, json.loads(line))) for line in p.read_text().splitlines()] for p in paths]
    require(len(runs) == REFERENCE_RUNS and all(len(r) == 6 for r in runs), 'three complete six-prompt runs required')
    for run in runs:
        require([row.get('i') for row in run] == list(range(6)) and all(isinstance(row.get('text'), str) for row in run),
                'ordered six-prompt reference rows required')
    first = [row['text'] for row in runs[0]]
    require(all([row['text'] for row in run] == first for run in runs[1:]),
            'PP4 greedy outputs are not reproducible across three runs; no reference recorded')
    return runs[0]


def identity_gate(before: dict[str, Json], after: dict[str, Json]) -> None:
    """Live PP4 rank == saved production rank except exactly the registered PP4 env, mounts and verify window."""
    require(before['id'] != after['id'], 'PP4 reused the control container')
    for key in ('image', 'engine', 'files', 'engram_identity'):
        require(before[key] == after[key], f'runtime drift: {key}')
    expected_env = {k: v for k, v in record(before['env']).items() if k not in switch.PP4_KEYS}
    expected_env.update(dict(switch.PP4_ENV))
    require(record(after['env']) == expected_env, 'PP4 runtime environment differs from the registered switch')
    mounted = {k: v for k, v in record(before['mounted']).items()
               if k != '/serve.sh' and not k.startswith(pf.DIST + switch.REMOTE + '/')}
    mounted.update(prereq.overlay_pins({'verifier_sha256': cast(Json, switch.VERIFIER_SHA256)}))
    require({k: v for k, v in record(after['mounted']).items() if k != '/serve.sh'} == mounted,
            'PP4 mounted engine bytes differ from production plus the qualified PP4 overlay')
    original = shlex.split(str(before['argv']).replace('\\\n', ' '))
    changed = shlex.split(str(after['argv']).replace('\\\n', ' '))
    require(original.count('--speculative-config') == 1, 'production argv lacks one speculative config')
    idx = original.index('--speculative-config') + 1
    require(len(changed) == len(original) and changed[idx] == switch.SPEC, 'PP4 argv lacks the fixed5 verify window')
    changed[idx] = original[idx]
    require(changed == original, 'PP4 argv changed beyond the fixed5 verify window')


class Backend(Protocol):
    def preflight(self) -> None: ...
    def open_hold(self) -> None: ...
    def controls(self, label: str) -> dict[str, list[dict[str, Json]]]: ...
    def switch(self) -> None: ...
    def gates(self) -> None: ...
    def final(self) -> None: ...
    def accept(self) -> None: ...
    def finish(self, passed: bool, touched: bool) -> None: ...
    def post_close(self, passed: bool) -> None: ...
    def rollback_watch(self) -> None: ...
    def clock_cleanup(self) -> None: ...


def window(backend: Backend, checkpoint: Callable[[dict[str, Json]], None]) -> dict[str, Json]:
    result: dict[str, Json] = {'verdict': 'FAIL'}
    attempted = touched = passed = False
    try:
        backend.preflight()
        attempted = True                    # an uncertain HOLD acknowledgement still requires fencing/recovery
        backend.open_hold()
        control = backend.controls('control')
        touched = True                      # a lost stop/switch/start reply must still restore
        backend.switch()
        backend.gates()                     # identity, PP4 greedy (1.1 nat), new references
        pp4 = backend.controls('pp4')
        result['G6'] = regression(control, pp4)
        backend.final()                     # versioned W8 check against the NEW references
        backend.accept()
        passed = True
        checkpoint({'gates': 'PASS'})
    except (Exception, KeyboardInterrupt) as exc:
        passed = False
        result['stop'] = f'{type(exc).__name__}: {exc}'
        checkpoint({'stop': result['stop']})
    finally:
        if attempted:
            with watch.masked_signals():
                try:
                    backend.finish(passed, touched)
                    result['close'] = 'PASS'
                except (Exception, KeyboardInterrupt) as exc:
                    result['close'] = f'FAILED: {type(exc).__name__}: {exc}; HOLD retained'
                    if passed:
                        try:                # an accepted-production close failure is a failed promotion
                            backend.finish(False, touched)
                            result['close'] = 'PASS'
                            result['stop'] = f'accepted-PP4 close failed: {exc}; rolled back'
                        except (Exception, KeyboardInterrupt) as rollback:
                            result['rollback_stop'] = f'{type(rollback).__name__}: {rollback}; HOLD retained'
                    passed = False
    if result.get('close') == 'PASS':
        try:
            backend.post_close(passed)
        except (Exception, KeyboardInterrupt) as exc:
            result['watch'] = f'FAILED: {exc}'
            try:
                checkpoint({'watch': result['watch']})
            except (Exception, KeyboardInterrupt) as evidence_error:
                result['watch_checkpoint'] = f'FAILED: {evidence_error}'
            if passed:
                with watch.masked_signals():
                    try:
                        backend.rollback_watch()
                        result['watch_rollback'] = 'PASS: prior production restored'
                    except (Exception, KeyboardInterrupt) as rollback:
                        result['watch_rollback'] = f'FAILED: {rollback}; HOLD retained/manual recovery required'
            passed = False
        else:
            result['watch'] = 'PASS'
            if passed:
                # Repo plan Phase 0: the folded 2300 default now carries the clock; drop the redundant override. A failure
                # leaves the cap unchanged (the line equals the default), so it never rolls the promotion back.
                try:
                    backend.clock_cleanup()
                    result['clock_cleanup'] = 'PASS: GPU_CLOCK_CAP=2300 override removed on both Sparks'
                except (Exception, KeyboardInterrupt) as exc:
                    result['clock_cleanup'] = f'FAILED: {type(exc).__name__}: {exc}; redundant 2300 line may remain'
    result['verdict'] = 'PASS: production left on PP4 sampled drafting' if passed else 'FAIL: see stop/close/watch evidence'
    return result


class StrictSupervisor(pf.Supervisor):
    """Production :8000 and strict zero swap growth (PROMO1's supervisor)."""
    def metrics(self) -> float:
        import urllib.request
        with urllib.request.urlopen(URL + '/metrics', timeout=3) as response:
            return production_health.requests(response.read().decode())

    def poll(self) -> None:
        super().poll()
        for host, swap in self.ready.items():
            require(self.swap.get(host, -1) >= swap, f'G8: {host} swap grew after readiness')


def x86host_apply_script() -> str:
    path = '$HOME/' + switch.DROPIN
    return '\n'.join((
        'set -euo pipefail',
        f'bash ~/dsv41-offload/prod_window.sh pair {PAIR_OWNER}',
        f'test "$(head -1 ~/dsv41-offload/PAIR_OK)" = {PAIR_OWNER}',
        f'sh ~/dsv41-offload/prod/pp4-runtime/{switch.PP4_ID}/attest.sh',
        f'test -d ~/dsv41-offload/prod/dspark-worker/{switch.PP4_COMMIT[:7]}/offload/dspark/worker',
        f'mkdir -p "$HOME/{switch.DROPIN_DIR}" && test ! -e "{path}"',
        f'printf %s {shlex.quote(DROPIN_TEXT)} > "{path}.tmp"',
        f'test "$(sha256sum < "{path}.tmp" | cut -d" " -f1)" = {DROPIN_SHA256} && mv "{path}.tmp" "{path}"',
        'systemctl --user daemon-reload',
        'PROD_WINDOW_READY_S=600 bash ~/dsv41-offload/prod_window.sh close',
        'bash ~/dsv41-offload/prod_window.sh check'))


def x86host_rollback_script() -> str:
    path = '$HOME/' + switch.DROPIN
    return '\n'.join((
        'set -euo pipefail',
        'P=~/dsv41-offload/PAIR_OK',
        f'if [ -e "$P" ]; then test "$(head -1 "$P")" = {PAIR_OWNER} || {{ echo "foreign pair window" >&2; exit 3; }}',
        f'elif [ -e "{path}" ] || [ -e "{path}.tmp" ]; then bash ~/dsv41-offload/prod_window.sh pair {PAIR_OWNER}; fi',
        f'rm -f -- "{path}.tmp"',
        f'if [ -e "{path}" ]; then test "$(sha256sum < "{path}" | cut -d" " -f1)" = {DROPIN_SHA256} '
        '|| { echo "foreign PP4 drop-in; refused" >&2; exit 3; }; rm -f -- "' + path + '"; fi',
        'systemctl --user daemon-reload',
        'if [ -e "$P" ]; then bash ~/dsv41-offload/prod_window.sh close; fi',
        'bash ~/dsv41-offload/prod_window.sh check'))


ENVIRONMENT = '/etc/environment'
CLOCK_OVERRIDE = 'GPU_CLOCK_CAP=2300'


def clock_cleanup_script(path: str = ENVIRONMENT) -> str:
    """After a PASSed watch: remove exactly the 2300 override; an explicit dial-back value (e.g. 2150) stays."""
    return '\n'.join((
        'set -euo pipefail',
        f"if grep -qx '{CLOCK_OVERRIDE}' {path}; then sudo -n sed -i '/^{CLOCK_OVERRIDE}$/d' {path}; fi",
        f"! grep -qx '{CLOCK_OVERRIDE}' {path}",
        f"echo \"CLOCK OVERRIDE CLEARED $(grep '^GPU_CLOCK_CAP=' {path} || echo 'none: folded default 2300')\""))


def clock_keep_script(path: str = ENVIRONMENT) -> str:
    """Rollback, before the prior W8 start: a GPU_CLOCK_CAP line must exist (2300 unless an explicit dial-back is there),
    so a rollback to a pre-fold production keeps the NDM 2300 cap."""
    return '\n'.join((
        'set -euo pipefail',
        f"if ! grep -q '^GPU_CLOCK_CAP=' {path}; then echo '{CLOCK_OVERRIDE}' | sudo -n tee -a {path} >/dev/null; fi",
        f"echo \"CLOCK OVERRIDE KEPT $(grep '^GPU_CLOCK_CAP=' {path})\""))


WORKER_PROBE = f"""import json, os, subprocess
pid = subprocess.run(['systemctl', '--user', 'show', '-p', 'MainPID', '--value', {switch.UNIT!r}],
                     check=True, capture_output=True, text=True).stdout.strip()
p = '/proc/' + pid
env = dict(x.split('=', 1) for x in open(p + '/environ').read().split('\\0') if '=' in x)
print(json.dumps({{'pid': pid, 'cmd': open(p + '/cmdline').read().split('\\0'), 'cwd': os.readlink(p + '/cwd'),
                  'sampled': env.get('DSPARK_SAMPLED_TOP32'), 'pythonpath': env.get('PYTHONPATH'),
                  'source': env.get('DSPARK_SOURCE_COMMIT'),
                  'status': json.load(open(os.path.expanduser('~/dsv41-offload/prod/run/dspark-worker.status.json')))}}))
"""


def worker_gate(probe: dict[str, Json], now: float) -> None:
    home_release = f'/dsv41-offload/prod/dspark-worker/{switch.PP4_COMMIT[:7]}/offload/dspark'
    argv = [str(x) for x in cast(list[Json], probe['cmd']) if x]
    require(bool(argv) and argv[0] == switch.PYTHON and '-m' in argv and argv[argv.index('-m') + 1] == 'worker.server',
            'PP4 unit runs an unqualified interpreter or entry point')
    require(argv[argv.index('--tp') + 1] == '2' and argv[argv.index('--max-model-len') + 1] == '1048576'
            and argv[argv.index('--ofl') + 1] == '10.202.28.1:18620', 'PP4 unit HELLO/listener bounds differ')
    require(probe.get('sampled') == '1' and probe.get('source') == switch.PP4_COMMIT, 'PP4 unit sample mode/source differs')
    require(str(probe.get('cwd', '')).endswith(home_release), 'PP4 unit cwd is not the staged release')
    runtime = f'/dsv41-offload/prod/pp4-runtime/{switch.PP4_ID}'
    require(re.fullmatch(rf'/[^:]+{runtime}/stock:/[^:]+{runtime}/facc:/[^:]+{home_release}', str(probe.get('pythonpath')))
            is not None, 'PP4 unit PYTHONPATH is not the attested runtime + release')
    status = record(probe['status'])
    require(status.get('tp') == 2 and status.get('tp_comm_errors') == 0, 'PP4 unit TP failure')
    stamp = datetime.datetime.fromisoformat(str(status['time']).replace('Z', '+00:00')).timestamp()
    require(abs(now - stamp) < 30, 'PP4 unit status stale')


class Live(watch.ScopedRecovery):
    """PROMO1's shared-scope pattern: every mutation registers its phase before launch; serial exclusive logs."""

    def __init__(self, tag: str, out: Path, commit: str, rollback_restore: Path, rollback_sha256: str,
                 admission: Path, measurement: Path) -> None:
        super().__init__(tag, out / 'restore-commands')
        self.run_out, self.commit = out, commit
        self.rollback_restore, self.rollback_sha256 = rollback_restore, rollback_sha256
        self.admission_path, self.measurement_path = admission, measurement
        self.sup = StrictSupervisor(out)
        self.hold_tag, self.token = tag + '-hold-open', uuid.uuid4().hex
        self.opened = 0.0
        self.ready_at = 0.0
        self.deadline = self.clock() + 120 * 60
        self.wall_deadline = self.wall() + 120 * 60
        self.submit = math.inf
        self.phase_serial = 0
        self.close_serial = 0
        self.capture_backend = pf.Live(out, self.sup, remote='dsv41-deploy', expected=commit,
            data_dir=Path('/opt/dsv41/data/bench-data'), tag=tag, stage_bundle=False)
        self.original_identity: list[dict[str, Json]] = []
        self.expected_new_sha = hashlib.sha256(NEW_RESTORE.read_bytes()).hexdigest()
        self.watch_recovery: watch.ScopedRecovery | None = None
        # Reference ownership (S91 #6): this run's receipt and where its rejected references go. A watchdog rollback
        # Live inherits both, so it reconciles the canonical file this run published.
        self.reference_receipt = out / 'new-references.json'
        self.reference_archive = out / 'greedy-post-pp4.rejected.jsonl'

    def command(self, label: str, host: str, script: str, *, mutate: bool = False, timeout: float = 120,
                inflight: int = 0) -> str:
        """`inflight`: requests this command itself keeps on production at once; any further demand is foreign."""
        if mutate:
            self.phase_serial += 1
            phase = self.tag + '-' + label + '-' + str(self.phase_serial)
            self.phases.append(phase)
            script = scoped_remote.command('launch', phase, ['bash', '-c', script], deadline=self.wall_deadline)
        argv = ['bash', '-c', script] if host == 'localhost' else [*pf.SSH, host, script]
        return self.sup.run(label, argv, timeout=timeout, inflight=inflight)

    def owned(self, phase: str, host: str, argv: list[str], limit: float, *,
              close_owner: str | None = None) -> tuple[int, str]:
        if phase != 'restore-close' and not phase.endswith('-restore-close'):
            self.phase_serial += 1
            phase += '-' + str(self.phase_serial)
        return super().owned(phase, host, argv, limit,
                             close_owner=self.tag if phase.endswith('-restore-close') else close_owner)

    def preflight(self) -> None:
        window_estimate()
        require(not chain.enabled() and not chain.last() and chain.read_marker() is None,
                'a promotion never runs chained (WINDOW_CHAIN / chain marker): it must start from and leave production serving')
        require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator only')
        require((CAMPAIGN / 'D2-GEN.PAUSE').exists(), 'D2 pause marker required (train windows are unqualified)')
        require(re.fullmatch('[0-9a-f]{40}', self.commit) is not None, 'full production package commit required')
        require(hashlib.sha256(self.rollback_restore.read_bytes()).hexdigest() == self.rollback_sha256,
                'rollback W8 differs from its pinned sha256')
        require(K1_POST.is_file() and not NEW_REFERENCES.exists(), 'K1-post present and PP4 references absent required')
        evidence = evidence_gate(self.admission_path, self.measurement_path)
        (self.run_out / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
        self.sup.begin()
        self.command('ledger', 'sparkhead', 'cd ~/dsv41-deploy && test ! -e HOLD && '
            f'test "$(git rev-parse HEAD)" = {self.commit} && test -z "$(git status --porcelain --untracked-files=no)" && '
            'test ! -e logs/pp4-switch && profile=$(python3 -c "import json;print(json.load(open(\'logs/k1-promotion/files.json\'))[\'profile\'])") && '
            "grep -qF '" + switch.CLOCK_DEFAULT + "' dsv41.env && test -s " + switch.FOLD + '/transaction.json && '
            'python3 offload/promote/k1/manage.py verify --profile "$profile" && '
            f'grep -c PP4-PROMOTE {" ".join(HOOKED)} | grep -c ":[1-9]" | grep -qx {len(HOOKED)}')
        text = self.command('pin-switch', 'sparkhead', f'cd ~/dsv41-deploy && sha256sum {"offload/launch/windows/pp4_promote_switch.py"}')
        require(text.split()[0] == pf.digest(ROOT / 'offload/launch/windows/pp4_promote_switch.py'), 'production switch tool drift')
        for host in HOSTS:
            self.command('staged-' + host, host, f'cd ~/dsv41-offload-prod/pp4/{switch.PP4_ID} && sha256sum -c --quiet SHA256SUMS && '
                         + ' && '.join(f'test "$(sha256sum verifier/{name} | cut -d" " -f1)" = {digest}'
                                       for name, digest in switch.VERIFIER_SHA256.items()))
        self.command('staged-x86host', 'x86host', f'sh ~/dsv41-offload/prod/pp4-runtime/{switch.PP4_ID}/attest.sh && '
                     f'test -d ~/dsv41-offload/prod/dspark-worker/{switch.PP4_COMMIT[:7]}/offload/dspark/worker && '
                     f'test "$(cat ~/dsv41-offload/prod/dspark-worker/{switch.PP4_COMMIT[:7]}/SOURCE_COMMIT)" = {switch.PP4_COMMIT} && '
                     f'test ! -e "$HOME/{switch.DROPIN}" && test ! -e ~/dsv41-offload/PAIR_OK && test ! -e ~/dsv41-offload/d2/TRAIN_OK')
        production_health.ProductionHealth().probe()
        self.sup.mark_ready()
        worker = self.command('worker', 'x86host', 'bash ~/dsv41-offload/prod_window.sh check')
        require('PRODUCTION HOLD OK' in worker.splitlines(), 'production worker not verified')
        self.original_identity = [self.capture_backend.capture(host, 'control') for host in HOSTS]
        bind_identities(evidence, self.original_identity)
        clock = {host: self.command('clock-' + host, host, "grep '^GPU_CLOCK_CAP=' /etc/environment || echo GPU_CLOCK_CAP=unset")
                 .strip().splitlines()[-1] for host in HOSTS}
        require(len(set(clock.values())) == 1, f'Spark clock overrides differ {clock}; one cap for both arms required')
        evidence['clock'] = {'fold': switch.CLOCK_FOLD_COMMIT, 'default': switch.CLOCK_DEFAULT, 'etc_environment': cast(Json, clock)}
        (self.run_out / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
        (self.run_out / 'rollback-identity.json').write_text(json.dumps(self.original_identity, indent=2) + '\n')

    def open_hold(self) -> None:
        self.wall_deadline = self.wall() + 21 * 60 + RESTORE_S
        self.phases.append(self.hold_tag)
        script = hold_admission.open_script(self.hold_tag, self.token, 'PP4-PROMOTE ' + self.tag + ': sampled drafting', 20 * 60)
        self.command('hold-open', 'sparkhead', scoped_remote.command('launch', self.hold_tag,
            ['bash', '-c', script], deadline=self.wall_deadline), timeout=21 * 60)
        text = self.command('hold-read', 'sparkhead', hold_admission.command('check', self.hold_tag, self.token))
        self.opened = pf.hold_opened(text, self.tag, self.wall())
        bounds = deadlines(self.opened)
        self.submit, self.wall_deadline = bounds['submit'], bounds['scope']
        self.deadline = self.clock() + bounds['restore'] - self.wall()
        self.sup.abort_at = bounds['abort']
        self.sup.bound = f'PP4-PROMOTE HOLD+{ABORT_S / 60:g}min abort bound (recovery reserve {RECOVERY_S / 60:g}min)'
        self.sup.tunnel([*pf.SSH, '-N', '-o', 'ExitOnForwardFailure=yes', '-L',
                         '127.0.0.1:18001:127.0.0.1:8000', 'sparkhead'])
        self.sup.run('cuts', [sys.executable, '-m', 'offload.gb10_budget.window_cells', '--execute',
            '--data-dir', '/opt/dsv41/data/bench-data', 'build', '--output', str(self.run_out / 'cuts.json')], timeout=300)

    def controls(self, label: str) -> dict[str, list[dict[str, Json]]]:
        result: dict[str, list[dict[str, Json]]] = {}
        for name in CONTROLS:
            result[name] = []
            for rep in range(REPEATS[name]):
                require(self.wall() + cells.CELLS[name].estimate_s < self.submit, 'cell exceeds submission deadline')
                output = self.run_out / f'{label}-{name}-{rep}.json'
                self.sup.run(f'{label}-{name}-{rep}', [sys.executable, '-m', 'offload.gb10_budget.window_cells',
                    '--execute', '--data-dir', '/opt/dsv41/data/bench-data', 'run', '--cell', name,
                    '--cuts', str(self.run_out / 'cuts.json'), '--output', str(output), '--deadline-utc', str(self.submit),
                    '--telemetry', *(str(self.run_out / f'telemetry-{host}.jsonl') for host in HOSTS),
                    '--capture-floor-kb', str(4 * 1024 ** 2), '--cache-salt-prefix', f'{self.tag}-{label}-{rep}',
                    '--ledger', str(self.run_out / 'requests.jsonl')], timeout=600, inflight=1)
                row = record(cast(Json, json.loads(output.read_text())))
                require(row.get('profiled') is False and row.get('cell') == name, 'profiled/mismatched control')
                streams = cast(list[Json], row['streams'])
                require(len(streams) == 1, 'control stream count differs')
                stream = record(streams[0])
                usage = record(stream['usage'])
                ttft, total = number(stream['ttft_s']), number(stream['total_s'])
                require(total > ttft, 'invalid decode interval')
                result[name].append({'ttft_s': ttft, 'decode_tps': (number(usage['completion_tokens']) - 1) / (total - ttft),
                                     'prompt_sha256': stream['prompt_sha256']})
        (self.run_out / f'controls-{label}.json').write_text(json.dumps(result, indent=2) + '\n')
        return result

    def switch(self) -> None:
        self.sup.unready()
        stop = hold_admission.command('stop', self.hold_tag, self.token, ['bash', '-c', 'cd ~/dsv41-deploy && ./stop.sh </dev/null'])
        self.command('stop', 'sparkhead', stop, mutate=True, timeout=180)
        for host in HOSTS:
            text = self.command('quiet-' + host, host, spark_quiet.command('PP4 QUIET', deadline=self.wall() + 350), timeout=360)
            require('PP4 QUIET' in text.splitlines(), 'unknown Spark compute/container')
        text = self.command('spark-apply', 'sparkhead', 'cd ~/dsv41-deploy && python3 offload/launch/windows/pp4_promote_switch.py spark-apply',
                            mutate=True, timeout=120)
        require('PP4 SWITCH APPLIED ' + switch.PP4_ID in text, 'Spark switch unconfirmed')
        text = self.command('x86host-apply', 'x86host', x86host_apply_script(), mutate=True, timeout=900)
        require('PRODUCTION HOLD OK' in text.splitlines() and 'PP4 RUNTIME ATTESTED' in text, 'PP4 worker swap unverified')
        worker_gate(record(cast(Json, json.loads(self.command('worker-probe', 'x86host', 'python3 -c ' + shlex.quote(WORKER_PROBE))
                                                          .strip().splitlines()[-1]))), self.wall())
        text = self.command('pp4-start', 'sparkhead', 'cd ~/dsv41-deploy && ./start.sh </dev/null', mutate=True, timeout=900)
        require('K1 START VERIFIED' in text and 'PP4 RUNNING VERIFIED' in text.splitlines(), 'PP4 start lacks K1/PP4 verification')
        self.ready_at = self.wall()
        self.sup.mark_ready()

    def gates(self) -> None:
        self.sup.poll()
        for before, host in zip(self.original_identity, HOSTS):
            identity_gate(before, self.capture_backend.capture(host, 'pp4'))
        output = self.run_out / 'pp4-greedy.json'
        self.sup.run('pp4-greedy', [sys.executable, '-m', 'offload.launch.windows.pp4_cells', '--execute', '--mode', 'greedy',
                                    '--output', str(output), '--deadline-utc', str(self.submit)], timeout=900, inflight=1)
        result = record(cast(Json, json.loads(output.read_text())))
        require(result.get('greedy_registration') == 'pp4-r14-near-tie-1p1', 'PP4 greedy ran an unregistered rule')
        runs = [self.run_out / f'reference-{i}.jsonl' for i in range(REFERENCE_RUNS)]
        for i, path in enumerate(runs):
            require(self.wall() + 180 < self.submit, 'reference run exceeds submission deadline')
            self.sup.run(f'reference-{i}', [sys.executable, str(GREEDY_CHECK), TUNNEL, str(path)], timeout=900, inflight=1)
        self.publish_references(runs[0], len(references_identical(runs)))
        self.sup.poll()

    def publish_references(self, source: Path, rows: int) -> None:
        """Receipt first, then an atomic exclusive publication: the canonical file is either absent or complete and
        always has this run's receipt, so any interrupted or rejected publication is reconcilable (S91 #6)."""
        data = source.read_bytes()
        staging = NEW_REFERENCES.with_name(f'.{NEW_REFERENCES.name}.{self.tag}.staging')
        with self.reference_receipt.open('x') as handle:
            handle.write(json.dumps({'path': str(NEW_REFERENCES), 'sha256': hashlib.sha256(data).hexdigest(),
                                     'rows': rows, 'staging': str(staging), 'run': self.tag}, indent=2) + '\n')
        with staging.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, NEW_REFERENCES)            # exclusive and atomic: never overwrites a recorded reference
        staging.unlink()

    def reject_references(self) -> dict[str, Json]:
        """A rejected PP4 production's references never serve a later W8 or block the next promotion: move only this
        run's hash-verified file (and its own staging leftover) into the run directory; refuse any foreign file."""
        receipt = record(cast(Json, json.loads(self.reference_receipt.read_text()))) if self.reference_receipt.exists() else None
        result: dict[str, Json] = {}
        if receipt is not None:
            require(receipt.get('path') == str(NEW_REFERENCES), 'PP4 reference receipt names another file; HOLD retained')
            staging = Path(str(receipt['staging']))
            if staging.exists():                    # interrupted publication: exclusively created under this run's tag
                staging.rename(self.reference_archive.with_suffix('.staging.jsonl'))
                result['staging'] = 'archived'
        if NEW_REFERENCES.exists() or NEW_REFERENCES.is_symlink():
            require(receipt is not None, f'{NEW_REFERENCES} exists without this run\'s receipt; foreign file kept; HOLD retained')
            require(not NEW_REFERENCES.is_symlink() and pf.digest(NEW_REFERENCES) == cast(dict[str, Json], receipt)['sha256'],
                    'PP4 reference changed after recording or is foreign; kept; HOLD retained')
            require(not self.reference_archive.exists(), 'rejected-reference archive already exists; HOLD retained')
            NEW_REFERENCES.rename(self.reference_archive)
            result['references'] = str(self.reference_archive)
        return result

    def final(self) -> None:
        self.sup.poll()
        # The W8 check sends its six greedy prompts one at a time: one owned request (PROMO1's low-greedy allowance).
        text = self.command('pp4-w8-check', 'localhost', shlex.join(['bash', str(NEW_RESTORE), 'check']), timeout=600,
                            inflight=1)
        watch.restore_verified('check', 0, text)

    def accept(self) -> None:
        tag = 'prod-pp4-' + time.strftime('%Y%m%d', time.gmtime())
        script = ('cd ~/dsv41-deploy && git diff --exit-code -- start.sh serve.sh && '
                  f'! git show-ref --verify --quiet refs/tags/{tag} && printf %s {tag} > logs/pp4-switch/production-tag && '
                  'git add -- dsv41.env && git commit --only -m "Promote PP4 sampled drafting after PP4-PROMOTE gates" -- dsv41.env && '
                  f'git rev-parse HEAD > logs/pp4-switch/accepted-commit && git tag {tag}')
        self.command('accept', 'sparkhead', script, mutate=True, timeout=120)

    def finish(self, passed: bool, touched: bool) -> None:
        self.sup.close()
        for phase in list(self.phases):
            self.quiesce(phase)
        rc, text = self.invoke('ownership', 'sparkhead', hold_admission.command('check', self.hold_tag, self.token))
        require(rc == 0, 'HOLD ownership unreadable; retained')
        if hold_admission.NOT_OWNED in text:
            require(not touched, 'production touched without owned HOLD; manual recovery required')
            return
        if not self.opened:
            self.opened = datetime.datetime.fromisoformat(text.split()[0].replace('Z', '+00:00')).timestamp()
            bounds = deadlines(self.opened)
            self.wall_deadline = bounds['scope']
            self.deadline = self.clock() + bounds['restore'] - self.wall()
        restore = NEW_RESTORE if passed else self.rollback_restore
        if not passed and touched:
            stop = hold_admission.command('stop', self.hold_tag, self.token,
                ['bash', '-c', 'cd ~/dsv41-deploy && ./stop.sh </dev/null'])
            rc, text = self.owned('rollback-stop', 'sparkhead', ['bash', '-c', stop], 180)
            require(rc == 0, 'rollback stop failed; HOLD retained')
            for host in HOSTS:
                rc, text = self.invoke('quiet-' + host, host, spark_quiet.command('PP4 QUIET', deadline=self.wall_deadline), 360)
                require(rc == 0 and 'PP4 QUIET' in text.splitlines(), 'rollback compute absence unproved')
            rc, text = self.owned('rollback-spark', 'sparkhead', ['bash', '-c', 'cd ~/dsv41-deploy && '
                'if [ -e logs/pp4-switch/transaction.json ]; then python3 offload/launch/windows/pp4_promote_switch.py spark-restore; '
                'else profile=$(python3 -c "import json;print(json.load(open(\'logs/k1-promotion/files.json\'))[\'profile\'])") && '
                'python3 offload/promote/k1/manage.py verify --profile "$profile"; fi'], 120)
            require(rc == 0, 'rollback Spark env restore failed; HOLD retained')
            rc, text = self.owned('rollback-x86host', 'x86host', ['bash', '-c', x86host_rollback_script()], 900)
            require(rc == 0 and 'PRODUCTION HOLD OK' in text.splitlines(), 'rollback worker swap unverified; HOLD retained')
            # A failed/lost acceptance may have committed and tagged: record the restored env in a NEW commit, never reset;
            # remove only a tag bound to our own saved acceptance commit.
            undo = ("set -euo pipefail; cd ~/dsv41-deploy; if [ -f logs/pp4-switch/accepted-commit ]; then "
                    "tag=$(cat logs/pp4-switch/production-tag); accepted=$(cat logs/pp4-switch/accepted-commit); "
                    "if git show-ref --verify --quiet refs/tags/\"$tag\"; then "
                    "test \"$(git rev-list -n1 \"$tag\")\" = \"$accepted\" && git tag -d \"$tag\" || exit 1; fi; fi; "
                    "if ! git diff --quiet HEAD -- dsv41.env; then git add -- dsv41.env && "
                    "git commit --only -m 'Restore production after failed PP4-PROMOTE acceptance' -- dsv41.env; fi")
            rc, text = self.owned('rollback-ledger', 'sparkhead', ['bash', '-c', undo], 120)
            require(rc == 0, 'rollback production ledger unverified; HOLD retained')
            for host in HOSTS:
                rc, text = self.owned('rollback-clock-' + host, host, ['bash', '-c', clock_keep_script()], 60)
                require(rc == 0 and any(line.startswith('CLOCK OVERRIDE KEPT GPU_CLOCK_CAP=') for line in text.splitlines()),
                        'rollback GPU_CLOCK_CAP override unverified; HOLD retained')
            rc, text = self.owned('restore-start', 'localhost', ['bash', str(restore), 'start'], 900)
            watch.restore_verified('start', rc, text)
        if not passed:
            # After the prior production is back and its W8 started: a foreign reference file refuses the close (HOLD
            # retained, prior production serving) instead of being moved or deleted.
            (self.run_out / 'rejected-references.json').write_text(json.dumps(self.reject_references(), indent=2) + '\n')
        self.close_serial += 1
        close_phase = f'close-txn-{self.close_serial}-restore-close'
        for step, timeout in (('check', 600), ('close', 120)):
            phase = close_phase if step == 'close' else 'restore-' + step
            rc, text = self.owned(phase, 'localhost', ['bash', str(restore), step], timeout)
            watch.restore_verified(step, rc, text)
        for phase in list(self.phases):
            self.quiesce(phase)
        require(self.deadline - self.clock() >= FINAL_CLOSE_RESERVE_S, 'final close lacks retention reserve')
        tag = self.tag + '-' + close_phase
        try:
            try:
                rc, text = self.invoke('commit-close', 'sparkhead', scoped_remote.command('commit-close', tag,
                    deadline=min(self.wall_deadline + 60, self.wall() + 25)), 30)
                require(rc == 0 and scoped_remote.CLOSED in text.splitlines(), 'owned HOLD close unconfirmed')
            finally:
                rc, text = self.invoke('cancel-close', 'sparkhead', scoped_remote.command('cancel-close', tag), 30)
                require(rc == 0 and 'CLOSE COMMIT FENCED' in text, 'close cancellation unconfirmed')
            require(self.clock() < self.deadline, 'final HOLD close completed after immutable deadline')
        except (Exception, KeyboardInterrupt):
            rc, text = self.invoke('retain-close', 'sparkhead', scoped_remote.command('retain-close', tag), 30)
            require(rc == 0 and scoped_remote.RETAINED in text, 'original HOLD retention unconfirmed')
            raise

    def post_close(self, passed: bool) -> None:
        recovery = watch.ScopedRecovery(self.tag + '-watch', self.run_out / 'watch-commands',
            restore=NEW_RESTORE if passed else self.rollback_restore,
            restore_sha256=self.expected_new_sha if passed else self.rollback_sha256)
        self.watch_recovery = recovery
        try:
            result = watch.watch(production_health.ProductionHealth(), recovery, log=lambda message: self.log(message))
        except (Exception, KeyboardInterrupt) as exc:
            (self.run_out / 'watch.json').write_text(json.dumps({'status': f'FAILED: {exc}'}, indent=2) + '\n')
            raise
        (self.run_out / 'watch.json').write_text(json.dumps(asdict(result), indent=2) + '\n')
        require(result.status == 'PASS', f'post-close watch: {result.status}')

    def clock_cleanup(self) -> None:
        """Outside the HOLD, after the watch PASSed: the accepted production env must carry the folded 2300 default; then
        the exact GPU_CLOCK_CAP=2300 override goes from /etc/environment on both Sparks (a dial-back line stays)."""
        rc, text = self.call('clock-fold-check', [*pf.SSH, 'sparkhead', "cd ~/dsv41-deploy && grep -qF '"
                                                  + switch.CLOCK_DEFAULT + "' dsv41.env && echo CLOCK FOLDED"], 60)
        require(rc == 0 and 'CLOCK FOLDED' in text.splitlines(), 'production env lacks the folded 2300 default; override kept')
        cleared: dict[str, Json] = {}
        for host in HOSTS:
            rc, text = self.call('clock-cleanup-' + host, [*pf.SSH, host, clock_cleanup_script()], 60)
            lines = [line for line in text.splitlines() if line.startswith('CLOCK OVERRIDE CLEARED ')]
            require(rc == 0 and len(lines) == 1, f'{host}: GPU_CLOCK_CAP override removal unverified')
            cleared[host] = lines[0]
        (self.run_out / 'clock.json').write_text(json.dumps({'fold': switch.CLOCK_FOLD_COMMIT, 'cleared': cleared}, indent=2) + '\n')

    def rollback_watch(self) -> None:
        """Prior-production restore under the watchdog's own lease (PROMO1 G9 pattern); never adopts a foreign HOLD."""
        recovery = self.watch_recovery
        if recovery is None:
            recovery = watch.ScopedRecovery(self.tag + '-watch', self.run_out / 'watch-commands',
                                            restore=NEW_RESTORE, restore_sha256=self.expected_new_sha)
            self.watch_recovery = recovery
        if not recovery.deadline:
            recovery.deadline = recovery.clock() + watch.RECOVERY_S
            recovery.wall_deadline = recovery.wall() + watch.RECOVERY_S - 90
        evidence: dict[str, Json] = {'deadline': recovery.deadline, 'status': 'FAILED; HOLD retained/manual recovery required'}
        try:
            require(recovery.deadline - recovery.clock() > 90, 'watchdog lease exhausted; manual recovery required')
            rollback_out = self.run_out / 'watch-rollback'
            rollback_out.mkdir()
            rollback = Live(recovery.tag, rollback_out, self.commit, self.rollback_restore, self.rollback_sha256,
                            self.admission_path, self.measurement_path)
            rollback.clock, rollback.wall = recovery.clock, recovery.wall
            rollback.deadline, rollback.wall_deadline = recovery.deadline, recovery.wall_deadline
            rollback.opened = recovery.wall()
            rollback.phases = list(recovery.phases)
            rollback.hold_tag, rollback.token = recovery.tag + '-hold-open', recovery.admission_token
            rollback.reference_receipt, rollback.reference_archive = self.reference_receipt, self.reference_archive
            for phase in list(rollback.phases):
                rollback.quiesce(phase)
            rc, text = rollback.invoke('watch-ownership', 'sparkhead',
                hold_admission.command('check', rollback.hold_tag, rollback.token))
            require(rc == 0, 'watchdog HOLD ownership unreadable; manual recovery required')
            if hold_admission.NOT_OWNED in text:
                rc, text = rollback.invoke('watch-hold-absent', 'sparkhead', 'cd ~/dsv41-deploy && test ! -e HOLD && test ! -L HOLD')
                require(rc == 0, 'foreign/changed watchdog HOLD; refused; manual recovery required')
                rollback.tag += '-rb'
                rollback.hold_tag, rollback.token = rollback.tag + '-hold-open', uuid.uuid4().hex
                marker = 'POST-CLOSE-WATCH ' + rollback.tag + ':'
                opened = hold_admission.open_script(rollback.hold_tag, rollback.token,
                    marker + ' prior-production restore after failed PP4 watch', 19 * 60, allow_unreachable=True)
                rc, text = rollback.owned('hold-open', 'sparkhead', ['bash', '-c', opened], 20 * 60)
                require(rc == 0 and 'window open:' in text and marker in text, 'watchdog admission unverified; HOLD retained')
            rollback.finish(False, True)
            evidence['status'] = 'PASS: prior production restored; its W8 start/check/close verified'
        finally:
            (self.run_out / 'watch-rollback.json').write_text(json.dumps(evidence, indent=2) + '\n')

    def log(self, message: str) -> None:
        with (self.run_out / 'watch.log').open('a') as handle:
            handle.write(message + '\n')


def plan() -> str:
    estimate = window_estimate()
    return ('DRY PLAN ONLY: no host access, commands or writes.\n'
            f'PP4 {switch.PP4_COMMIT} sampled drafting (pp4-r14-near-tie-1p1), fixed5, ordinary verifier, '
            f'drop-in sha {DROPIN_SHA256[:12]}.\n'
            'Preflight: QUALIFIED admission (prereq.qualification + promoted byte pins + primary artifacts) and a COMPLETE '
            'production-policy v2 measurement carrying it (pooled-rate primary) that recomputes with no LOSS, staged '
            'Spark/x86host bytes, hooked '
            'production commit, live K1 profile verified, rollback W8 pinned.\n'
            'One owned HOLD: control short/32k x3 (win2-v1 decision cells) + 150k x1 (reported) on live production -> stop '
            '-> Spark env switch -> x86host drop-in -> guarded start -> identity, PP4 greedy, three identical NEW references '
            '-> same controls on PP4 (no decision cell LOSS under decide_cell: decode tok/s, TTFT guard; no win needed) '
            '-> restore-pp4.sh check -> commit + tag prod-pp4-YYYYMMDD (production commit carries the 420be24 2300 clock '
            'fold); any failure restores the '
            'prior production (GPU_CLOCK_CAP override kept on both Sparks before its W8 start) and its W8; 15-min '
            'post-close watch outside HOLD; after it PASSes the redundant GPU_CLOCK_CAP=2300 override is removed.\n'
            f'HOLD ESTIMATED: success {estimate["success_hold_s"] / 60:g}min; forward {estimate["forward_s"] / 60:g}min <= '
            f'abort +{ABORT_S / 60:g}; failure at the abort bound {estimate["failure_at_abort_hold_s"] / 60:g}min and an '
            f'accepted close failing after it {estimate["late_close_rollback_hold_s"] / 60:g}min <= {RESTORE_S / 60:g} '
            f'(accepted close {SUCCESS_CLOSE_S / 60:g} + recovery {RECOVERY_S / 60:g} + slack {UNPRICED_SLACK_S / 60:g} '
            f'+ final close {FINAL_CLOSE_RESERVE_S / 60:g}); submit +{SUBMIT_S / 60:g}. Starts, W8 and its check MEASURED; '
            'the rest ESTIMATED.')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--production-commit', default='')
    parser.add_argument('--rollback-restore', type=Path)
    parser.add_argument('--rollback-restore-sha256', default='')
    parser.add_argument('--admission-receipt', type=Path)
    parser.add_argument('--measurement-summary', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(plan())
        return 0
    try:
        estimate = window_estimate()        # BEFORE mkdir, SSH, telemetry, HTTP, inputs or signal hooks
        require(None not in (args.out, args.rollback_restore, args.admission_receipt, args.measurement_summary),
                'out/rollback-restore/admission-receipt/measurement-summary required')
        require(re.fullmatch('[0-9a-f]{64}', args.rollback_restore_sha256) is not None, 'pinned rollback W8 sha256 required')
        require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator only')
        out = cast(Path, args.out)
        require(out.parent.resolve() == CAMPAIGN.resolve(), 'new direct campaign output directory required')
        out.mkdir()
        (out / 'budget-estimate.json').write_text(json.dumps({'kind': 'ESTIMATED',
            'source': 'MEASURED PF2 DESIGN section 8 starts 6.5-6.8 min, W8 7.4 min, PC1 W8 check 20 s; other components estimated',
            'components_s': estimate}, indent=2) + '\n')
        live = Live(out.name, out, args.production_commit, cast(Path, args.rollback_restore), args.rollback_restore_sha256,
                    cast(Path, args.admission_receipt), cast(Path, args.measurement_summary))

        def interrupted(signum: int, frame: object) -> None:
            raise KeyboardInterrupt(f'signal {signum}')
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupted)

        def checkpoint(row: dict[str, Json]) -> None:
            with (out / 'checkpoints.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
        try:
            result = window(live, checkpoint)
        finally:
            live.sup.close()
        (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        if not str(result['verdict']).startswith('PASS'):
            return 1
        return 0 if str(result.get('clock_cleanup', '')).startswith('PASS') else 3
    except (ValueError, OSError) as exc:
        print(f'STOP before execution: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
