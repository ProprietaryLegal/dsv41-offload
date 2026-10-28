"""UP2-PROMOTE coordinator runner: turn 0075 replay skip ON in production. Dry by default.

One owned HOLD (PROMO1/PP4-PROMOTE structure): same-HOLD controls on the LIVE production (ops1 596ba1c or PROMO1's
327680-low; pinned at preflight) -> stop Spark serve -> Spark env switch (switch.py spark-apply, byte-exact snapshot,
exactly the qualified arm's flag and bytes) -> guarded production start (K1 START VERIFIED + UP2 RUNNING VERIFIED) ->
identity (= production + the one flag + the staged package) -> six greedy vs K1-post at 0.9+1e-3 nats
(up2-0075-adopt-0p9) -> 150K needle 6/6 -> three identical runs recorded as NEW greedy references -> the same controls on
0075 (win2-v1: no decision-cell LOSS; no win needed) -> versioned W8 check (restore-up2.sh) -> production commit + tag.
Any failure, lost reply or STOP restores the exact prior production and its own W8. A PP4 production refuses here: the
PP4+0075 package exists (packages/pp4-composed.json) but was never measured.
No command runs at import or in the dry plan. Fake backends exercise every path (tests/test_promote_window.py).
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
import sys
import time
from typing import Protocol, cast
import uuid

from offload.launch.windows import chain, decision_v2 as dv2
from offload.launch.windows import hold_admission, production_health, scoped_remote, spark_quiet
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import post_close_watch as watch
from offload.launch.windows import up3_window as up3
from offload.prefill_maxlen.evidence import Json, number, record
from offload.promote.up2 import hooks, switch
from offload.up3 import cells

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CAMPAIGN = pf.CAMPAIGN
PACKAGE = switch.package('qualified', HERE)
PACKAGE_ID = str(PACKAGE['id'])
PACKAGE_FILES = cast(dict[str, str], PACKAGE['files'])
NEW_RESTORE = HERE / 'restore-up2.sh'
K1_POST = CAMPAIGN / 'K1-production/greedy-post.jsonl'
NEW_REFERENCES = CAMPAIGN / f'K1-production/greedy-post-up2-{PACKAGE_ID}.jsonl'
GREEDY_CHECK = CAMPAIGN / 'W5/greedy_check.py'
DATA_DIR = Path('/opt/dsv41/data/bench-data')
REGISTRATION = cells.REPLAY_REGISTRATION
# win2-v1 on the production path, before vs after the switch (not interleaved: Welch interval). Same decision cells and
# metrics as the qualification window; the first metric decides, ms/round only guards (LOSS only).
DECISION_CELLS = up3.REPLAY_DECISION_CELLS
# S97 FIX 4 (CX3): PROMO1-low is refused until PROMO1's low patch is re-pinned onto the 420be24-folded env. Its 2300 line is
# PROMO1's own text, so PP4's fold-clock refuses to write the ledger CLOCK_FOLD_CHECK requires (switch.py keeps its support).
BASES = ('ops1',)
REFERENCE_RUNS = 3
NEEDLE_S = up3.NEEDLE_S
SUBMIT_S, ABORT_S, RESTORE_S = 60 * 60, 65 * 60, 85 * 60
HOSTS = ('sparkhead', 'sparkworker-wired')
URL = production_health.PRODUCTION
TUNNEL = 'http://127.0.0.1:18001'
# MEASURED: PF2 DESIGN section 8 arm starts 6.5-6.8 min, W8 7.4 min. Everything else is an ESTIMATE, never evidence.
MEASURED_START_S, MEASURED_W8_S = 6.8 * 60, 7.4 * 60
# Repo plan Phase 0: the production this switch stacks on carries 420be24's 2300 clock fold, recorded in the K1 guard by
# PP4-PROMOTE's reused `pp4_promote_switch.py fold-clock` (its ledger). Order: 420be24 -> hooks -> push sparkhead ->
# fold-clock -> W8 start (UP2-REPIN.md).
CLOCK_FOLD_CHECK = ("grep -qx 'GPU_CLOCK_CAP=${GPU_CLOCK_CAP-2300} .*' dsv41.env && "
                    'test -s logs/clock-fold/transaction.json')
# S97 FIX 5: fold-clock writes the production K1 record, so the coordinator runs it only after the staged copy's bytes are
# proved to be the reviewed tool (blob 9dcb9eb, identical in 8548a7c and this branch). README / UP2-REPIN carry FOLD_COMMAND.
FOLD_TOOL = 'offload/launch/windows/pp4_promote_switch.py'
FOLD_TOOL_SHA256 = 'e722fc56913673e4ac23b9c19b64401f81c5115f5650863c309078be9715ee49'
FOLD_COMMAND = (f"ssh sparkhead 'cd ~/dsv41-deploy-up2a && echo \"{FOLD_TOOL_SHA256}  {FOLD_TOOL}\" | sha256sum -c --quiet && "
                f"python3 {FOLD_TOOL} fold-clock --root ~/dsv41-deploy'")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def window_estimate() -> dict[str, float]:
    """Aggregate admission guard; refuses before any host contact when the ESTIMATE exceeds the HOLD."""
    controls = sum(cells.ESTIMATES[name] * cells.REPEATS[name] for name in DECISION_CELLS)
    components = {'controls-production': float(controls), 'controls-0075': float(controls), 'cuts': 120.,
                  'stop_quiet_switch': 4 * 60., 'guarded_start': MEASURED_START_S + 72, 'greedy_0p9': 4 * 60.,
                  'needle_150k': float(NEEDLE_S), 'new_references': REFERENCE_RUNS * 2 * 60., 'final_w8_check': 6 * 60.,
                  'accept_close': 3 * 60.}
    # Rollback: Spark stop/quiet/env restore, old W8 start/check/close.
    components['rollback_reserve'] = 3 * 60 + MEASURED_W8_S + 4 * 60
    total = sum(components.values())
    require(math.isfinite(total) and total <= RESTORE_S,
            f'window estimate {total / 60:g}min exceeds {RESTORE_S / 60:g}min; refused before admission')
    return {**components, 'total_s': total}


def staged_files(replay: dict[str, str]) -> dict[str, str]:
    """UP3 replay manifest paths -> the promotion layout (remote package files under remote_dspark/)."""
    prefix = 'vllm/v1/worker/gpu/spec_decode/remote_dspark/'
    return {('remote_dspark/' + p.removeprefix(prefix) if p.startswith(prefix) else p): v for p, v in replay.items()}


def evidence_gate(summary: dict[str, Json]) -> dict[str, Json]:
    """The UP2-ADOPT window's own summary: COMPLETE, restored, ADOPT CANDIDATE, for exactly these package bytes."""
    require(summary.get('status') == 'COMPLETE' and 'stop' not in summary, 'qualification window not COMPLETE')
    require(str(record(summary['restore']).get('status', '')).startswith('RESTORED'), 'qualification window not restored')
    decision = record(summary['decision'])
    require(decision.get('rule') == dv2.RULE and decision.get('adopt') == 'ADOPT CANDIDATE'
            and decision.get('registration') == REGISTRATION,
            f'qualification decision is not a {dv2.RULE} ADOPT CANDIDATE under {REGISTRATION}')
    require(staged_files(cast(dict[str, str], decision.get('replay_files', {}))) == PACKAGE_FILES,
            f'qualified bytes differ from package {PACKAGE_ID}')
    return {'qualification': 'ADOPT CANDIDATE', 'win2_status': decision.get('status'), 'package': PACKAGE_ID}


def references_identical(paths: list[Path]) -> list[dict[str, Json]]:
    runs = [[record(cast(Json, json.loads(line))) for line in p.read_text().splitlines()] for p in paths]
    require(len(runs) == REFERENCE_RUNS and all(len(r) == 6 for r in runs), 'three complete six-prompt runs required')
    for run in runs:
        require([row.get('i') for row in run] == list(range(6)) and all(isinstance(row.get('text'), str) for row in run),
                'ordered six-prompt reference rows required')
    first = [row['text'] for row in runs[0]]
    require(all([row['text'] for row in run] == first for run in runs[1:]),
            '0075 greedy outputs are not reproducible across three runs; no reference recorded')
    return runs[0]


def identity_gate(before: dict[str, Json], after: dict[str, Json]) -> None:
    """Live 0075 rank == saved production rank plus exactly the flag and the staged package's 24 files."""
    require(before['id'] != after['id'], '0075 reused the control container')
    for key in ('image', 'engine', 'files', 'engram_identity'):
        require(before[key] == after[key], f'runtime drift: {key}')
    key, value = switch.SWITCH_ENV
    require(key not in record(before['env']), f'production already sets {key}')
    require(record(after['env']) == {**record(before['env']), key: value}, '0075 environment differs from the one flag')
    remote = pf.DIST + switch.REMOTE.removeprefix('/') + '/'
    mounted = {k: v for k, v in record(before['mounted']).items() if k != '/serve.sh' and not k.startswith(remote)}
    mounted.update({pf.DIST + (switch.REMOTE + '/' + p.removeprefix('remote_dspark/') if p.startswith('remote_dspark/')
                               else p): v for p, v in PACKAGE_FILES.items()})
    require({k: v for k, v in record(after['mounted']).items() if k != '/serve.sh'} == mounted,
            'mounted engine bytes differ from production plus the qualified 0075 package')
    require(str(after['argv']) == str(before['argv']), '0075 changed the production launch argv')


def regression(control: dict[str, dict[str, Json]], candidate: dict[str, dict[str, Json]]) -> dict[str, Json]:
    """win2-v1 over the same cells before/after the switch; any decision-cell LOSS refuses (no win required)."""
    require(set(control) == set(candidate) == set(DECISION_CELLS), 'G6: control set incomplete')
    series: list[dv2.Series] = []
    for name in DECISION_CELLS:
        for metric, higher in up3.REPLAY_METRICS[name]:
            series.append(dv2.Series(name, metric, higher, tuple(up3.repeat_values(control[name], metric)),
                                     tuple(up3.repeat_values(candidate[name], metric)), paired=False))
    result = dv2.decide_window(series, DECISION_CELLS)
    require(result['status'] != 'NOT A CANDIDATE', 'G6: a decision cell LOST on the production path: ' +
            json.dumps({c: record(v)['verdict'] for c, v in record(result['cells']).items()}))
    return result


class Backend(Protocol):
    def preflight(self) -> None: ...
    def open_hold(self) -> None: ...
    def controls(self, label: str) -> dict[str, dict[str, Json]]: ...
    def switch(self) -> None: ...
    def gates(self) -> None: ...
    def final(self) -> None: ...
    def accept(self) -> None: ...
    def finish(self, passed: bool, touched: bool) -> None: ...
    def post_close(self, passed: bool) -> None: ...
    def rollback_watch(self) -> None: ...


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
        backend.gates()                     # identity, greedy 0.9 nat, 150K needle, new references
        replay = backend.controls('replay')
        result['G6'] = regression(control, replay)
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
                            result['stop'] = f'accepted-0075 close failed: {exc}; rolled back'
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
    result['verdict'] = 'PASS: production left on 0075 replay skip' if passed else 'FAIL: see stop/close/watch evidence'
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


class Live(watch.ScopedRecovery):
    """Reuses shared scope, HOLD admission, quiet proof, health and post-close watch (PROMO1 structure)."""
    def __init__(self, tag: str, out: Path, commit: str, base: str, rollback_restore: Path, rollback_sha256: str,
                 qualification: Path) -> None:
        super().__init__(tag, out / 'restore-commands')
        require(base in BASES, f'production base {base!r}: only {BASES} compose with the measured 0075 package')
        self.run_out, self.commit, self.base = out, commit, base
        self.rollback_restore, self.rollback_sha256 = rollback_restore, rollback_sha256
        self.qualification = qualification
        self.sup = StrictSupervisor(out)
        self.hold_tag, self.token = tag + '-hold-open', uuid.uuid4().hex
        self.opened = 0.0
        self.deadline = self.clock() + 120 * 60
        self.wall_deadline = self.wall() + 120 * 60
        self.submit = math.inf
        self.phase_serial = 0
        self.close_serial = 0
        self.capture_backend = pf.Live(out, self.sup, remote='dsv41-deploy', expected=commit,
            data_dir=DATA_DIR, tag=tag, stage_bundle=False)
        self.original_identity: list[dict[str, Json]] = []
        self.expected_new_sha = hashlib.sha256(NEW_RESTORE.read_bytes()).hexdigest()
        self.watch_recovery: watch.ScopedRecovery | None = None

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

    def driver(self, label: str, output: Path, *args: str, timeout: float, inflight: int = 1) -> dict[str, Json]:
        self.sup.run(label, [sys.executable, '-m', 'offload.up3.cells', '--execute', '--output', str(output),
                             '--data-dir', str(DATA_DIR), '--deadline', str(self.submit), *args], timeout=timeout,
                     inflight=inflight)
        return record(cast(Json, json.loads(output.read_text())))

    def preflight(self) -> None:
        window_estimate()
        require(not chain.enabled() and not chain.last() and chain.read_marker() is None,
                'a promotion never runs chained (WINDOW_CHAIN / chain marker): it must start from and leave production serving')
        require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator only')
        require((CAMPAIGN / 'D2-GEN.PAUSE').exists(), 'D2 pause marker required')
        require(re.fullmatch('[0-9a-f]{40}', self.commit) is not None, 'full production package commit required')
        require(hashlib.sha256(self.rollback_restore.read_bytes()).hexdigest() == self.rollback_sha256,
                'rollback W8 differs from its pinned sha256')
        require(K1_POST.is_file() and not NEW_REFERENCES.exists(), 'K1-post present and 0075 references absent required')
        evidence = evidence_gate(record(cast(Json, json.loads(self.qualification.read_text()))))
        (self.run_out / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
        self.sup.begin()
        profile = 'profile=$(python3 -c "import json;print(json.load(open(\'logs/k1-promotion/files.json\'))[\'profile\'])")'
        text = self.command('ledger', 'sparkhead', 'cd ~/dsv41-deploy && test ! -e HOLD && '
            f'test "$(git rev-parse HEAD)" = {self.commit} && test -z "$(git status --porcelain --untracked-files=no)" && '
            f'test ! -e {switch.BACKUP} && {CLOCK_FOLD_CHECK} && {profile} && '
            'python3 offload/promote/k1/manage.py verify --profile "$profile" && '
            f'grep -c {hooks.TAG} {" ".join(hooks.HOOKS_PATHS)} | grep -c ":[1-9]" | grep -qx {len(hooks.HOOKS_PATHS)} && '
            'python3 offload/promote/up2/switch.py base --env dsv41.env --profile "$profile"')
        require(text.strip().splitlines()[-1:] == [self.base], f'live production base is not the pinned {self.base}')
        for rel in (hooks.SWITCH, *hooks.PACKAGES):
            text = self.command('pin-' + Path(rel).name.replace('.', '-'), 'sparkhead', f'cd ~/dsv41-deploy && sha256sum {rel}')
            require(text.split()[0] == pf.digest(ROOT / rel), f'production {rel} drift')
        for host in HOSTS:
            self.command('staged-' + host, host, f'cd ~/dsv41-offload-prod/up2/{PACKAGE_ID} && sha256sum -c --quiet SHA256SUMS && '
                         f'test "$(sha256sum < SHA256SUMS | cut -c1-12)" = {PACKAGE_ID}')
        production_health.ProductionHealth().probe()
        self.sup.mark_ready()
        worker = self.command('worker', 'x86host', 'bash ~/dsv41-offload/prod_window.sh check')
        require('PRODUCTION HOLD OK' in worker.splitlines(), 'production worker not verified')
        self.original_identity = [self.capture_backend.capture(host, 'control') for host in HOSTS]
        (self.run_out / 'rollback-identity.json').write_text(json.dumps(self.original_identity, indent=2) + '\n')

    def open_hold(self) -> None:
        self.wall_deadline = self.wall() + 21 * 60 + RESTORE_S
        self.phases.append(self.hold_tag)
        script = hold_admission.open_script(self.hold_tag, self.token, 'UP2-PROMOTE ' + self.tag + ': 0075 replay skip', 20 * 60)
        self.command('hold-open', 'sparkhead', scoped_remote.command('launch', self.hold_tag,
            ['bash', '-c', script], deadline=self.wall_deadline), timeout=21 * 60)
        text = self.command('hold-read', 'sparkhead', hold_admission.command('check', self.hold_tag, self.token))
        self.opened = pf.hold_opened(text, self.tag, self.wall())
        self.submit = self.opened + SUBMIT_S
        self.wall_deadline = self.opened + RESTORE_S - 90
        self.deadline = self.clock() + self.opened + RESTORE_S - self.wall()
        self.sup.abort_at = self.opened + ABORT_S
        self.sup.bound = 'UP2-PROMOTE HOLD+65min measurement bound'
        self.sup.tunnel([*pf.SSH, '-N', '-o', 'ExitOnForwardFailure=yes', '-L',
                         '127.0.0.1:18001:127.0.0.1:8000', 'sparkhead'])
        self.driver('cuts', self.run_out / 'cuts.json', '--mode', 'cuts', timeout=300)

    def controls(self, label: str) -> dict[str, dict[str, Json]]:
        result: dict[str, dict[str, Json]] = {}
        for name in DECISION_CELLS:
            require(self.wall() + cells.ESTIMATES[name] * cells.REPEATS[name] < self.submit, 'cell exceeds submission deadline')
            result[name] = self.driver(f'{label}-{name}', self.run_out / f'{label}-{name}.json', '--mode', 'cell',
                '--cell', name, '--cuts', str(self.run_out / 'cuts.json'), '--salt', f'{self.tag}-{label}',
                '--ledger', str(self.run_out / 'requests.jsonl'), timeout=cells.ESTIMATES[name] * cells.REPEATS[name] * 2 + 30)
            require(result[name].get('cell') == name, 'control cell identity differs')
        return result

    def switch(self) -> None:
        self.sup.unready()
        stop = hold_admission.command('stop', self.hold_tag, self.token, ['bash', '-c', 'cd ~/dsv41-deploy && ./stop.sh </dev/null'])
        self.command('stop', 'sparkhead', stop, mutate=True, timeout=180)
        for host in HOSTS:
            text = self.command('quiet-' + host, host, spark_quiet.command('UP2 QUIET', deadline=self.wall() + 350), timeout=360)
            require('UP2 QUIET' in text.splitlines(), 'unknown Spark compute/container')
        text = self.command('spark-apply', 'sparkhead', 'cd ~/dsv41-deploy && python3 offload/promote/up2/switch.py spark-apply',
                            mutate=True, timeout=120)
        require(f'UP2 SWITCH APPLIED {PACKAGE_ID} on base {self.base}' in text, 'Spark switch unconfirmed')
        text = self.command('up2-start', 'sparkhead', 'cd ~/dsv41-deploy && ./start.sh </dev/null', mutate=True, timeout=900)
        require('K1 START VERIFIED' in text and 'UP2 RUNNING VERIFIED' in text.splitlines(), '0075 start lacks K1/UP2 verification')
        self.sup.mark_ready()

    def gates(self) -> None:
        self.sup.poll()
        for before, host in zip(self.original_identity, HOSTS):
            identity_gate(before, self.capture_backend.capture(host, 'replay'))
        greedy = self.driver('up2-greedy', self.run_out / 'up2-greedy.json', '--mode', 'greedy',
                             '--near-tie-nats', repr(cells.REPLAY_NEAR_TIE_NATS), timeout=up3.GREEDY_S * 15)
        up3.greedy_checks(greedy)
        require(greedy.get('greedy_registration') == REGISTRATION, '0075 greedy ran an unregistered rule')
        needle = self.run_out / 'needle150000.json'
        self.sup.run('needle150000', [sys.executable, '-m', 'offload.launch.windows.up2_request', '--needle', '--target',
            '150000', '--budget', '300', '--salt', f'{self.tag}:needle150000', '--output', str(needle), '--ledger',
            str(self.run_out / 'requests.jsonl'), '--deadline', str(self.submit), '--data-dir', str(DATA_DIR)],
            timeout=2 * NEEDLE_S, inflight=1)
        row = record(cast(Json, json.loads(needle.read_text())))
        require(row.get('recall') == 6 and row.get('of') == 6 and row.get('foreign_codes') == 0, '150K needle 6/6 failed')
        runs = [self.run_out / f'reference-{i}.jsonl' for i in range(REFERENCE_RUNS)]
        for i, path in enumerate(runs):
            require(self.wall() + 180 < self.submit, 'reference run exceeds submission deadline')
            self.sup.run(f'reference-{i}', [sys.executable, str(GREEDY_CHECK), TUNNEL, str(path)], timeout=900, inflight=1)
        reference = references_identical(runs)
        with NEW_REFERENCES.open('x') as handle:   # exclusive: never overwrite a recorded reference
            handle.write(runs[0].read_text())
        (self.run_out / 'new-references.json').write_text(json.dumps(
            {'path': str(NEW_REFERENCES), 'sha256': pf.digest(NEW_REFERENCES), 'rows': len(reference)}, indent=2) + '\n')
        self.sup.poll()

    def final(self) -> None:
        self.sup.poll()
        # The W8 check sends its six greedy prompts one at a time: one owned request (PROMO1's low-greedy allowance).
        text = self.command('up2-w8-check', 'localhost', shlex.join(['bash', str(NEW_RESTORE), 'check']), timeout=600,
                            inflight=1)
        watch.restore_verified('check', 0, text)

    def accept(self) -> None:
        tag = 'prod-up2-' + time.strftime('%Y%m%d', time.gmtime())
        script = ('cd ~/dsv41-deploy && git diff --exit-code -- start.sh serve.sh && '
                  f'! git show-ref --verify --quiet refs/tags/{tag} && printf %s {tag} > {switch.BACKUP}/production-tag && '
                  'git add -- dsv41.env && git commit --only -m "Promote 0075 replay skip after UP2-PROMOTE gates" -- dsv41.env && '
                  f'git rev-parse HEAD > {switch.BACKUP}/accepted-commit && git tag {tag}')
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
            self.wall_deadline = self.opened + RESTORE_S - 90
            self.deadline = self.clock() + self.opened + RESTORE_S - self.wall()
        restore = NEW_RESTORE if passed else self.rollback_restore
        recorded = self.run_out / 'new-references.json'
        if not passed and recorded.exists() and NEW_REFERENCES.exists():
            # A rejected promotion's reference must never serve a later W8: keep it as this run's evidence only.
            require(pf.digest(NEW_REFERENCES) == record(cast(Json, json.loads(recorded.read_text())))['sha256'],
                    '0075 reference changed after recording; HOLD retained')
            NEW_REFERENCES.rename(self.run_out / f'{NEW_REFERENCES.stem}.rejected.jsonl')
        if not passed and touched:
            stop = hold_admission.command('stop', self.hold_tag, self.token,
                ['bash', '-c', 'cd ~/dsv41-deploy && ./stop.sh </dev/null'])
            rc, text = self.owned('rollback-stop', 'sparkhead', ['bash', '-c', stop], 180)
            require(rc == 0, 'rollback stop failed; HOLD retained')
            for host in HOSTS:
                rc, text = self.invoke('quiet-' + host, host, spark_quiet.command('UP2 QUIET', deadline=self.wall_deadline), 360)
                require(rc == 0 and 'UP2 QUIET' in text.splitlines(), 'rollback compute absence unproved')
            rc, text = self.owned('rollback-spark', 'sparkhead', ['bash', '-c', 'cd ~/dsv41-deploy && '
                f'if [ -e {switch.BACKUP}/transaction.json ]; then python3 offload/promote/up2/switch.py spark-restore; '
                'else profile=$(python3 -c "import json;print(json.load(open(\'logs/k1-promotion/files.json\'))[\'profile\'])") && '
                'python3 offload/promote/k1/manage.py verify --profile "$profile"; fi'], 120)
            require(rc == 0, 'rollback Spark env restore failed; HOLD retained')
            # A failed/lost acceptance may have committed and tagged: record the restored env in a NEW commit, never reset;
            # remove only a tag bound to our own saved acceptance commit.
            undo = (f"set -euo pipefail; cd ~/dsv41-deploy; if [ -f {switch.BACKUP}/accepted-commit ]; then "
                    f"tag=$(cat {switch.BACKUP}/production-tag); accepted=$(cat {switch.BACKUP}/accepted-commit); "
                    "if git show-ref --verify --quiet refs/tags/\"$tag\"; then "
                    "test \"$(git rev-list -n1 \"$tag\")\" = \"$accepted\" && git tag -d \"$tag\" || exit 1; fi; fi; "
                    "if ! git diff --quiet HEAD -- dsv41.env; then git add -- dsv41.env && "
                    "git commit --only -m 'Restore production after failed UP2-PROMOTE acceptance' -- dsv41.env; fi")
            rc, text = self.owned('rollback-ledger', 'sparkhead', ['bash', '-c', undo], 120)
            require(rc == 0, 'rollback production ledger unverified; HOLD retained')
            rc, text = self.owned('restore-start', 'localhost', ['bash', str(restore), 'start'], 900)
            watch.restore_verified('start', rc, text)
        self.close_serial += 1
        close_phase = f'close-txn-{self.close_serial}-restore-close'
        for step, timeout in (('check', 600), ('close', 120)):
            phase = close_phase if step == 'close' else 'restore-' + step
            rc, text = self.owned(phase, 'localhost', ['bash', str(restore), step], timeout)
            watch.restore_verified(step, rc, text)
        for phase in list(self.phases):
            self.quiesce(phase)
        require(self.deadline - self.clock() >= 90, 'final close lacks retention reserve')
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
            rollback = Live(recovery.tag, rollback_out, self.commit, self.base, self.rollback_restore,
                            self.rollback_sha256, self.qualification)
            rollback.clock, rollback.wall = recovery.clock, recovery.wall
            rollback.deadline, rollback.wall_deadline = recovery.deadline, recovery.wall_deadline
            rollback.opened = recovery.wall()
            rollback.phases = list(recovery.phases)
            rollback.hold_tag, rollback.token = recovery.tag + '-hold-open', recovery.admission_token
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
                    marker + ' prior-production restore after failed 0075 watch', 19 * 60, allow_unreachable=True)
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
            f'0075 replay skip, package {PACKAGE_ID} (= the UP3 replay arm bytes), flag {"=".join(switch.SWITCH_ENV)}, '
            f'greedy {REGISTRATION} ({cells.REPLAY_NEAR_TIE_NATS} nat).\n'
            'Preflight: the qualification window summary (COMPLETE, RESTORED, ADOPT CANDIDATE, same bytes); staged package on '
            'both Sparks; hooked production commit; live base pinned (ops1 only: PROMO1-low refused until PROMO1 carries the '
            '420be24 fold, PP4 refused: never measured); '
            'rollback W8 pinned; no chain.\n'
            'One owned HOLD: controls 150K B1 x3 + 32K B1 x4 on live production -> stop -> Spark env switch -> guarded start '
            '-> identity, greedy 0.9 nat, 150K needle 6/6, three identical NEW references -> same controls on 0075 (win2-v1, '
            'Welch: no decision-cell LOSS; no win needed) -> restore-up2.sh check -> commit + tag prod-up2-YYYYMMDD; any '
            'failure restores the prior production and its W8; 15-min post-close watch outside HOLD.\n'
            f'HOLD ESTIMATED {estimate["total_s"] / 60:g}min <= {RESTORE_S / 60:g}min (starts and W8 MEASURED; the rest '
            'ESTIMATED); submit+60, abort+65, recovery/close by+85.')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--production-commit', default='')
    parser.add_argument('--base', choices=BASES)
    parser.add_argument('--rollback-restore', type=Path)
    parser.add_argument('--rollback-restore-sha256', default='')
    parser.add_argument('--qualification-summary', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(plan())
        return 0
    try:
        estimate = window_estimate()        # BEFORE mkdir, SSH, telemetry, HTTP, inputs or signal hooks
        require(None not in (args.out, args.base, args.rollback_restore, args.qualification_summary),
                'out/base/rollback-restore/qualification-summary required')
        require(re.fullmatch('[0-9a-f]{64}', args.rollback_restore_sha256) is not None, 'pinned rollback W8 sha256 required')
        require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator only')
        out = cast(Path, args.out)
        require(out.parent.resolve() == CAMPAIGN.resolve(), 'new direct campaign output directory required')
        out.mkdir()
        (out / 'budget-estimate.json').write_text(json.dumps({'kind': 'ESTIMATED',
            'source': 'MEASURED PF2 DESIGN section 8 starts 6.5-6.8 min, W8 7.4 min; other components estimated',
            'components_s': estimate}, indent=2) + '\n')
        live = Live(out.name, out, args.production_commit, str(args.base), cast(Path, args.rollback_restore),
                    args.rollback_restore_sha256, cast(Path, args.qualification_summary))

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
        return 0 if str(result['verdict']).startswith('PASS') else 1
    except (ValueError, OSError) as exc:
        print(f'STOP before execution: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
