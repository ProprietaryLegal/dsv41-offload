"""TV1 coordinator-only, interleaved OFF/one-switch window; dry by default. W8 restore always follows HOLD.

Reuses PF1 supervision/snapshot/start/rollback and SP1 profiler collection. No lane
runs this or its preflight on real hosts. One feature per HOLD, fixed C/arm/C.
"""
from __future__ import annotations

import argparse
import json
import math
import uuid
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import sp1_window as sp
from offload.launch.windows import scoped_remote as owner
from offload.launch.windows import hold_admission, spark_quiet
from offload.launch.windows import post_close_watch as watchdog
from offload.launch.windows import up2_checks as ck
from offload.launch.windows.up2_low import X86HOST_PROBE, terminate
from offload.prefill_trace.telemetry import FIELDS
from offload.prefill_maxlen.evidence import Json, record
from offload.tv1 import build_overlay as overlay
from offload.tv1 import cells
from offload.tv1 import production
from offload.tv1 import qualification
from offload.tv1 import schedule
from offload.tv1 import engine_logs
from offload.tv1 import init_smoke
from offload.tv1 import candidate_init_smoke
from offload.tv1 import greedy
from offload.tv1 import oracle_prestart
from offload.tv1 import oracle_memory
from offload.tv1.candidate_cases import validate_engine_receipts
from offload.tv1.deadline import ABORT_S as ABORT_S, RESTORE_S as RESTORE_S, SUBMIT_S as SUBMIT_S, HoldBudget as HoldBudget

CONFIG = 'offload/tv1/window.env'
PIN_FILES = (*pf.PIN_FILES, 'offload/launch/configs/SP1-profile.env', CONFIG,
             'offload/launch/windows/tv1_window.py', 'offload/tv1/build_overlay.py',
             'offload/tv1/inputs.lock.json', 'offload/tv1/verify_runtime.py',
             'offload/tv1/window_config.py', 'offload/tv1/cells.py', 'offload/tv1/metrics.py', 'offload/tv1/gb10_oracle.py',
             'offload/tv1/candidate.py', 'offload/tv1/candidate_kernel.py', 'offload/tv1/compact.py',
             'offload/tv1/patches/candidate.patch', 'offload/tv1/patches/compact.patch',
             'offload/tv1/greedy-references.json', 'offload/tv1/analyze.py', 'offload/tv1/greedy.py',
             'offload/launch/windows/nbw_cells.py',
             'offload/launch/windows/scoped_remote.py', 'offload/launch/windows/hold_admission.py',
             'offload/launch/windows/spark_quiet.py', 'offload/launch/windows/post_close_watch.py',
             'offload/launch/windows/production_health.py', 'offload/tv1/production.py',
             'offload/tv1/candidate_cases.py', 'offload/tv1/candidate_metadata.py',
             'offload/tv1/qualification.py', 'offload/tv1/patches/candidate-compact.patch', 'offload/tv1/deadline.py',
             'offload/tv1/schedule.py', 'offload/tv1/engine_logs.py', 'offload/tv1/init_smoke.py',
             'offload/tv1/candidate_init_smoke.py', 'offload/tv1/oracle_prestart.py',
             'offload/tv1/oracle_memory.py')
OWNER_HOSTS = (*owner.NODES, 'localhost')
ORACLE_FLOOR_KB = oracle_memory.FLOOR_KB
FINAL_CLOSE_RESERVE_S, FINAL_CLOSE_CALL_S = 90, 20
QUIET = 'TV1 EXPERIMENT ABSENT'


@dataclass(frozen=True)
class Arm(pf.Arm):
    role: str = 'off'

    @property
    def config(self) -> str:
        return CONFIG


def arms(feature: str) -> tuple[Arm, ...]:
    ck.require(feature in ('candidate', 'compact'), 'TV1 one known feature per HOLD')
    return tuple(Arm(f's{i+1}-{role}', 327680, 0, role=role)
                 for i, role in enumerate(schedule.roles(feature)))


def source_verifier(manifest: overlay.Manifest) -> str:
    k1 = ck.read(pf.ROOT / 'offload/kvpack/MANIFEST.json')
    pins = {str(e['container_path']): str(e['overlay_sha256']) for e in ck.rows(k1['files'])}
    pins.update({pf.DIST + e['path']: e['sha256'] for e in manifest['files']})
    return ('import hashlib,pathlib\n' + f'pins={pins!r}\n'
            'for name,want in pins.items():\n'
            ' if hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()!=want:\n'
            '  raise ValueError("TV1 runtime source hash mismatch: "+name)\n'
            'print("TV1 verified",len(pins),"engine sources")\n')


def runtime(prod: dict[str, Json], live: dict[str, Json], arm: Arm, manifest: overlay.Manifest, run: str) -> None:
    production.runtime(prod, live, arm, manifest, run)


def engine_oracle_script() -> str:
    return '''import json,subprocess
records=[]
p=subprocess.Popen(['docker','logs','dsv41-serve'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
for line in p.stdout:
 if 'TV1_ENGINE_ORACLE ' in line:
  records.append(json.loads(line.split('TV1_ENGINE_ORACLE ',1)[1]))
if p.wait(): raise ValueError('TV1 engine oracle log read failed')
print(json.dumps({'records':records}))
'''


class TV1Live(sp.SP1Live):
    def __init__(self, out: Path, sup: pf.Supervisor, *, remote: str, expected: str,
                 data_dir: Path, tag: str, stage_bundle: bool, feature: str,
                 call: pf.Call | None = None,
                 plain: Callable[[str, list[str], float], tuple[int, str]] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        super().__init__(out, sup, remote=remote, expected=expected, data_dir=data_dir, tag=tag,
                         stage_bundle=stage_bundle, call=call, plain=plain, clock=clock)
        self.feature = feature
        self.active: Arm | None = None
        self.root_out, self.root_tag = out, tag
        self.manifests: dict[str, overlay.Manifest] = {}
        self.base_call = self.call
        self.call = self.tagged_call
        self.owner_tags: list[str] = []
        self.owner_deadlines: dict[str, float] = {}
        self.recovery_serial = 0
        self.restore_log_serial = 0
        self.engine_log_serial = 0
        self.experiment_attempted = False
        self.start_failed = False
        self.engine_log_errors: list[str] = []
        self.admission_token = uuid.uuid4().hex
        self.open_deadline = float('inf')
        self.hold_scope_deadline = float('inf')
        self.budget: HoldBudget | None = None
        self.open_admitted: float | None = None

    def remember(self, tag: str, deadline: float) -> None:
        ck.require(math.isfinite(deadline), 'TV1 finite scoped deadline required')
        ck.require(self.owner_deadlines.get(tag, deadline) == deadline, 'TV1 immutable phase deadline changed')
        if tag not in self.owner_tags:
            self.owner_tags.append(tag)  # before any submission, including a lost reply
            self.owner_deadlines[tag] = deadline

    def ssh(self, label: str, host: str, command: str, stdin: str | None = None, timeout: float = 120,
            inflight: int = 0) -> str:
        if label.endswith('-start'):
            tag = f'{self.root_tag}-{label}'
            self.remember(tag, self.sup.abort_at)
            command = hold_admission.command('stop', self.root_tag + '-hold', self.admission_token,
                                             ['bash', '-c', command])
            command = owner.command('launch', tag, ['bash', '-c', command], deadline=self.sup.abort_at)
        return super().ssh(label, host, command, stdin, timeout, inflight)

    def local(self, name: str, command: list[str], timeout: float) -> tuple[int, str]:
        """Only this local transport group; shared scopes own W8 and every receiving hop."""
        self.restore_log_serial += 1
        log = self.root_out / 'cmd' / f'restore-{self.restore_log_serial:04d}-{name}.log'
        with log.open('x') as handle:
            proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True,
                                    start_new_session=True)
            try:
                try:
                    rc = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    rc = 124
            finally:
                terminate(proc)
        return rc, log.read_text()

    def tagged_call(self, label: str, command: list[str], *, stdin: str | None = None,
                    timeout: float = 120, inflight: int = 0) -> str:
        if '--cell' in command and command[1:3] == ['-m', 'offload.tv1.cells']:
            cell = cells.CELLS[command[command.index('--cell')+1]]
            timeout = max(timeout, 3*cell.estimate_s*cells.repetitions(cell)+600)
        if self.budget is not None:
            left = self.measure_by - self.clock()
            ck.require(left > 0, 'TV1 HOLD+73 measurement deadline reached; reserve recovery within +85')
            timeout = min(timeout, left)
        prefix = self.active.name if self.active is not None else 'tv1'
        return self.base_call(prefix + '-' + label, command, stdin=stdin, timeout=timeout, inflight=inflight)

    def preserve_engine_logs(self, reason: str, *, recovering: bool = False) -> list[str]:
        """Use unsupervised reads so STOP/start failure cannot discard the root cause.

        Errors stop a subsequent experiment, but must never prevent W8 recovery.
        Every rank is attempted independently, even if the other rank is missing.
        """
        ck.require(self.budget is not None, 'TV1 log capture requires original HOLD budget')
        self.engine_log_serial += 1
        directory = self.root_out / 'engine-logs' / f'{self.engine_log_serial:04d}-{reason}'
        errors: list[str] = []
        deadline = cast(HoldBudget, self.budget).restore if recovering else self.measure_by
        for host in pf.HOSTS:
            try:
                left = deadline - self.clock()
                ck.require(left > 0, 'TV1 no time for engine logs inside original deadline')
                rc, text = self.plain(f'engine-logs-{self.engine_log_serial}-{host}',
                    [*pf.SSH, host, 'python3 -c ' + shlex.quote(engine_logs.script())], min(30., left))
                ck.require(rc == 0, f'TV1 engine log transport rc{rc}')
                # Read each rank even when diagnostic storage is unavailable.
                # These writes must never escape into experiment/W8 recovery.
                directory.mkdir(parents=True, exist_ok=True)
                receipt = engine_logs.save(text, directory / f'{host}.log')
                ck.require(receipt.get('complete') is True,
                    f'TV1 incomplete engine logs: {receipt.get("error", "size cap exceeded")}')
            except Exception as exc:
                error = f'{reason}/{host}: {type(exc).__name__}: {exc}'
                errors.append(error)
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / f'{host}.error.txt').write_text(error + '\n')
                except Exception as receipt_exc:
                    errors.append(f'{reason}/{host}: error receipt write failed: '
                                  f'{type(receipt_exc).__name__}: {receipt_exc}')
        self.engine_log_errors.extend(errors)
        return errors

    @property
    def pin_files(self) -> tuple[str, ...]:
        return PIN_FILES

    def hold_purpose(self) -> str:
        return f'TV1-WINDOW {self.root_tag}: OFF/{self.feature}/OFF, default-OFF target experiments'

    @property
    def measure_by(self) -> float:
        ck.require(self.budget is not None, 'TV1 measurement requires original HOLD budget')
        return min(self.submit, cast(HoldBudget, self.budget).opened + schedule.WINDOW_S - schedule.RECOVERY_S)

    def preflight(self) -> None:
        """Current OPS1 reconciliation; historical PF1 policy stays unchanged."""
        if self.feature == 'candidate':
            tree = self.root_out / 'candidate-init-source-overlay'
            manifest = overlay.build(tree, ('candidate',))
            overlay.verify(tree, manifest)
            receipt = candidate_init_smoke.check(tree)
            receipt['inputs_lock_sha256'] = manifest['inputs_lock_sha256']
            (self.root_out / 'candidate-init-source-smoke.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if self.feature == 'compact':
            tree = self.root_out / 'compact-init-source-overlay'
            manifest = overlay.build(tree, ('compact',))
            overlay.verify(tree, manifest)
            receipt = init_smoke.check(tree)
            receipt['inputs_lock_sha256'] = manifest['inputs_lock_sha256']
            (self.root_out / 'compact-init-source-smoke.json').write_text(json.dumps(receipt, indent=2) + '\n')
        self.sup.begin()
        ck.require(pf.digest(pf.RESTORE) == pf.RESTORE_SHA256, 'W8 restore changed')
        from offload.prefill_trace.window_requests import corpus
        corpus(self.data_dir)
        ck.require((pf.CAMPAIGN / 'D2-GEN.PAUSE').exists() or subprocess.run(
            ['tmux', 'has-session', '-t', '=d2-gen'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0,
            'D2 generation active without pause')
        self.ssh('ledger', 'sparkhead', production.ledger_script())
        checkout = (f'cd ~/{shlex.quote(self.remote)} && test "$(git rev-parse HEAD)" = {self.expected} && '
                    'test -z "$(git status --porcelain --untracked-files=no)"')
        try:
            self.ssh('checkout', 'sparkhead', checkout)
        except ValueError:
            ck.require(self.stage_bundle, 'TV1 experiment checkout is not reviewed; stage bundle required')
            self.bundle()
            self.ssh('checkout-staged', 'sparkhead', checkout)
        pins = self.ssh('pins', 'sparkhead', f'cd ~/{shlex.quote(self.remote)} && sha256sum ' + shlex.join(self.pin_files))
        ck.require(pins.strip() == '\n'.join(f'{pf.digest(pf.ROOT / p)}  {p}' for p in self.pin_files), 'TV1 staged sources differ')
        self.ssh('prod-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/health')
        models = ck.rows(self.json('prod-models', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/v1/models')['data'])
        ck.require(len(models) == 1 and models[0].get('max_model_len') == 327680
                   and models[0].get('root') == '/srv/dsv41/legal-v2a', 'TV1 production model/cap mismatch')
        pf.worker(self.json('worker', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        for host in pf.HOSTS:
            rows = [record(cast(Json, json.loads(x))) for x in self.ssh(
                'telemetry-probe-' + host, host, 'python3 -c ' + shlex.quote(pf.TELEMETRY)
                + ' ' + shlex.quote(FIELDS) + ' 1').splitlines() if x.startswith('{')]
            ck.require([r.get('kind') for r in rows] == ['memory', 'gpu'], 'TV1 telemetry fields unsupported')
            pf.gpu_row(rows[1])
            ck.require(pf.memory_row(rows[0])[0] >= pf.START_FLOOR_KB, 'TV1 pre-window memory below 5GiB')
            if self.feature == 'candidate':
                probe = self.json('oracle-memory-prehold-' + host, host,
                                  'python3 -c ' + shlex.quote(oracle_memory.probe_script()))
                # Keep raw diagnostics even when the assessment refuses HOLD.
                (self.root_out / f'oracle-memory-prehold-{host}.probe.json').write_text(json.dumps(probe) + '\n')
                assessment = oracle_memory.assess(probe)
                (self.root_out / f'oracle-memory-prehold-{host}.json').write_text(json.dumps(assessment) + '\n')
                oracle_memory.require(assessment)
            text = self.ssh('profile-dir-' + host, host, f'test ! -e ~/{sp.PROFILE_DIR} && df -Pk ~ | tail -1')
            ck.require(int(text.split()[-3]) >= sp.PRE_DISK_KB, 'TV1 home disk below pre-window floor')

    def capture(self, host: str, label: str) -> dict[str, Json]:
        result = super().capture(host, label)
        config = self.json(label + '-all-env-' + host, host, "docker inspect --format '{{json .Config}}' dsv41-serve")
        env: dict[str, Json] = {}
        for entry in ck.strings(config['Env']):
            key, sep, value = entry.partition('=')
            ck.require(bool(sep) and key not in env, 'TV1 malformed/duplicate container env key')
            env[key] = value
        result['env'] = env
        return result

    def snapshot(self) -> None:
        super().snapshot()
        path = self.root_out / 'rollback.json'
        data = ck.read(path)
        data.update(production.rollback_metadata())
        path.write_text(json.dumps(data, indent=1) + '\n')

    def open_hold(self) -> float:
        ck.require(self.open_admitted is None, 'TV1 HOLD cannot be reopened or renewed')
        # S55: freeze AFTER preflight, at the one-shot submission. The opening
        # ceiling and placeholder lease share an origin; neither can be renewed.
        self.open_admitted = self.clock()
        self.open_deadline = self.open_admitted + pf.DRAIN_S + 60
        self.hold_scope_deadline = self.open_deadline + RESTORE_S
        tag = self.root_tag + '-hold'
        self.remember(tag, self.hold_scope_deadline)
        try:
            script = hold_admission.open_script(tag, self.admission_token, self.hold_purpose(), pf.DRAIN_S)
            self.ssh('hold-open', 'sparkhead', owner.command('launch', tag, ['bash', '-c', script],
                      deadline=self.hold_scope_deadline), timeout=pf.DRAIN_S + 60)
        finally:
            text = self.ssh('hold-read', 'sparkhead', hold_admission.command('check', tag, self.admission_token))
            self.hold_written = f'TV1-WINDOW {self.root_tag}:' in text
            if self.hold_written:
                opened = pf.hold_opened(text, self.root_tag, self.clock())
                self.budget = HoldBudget(opened)
                self.submit, self.sup.abort_at = self.budget.submission, self.budget.abort
                self.sup.bound = '86-minute TV1 abort bound; 12 minutes reserved for recovery'
                (self.root_out / 'deadlines.json').write_text(json.dumps({'opened': opened,
                    'submission': self.submit, 'abort': self.sup.abort_at, 'restore_by': self.budget.restore,
                    'measurement_by': self.measure_by, 'estimated_window_s': schedule.total_s(self.feature),
                    'planned_restore_by': opened + schedule.WINDOW_S}) + '\n')
        ck.require(self.hold_written and self.budget is not None, 'TV1 exact owned HOLD marker required')
        opened = pf.hold_opened(text, self.root_tag, self.clock())
        ck.require(opened <= self.open_deadline and cast(HoldBudget, self.budget).restore <= self.hold_scope_deadline,
                   'TV1 HOLD published after opening ceiling; placeholder lease would not cover recovery')
        self.sup.tunnel([*pf.SSH, '-N', '-o', 'ExitOnForwardFailure=yes', '-L', '127.0.0.1:18001:127.0.0.1:8001', 'sparkhead'])
        return opened

    def fence_phase(self, tag: str) -> None:
        def invoke(host: str, command: str) -> tuple[int, str]:
            mode = command.rsplit(' ', 2)[-2]
            label = f'{mode}-{tag}-{host}'
            argv = ['bash', '-c', command] if host == 'localhost' else [*pf.SSH, host, command]
            return self.plain(label, argv, 60)
        owner.fence_all(tag, invoke, OWNER_HOSTS)

    def recovery(self, label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        ck.require(self.budget is not None, 'TV1 owned recovery budget missing')
        self.recovery_serial += 1
        # W8 close requires this exact suffix for shared HOLD unlink deferral.
        tag = (f'{self.root_tag}-restore-close' if label == 'close'
               else f'{self.root_tag}-restore-{label}-{self.recovery_serial}')
        deadline = cast(HoldBudget, self.budget).restore - FINAL_CLOSE_RESERVE_S
        ck.require(self.clock() < deadline, 'TV1 recovery deadline reserve exhausted; HOLD stays open')
        self.remember(tag, deadline)
        try:
            return self.plain(label, ['bash', '-c', owner.command('launch', tag, argv, deadline=deadline)], timeout)
        finally:
            self.fence_phase(tag)

    def restore(self, *, production_stopped: bool) -> dict[str, Json]:
        # If the HOLD reply was unreadable, use the admission timestamp as a
        # conservative earlier bound. Never renew an existing HOLD to recover.
        budget = self.budget
        if budget is None:
            ck.require(self.open_admitted is not None, 'TV1 missing HOLD admission budget')
            budget = HoldBudget(cast(float, self.open_admitted))
            self.budget = budget
        original = self.plain
        def bounded(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
            cap = budget.cap(label, timeout, self.clock())
            result = original(label, argv, cap)
            ck.require(self.clock() < budget.restore, f'TV1 recovery deadline reached during {label}; HOLD stays open')
            return result
        self.plain = bounded
        try:
            result = self._restore_owned(production_stopped=production_stopped)
            result['restore_by'] = budget.restore
            result['restore_deadline_met'] = self.clock() < budget.restore
            return result
        finally:
            self.plain = original

    def _restore_owned(self, *, production_stopped: bool) -> dict[str, Json]:
        self.sup.close()
        (self.root_out / 'arm-memory.json').write_text(json.dumps(self.sup.arm_memory) + '\n')
        for tag in self.owner_tags:
            self.fence_phase(tag)
        out: dict[str, Json] = {}
        if not self.hold_written:
            rc, text = self.plain('hold-reread', [*pf.SSH, 'sparkhead', hold_admission.command('check', self.root_tag + '-hold', self.admission_token)], 60)
            ck.require(rc == 0, 'TV1 HOLD ownership unreadable; manual restore required')
            self.hold_written = f'TV1-WINDOW {self.root_tag}:' in text
        if not self.hold_written:
            ck.require(not production_stopped, 'TV1 production may be stopped without owned HOLD')
            return {'status': 'NOT NEEDED: no owned HOLD; remote mutators fenced and absent'}
        if production_stopped:
            if self.experiment_attempted:
                # All launchers have been fenced above. Preserve both ranks by
                # immutable container ID BEFORE stop.sh removes their logs.
                self.preserve_engine_logs('start-failed' if self.start_failed else 'restore-stop', recovering=True)
            # Docker-created processes leave the client scope. Stop only the
            # reviewed experiment, then independently prove containers/JIT/GPU quiet.
            rc, text = self.recovery('experiment-stop', [*pf.SSH, 'sparkhead',
                f'cd ~/{self.remote} && ./stop.sh </dev/null'], 120)
            ck.require(rc == 0, 'TV1 experiment stop failed; HOLD stays open')
            for host in pf.HOSTS:
                budget = cast(HoldBudget, self.budget)
                rc, text = self.plain('jit-wait-' + host, [*pf.SSH, host,
                    spark_quiet.command(QUIET, deadline=min(budget.restore - FINAL_CLOSE_RESERVE_S, self.clock()+350))], 360)
                ck.require(rc == 0 and QUIET in text.splitlines(),
                           f'{host}: experiment/JIT/compute absence unproved; HOLD stays open')
                out['jit_' + host] = 'PASS'
        for host in sorted(self.created):
            rc, text = self.recovery('profile-dir-' + host, [*pf.SSH, host,
                f'if [ -d ~/{sp.PROFILE_DIR} ]; then rmdir --ignore-fail-on-non-empty ~/{sp.PROFILE_DIR}; fi'], 60)
            ck.require(rc == 0, f'{host}: profile cleanup failed; HOLD stays open')
            out['profile_dir_' + host] = text.strip()
        steps = ('start', 'check', 'close') if production_stopped else ('check', 'close')
        for step in steps:
            rc, text = self.recovery(step, ['bash', str(pf.RESTORE), step],
                                     {'start': 900, 'check': 600, 'close': 120}[step])
            pf.restored(step, rc, text)
            out[step] = 'PASS'
        for tag in list(self.owner_tags):
            self.fence_phase(tag)
        # W8's final HOLD unlink was deferred by its registered close shim.
        # Only now, after every destination proved close-phase absence, commit
        # removal of the exact owned HOLD under the same persistent fence lock.
        budget = self.budget or HoldBudget(cast(float, self.open_admitted))
        ck.require(budget.restore - self.clock() >= FINAL_CLOSE_RESERVE_S,
                   'TV1 final close needs 90s reserve for commit/cancel/retention; HOLD stays open')
        cap = budget.cap('commit-close', FINAL_CLOSE_CALL_S, self.clock())
        tag = f'{self.root_tag}-restore-close'
        lease = min(budget.restore - 5, self.clock() + cap - 5)
        try:
            try:
                rc, text = self.plain('commit-close', [*pf.SSH, 'sparkhead',
                                     owner.command('commit-close', tag, deadline=lease)], cap)
                ck.require(rc == 0 and owner.CLOSED in text.splitlines(), 'TV1 final owned HOLD close unconfirmed')
            finally:
                cancel_rc, cancel_text = self.plain('cancel-close', [*pf.SSH, 'sparkhead', owner.command('cancel-close', tag)], FINAL_CLOSE_CALL_S)
                ck.require(cancel_rc == 0 and 'SCOPED CLOSE COMMIT FENCED' in cancel_text.splitlines(),
                           'TV1 final close fence unconfirmed; inspect HOLD before any further action')
        except (Exception, KeyboardInterrupt):
            # A lost acknowledgement can follow a completed unlink. Fence any
            # delayed commit and restore the original HOLD text/timestamp,
            # under the same lock/lease; never overwrite another owner's HOLD.
            remaining = budget.cap('retain-close', FINAL_CLOSE_CALL_S, self.clock())
            retained_rc, retained = self.plain('retain-close', [*pf.SSH, 'sparkhead',
                owner.command('retain-close', tag, deadline=min(budget.restore, self.clock()+remaining-1))], remaining)
            ck.require(retained_rc == 0 and owner.RETAINED in retained.splitlines(),
                       'TV1 original HOLD retention unconfirmed; coordinator must inspect HOLD')
            raise
        out['status'] = 'RESTORED: K1 START VERIFIED, greedy 6/6 IDENTICAL, PRODUCTION HOLD OK, window closed'
        out['engine_log_errors'] = list(self.engine_log_errors)
        return out

    def stage(self) -> None:
        base = f'offload/tv1/generated-{self.root_tag}'
        self.ssh('stage-dir', 'sparkhead', f'cd ~/{shlex.quote(self.remote)} && test ! -e {base} && mkdir {base}')
        for role in ('off', self.feature):
            local = self.root_out / 'generated' / role
            manifest = overlay.build(local, () if role == 'off' else (role,))
            overlay.verify(local, manifest)
            self.manifests[role] = manifest
            sidecar = local.with_suffix('.manifest.json')
            sidecar.write_text(json.dumps(manifest, indent=2) + '\n')
            self.call(f'stage-{role}', ['scp', '-rq', '-o', 'BatchMode=yes', str(local), str(sidecar),
                                       f'sparkhead:{self.remote}/{base}/'])
            code = ('import hashlib,json,pathlib\n' + f'root=pathlib.Path({base + "/" + role!r})\n'
                    'm=json.loads(root.with_suffix(".manifest.json").read_text())\n'
                    f'assert m=={manifest!r},"TV1 staged manifest drift"\n'
                    'for e in m["files"]:\n'
                    ' assert hashlib.sha256((root/e["path"]).read_bytes()).hexdigest()==e["sha256"],e["path"]\n'
                    'print("TV1 staged hashes verified")\n')
            text = self.ssh(f'stage-verify-{role}', 'sparkhead', f'cd ~/{shlex.quote(self.remote)} && python3 -', code)
            ck.require('TV1 staged hashes verified' in text, 'TV1 remote stage was not hash-verified')

    def arm_env(self, arm: pf.Arm) -> str:
        selected = cast(Arm, arm)
        return (f' TV1_ROLE={selected.role} TV1_TREE=offload/tv1/generated-{self.root_tag}/{selected.role}'
                + (f' TV1_ORACLE_RUN={shlex.quote(self.root_tag)}' if selected.role == 'candidate' else ''))

    def runtime_check(self, rank: int, live: dict[str, Json], arm: pf.Arm) -> None:
        selected = cast(Arm, arm)
        runtime(self.production[rank], live, selected, self.manifests[selected.role], self.run_name(arm))

    def verify(self, host: str, label: str, trace: int, *, cause: bool = False, fix: bool = False) -> None:
        if label == 'prod':
            super().verify(host, label, trace, cause=cause, fix=fix)
            return
        ck.require(self.active is not None, 'TV1 missing active arm')
        manifest = self.manifests[cast(Arm, self.active).role]
        text = self.ssh(label + '-tv1-hashes-' + host, host, 'docker exec -i dsv41-serve python3 -',
                        source_verifier(manifest))
        ck.require('TV1 verified' in text, 'TV1 runtime hash verifier did not pass')

    def driver(self, *extra: str) -> list[str]:
        argv = list(extra)
        if '--telemetry' in argv:
            i = argv.index('--telemetry')
            argv[i+1:i+3] = [str(self.root_out / f'telemetry-{host}.jsonl') for host in pf.HOSTS]
        if '--deadline-utc' in argv:
            argv[argv.index('--deadline-utc')+1] = f'{self.measure_by:.3f}'
        return [sys.executable, '-m', 'offload.tv1.cells', '--execute', '--data-dir', str(self.data_dir), *argv]

    def begin(self, arm: Arm) -> None:
        ck.require(self.budget is not None, 'TV1 start requires original HOLD budget')
        schedule.admit(self.feature, arms(self.feature).index(arm),
                       opened=cast(HoldBudget, self.budget).opened, now=self.clock())
        if self.experiment_attempted:
            errors = self.preserve_engine_logs('before-' + arm.name)
            ck.require(not errors, 'TV1 engine log preservation failed before next stop: ' + '; '.join(errors))
        self.active = arm
        self.never_started = None
        self.manifest = []
        self.node_memory(arm.name)
        for host in pf.HOSTS:
            self.ssh('profile-dir-' + host, host, f'mkdir -p ~/dsv41-offload-run && mkdir ~/{sp.PROFILE_DIR}')
            self.created.add(host)
        self.sup.arm_minima = {}
        self.sup.arm_memory[arm.name] = self.sup.arm_minima
        self.experiment_attempted = True  # set before submission, including lost replies
        try:
            pf.Live.start(self, arm)
        except (Exception, KeyboardInterrupt):
            self.start_failed = True
            raise  # recovery fences this phase, captures both ranks, then stops
        for host in pf.HOSTS:
            mounts = ck.rows(json.loads(self.ssh('mounts-' + host, host,
                            "docker inspect --format '{{json .Mounts}}' dsv41-serve")))
            sp.trace_mount(mounts)
            for entry in self.manifests[arm.role]['files']:
                found = [m for m in mounts if m.get('Destination') == pf.DIST + entry['path']]
                ck.require(len(found) == 1 and found[0].get('Type') == 'bind' and found[0].get('RW') is False,
                           'TV1 duplicate/missing/writable feature mount')
            files, free = sp.listing(self.ssh('empty-profile-' + host, host, self.find()))
            ck.require(not files and free >= sp.DISK_FLOOR_KB, 'TV1 profile directory not empty or disk too low')
            ck.require(self.log_hits(host, sp.PROFILER_STARTED) == 0, 'TV1 profiler already started')
        ck.require(self.log_hits('sparkhead', sp.PROFILER_ON) >= 1, 'TV1 profiler endpoints absent')
        (self.root_out / 'arms' / arm.name).mkdir(parents=True)
        if arm.role == 'candidate':
            # Completed before server startup by the scoped launcher's hook.
            # Never create a competing CUDA context in a live server container.
            result = self.json('prestart-oracles', 'sparkhead',
                'cat ~/dsv41-offload-run/' + oracle_prestart.receipt_path(self.root_tag).name)
            oracle_prestart.validate(result, self.root_tag)
            (self.root_out / 'arms' / arm.name / 'prestart-oracles.json').write_text(json.dumps(result) + '\n')

    def qualify(self, arm: Arm) -> dict[str, Json]:
        out = self.root_out / 'arms' / arm.name / 'qualification.json'
        argv = [sys.executable, '-m', 'offload.tv1.qualification', '--execute', '--data-dir', str(self.data_dir),
                '--cuts', str(self.cuts), '--deadline-utc', str(self.measure_by),
                '--telemetry', *(str(self.root_out / f'telemetry-{host}.jsonl') for host in pf.HOSTS),
                '--capture-floor-kb', str(sp.CAPTURE_FLOOR_KB), '--cache-salt-prefix', self.root_tag+'-'+arm.name,
                '--output', str(out)]
        self.call('candidate-qualification', argv, timeout=qualification.ESTIMATE_S + 120, inflight=4)
        traffic = ck.read(out)
        ck.require(traffic.get('type') == 'tv1-qualification-traffic-v2' and traffic.get('qualification_only') is True,
                   'TV1 qualification traffic did not finish')
        receipts: dict[str, Json] = {}
        for host in pf.HOSTS:
            result = self.json('engine-oracle-' + host, host, 'python3 -c ' + shlex.quote(engine_oracle_script()))
            validate_engine_receipts(result.get('records'))
            receipts[host] = result['records']
        (out.parent / 'engine-oracle.json').write_text(json.dumps(receipts) + '\n')
        return {'traffic': str(out.relative_to(self.root_out)), 'both_rank_metadata_identical': True}

    def identities(self, arm: Arm) -> dict[str, Json]:
        out = self.root_out / 'arms' / arm.name / 'greedy.json'
        try:
            self.call('greedy', self.driver('greedy', '--deadline-utc', str(self.submit), '--output', str(out)),
                      timeout=300, inflight=1)
            result = ck.read(out)
            greedy.validate(result)
        except (Exception, KeyboardInterrupt) as exc:
            if out.is_file():
                raise greedy.Failure(ck.read(out), exc) from exc
            raise
        return result

    def run_cell(self, arm: Arm, name: str) -> dict[str, Json]:
        cell = cells.CELLS[name]
        ck.require(name in cells.order(self.feature), 'TV1 cell outside fixed feature schedule')
        ck.require(self.clock()+cell.estimate_s*cells.repetitions(cell) <= self.measure_by,
                   'TV1 full repeated cell cannot fit before submission deadline')
        previous_out, previous_tag = self.out, self.tag
        self.out, self.tag = self.root_out / 'arms' / arm.name, self.root_tag + '-' + arm.name
        try:
            result = super().cell(name)
            if cells.CELLS[name].captures:
                captures = ck.rows(result['captures'])
                ck.require(len(captures) == 1 and captures[0].get('kind') == 'decode'
                           and captures[0].get('status') in ('TAKEN', 'ENDED'), 'TV1 required decode capture skipped/incomplete')
            return result
        finally:
            self.out, self.tag = previous_out, previous_tag

    def finish_arm(self, arm: Arm) -> dict[str, Json]:
        old = self.out
        self.out = self.root_out / 'arms' / arm.name
        try:
            return super().finish()
        finally:
            self.out = old


class Backend(Protocol):
    def preflight(self) -> None: ...
    def open_hold(self) -> float: ...
    def snapshot(self) -> None: ...
    def stage(self) -> None: ...
    def begin(self, arm: Arm) -> None: ...
    def identities(self, arm: Arm) -> dict[str, Json]: ...
    def qualify(self, arm: Arm) -> dict[str, Json]: ...
    def build(self) -> dict[str, Json]: ...
    def run_cell(self, arm: Arm, name: str) -> dict[str, Json]: ...
    def finish_arm(self, arm: Arm) -> dict[str, Json]: ...
    def restore(self, *, production_stopped: bool) -> dict[str, Json]: ...


def window(backend: Backend, feature: str, checkpoint: Callable[[dict[str, Json]], None]) -> dict[str, Json]:
    result: dict[str, Json] = {'status': 'INCOMPLETE', 'feature': feature, 'design': 'C/arm/C-v4', 'starts': {},
                             'registration_id': greedy.REGISTRATION_ID, 'greedy_rule': greedy.RULE}
    hold = stopped = False
    try:
        selected = arms(feature)
        backend.preflight()
        hold = True  # open may time out AFTER writing HOLD; always attempt W8 recovery
        result['hold_opened'] = backend.open_hold()
        backend.snapshot()
        backend.stage()
        for index, arm in enumerate(selected):
            stopped = True
            backend.begin(arm)
            row: dict[str, Json] = {'role': arm.role, 'cells': {}}
            record(result['starts'])[arm.name] = row
            try:
                row['greedy'] = backend.identities(arm)
            except greedy.Failure as exc:
                row['greedy'] = exc.receipt
                raise
            if index == 0:
                result['prompts'] = backend.build()
            if arm.role == 'candidate':
                row['qualification'] = backend.qualify(arm)
            for name in cells.order(feature):
                record(row['cells'])[name] = backend.run_cell(arm, name)
                checkpoint({'arm': arm.name, 'cell': name})
            row['checks'] = backend.finish_arm(arm)
        result['status'] = 'COMPLETE'
    except (Exception, KeyboardInterrupt) as exc:
        result['stop'] = f'{type(exc).__name__}: {exc}'
    finally:
        saved = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            if hold:
                try:
                    result['restore'] = backend.restore(production_stopped=stopped)
                    if record(result['restore']).get('engine_log_errors'):
                        result['status'] = 'INCOMPLETE'
                        result['engine_log_errors'] = record(result['restore'])['engine_log_errors']
                except (Exception, KeyboardInterrupt) as exc:
                    result['restore'] = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
            else:
                result['restore'] = {'status': 'NOT NEEDED: no HOLD written, production untouched'}
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
    return result


def plan(feature: str) -> str:
    return ('DRY RUN: no command, network or file. Coordinator Deepseek/coordinator only.\n'
            + f'Production {production.PROD_COMMIT} / {production.PROD_TAG}; exact OFF env, Engram REQUIRED=0.\n'
            + 'Fresh starts: ' + ' -> '.join(a.name for a in arms(feature)) + '\n'
            + f'Experimental greedy registration: {greedy.REGISTRATION_ID}; {greedy.RULE}.\n'
            + 'Each start: six indexed K1 checks via unchanged NBW checker; warmup; '
            + ', '.join(cells.controls(feature)) + ' controls and matching two-rank decode profiles.\n'
            + f'ESTIMATED HOLD incl recovery: {schedule.total_s(feature)/60:.2f}min; '
            + f'five starts {schedule.total_s(feature, five=True)/60:.2f}min exceed85.\n'
            'Fixed C/arm/C: gain must pass against EACH OFF independently; OFF drift and four-request pooling mandatory.\n'
            'Decision: unprofiled server decode ms/spec request-round; >=4 completed requests/cell/start, 512 outputs each.\n'
            'Candidate target150K B1; compact target32K B4 round time AND aggregate tok/s; profiles diagnostic.\n'
            'Measurement ends+73min, reserving12min recovery within+85; remaining-schedule admission refuses over-budget starts.\n'
            'Original submit+85/abort+86/recovery+98 remain immutable outer safety ceilings.\n'
            'Candidate retains8GiB floor,288-case exact installed gate and735s actual-forward qualification+240s oracle allowance.\n'
            'Shared OWN1 scopes+nonce/inode HOLD admission; fence/reap/prove ALL four destinations before close.\n'
            'After verified close: shared15min health watch, at most one scoped W8 recovery.\n'
            'ALWAYS after HOLD: W8/restore.sh start(if stopped),check,close; exactly six indexed IDENTICAL lines(S48).\n'
            'No automatic promotion. Apply offload/tv1/WINDOW.md pre-registered numerical and timing gates.\n')


def post_close(out: Path, tag: str) -> dict[str, Json]:
    def log(message: str) -> None:
        print(message, flush=True)
        with (out / 'post-close-watch.log').open('a') as handle:
            handle.write(message + '\n')
    watched = watchdog.watch(watchdog.ProductionHealth(),
        watchdog.ScopedRecovery(tag + '-watch', out / 'post-close-commands'), log=log)
    return cast(dict[str, Json], dict(watched.__dict__))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature', choices=('candidate', 'compact'), default='candidate')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--out', type=Path)
    p.add_argument('--expected-experiment-commit')
    p.add_argument('--remote-repo', default='dsv41-deploy-offload')
    p.add_argument('--stage-bundle', action='store_true')
    p.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    a = p.parse_args(argv)
    if not a.execute:
        print(plan(a.feature), end='')
        return 0
    try:
        ck.require(not a.dry_run, 'choose --execute OR --dry-run')
        ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'TV1 coordinator identity required')
        ck.require(isinstance(a.out, Path) and not a.out.exists(), 'TV1 fresh --out required')
        ck.require(re.fullmatch('[0-9a-f]{40}', a.expected_experiment_commit or '') is not None,
                   'TV1 reviewed full commit required')
        ck.require(re.fullmatch('dsv41-deploy-[A-Za-z0-9_-]+', a.remote_repo) is not None, 'TV1 isolated remote checkout required')
        local = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pf.ROOT, text=True).strip()
        ck.require(local == a.expected_experiment_commit, 'TV1 local HEAD is not reviewed commit')
        ck.require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                              cwd=pf.ROOT, text=True).strip(), 'TV1 tracked source dirty')
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP before any remote action: {exc}', file=sys.stderr)
        return 1
    out = cast(Path, a.out)
    out.mkdir()
    tag = re.sub('[^A-Za-z0-9_-]', '-', out.name)
    def interrupted(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f'signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    def checkpoint(row: dict[str, Json]) -> None:
        with (out / 'checkpoints.jsonl').open('a') as handle:
            handle.write(json.dumps({'time': time.time(), **row}, allow_nan=False) + '\n')
    sup = pf.Supervisor(out)
    live = TV1Live(out, sup, remote=a.remote_repo, expected=a.expected_experiment_commit, data_dir=a.data_dir,
                   tag=tag, stage_bundle=a.stage_bundle, feature=a.feature)
    result = window(live, a.feature, checkpoint)
    sup.close()
    if str(record(result['restore']).get('status', '')).startswith('RESTORED'):
        result['post_close_watch'] = post_close(out, tag)
    result.update({'experiment_commit': a.expected_experiment_commit, 'restore_sha256': pf.RESTORE_SHA256,
                   'worker_source': pf.WORKER,
                   'analysis_sha256': pf.digest(overlay.HERE / 'analyze.py'),
                   'artifact_sha256': {str(p.relative_to(out)): pf.digest(p) for p in sorted(out.rglob('*'))
                        if p.is_file() and p.suffix in ('.json', '.jsonl') and not p.name.startswith('telemetry-')}})
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    restored = str(record(result['restore']).get('status', ''))
    print(json.dumps({'status': result['status'], 'restore': restored, 'stop': result.get('stop')}))
    if 'post_close_watch' in result and record(result['post_close_watch']).get('status') != 'PASS':
        print('POST-CLOSE PRODUCTION HEALTH FAILED; inspect watchdog log and HOLD before further work.', file=sys.stderr)
        return 2
    if not restored.startswith(('RESTORED', 'NOT NEEDED')):
        print('PRODUCTION NOT VERIFIED. Inspect restore logs and HOLD; a final close reply may be ambiguous. '
              'Coordinator must verify before further action and use W8 start/check/close as needed.',
              file=sys.stderr)
        return 2
    return 1 if 'stop' in result else 0


if __name__ == '__main__':
    raise SystemExit(main())
