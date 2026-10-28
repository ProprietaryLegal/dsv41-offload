"""Six-probe qualification HOLD; no headline decision or promotion.

Imported lazily by pp4_window. Shared ownership, original HOLD deadline, strict
W8 restoration and post-close watch are inherited, never reimplemented here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import pp4_admission_prepare as prep
from offload.launch.windows import pp4_budget as budget
from offload.launch.windows import pp4_cells as cells
from offload.launch.windows import pp4_prerequisites as prereq
from offload.launch.windows import pp4_window as w
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record

GATES = ('both_rank_q_failure_target_only', 'grammar_penalty_stop_eos', 'tool_json_reasoning_parser',
         'cancel_reset_reconnect_timeout_late_reply', 'gb10_image_and_abi_reconciliation',
         'complete_round_instrumentation')
SEED_GATES = tuple(name for name in prereq.GATES if name not in GATES)
MAX_HOLD_S = 85*60
# Historical MEASURED worker333s/Spark388–404s inputs; all other costs are
# explicit ESTIMATED allowances, not measured PP4 complete blocks.
DEFAULT_ESTIMATE: dict[str, Json] = {'setup_s': 120, 'pair_s': 600,
    'worker_s': 333, 'engine_s': 404, 'warm_s': 30, 'greedy_s': 42,
    'cuts_s': 120, 'probes_s': 1800, 'profile_s': 320, 'transition_s': 60, 'restore_s': 720}


def estimate(data: dict[str, Json]) -> dict[str, Json]:
    plan = record(data.get('admission_estimate', DEFAULT_ESTIMATE))
    ck.require(set(plan) == set(DEFAULT_ESTIMATE), 'admission estimate must price every registered probe/start/profile/restore component')
    values = {key: number(value) for key, value in plan.items()}
    ck.require(all(math.isfinite(value) and value >= number(DEFAULT_ESTIMATE[key]) for key, value in values.items()),
               'admission costs must retain registered positive/finite allowances')
    historical = budget.estimate()
    ck.require(values['pair_s'] >= number(record(historical['pair'])['seconds']), 'admission pair reserve below measured qualification duration')
    total = sum(values.values())
    ck.require(total <= MAX_HOLD_S, f'admission ESTIMATED {total/60:.2f}min exceeds85min; refuses before host contact')
    return {**plan, 'total_s': total, 'historical_budget_sha256': historical['source_sha256'], 'pair': historical['pair'], 'classification': 'ESTIMATED from historical MEASURED startups; no calibration blocks'}


def seed(path: Path, expected: str) -> dict[str, Json]:
    data = prereq.staging(path, expected)
    gates = record(data['gates'])
    ck.require(set(gates) == set(SEED_GATES), 'admission requires exactly the six existing reviewed gates; cannot seed missing six')
    for name in SEED_GATES:
        checked_gate(record(gates[name]), name)
    ordinary = record(ck.read(pf.ROOT/'offload/dspark/probabilistic/pp4/verifier-k1-receipt.json')['output_sha256'])
    ck.require(data.get('verifier_sha256') == ordinary, 'ordinary K1 verifier receipt differs')
    outputs = record(data['admission_verifier_sha256'])
    # Round 16: the admission remote_dspark also carries the client seam (package-relative, torch-free proxy).
    expected_outputs = {*record(data['verifier_sha256']), prep.HELPER, 'remote_dspark/'+prep.SEAM,
        *('remote_dspark/'+p.name for p in (pf.ROOT/'offload/dspark/spark/remote_dspark').glob('*.py'))}
    ck.require(set(outputs) == expected_outputs and all(re.fullmatch('[0-9a-f]{64}', str(v)) for v in outputs.values()),
               'exact separate admission helper/client overlay hashes required')
    data['admission_budget'] = estimate(data)
    return data


def checked_gate(row: dict[str, Json], name: str) -> None:
    path = Path(str(row.get('evidence_path', '')))
    ck.require(row.get('pass') is True and path.is_absolute() and path.is_file()
               and pf.digest(path) == row.get('sha256'), name+': primary evidence absent/changed')


def write(path: Path, data: dict[str, Json]) -> None:
    path.write_text(json.dumps(data, indent=1, allow_nan=False)+'\n')


class Backend(Protocol):
    out: Path
    expected: str
    tag: str
    admission: dict[str, Json]
    def probes(self, during: Callable[[], None]) -> dict[str, Json]: ...
    def reconcile(self) -> dict[str, Json]: ...
    def q_failures(self) -> dict[str, Json]: ...
    def api_gate(self, case: str) -> dict[str, Json]: ...
    def lifecycle(self) -> dict[str, Json]: ...


class Session:
    def __init__(self, backend: Backend, checkpoint: Callable[[dict[str, Json]], None]) -> None:
        ck.require(isinstance(backend.tag, str) and bool(backend.tag.strip()), 'nonempty backend admission run required')
        self.b = backend
        self.checkpoint = checkpoint
        self.current = 'physical_pair'
        self.data: dict[str, Json] = {**backend.admission, 'schema': 'pp4-live-admission-v1', 'depth': 5,
            'source_commit': backend.expected, 'admission_run': backend.tag,
            'verdict': 'NOT QUALIFIED', 'failing_gate': 'physical_pair',
            'gates': dict(record(backend.admission['gates'])), 'retired_gates': prereq.RETIRED_GATES,
            'headline_measurements': False}
        self.save()

    def save(self) -> None:
        write(self.b.out/'admission-receipt.json', self.data)

    def gate(self, name: str, operation: Callable[[], dict[str, Json]]) -> None:
        self.current = name
        self.data['failing_gate'] = name
        self.save()  # failure/interrupt leaves the current gate visible
        evidence = operation()
        ck.require(evidence.get('complete') is True, name+': incomplete primary evidence')
        path = self.b.out/(name+'.json')
        write(path, {'source_commit': self.b.expected, **evidence})
        record(self.data['gates'])[name] = {'pass': True, 'evidence_path': str(path.resolve()), 'sha256': pf.digest(path)}
        self.save()
        self.checkpoint({'admission_gate': name, 'status': 'PASS', 'sha256': pf.digest(path)})

    def run(self) -> dict[str, Json]:
        self.current = 'admission_on_start'
        self.data['failing_gate'] = self.current
        self.save()
        def on_probes() -> None:
            self.gate('gb10_image_and_abi_reconciliation', self.b.reconcile)
            self.gate('both_rank_q_failure_target_only', self.b.q_failures)
            self.gate('grammar_penalty_stop_eos', lambda: self.b.api_gate('semantics'))
            self.gate('tool_json_reasoning_parser', lambda: self.b.api_gate('tools'))
            self.gate('cancel_reset_reconnect_timeout_late_reply', self.b.lifecycle)
            self.current = 'complete_round_instrumentation'
            self.data['failing_gate'] = self.current
            self.save()
        b = self.b.probes(on_probes)
        def instrumentation() -> dict[str, Json]:
            profiles = record(b['profiles'])
            ck.require(set(profiles) == set(cells.CELLS), 'missing complete-round cells')
            for value in profiles.values():
                row = record(value)
                ck.require(set(record(row['by_family'])) == set(cells.FAMILIES), 'missing complete-round family')
                for value in record(row['by_family']).values():
                    r = record(value)
                    ck.require(number(r['profiled_rounds']) >= 16 and number(r['p95_round_ms']) > 0,
                               'insufficient complete both-rank round instrumentation')
            return {'complete': True, 'profiles': profiles, 'raw_artifacts': self.artifacts(),
                    'scope': 'ON admission instrumentation only; no A/B calibration or headline measurements'}
        self.gate('complete_round_instrumentation', instrumentation)
        self.current = 'restore_and_post_close_watch'
        self.data['failing_gate'] = self.current
        self.save()
        return {'status': 'GATES_COMPLETE_PENDING_RESTORE', 'headline_measurements': False}

    def artifacts(self) -> dict[str, Json]:
        return {str(p.relative_to(self.b.out)): pf.digest(p) for p in sorted(self.b.out.rglob('*'))
                if p.is_file() and p.name not in ('admission-receipt.json', 'checkpoints.jsonl', 'summary.json')}

    def finish(self, result: dict[str, Json]) -> None:
        gates = record(self.data['gates'])
        restored = record(result['restore'])
        qualified = ('stop' not in result and set(gates) == set(prereq.GATES)
            and restored.get('status') == 'RESTORED: K1 START VERIFIED, greedy 6/6 IDENTICAL, PRODUCTION HOLD OK, window closed'
            and restored.get('restore_deadline_met') is True
            and record(result.get('post_close_watch', {})).get('status') == 'PASS')
        for name, value in gates.items():
            try:
                checked_gate(record(value), name)
            except (ValueError, OSError) as exc:
                qualified = False
                self.current = name
                result['stop'] = str(exc)
        # Bind measurement's pair-duration input to THIS admission's primary
        # physical step. A prior pair-only/solo summary cannot price this HOLD.
        try:
            from offload.dspark.probabilistic.pp4.pair_contract import validate
            physical = record(result['physical_pair'])
            path = self.b.out/'physical-pair-qualification.json'
            ck.require(self.data.get('admission_run') == self.b.tag, 'parent admission run differs from backend')
            ck.require(path.resolve().parent == self.b.out.resolve(), 'admission physical-pair primary directory differs')
            ck.require(ck.read(path) == physical, 'admission physical-pair primary differs')
            validate(physical, self.b.expected, self.b.tag)
            duration = number(physical['seconds'])
            ck.require(math.isfinite(duration) and duration > 0, 'admission pair duration absent')
            self.data['physical_pair_evidence'] = {'pass': True, 'evidence_path': str(path.resolve()),
                'sha256': pf.digest(path), 'run': physical['run']}
        except (KeyError, ValueError, OSError) as exc:
            qualified = False
            # Preserve an earlier semantic/probe failure as the primary cause.
            if 'stop' not in result:
                self.current = 'physical_pair'
                result['stop'] = str(exc)
        self.data.update({'verdict': 'QUALIFIED' if qualified else 'NOT QUALIFIED',
            'failing_gate': None if qualified else self.current, 'stop': result.get('stop'),
            'restore': restored, 'post_close_watch': result.get('post_close_watch'), 'artifact_sha256': self.artifacts()})
        self.save()


class Live(w.Live):
    """Use real inherited starts/cells/cleanup; only add owned probes."""
    restore_seconds = MAX_HOLD_S

    def preflight(self) -> None:
        super().preflight()
        outputs = record(self.admission['admission_verifier_sha256'])
        manifest = '\n'.join(f'{sha}  offload/launch/windows/pp4-admission-staged/verifier/{name}' for name, sha in outputs.items())
        self.ssh('admission-prepared', 'sparkhead', f'cd ~/{self.remote} && printf %s '+shlex.quote(manifest+'\n')+' | sha256sum -c -')

    def reconcile(self) -> dict[str, Json]:
        from offload.dspark.probabilistic.pp3.prepare_verifier import ABI
        ranks: list[Json] = []
        for rank, host in enumerate(pf.HOSTS):
            self.verify_overlay(host, 'admission')
            script = ('import json,torch; from vllm.v1.worker.gpu import model_runner as r; '
                'from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as s; '
                f'assert r.REMOTE_DRAFT_PROBS_ABI == s.REMOTE_DRAFT_PROBS_ABI == {ABI!r}; '
                'assert torch.cuda.get_device_capability()==(12,1); '
                'print(json.dumps({"abi":r.REMOTE_DRAFT_PROBS_ABI,"capability":[12,1],"torch":torch.__version__}))')
            # Read/import under owned docker-exec seam INSIDE HOLD, never precheck.
            text = self.owned('admission-abi-'+host, host, self.tag+'-abi-'+host,
                'docker exec dsv41-serve python3 -c '+shlex.quote(script), timeout=60)
            info = record(cast(Json, json.loads(text)))
            capture = self.capture(host, 'admission-reconcile')
            w.runtime(self.production[rank], capture, w.Arm('admission-B', 327680, 0, role='B', admission=True), self.native, self.overlay, admission_run=self.tag+'-admission-B')
            ranks.append({'rank': rank, 'abi': info, 'container': capture})
        return {'complete': True, 'ranks': ranks, 'source_sha256': self.overlay}

    def api_gate(self, case: str) -> dict[str, Json]:
        output = self.out/('api-'+case+'-'+str(self.clock()).replace('.', '-')+'.json')
        extra: list[str] = []
        eos: int | None = None
        directory = f'dsv41-offload-run/pp4-round/{self.tag}-{self.probe_arm.name}'
        armed: list[str] = []
        try:
            if case == 'semantics':
                text = self.ssh('admission-model-config', 'sparkhead', 'docker exec dsv41-serve cat /srv/dsv41/legal-v2a/config.json')
                config = record(cast(Json, json.loads(text)))
                value = config['eos_token_id']
                ck.require(isinstance(value, int), 'one verified EOS token required')
                eos = int(number(value))
                extra = ['--eos', str(eos)]
                write(self.out/'admission-eos-source.json', {'config_sha256': hashlib.sha256(text.encode()).hexdigest(), 'eos': eos})
                payload = json.dumps({'run': self.tag+'-'+self.probe_arm.name, 'eos': eos,
                                      'expires': min(self.submit, self.clock()+180)})
                for host in pf.HOSTS:
                    armed.append(host)
                    self.owned('admission-eos-'+host, host, self.tag+'-eos-'+host,
                        f'printf %s {shlex.quote(payload)} > ~/{directory}/EOS.json', timeout=30)
            self.call('admission-api-'+case, [prereq.PYTHON, '-m', 'offload.launch.windows.pp4_admission_requests',
                '--execute', '--case', case, '--out', str(output), '--deadline', str(min(self.submit, self.block_deadline)), *extra],
                timeout=180 if case != 'tools' else 300, inflight=1)
            data = ck.read(output)
            ck.require(data.get('complete') is True, 'incomplete API qualification')
            if eos is not None:
                records = self.engine_events('eos')
                for rank in (0, 1):
                    ck.require(any(row.get('kind') == 'model_eos_sampled' and row.get('rank') == rank
                                   and row.get('eos') == eos and row.get('observed') is True for row in records),
                               'missing actual sampled EOS on rank'+str(rank))
                data['actual_eos'] = cast(Json, [row for row in records if row.get('kind') == 'model_eos_sampled'])
            return data
        finally:
            errors: list[str] = []
            for host in armed:
                rc, _ = self.recovery('admission-eos-off-'+host, [*pf.SSH, host, f'rm -f ~/{directory}/EOS.json'], 30)
                if rc != 0: errors.append(host)
            ck.require(not errors, 'EOS observation cleanup failed: '+','.join(errors))

    def engine_events(self, label: str) -> list[dict[str, Json]]:
        directory = f'dsv41-offload-run/pp4-round/{self.tag}-{self.probe_arm.name}'
        rows: list[dict[str, Json]] = []
        for host in pf.HOSTS:
            script = ("from pathlib import Path; p=Path.home()/"+repr(directory)+"; "
                "files=sorted(p.glob('admission-*.jsonl')); assert files; "
                "assert sum(f.stat().st_size for f in files)<8388608; "
                "print(''.join(f.read_text() for f in files),end='')")
            text = self.ssh('admission-events-'+label+'-'+host, host, 'python3 -c '+shlex.quote(script))
            path = self.out/('events-'+label+'-'+host+'.jsonl'); path.write_text(text)
            rows.extend(record(cast(Json, json.loads(line))) for line in text.splitlines())
        ck.require(all(row.get('run') == self.tag+'-'+self.probe_arm.name for row in rows), 'engine event run differs')
        return rows

    def fault(self, mode: str) -> dict[str, Json]:
        arm = self.probe_arm
        directory = f'dsv41-offload-run/pp4-round/{self.tag}-{arm.name}'
        identifier = self.tag+'-'+mode
        payload = json.dumps({'run': self.tag+'-'+arm.name, 'id': identifier, 'mode': mode,
                              'expires': min(self.submit, self.clock()+180)})
        before = self.stats('admission-fault-before-'+mode)
        armed: list[str] = []
        try:
            for host in pf.HOSTS:
                armed.append(host)
                self.owned('admission-fault-'+mode+'-'+host, host, self.tag+'-fault-'+mode+'-'+host,
                    f'printf %s {shlex.quote(payload)} > ~/{directory}/FAULT.json', timeout=30)
            response = self.api_gate('probe')
            # Several fresh requests allow background HELLO/resync to complete;
            # the recovery evidence must advance epoch, not merely return HTTP200.
            for _ in range(3):
                self.api_gate('probe')
            rows = self.engine_events(mode)
            selected = [row for row in rows if row.get('id') == identifier]
            if mode.startswith('q-rank'):
                for rank in (0, 1):
                    ck.require(any(row.get('kind') == 'q_invalidated' and row.get('rank') == rank and row.get('all_invalid') is True for row in selected),
                               mode+': missing rank invalidation')
                    ck.require(any(row.get('kind') == 'target_only_exact' and row.get('rank') == rank and row.get('exact') is True
                                   and row.get('lengths_one') is True for row in selected), mode+': missing exact target-only distribution coupling')
                emitted = {rank: [(row['emitted_sha256'], row['requests']) for row in selected
                                  if row.get('kind') == 'target_only_exact' and row.get('rank') == rank] for rank in (0, 1)}
                ck.require(emitted[0] == emitted[1], mode+': TP ranks disagree on exact target-only samples')
            else:
                applied = [row for row in selected if row.get('kind') == 'client_fault']
                recovered = [row for row in selected if row.get('kind') == 'client_recovered']
                ck.require(len(applied) == 1 and bool(recovered), mode+': no real client fault/recovery evidence')
                ck.require(any(number(row['epoch']) > number(applied[0]['epoch']) for row in recovered), mode+': no new HELLO epoch')
                if mode == 'reset':
                    ck.require(applied[0]['session_before'] != applied[0]['session_after'], 'reset failed to change session')
            after = self.stats('admission-fault-after-'+mode)
            bc, ac = record(before['counters']), record(after['counters'])
            ck.require(number(ac['rounds_ok']) > number(bc['rounds_ok']), mode+': no successful real draft rounds')
            if not mode.startswith('q-rank'):
                ck.require(sum(number(ac[k])-number(bc[k]) for k in ('rounds_failed', 'rounds_some_failed')) > 0,
                           mode+': injected lifecycle fault did not fail a real round')
            return {'complete': True, 'mode': mode, 'events': cast(Json, selected), 'public_response': response,
                    'proxy_before': before, 'proxy_after': after}
        finally:
            errors: list[str] = []
            for host in armed:
                rc, _ = self.recovery('admission-fault-off-'+mode+'-'+host,
                    [*pf.SSH, host, f'rm -f ~/{directory}/FAULT.json'], 30)
                if rc != 0: errors.append(host)
            ck.require(not errors, 'fault marker cleanup failed: '+','.join(errors))

    def q_failures(self) -> dict[str, Json]:
        return {'complete': True, 'rank0': self.fault('q-rank0'), 'rank1': self.fault('q-rank1'),
                'exactness': 'actual installed verifier vs target-only same processed logits/seeds/positions; every observed emitted token and length exact'}

    def lifecycle(self) -> dict[str, Json]:
        evidence: dict[str, Json] = {'cancel': self.api_gate('cancel')}
        for mode in ('reset', 'reconnect', 'timeout', 'late-reply'):
            evidence[mode] = self.fault(mode)
        return {'complete': True, 'checks': evidence}

    probe_arm: w.Arm

    def probes(self, during: Callable[[], None]) -> dict[str, Json]:
        """Start ON once; semantic probes then independent profile companions only."""
        arm = w.Arm('admission-B', 327680, 0, role='B', admission=True)
        self.probe_arm = arm
        self.start(arm)
        self.build()
        self.warm(arm)
        self.greedy(arm)
        during()
        # Injected lifecycle failures precede the one immutable baseline.
        self.begin_measurement(arm)
        profiles = {name: self.profile_cell(arm, name) for name in cells.CELLS}
        self.finish(arm)
        result: dict[str, Json] = {'profiles': cast(Json, profiles), 'admission_only': True}
        write(self.out/'probe-profiles.json', result)
        return result

    def open_hold(self) -> float:
        try:
            return super().open_hold()
        finally:
            # Also bind85 after a lost HOLD/tunnel acknowledgement. Never let
            # any inherited ceiling escape on an exception.
            if self.hold_written:
                opened = self.submit-w.SUBMIT_S
                self.submit = self.sup.abort_at = opened+73*60
                self.restore_by = opened+MAX_HOLD_S
                self.sup.bound = '73-minute admission abort; restore by original HOLD+85'
                write(self.out/'deadlines.json', {'opened': opened, 'submit': self.submit,
                    'abort': self.sup.abort_at, 'restore_by': self.restore_by})

    def block_budget(self, arm: w.Arm) -> float:
        plan = record(self.admission['admission_budget'])
        return sum(number(plan[key]) for key in DEFAULT_ESTIMATE if key not in ('setup_s', 'pair_s', 'restore_s'))


def configure(live: w.Live) -> Live:
    ck.require(isinstance(live, Live), 'initialized admission Live required')
    obj = cast(Live, live)
    outputs = record(obj.admission['admission_verifier_sha256'])
    obj.overlay = {pf.DIST+(('vllm/v1/worker/gpu/spec_decode/'+name) if name.startswith('remote_dspark/') else name): sha
                   for name, sha in outputs.items()}
    # Pair-only bypasses the measurement admission budget check, but does NOT
    # bypass pair qualification. Local estimate and original85min bound apply.
    obj.pair_only = True
    return obj
