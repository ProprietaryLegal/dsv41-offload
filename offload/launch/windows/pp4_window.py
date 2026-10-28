"""PP4 sampled K2 OFF/ON/OFF window. Dry by default, coordinator only; fenced mutations and strict W8 restore."""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from offload.launch.windows import hold_admission, spark_quiet
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import post_close_watch as watchdog
from offload.launch.windows import pp4_budget as budget
from offload.launch.windows import pp4_cells as cells
from offload.launch.windows import pp4_metrics as metric
from offload.launch.windows import pp4_prerequisites as prereq
from offload.launch.windows import scoped_remote as owner
from offload.launch.windows import up2_checks as ck
from offload.launch.windows.up2_low import X86HOST_PROBE, terminate
from offload.prefill_maxlen.evidence import Json, number, record

LOG = logging.getLogger(__name__)

CONFIG = 'offload/launch/windows/PP4-sampled.env'
PROD_COMMIT = '596ba1cf4e3b11be29b41a57e7f69a5d8efa4799'
PROD_TAG = 'prod-k1-327-ops1-20261026'
SUBMIT_S, ABORT_S, RESTORE_S = 73 * 60, 73 * 60, 85 * 60
# Pair qualification is reserved separately; complete starts are priced by pp4_budget.
PAIR_QUALIFY_S = 10 * 60
KERNEL, DRIVER = '7.0.0-1019-nvidia', '580.178.04'
IDENTITY_KEYS = ('VLLM_DSPARK_REMOTE_IDENTITY', 'VLLM_DSPARK_REMOTE_MARKOV_IDENTITY')
# NBW round2's reviewed E/production TP2 cache, copied privately before the window.
TUNE_CACHE = '/opt/dsv41/x86host-data/pp4/source/exl3_tune_cache-reviewed.bin'
TUNE_SHA256 = '937e9633334be0a5a7a97557e83f962bfbc33dce84a76b93e331b8017b0d405e'
PIN_FILES = ('offload/launch/windows/pp4_admission.py', 'offload/launch/windows/pp4_admission_prepare.py',
             'offload/launch/windows/pp4_admission_engine.py', 'offload/launch/windows/pp4_admission_requests.py', CONFIG, 'offload/launch/start_exp.sh', 'offload/launch/serve_exp.sh',
             'offload/launch/dsv41_tp2_offload.sh', 'offload/kvpack/MANIFEST.json',
             'offload/launch/windows/pp4_window.py', 'offload/launch/windows/pp4_cells.py',
             'offload/launch/windows/pp4_metrics.py', 'offload/launch/windows/pp4_budget.py',
             'offload/launch/windows/nbw_cells.py',
             'offload/dspark/probabilistic/pp4/round6-budget.json',
             'offload/launch/windows/scoped_remote.py', 'offload/launch/windows/hold_admission.py',
             'offload/launch/windows/post_close_watch.py', 'offload/launch/windows/production_health.py',
             'offload/launch/windows/spark_quiet.py', 'offload/launch/windows/pp4_prerequisites.py',
             'offload/launch/windows/pp4_round_trace.py', 'offload/launch/windows/pp4_prepare.py',
             'offload/dspark/probabilistic/pp4/pair_qualify.py', 'offload/dspark/probabilistic/pp4/pair_contract.py',
             'offload/dspark/probabilistic/pp4/pair_probe.py', 'offload/dspark/probabilistic/pp4/full_worker.py')

OWNER_HOSTS = (*owner.NODES, 'localhost')
FINAL_CLOSE_RESERVE_S, FINAL_CLOSE_CALL_S = 90, 20

@dataclass(frozen=True)
class Arm(pf.Arm):
    role: str = 'A'
    admission: bool = False

    @property
    def config(self) -> str:
        return CONFIG


ARMS = tuple(Arm(f's{i}-{role}', 327680, 0, role=role) for i, role in enumerate(('A', 'B', 'A'), 1))


@dataclass(frozen=True)
class Worker:
    tree: str
    commit: str
    identity: str
    markov: str

    def validate(self) -> None:
        ck.require(re.fullmatch(r'/opt/dsv41/x86host-data/pp4/[A-Za-z0-9_/-]+/repo', self.tree) is not None
                   and '..' not in self.tree.split('/'), 'separate K2 tree under /opt/dsv41/x86host-data/pp4 required')
        ck.require(re.fullmatch('[0-9a-f]{40}', self.commit) is not None, 'full reviewed PP4 source commit required')
        ck.require(all(re.fullmatch('[0-9a-f]{64}', v) is not None for v in (self.identity, self.markov)),
                   'reviewed native drafter and Markov digests required')



def host_check(text: str) -> None:
    ck.require(text.splitlines()[:2] == [KERNEL, DRIVER], 'Spark kernel/driver differs from current production brief')
    available = re.findall(r'^MemAvailable:\s+(\d+) kB$', text, re.MULTILINE)
    ck.require(len(available) == 1 and int(available[0]) >= pf.START_FLOOR_KB, 'MemAvailable below 5 GiB before start')


COMMON_ENV = {'VLLM_DSV41_ADAPTIVE_K': '0', 'VLLM_DSPARK_DEPTH_POLICY': 'fixed:5',
              'VLLM_DSPARK_DEPTH_FILE': '', 'VLLM_DSPARK_LOOKUP': '',
              'VLLM_DSPARK_REMOTE_WANT_MARGIN': '0', 'VLLM_PP4_ROUND_DIR': '/pp4-round'}
SPEC = '{"method":"dspark","num_speculative_tokens":5}'


def runtime(prod: dict[str, Json], live: dict[str, Json], arm: Arm, native: Worker,
            overlay: dict[str, Json], *, admission_run: str | None = None) -> None:
    """Exact production identity plus only the registered fixed-depth/sample/mount experiment differences."""
    before, after = record(prod['env']), record(live['env'])
    expected = {**before, **COMMON_ENV, 'VLLM_DSPARK_SAMPLED_TOP32': '1' if arm.role == 'B' else '0',
                **dict(zip(IDENTITY_KEYS, (native.identity, native.markov)))}
    if arm.admission:
        ck.require(bool(admission_run), 'exact admission run identity required')
        expected['VLLM_PP4_ADMISSION_RUN'] = admission_run
    ck.require(after == expected, 'PP4 runtime environment drift')
    mounted = {**record(prod['mounted']), **overlay}
    # Compare the exact argv including fixed5, then normalise only those declared differences for PF1's comparator.
    original = shlex.split(str(prod['argv']).replace('\\\n', ' '))
    changed = shlex.split(str(live['argv']).replace('\\\n', ' '))
    idx = original.index('--speculative-config') + 1
    ck.require(changed[idx] == SPEC, 'PP4 fixed-depth speculative config differs')
    changed[idx] = original[idx]
    a = {**prod, 'env': {**before, 'VLLM_ENGRAM_REMOTE_REQUIRED': '0'}, 'mounted': mounted}
    b = {**live, 'env': {**before, 'VLLM_ENGRAM_REMOTE_REQUIRED': '1'}, 'argv': shlex.join(changed)}
    pf.runtime(a, b, arm, arm.name)


def decision(starts: dict[str, Json], *, measured_cells: tuple[str, ...] = budget.ALL_CELLS) -> dict[str, Json]:
    """Pre-registered Round5 DEP1 form: strict point AND paired95% B1 win."""
    if list(starts) != [arm.name for arm in ARMS]:
        return {'status': 'INCONCLUSIVE', 'reason': 'one complete OFF/ON/OFF HOLD required'}
    ck.require(measured_cells in (budget.ALL_CELLS, budget.B1_CELLS), 'unregistered measurement cell set')
    blocks = [record(starts[a.name]) for a in ARMS]
    for arm, block in zip(ARMS, blocks):
        ck.require(block.get('role') == arm.role and set(record(block['cells'])) == set(measured_cells),
                   'registered arm/cell set differs')
        if block.get('status') != 'COMPLETE':
            return {'status': 'INCONCLUSIVE', 'reason': 'incomplete start'}
        if arm.role == 'B':
            greedy = record(block['greedy'])
            checks = ck.rows(greedy['checks'])
            ck.require([r.get('i') for r in checks] == list(range(6))
                       and all(r.get('identical') is True or r.get('near_tie') is True for r in checks),
                       'six ON greedy/qualified near-tie checks required')
    points: dict[str, Json] = {}
    win = False
    no_loss = True
    guards = True
    for name in measured_cells:
        cell = cells.CELLS[name]
        values = [record(record(block['cells'])[name]) for block in blocks]
        observations = [metric.waves(row, cell.batch) for row in values]
        signatures = [[(r['family'], [s['prompt_sha256'] for s in ck.rows(r['streams'])]) for r in rows]
                      for rows in observations]
        ck.require(all(h == signatures[0] for h in signatures), 'OFF/ON/OFF matched inputs/families differ')
        salts = [str(s['cache_salt_sha256']) for rows in observations for r in rows for s in ck.rows(r['streams'])]
        ck.require(len(set(salts)) == len(salts), 'fresh cold salts required across every start/group/stream')
        point = metric.compare(observations)
        win |= name.endswith('b1') and point['win'] is True
        no_loss &= point['no_loss_beyond_spread'] is True
        # Admission proves complete-round instrumentation. Measurement omits
        # profiles/p95, retaining TTFT and server request-round cost guards.
        timing: dict[str, Json] = {}
        for key in ('median_ttft_s', 'server_decode_ms_per_spec_request_round'):
            if key == 'server_decode_ms_per_spec_request_round':
                v = [number(metric.pool(rows)[key]) for rows in observations]
            else:
                if any(row.get(key) is None for row in values):
                    return {'status': 'INCONCLUSIVE', 'reason': 'missing measured '+key}
                v = [number(row[key]) for row in values]
            ck.require(all(math.isfinite(x) and x > 0 for x in v), 'invalid timing guard')
            ratio = v[1]/((v[0]+v[2])/2)-1
            timing[key+'_regression'] = ratio
            guards &= v[1] <= ((v[0]+v[2])/2)*1.05
        for family in cells.FAMILIES:
            ttft = [statistics.median(number(r['median_ttft_s']) for r in rows if r['family'] == family) for rows in observations]
            cost = [number(metric.pool([r for r in rows if r['family'] == family])['server_decode_ms_per_spec_request_round']) for rows in observations]
            ck.require(all(math.isfinite(x) and x > 0 for x in (*ttft, *cost)), 'invalid family timing guard')
            guards &= ttft[1] <= (ttft[0]+ttft[2])/2*1.05 and cost[1] <= (cost[0]+cost[2])/2*1.05
            timing[family] = {'ttft_regression': ttft[1]/((ttft[0]+ttft[2])/2)-1,
                              'server_ms_per_round_regression': cost[1]/((cost[0]+cost[2])/2)-1}
        points[name] = {**point, 'timing_guards': timing}
    return {'status': 'CANDIDATE FOR REVIEW' if win and no_loss and guards else 'INCONCLUSIVE',
        'cells': points, 'bootstrap_seed': metric.SEED, 'resamples': metric.RESAMPLES,
        'metric_registration': metric.REGISTRATION, 'source_families': len(cells.FAMILIES),
        'decision_metric': 'pooled (R+A)/D per cell/start; reference mean of two OFF rates',
        'rule': 'one B1 point and paired95% lower delta >max(2%reference, OFF spread); '
                'all measured cells point/lower loss <=their spread; retained5% TTFT/server-ms-per-round guards',
        'measured_cells': list(measured_cells),
        'full_round_p95_guard': 'omitted in measurement; complete-round instrumentation qualified in admission',
        'client_tps': 'report-only', 'promotion': 'none; separate independent review required'}


def round_summary(ranks: dict[int, list[dict[str, Json]]], cell: str, batch: int) -> dict[str, Json]:
    """Exact both-rank completion/membership match, then worst-rank completed full model-runner round p95."""
    selected: dict[int, dict[int, dict[str, Json]]] = {}
    for rank in (0, 1):
        rows: dict[int, dict[str, Json]] = {}
        for row in ranks.get(rank, []):
            if row.get('cell') != cell:
                continue
            ck.require(row.get('schema') == 'pp4-complete-round-v1' and row.get('rank') == rank
                       and row.get('failed') is False, 'failed/malformed complete-round trace')
            i = int(number(row['round']))
            ck.require(i not in rows, 'duplicate complete-round identity')
            ck.require(number(row['complete_wall_ms']) > 0 and number(row['cuda_elapsed_ms']) >= 0,
                       'invalid completed round duration')
            rows[i] = row
        selected[rank] = rows
        ck.require(set(rows) == set(range(len(rows))), 'missing complete-round sequence; never drop a slow round')
    ck.require(set(selected[0]) == set(selected[1]) and bool(selected[0]), 'both ranks must complete every profiled round')
    times: list[float] = []
    for i in sorted(selected[0]):
        a, b = selected[0][i], selected[1][i]
        ck.require(a['request_hash'] == b['request_hash'] and a['batch'] == b['batch'], 'TP round membership differs')
        if a['batch'] == batch:
            times.append(max(number(a['complete_wall_ms']), number(b['complete_wall_ms'])))
    ck.require(len(times) >= 16, 'fewer than 16 complete matched rounds at the registered batch size')
    ordered = sorted(times)
    return {'p95_round_ms': ordered[math.ceil(.95 * len(ordered)) - 1],
            'p50_round_ms': statistics.median(ordered), 'p99_round_ms': ordered[math.ceil(.99 * len(ordered)) - 1],
            'profiled_rounds': len(times), 'batch': batch, 'profiled': True,
            'scope': 'worst TP rank execute_model entry through sample_tokens completion plus end-event sync; pure decode only',
            'headline_rates': 'separate unprofiled request observations; profiled request rates excluded'}


# Pure command builders. Commands are never run by a dry plan or by this lane.
def worker_script(native: Worker, tag: str, sampled: bool) -> str:
    """Exact K2 launch options from production; private source/status/tune; never change a unit/current link."""
    return ('set -euo pipefail\n'
            # owner.launch registration necessarily precedes this production mutation (NBW round2 / S49).
            'timeout --signal=TERM --kill-after=15s --foreground 120 bash ~/dsv41-offload/prod_window.sh pair COORD\n'
            'test "$(head -1 ~/dsv41-offload/PAIR_OK)" = COORD\n'
            f'cd {shlex.quote(native.tree)}/offload/dspark\n'
            f'test "$(git rev-parse HEAD)" = {native.commit}\n'
            'test -z "$(git status --porcelain --untracked-files=no)"\n'
            'export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 '
            'OFL_PY_DIR=$HOME/dsv41-offload/prod/dspark-worker/lib '
            'OFL_LIB=$HOME/dsv41-offload/prod/dspark-worker/lib/libofl.so '
            f'DSPARK_P2P_PREBUILT={prereq.RUNTIME}/p2p '
            f'PYTHONPATH={prereq.RUNTIME}/stock:{prereq.RUNTIME}/facc:$PWD '
            f'CUDA_VISIBLE_DEVICES={pf.WORKER_GPUS} DSPARK_WORKER_ARM=k2-tp2 '
            f'DSPARK_SAMPLED_TOP32={int(sampled)} DSPARK_SOURCE_COMMIT={native.commit} '
            f'EXLLAMAV3_TUNE_CACHE={TUNE_CACHE}\n'
            'unset DSPARK_V1_DUMP_DIR DSPARK_WORKER_TP_EMULATE DSPARK_D2_ADAPTER\n'
            'exec sh worker/run/gpu_locked.sh "$HOME/dsv41-offload/locks/gpu-c4e7a371.lock" '
            '"$HOME/dsv41-offload/locks/gpu-7744b44b.lock" -- '
            f'{prereq.PYTHON} -u -m worker.server '
            '--tp 2 --root /opt/dsv41/x86host-data/dspark --max-model-len 1048576 '
            f'--ofl 10.202.28.1:18620 --status-file "$HOME/dsv41-offload-run/pp4/{tag}/status.json"')


def native_probe_script(native: Worker, tag: str) -> str:
    return f"""import json, os, re
from pathlib import Path
root=Path.home()
status=json.loads((root/'dsv41-offload-run/pp4/{tag}/status.json').read_text())
p=Path('/proc')/str(status['pid'])
env=dict(x.split('=',1) for x in (p/'environ').read_text().split('\\0') if '=' in x)
log=(root/'dsv41-offload-run/pp4/{tag}/worker.log').read_text()
ident=re.findall(r'TP=2 worker ready.*drafter_identity ([0-9a-f]{{64}}), markov_identity ([0-9a-f]{{64}})',log)
print(json.dumps({{'status':status,'cwd':os.readlink(p/'cwd'),
 'cmd':(p/'cmdline').read_text().split('\\0'),
 'gpus':env.get('CUDA_VISIBLE_DEVICES'),'sampled':env.get('DSPARK_SAMPLED_TOP32'),'pythonpath':env.get('PYTHONPATH'),
 'identity':ident, 'revision':env.get('DSPARK_SOURCE_COMMIT'),'expected_cwd':str(root/{native.tree!r}/'offload/dspark')}}))
"""


def native_worker(probe: dict[str, Json], native: Worker, now: float, sampled: bool) -> dict[str, Json]:
    status = record(probe['status'])
    ck.require(probe.get('revision') == native.commit and probe.get('cwd') == probe.get('expected_cwd'), 'K2 source identity changed')
    ck.require(probe.get('gpus') == pf.WORKER_GPUS and probe.get('sampled') == str(int(sampled)), 'K2 GPU/sample mode differs')
    ck.require(probe.get('identity') == [[native.identity, native.markov]] and status.get('worker') == 'e-dspark-3090x2',
               'K2 startup checkpoint identity refusal')
    ck.require(status.get('tp') == 2 and status.get('tp_comm_errors') == 0, 'K2 TP failure')
    argv = ck.strings(probe['cmd'])
    ck.require(argv[0] == prereq.PYTHON, 'worker substituted an unqualified interpreter')
    ck.require(str(probe.get('pythonpath', '')).startswith(prereq.RUNTIME + '/stock:' + prereq.RUNTIME + '/facc:'),
               'worker is not using the reviewed CPython3.13 EXL3 runtime')
    ck.require(argv[argv.index('--tp') + 1] == '2' and argv[argv.index('--max-model-len') + 1] == '1048576', 'K2 HELLO bounds differ')
    stamp = datetime.datetime.fromisoformat(str(status['time']).replace('Z', '+00:00')).timestamp()
    ck.require(abs(now - stamp) < 30, 'K2 worker status stale')
    return status


class Live(pf.Live):
    restore_seconds = RESTORE_S

    def __init__(self, out: Path, sup: pf.Supervisor, *, native: Worker, remote: str, expected: str,
                 data_dir: Path, tag: str, admission: dict[str, Json], call: pf.Call | None = None,
                 plain: Callable[[str, list[str], float], tuple[int, str]] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        super().__init__(out, sup, remote=remote, expected=expected, data_dir=data_dir, tag=tag,
                         stage_bundle=False, call=call, plain=plain, clock=clock)
        self.native = native
        self.admission = admission
        self.overlay = prereq.overlay_pins(admission)
        self.owner_tags: list[str] = []
        self.owner_deadlines: dict[str, float] = {}
        self.recovery_serial = 0
        self.local_serial = 0
        self.open_deadline = self.clock() + pf.DRAIN_S + 60
        self.hold_scope_deadline = self.open_deadline + self.restore_seconds
        self.admission_token = uuid.uuid4().hex
        self.native_tags: list[str] = []
        self.pair_open = False
        self.pair_only = False
        self.restore_by = self.clock() + self.restore_seconds
        self.block_deadline = float('inf')
        self.open_admitted = False
        self.close_sequence = 0
        self.measurement_baselines: dict[str, dict[str, Json]] = {}
        self.measurement_latest: dict[str, dict[str, Json]] = {}

    def local(self, name: str, command: list[str], timeout: float) -> tuple[int, str]:
        """Every repeated proof/retry retains a distinct log and an owned local process group."""
        self.local_serial += 1
        log = self.out / 'cmd' / f'restore-{self.local_serial:04d}-{name}.log'
        with log.open('x') as handle:
            proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                    text=True, start_new_session=True)
            try:
                try:
                    rc = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    rc = 124
            finally:
                terminate(proc)  # exact group created here; remote scopes fenced separately
        return rc, log.read_text()

    @property
    def pin_files(self) -> tuple[str, ...]:
        return PIN_FILES

    def hold_purpose(self) -> str:
        return f'PP4-WINDOW {self.tag}: K1-327680 K2 sampled fixed5 OFF/ON/OFF; exact restore required'

    def reap_script(self) -> str:
        # Every PP4 mutating launcher has an exact run-scoped ownership record and was proved absent above.
        # Do not use the inherited name-based reaper to signal another coordinator's process.
        return "echo 'reap done: PP4 registered launcher groups fenced and absent'"

    def preflight(self) -> None:
        self.sup.begin()
        ck.require(pf.digest(pf.RESTORE) == pf.RESTORE_SHA256, 'W8 restore changed')
        cells.greedy_inputs()
        from offload.prefill_trace.window_requests import corpus
        corpus(self.data_dir)
        ck.require((pf.CAMPAIGN / 'D2-GEN.PAUSE').exists() or subprocess.run(
            ['tmux', 'has-session', '-t', '=d2-gen'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode != 0,
            'D2 generation active without pause')
        self.ssh('ledger', 'sparkhead', 'cd ~/dsv41-deploy && test ! -e HOLD && '
                 f'test "$(git rev-parse HEAD)" = {PROD_COMMIT} && '
                 f'test "$(git rev-parse HEAD)" = "$(git rev-list -n1 {PROD_TAG})" && '
                 'git diff --exit-code HEAD -- dsv41.env serve.sh start.sh stop.sh offload/launch/window.sh '
                 'offload/launch/warmup.py offload/launch/warmup_serve.sh')
        checkout = (f'cd ~/{shlex.quote(self.remote)} && test "$(git rev-parse HEAD)" = {self.expected} && '
                    'test -z "$(git status --porcelain --untracked-files=no)"')
        self.ssh('checkout', 'sparkhead', checkout)
        remote = self.ssh('pins', 'sparkhead', f'cd ~/{shlex.quote(self.remote)} && sha256sum ' + shlex.join(PIN_FILES))
        ck.require(remote.strip() == '\n'.join(f'{pf.digest(pf.ROOT / p)}  {p}' for p in PIN_FILES), 'staged source pins differ')
        self.ssh('prod-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/health')
        models = ck.rows(self.json('prod-models', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/v1/models')['data'])
        ck.require(len(models) == 1 and models[0].get('max_model_len') == 327680
                   and models[0].get('root') == '/srv/dsv41/legal-v2a', 'production model/cap differs')
        pf.worker(self.json('worker', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        for host in pf.HOSTS:
            host_check(self.ssh(f'preflight-host-{host}', host, pf.HOST_PARITY))
        self.ssh('k2-tree', 'x86host', f'cd {self.native.tree} && '
                 f'test "$(git rev-parse HEAD)" = {self.native.commit} && '
                 f'test "$(cat SOURCE_COMMIT)" = {self.native.commit} && '
                 'test -z "$(git status --porcelain --untracked-files=no)" && '
                 f'echo "{TUNE_SHA256}  {TUNE_CACHE}" | sha256sum -c -')
        artifacts = prereq.runtime_artifacts(self.admission)
        self.ssh('runtime-artifacts', 'x86host', ' && '.join(
            f'test "$(sha256sum {prereq.RUNTIME}/{name} | cut -d \' \' -f1)" = {sha}'
            for name, sha in artifacts.items()))
        version = "import sys,torch; assert sys.version_info[:2]==(3,13); assert torch.__version__=='2.12.1+cu130'; assert torch.version.cuda=='13.0'; print('PP4 PYTHON313 TORCH212 VERIFIED')"
        ck.require('PP4 PYTHON313 TORCH212 VERIFIED' in self.ssh('runtime-version', 'x86host', prereq.PYTHON + ' -c ' + shlex.quote(version)),
                   'runtime interpreter/version attestation missing')
        phases = [self.tag + '-hold', self.tag + '-pair-qualification-worker', *(f'{self.tag}-{arm.name}-{kind}' for arm in ARMS
                    for kind in ('worker', 'engine')), *(f'{self.tag}-restore-{step}' for step in ('start', 'check', 'close')),
                    *(f'{self.tag}-pair-close-{i}' for i in range(1, 7))]
        self.fresh_controls(phases)
        for host in ('sparkhead',):
            # Both-rank sources are verified after mount; prepared files must already exist before HOLD.
            self.ssh('prepared-' + host, host, f'cd ~/{self.remote} && ' + ' && '.join(
                f'test "$(sha256sum {prereq.OVERLAY_REL}/{name} | cut -d \' \' -f 1)" = {sha}'
                for name, sha in record(self.admission['verifier_sha256']).items()))

    def fresh_controls(self, phases: list[str]) -> None:
        command = ' && '.join(f'test ! -e ~/{owner.CONTROL}/{phase}' for phase in phases)
        for host in OWNER_HOSTS:
            if host == 'localhost':
                self.call('fresh-control-' + host, ['bash', '-c', command], timeout=60)
            else:
                self.ssh('fresh-control-' + host, host, command)

    def remember(self, tag: str, deadline: float) -> None:
        ck.require(math.isfinite(deadline), 'finite scoped deadline required')
        ck.require(self.owner_deadlines.get(tag, deadline) == deadline, 'immutable phase deadline changed')
        if tag not in self.owner_tags:
            self.owner_tags.append(tag)  # BEFORE submitting any possibly ambiguous command
            self.owner_deadlines[tag] = deadline


    def recovery(self, label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        # Unique attempt identities allow a retry without ever reopening a fence.
        self.recovery_serial += 1
        tag = f'{self.tag}-restore-{label}-{self.recovery_serial}'
        deadline = self.restore_by - 90  # reserve final commit/cancel/retention
        self.remember(tag, deadline)
        ck.require(self.clock() < deadline, 'PP4 insufficient time for owned recovery; HOLD stays open')
        try:
            return self.plain(label, ['bash', '-c', owner.command('launch', tag, argv, deadline=deadline)], timeout)
        finally:
            self.fence_phase(tag)


    def owned(self, label: str, host: str, tag: str, script: str, *, timeout: float) -> str:
        deadline = self.hold_scope_deadline if tag.endswith('-hold') else self.sup.abort_at
        self.remember(tag, deadline)
        return self.ssh(label, host, owner.command('launch', tag, ['bash', '-c', script], deadline=deadline), timeout=timeout)

    def fence_phase(self, tag: str, *, guarded: bool = False) -> None:
        def invoke(host: str, command: str) -> tuple[int, str]:
            mode = command.rsplit(' ', 2)[-2]
            label = f'owned-{mode}-{tag}-{host}'
            argv = ['bash', '-c', command] if host == 'localhost' else [*pf.SSH, host, command]
            if guarded:
                return 0, self.call(label, argv, timeout=60)
            return self.plain(label, argv, 60)
        owner.fence_all(tag, invoke, (*owner.NODES, 'localhost'))

    def open_hold(self) -> float:
        ck.require(not self.open_admitted, 'PP4 HOLD cannot be reopened or renewed')
        self.open_admitted = True
        # Freeze the opening ceiling and the placeholder lease together AFTER preflight: a HOLD published by the ceiling
        # always has restore_by = HOLD + RESTORE_S <= the lease (S55 FIX 1).
        self.open_deadline = self.clock() + pf.DRAIN_S + 60
        self.hold_scope_deadline = self.open_deadline + self.restore_seconds
        # Establish PP4 bounds as soon as HOLD is read, including if open/tunnel fails.
        try:
            self.owned('hold-open', 'sparkhead', self.tag + '-hold',
                hold_admission.open_script(self.tag + '-hold', self.admission_token,
                    self.hold_purpose(), pf.DRAIN_S), timeout=pf.DRAIN_S + 60)
        finally:
            text = self.ssh('hold-read', 'sparkhead',
                hold_admission.command('check', self.tag + '-hold', self.admission_token))
            self.hold_written = f'PP4-WINDOW {self.tag}:' in text
            if self.hold_written:
                opened = pf.hold_opened(text, self.tag, self.clock())
                self.submit, self.sup.abort_at, self.restore_by = opened + SUBMIT_S, opened + ABORT_S, opened + self.restore_seconds
                self.sup.bound = '73-minute PP4 abort bound; 12 minutes recovery reserve'
                (self.out / 'deadlines.json').write_text(json.dumps({'opened': opened, 'submit': self.submit,
                    'abort': self.sup.abort_at, 'restore_by': self.restore_by}) + '\n')
        ck.require(self.hold_written, 'PP4 HOLD does not belong to this exact run')
        opened = pf.hold_opened(text, self.tag, self.clock())
        ck.require(opened <= self.open_deadline and self.restore_by <= self.hold_scope_deadline,
                   'PP4 HOLD published after its opening ceiling; placeholder lease would not cover the window')
        self.sup.tunnel([*pf.SSH, '-N', '-o', 'ExitOnForwardFailure=yes', '-L', '127.0.0.1:18001:127.0.0.1:8001', 'sparkhead'])
        return opened

    def snapshot(self) -> None:
        super().snapshot()
        path = self.out / 'rollback.json'
        data = ck.read(path)
        data.update({'production_commit': PROD_COMMIT, 'tag': PROD_TAG})
        path.write_text(json.dumps(data, indent=1) + '\n')

    def block_budget(self, arm: Arm) -> float:
        budgets = record(self.admission['block_budget_s'] if 'block_budget_s' in self.admission else budget.estimate()['block_budget_s'])
        return number(budgets[arm.name])

    def fits(self, arm: Arm) -> bool:
        remaining = ARMS[ARMS.index(arm):] if arm in ARMS else (arm,)
        return self.clock() + sum(self.block_budget(a) for a in remaining) <= self.submit

    def probe_worker(self, label: str, arm: Arm) -> dict[str, Json]:
        tag = f'{self.tag}-{arm.name}-worker'
        return native_worker(self.json(label, 'x86host', 'python3 -c ' + shlex.quote(native_probe_script(self.native, tag))),
                             self.native, self.clock(), arm.role == 'B')

    def qualify_pair(self) -> dict[str, Json]:
        """First GPU operation inside each HOLD. Solo evidence cannot waive it."""
        from offload.dspark.probabilistic.pp4.pair_contract import validate
        from offload.dspark.probabilistic.pp4.pair_qualify import SOURCES
        pair_seconds = (number(record(record(self.admission['registered_budget'])['pair'])['seconds'])
            if 'registered_budget' in self.admission else number(record(self.admission['admission_budget'])['pair_s'])
            if 'admission_budget' in self.admission else PAIR_QUALIFY_S)
        ck.require(self.pair_only
                   or self.clock() + pair_seconds + sum(self.block_budget(a) for a in ARMS) <= self.submit,
                   'pair qualification plus all three measurement blocks cannot fit; restore')
        tag = self.tag + '-pair-qualification-worker'
        self.native_tags.append(tag)
        self.pair_open = True  # pair-open may take effect despite a lost reply
        self.sup.unready()
        path = '/opt/dsv41/x86host-data/pp4/qualification/' + self.tag
        # Share worker's exact source/runtime/lock preparation; replace only its
        # server command with the finite qualification subprocess sequence.
        prefix = worker_script(self.native, tag, True).split('exec sh worker/run/gpu_locked.sh', 1)[0]
        script = prefix + (
            'export WINDOW_COORDINATOR=Deepseek/coordinator PP4_PAIR_QUALIFICATION=1 '
            'PP4_PAIR_LOCK_FD0=3 PP4_PAIR_LOCK_FD1=4\n'
            f'export PYTHONPATH={prereq.RUNTIME}/stock:{prereq.RUNTIME}/facc:{self.native.tree}:$PWD\n'
            f'export TMPDIR=/opt/dsv41/x86host-data/pp4/qualification/tmp '
            f'TRITON_CACHE_DIR=/opt/dsv41/x86host-data/pp4/qualification/triton '
            'TORCH_EXTENSIONS_DIR=/opt/dsv41/x86host-data/pp4/qualification/torch_extensions '
            'CUDA_CACHE_PATH=/opt/dsv41/x86host-data/pp4/qualification/cuda_cache\n'
            'mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"\n'
            'exec sh worker/run/gpu_locked.sh "$HOME/dsv41-offload/locks/gpu-c4e7a371.lock" '
            '"$HOME/dsv41-offload/locks/gpu-7744b44b.lock" -- taskset -c 12-15,28-31 nice -n 19 '
            f'{prereq.PYTHON} -m offload.dspark.probabilistic.pp4.pair_qualify --execute '
            f'--out {path} --commit {self.native.commit} --run {self.tag} '
            f'--deadline {self.clock()+pair_seconds-30}')
        self.owned('pair-qualification', 'sparkhead', tag,
            hold_admission.command('stop', self.tag + '-hold', self.admission_token,
                [*pf.SSH, 'x86host', 'bash -c ' + shlex.quote(script)]), timeout=pair_seconds)
        data = self.json('pair-qualification-receipt', 'x86host', f'cat {path}/receipt.json')
        validate(data, self.native.commit, self.tag)
        ck.require(data['source_sha256'] == {name: pf.digest(pf.ROOT/name) for name in SOURCES},
                   'physical-peer executed source differs')
        (self.out/'physical-pair-qualification.json').write_text(json.dumps(data, allow_nan=False)+'\n')
        self.pair_close(guarded=True)  # prove process absence and restore pair before engine starts
        return data

    def pair_close(self, *, guarded: bool) -> None:
        for tag in list(self.owner_tags):
            if guarded and not tag.endswith('-worker'):
                continue  # its private placeholder server protects the gaps between arms
            self.fence_phase(tag, guarded=guarded)
        script = 'PROD_WINDOW_WAIT_S=60 PROD_WINDOW_READY_S=180 bash ~/dsv41-offload/prod_window.sh close'
        self.recovery_serial += 1
        tag = f'{self.tag}-pair-close-{self.recovery_serial}'
        deadline = self.sup.abort_at if guarded else self.restore_by - 90
        self.remember(tag, deadline)
        command = [*pf.SSH, 'x86host', owner.command('launch', tag, ['bash', '-c', script], deadline=deadline)]
        try:
            if guarded:
                text = self.call('pair-close', command, timeout=300)
                rc = 0
            else:
                rc, text = self.plain('pair-close', command, 300)
        finally:
            self.fence_phase(tag, guarded=guarded)
        ck.require(rc == 0 and 'PRODUCTION HOLD OK' in text.splitlines(),
                   'production TP2 worker did not return: PRODUCTION HOLD OK missing')
        self.pair_open = False

    def start(self, base_arm: pf.Arm) -> None:
        start_began = time.monotonic()
        ck.require(isinstance(base_arm, Arm), 'PP4 arm required')
        arm = cast(Arm, base_arm)
        ck.require(self.fits(arm), f'{arm.name}: full block cannot fit before submission deadline')
        self.block_deadline = self.clock() + self.block_budget(arm)
        self.sup.abort_at = min(self.block_deadline, self.submit + (ABORT_S - SUBMIT_S))
        self.sup.unready()
        if self.pair_open:
            self.pair_close(guarded=True)
        for host in pf.HOSTS:
            host_check(self.ssh(f'{arm.name}-host-{host}', host, pf.HOST_PARITY))
        self.sup.arm_minima = {}
        self.sup.arm_memory[arm.name] = self.sup.arm_minima
        if True:  # both A and B start the exact same reviewed K2 source, sample switch differs
            tag = f'{self.tag}-{arm.name}-worker'
            self.native_tags.append(tag)
            self.remember(tag, self.sup.abort_at)  # BEFORE pair-open, including an ambiguous SSH reply
            self.pair_open = True
            directory = f'dsv41-offload-run/pp4/{tag}'
            launch = owner.command('launch', tag, ['bash', '-c', worker_script(self.native, tag, arm.role == 'B')],
                                   deadline=min(self.sup.abort_at, self.restore_by - 5))
            worker_start = f'mkdir -p ~/{directory} && (nohup {launch} > ~/{directory}/worker.log 2>&1 < /dev/null &)'
            self.owned(f'{arm.name}-native-launch', 'sparkhead', tag + '-admit',
                hold_admission.command('stop', self.tag + '-hold', self.admission_token,
                    [*pf.SSH, 'x86host', worker_start]), timeout=30)
            until = self.clock() + 10 * 60
            while True:
                # A missing status while prepacking is expected; identity failures after JSON appears are fatal.
                text = self.ssh(f'{arm.name}-native-status', 'x86host', f'if [ -f ~/{directory}/status.json ]; then '
                    'python3 -c ' + shlex.quote(native_probe_script(self.native, tag)) + '; else echo WAIT; fi')
                if text.strip() != 'WAIT':
                    native_worker(record(cast(Json, json.loads(text))), self.native, self.clock(), arm.role == 'B')
                    break
                ck.require(self.clock() < until, 'native worker failed to start within ten minutes')
                self.sup.poll()
                time.sleep(1)
        worker_done = time.monotonic()
        self.start_timings = {'worker_start_s': worker_done-start_began}
        env = f'PP4_ROLE={arm.role} PP4_NATIVE_IDENTITY={self.native.identity} PP4_NATIVE_MARKOV_IDENTITY={self.native.markov}'
        env += f' PP4_TRACE_RUN={self.tag}-{arm.name}'
        if arm.admission:
            env += ' PP4_ADMISSION=1'
        for host in pf.HOSTS:
            self.owned(arm.name + '-trace-dir-' + host, host, f'{self.tag}-{arm.name}-trace-{host}',
                     f'mkdir -p ~/dsv41-offload-run/pp4-round && mkdir ~/dsv41-offload-run/pp4-round/{self.tag}-{arm.name}', timeout=60)
        self.owned(f'{arm.name}-start', 'sparkhead', f'{self.tag}-{arm.name}-engine',
                   hold_admission.command('stop', self.tag + '-hold', self.admission_token,
                       ['bash', '-c', f'cd ~/{self.remote} && {env} bash offload/launch/start_exp.sh {CONFIG}']), timeout=pf.START_S)
        self.ssh(f'{arm.name}-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/health')
        models = ck.rows(self.json(f'{arm.name}-models', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/v1/models')['data'])
        ck.require(len(models) == 1 and models[0].get('max_model_len') == 327680
                   and models[0].get('root') == '/srv/dsv41/legal-v2a', 'experiment model/cap differs')
        current = [self.capture(host, arm.name) for host in pf.HOSTS]
        for rank, host in enumerate(pf.HOSTS):
            runtime(self.production[rank], current[rank], arm, self.native, self.overlay,
                    admission_run=self.tag+'-'+arm.name if arm.admission else None)
            ck.require(str(current[rank]['id']) not in self.seen, 'experiment reused a container')
            self.verify_overlay(host, arm.name)
            logs = pf.logs(self.json(f'{arm.name}-logs-{host}', host, 'python3 -c ' + shlex.quote(pf.LOG_FILTER)
                + ' ' + shlex.quote(ck.STOP_LOG.pattern)), rank, 327680)
            if rank == 0:
                self.proxy_pid = int(number(cast(list[Json], logs['proxy_pids'])[0]))
        self.ids = [str(row['id']) for row in current]
        self.seen.update(self.ids)
        self.probe_worker(f'{arm.name}-worker', arm)
        (self.out / f'{arm.name}-identity.json').write_text(json.dumps({'ranks': cast(Json, current)}, indent=1) + '\n')
        self.sup.mark_ready()
        self.start_timings['engine_start_s'] = time.monotonic()-worker_done

    def verify_overlay(self, host: str, label: str) -> None:
        script = ('import hashlib,json; from pathlib import Path; expected=' + repr(self.overlay)
                  + '; got={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in expected}; '
                  'assert got == expected, "PP4 mounted source hash mismatch"; print("PP4 OVERLAY VERIFIED")')
        text = self.ssh(label + '-verify-' + host, host, 'docker exec -i dsv41-serve python3 -', script)
        ck.require('PP4 OVERLAY VERIFIED' in text, 'PP4 overlay verification missing')

    def driver(self, mode: str, output: Path, *extra: str) -> list[str]:
        return [sys.executable, '-m', 'offload.launch.windows.pp4_cells', '--execute', '--mode', mode,
                '--data-dir', str(self.data_dir), '--output', str(output), '--deadline-utc', str(min(self.submit, self.block_deadline)), *extra]

    def build(self) -> None:
        self.call('public-cuts', self.driver('build', self.out / 'cuts.json'), timeout=300)

    def warm(self, arm: Arm) -> None:
        self.call(arm.name + '-warmup', self.driver('warm', self.out / (arm.name + '-warm.json'),
            '--cuts', str(self.out / 'cuts.json'), '--salt', self.tag + '-' + arm.name + '-warm'), timeout=180, inflight=1)

    def greedy(self, arm: Arm) -> dict[str, Json]:
        output = self.out / f'{arm.name}-greedy.json'
        self.call(f'{arm.name}-greedy', self.driver('greedy', output), timeout=300, inflight=1)
        result = ck.read(output)
        rows = ck.rows(result['checks'])
        ck.require([r.get('i') for r in rows] == list(range(6))
                   and all(r.get('identical') is True or r.get('near_tie') is True for r in rows)
                   and result.get('identical') == sum(r.get('identical') is True for r in rows)
                   and result.get('near_tie', 0) == sum(r.get('identical') is not True for r in rows),
                   'native greedy divergence or missing checks')
        return result

    def begin_measurement(self, arm: Arm) -> None:
        ck.require(arm.name not in self.measurement_baselines, 'PP4 measurement baseline cannot be renewed')
        stats = self.stats(f'{arm.name}-post-warmup-post-greedy-baseline')
        cs = record(stats['counters'])
        ck.require({'rounds_failed', 'rounds_some_failed', 'rounds_ok'} <= cs.keys(), 'missing failed-round baseline evidence')
        ck.counters(stats, before=stats)  # earlier totals are recorded, never zeroed
        self.measurement_baselines[arm.name] = stats
        self.measurement_latest[arm.name] = stats
        (self.out / f'{arm.name}-measurement-baseline.json').write_text(json.dumps(stats) + '\n')

    def measured_stats(self, arm: Arm, label: str) -> dict[str, Json]:
        ck.require(arm.name in self.measurement_baselines, 'PP4 post-warmup measurement baseline absent')
        stats = self.stats(label)
        baseline = self.measurement_baselines[arm.name]
        previous = record(self.measurement_latest[arm.name]['counters'])
        current, initial = record(stats['counters']), record(baseline['counters'])
        ck.require(previous.keys() <= current.keys(), f'{label}: counter evidence disappeared')
        for key, value in previous.items():
            ck.require(number(current[key]) >= number(value), f'{label}: {key} counter reset')
        ck.counters(stats, before=baseline)
        for key in ('rounds_failed', 'rounds_some_failed'):
            ck.require(number(current[key]) == number(initial[key]), f'{label}: {key} grew after measurement began')
        ck.require(number(record(stats['async_watchdog'])['fires']) == number(record(baseline['async_watchdog'])['fires']),
                   f'{label}: watchdog grew during measurement')
        self.measurement_latest[arm.name] = stats
        return stats

    def cell(self, arm: Arm, name: str) -> dict[str, Json]:
        before = self.measured_stats(arm, f'{arm.name}-{name}-before')
        steps = number(self.probe_worker(f'{arm.name}-{name}-worker-before', arm).get('steps', 0))
        output = self.out / f'{arm.name}-{name.replace(":", "-")}.json'
        self.call(f'{arm.name}-{name}', self.driver('cell', output, '--cell', name, '--cuts', str(self.out / 'cuts.json'),
            '--salt', f'{self.tag}-{arm.name}', '--ledger', str(self.out / 'requests-ledger.jsonl')),
            timeout=cells.CELLS[name].estimate_s * 3 + 60, inflight=cells.CELLS[name].batch)
        result = ck.read(output)
        ck.require(result.get('cell') == name and result.get('batch') == cells.CELLS[name].batch, 'cell artifact identity mismatch')
        after = self.measured_stats(arm, f'{arm.name}-{name}-after')
        ck.counters(after, 0, before)
        ck.require(number(record(after['counters'])['rounds_ok']) > number(record(before['counters'])['rounds_ok']), 'no drafting during cell')
        ck.require(number(self.probe_worker(f'{arm.name}-{name}-worker-after', arm).get('steps', 0)) > steps, 'worker took no steps')
        result['proxy_before'], result['proxy_after'] = before, after
        result['full_round_p95_guard'] = 'omitted in measurement; no profile companions'
        output.write_text(json.dumps(result, allow_nan=False) + '\n')
        return result

    def profile_cell(self, arm: Arm, name: str) -> dict[str, Json]:
        profiles = {family: self.profile_family(arm, name, family, wave)
                    for wave, family in enumerate(cells.FAMILIES)}
        return {'p95_round_ms': max(number(row['p95_round_ms']) for row in profiles.values()),
                'by_family': cast(Json, profiles), 'profiled': True}

    def profile_family(self, arm: Arm, name: str, family: str, wave: int) -> dict[str, Json]:
        """Separate matched request, never use its instrumented throughput as the headline rate."""
        marker = family+':'+name
        before = self.measured_stats(arm, f'{arm.name}-{family}-{name}-profile-before')
        directory = f'dsv41-offload-run/pp4-round/{self.tag}-{arm.name}'
        output = self.out / f'{arm.name}-{family}-{name}-profile-request.json'
        armed: list[str] = []
        try:
            for host in pf.HOSTS:
                armed.append(host)  # include a marker write whose reply is lost
                self.owned(arm.name + '-profile-on-' + family + '-' + host, host, f'{self.tag}-{arm.name}-profile-{family}-{name}-{host}',
                         f'printf %s {shlex.quote(marker)} > ~/{directory}/ACTIVE', timeout=60)
            self.call(arm.name + '-profile-' + family + '-' + name, self.driver('cell', output, '--cell', name,
                '--cuts', str(self.out / 'cuts.json'), '--salt', f'{self.tag}-{arm.name}-profile-{family}',
                '--profile-wave', str(wave),
                '--ledger', str(self.out / 'requests-ledger.jsonl')),
                timeout=cells.CELLS[name].estimate_s * 3 / metric.WAVES + 60, inflight=cells.CELLS[name].batch)
        finally:
            # Unsup cleanup also works after STOP; full restore fences the engine before W8 on any failure.
            errors: list[str] = []
            for host in armed:
                rc, text = self.recovery(arm.name + '-profile-off-' + family + '-' + host,
                    [*pf.SSH, host, f'rm -f ~/{directory}/ACTIVE'], 30)
                if rc != 0:
                    errors.append(host + ': marker removal failed: ' + text[-100:])
            ck.require(not errors, '; '.join(errors))
        ranks: dict[int, list[dict[str, Json]]] = {}
        for rank, host in enumerate(pf.HOSTS):
            text = self.ssh(arm.name + '-round-trace-' + family + '-' + host, host,
                f'test "$(stat -c %s ~/{directory}/rank{rank}.jsonl)" -le 8388608 && cat ~/{directory}/rank{rank}.jsonl')
            rows = [record(cast(Json, json.loads(line))) for line in text.splitlines()]
            path = self.out / f'{arm.name}-{family}-rank{rank}-rounds.jsonl'
            path.write_text(text)
            ranks[rank] = rows
        after = self.measured_stats(arm, f'{arm.name}-{family}-{name}-profile-after')
        return {**round_summary(ranks, marker, cells.CELLS[name].batch), 'proxy_before': before, 'proxy_after': after}

    def finish(self, arm: Arm) -> dict[str, Json]:
        checks: dict[str, Json] = {}
        stats = self.measured_stats(arm, f'{arm.name}-final-stats')
        checks['post_warmup_baseline'] = self.measurement_baselines[arm.name]
        checks['proxy'] = stats
        for rank, host in enumerate(pf.HOSTS):
            ck.require(self.ssh(f'{arm.name}-final-id-{host}', host, "docker inspect --format '{{.Id}}' dsv41-serve").strip() == self.ids[rank],
                       'experiment container changed during measurement')
            checks[host] = pf.logs(self.json(f'{arm.name}-final-logs-{host}', host, 'python3 -c '
                + shlex.quote(pf.LOG_FILTER) + ' ' + shlex.quote(ck.STOP_LOG.pattern)), rank, 327680)
        minima = self.sup.arm_minima
        ck.require(minima is not None and set(minima) == set(pf.HOSTS), 'missing memory minima')
        checks['memory_min_kb'] = cast(Json, dict(cast(dict[str, int], minima)))
        return checks

    def restore(self, *, production_stopped: bool) -> dict[str, Json]:
        self.sup.close()
        original = self.plain
        # Clamp cleanup, W8 and close transactions to ONE original HOLD deadline.
        def bounded(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
            if label in ('cancel-close', 'retain-close'):
                return original(label, argv, min(30, timeout))
            left = self.restore_by - self.clock()
            ck.require(left > 0, f'PP4 HOLD+{self.restore_seconds//60} exhausted; manual restore required; HOLD stays open')
            return original(label, argv, min(left, timeout))
        self.plain = bounded
        try:
            # Fence the HOLD writer BEFORE rereading it: a late opener must not
            # create a HOLD after recovery decided that production was untouched.
            for tag in list(self.owner_tags):
                self.fence_phase(tag)
            if not self.hold_written:
                rc, text = self.plain('pp4-hold-reread', [*pf.SSH, 'sparkhead',
                    hold_admission.command('check', self.tag + '-hold', self.admission_token)], 60)
                ck.require(rc == 0, 'PP4 HOLD ownership unreadable; HOLD stays open')
                self.hold_written = f'PP4-WINDOW {self.tag}:' in text
                if self.hold_written:
                    opened = pf.hold_opened(text, self.tag, self.clock())
                    self.restore_by = opened + self.restore_seconds
            if not self.hold_written:
                ck.require(not production_stopped, 'PP4 production may be stopped without owned HOLD')
                return {'status': 'NOT NEEDED: no owned HOLD; all remote mutators fenced and absent'}
            ck.require(math.isfinite(self.restore_by), 'PP4 missing immutable restore deadline')
            self.pair_close(guarded=False)
            if production_stopped:
                # dockerd children are outside cgroups: explicitly stop/remove
                # the same named experiment containers and require no GPU apps.
                rc, text = self.recovery('experiment-stop', [*pf.SSH, 'sparkhead',
                    f'cd ~/{self.remote} && ./stop.sh </dev/null'], 120)
                ck.require(rc == 0, 'PP4 experiment stop failed; HOLD stays open')
                for host in pf.HOSTS:
                    rc, text = self.plain('jit-wait-' + host, [*pf.SSH, host,
                        spark_quiet.command('PP4 EXPERIMENT ABSENT', deadline=min(self.restore_by - 90, self.clock() + 350))], 360)
                    ck.require(rc == 0 and 'PP4 EXPERIMENT ABSENT' in text.splitlines(),
                               f'{host}: experiment/container/compute absence unproved; HOLD stays open')
            result: dict[str, Json] = {'pair_close': 'PRODUCTION HOLD OK'}
            steps = ('start', 'check', 'close') if production_stopped else ('check', 'close')
            for step in steps:
                # Close uses an exact suffix to install the HOLD-unlink shim on
                # every receiving host. W8 still performs its original checks.
                if step == 'close':
                    tag = f'{self.tag}-restore-close'
                    deadline = self.restore_by - 90
                    self.remember(tag, deadline)
                    ck.require(self.clock() < deadline, 'PP4 final close budget exhausted; HOLD stays open')
                    try:
                        rc, text = self.plain(step, ['bash', '-c', owner.command('launch', tag,
                            ['bash', str(pf.RESTORE), step], deadline=deadline)], 120)
                    finally:
                        self.fence_phase(tag)
                else:
                    rc, text = self.recovery(step, ['bash', str(pf.RESTORE), step], 900 if step == 'start' else 600)
                pf.restored(step, rc, text)  # requires six distinct IDENTICAL, not near ties
                result[step] = 'PASS'
            for prior in list(self.owner_tags):
                self.fence_phase(prior)
            ck.require(self.restore_by - self.clock() >= 90, 'PP4 final close needs 90s reserve; HOLD stays open')
            tag = f'{self.tag}-restore-close'
            try:
                try:
                    rc, text = self.plain('commit-close', [*pf.SSH, 'sparkhead', owner.command('commit-close', tag,
                        deadline=min(self.restore_by-5, self.clock()+25))], 30)
                    ck.require(rc == 0 and owner.CLOSED in text.splitlines(), 'PP4 final owned HOLD close unconfirmed')
                finally:
                    rc, text = self.plain('cancel-close', [*pf.SSH, 'sparkhead', owner.command('cancel-close', tag)], 30)
                    ck.require(rc == 0 and 'SCOPED CLOSE COMMIT FENCED' in text.splitlines(), 'PP4 final close fence unconfirmed')
                result['restore_deadline_met'] = self.clock() < self.restore_by
                ck.require(result['restore_deadline_met'] is True, 'PP4 restore completed late')
            except (Exception, KeyboardInterrupt):
                rc, text = self.plain('retain-close', [*pf.SSH, 'sparkhead', owner.command('retain-close', tag)], 30)
                ck.require(rc == 0 and owner.RETAINED in text.splitlines(), 'PP4 original HOLD retention unconfirmed')
                raise
            result['status'] = 'RESTORED: K1 START VERIFIED, greedy 6/6 IDENTICAL, PRODUCTION HOLD OK, window closed'
            return result
        finally:
            self.plain = original
            (self.out / 'arm-memory.json').write_text(json.dumps(self.sup.arm_memory) + '\n')

class Backend(Protocol):
    def preflight(self) -> None: ...
    def open_hold(self) -> float: ...
    def snapshot(self) -> None: ...
    def qualify_pair(self) -> dict[str, Json]: ...
    def fits(self, arm: Arm) -> bool: ...
    def start(self, arm: Arm) -> None: ...
    def build(self) -> None: ...
    def warm(self, arm: Arm) -> None: ...
    def greedy(self, arm: Arm) -> dict[str, Json]: ...
    def begin_measurement(self, arm: Arm) -> None: ...
    def cell(self, arm: Arm, name: str) -> dict[str, Json]: ...
    def finish(self, arm: Arm) -> dict[str, Json]: ...
    def restore(self, *, production_stopped: bool) -> dict[str, Json]: ...


def window(b: Backend, checkpoint: Callable[[dict[str, Json]], None], *,
           selected: tuple[Arm, ...] = ARMS, prior: dict[str, Json] | None = None,
           admission_run: Callable[[], dict[str, Json]] | None = None,
           measured_cells: tuple[str, ...] = budget.ALL_CELLS) -> dict[str, Json]:
    result: dict[str, Json] = {'status': 'INCOMPLETE', 'comparison': 'CONTROLLED fixed5 PP4 runtime OFF vs ON; not production-exact', 'starts': {}, 'decision': {'status': 'NOT A CANDIDATE', 'reason': 'not measured completely'}}
    starts = record(result['starts'])
    hold = stopped = False
    try:
        ck.require(prior is None and selected in ((), ARMS), 'Round5 requires a single complete OFF/ON/OFF HOLD; no imported segment')
        ck.require(measured_cells in (budget.ALL_CELLS, budget.B1_CELLS), 'unregistered measurement cell set')
        result['measurement_cells'] = list(measured_cells) if selected else []
        b.preflight()
        hold = True
        result['hold_opened'] = b.open_hold()
        b.snapshot()
        stopped = True  # qualification's owned pair-open can lose its reply
        result['physical_pair'] = b.qualify_pair()
        if admission_run is not None:
            ck.require(not selected and prior is None, 'admission cannot run headline measurements')
            result['admission'] = admission_run()
        for arm in selected:
            i = ARMS.index(arm)
            ck.require(b.fits(arm), f'{arm.name}: mandatory block cannot fit')
            stopped = True  # native pair operation or start_exp may mutate despite a lost reply
            row: dict[str, Json] = {'status': 'STARTED', 'role': arm.role, 'cells': {}}
            starts[arm.name] = row
            checkpoint({'start': arm.name, 'status': 'STARTED'})
            b.start(arm)
            if arm == selected[0]:
                b.build()
            b.warm(arm)
            if arm.role == 'B':
                row['greedy'] = b.greedy(arm)
            b.begin_measurement(arm)
            names = measured_cells
            for name in names[i % len(names):] + names[:i % len(names)]:
                record(row['cells'])[name] = b.cell(arm, name)
                checkpoint({'start': arm.name, 'cell': name})
            row['checks'] = b.finish(arm)
            row['status'] = 'COMPLETE'
            checkpoint({'start': arm.name, 'status': 'COMPLETE'})
        if admission_run is None:
            result['decision'] = decision(starts, measured_cells=measured_cells)
        result['status'] = 'ADMISSION_GATES_COMPLETE' if admission_run is not None else 'PAIR_QUALIFIED' if not selected else 'COMPLETE'
    except (Exception, KeyboardInterrupt) as exc:
        LOG.exception('PP4 window stopped; restoration remains mandatory')
        result['stop'] = f'{type(exc).__name__}: {exc}'
        result['decision'] = {'status': 'REJECT', 'reason': 'execution, semantic or lifecycle gate failed'}
    finally:
        saved = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            if hold:
                try:
                    result['restore'] = b.restore(production_stopped=stopped)
                except (Exception, KeyboardInterrupt) as exc:
                    LOG.exception('PP4 production restoration failed; retain HOLD')
                    result['restore'] = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
            else:
                result['restore'] = {'status': 'NOT NEEDED: preflight only'}
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
    restoration = record(result['restore'])
    if hold and (restoration.get('status') != 'RESTORED: K1 START VERIFIED, greedy 6/6 IDENTICAL, PRODUCTION HOLD OK, window closed'
                 or restoration.get('restore_deadline_met') is not True):
        result['decision'] = {'status': 'REJECT', 'reason': 'production restore unverified'}
    return result


def validate_prior(prior: dict[str, Json], commit: str) -> None:
    raise ValueError('Round5 retires split-window ABBA; one complete OFF/ON/OFF HOLD required')


def plan() -> str:
    registered = budget.estimate()
    return ('DRY RUN: no command, network or write. Coordinator Deepseek/coordinator only.\n'
            'Admission: six seed gates plus six primary probes, one ON start; no calibration/headline blocks.\n'
            'Pair-only: physical qualification and strict W8 restore; no measurements.\n'
            'CONTROLLED fixed5 OFF/ON/OFF in one HOLD, same target/weights/pair; not production-exact.\n'
            'T1/top_p1/high/priority10; forced512 outputs/length finish; fresh salts/matched42+wave seeds.\n'
            'short B1/32K B1:4 requests; 32K B4:2 whole concurrent groups, no-regression only.\n'
            'Measurement omits profile companions/full-round-p95 guard; retains5% TTFT/server ms-per-round guards.\n'
            'Six NBW exact-or-qualified-near-tie greedy checks on ON; W8 six distinct exact IDENTICAL.\n'
            f"Measurement ESTIMATED {number(registered['total_minutes']):.2f}min; <=85 required, else REFUSE before host contact.\n"
            'Includes separate333s worker and404s Spark startups; measurement requires hashed admission pair duration.\n'
            'Pair reserve=max(180s,1.5*admission driver seconds); drop B4 only if needed to fit.\n'
            f"Dry600s-bound illustration selects {registered['cells']}; no measurement admission without evidence.\n"
            'Late starts need every remaining block; abort original HOLD+73, restore+85, no renewal.\n'
            'All receiving-host mutations use shared scopes; cleanup/absence failure retains HOLD.\n'
            'Pooled(R+A)/D; reference mean of OFF rates; OFF spread absolute difference.\n'
            'A B1 wins only if point AND paired95% lower delta >max(2%reference,spread); all cells loss<=spread.\n'
            '2000 paired whole-group draws seed4103; passing=CANDIDATE FOR REVIEW only.\n'
            'Strict pinned W8, then shared15min post-close watch outside HOLD. No automatic promotion.')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--remote-repo', default='dsv41-deploy-offload')
    ap.add_argument('--expected-experiment-commit', default='')
    ap.add_argument('--qualification', type=Path)
    qualification_modes = ap.add_mutually_exclusive_group()
    qualification_modes.add_argument('--admission', action='store_true', help='six-probe qualification only; no headline measurements')
    qualification_modes.add_argument('--pair-only', action='store_true', help='qualification HOLD only, uses reviewed staging receipt; no measurement')
    ap.add_argument('--segment', choices=('single',), default='single')
    ap.add_argument('--prior', type=Path, help='first segment summary, required for second')
    ap.add_argument('--worker-tree', default='')
    ap.add_argument('--worker-commit', default='')
    ap.add_argument('--worker-identity', default='')
    ap.add_argument('--worker-markov-identity', default='')
    ap.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    args = ap.parse_args(argv)
    if not args.execute:
        print(plan())
        return 0
    native = Worker(args.worker_tree, args.worker_commit, args.worker_identity, args.worker_markov_identity)
    try:
        ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'WINDOW_COORDINATOR=Deepseek/coordinator required')
        ck.require(args.qualification is not None, 'hash-bound --qualification is mandatory; PP4 is not live-qualified')
        from offload.launch.windows import pp4_admission as live_admission
        admission = (live_admission.seed if args.admission else prereq.staging if args.pair_only else prereq.qualification)(cast(Path, args.qualification), args.expected_experiment_commit)
        ck.require(not (args.pair_only or args.admission) or args.segment == 'single' and args.prior is None, 'pair-only has no measurement segment')
        prior = ck.read(args.prior) if args.prior is not None else None
        ck.require(prior is None, 'Round5 is one HOLD; no prior segment may enter')
        native.validate()
        ck.require(native.commit == args.expected_experiment_commit, 'worker and experiment commits differ')
        ck.require(admission['worker_identity'] == native.identity and admission['markov_identity'] == native.markov,
                   'worker identities differ from reviewed qualifications')
        ck.require(args.out is not None and args.out.resolve().parent == pf.CAMPAIGN and not args.out.exists(), 'fresh --out inside campaign required')
        ck.require(re.fullmatch('[0-9a-f]{40}', args.expected_experiment_commit) is not None, 'full reviewed experiment commit required')
        ck.require(re.fullmatch('[A-Za-z0-9_.-]+', args.remote_repo) is not None and args.remote_repo != 'dsv41-deploy', 'separate experiment checkout required')
        ck.require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pf.ROOT, text=True).strip() == args.expected_experiment_commit,
                   'local HEAD differs from reviewed commit')
        ck.require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=pf.ROOT, text=True).strip(), 'tracked source dirty')
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f'STOP before any remote action: {exc}', file=sys.stderr)
        return 1
    out = cast(Path, args.out)
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
    live_class = live_admission.Live if args.admission else Live
    live = live_class(out, sup, native=native, remote=args.remote_repo, expected=args.expected_experiment_commit,
                     data_dir=args.data_dir, tag=tag, admission=admission)
    live.pair_only = args.pair_only
    session = live_admission.Session(live_admission.configure(live), checkpoint) if args.admission else None
    measured_cells = (tuple(str(x) for x in cast(list[Json], record(admission['registered_budget'])['cells']))
                      if not (args.pair_only or args.admission) else budget.ALL_CELLS)
    result = window(live, checkpoint, admission_run=session.run if session else None,
        selected=() if args.pair_only or args.admission else ARMS, prior=prior, measured_cells=measured_cells)
    sup.close()
    if str(record(result['restore']).get('status', '')).startswith('RESTORED'):
        def watch_log(message: str) -> None:
            print(message, flush=True)
            with (out / 'post-close-watch.log').open('a') as handle:
                handle.write(message + '\n')
        watched = watchdog.watch(watchdog.ProductionHealth(),
            watchdog.ScopedRecovery(tag + '-watch', out / 'post-close-commands'), log=watch_log)
        result['post_close_watch'] = cast(Json, dict(watched.__dict__))
        if watched.status != 'PASS':
            result['decision'] = {'status': 'REJECT', 'reason': 'post-close production health unverified'}
    if session is not None:
        session.finish(result)
    result.update({'registered_budget': admission.get('registered_budget'), 'metric_registration': metric.REGISTRATION, 'segment': 'admission' if args.admission else 'pair-only' if args.pair_only else args.segment, 'prior_sha256': pf.digest(args.prior) if args.prior else None, 'experiment_commit': args.expected_experiment_commit, 'production_commit': PROD_COMMIT,
                   'restore_sha256': pf.RESTORE_SHA256, 'config_sha256': pf.digest(pf.ROOT / CONFIG),
                   'unavailable_traces': 'raw confidence and exposed draft wait not exported; no measurement profiles/full-round-p95 guard; instrumentation qualified in admission',
                   'artifact_sha256': {str(p.relative_to(out)): pf.digest(p) for p in sorted(out.rglob('*.json*')) if p.is_file()}})
    (out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'stop': result.get('stop'), 'decision': result['decision'], 'restore': result['restore']}))
    if 'post_close_watch' in result and record(result['post_close_watch']).get('status') != 'PASS':
        return 2
    if str(record(result['restore']).get('status', '')).startswith(('RESTORED', 'NOT NEEDED')):
        return 0 if 'stop' not in result else 1
    print('PRODUCTION NOT VERIFIED; HOLD remains open. Inspect cmd/restore-*.log; restore pair first, then W8 start/check/close.', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
