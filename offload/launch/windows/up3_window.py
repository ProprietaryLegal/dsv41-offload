"""UP3 current-production window: legacy OFF/arm/OFF groups or WIN2 multi-arm groups (win2_plan). Dry by default;
coordinator only, one group per HOLD."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from offload.launch import warmup as ops1_warmup
from offload.launch.windows import pf1_window as pf
from offload.launch.windows import decision_v2 as dv2
from offload.launch.windows import win2_history
from offload.launch.windows import nbw_cells as greedy_driver
from offload.launch.windows import up2_checks as ck
from offload.launch.windows import chain, hold_admission, spark_quiet, production_health
from offload.launch.windows import post_close_watch as watchdog
from offload.launch.windows import scoped_remote as owner
from offload.launch.windows.up2_low import X86HOST_PROBE, terminate
from offload.prefill_maxlen.evidence import Json, number, record
from offload.up3 import build as b, cells, gates, metrics, host_gates, win2_plan, upx_plan, upf_plan, upf_driver
from offload.upfix import configs as upf_configs, windows as upf_windows
from offload.upx import band

CONFIG = 'offload/up3/window.env'
SUBMIT_S, ABORT_S, RESTORE_S = 73 * 60, 73 * 60, 85 * 60
# ESTIMATED from measured startup + NBW-W3 cold cells; budget.json records
# hashes, arithmetic and unmeasured mixed/B4/host-evidence extrapolations explicitly.
STARTUP_S, WARMUP_S, TRANSITION_S, QUALIFY_S = 404, 90, 60, 30
GREEDY_S = 60  # ESTIMATED six W5 checks plus first-divergence diagnostics, not a GB10 measurement
SETUP_S, RECOVERY_S = 120, 720
# WIN2 chain (coordinator 2026-10-27 for R-UPX: UPX-0064 needs ~78 min of starts): production is down for the whole
# chain, so a chained window (WINDOW_CHAIN=1, last or not) submits/aborts until HOLD+120 min and restores by HOLD+132,
# the same 12-minute recovery reserve after the last submission. Unchained windows keep 73/73/85.
CHAINED_SUBMIT_S = CHAINED_ABORT_S = 120 * 60
CHAINED_RESTORE_S = CHAINED_SUBMIT_S + RECOVERY_S


def hold_bounds(group: str = '') -> tuple[int, int, int]:
    """(submit, abort, restore) seconds after the HOLD publication: an R-UPF group's own bounds (sized from its budget,
    restore <= HOLD+132; coordinator 2026-10-27), else this process's chain mode."""
    if group in upf_plan.GROUPS:
        return upf_plan.bounds(group)
    return (CHAINED_SUBMIT_S, CHAINED_ABORT_S, CHAINED_RESTORE_S) if chain.enabled() else (SUBMIT_S, ABORT_S, RESTORE_S)
assert upf_plan.RECOVERY_S == RECOVERY_S, 'R-UPF bounds keep the UP3 recovery reserve'
PREPARE_S = STARTUP_S + WARMUP_S + TRANSITION_S + GREEDY_S
BLOCK_S = PREPARE_S + sum(cells.ESTIMATES[n] * cells.REPEATS[n] for n in cells.CELLS)
KERNEL, DRIVER = '7.0.0-1019-nvidia', '580.178.04'
BUDGET = {'control': BLOCK_S, **{arm: BLOCK_S + QUALIFY_S for arm in gates.CASES}}
TARGETS = {'split': ('mixed150k-2decode',), 'cpu-pin': ('32k-b3',),
           'vor': ('32k-b1',), 'k5': ('32k-b3',), 'dual-ring': ('32k-b3',)}
# Roles that run the K5 speculative schedule [[1,4,5]] (fixed2 adds DEP1's depth env).
SPEC_ROLES = ('k5', 'fixed2')
# Config-only roles without host evidence; ring-defer proves the dual ring with dual-ring's host gate.
NO_HOST_GATE = ('k5', 'fixed2', 'defer')
HOST_GATE = {'ring-defer': 'dual-ring'}
# start_exp.sh prints each node's applied ceiling ("spark clock cap 2300 MHz"); NDM's persistent 2300 MHz from ~18:55Z
# 2026-10-27, and THERMAL-WATCH may dial it back to 2150 at any time, so every start records the cap it actually got.
CLOCK_LINE = re.compile(r'^(spark(?:worker)?|sparkhead|sparkworker) clock cap ([0-9]+) MHz', re.M)
# WIN2 groups (offload/up3/win2_plan.py): ring-depth2, defer-shares (0051) and k5-batch; R-UPX groups
# (offload/up3/upx_plan.py): upx-0064, upx-pin, upx-0067; R-UPF groups (offload/up3/upf_plan.py): upf-topk, upf-0091
# (upf-w1 is refused by the HOLD+132 ceiling); legacy groups stay C/arm/C.
GROUPS = (*gates.CASES, *win2_plan.GROUPS, *upx_plan.GROUPS, *upf_plan.GROUPS)
# WIN2 chain (NDM 2026-10-27 "go window after window"): a chained window cannot capture production (it is down), so it
# reuses the chain's first UP3 capture, sha-pinned through the marker. UP3 compares the FULL container environment.
IDENTITY_KEYS = ('id', 'image', 'engine', 'files', 'mounted', 'env', 'argv', 'engram_identity')
# WIN2-R2 placeholder rule: the hold placeholders stay up for the whole chain. UP3's own placeholders live on the HOLD
# phase's PRIVATE tmux socket (scoped_remote's tmux shim) and die with its fence, and that shim also rewrites
# chain.CHAINED_CLOSE's `tmux has-session -t dsv41-hold` to the close scope's private socket, so CHAINED_CLOSE can never
# pass inside a UP3 scope (reported to the coordinator). UP3 therefore (1) keeps chain placeholders on the DEFAULT socket,
# outside every window scope, exactly as window.sh open does (sleep 86400), (2) proves them with chain.PLACEHOLDER_PROBE
# over plain ssh, (3) runs only the HOLD unlink inside its own -restore-close scope (the rm shim records it) followed by the
# usual fenced commit-close, and (4) takes them down after a verified strict W8 restore (window.sh close's kill-session
# is redirected to a private socket inside the scope, so it cannot).
PLACEHOLDER_UP = 'UP3 CHAIN PLACEHOLDER UP'
KEEP_PLACEHOLDER = ('tmux has-session -t dsv41-hold 2>/dev/null || tmux new-session -d -s dsv41-hold '
                    + shlex.quote('exec -a "vllm serve [DSV41-OFFLOAD hold]" sleep 86400')
                    + f'; {chain.PLACEHOLDER_PROBE} && echo {shlex.quote(PLACEHOLDER_UP)}')
PROBE_PLACEHOLDER = f'if {chain.PLACEHOLDER_PROBE}; then echo PRESENT; else echo ABSENT; fi'
RELEASE_PLACEHOLDER = ('tmux kill-session -t dsv41-hold 2>/dev/null || true; '
                       'if tmux has-session -t dsv41-hold 2>/dev/null; then echo HELD; else echo RELEASED; fi')
CHAIN_CLOSE_RECORDED = 'UP3 CHAINED CLOSE RECORDED'
CLOSE_OWN_HOLD = 'rm -f "$HOME/dsv41-deploy/HOLD"; echo ' + shlex.quote(CHAIN_CLOSE_RECORDED)
# DEP1 depth 2 near-tie diagnostics (one tokenize pair + one top-20 probe per divergent check, DEP1 diverged on its first
# three of six): the greedy call may take longer than GREEDY_S; time alone must never stop a fixed2 start (coordinator).
FIXED2_GREEDY_S = 180
# R-UPF T0-T2 (offload/upfix WINDOW.md), on sparkhead in the experiment checkout inside the HOLD, each under the HOLD's
# 'stop' admission. T1/T2 always print their docker run's exit code: a non-zero code skips X-topk only; a timed-out
# (124/137) container is removed by name so it cannot hold the CPU or the GPU.
UPF_T0_IDLE = 'UPF T0 IDLE'
UPF_T0_S = 180
UPF_BUILD = upf_configs.TOPK_BUILD


def upf_t0(remote: str) -> str:
    return (f'cd ~/{remote} && set -a && . ./dsv41.env && set +a && ./stop.sh && for n in $HEAD $WORKER; do '
            'busy=$(ssh -o BatchMode=yes $n "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader" '
            "| grep -v '^$' || true); if [ -n \"$busy\" ]; then echo \"GPU busy on $n after stop: $busy\" >&2; exit 1; fi; "
            'done; echo ' + shlex.quote(UPF_T0_IDLE))


def upf_docker(remote: str, step: str, gpus: bool, module: str, arguments: str, guard: str) -> str:
    """One bounded in-image docker run (WINDOW.md T1/T2 verbatim flags), named upfix-<step>."""
    name = f'upfix-{step.lower()}'
    return (f'cd ~/{remote} && set -a && . ./dsv41.env && set +a && B={UPF_BUILD} && rc=0 && {guard}'
            f'timeout -k 30 {upf_windows.T2_TIMEOUT_S} docker run --rm --name {name} --pull never --network none '
            + ('--gpus all ' if gpus else '--cpus 16 ')
            + '--user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/opt/repo:ro" -v "$PWD/$B:/opt/repo/$B" -w /opt/repo '
            f'--entrypoint python3 "$IMAGE" -m {module} --execute {arguments} || rc=$?; fi; '
            f'case $rc in 124|137) docker rm -f {name} >/dev/null 2>&1 || true ;; esac; echo "UPF {step} RC $rc"')


def upf_t1(remote: str) -> str:
    guard = (f"if [ \"$(docker image inspect --format '{{{{.Id}}}}' \"$IMAGE\" 2>/dev/null)\" != {pf.IMAGE} ]; then rc=90; "
             'else rm -rf $B/jit $B/receipt.json $B/bitcompare.json && ')
    return upf_docker(remote, 'T1', False, 'offload.upfix.topk.jit_build', '--build-dir $B/jit --receipt $B/receipt.json', guard)


def upf_t2(remote: str) -> str:
    guard = 'if [ ! -f $B/jit/dsv41_topk.so ]; then rc=91; else '
    return upf_docker(remote, 'T2', True, 'offload.upfix.topk.bitcompare',
                      '--library $B/jit/dsv41_topk.so --out $B/bitcompare.json', guard)


def upf_receipts(remote: str) -> str:
    return (f'cd ~/{remote} && B={UPF_BUILD} && for f in receipt bitcompare; do echo "UPF RECEIPT $f"; '
            "cat $B/$f.json 2>/dev/null | tr -d '\\n' || true; echo; done; "
            'python3 -m offload.upfix.configs verify-topk 2>&1 | tail -n 3; echo "UPF VERIFY RC ${PIPESTATUS[0]}"')


class ArmStop(ValueError):
    """R-UPF arm-level STOP (WINDOW.md: "arm-level stops are not window stops"): a CANDIDATE's identity gate failed; the
    window records it and goes on to the next start. Raised only for candidate arms of R-UPF groups."""

    def __init__(self, gate: str, message: str, evidence: dict[str, Json] | None = None) -> None:
        super().__init__(f'{gate}: {message}')
        self.gate, self.evidence = gate, evidence or {}


@dataclass(frozen=True)
class Arm(pf.Arm):
    role: str = 'control'

    @property
    def config(self) -> str:
        return upf_configs.CONFIG if self.role in upf_configs.ARMS else CONFIG


def is_control(role: str) -> bool:
    return role == 'control' or upf_plan.is_control(role)


def arms(group: str) -> tuple[Arm, ...]:
    ck.require(group in GROUPS, 'one isolated group required')
    if group in upf_plan.GROUPS:   # the design's own start labels (C1, X-topk, C2): its archive rows and start logs use them
        return tuple(Arm(s.label, 327680, 0, role=s.arm) for s in upf_plan.starts(group))
    plan = (win2_plan.roles(group) if group in win2_plan.GROUPS else upx_plan.roles(group) if group in upx_plan.GROUPS
            else ('control', group, 'control'))
    return tuple(Arm(f's{i}-{role}', 327680, 0, role=role) for i, role in enumerate(plan, 1))


def admit(group: str) -> None:
    """Every non-control role of the group must be an admitted config-only arm (refused BEFORE HOLD). split (0064) and
    vor (0067) stay parked everywhere except their R-UPX group, where registration upx-0p9 qualifies them. R-UPF groups
    are config-only UPF.env arms; upf-w1 is refused by the HOLD+132 ceiling (upf_plan.admit)."""
    if group in upf_plan.GROUPS:
        upf_plan.admit(group)
        return
    for role in {a.role for a in arms(group)} - {'control'}:
        if not (group in upx_plan.GROUPS and role in upx_plan.UPX_ROLES):
            host_gates.admit(role)


def position(group: str, arm: 'Arm') -> int:
    return arms(group).index(arm)


def cells_for(group: str) -> tuple[str, ...]:
    if group in upf_plan.GROUPS:
        return upf_plan.CELLS
    if group in upx_plan.GROUPS:
        return upx_plan.group_for(group).cells
    return win2_plan.plan_for(group).cells if group in win2_plan.GROUPS else cells.CELLS


def repeats_for(group: str, name: str, arm: 'Arm') -> int:
    if group in upf_plan.GROUPS:
        return upf_plan.repeats(group, position(group, arm), name)
    if group in upx_plan.GROUPS:
        return upx_plan.repeats(group, name, position(group, arm))
    return win2_plan.plan_for(group).repeats[name] if group in win2_plan.GROUPS else cells.REPEATS[name]


def greedy_s(arm: 'Arm') -> int:
    return FIXED2_GREEDY_S if arm.role == 'fixed2' else GREEDY_S


def allowance(group: str, arm: 'Arm', first: bool) -> int:
    """Pre-measurement allowance from start: startup/warmup/greedy plus that start's qualification evidence."""
    if group in upf_plan.GROUPS:   # R-UPF needles run inside qualify (ESTIMATED GATE_S of the design)
        return PREPARE_S + QUALIFY_S + upf_plan.gate_s(group, position(group, arm))
    if group in upx_plan.GROUPS:   # R-UPX gates run inside qualify (ESTIMATED GATE_S of the design)
        return PREPARE_S + QUALIFY_S + math.ceil(upx_plan.gate_s(group, position(group, arm)))
    return PREPARE_S + (QUALIFY_S if arm.role != 'control' else 0) + greedy_s(arm) - GREEDY_S


def block_s(group: str, arm: 'Arm', first: bool) -> int:
    if group in upf_plan.GROUPS:
        return upf_plan.block_s(group, position(group, arm))
    if group in win2_plan.GROUPS:
        return win2_plan.block_s(group, arm.role)
    if group in upx_plan.GROUPS:
        return upx_plan.block_s(group, position(group, arm))
    return BUDGET[arm.role]


def near_tie_nats(group: str, arm: 'Arm') -> float:
    """Registered greedy near-tie limit: WIN2 groups use NDM's 0.9 / DEP1 fixed2 1.1; R-UPX groups 0.9 (upx-0p9);
    legacy groups NBW's 0.501; R-UPF groups 0.9 (upf-0p9)."""
    if group in upf_plan.GROUPS:
        return upf_configs.GREEDY_NEAR_TIE_NATS
    if group in upx_plan.GROUPS:
        return band.NEAR_TIE_NATS
    return win2_plan.NEAR_TIE_NATS[arm.role] if group in win2_plan.GROUPS else greedy_driver.NEAR_TIE_NATS


def archive_path() -> Path:
    """WIN2 control archive (append-only, sha-chained)."""
    return pf.CAMPAIGN / 'control-archive.jsonl'


def source_pins(arm: str) -> dict[str, str]:
    lock = cast(b.Lock, json.loads((b.HERE / 'inputs.json').read_text()))
    pins: dict[str, str] = {}
    for key, sha in lock['inputs'].items():
        if key.startswith('git:offload/kvpack/overlay/'):
            pins[b.DIST + key.removeprefix('git:offload/kvpack/overlay/')] = sha
        elif key.startswith('git:offload/dspark/spark/remote_dspark/'):
            # stage_overlay.sh mounts only the package's *.py files; hostflag.c is compiled into
            # ofl/libdsv41_hostflag.so (UP3-k5-W-20261027T040705Z stopped on the absent .c).
            if key.endswith('.py'):
                pins[b.DIST + 'vllm/v1/worker/gpu/spec_decode/remote_dspark/' + key.rsplit('/', 1)[1]] = sha
        elif key.startswith('git:offload/dspark/spark/overlay/'):
            pins[b.DIST + key.removeprefix('git:offload/dspark/spark/overlay/')] = sha
        elif key.startswith('image:'):
            pins[b.DIST + key.removeprefix('image:')] = sha
    pins.update({b.DIST + name: sha for name, sha in b.verify(arm)['files'].items()})
    return pins


def runtime(prod: dict[str, Json], live: dict[str, Json], arm: Arm) -> None:
    before, after = record(prod['env']), record(live['env'])
    expected = {**before, **b.ENV[arm.role]}
    ck.require(after == expected, 'UP3 env drift beyond isolated documented switch')
    expected_mounts = {key: value for key, value in record(prod['mounted']).items() if key != '/serve.sh'}
    expected_mounts.update({b.DIST + name: sha for name, sha in b.verify(arm.role)['files'].items()})
    ck.require({key: value for key, value in record(live['mounted']).items() if key != '/serve.sh'} == expected_mounts,
               'UP3 mounted source drift')
    a = shlex.split(str(prod['argv']).replace('\\\n', ' '))
    actual = shlex.split(str(live['argv']).replace('\\\n', ' '))
    expected_argv = list(a)
    expected_argv[expected_argv.index('--port') + 1] = '8001'
    if arm.role in SPEC_ROLES:
        expected_argv[expected_argv.index('--speculative-config') + 1] = b.SPEC
    ck.require(actual == expected_argv, 'UP3 argv drift beyond experiment port/K5 schedule')
    # Only after proving exact actual equality, use PF1's other engine/target/image/Engram invariants.
    normal_prod = {**prod, 'env': {**before, 'VLLM_ENGRAM_REMOTE_REQUIRED': '0'}}
    normal_live = {**live, 'env': {**before, 'VLLM_ENGRAM_REMOTE_REQUIRED': '1'},
                   'mounted': prod['mounted'], 'argv': shlex.join([('8001' if i == a.index('--port') + 1 else v)
                                                               for i, v in enumerate(a)])}
    pf.runtime(normal_prod, normal_live, arm, arm.name)


def greedy_checks(result: dict[str, Json]) -> None:
    """DEP1 9b72534 premeasurement gate; driver enforces NBW top20/0.501nat diagnosis."""
    checks = ck.rows(result.get('checks', []))
    ck.require([r.get('i') for r in checks] == list(range(6))
        and all(r.get('identical') is True or r.get('near_tie') is True for r in checks)
        and number(result.get('identical', -1)) + number(result.get('near_tie', -1)) == 6,
        'UP3 greedy divergence that is not a near-tie, or missing checks')
    ck.require(result.get('identical') == sum(r.get('identical') is True for r in checks)
        and result.get('near_tie') == sum(r.get('identical') is not True for r in checks),
        'greedy aggregate evidence differs from ordered checks')


def decision(group: str, starts: dict[str, Json], pre: dict[str, Json] | None = None) -> dict[str, Json]:
    if group in upf_plan.GROUPS:   # identity / no-loss per candidate; SKIPPED and arm-STOPPED candidates are verdicts too
        return upf_plan.decide(group, starts, pre or {}, archive_path(), b.PRODUCTION)
    expected = arms(group)
    if list(starts) != [a.name for a in expected] or any(record(v).get('status') != 'COMPLETE' for v in starts.values()):
        return {'status': 'NOT A CANDIDATE', 'reason': 'all mandatory complete bracket starts required'}
    ck.require(all(record(starts[a.name]).get('role') == a.role for a in expected), 'start role mismatch')
    for arm in expected:
        greedy_checks(record(record(starts[arm.name]).get('greedy', {})))
    if group in win2_plan.GROUPS:
        return win2_plan.decide(group, starts, archive_path(), b.PRODUCTION)
    if group in upx_plan.GROUPS:
        return upx_plan.decide(group, starts, archive_path(), b.PRODUCTION)
    points: dict[str, Json] = {}
    targets: list[str] = []
    passes = True
    for name in cells.CELLS:
        measured = [record(record(record(starts[a.name])['cells'])[name]) for a in expected]
        point = metrics.compare(measured, cells.batch_for(name), cells.REPEATS[name])
        tails: list[Json] = []
        # Client rate is diagnostic. TTFT and mixed p99 retain separate gates.
        for repeat in range(cells.REPEATS[name]):
            streams = [ck.rows(ck.rows(row['repetitions'])[repeat]['streams']) for row in measured]
            for j in range(cells.batch_for(name)):
                for metric in ('ttft_s', 'p99_inter_token_delivery_gap_ms') if name == cells.CELLS[-1] else ('ttft_s',):
                    values = [number(items[j][metric]) for items in streams]
                    ck.require(all(math.isfinite(v) and v >= 0 for v in values), 'invalid tail timing')
                    width = max(values[::2]) - min(values[::2])
                    okay = all(values[i] <= (values[i-1] + values[i+1])/2 + width for i in range(1, len(values), 2))
                    tails.append({'repeat': repeat, 'stream': j, 'metric': metric, 'pass': okay, 'C_spread': width})
                    passes &= okay
        point['tail_gates'] = tails
        passes &= point['no_loss_beyond_spread'] is True
        if name in TARGETS[group] and point['win'] is True:
            targets.append(name)
        points[name] = point
    return {'status': 'CANDIDATE FOR REVIEW' if passes and targets else 'NOT A CANDIDATE', 'cells': points,
            'passing_target_cells': cast(Json, targets),
            'rule': 'pooled (R+A)/D; point and paired 95% lower bound >max(2%,C spread); no cell loss beyond spread; TTFT/mixed p99 gates',
            'design': 'whole simultaneous request groups paired across starts; B4 one group is no-regression only; no task-family inference',
            'promotion': 'none; combine only individually passing arms in a later reviewed window'}


def chain_identity(explicit: Path | None) -> tuple[Path, dict[str, Json]]:
    """The production identity a chained window compares against: the marker's production snapshot
    (chain.production_snapshot, sha-checked; WIN2-R2), or an explicit UP3 rollback.json (coordinator option when the
    snapshot is another runner's filtered capture). UP3 compares the FULL container env, so it must be a full capture."""
    path = chain.production_snapshot()[0] if explicit is None else explicit
    ck.require(path.is_absolute() and path.resolve().parent.parent == pf.CAMPAIGN and path.name == 'rollback.json',
               'chain production identity must be a campaign window rollback.json')
    data = ck.read(path)
    ck.require((data.get('commit') or data.get('production_commit')) == b.PRODUCTION and data.get('tag') == b.TAG
               and data.get('restore_sha256') == pf.RESTORE_SHA256, 'chain production identity is another production')
    ranks = [record(r) for r in ck.rows(data['ranks'])]
    ck.require(len(ranks) == len(pf.HOSTS) and all(set(r) >= set(IDENTITY_KEYS) for r in ranks),
               'chain production identity lacks ranks/keys')
    # pf1_window.capture keeps only these runtime keys; UP3's capture keeps the full container environment.
    filtered = ('VLLM_', 'EXL3_', 'NCCL_', 'PYTORCH_', 'CUDA_', 'PYTHONPATH', 'DSV41_ENGRAM_OVERLAY')
    ck.require(all(any(not key.startswith(filtered) for key in record(r['env'])) for r in ranks),
               'chain identity env is filtered, not the full container env; pass --production-identity '
               '<a UP3 rollback.json of this production>')
    ck.require(ranks[0]['files'] == ranks[1]['files'] and all(r.get('image') == pf.IMAGE for r in ranks),
               'chain production identity ranks/image differ')
    return path, data


class Live(pf.Live):
    def __init__(self, out: Path, sup: pf.Supervisor, *, group: str,
                 remote: str, expected: str, data_dir: Path, tag: str, call: pf.Call | None = None,
                 plain: Callable[[str, list[str], float], tuple[int, str]] | None = None,
                 clock: Callable[[], float] = time.time, production_identity: Path | None = None) -> None:
        super().__init__(out, sup, remote=remote, expected=expected, data_dir=data_dir, tag=tag,
                         stage_bundle=False, call=call, plain=plain, clock=clock)
        admit(group)
        self.group = group
        self.bounds = hold_bounds(group)   # fixed for the life of the window (chain mode is read once)
        self.restore_by = self.clock() + self.bounds[2]
        self.owner_tags: list[str] = []
        self.owner_deadlines: dict[str, float] = {}
        self.recovery_serial = 0
        self.open_deadline = self.clock() + pf.DRAIN_S + 60
        self.hold_scope_deadline = self.open_deadline + self.bounds[2]
        self.open_admitted = False
        self.admission_token = uuid.uuid4().hex
        self.round_baseline: dict[str, dict[str, Json]] = {}
        self.counter_seen: dict[str, dict[str, float]] = {}
        self.started_at = 0.0
        self.clock_caps: dict[str, list[int]] = {}
        self.upx_state: dict[str, Path] = {}       # R-UPX: this window's gate artifacts by kind-position
        self.upx_screen: dict[str, Json] | None = None
        # R-UPF (upf_plan): skipped starts, candidates' greedy files for C2's KL, C1's capture per rank (identity is
        # relative to C1), C1's soak growth (OFF), and the verified dsv41_topk library sha256 (T1 receipt, T2 PASS).
        self.skipped: dict[str, str] = {}
        self.upf_state: dict[str, Path] = {}
        self.upf_reference: dict[int, dict[str, Json]] = {}
        self.upf_c1_diff: dict[str, Json] = {}
        self.upf_soak_off: dict[str, float] | None = None
        self.upf_library = ''
        # WIN2 chain: 'production' (serving at preflight) or 'chained' (marker, production down, Sparks quiet).
        self.chain_mode = 'production'
        self.explicit_identity = production_identity
        self.inherited: tuple[Path, dict[str, Json]] | None = None

    @property
    def pin_files(self) -> tuple[str, ...]:
        paths = [p for p in (b.ROOT / 'offload/up3').rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and 'tests' not in p.parts]
        paths += [b.ROOT / name for name in ('offload/launch/windows/up3_window.py',
            'offload/launch/windows/nbw_cells.py', 'offload/launch/windows/pf1_window.py', 'offload/launch/windows/scoped_remote.py',
            'offload/launch/windows/hold_admission.py', 'offload/launch/windows/spark_quiet.py',
            'offload/launch/windows/post_close_watch.py', 'offload/launch/windows/production_health.py',
            'offload/launch/start_exp.sh', 'offload/launch/serve_exp.sh', 'offload/launch/dsv41_tp2_offload.sh',
            'offload/launch/warmup_serve.sh', 'offload/launch/warmup.py')]
        if self.group in upf_plan.GROUPS:   # UPF.env, its K1 base, overlays, configs verify-topk, T1/T2 sources
            built = b.ROOT / UPF_BUILD
            paths += [p for p in (b.ROOT / 'offload/upfix').rglob('*') if p.is_file() and '__pycache__' not in p.parts
                      and 'tests' not in p.parts and built not in p.parents]
            paths.append(b.ROOT / 'offload/launch/configs/D-kv6-async-pack.env')
        return tuple(sorted({str(p.relative_to(b.ROOT)) for p in paths}))

    def remember(self, tag: str, deadline: float) -> None:
        ck.require(math.isfinite(deadline), 'finite scoped deadline required')
        ck.require(self.owner_deadlines.get(tag, deadline) == deadline, 'immutable phase deadline changed')
        if tag not in self.owner_tags:
            self.owner_tags.append(tag)  # BEFORE submitting any possibly ambiguous command
            self.owner_deadlines[tag] = deadline

    def owned(self, label: str, host: str, tag: str, script: str, *, timeout: float, inflight: int = 0) -> str:
        deadline = self.hold_scope_deadline if tag.endswith('-hold') else self.sup.abort_at
        self.remember(tag, deadline)
        return self.ssh(label, host, owner.command('launch', tag, ['bash', '-c', script], deadline=deadline),
                        timeout=timeout, inflight=inflight)

    def fence_phase(self, tag: str, *, guarded: bool = False) -> None:
        def invoke(host: str, command: str) -> tuple[int, str]:
            mode = command.rsplit(' ', 2)[-2]
            label = f'owned-{mode}-{tag}-{host}'
            argv = ['bash', '-c', command] if host == 'localhost' else [*pf.SSH, host, command]
            if guarded:
                return 0, self.call(label, argv, timeout=60)
            return self.plain(label, argv, 60)
        owner.fence_all(tag, invoke, (*owner.NODES, 'localhost'))

    def recovery(self, label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        # Unique attempt identities allow a retry without ever reopening a fence.
        self.recovery_serial += 1
        tag = f'{self.tag}-restore-{label}-{self.recovery_serial}'
        deadline = self.restore_by - 90  # reserve final commit/cancel/retention
        self.remember(tag, deadline)
        ck.require(self.clock() < deadline, 'UP3 insufficient time for owned recovery; HOLD stays open')
        try:
            return self.plain(label, ['bash', '-c', owner.command('launch', tag, argv, deadline=deadline)], timeout)
        finally:
            self.fence_phase(tag)

    def preflight(self) -> None:
        admit(self.group)
        b.verify_inputs()
        for arm in b.ARMS:
            b.verify(arm)
        self.sup.begin()
        ck.require(pf.digest(pf.RESTORE) == pf.RESTORE_SHA256, 'W8 restore pin differs')
        greedy_driver.greedy_inputs()
        for name in ('offload/launch/warmup.py', 'offload/launch/warmup_serve.sh'):
            ck.require((b.ROOT / name).read_bytes() == b.production_file(name), 'OPS1 warm-up source drift')
        from offload.prefill_trace.window_requests import corpus
        corpus(self.data_dir)
        ck.require((pf.CAMPAIGN / 'D2-GEN.PAUSE').exists() or subprocess.run(
            ['tmux', 'has-session', '-t', '=d2-gen'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0,
            'D2 generation active without pause')
        production_files = ['dsv41.env', 'serve.sh', 'start.sh', 'stop.sh', 'firewall.sh',
                            'offload/launch/warmup.py', 'offload/launch/warmup_serve.sh', 'offload/launch/dsv41_tp2_offload.sh',
                            'offload/launch/window.sh']
        pin_lines = '\n'.join(b.digest(b.production_file(p)) + '  ' + p for p in production_files)
        probe = self.ssh('chain-probe', 'sparkhead', 'if test -e ~/dsv41-deploy/HOLD; then echo HOLD; else echo NOHOLD; fi; '
            'if curl -fsS --max-time 10 http://localhost:8000/health >/dev/null 2>&1; then echo SERVING; else echo DOWN; fi').split()
        ck.require(len(probe) == 2 and probe[0] in ('HOLD', 'NOHOLD') and probe[1] in ('SERVING', 'DOWN'), 'chain probe unreadable')
        present: bool | None = None
        if chain.marker_present():   # WIN2-R2: a chained start must find the hold placeholders up on both Sparks
            present = all(self.ssh('chain-placeholders-' + host, host, PROBE_PLACEHOLDER).split() == ['PRESENT']
                          for host in pf.HOSTS)
        self.chain_mode = chain.preflight(production_serving=probe[1] == 'SERVING', hold_exists=probe[0] == 'HOLD',
                                          placeholders_present=present)
        ck.require(self.explicit_identity is None or self.chain_mode == 'chained', '--production-identity is for chained windows only')
        self.ssh('ledger', 'sparkhead', 'cd ~/dsv41-deploy && test ! -e HOLD && '
            f'test "$(git rev-parse HEAD)" = {b.PRODUCTION} && '
            f'test "$(git rev-list -n1 {b.TAG})" = {b.PRODUCTION} && git diff --quiet HEAD -- . && '
            'printf %s ' + shlex.quote(pin_lines + '\n') + ' | sha256sum --check --status -')
        for host in pf.HOSTS:
            self.ssh('checkout-' + host, host, f'cd ~/{self.remote} && test "$(git rev-parse HEAD)" = {self.expected} && '
                     'test -z "$(git status --porcelain --untracked-files=no)"')
            pins = self.ssh('pins-' + host, host, f'cd ~/{self.remote} && sha256sum ' + shlex.join(self.pin_files))
            ck.require(pins.strip() == '\n'.join(f'{pf.digest(b.ROOT / p)}  {p}' for p in self.pin_files), 'remote reviewed source pins differ')
        if self.chain_mode == 'chained':
            self.inherited = chain_identity(self.explicit_identity)
            for host in pf.HOSTS:
                text = self.ssh('chain-quiet-' + host, host, spark_quiet.command('UP3 CHAIN QUIET', deadline=self.clock() + 60), timeout=70)
                ck.require('UP3 CHAIN QUIET' in text.splitlines(), f'{host}: chained Spark quiet unproved')
        else:
            self.ssh('prod-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/health')
            model = ck.rows(self.json('prod-models', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8000/v1/models')['data'])
            ck.require(len(model) == 1 and model[0].get('max_model_len') == 327680
                and model[0].get('root') == '/srv/dsv41/legal-v2a', 'production cap/model differs')
        pf.worker(self.json('worker', 'x86host', 'python3 -c ' + shlex.quote(X86HOST_PROBE)), self.clock())
        for host in pf.HOSTS:
            host_check(self.ssh('host-' + host, host, pf.HOST_PARITY))
            self.ssh('fresh-' + host, host, f'test ! -e ~/{owner.CONTROL}/{self.tag}-hold')

    def open_hold(self) -> float:
        ck.require(not self.open_admitted, 'UP3 HOLD cannot be reopened or renewed')
        self.open_admitted = True
        self.open_deadline = self.clock() + pf.DRAIN_S + 60
        self.hold_scope_deadline = self.open_deadline + self.bounds[2]
        # Establish UP3 bounds as soon as HOLD is read, including if open/tunnel fails.
        try:
            self.owned('hold-open', 'sparkhead', self.tag + '-hold',
                hold_admission.open_script(self.tag + '-hold', self.admission_token,
                    f'UP3-WINDOW {self.tag}: isolated {self.group}', pf.DRAIN_S,
                    allow_unreachable=chain.allow_unreachable(self.chain_mode)), timeout=pf.DRAIN_S + 60)
        finally:
            text = self.ssh('hold-read', 'sparkhead',
                hold_admission.command('check', self.tag + '-hold', self.admission_token))
            self.hold_written = f'UP3-WINDOW {self.tag}:' in text
            if self.hold_written:
                opened = pf.hold_opened(text, self.tag, self.clock())
                submit, abort, restore = self.bounds
                self.submit, self.sup.abort_at, self.restore_by = opened + submit, opened + abort, opened + restore
                self.sup.bound = f'UP3 HOLD+{abort // 60} abort bound'
                (self.out / 'deadlines.json').write_text(json.dumps({'opened': opened, 'submit': self.submit,
                    'abort': self.sup.abort_at, 'restore_by': self.restore_by}) + '\n')
        ck.require(self.hold_written, 'UP3 exact owned HOLD marker required')
        opened = pf.hold_opened(text, self.tag, self.clock())
        ck.require(opened <= self.open_deadline and self.restore_by <= self.hold_scope_deadline,
                   'UP3 late HOLD publication exceeds opening ceiling/placeholder lease')
        self.sup.tunnel([*pf.SSH, '-N', '-o', 'ExitOnForwardFailure=yes', '-L', '127.0.0.1:18001:127.0.0.1:8001', 'sparkhead'])
        return opened

    def capture(self, host: str, label: str) -> dict[str, Json]:
        current = super().capture(host, label)
        config = self.json(label + '-fullenv-' + host, host, "docker inspect --format '{{json .Config}}' dsv41-serve")
        env: dict[str, Json] = {}
        for value in ck.strings(config['Env']):
            key, equal, item = value.partition('=')
            ck.require(equal == '=' and key not in env, 'duplicate/malformed full container environment')
            env[key] = item
        current['env'] = env
        return current

    def verify_sources(self, host: str, arm: str) -> None:
        pins = self.upf_pins(arm) if arm in upf_configs.ARMS else source_pins(arm)
        got = self.json(arm + '-sources-' + host, host,
            'docker exec -i dsv41-serve python3 - ' + shlex.join(pins), pf.IMAGE_BASES)
        ck.require(got == pins, 'installed production/arm bytes differ')

    def snapshot(self) -> None:
        if self.chain_mode == 'chained':
            # Production is down: reuse the chain's sha-pinned UP3 capture (re-checked here, after HOLD).
            ck.require(self.inherited is not None, 'chained snapshot without preflight identity')
            path, data = cast(tuple[Path, dict[str, Json]], self.inherited)
            ck.require(chain_identity(self.explicit_identity)[0] == path, 'chain production identity changed after preflight')
            self.production = [record(r) for r in ck.rows(data['ranks'])]
            self.seen.update(str(r['id']) for r in self.production)
            (self.out / 'rollback.json').write_text(json.dumps({'commit': b.PRODUCTION, 'tag': b.TAG,
                'restore_sha256': pf.RESTORE_SHA256, 'ranks': data['ranks'], 'chained_from': str(path),
                'chained_from_sha256': pf.digest(path)}, indent=1) + '\n')
            if self.group in upf_plan.GROUPS:
                self.upf_image_bases()   # recorded UNCHECKED: no production container to read
            return
        for host in pf.HOSTS:
            current = self.capture(host, 'prod')
            self.verify_sources(host, 'control')
            ck.require(current.get('image') == pf.IMAGE, 'production image pin differs')
            self.production.append(current)
            self.seen.add(str(current['id']))
        ck.require(self.production[0]['files'] == self.production[1]['files'], 'production target metadata differs by rank')
        if self.group in upf_plan.GROUPS:
            self.upf_image_bases()
        self.production_idle()
        (self.out / 'rollback.json').write_text(json.dumps({'commit': b.PRODUCTION, 'tag': b.TAG,
            'restore_sha256': pf.RESTORE_SHA256, 'ranks': cast(Json, self.production)}, indent=1) + '\n')

    def fits(self, arm: Arm) -> bool:
        planned = arms(self.group)
        ck.require(arm in planned, 'start is outside this isolated OFF/arm/OFF plan')
        remaining = sum(block_s(self.group, a, i == 0) for i, a in enumerate(planned)
                        if i >= planned.index(arm) and a.name not in self.skipped)
        return self.clock() + remaining <= self.submit

    def start(self, base_arm: pf.Arm) -> None:
        ck.require(isinstance(base_arm, Arm), 'UP3 arm required')
        arm = cast(Arm, base_arm)
        ck.require(self.fits(arm), 'full mandatory start block cannot fit')
        self.sup.unready()
        for tag in list(self.owner_tags):
            if tag != self.tag + '-hold':
                self.fence_phase(tag, guarded=True)
        self.sup.arm_minima = {}
        self.sup.arm_memory[arm.name] = self.sup.arm_minima
        for host in pf.HOSTS:
            host_check(self.ssh(arm.name + '-host-' + host, host, pf.HOST_PARITY))
        self.started_at = self.clock()
        startup_limit = STARTUP_S + TRANSITION_S + (QUALIFY_S if not is_control(arm.role) else 0)
        switch = upf_plan.start_env(arm.role) if arm.role in upf_configs.ARMS else f'UP3_ARM={arm.role}'
        start = (f'cd ~/{self.remote} && {switch} '
            f'timeout --foreground {startup_limit} bash offload/launch/start_exp.sh {arm.config}')
        start = hold_admission.command('stop', self.tag + '-hold', self.admission_token, ['bash', '-c', start])
        text = self.owned(arm.name + '-start', 'sparkhead', self.tag + '-' + arm.name + '-engine', start, timeout=startup_limit)
        self.clock_caps[arm.name] = sorted({int(m.group(2)) for m in CLOCK_LINE.finditer(text)})
        self.ssh(arm.name + '-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/health')
        model = ck.rows(self.json(arm.name + '-model', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/v1/models')['data'])
        ck.require(len(model) == 1 and model[0].get('max_model_len') == 327680
            and model[0].get('root') == '/srv/dsv41/legal-v2a', 'experiment model/cap differs')
        current = [self.capture(host, arm.name) for host in pf.HOSTS]
        for rank, host in enumerate(pf.HOSTS):
            if self.group in upf_plan.GROUPS:
                self.upf_identity(arm, rank, current[rank])
            else:
                runtime(self.production[rank], current[rank], arm)
            ck.require(str(current[rank]['id']) not in self.seen, 'container reused across starts')
            self.verify_sources(host, arm.role)
            summary = self.rank_logs(arm, host, rank)
            if rank == 0:
                self.proxy_pid = int(number(cast(list[Json], summary['proxy_pids'])[0]))
        self.ids = [str(row['id']) for row in current]
        self.seen.update(self.ids)
        self.sup.mark_ready()
        (self.out / (arm.name + '-identity.json')).write_text(json.dumps({'ranks': cast(Json, current)}) + '\n')

    def allowed(self, arm: Arm) -> int:
        return allowance(self.group, arm, arm == arms(self.group)[0])

    def greedy(self, arm: Arm) -> dict[str, Json]:
        # Port of DEP1 9b72534. No alternate target comparison or near-tie implementation.
        deadline = min(self.submit, self.started_at + self.allowed(arm))
        ck.require(self.clock() < deadline, 'premeasurement greedy budget exhausted')
        if self.group in upf_plan.GROUPS:
            return self.upf_greedy(arm, deadline)
        output = self.out / f'{arm.name}-greedy.json'
        limit = near_tie_nats(self.group, arm)
        threshold = ('--near-tie-nats', repr(limit)) if limit != greedy_driver.NEAR_TIE_NATS else ()
        if self.group in upx_plan.GROUPS:
            argv = self.upx_driver('greedy', output, deadline, f'{self.tag}-{arm.name}-greedy')
        else:
            argv = self.driver('greedy', output, '--deadline', str(deadline), *threshold)
        self.call(f'{arm.name}-greedy', argv, timeout=min(greedy_s(arm), deadline-self.clock()), inflight=1)
        result = ck.read(output)
        greedy_checks(result)
        ck.require(not threshold or result.get('near_tie_nats') == limit, 'greedy ran without its registered near-tie limit')
        if self.group in upx_plan.GROUPS:
            ck.require(result.get('greedy_registration') == upx_plan.REGISTRATION, 'greedy ran without upx-0p9')
            self.upx_state[f'greedy-{position(self.group, arm)}'] = output
        ck.require(self.clock() < deadline, 'experimental greedy exceeded premeasurement allowance')
        return result

    def warmup(self, arm: Arm) -> None:
        ck.require(arm.name not in self.round_baseline, 'measurement baseline cannot be replaced')
        # DEP1 ordering: six greedy checks, then unchanged discarded OPS1 shapes1..4.
        deadline = min(self.submit, self.started_at + self.allowed(arm))
        ck.require(self.clock() < deadline, 'premeasurement warmup budget exhausted')
        # The OPS1 warm-up's own waves (up to max(SHAPES) concurrent) are owned demand, not foreign demand on :8001
        # (UP3-k5-W-20261027T041627Z stopped at the concurrency-3 wave with owned 0).
        self.owned(arm.name + '-warmup', 'sparkhead', self.tag + '-' + arm.name + '-warmup', f'cd ~/{self.remote} && bash offload/launch/warmup_serve.sh 8001 '
            'DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2', timeout=min(WARMUP_S, deadline-self.clock()),
            inflight=max(ops1_warmup.SHAPES))
        # DEP1 9b72534: establish one FULL baseline after greedy and OPS1.
        baseline = self.stats(f'{arm.name}-baseline')
        ck.counters(baseline, before=baseline)
        self.round_baseline[arm.name] = baseline
        self.counter_seen[arm.name] = {}
        self.baseline_check(arm, baseline)

    def baseline_check(self, arm: Arm, current: dict[str, Json]) -> None:
        """DEP1 full baseline, retaining every subsequently observed sparse key too."""
        ck.require(arm.name in self.round_baseline, 'missing post-greedy/post-warmup measurement baseline')
        baseline = self.round_baseline[arm.name]
        ck.counters(current, before=baseline)
        before, after = record(baseline['counters']), record(current['counters'])
        seen = self.counter_seen[arm.name]
        for key, value in seen.items():
            ck.require(key in after and number(after[key]) >= value, f'{key}: missing/reset counter')
        for key, field in after.items():
            n = number(field)
            ck.require(math.isfinite(n) and n >= 0, f'{key}: invalid measurement counter')
        for key in ('rounds_failed', 'rounds_some_failed'):
            ck.require(number(after.get(key, 0)) == number(before.get(key, 0)), f'{key} grew after warmup')
        fires = number(record(current['async_watchdog'])['fires'])
        ck.require(math.isfinite(fires) and fires >= 0
            and fires == number(record(baseline['async_watchdog'])['fires']), 'watchdog grew after warmup')
        self.counter_seen[arm.name] = {key: number(value) for key, value in after.items()}

    def rank_logs(self, arm: Arm, host: str, rank: int) -> dict[str, Json]:
        summary = pf.logs(self.json(arm.name + '-logs-' + host, host, 'python3 -c ' + shlex.quote(pf.LOG_FILTER)
            + ' ' + shlex.quote(ck.STOP_LOG.pattern + '|fused SWA metadata off|warmup compile skipped|NCCL stays')),
            rank, 327680)
        return summary

    def upx_driver(self, mode: str, output: Path, deadline: float, salt: str, *extra: str) -> list[str]:
        return [sys.executable, '-m', 'offload.up3.upx_driver', '--execute', '--mode', mode, '--output', str(output),
                '--data-dir', str(self.data_dir), '--deadline', str(deadline), '--salt', salt, *extra]

    def upx_gates(self, arm: Arm, deadline: float) -> dict[str, Json]:
        """R-UPX design gates of this start (offload/upx WINDOWS.md), executed through upx_driver; any failure STOPs."""
        where = position(self.group, arm)
        before = self.stats(arm.name + '-upx-before')
        self.baseline_check(arm, before)
        state = self.upx_state
        out: dict[str, Json] = {}

        def drive(kind: str, *extra: str) -> Path:
            path = self.out / f'{arm.name}-upx-{kind}.json'
            ck.require(self.clock() < deadline, f'{kind}: R-UPX gate allowance exhausted before submission')
            self.call(f'{arm.name}-upx-{kind}', self.upx_driver(kind, path, deadline, f'{self.tag}-{arm.name}-{kind}', *extra),
                timeout=min(2 * upx_plan.GATE_S[kind if kind != 'needle' else 'needle-150k'] + 30, deadline - self.clock()),
                inflight=1)
            ck.require(self.clock() < deadline, f'{kind}: R-UPX gate exceeded its allowance')
            return path

        for kind in upx_plan.gates(self.group, where):
            if kind == 'regions':
                path = drive('regions', *(('--pins', str(state['regions-0'])) if where else ()))
                state[f'regions-{where}'] = path
                out['regions'] = record(ck.read(path))['pins']
            elif kind == 'band-probes':
                state[f'band-probes-{where}'] = drive('band-probes', '--regions', str(state['regions-0']))
                if where == 2:
                    probes = [record(ck.read(state[f'band-probes-{i}'])['probes']) for i in range(3)]
                    self.upx_screen = band.kl_screen(*probes)
                    ck.require(self.upx_screen['pass'] is True, 'band KL screen failed (KL(C1||X) above max(0.005, 2 x A/A))')
                    out['band-screen'] = self.upx_screen
                else:
                    out['band-probes'] = 'RECORDED'
            elif kind in ('band-greedy', 'long-greedy'):
                reference = ('--reference', str(state[f'{kind}-0'])) if where else ()
                state[f'{kind}-{where}'] = drive(kind, '--regions', str(state['regions-0']), *reference)
                result = ck.read(state[f'{kind}-{where}'])
                out[kind] = {k: result.get(k) for k in ('identical', 'near_tie', 'threshold_nats')}
            elif kind == 'engagement':
                rows: dict[str, Json] = {}
                for host in pf.HOSTS:
                    text = self.ssh(f'{arm.name}-engagement-{host}', host, 'docker logs dsv41-serve 2>&1 | grep -F -m1 '
                                    + shlex.quote(band.SPLIT_LOG) + ' || true')
                    engaged = band.split_engaged(text)
                    ck.require((engaged is not None) == (arm.role == 'split'),
                               f'{host}: 0064 engagement {engaged!r} contradicts arm {arm.role}')
                    rows[host] = engaged
                out['engagement'] = rows
            elif kind == 'kl':
                candidate = [state[k] for k in (f'greedy-{where - 1}', f'band-greedy-{where - 1}', f'long-greedy-{where - 1}')
                             if k in state]
                bound = upx_plan.kl_bound(self.upx_screen)
                path = drive('kl', '--candidate', *map(str, candidate), '--bound', repr(bound))
                result = ck.read(path)
                ck.require(result.get('pass') is True, 'greedy-divergence KL above the registered bound')
                out['kl'] = {'bound': bound, 'compared': result.get('compared')}
            else:
                ck.require(kind == 'needle-150k', f'unknown R-UPX gate {kind}')
                path = drive('needle', '--ledger', str(self.out / 'upx-needle-ledger.jsonl'))
                out['needle-150k'] = {k: ck.read(path).get(k) for k in ('recall', 'of', 'foreign_codes')}
        after = self.stats(arm.name + '-upx-after')
        self.baseline_check(arm, after)
        ck.counters(after, before=before)
        return out

    # ------------------------------------------------------------ R-UPF (offload/upfix WINDOW.md through upf_plan)
    def upf_driver(self, mode: str, output: Path, deadline: float, salt: str, *extra: str) -> list[str]:
        return [sys.executable, '-m', 'offload.up3.upf_driver', '--execute', '--mode', mode, '--output', str(output),
                '--data-dir', str(self.data_dir), '--deadline', str(deadline), '--salt', salt, *extra]

    def upf_image_bases(self) -> None:
        """S93 runner note: every UPF overlay's base file in the image equals its manifest base_sha256 before any start.
        Production (K1) mounts none of these paths, so its running container shows the image bytes. A chained window has
        no production container: recorded as unchecked."""
        expected = upf_plan.image_bases(self.group)
        status: dict[str, Json] = {'expected': cast(Json, expected)}
        if self.chain_mode == 'chained':
            status['status'] = 'UNCHECKED: chained window, production down'
        else:
            for host in pf.HOSTS:
                got = self.json(f'upf-image-bases-{host}', host,
                                'docker exec -i dsv41-serve python3 - ' + shlex.join(expected), pf.IMAGE_BASES)
                ck.require(got == expected, f'{host}: UPF overlay base files in the image differ from their manifests')
            status['status'] = 'MATCH'
        (self.out / 'upf-image-bases.json').write_text(json.dumps(status, indent=1) + '\n')

    def upf_pre(self) -> dict[str, Json]:
        """R-UPF T0-T2 after the HOLD and the production snapshot, before C1. T0 (./stop.sh + GPU-idle on both Sparks)
        failing stops the window; a failed, crashed or timed-out T1/T2, or a refused verify-topk, skips X-topk only."""
        kinds = upf_plan.pre(self.group) if self.group in upf_plan.GROUPS else ()
        out: dict[str, Json] = {'kinds': list(kinds), 'skip': {}}
        if not kinds:
            return out
        ck.require(kinds == ('stop_idle', 'jit_build', 'bitcompare'), f'unknown R-UPF pre-steps {kinds}')

        def step(name: str, script: str, timeout: float) -> str:
            ck.require(self.clock() < self.submit, f'UPF {name}: HOLD submit bound reached')
            command = hold_admission.command('stop', self.tag + '-hold', self.admission_token, ['bash', '-c', script])
            return self.owned('upf-' + name, 'sparkhead', f'{self.tag}-upf-{name}', command, timeout=timeout)

        def rc(text: str, name: str) -> int:
            found = re.findall(rf'^UPF {name} RC ([0-9]+)$', text, re.M)
            ck.require(len(found) == 1, f'UPF {name}: exit code unreported')
            return int(found[0])

        ck.require(UPF_T0_IDLE in step('t0', upf_t0(self.remote), UPF_T0_S).splitlines(),
                   'UPF T0: production stop / GPU-idle check on both Sparks unproved')
        out['t0'] = 'IDLE'
        topk: dict[str, Json] = {'status': 'SKIPPED'}
        t1 = rc(step('t1', upf_t1(self.remote), upf_plan.T_TIMEOUT_S + 60), 'T1')
        out['t1'] = {'rc': t1}
        if t1 != 0:
            topk['reason'] = f'T1 jit_build rc {t1}' + (' (image id differs)' if t1 == 90 else '')
        else:
            t2 = rc(step('t2', upf_t2(self.remote), upf_plan.T_TIMEOUT_S + 60), 'T2')
            out['t2'] = {'rc': t2, 'meaning': 'PASS' if t2 == 0 else 'FAIL' if t2 == 2 else
                         'TIMED OUT' if t2 in (124, 137) else 'CRASHED'}
            text = self.ssh('upf-receipts', 'sparkhead', upf_receipts(self.remote), timeout=120)
            lines = text.splitlines()
            receipts: dict[str, dict[str, Json]] = {}
            for name in ('receipt', 'bitcompare'):
                at = lines.index(f'UPF RECEIPT {name}')
                body = lines[at + 1].strip() if at + 1 < len(lines) else ''
                receipts[name] = record(cast(Json, json.loads(body))) if body.startswith('{') else {}
                (self.out / f'upf-{name}.json').write_text(json.dumps(receipts[name], indent=1) + '\n')
            verify = rc(text, 'VERIFY')
            compare = receipts['bitcompare']
            topk['bitcompare'] = {k: compare.get(k) for k in ('verdict', 'complete', 'cases', 'cases_done', 'capability')}
            topk['verify_topk_rc'] = verify
            library = str(receipts['receipt'].get('library_sha256', ''))
            if t2 != 0:
                topk['reason'] = f'T2 bit-compare rc {t2} ({record(out["t2"])["meaning"]})'
            elif verify != 0 or re.fullmatch('[0-9a-f]{64}', library) is None or compare.get('library_sha256') != library:
                topk['reason'] = f'verify-topk refused the receipts (rc {verify})'
            else:
                topk.update(status='READY', library_sha256=library)
                self.upf_library = library
        out['topk'] = topk
        if topk['status'] != 'READY':
            for start in upf_plan.starts(self.group):
                if start.arm == upf_plan.TOPK:
                    self.skipped[start.label] = str(topk['reason'])
            out['skip'] = cast(Json, dict(self.skipped))
        (self.out / 'upf-pre.json').write_text(json.dumps(out, indent=1) + '\n')
        return out

    def upf_identity(self, arm: Arm, rank: int, live: dict[str, Json]) -> None:
        """Identity relative to C1 (coordinator 2026-10-27). C1: same image, engine, model target files and Engram
        identity as production; env / mounts / argv differences are RECORDED (UPF.env starts from D-kv6-async-pack.env).
        Every later start: C1 plus exactly its arm's dist-packages mounts and container environment, same argv."""
        prod = self.production[rank]
        ck.require(live.get('image') == prod.get('image') == pf.IMAGE, f'UPF {arm.name}: image differs from production')
        for key in ('engine', 'files', 'engram_identity'):
            ck.require(live.get(key) == prod.get(key), f'UPF {arm.name}: {key} differs from production')
        mounts = {k: v for k, v in record(live['mounted']).items() if k != '/serve.sh'}
        argv = shlex.split(str(live['argv']).replace('\\\n', ' '))
        if position(self.group, arm) == 0:
            ck.require(upf_plan.is_control(arm.role), 'R-UPF C1 must be the control')
            env, base_env = record(live['env']), record(prod['env'])
            base_mounts = {k: v for k, v in record(prod['mounted']).items() if k != '/serve.sh'}
            base_argv = shlex.split(str(prod['argv']).replace('\\\n', ' '))
            if '--port' in base_argv:
                base_argv[base_argv.index('--port') + 1] = '8001'
            self.upf_c1_diff[pf.HOSTS[rank]] = {
                'env_added': cast(Json, sorted(set(env) - set(base_env))),
                'env_removed': cast(Json, sorted(set(base_env) - set(env))),
                'env_changed': cast(Json, sorted(k for k in set(env) & set(base_env) if env[k] != base_env[k])),
                'mounts_added': cast(Json, sorted(set(mounts) - set(base_mounts))),
                'mounts_removed': cast(Json, sorted(set(base_mounts) - set(mounts))),
                'mounts_changed': cast(Json, sorted(k for k in set(mounts) & set(base_mounts) if mounts[k] != base_mounts[k])),
                'argv_equal_but_port': argv == base_argv}
            (self.out / 'upf-c1-vs-production.json').write_text(json.dumps(self.upf_c1_diff, indent=1) + '\n')
            self.upf_reference[rank] = live
            return
        ck.require(rank in self.upf_reference, 'R-UPF start before C1 identity')
        ref = self.upf_reference[rank]
        delta = upf_plan.env_delta(arm.role)
        ck.require(record(live['env']) == {**record(ref['env']), **delta},
                   f'UPF {arm.name}: container env differs from C1 + {sorted(delta)}')
        expected = {k: v for k, v in record(ref['mounted']).items() if k != '/serve.sh'}
        expected.update(upf_plan.mount_delta(arm.role))
        ck.require(mounts == expected, f'UPF {arm.name}: mounted sources differ from C1 + its documented overlays')
        ck.require(argv == shlex.split(str(ref['argv']).replace('\\\n', ' ')), f'UPF {arm.name}: argv differs from C1')

    def upf_pins(self, role: str) -> dict[str, str]:
        """Installed bytes of a UPF arm: production's control pins + the arm's overlays + the T1/T2-verified library."""
        pins = source_pins('control')
        pins.update(upf_plan.mount_delta(role))
        if upf_configs.ARMS[role]['UPF_TOPK'] == '1':
            ck.require(bool(self.upf_library), 'X-topk without a T1/T2-verified dsv41_topk library')
            pins[upf_configs.TOPK_SO_CONTAINER] = self.upf_library
        return pins

    def upf_greedy(self, arm: Arm, deadline: float) -> dict[str, Json]:
        """Six K1 reference greedy at 0.9+1e-3 nats with distributions recorded (C1 must be 6/6 IDENTICAL); C2's step is
        the KL re-probe of every candidate's recorded divergences."""
        where = position(self.group, arm)
        if 'kl' in upf_plan.kinds(self.group, where):
            return self.upf_kl(arm, deadline)
        ck.require('greedy' in upf_plan.kinds(self.group, where), f'{arm.name}: no greedy step in the design')
        output = self.out / f'{arm.name}-greedy.json'
        self.call(f'{arm.name}-greedy', self.upf_driver('greedy', output, deadline, f'{self.tag}-{arm.name}-greedy'),
                  timeout=min(greedy_s(arm), deadline - self.clock()), inflight=1)
        result = ck.read(output)
        ck.require(result.get('greedy_registration') == upf_driver.REGISTRATION
                   and result.get('near_tie_nats') == upf_configs.GREEDY_NEAR_TIE_NATS, 'greedy ran without upf-0p9')
        try:
            greedy_checks(result)
            if upf_plan.is_control(arm.role):
                ck.require(result.get('identical') == 6,
                           f'UPF {arm.name} greedy is not 6/6 IDENTICAL to the K1 reference: the base drifted')
        except ValueError as exc:
            if upf_plan.is_control(arm.role):
                raise
            raise ArmStop('greedy', str(exc), {k: result.get(k) for k in ('identical', 'near_tie')}) from exc
        if not upf_plan.is_control(arm.role):
            self.upf_state[arm.name] = output
        ck.require(self.clock() < deadline, 'experimental greedy exceeded premeasurement allowance')
        return {**result, 'pass': True}

    def upf_kl(self, arm: Arm, deadline: float) -> dict[str, Json]:
        # C1 is 6/6 IDENTICAL by gate, so no C1/C2 A/A rows exist: the bound is the registered 0.005 floor.
        bound = upf_windows.KL_BOUND_FLOOR
        out: dict[str, Json] = {'bound': bound, 'aa_floor': None, 'registration': upf_driver.REGISTRATION, 'kl': {}}
        for label, source in self.upf_state.items():
            path = self.out / f'{arm.name}-kl-{label}.json'
            ck.require(self.clock() < deadline, f'KL re-probe of {label}: allowance exhausted before submission')
            self.call(f'{arm.name}-kl-{label}', self.upf_driver('kl', path, deadline, f'{self.tag}-{arm.name}-kl-{label}',
                      '--candidate', str(source), '--bound', repr(bound)),
                      timeout=min(2 * upf_windows.GATE_S['kl'] + 30, deadline - self.clock()), inflight=1)
            result = ck.read(path)
            record(out['kl'])[label] = {'pass': result.get('pass') is True, 'compared': result.get('compared'),
                                        'bound': result.get('bound')}
        return out

    def upf_gates(self, arm: Arm, deadline: float) -> dict[str, Json]:
        """Needles of a candidate (WINDOW.md order); a failed needle is an arm-level STOP."""
        where = position(self.group, arm)
        if upf_plan.is_control(arm.role):
            out: dict[str, Json] = {'status': 'identity relative to C1 verified', 'registration': upf_driver.REGISTRATION}
            if where == 0:
                out['c1_vs_production'] = cast(Json, self.upf_c1_diff)
            return out
        needles: dict[str, Json] = {}
        for kind in upf_plan.gates(self.group, where):
            target = 150000 if kind == 'needle-150k' else 300000
            path = self.out / f'{arm.name}-{kind}.json'
            ck.require(self.clock() < deadline, f'{kind}: R-UPF gate allowance exhausted before submission')
            self.call(f'{arm.name}-{kind}', self.upf_driver('needle', path, deadline, f'{self.tag}-{arm.name}-{kind}',
                      '--target', str(target), '--ledger', str(self.out / 'upf-needle-ledger.jsonl')),
                      timeout=min(2 * upf_windows.GATE_S[kind] + 60, deadline - self.clock()), inflight=1)
            result = ck.read(path)
            needles[kind] = {k: result.get(k) for k in ('pass', 'target', 'recall', 'of', 'foreign_codes', 'error')}
            if result.get('pass') is not True:
                raise ArmStop(kind, str(result.get('error') or 'needle failed'), {'needles': needles})
            ck.require(self.clock() < deadline, f'{kind}: R-UPF gate exceeded its allowance')
        return {'policy': 'R-UPF identity gates (offload/upfix WINDOW.md)', 'registration': upf_driver.REGISTRATION,
                'needles': needles}

    def upf_soak(self, arm: Arm) -> dict[str, Json]:
        """600-s four-stream soak with per-process RSS sampling; C1 records OFF, the candidate is judged by soak_ok."""
        path = self.out / f'{arm.name}-soak.json'
        before = self.stats(arm.name + '-soak-before')
        self.baseline_check(arm, before)
        self.call(f'{arm.name}-soak', self.upf_driver('soak', path, self.submit, f'{self.tag}-{arm.name}',
                  '--cuts', str(self.out / 'cuts.json'), '--seconds', str(upf_windows.SOAK_S)),
                  timeout=upf_windows.SOAK_S + 300, inflight=upf_driver.SOAK_STREAMS)
        result = ck.read(path)
        growth = {k: number(v) for k, v in record(result['growth_mib']).items()}
        out: dict[str, Json] = {'growth_mib': cast(Json, growth), 'requests': result.get('requests'),
                                'samples': len(ck.rows(result['samples'])), 'transient_roles': result.get('transient_roles')}
        if upf_plan.is_control(arm.role):
            self.upf_soak_off = growth
            out['mode'] = 'OFF'
            return out
        ck.require(self.upf_soak_off is not None, 'soak ON without C1 soak OFF')
        out['mode'] = 'ON'
        try:
            out['ok'] = upf_windows.soak_ok(cast(dict[str, float], self.upf_soak_off), growth)
        except ValueError as exc:
            out.update(ok=False, error=str(exc))
        return out

    def qualify(self, arm: Arm) -> dict[str, Json]:
        deadline = min(self.submit, self.started_at + self.allowed(arm))
        ck.require(self.clock() < deadline, 'startup/warmup/qualification budget exhausted')
        if self.group in upf_plan.GROUPS:
            return self.upf_gates(arm, deadline)
        upx: dict[str, Json] = {}
        if self.group in upx_plan.GROUPS:
            upx = {'registration': upx_plan.REGISTRATION, 'upx_gates': self.upx_gates(arm, deadline)}
            if arm.role in upx_plan.UPX_ROLES:
                return {'policy': 'upx-0p9: API-level gates replace the never-built resident-worker harness', **upx}
        if arm.role == 'control':
            return {'status': 'unchanged production source/env verified', **upx}
        host_gates.admit(arm.role)
        results: dict[str, Json] = {'policy': 'Round4 config-only; resident adapter not required', **upx}
        if arm.role in NO_HOST_GATE:
            return results
        gate = HOST_GATE.get(arm.role, arm.role)
        for rank, host in enumerate(pf.HOSTS):
            ck.require(self.clock() < deadline, 'qualification deadline exhausted before submission')
            command = (f'cd ~/{self.remote} && python3 -m offload.up3.host_gates '
                f'--arm {gate} --rank {rank} --container {shlex.quote(self.ids[rank])}')
            text = self.ssh(arm.name + '-host-gate-' + host, host, command,
                timeout=min(15, deadline - self.clock()))
            receipt = record(cast(Json, json.loads(text)))
            ck.require(self.clock() < deadline, 'host evidence exceeded its full block allowance')
            host_gates.validate(receipt, gate, rank, self.ids[rank])
            results[host] = receipt
        return results

    def driver(self, mode: str, output: Path, *extra: str) -> list[str]:
        return [sys.executable, '-m', 'offload.up3.cells', '--execute', '--mode', mode, '--output', str(output),
            '--data-dir', str(self.data_dir), '--deadline', str(self.submit), *extra]

    def build_cuts(self) -> None:
        self.call('public-cuts', self.driver('cuts', self.out / 'cuts.json'), timeout=180)

    def cell(self, arm: Arm, name: str) -> dict[str, Json]:
        before = self.stats(arm.name + '-' + name + '-before')
        self.baseline_check(arm, before)
        output = self.out / (arm.name + '-' + name + '.json')
        repeats = repeats_for(self.group, name, arm)
        if name in band.BAND_CELLS:
            self.call(arm.name + '-' + name, self.upx_driver('band-cell', output, self.submit, self.tag + '-' + arm.name,
                '--cell', name, '--repeats', str(repeats), '--cuts', str(self.out / 'cuts.json'),
                '--regions', str(self.upx_state['regions-0'])),
                timeout=band.BAND_CELLS[name].estimate_s * repeats * 2 + 30, inflight=band.DECODERS + 1)
        else:
            extra = ('--repeats', str(repeats)) if self.group in (*win2_plan.GROUPS, *upx_plan.GROUPS, *upf_plan.GROUPS) else ()
            self.call(arm.name + '-' + name, self.driver('cell', output, '--cell', name, '--cuts', str(self.out / 'cuts.json'),
                '--salt', self.tag + '-' + arm.name, '--ledger', str(self.out / 'requests-ledger.jsonl'), *extra),
                timeout=cells.ESTIMATES[name] * repeats * 2 + 30, inflight=3 if name == cells.CELLS[-1] else int(name[-1]))
        result = ck.read(output)
        ck.require(result.get('cell') == name, 'cell artifact identity differs')
        after = self.stats(arm.name + '-' + name + '-after')
        self.baseline_check(arm, after)
        ck.counters(after, before=before)
        ck.require(number(record(after['counters']).get('rounds_ok', 0)) > number(record(before['counters']).get('rounds_ok', 0)),
            'remote drafter took no successful round')
        result['proxy_before'], result['proxy_after'] = before, after
        output.write_text(json.dumps(result, allow_nan=False) + '\n')
        return result

    def finish(self, arm: Arm) -> dict[str, Json]:
        result: dict[str, Json] = {}
        if self.group in upf_plan.GROUPS and 'soak' in upf_plan.kinds(self.group, position(self.group, arm)):
            result['soak'] = self.upf_soak(arm)   # WINDOW.md: the soak follows the cells
        final = self.stats(arm.name + '-final-stats')
        self.baseline_check(arm, final)
        result['startup_baseline'] = self.round_baseline[arm.name]
        result['proxy'] = final
        result['observed_counter_keys'] = cast(Json, sorted(self.counter_seen[arm.name]))
        for rank, host in enumerate(pf.HOSTS):
            ck.require(self.ssh(arm.name + '-id-' + host, host, "docker inspect --format '{{.Id}}' dsv41-serve").strip() == self.ids[rank],
                'container changed during arm')
            result[host] = self.rank_logs(arm, host, rank)
        ck.require(self.sup.arm_minima is not None and set(self.sup.arm_minima) == set(pf.HOSTS), 'missing rank memory minima')
        result['memory_min_kb'] = cast(Json, dict(cast(dict[str, int], self.sup.arm_minima)))
        result['clock_caps_mhz'] = cast(Json, self.clock_caps.get(arm.name, []))
        return result

    def restore(self, *, production_stopped: bool) -> dict[str, Json]:
        self.sup.close()
        original = self.plain
        original_call = self.call
        def recovery_call(label: str, argv: list[str], *, stdin: str | None = None,
                          timeout: float = 120, inflight: int = 0) -> str:
            if stdin is not None:
                import base64
                encoded = base64.b64encode(stdin.encode()).decode()
                argv = [*argv[:-1], 'printf %s ' + shlex.quote(encoded) + ' | base64 -d | (' + argv[-1] + ')']
            rc, text = self.plain('verify-' + label, argv, timeout)
            ck.require(rc == 0, 'recovery identity command failed; HOLD stays open')
            return text
        self.call = recovery_call
        # Clamp cleanup, W8 and close transactions to ONE original HOLD deadline.
        def bounded(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
            if label in ('cancel-close', 'retain-close'):
                return original(label, argv, min(30, timeout))
            left = self.restore_by - self.clock()
            ck.require(left > 0, f'UP3 HOLD+{self.bounds[2] // 60} exhausted; manual restore required; HOLD stays open')
            result = original(label, argv, min(left, timeout))
            ck.require(self.clock() < self.restore_by, 'UP3 recovery command exceeded original deadline')
            return result
        self.plain = bounded
        try:
            # Fence the HOLD writer BEFORE rereading it: a late opener must not
            # create a HOLD after recovery decided that production was untouched.
            for tag in list(self.owner_tags):
                self.fence_phase(tag)
            if not self.hold_written:
                rc, text = self.plain('hold-reread', [*pf.SSH, 'sparkhead',
                    hold_admission.command('check', self.tag + '-hold', self.admission_token)], 60)
                ck.require(rc == 0, 'UP3 HOLD ownership unreadable; HOLD stays open')
                self.hold_written = f'UP3-WINDOW {self.tag}:' in text
                if self.hold_written:
                    opened = pf.hold_opened(text, self.tag, self.clock())
                    self.restore_by = opened + self.bounds[2]
            if not self.hold_written:
                ck.require(not production_stopped, 'UP3 production may be stopped without owned HOLD')
                if chain.end_action(True, False) == 'abort':   # S92 FIX 1: production is down under the marker
                    return self.chain_abort()
                return {'status': 'NOT NEEDED: no owned HOLD; all remote mutators fenced and absent'}
            ck.require(math.isfinite(self.restore_by), 'UP3 missing immutable restore deadline')
            # A chained window found production down: W8 must start it even if no arm ran.
            production_stopped = production_stopped or self.chain_mode == 'chained'
            if production_stopped:
                # dockerd children are outside cgroups: explicitly stop/remove
                # the same named experiment containers and require no GPU apps.
                rc, text = self.recovery('experiment-stop', [*pf.SSH, 'sparkhead',
                    f'cd ~/{self.remote} && ./stop.sh </dev/null'], 120)
                ck.require(rc == 0, 'UP3 experiment stop failed; HOLD stays open')
                for host in pf.HOSTS:
                    rc, text = self.plain('jit-wait-' + host, [*pf.SSH, host,
                        spark_quiet.command('UP3 EXPERIMENT ABSENT',
                            deadline=min(self.restore_by - 90, self.clock() + 350))], 360)
                    ck.require(rc == 0 and 'UP3 EXPERIMENT ABSENT' in text.splitlines(),
                               f'{host}: experiment/container/compute absence unproved; HOLD stays open')
            result: dict[str, Json] = {}
            steps = ('start', 'check', 'close') if production_stopped else ('check', 'close')
            for step in steps:
                # Close uses an exact suffix to install the HOLD-unlink shim on
                # every receiving host. W8 still performs its original checks.
                if step == 'close':
                    # One -restore-close scope per window (the rm shim needs HOLD's own tag); after a chained close used
                    # it, production is restored and verified above but this HOLD must be closed by the coordinator.
                    ck.require(f'{self.tag}-restore-close' not in self.owner_tags,
                               'close scope already spent by the chained close; production restored and verified; '
                               'HOLD stays open for the coordinator')
                    # Compare complete production identity BEFORE permitting deferred close.
                    for rank, host in enumerate(pf.HOSTS):
                        self.verify_sources(host, 'control')
                        restored = self.capture(host, 'restored')
                        if self.production:
                            prod = self.production[rank]
                            ck.require(all(restored[k] == prod[k] for k in
                                ('image', 'engine', 'files', 'mounted', 'env', 'argv', 'engram_identity')),
                                'restored production identity differs; HOLD stays open')
                    tag = f'{self.tag}-restore-close'
                    deadline = self.restore_by - 90
                    self.remember(tag, deadline)
                    ck.require(self.clock() < deadline, 'UP3 final close budget exhausted; HOLD stays open')
                    try:
                        rc, text = self.plain(step, ['bash', '-c', owner.command('launch', tag,
                            ['bash', str(pf.RESTORE), step], deadline=deadline)], 120)
                    finally:
                        self.fence_phase(tag)
                else:
                    rc, text = self.recovery(step, ['bash', str(pf.RESTORE), step], 900 if step == 'start' else 600)
                strict_restored(step, rc, text)  # requires six distinct IDENTICAL, not near ties
                result[step] = 'PASS'
            for prior in list(self.owner_tags):
                self.fence_phase(prior)
            ck.require(self.restore_by - self.clock() >= 90, 'UP3 final close needs 90s reserve; HOLD stays open')
            tag = f'{self.tag}-restore-close'
            try:
                try:
                    rc, text = self.plain('commit-close', [*pf.SSH, 'sparkhead', owner.command('commit-close', tag,
                        deadline=min(self.restore_by-5, self.clock()+25))], 30)
                    ck.require(rc == 0 and owner.CLOSED in text.splitlines(), 'UP3 final owned HOLD close unconfirmed')
                finally:
                    rc, text = self.plain('cancel-close', [*pf.SSH, 'sparkhead', owner.command('cancel-close', tag)], 30)
                    ck.require(rc == 0 and 'SCOPED CLOSE COMMIT FENCED' in text.splitlines(), 'UP3 final close fence unconfirmed')
                result['restore_deadline_met'] = self.clock() < self.restore_by
                ck.require(result['restore_deadline_met'] is True, 'UP3 restore completed late')
            except (Exception, KeyboardInterrupt):
                rc, text = self.plain('retain-close', [*pf.SSH, 'sparkhead', owner.command('retain-close', tag)], 30)
                ck.require(rc == 0 and owner.RETAINED in text.splitlines(), 'UP3 original HOLD retention unconfirmed')
                raise
            result['status'] = 'RESTORED: K1 START VERIFIED, greedy 6/6 IDENTICAL, PRODUCTION HOLD OK, window closed'
            return result
        finally:
            self.plain = original
            self.call = original_call
            (self.out / 'arm-memory.json').write_text(json.dumps(self.sup.arm_memory) + '\n')

    def archive_controls(self, starts: dict[str, Json]) -> dict[str, Json]:
        """WIN2 control archive (schema v3): every COMPLETE control start's decision-cell repeats (also after a later
        stop), each with its own start's proven clock cap (win2_history.start_clock_cap: this window's start log). R-UPF
        rows are windows.archive_rows, which reads the production commit from summary.json: main() appends them."""
        if self.group in upf_plan.GROUPS:
            return {'status': 'DEFERRED', 'reason': 'R-UPF rows are appended by main() once summary.json exists (upf_archive)'}
        if self.group not in (*win2_plan.GROUPS, *upx_plan.GROUPS):
            return {'rows': 0}
        rows_for = win2_plan.archive_rows if self.group in win2_plan.GROUPS else upx_plan.archive_rows
        rows = rows_for(self.group, self.out.name, starts, production=b.PRODUCTION, tag=b.TAG,
                        date=time.strftime('%Y%m%d', time.gmtime(self.clock())),
                        cap_of=lambda start: win2_history.start_clock_cap(self.out, start))
        before = dv2.archive_head(archive_path())
        if rows:
            dv2.archive_append(archive_path(), rows)
        return {'rows': len(rows), 'head_before': before, 'head_after': dv2.archive_head(archive_path())}

    def chain_abort(self) -> dict[str, Json]:
        """WIN2-R2 chain.abort: no owned HOLD while the chain marker is present. Strict W8 start|check|close under the
        reviewed scoped recovery's own HOLD + the 15-min watch; clear() only on RESTORED + watch PASS."""
        summary = chain.abort(self.out / 'chain-abort', log=lambda message: print(message, flush=True))
        restore = record(summary.get('restore', {}))
        restored = str(summary.get('status', '')).startswith('RESTORED')
        return {**restore, 'status': str(restore.get('status')) if restored else f'FAILED: chain abort: {summary.get("status")}',
                'chain_abort': summary}

    def placeholders(self, label: str, command: str, marker: str) -> dict[str, Json]:
        """Unscoped default-socket placeholder command on both Sparks (keep / release; see KEEP_PLACEHOLDER)."""
        out: dict[str, Json] = {}
        for host in pf.HOSTS:
            rc, text = self.plain(f'{label}-{host}', [*pf.SSH, host, command], 60)
            ck.require(rc == 0 and marker in text.splitlines(), f'{host}: hold placeholder {label} unconfirmed')
            out[host] = marker
        return out

    def release_placeholders(self) -> dict[str, Json]:
        return self.placeholders('placeholder-release', RELEASE_PLACEHOLDER, 'RELEASED')

    def chained_end(self) -> dict[str, Json]:
        """WIN2 chain (non-last window, no stop): chain placeholders kept up, experiment stopped, Sparks proven quiet,
        own HOLD removed (unlink recorded in the -restore-close scope, then the fenced commit-close), production LEFT
        DOWN with the CHAINED marker. Any failure falls back to the strict restore while the HOLD is ours."""
        self.sup.close()
        try:
            ck.require(self.hold_written and math.isfinite(self.restore_by), 'chained end without an owned HOLD')
            # Default-socket placeholders BEFORE the HOLD phase (and its private placeholders) is fenced.
            kept = self.placeholders('placeholder-keep', KEEP_PLACEHOLDER, PLACEHOLDER_UP)
            for tag in list(self.owner_tags):
                self.fence_phase(tag)
            rc, _ = self.recovery('experiment-stop', [*pf.SSH, 'sparkhead', f'cd ~/{self.remote} && ./stop.sh </dev/null'], 120)
            ck.require(rc == 0, 'UP3 chained experiment stop failed; HOLD stays open')
            quiet: dict[str, Json] = {}
            for host in pf.HOSTS:
                rc, text = self.plain('jit-wait-' + host, [*pf.SSH, host, spark_quiet.command('UP3 EXPERIMENT ABSENT',
                    deadline=min(self.restore_by - 90, self.clock() + 350))], 360)
                ck.require(rc == 0 and 'UP3 EXPERIMENT ABSENT' in text.splitlines(),
                           f'{host}: experiment/container/compute absence unproved; HOLD stays open')
                quiet[host] = 'UP3 EXPERIMENT ABSENT'
            self.placeholders('placeholder-probe', PROBE_PLACEHOLDER, 'PRESENT')
            tag = f'{self.tag}-restore-close'
            deadline = self.restore_by - 90
            self.remember(tag, deadline)
            ck.require(self.clock() < deadline, 'UP3 chained close budget exhausted; HOLD stays open')
            try:
                rc, text = self.plain('chained-close', ['bash', '-c', owner.command('launch', tag,
                    [*pf.SSH, 'sparkhead', CLOSE_OWN_HOLD], deadline=deadline)], 120)
            finally:
                self.fence_phase(tag)
            ck.require(rc == 0 and CHAIN_CLOSE_RECORDED in text.splitlines(), 'UP3 chained HOLD unlink unrecorded; HOLD stays open')
            for prior in list(self.owner_tags):
                self.fence_phase(prior)
            ck.require(self.restore_by - self.clock() >= 90, 'UP3 chained close needs 90s reserve; HOLD stays open')
            try:
                try:
                    rc, text = self.plain('commit-close', [*pf.SSH, 'sparkhead', owner.command('commit-close', tag,
                        deadline=min(self.restore_by - 5, self.clock() + 25))], 30)
                    ck.require(rc == 0 and owner.CLOSED in text.splitlines(), 'UP3 chained owned HOLD close unconfirmed')
                finally:
                    rc, text = self.plain('cancel-close', [*pf.SSH, 'sparkhead', owner.command('cancel-close', tag)], 30)
                    ck.require(rc == 0 and 'SCOPED CLOSE COMMIT FENCED' in text.splitlines(), 'UP3 chained close fence unconfirmed')
            except (Exception, KeyboardInterrupt):
                rc, text = self.plain('retain-close', [*pf.SSH, 'sparkhead', owner.command('retain-close', tag)], 30)
                ck.require(rc == 0 and owner.RETAINED in text.splitlines(), 'UP3 original HOLD retention unconfirmed')
                raise
            chain.finish_chained(self.out, {'runner': 'up3', 'group': self.group, 'sparks': quiet, 'placeholders': kept},
                                 production_snapshot=self.out / 'rollback.json')
            return {'status': 'CHAINED: production left down (NDM 2026-10-27 chain); experiment stopped, Sparks quiet, '
                              'own HOLD closed, hold placeholders kept up', 'sparks': quiet, 'placeholders': kept,
                    'production_snapshot': chain.snapshot_ref(self.out / 'rollback.json'),
                    'closed_before_restore_by': self.clock() < self.restore_by}
        finally:
            (self.out / 'arm-memory.json').write_text(json.dumps(self.sup.arm_memory) + '\n')

def strict_restored(step: str, rc: int, text: str) -> None:
    pf.restored(step, rc, text)
    if step == 'check':
        lines = [line for line in text.splitlines() if line.startswith('greedy K1-post vs restored:')]
        matches = [re.fullmatch(r'greedy K1-post vs restored: ([0-5]) IDENTICAL [0-9]+', line) for line in lines]
        ck.require(len(matches) == 6 and all(match is not None for match in matches)
            and sorted(match.group(1) for match in matches if match is not None) == list('012345'),
            'restore requires six distinct complete IDENTICAL lines0..5')


def host_check(text: str) -> None:
    ck.require(text.splitlines()[:2] == [KERNEL, DRIVER], 'Spark kernel/driver differs')
    memory = re.findall(r'^MemAvailable:\s+(\d+) kB$', text, re.M)
    ck.require(len(memory) == 1 and int(memory[0]) >= pf.START_FLOOR_KB, 'MemAvailable below 5GiB before start')


class Backend(Protocol):
    def preflight(self) -> None: ...
    def open_hold(self) -> float: ...
    def snapshot(self) -> None: ...
    def fits(self, arm: Arm) -> bool: ...
    def start(self, arm: Arm) -> None: ...
    def greedy(self, arm: Arm) -> dict[str, Json]: ...
    def warmup(self, arm: Arm) -> None: ...
    def qualify(self, arm: Arm) -> dict[str, Json]: ...
    def build_cuts(self) -> None: ...
    def cell(self, arm: Arm, name: str) -> dict[str, Json]: ...
    def finish(self, arm: Arm) -> dict[str, Json]: ...
    def restore(self, *, production_stopped: bool) -> dict[str, Json]: ...
    def chained_end(self) -> dict[str, Json]: ...
    def chain_abort(self) -> dict[str, Json]: ...
    def release_placeholders(self) -> dict[str, Json]: ...
    def archive_controls(self, starts: dict[str, Json]) -> dict[str, Json]: ...
    def upf_pre(self) -> dict[str, Json]: ...


def window(backend: Backend, group: str, checkpoint: Callable[[dict[str, Json]], None]) -> dict[str, Json]:
    result: dict[str, Json] = {'status': 'INCOMPLETE', 'starts': {}, 'decision': {'status': 'NOT A CANDIDATE'}}
    hold = stopped = False
    try:
        admit(group)
        backend.preflight()
        hold = True
        result['opened'] = backend.open_hold()
        backend.snapshot()
        if group in upf_plan.GROUPS and upf_plan.pre(group):
            stopped = True   # R-UPF T0 runs ./stop.sh: production is down from here
            result['pre'] = backend.upf_pre()
        skipped = record(record(result.get('pre', {})).get('skip', {}))
        for i, arm in enumerate(arms(group)):
            if arm.name in skipped:   # R-UPF: T1/T2/verify-topk failed; only this arm is skipped
                record(result['starts'])[arm.name] = {'status': 'SKIPPED', 'role': arm.role, 'reason': skipped[arm.name]}
                checkpoint({'arm': arm.name, 'status': 'SKIPPED'})
                continue
            ck.require(backend.fits(arm), 'mandatory remaining OFF/arm/OFF blocks cannot fit; restore now')
            stopped = True
            row: dict[str, Json] = {'status': 'STARTED', 'role': arm.role, 'cells': {}}
            record(result['starts'])[arm.name] = row
            checkpoint({'arm': arm.name, 'status': 'STARTED'})
            try:
                backend.start(arm)
                row['greedy'] = backend.greedy(arm)
                backend.warmup(arm)
                row['gates'] = backend.qualify(arm)
                checkpoint({'arm': arm.name, 'gates': 'PASS'})
                if i == 0:
                    backend.build_cuts()
                for name in cells_for(group):
                    record(row['cells'])[name] = backend.cell(arm, name)
                    checkpoint({'arm': arm.name, 'cell': name})
                row['checks'] = backend.finish(arm)
            except ArmStop as exc:   # R-UPF candidate gate: record and go on to the next start (WINDOW.md)
                if group not in upf_plan.GROUPS or is_control(arm.role):
                    raise
                row.update(status='STOPPED', stop=str(exc), failed_gate=exc.gate, gate_evidence=exc.evidence)
                checkpoint({'arm': arm.name, 'status': 'STOPPED', 'gate': exc.gate})
                continue
            row['status'] = 'COMPLETE'
            checkpoint({'arm': arm.name, 'status': 'COMPLETE'})
        result['decision'] = decision(group, record(result['starts']), record(result.get('pre', {})))
        result['status'] = 'COMPLETE'
    except (Exception, KeyboardInterrupt) as exc:
        result['stop'] = f'{type(exc).__name__}: {exc}'
    finally:
        saved = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            # WIN2-R2 (S92 FIX 1): the end follows chain.end_action, never "did I write a HOLD" alone.
            action = chain.end_action('stop' in result, hold)
            result['end_action'] = action
            if action == 'chained':
                try:
                    result['restore'] = backend.chained_end()
                    result['chain'] = 'CHAINED'  # WIN2.md: the coordinator's chain_prev_ok.sh reads this field
                except (Exception, KeyboardInterrupt) as exc:
                    # A failed chained end falls back to the strict W8 restore while the HOLD is still ours.
                    result['chained_end_error'] = f'{type(exc).__name__}: {exc}'
                    action = 'restore'
            if action == 'restore':
                try:
                    result['restore'] = backend.restore(production_stopped=stopped)
                except (Exception, KeyboardInterrupt) as exc:
                    result['restore'] = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
            elif action == 'abort':
                # No HOLD of ours, but the chain left production down: the coordinator chain abort (strict W8 + watch).
                try:
                    result['restore'] = backend.chain_abort()
                except (Exception, KeyboardInterrupt) as exc:
                    result['restore'] = {'status': 'FAILED', 'error': f'chain abort: {type(exc).__name__}: {exc}'}
            elif action == 'none':
                result['restore'] = {'status': 'NOT NEEDED: preflight only'}
            restored = str(record(result['restore']).get('status', '')).startswith('RESTORED')
            if restored and (chain.enabled() or chain.marker_present()):
                # Inside the scoped W8 close, window.sh close's kill-session hits a private socket: release explicitly.
                try:
                    result['placeholders'] = backend.release_placeholders()
                except (Exception, KeyboardInterrupt) as exc:
                    result['placeholders'] = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
    try:
        result['archive'] = backend.archive_controls(record(result['starts']))
    except (Exception, KeyboardInterrupt) as exc:
        result['archive'] = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
    if 'stop' in result or str(record(result['restore']).get('status', '')).startswith('FAILED'):
        result['decision'] = {'status': 'NOT A CANDIDATE', 'reason': 'STOP or production restore unverified'}
    return result


def budget_minutes(group: str = 'split') -> float:
    if group in upf_plan.GROUPS:
        return upf_plan.budget_s(group) / 60
    return (SETUP_S + sum(block_s(group, a, i == 0) for i, a in enumerate(arms(group))) + RECOVERY_S) / 60


def win2_plan_text(group: str) -> str:
    plan = win2_plan.plan_for(group)
    blocks = [block_s(group, a, i == 0) for i, a in enumerate(arms(group))]
    decide = ', '.join(f'{n} x{plan.repeats[n]}' for n in plan.decision_cells)
    report = ', '.join(f'{n} x{plan.repeats[n]}' for n in plan.cells if n not in plan.decision_cells) or 'none'
    limits = ', '.join(f'{a.role} {win2_plan.NEAR_TIE_NATS[a.role]:.3f}' for a in arms(group)[1:-1])
    return ('DRY RUN: no network, file or command. Coordinator Deepseek/coordinator only.\n'
        f'UP3 WIN2 {group} on CURRENT production {b.PRODUCTION[:7]} / {b.TAG}: ' + '/'.join(a.role for a in arms(group)) + '.\n'
        'Arms (config-only, b.ENV): dual-ring EXL3_RDMA_DEVICES=rocep1s0f1,roceP2p1s0f1 (host RDMA evidence); fixed2 DEP1 '
        'VLLM_DSPARK_DEPTH_POLICY=fixed:2 VLLM_DSV41_ADAPTIVE_K=0 + [[1,4,5]]; defer 0051 VLLM_DSV41_INDEXER_DEFER=1; '
        'ring-defer both; k5 [[1,4,5]] (k 4->5 at B3/B4 only).\n'
        f'Greedy six checks per start; near-tie limit nats: control 0.901, {limits} (NDM 2026-10-27: 0.9; 1.1 DEP1 fixed2).\n'
        f'Cells per start, 512 tokens/request: DECIDE {decide} (win2-v1); reported {report}.\n'
        'Each candidate repeat paired with the mean of the two bracketing control repeats. Round-speed arms: pooled ms/round '
        'decides, accepted/round guards; fixed2/k5: pooled committed tokens per decode second (R+A)/D decides.\n'
        f'MEASURED-basis blocks {"/".join(str(x) for x in blocks)}s ({win2_plan.MEASURED_SOURCE}); '
        f'ESTIMATED HOLD {budget_minutes(group):.2f}min incl. 120s setup + 720s recovery; HOLD bounds submit/restore '
        f'{SUBMIT_S // 60}/{RESTORE_S // 60}min, chained {CHAINED_SUBMIT_S // 60}/{CHAINED_RESTORE_S // 60}min.\n'
        'Control repeats go to the WIN2 archive (max_tokens 512). WINDOW_CHAIN=1: a clean non-last window leaves production '
        'DOWN + CHAINED marker; any stop or WINDOW_CHAIN_LAST=1 runs the strict W8 restore + 15min watch.\n'
        'No production change or promotion is authorized by this window.')


def upx_plan_text(group: str) -> str:
    spec = upx_plan.group_for(group)
    blocks = [block_s(group, a, i == 0) for i, a in enumerate(arms(group))]
    starts = '; '.join(f'{a.name}: gates {",".join(upx_plan.gates(group, i)) or "none"}, cells '
                       + ', '.join(f'{c} x{upx_plan.repeats(group, c, i)}' for c in spec.cells) for i, a in enumerate(arms(group)))
    return ('DRY RUN: no network, file or command. Coordinator Deepseek/coordinator only.\n'
        f'R-UPX {spec.design} in the UP3 runner on CURRENT production {b.PRODUCTION[:7]}: ' + '/'.join(a.role for a in arms(group))
        + f' (offload/upx/WINDOWS.md; registration {upx_plan.REGISTRATION}).\n'
        f'Every start: six W5/K1 greedy at 0.9+1e-3 nats with recorded distributions; OPS1 warm-up. {starts}.\n'
        f'Decision cells {", ".join(spec.spec.decision_cells)} (win2-v1, C1+C2 pooled vs X, unpaired); '
        'windows.arm_verdict: CANDIDATE FOR PROMOTION / NO GAIN SHOWN (stays off) / NOT A CANDIDATE.\n'
        f'Blocks {"/".join(str(x) for x in blocks)}s (MEASURED-basis prepare {win2_plan.PREPARE_S:.0f}s + design ESTIMATES); '
        f'ESTIMATED HOLD {budget_minutes(group):.2f}min incl. 120s setup + 720s recovery; HOLD bounds submit/restore '
        f'{SUBMIT_S // 60}/{RESTORE_S // 60}min, chained {CHAINED_SUBMIT_S // 60}/{CHAINED_RESTORE_S // 60}min.\n'
        'Any failed gate STOPs and runs the strict W8 restore. No production change or promotion is authorized.')


def upf_plan_text(group: str) -> str:
    starts = upf_plan.starts(group)
    blocks = [upf_plan.block_s(group, i) for i in range(len(starts))]
    submit, _, restore = hold_bounds(group)
    lines = '; '.join(f'{s.label} {s.arm}: ' + ', '.join(
        f'{st.cell} x{st.repeats}' if isinstance(st, upf_windows.CellStep) else st.kind for st in s.steps) for s in starts)
    try:
        upf_plan.admit(group)
        admitted = f'HOLD bounds submit/abort HOLD+{submit // 60}, restore HOLD+{restore // 60} (<= +{upf_plan.CEILING_S // 60}).'
    except ValueError as exc:
        admitted = f'REFUSED: {exc}.'
    return ('DRY RUN: no network, file or command. Coordinator Deepseek/coordinator only.\n'
        f'R-UPF {upf_plan.GROUPS[group]} in the UP3 runner on CURRENT production {b.PRODUCTION[:7]}, a SINGLE window '
        f'(offload/upfix WINDOW.md; arms = {upf_configs.CONFIG} + configs.ARMS environment; registration '
        f'{upf_plan.REGISTRATION}).\n'
        + (f'Pre-steps in the HOLD: {", ".join(upf_plan.pre(group))} (T0 ./stop.sh + GPU idle; T1 SM121 jit_build in the '
           f'image; T2 bit-compare under timeout -k 30 {upf_windows.T2_TIMEOUT_S}, container upfix-t2; any T1/T2/verify-topk '
           'failure skips X-topk only).\n' if upf_plan.pre(group) else '')
        + f'Starts: {lines}.\n'
        'Identity relative to C1 (C1 vs production recorded; C1 greedy 6/6 IDENTICAL or STOP); OPS1 warm-up stands in for '
        'the 64-token warm-up cell; greedy 0.9+1e-3 nats with recorded distributions; C2 KL re-probe at 0.005.\n'
        f'Decision cells {", ".join(upf_plan.CELLS)} at {upf_windows.MAX_TOKENS} tokens, C1+C2 pooled vs X (unpaired, '
        f'win2-v1 revision {upf_windows.RULE_REVISION}); windows.arm_verdict: PROMOTABLE (no loss) / NOT A CANDIDATE / STOP.\n'
        f'Blocks {"/".join(str(x) for x in blocks)}s; ESTIMATED HOLD {budget_minutes(group):.2f}min incl. '
        f'{upf_windows.FENCES_S}s fences, {upf_plan.pre_s(group)}s pre-steps and {RECOVERY_S}s recovery; {admitted}\n'
        'Arm-level stops are not window stops; a control failure stops the window and runs the strict W8 restore.\n'
        'No production change or promotion is authorized.')


def plan(group: str = 'split') -> str:
    if group in upf_plan.GROUPS:
        return upf_plan_text(group)
    if group in upx_plan.GROUPS:
        return upx_plan_text(group)
    if group in win2_plan.GROUPS:
        return win2_plan_text(group)
    return (f'DRY RUN: no network, file or command. Coordinator Deepseek/coordinator only.\n'
        'OPS1 596ba1c / prod-k1-327-ops1-20261026; K1 327680, KV2502600, TP2 K2 fb68c52 unchanged.\n'
        f'One isolated group/HOLD: OFF/{group}/OFF; all three mandatory. Config-only host gates; split/vor parked BEFORE HOLD.\n'
        'Public short B1;32K B1/B3/B4;150K+2 active decoders.512 outputs/request;4/4/6/4/6 requests/cell/start.\n'
        'T1/top_p1/seeds42+repeat/high/priority10; matched public cuts and fresh cold salts. TTFT/client rate/mixed p99 recorded.\n'
        'Decide pooled (R+A)/D; paired whole-group95% lower bound and point >max(2%,C spread). No cell loss beyond spread.\n'
        'B4 one simultaneous group is no-regression only; TTFT/mixed p99 no-regression. No restart or task-family uncertainty claim.\n'
        'Each start: six ordered W5/K1 experimental greedy checks; top20/<=0.501nat first divergence or STOP.\n'
        'Then OPS1 warmup; measurement; strict W8 identities remain separate.\n'
        'Shared scopes/HOLD admission/quiet/health/watch; every receiving-host mutation, nested SSH/SCP and W8 owned/fenced.\n'
        f'MEASURED startup388--404s; ESTIMATED block C={BLOCK_S}s,A={BLOCK_S+QUALIFY_S}s including30s host-evidence and60s/start greedy reserves.\n'
        f'ESTIMATED HOLD {budget_minutes(group):.2f}min <=85, including120s setup+720s recovery; over-budget plans/remaining starts refuse.\n'
        'HOLD publication+73 submit/abort,+85 recovery deadline; never renew. Always restore on STOP/signals/lost replies.\n'
        'Strict six distinct IDENTICAL0..5; all-host/container/JIT absence; deferred owned close;15min shared watch outside HOLD.\n'
        'WIN2 chain (WINDOW_CHAIN=1): a clean non-last window stops the experiment, proves Sparks quiet, closes its own HOLD,\n'
        'leaves production DOWN + CHAINED marker; any stop or WINDOW_CHAIN_LAST=1 runs strict W8 restore + watch.\n'
        'Config-only k5/dual-ring/cpu-pin: no worker adapter. Split/vor parked until reviewed resident integration.\n'
        'Source/CPU preparation only; no production execution or promotion authorized by this dry plan.')


def plain_owned(out: Path) -> Callable[[str, list[str], float], tuple[int, str]]:
    # Restore fences every owned phase again after close; a serial keeps each
    # attempt's exclusive log distinct (TV1-COMPACT-W-20261027T031304Z kept HOLD
    # on a FileExistsError for the second hold-phase fence log).
    serial = itertools.count(1)
    def run(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
        log = out / 'cmd' / f'restore-{next(serial):04d}-{label}.log'
        log.parent.mkdir(exist_ok=True)
        with log.open('x') as handle:
            process = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, text=True)
            try:
                rc = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                terminate(process)
                rc = 124
        return rc, log.read_text()
    return run


def reanalyze_arms(out: Path, group: str) -> dict[str, Json]:
    """WIN2-R2 audit: win2_history.reanalyze over this window's own files, each candidate role its own arm ('arms' for
    dual ring + DEP1 and for 0051 alone / with the ring). Its primary is ms/round for every arm, so for fixed2 / k5 it is
    reported beside the registered (R+A)/D decision, never instead of it."""
    try:
        return win2_history.reanalyze(out, win2_plan.plan_for(group).decision_cells)
    except (OSError, ValueError, KeyError) as exc:
        return {'status': 'UNAVAILABLE', 'error': f'{type(exc).__name__}: {exc}'}


def upf_archive(out: Path, group: str, result: dict[str, Json]) -> dict[str, Json]:
    """R-UPF control archive (WIN2-R2 v3): windows.archive_rows for every COMPLETE control start, also after a stop.
    Runs once summary.json exists, because win2_history.production_commit reads the production commit from it."""
    try:
        starts = record(result['starts'])
        rows: list[dict[str, Json]] = []
        for start in upf_plan.starts(group):
            row = record(starts.get(start.label, {}))
            if upf_plan.is_control(start.arm) and row.get('status') == 'COMPLETE':
                rows += upf_windows.archive_rows(out, start.label, upf_plan.control_values(row))
        before = dv2.archive_head(archive_path())
        if rows:
            dv2.archive_append(archive_path(), rows)
        return {'rows': len(rows), 'head_before': before, 'head_after': dv2.archive_head(archive_path())}
    except (OSError, ValueError, KeyError) as exc:
        return {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}


def end_chain(result: dict[str, Json]) -> None:
    """WIN2.md: the chain marker is cleared only after a strict W8 restore verified AND the post-close watch passed; a
    failed or missing watch leaves it in place for the coordinator (chain.abort clears its own)."""
    if (str(record(result['restore']).get('status', '')).startswith('RESTORED')
            and record(result.get('post_close_watch', {})).get('status') == 'PASS'):
        chain.clear()


def post_close(result: dict[str, Json], out: Path, tag: str) -> None:
    """Watch even a stopped experiment after verified restoration; never an open HOLD."""
    if not str(record(result['restore']).get('status', '')).startswith('RESTORED'):
        return
    aborted = record(record(result['restore']).get('chain_abort', {}))
    if 'post_close_watch' in aborted:   # chain.abort already ran the 15-minute watch
        result['post_close_watch'] = aborted['post_close_watch']
        return
    def log(message: str) -> None:
        print(message, flush=True)
        with (out / 'post-close-watch.log').open('a') as handle:
            handle.write(message + '\n')
    try:
        watched = watchdog.watch(production_health.ProductionHealth(),
            watchdog.ScopedRecovery(tag + '-watch', out / 'post-close-commands'), log=log)
        result['post_close_watch'] = cast(Json, dict(watched.__dict__))
    except (Exception, KeyboardInterrupt) as exc:
        print(f'UP3 post-close watch unverified: {exc}', file=sys.stderr)
        result['post_close_watch'] = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
    if record(result['post_close_watch']).get('status') != 'PASS':
        result['decision'] = {'status': 'NOT A CANDIDATE', 'reason': 'post-close production health failed'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--group', choices=GROUPS, default='split')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--remote-repo', default='dsv41-deploy-offload')
    parser.add_argument('--expected-experiment-commit', default='')
    parser.add_argument('--data-dir', type=Path, default=Path('/opt/dsv41/data/bench-data'))
    parser.add_argument('--production-identity', type=Path, default=None,
        help='WIN2 chain only: UP3 rollback.json to compare against when the marker carries no UP3 production_identity')
    args = parser.parse_args(argv)
    if not args.execute:
        print(plan(args.group))
        return 0
    try:
        admit(args.group)
        limit = hold_bounds(args.group)[2] / 60
        ck.require(budget_minutes(args.group) <= limit,
            f'UP3 matrix needs ESTIMATED {budget_minutes(args.group):.2f}min >{limit:.0f}; redesign required before any host access')
        ck.require(os.environ.get('WINDOW_COORDINATOR') == 'Deepseek/coordinator', 'coordinator required')
        ck.require(args.out is not None and args.out.resolve().parent == pf.CAMPAIGN and not args.out.exists(), 'fresh campaign --out required')
        ck.require(re.fullmatch('[0-9a-f]{40}', args.expected_experiment_commit) is not None, 'full reviewed commit required')
        ck.require(re.fullmatch('[A-Za-z0-9_-]+', args.remote_repo) is not None and args.remote_repo != 'dsv41-deploy', 'separate experiment checkout required')
        ck.require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=b.ROOT, text=True).strip() == args.expected_experiment_commit,
            'local HEAD differs from reviewed commit')
        ck.require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=b.ROOT, text=True).strip(), 'tracked tree dirty')
        tag = cast(Path, args.out).name
        ck.require(re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_-]{0,49}', tag) is not None, 'run name must be <=50 safe characters')
        owner.root_for(tag + '-restore-experiment-stop-999999')
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP before remote action: {exc}', file=sys.stderr)
        return 1
    out = cast(Path, args.out)
    out.mkdir()
    def interrupted(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f'signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    def checkpoint(row: dict[str, Json]) -> None:
        with (out / 'checkpoints.jsonl').open('a') as handle:
            handle.write(json.dumps({'time': time.time(), **row}, allow_nan=False) + '\n')
    sup = pf.Supervisor(out)
    live = Live(out, sup, group=args.group, remote=args.remote_repo, expected=args.expected_experiment_commit,
        data_dir=args.data_dir, tag=tag, plain=plain_owned(out), production_identity=args.production_identity)
    result = window(live, args.group, checkpoint)
    sup.close()
    post_close(result, out, tag)
    end_chain(result)
    result.update({'production': b.PRODUCTION, 'experiment': args.expected_experiment_commit, 'group': args.group,
        'shared_owner_sha256': pf.digest(b.ROOT / 'offload/launch/windows/scoped_remote.py'),
        'budget_minutes_estimated': budget_minutes(args.group), 'budget_sha256': pf.digest(b.HERE / 'budget.json'),
        'restore_sha256': pf.RESTORE_SHA256, 'qualification_policy': 'Round4 config-only',
        'chain_context': {'enabled': chain.enabled(), 'last': chain.last(), 'mode': live.chain_mode},
        'artifact_sha256': {str(p.relative_to(out)): pf.digest(p) for p in out.rglob('*.json*') if p.is_file()}})
    (out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    if args.group in win2_plan.GROUPS:
        result['reanalyze'] = reanalyze_arms(out, args.group)
        (out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    if args.group in upf_plan.GROUPS:
        result['archive'] = upf_archive(out, args.group, result)
        (out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'stop': result.get('stop'), 'decision': result['decision'], 'restore': result['restore']}))
    chained = str(record(result['restore']).get('status', '')).startswith('CHAINED')
    return 2 if str(record(result['restore']).get('status', '')).startswith('FAILED') else (
        0 if result['status'] == 'COMPLETE' and (chained or record(result.get('post_close_watch', {})).get('status') == 'PASS') else 1)


if __name__ == '__main__':
    raise SystemExit(main())
