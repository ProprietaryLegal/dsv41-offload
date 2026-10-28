"""UP3 current-production OFF/arm/OFF window. Dry by default; coordinator only, one arm per HOLD."""
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
from offload.launch.windows import nbw_cells as greedy_driver
from offload.launch.windows import up2_checks as ck
from offload.launch.windows import chain, hold_admission, spark_quiet, production_health, win2_history
from offload.launch.windows import post_close_watch as watchdog
from offload.launch.windows import scoped_remote as owner
from offload.launch.windows.up2_low import X86HOST_PROBE, terminate
from offload.prefill_maxlen.evidence import Json, number, record
from offload.up3 import build as b, cells, gates, metrics, host_gates

CONFIG = 'offload/up3/window.env'
SUBMIT_S, ABORT_S, RESTORE_S = 73 * 60, 73 * 60, 85 * 60
# ESTIMATED from measured startup + NBW-W3 cold cells; budget.json records
# hashes, arithmetic and unmeasured mixed/B4/host-evidence extrapolations explicitly.
STARTUP_S, WARMUP_S, TRANSITION_S, QUALIFY_S = 404, 90, 60, 30
GREEDY_S = 60  # ESTIMATED six W5 checks plus first-divergence diagnostics, not a GB10 measurement
SETUP_S, RECOVERY_S = 120, 720
PREPARE_S = STARTUP_S + WARMUP_S + TRANSITION_S + GREEDY_S
BLOCK_S = PREPARE_S + sum(cells.ESTIMATES[n] * cells.REPEATS[n] for n in cells.CELLS)
KERNEL, DRIVER = '7.0.0-1019-nvidia', '580.178.04'
BUDGET = {'control': BLOCK_S, **{arm: BLOCK_S + QUALIFY_S for arm in gates.CASES}}
TARGETS = {'split': ('mixed150k-2decode',), 'cpu-pin': ('32k-b3',),
           'vor': ('32k-b1',), 'k5': ('32k-b3',), 'dual-ring': ('32k-b3',)}
# UP2-ADOPT (NDM 2026-10-27: "Look at past results and if there are optimizations that can now be approved based on .9
# nats, and will improve speed, implement those too" / "do this yes"): 0075 replay skip on CURRENT production. Its gain is
# the MEASURED 2026-10-25 300K cold request (ON 130.3-130.9 s vs OFF 139.6-141.8 s); this window qualifies correctness and
# no-regression only. API-level gates (no in-engine receipt): six greedy at 0.9+1e-3 nats on every start; first-token
# top-20 KL at 32K and 150K against a same-start A/A floor (UP2's registered numerical floor); 150K needle 6/6; UP2's
# replay-provenance counters and log strings (ck.STOP_COUNTERS / ck.STOP_LOG) on every start.
REPLAY = 'replay'
GROUPS = (*gates.CASES, REPLAY)
REPLAY_REGISTRATION = cells.REPLAY_REGISTRATION
REPLAY_NEAR_TIE_NATS = cells.REPLAY_NEAR_TIE_NATS
# WIN2 (NDM 2026-10-27 "Fairer win rule is essential"; Fable review row 4): decision cells 150K cold TTFT x3 and 32K TTFT
# (no regression); short B1 decode is reported only. The mixed cell is not part of this window.
REPLAY_CELLS = ('150k-b1', '32k-b1', 'short-b1')
REPLAY_DECISION_CELLS = ('150k-b1', '32k-b1')
# Per cell: (metric, higher_is_better). In a decision cell the FIRST metric decides and the rest are guards (LOSS only);
# pooled server ms/round guards decode in both decision cells, since short B1 cannot be a third decision cell.
REPLAY_METRICS: dict[str, tuple[tuple[str, bool], ...]] = {
    '150k-b1': (('ttft_s', False), ('ms_per_round', False)),
    '32k-b1': (('ttft_s', False), ('ms_per_round', False)),
    'short-b1': (('ms_per_round', False), ('accepted_per_round', True), ('decode_tps', True))}
# ESTIMATED from MEASURED inputs: 150K cold TTFT 68.6-69.4 s, prompt construction 19-33 s, decode >=52 tok/s (S30);
# 32K cold TTFT 14.4-14.9 s (UP3-dual-ring-W-20261027T165101Z, all three starts).
NUMERIC_TARGETS = (32000, 150000)   # UP2's registered 32K/150K numeric contexts (up2_low.py)
NUMERIC_S = {32000: 40,    # 14.9 + construction (<=33 s at 150K, taken whole) - rounded up; 64 tokens/52 tok/s
             150000: 105}  # 69.4 + 33 + 64 tokens/52 tok/s, rounded up
NEEDLE_S = 120    # 69.4 + needle prompt construction (<=33 s, estimated as numeric) + 300 tokens/52 tok/s, rounded up
AA_S = 2 * sum(NUMERIC_S.values()) + 30        # first control start: same-start A/A numerical floor
REPLAY_GATES_S = sum(NUMERIC_S.values()) + NEEDLE_S + 30
# WIN2 opt-in greedy refinement on the last control: at most six short re-probes of the replay arm's divergences.
GREEDY_KL_S = 60
REPLAY_CELL_S = sum(cells.ESTIMATES[n] * cells.REPEATS[n] for n in REPLAY_CELLS)


def cells_for(group: str) -> tuple[str, ...]:
    return REPLAY_CELLS if group == REPLAY else cells.CELLS


def admit(group: str) -> None:
    """Replay is a config arm qualified by API-level gates; every other group keeps UP3's host-gate admission."""
    if group != REPLAY:
        host_gates.admit(group)


def allowance(group: str, arm: 'Arm', first: bool) -> int:
    """Pre-measurement allowance from start: startup/warmup/greedy plus that start's qualification evidence."""
    if group == REPLAY:
        return PREPARE_S + (REPLAY_GATES_S if arm.role == REPLAY else AA_S if first else GREEDY_KL_S)
    return PREPARE_S + (QUALIFY_S if arm.role != 'control' else 0)


def block_s(group: str, arm: 'Arm', first: bool) -> int:
    if group == REPLAY:
        return allowance(group, arm, first) + REPLAY_CELL_S
    return BUDGET[arm.role]
# WIN2 chain (NDM 2026-10-27 "go window after window"): a chained window cannot capture production (it is down), so it
# reuses the chain's first UP3 capture, sha-pinned through the marker. UP3 compares the FULL container environment.
IDENTITY_KEYS = ('id', 'image', 'engine', 'files', 'mounted', 'env', 'argv', 'engram_identity')


@dataclass(frozen=True)
class Arm(pf.Arm):
    role: str = 'control'

    @property
    def config(self) -> str:
        return CONFIG


def arms(group: str) -> tuple[Arm, ...]:
    ck.require(group in GROUPS, 'one isolated group required')
    return tuple(Arm(f's{i}-{role}', 327680, 0, role=role)
                 for i, role in enumerate(('control', group, 'control'), 1))


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
    if arm.role == 'k5':
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


def tail_gates(name: str, measured: list[dict[str, Json]]) -> tuple[list[Json], bool]:
    """Per-stream C/A/C tail gates: TTFT for every cell, plus delivery-gap p99 for the mixed cell."""
    tails: list[Json] = []
    passes = True
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
    return tails, passes


def bracket(group: str, starts: dict[str, Json]) -> tuple[Arm, ...] | None:
    expected = arms(group)
    if list(starts) != [a.name for a in expected] or any(record(v).get('status') != 'COMPLETE' for v in starts.values()):
        return None
    ck.require(all(record(starts[a.name]).get('role') == a.role for a in expected), 'start role mismatch')
    for arm in expected:
        greedy_checks(record(record(starts[arm.name]).get('greedy', {})))
    return expected


def archive_path() -> Path:
    """WIN2 control archive (append-only, sha-chained): every replay-window control cell, per repeat and metric."""
    return pf.CAMPAIGN / 'control-archive.jsonl'


def repeat_values(cell: dict[str, Json], metric: str) -> list[float]:
    """One value per repeat of a B1 cell, from fields cells.run already records (stream TTFT/decode rate; Prometheus
    request decode time over speculative rounds; accepted tokens over rounds)."""
    rows = ck.rows(cell['repetitions'])
    ck.require(all(len(ck.rows(r['streams'])) == 1 for r in rows), 'replay decision cells are single-request (B1)')
    if metric in ('ttft_s', 'decode_tps'):
        return [number(ck.rows(r['streams'])[0][metric]) for r in rows]
    rounds = [number(r['spec_request_rounds']) for r in rows]
    ck.require(all(math.isfinite(n) and n > 0 for n in rounds), 'no speculative rounds in a measured repeat')
    if metric == 'ms_per_round':
        return [number(r['server_decode_ms']) / n for r, n in zip(rows, rounds)]
    ck.require(metric == 'accepted_per_round', f'unknown replay metric {metric}')
    return [number(r['accepted_tokens']) / n for r, n in zip(rows, rounds)]


def replay_series(starts: dict[str, Json], expected: tuple[Arm, ...]) -> list[dv2.Series]:
    """Paired C/X/C: repeat r of the arm against the mean of repeat r of the two bracketing controls."""
    series: list[dv2.Series] = []
    for name in REPLAY_CELLS:
        measured = [record(record(record(starts[a.name])['cells'])[name]) for a in expected]
        prompts = [[(s['prompt_sha256'], s['prompt_tokens'], s['budget'])
                    for r in ck.rows(c['repetitions']) for s in ck.rows(r['streams'])] for c in measured]
        ck.require(all(h == prompts[0] for h in prompts) and len(prompts[0]) == cells.REPEATS[name],
                   f'{name}: public prompt or repeat count differs between starts')
        for metric, higher in REPLAY_METRICS[name]:
            before, arm, after = (repeat_values(c, metric) for c in measured)
            series.append(dv2.Series(name, metric, higher, tuple((x + y) / 2 for x, y in zip(before, after)), tuple(arm)))
    return series


def replay_decision(starts: dict[str, Json]) -> dict[str, Json]:
    """The summary decision IS decide_window's win2-v1 dict (WIN2 checklist 3), plus `adopt`: ADOPT CANDIDATE iff every
    correctness gate passed and no decision cell LOSES. A speed win is NOT required: the gain is the MEASURED 2026-10-25
    300K cold-request evidence, cited, not re-measured; a 150K or 32K TTFT WIN (status CANDIDATE) is additional evidence."""
    refused = {'status': 'NOT A CANDIDATE', 'adopt': 'NOT A CANDIDATE', 'registration': REPLAY_REGISTRATION}
    expected = bracket(REPLAY, starts)
    if expected is None:
        return {**refused, 'reason': 'all mandatory complete bracket starts required'}
    for arm in expected:
        greedy = record(record(starts[arm.name])['greedy'])
        ck.require(greedy.get('greedy_registration') == REPLAY_REGISTRATION
            and greedy.get('near_tie_nats') == REPLAY_NEAR_TIE_NATS, 'replay greedy registration/threshold missing')
    gate = record(record(starts[expected[1].name]).get('gates', {}))
    if gate.get('status') != 'PASS' or gate.get('registration') != REPLAY_REGISTRATION:
        return {**refused, 'reason': 'replay numerical/needle gates did not pass'}
    kl = record(record(record(starts[expected[2].name]).get('gates', {})).get('greedy_kl', {}))
    if kl.get('status') != 'PASS' or kl.get('registration') != REPLAY_REGISTRATION:
        return {**refused, 'reason': 'greedy divergence KL re-probe on the following control did not pass'}
    v2 = dv2.decide_window(replay_series(starts, expected), REPLAY_DECISION_CELLS)
    # Day-to-day control spread beside the in-window spread (reported; the required effect stays 2 %). Archive v3: one
    # definition = max_tokens + production commit + the clock cap this window's control starts proved.
    caps = {record(record(record(starts[a.name]).get('checks', {})).get('archive', {})).get('clock_cap_mhz')
            for a in expected if a.role == 'control'}
    cap = caps.pop() if len(caps) == 1 else None
    where: dict[str, Json] = {'max_tokens': cells.BUDGET, 'commit': b.PRODUCTION, 'clock_cap_mhz': cap}
    spreads: dict[str, Json] = {}
    for name in REPLAY_DECISION_CELLS:
        for metric, _ in REPLAY_METRICS[name]:
            try:
                ck.require(isinstance(cap, int), 'control starts prove no single clock cap')
                spreads[f'{name}/{metric}'] = dv2.archive_spread(archive_path(), name, metric, where=where)
            except ValueError as exc:
                spreads[f'{name}/{metric}'] = f'UNREADABLE: {exc}'
    # S97 FIX 2: an approved-on-evidence lever needs no win, but a decision cell whose primary (cold TTFT) is
    # significantly WORSE (whole 95 % interval below zero) contradicts that evidence: never an ADOPT CANDIDATE, even when
    # every repeat stays inside the control spread (INCONCLUSIVE, not LOSS).
    slower = [c for c in REPLAY_DECISION_CELLS if cast(list[float], record(record(record(record(v2['cells'])[c])['metrics'])
              [REPLAY_METRICS[c][0][0]])['interval95'])[1] < 0]
    return {**v2, 'adopt': 'NOT A CANDIDATE' if v2['status'] == 'NOT A CANDIDATE' or slower else 'ADOPT CANDIDATE',
            'significantly_slower': cast(Json, slower),
            'archive_spread_max_tokens_512': spreads, 'archive_spread_where': where, 'registration': REPLAY_REGISTRATION,
            # The exact measured bytes: the promotion evidence gate requires its package to equal these.
            'replay_files': cast(Json, dict(b.verify(REPLAY)['files'])),
            'adoption_rule': 'every correctness gate PASS (greedy 0.9+1e-3 with divergence KL, KL 32K/150K, needle 150K, '
                             'provenance counters/logs); win2-v1 on 150K and 32K cold TTFT (ms/round guard): no decision '
                             'cell LOSS and no decision cell significantly slower (cold TTFT 95% interval wholly below '
                             'zero); no win required',
            'gain_evidence': 'MEASURED UP2-screen-300K-20261025T131508Z: 300K cold request ON 130.3-130.9 s vs OFF '
                             '139.6-141.8 s (paired median 0.9317, spread 0.105%) on the UP1 composition',
            'promotion': 'coordinator promotion window only (offload/promote/up2); this window never changes production'}


def decision(group: str, starts: dict[str, Json]) -> dict[str, Json]:
    if group == REPLAY:
        return replay_decision(starts)
    expected = bracket(group, starts)
    if expected is None:
        return {'status': 'NOT A CANDIDATE', 'reason': 'all mandatory complete bracket starts required'}
    points: dict[str, Json] = {}
    targets: list[str] = []
    passes = True
    for name in cells.CELLS:
        measured = [record(record(record(starts[a.name])['cells'])[name]) for a in expected]
        point = metrics.compare(measured, cells.batch_for(name), cells.REPEATS[name])
        point['tail_gates'], okay = tail_gates(name, measured)
        passes &= okay
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
    """The production identity a chained window compares against: the marker's UP3 reference, or an explicit UP3
    rollback.json (coordinator option when the previous window was another runner). Sha-checked; full env required."""
    if explicit is None:
        marker = chain.read_marker()
        ck.require(marker is not None, 'chained window without a chain marker')
        reference = record(record(cast(dict[str, Json], marker).get('quiet_proof', {})).get('production_identity', {}))
        ck.require(set(reference) >= {'path', 'sha256'}, 'chain marker lacks a UP3 production_identity; '
                   'pass --production-identity <UP3 rollback.json> or restore W8')
        path = Path(str(reference['path']))
        ck.require(pf.digest(path) == reference['sha256'], 'chain production identity changed since the marker')
    else:
        path = explicit
    ck.require(path.is_absolute() and path.resolve().parent.parent == pf.CAMPAIGN and path.name == 'rollback.json',
               'chain production identity must be a campaign window rollback.json')
    data = ck.read(path)
    ck.require(data.get('commit') == b.PRODUCTION and data.get('tag') == b.TAG
               and data.get('restore_sha256') == pf.RESTORE_SHA256, 'chain production identity is another production')
    ranks = [record(r) for r in ck.rows(data['ranks'])]
    ck.require(len(ranks) == len(pf.HOSTS) and all(set(r) >= set(IDENTITY_KEYS) for r in ranks),
               'chain production identity lacks ranks/keys')
    # pf1_window.capture keeps only these runtime keys; UP3's capture keeps the full container environment.
    filtered = ('VLLM_', 'EXL3_', 'NCCL_', 'PYTORCH_', 'CUDA_', 'PYTHONPATH', 'DSV41_ENGRAM_OVERLAY')
    ck.require(all(any(not key.startswith(filtered) for key in record(r['env'])) for r in ranks),
               'chain identity env is filtered, not the full container env')
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
        self.numeric_aa: dict[int, list[dict[str, Json]]] = {}
        self.kl_bound = 0.        # set by the replay arm's numeric gates: max(.005, 2 x the window's worst A/A KL)
        self.control_cells: dict[str, dict[str, dict[str, Json]]] = {}
        self.restore_by = self.clock() + RESTORE_S
        self.owner_tags: list[str] = []
        self.owner_deadlines: dict[str, float] = {}
        self.recovery_serial = 0
        self.open_deadline = self.clock() + pf.DRAIN_S + 60
        self.hold_scope_deadline = self.open_deadline + RESTORE_S
        self.open_admitted = False
        self.admission_token = uuid.uuid4().hex
        self.round_baseline: dict[str, dict[str, Json]] = {}
        self.counter_seen: dict[str, dict[str, float]] = {}
        self.started_at = 0.0
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
            'offload/launch/warmup_serve.sh', 'offload/launch/warmup.py',
            # start_exp.sh runs the replay guard on sparkhead for REPLAY=1; it reads UP2's manifest and builder.
            'offload/upstream_adopt/replay_guard.py', 'offload/upstream_adopt/build_overlay.py',
            'offload/upstream_adopt/MANIFEST.json')]
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
        # WIN2-R2: between chained windows the hold placeholders must still be up on both Sparks (chain.gate's rule:
        # probed only while the marker is present; chain.PLACEHOLDER_PROBE, logged here like every other probe).
        placeholders: bool | None = None
        if chain.marker_present():
            answers = [self.ssh('chain-placeholders-' + host, host,
                                f'if {chain.PLACEHOLDER_PROBE}; then echo PRESENT; else echo ABSENT; fi').split()
                       for host in chain.PLACEHOLDER_HOSTS]
            ck.require(all(a in (['PRESENT'], ['ABSENT']) for a in answers), 'chain placeholder probe unreadable')
            placeholders = all(a == ['PRESENT'] for a in answers)
        self.chain_mode = chain.preflight(production_serving=probe[1] == 'SERVING', hold_exists=probe[0] == 'HOLD',
                                          placeholders_present=placeholders)
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
        self.hold_scope_deadline = self.open_deadline + RESTORE_S
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
                self.submit, self.sup.abort_at, self.restore_by = opened + SUBMIT_S, opened + ABORT_S, opened + RESTORE_S
                self.sup.bound = 'UP3 HOLD+73 abort bound'
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
        pins = source_pins(arm)
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
            return
        for host in pf.HOSTS:
            current = self.capture(host, 'prod')
            self.verify_sources(host, 'control')
            ck.require(current.get('image') == pf.IMAGE, 'production image pin differs')
            self.production.append(current)
            self.seen.add(str(current['id']))
        ck.require(self.production[0]['files'] == self.production[1]['files'], 'production target metadata differs by rank')
        self.production_idle()
        (self.out / 'rollback.json').write_text(json.dumps({'commit': b.PRODUCTION, 'tag': b.TAG,
            'restore_sha256': pf.RESTORE_SHA256, 'ranks': cast(Json, self.production)}, indent=1) + '\n')

    def fits(self, arm: Arm) -> bool:
        planned = arms(self.group)
        ck.require(arm in planned, 'start is outside this isolated OFF/arm/OFF plan')
        remaining = sum(block_s(self.group, a, i == 0) for i, a in enumerate(planned) if i >= planned.index(arm))
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
        startup_limit = STARTUP_S + TRANSITION_S + (QUALIFY_S if arm.role != 'control' else 0)
        start = (f'cd ~/{self.remote} && UP3_ARM={arm.role} '
            f'timeout --foreground {startup_limit} bash offload/launch/start_exp.sh {CONFIG}')
        start = hold_admission.command('stop', self.tag + '-hold', self.admission_token, ['bash', '-c', start])
        self.owned(arm.name + '-start', 'sparkhead', self.tag + '-' + arm.name + '-engine', start, timeout=startup_limit)
        self.ssh(arm.name + '-health', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/health')
        model = ck.rows(self.json(arm.name + '-model', 'sparkhead', 'curl -fsS --max-time 10 http://localhost:8001/v1/models')['data'])
        ck.require(len(model) == 1 and model[0].get('max_model_len') == 327680
            and model[0].get('root') == '/srv/dsv41/legal-v2a', 'experiment model/cap differs')
        current = [self.capture(host, arm.name) for host in pf.HOSTS]
        for rank, host in enumerate(pf.HOSTS):
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
        output = self.out / f'{arm.name}-greedy.json'
        threshold: tuple[str, ...] = ('--near-tie-nats', repr(REPLAY_NEAR_TIE_NATS)) if self.group == REPLAY else ()
        if self.group == REPLAY and arm.role == REPLAY:
            # WIN2 opt-in: keep each divergence's top-20 and prefix for the following control's KL re-probe.
            threshold += ('--record-distribution', '--salt', f'{self.tag}-{arm.name}-greedy')
        self.call(f'{arm.name}-greedy', self.driver('greedy', output, '--deadline', str(deadline), *threshold),
            timeout=min(GREEDY_S, deadline-self.clock()), inflight=1)
        result = ck.read(output)
        greedy_checks(result)
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
        # S97 FIX 1: UP2's provenance definition (up2_low.clean; NBW checks the same) is STOP counters at ZERO over the
        # whole replay-group start (engine start, six greedy, OPS1 warm-up), not merely unchanged from this baseline.
        ck.counters(baseline, before=None if self.group == REPLAY else baseline)
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

    def request(self, arm: Arm, label: str, argv: list[str], deadline: float, limit: float) -> dict[str, Json]:
        """One UP2 public-corpus probe through the same 18001 tunnel; separate ledger, hashes/numbers only."""
        output = self.out / f'{arm.name}-{label}.json'
        ck.require(self.clock() < deadline, label + ': replay gate allowance exhausted before submission')
        self.call(f'{arm.name}-{label}', [sys.executable, '-m', 'offload.launch.windows.up2_request', *argv,
            '--output', str(output), '--ledger', str(self.out / 'replay-gates-ledger.jsonl'), '--deadline', str(deadline),
            '--data-dir', str(self.data_dir)], timeout=min(limit, deadline - self.clock()), inflight=1)
        ck.require(self.clock() < deadline, label + ': replay gate exceeded its allowance')
        return ck.read(output)

    def numeric(self, arm: Arm, target: int, rep: int, deadline: float) -> dict[str, Json]:
        # UP2's diagnostic numerical probe: T0, 64 tokens, first-token top-20; distinct cold salt per repetition.
        return self.request(arm, f'numeric{target}-{rep}', ['--target', str(target), '--budget', '64', '--numeric',
            '--salt', f'{self.tag}:num{target}:{rep}', '--rep', str(rep), '--kind', 'numeric'], deadline, 2 * NUMERIC_S[target])

    def greedy_kl(self, arm: Arm, deadline: float) -> dict[str, Json]:
        """Last control: re-probe the replay arm's recorded greedy divergences (none when all six were IDENTICAL)."""
        replay = arms(self.group)[1]
        ck.require(self.kl_bound >= .005, 'replay numeric gates must set the greedy-KL bound first')
        output = self.out / f'{arm.name}-greedy-kl.json'
        self.call(f'{arm.name}-greedy-kl', self.driver('greedy-kl', output, '--deadline', str(deadline), '--candidate',
            str(self.out / f'{replay.name}-greedy.json'), '--kl-bound', repr(self.kl_bound), '--salt',
            f'{self.tag}-{arm.name}-kl'), timeout=min(GREEDY_KL_S, deadline - self.clock()), inflight=1)
        result = ck.read(output)
        ck.require(result.get('status') == 'PASS' and result.get('bound') == self.kl_bound,
                   'replay greedy divergence distribution differs beyond the registered KL bound')
        return {'status': 'unchanged production source/env verified', 'greedy_kl': result}

    def replay_gates(self, arm: Arm, deadline: float) -> dict[str, Json]:
        before = self.stats(arm.name + '-gates-before')
        self.baseline_check(arm, before)
        result: dict[str, Json]
        if arm.role == 'control':
            # Same-start A/A on the FIRST control only: UP2's numerical floor for this window, per context.
            self.numeric_aa = {n: [self.numeric(arm, n, rep, deadline) for rep in (0, 1)] for n in NUMERIC_TARGETS}
            result = {'status': 'unchanged production source/env verified', 'numeric_aa': {
                str(n): [{k: r.get(k) for k in ('prompt_sha256', 'text_sha256', 'finish')} for r in rows]
                for n, rows in self.numeric_aa.items()}}
        else:
            ck.require(sorted(self.numeric_aa) == list(NUMERIC_TARGETS)
                       and all(len(rows) == 2 for rows in self.numeric_aa.values()),
                       'same-window A/A numerical floor required before the replay arm')
            numerical: dict[str, Json] = {}
            text: dict[str, Json] = {}
            for n in NUMERIC_TARGETS:
                on = self.numeric(arm, n, 0, deadline)
                a, aa = self.numeric_aa[n]
                ck.require(a['prompt_sha256'] == aa['prompt_sha256'] == on['prompt_sha256'], 'numeric prompt drift')
                numerical[str(n)] = ck.numerical(a, aa, on)  # STOP beyond max(.005, 2*A/A KL) or captured mass < .99
                text[str(n)] = {'aa_equal': a['text_sha256'] == aa['text_sha256'],
                                'on_equals_off': a['text_sha256'] == on['text_sha256']}
            # The greedy-KL registration reuses UP2's numeric bound with this window's worst A/A floor.
            self.kl_bound = max(.005, 2 * max(number(record(v)['aa_kl']) for v in numerical.values()))
            needle = self.request(arm, 'needle150000', ['--needle', '--target', '150000', '--budget', '300',
                '--salt', f'{self.tag}:needle150000'], deadline, 2 * NEEDLE_S)
            ck.require(needle.get('recall') == 6 and needle.get('of') == 6 and needle.get('foreign_codes') == 0,
                       'replay 150K needle 6/6 or foreign-code failure')
            result = {'status': 'PASS', 'registration': REPLAY_REGISTRATION, 'numerical': numerical,
                'needle_150000': {k: needle.get(k) for k in ('recall', 'of', 'foreign_codes')},
                'text_sha_equality_INFORMATIONAL': text,
                'native_checks': 'NOT RUN: UP2 native 12-check apparatus absent (roadmap row 24); provenance audited by '
                                 'UP2 STOP_COUNTERS/STOP_LOG on every start'}
        after = self.stats(arm.name + '-gates-after')
        self.baseline_check(arm, after)
        ck.counters(after, before=before)
        return result

    def qualify(self, arm: Arm) -> dict[str, Json]:
        deadline = min(self.submit, self.started_at + self.allowed(arm))
        ck.require(self.clock() < deadline, 'startup/warmup/qualification budget exhausted')
        if self.group == REPLAY and (arm.role == REPLAY or arm == arms(self.group)[0]):
            return self.replay_gates(arm, deadline)
        if self.group == REPLAY:
            return self.greedy_kl(arm, deadline)
        if arm.role == 'control':
            return {'status': 'unchanged production source/env verified'}
        host_gates.admit(arm.role)
        results: dict[str, Json] = {'policy': 'Round4 config-only; resident adapter not required'}
        if arm.role == 'k5':
            return results
        for rank, host in enumerate(pf.HOSTS):
            ck.require(self.clock() < deadline, 'qualification deadline exhausted before submission')
            command = (f'cd ~/{self.remote} && python3 -m offload.up3.host_gates '
                f'--arm {arm.role} --rank {rank} --container {shlex.quote(self.ids[rank])}')
            text = self.ssh(arm.name + '-host-gate-' + host, host, command,
                timeout=min(15, deadline - self.clock()))
            receipt = record(cast(Json, json.loads(text)))
            ck.require(self.clock() < deadline, 'host evidence exceeded its full block allowance')
            host_gates.validate(receipt, arm.role, rank, self.ids[rank])
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
        self.call(arm.name + '-' + name, self.driver('cell', output, '--cell', name, '--cuts', str(self.out / 'cuts.json'),
            '--salt', self.tag + '-' + arm.name, '--ledger', str(self.out / 'requests-ledger.jsonl')),
            timeout=cells.ESTIMATES[name] * cells.REPEATS[name] * 2 + 30, inflight=3 if name == cells.CELLS[-1] else int(name[-1]))
        result = ck.read(output)
        ck.require(result.get('cell') == name, 'cell artifact identity differs')
        after = self.stats(arm.name + '-' + name + '-after')
        self.baseline_check(arm, after)
        ck.counters(after, before=before)
        ck.require(number(record(after['counters']).get('rounds_ok', 0)) > number(record(before['counters']).get('rounds_ok', 0)),
            'remote drafter took no successful round')
        result['proxy_before'], result['proxy_after'] = before, after
        output.write_text(json.dumps(result, allow_nan=False) + '\n')
        if self.group == REPLAY and arm.role == 'control':
            self.control_cells.setdefault(arm.name, {})[name] = result
        return result

    def finish(self, arm: Arm) -> dict[str, Json]:
        result: dict[str, Json] = {}
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
        if self.group == REPLAY and arm.role == 'control':
            # WIN2 checklist 4: every control repeat of a FINISHED start, kept even if a later start stops. Archive v3
            # (WIN2-R2): the production commit, and the clock cap both Sparks printed in THIS start's own start log.
            cap, source = win2_history.start_clock_cap(self.out, arm.name)
            before = dv2.archive_head(archive_path())
            dv2.archive_append(archive_path(), [{'window': self.out.name, 'start': arm.name, 'cell': name, 'metric': metric,
                'repeat': i, 'value': value, 'max_tokens': cells.BUDGET, 'commit': b.PRODUCTION,
                'production_tag': b.TAG, 'clock_cap_mhz': cap, 'clock_source': source,
                'date': time.strftime('%Y%m%d', time.gmtime(self.clock()))}
                for name, cell in self.control_cells.get(arm.name, {}).items()
                for metric, _ in REPLAY_METRICS[name] for i, value in enumerate(repeat_values(cell, metric))])
            result['archive'] = {'clock_cap_mhz': cap, 'clock_source': source, 'before': before,
                                 'after': dv2.archive_head(archive_path())}
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
        self.plain = self.hold_bounded(original)
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
                    self.restore_by = opened + RESTORE_S
            if not self.hold_written:
                ck.require(not production_stopped, 'UP3 production may be stopped without owned HOLD')
                return {'status': 'NOT NEEDED: no owned HOLD; all remote mutators fenced and absent' + (
                    '; chain marker present: the window ends with chain.abort' if chain.marker_present() else '')}
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
            self.commit_close(result, f'{self.tag}-restore-close')
            result['status'] = 'RESTORED: K1 START VERIFIED, greedy 6/6 IDENTICAL, PRODUCTION HOLD OK, window closed'
            return result
        finally:
            self.plain = original
            self.call = original_call
            (self.out / 'arm-memory.json').write_text(json.dumps(self.sup.arm_memory) + '\n')

    def hold_bounded(self, original: Callable[[str, list[str], float], tuple[int, str]]
                     ) -> Callable[[str, list[str], float], tuple[int, str]]:
        """Clamp cleanup, W8 and close transactions to ONE original HOLD deadline."""
        def bounded(label: str, argv: list[str], timeout: float) -> tuple[int, str]:
            if label in ('cancel-close', 'retain-close'):
                return original(label, argv, min(30, timeout))
            left = self.restore_by - self.clock()
            ck.require(left > 0, 'UP3 HOLD+85 exhausted; manual restore required; HOLD stays open')
            result = original(label, argv, min(left, timeout))
            ck.require(self.clock() < self.restore_by, 'UP3 recovery command exceeded original deadline')
            return result
        return bounded

    def commit_close(self, result: dict[str, Json], tag: str) -> None:
        """Remove the window's own HOLD whose unlink the `tag` (-restore-close) scope's rm shim recorded (W8 close, or
        chain.CHAINED_CLOSE): every owned phase fenced and proved absent first; any failure retains the original HOLD."""
        for prior in list(self.owner_tags):
            self.fence_phase(prior)
        ck.require(self.restore_by - self.clock() >= 90, 'UP3 final close needs 90s reserve; HOLD stays open')
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

    def chained_end(self) -> dict[str, Json]:
        """WIN2 chain (non-last window, no stop): experiment stopped, Sparks proven quiet, own HOLD removed, production
        LEFT DOWN with the CHAINED marker and the hold placeholders LEFT UP (WIN2-R2). Any failure here falls back to the
        strict restore while the HOLD is ours."""
        self.sup.close()
        original = self.plain
        self.plain = self.hold_bounded(original)
        try:
            ck.require(self.hold_written and math.isfinite(self.restore_by), 'chained end without an owned HOLD')
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
            rollback = self.out / 'rollback.json'
            identity: dict[str, Json] = {'path': str(rollback), 'sha256': pf.digest(rollback)}
            # chain.CHAINED_CLOSE proves both placeholders up and removes only the HOLD file, inside the window's own
            # -restore-close scope, whose rm shim RECORDS the unlink; commit_close removes HOLD once every phase is fenced.
            # A numbered close transaction of this window's HOLD (scoped_remote close_owner): a fallback strict restore
            # still has the unfenced, uncommitted `<tag>-restore-close` scope for its W8 close.
            tag = f'{self.tag}-close-txn-1-restore-close'
            deadline = self.restore_by - 90
            self.remember(tag, deadline)
            ck.require(self.clock() < deadline, 'UP3 chained close budget exhausted; HOLD stays open')
            try:
                rc, text = self.plain('chained-close', [*pf.SSH, 'sparkhead', owner.command('launch', tag,
                    ['bash', '-c', chain.CHAINED_CLOSE], deadline=deadline, close_owner=self.tag)], 120)
            finally:
                self.fence_phase(tag)
            ck.require(rc == 0 and chain.CHAINED_CLOSE_OK in text.splitlines(), 'UP3 chained HOLD close unrecorded; HOLD stays open')
            result: dict[str, Json] = {'sparks': quiet, 'production_identity': identity}
            self.commit_close(result, tag)
            chain.finish_chained(self.out, {'runner': 'up3', 'group': self.group, 'sparks': quiet,
                                            'production_identity': identity})
            return {**result, 'status': 'CHAINED: production left down (NDM 2026-10-27 chain); experiment stopped, Sparks '
                                        'quiet, own HOLD closed, hold placeholders kept up'}
        finally:
            self.plain = original
            (self.out / 'arm-memory.json').write_text(json.dumps(self.sup.arm_memory) + '\n')

    def abort(self) -> dict[str, Json]:
        """WIN2-R2 'abort' end: this window owns no HOLD but the chain marker is present (production is down from the
        previous chained window): the shared strict W8 recovery under its own scoped HOLD + post-close watch."""
        self.sup.close()
        return chain.abort(self.out / 'chain-abort')


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
    def abort(self) -> dict[str, Json]: ...
    hold_written: bool


def window(backend: Backend, group: str, checkpoint: Callable[[dict[str, Json]], None]) -> dict[str, Json]:
    result: dict[str, Json] = {'status': 'INCOMPLETE', 'starts': {}, 'decision': {'status': 'NOT A CANDIDATE'}}
    hold = stopped = False
    try:
        admit(group)
        backend.preflight()
        hold = True
        result['opened'] = backend.open_hold()
        backend.snapshot()
        for i, arm in enumerate(arms(group)):
            ck.require(backend.fits(arm), 'mandatory remaining OFF/arm/OFF blocks cannot fit; restore now')
            stopped = True
            row: dict[str, Json] = {'status': 'STARTED', 'role': arm.role, 'cells': {}}
            record(result['starts'])[arm.name] = row
            checkpoint({'arm': arm.name, 'status': 'STARTED'})
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
            row['status'] = 'COMPLETE'
            checkpoint({'arm': arm.name, 'status': 'COMPLETE'})
        result['decision'] = decision(group, record(result['starts']))
        result['status'] = 'COMPLETE'
    except (Exception, KeyboardInterrupt) as exc:
        result['stop'] = f'{type(exc).__name__}: {exc}'
    finally:
        saved = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            # WIN2-R2 (S92 FIX 1): `hold` means a HOLD may be ours (preflight passed); restore() fences every phase and
            # re-reads it, and a window that owns none while the chain marker is present takes chain.abort, never
            # "NOT NEEDED" (production is down from the previous chained window).
            action = chain.end_action('stop' in result, hold)
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
                else:
                    if not backend.hold_written:
                        action = chain.end_action(True, False)
                        result['own_restore'] = result['restore']
            if action == 'abort':
                try:
                    result['restore'] = backend.abort()
                except (Exception, KeyboardInterrupt) as exc:
                    result['restore'] = {'status': 'FAILED', 'error': f'chain abort: {type(exc).__name__}: {exc}'}
                # chain.abort runs its own post-close watch (and clears the marker only on RESTORED + watch PASS).
                result['post_close_watch'] = record(result['restore']).get('post_close_watch', {'status': 'NOT RUN'})
            elif action == 'none' and not hold:
                result['restore'] = {'status': 'NOT NEEDED: preflight only'}
            result['end_action'] = action
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
    if 'stop' in result or record(result['restore']).get('status') == 'FAILED':
        result['decision'] = {'status': 'NOT A CANDIDATE', 'reason': 'STOP or production restore unverified'}
    return result


def budget_minutes(group: str = 'split') -> float:
    return (SETUP_S + sum(block_s(group, a, i == 0) for i, a in enumerate(arms(group))) + RECOVERY_S) / 60


def replay_plan() -> str:
    blocks = [block_s(REPLAY, a, i == 0) for i, a in enumerate(arms(REPLAY))]
    return ('DRY RUN: no network, file or command. Coordinator Deepseek/coordinator only.\n'
        'UP2-ADOPT 0075 replay skip on CURRENT production 596ba1c / prod-k1-327-ops1-20261026 (K1 327680, KV2502600, TP2 K2 fb68c52).\n'
        'One HOLD: OFF/replay/OFF. Arm = production + VLLM_DSV41_REPLAY_SKIP=1 + offload/up3/overlay/replay (4 vLLM files, '
        'remote_dspark as one directory replacing production\'s mount); replay guard on sparkhead.\n'
        f'Every start: six greedy vs K1-post at {REPLAY_NEAR_TIE_NATS}nat ({REPLAY_REGISTRATION}); OPS1 warmup. The replay arm '
        'records each divergence top-20; the last control re-probes them (WIN2 greedy_kl, bound max(.005, 2 x worst A/A KL)).\n'
        'First control: same-start A/A 32K and 150K numeric (T0, 64 tokens, first-token top20). Replay: each within '
        'max(.005, 2*A/A KL) and captured mass >=.99; 150K needle 6/6, no foreign codes; UP2 provenance counters/logs '
        'on every start. Native 12-check receipts NOT RUN (apparatus absent).\n'
        'Cells 150K B1 x3 and 32K B1 x4 DECIDE (win2-v1: cold TTFT, ms/round guard; LOSS = worse than own control spread '
        'in >=2/3 repeats); short B1 x4 reported. ADOPT = gates PASS and no decision LOSS; no win required: the gain is the '
        'MEASURED 2026-10-25 300K cold request (ON 130.3-130.9 s vs OFF 139.6-141.8 s). Controls go to the WIN2 archive.\n'
        f'ESTIMATED blocks {blocks[0]}/{blocks[1]}/{blocks[2]}s; HOLD {budget_minutes(REPLAY):.2f}min <=85 incl. 120s setup + 720s recovery.\n'
        'MEASURED basis UP3-dual-ring-W-20261027T165101Z (same production): start-to-gates 420-440s, short-b1 49-51s, '
        '32k-b1 106-109s, restore 525s; 150K B1 ESTIMATED from MEASURED 69.4s TTFT + 512 tokens at 40-42 tok/s.\n'
        'Strict six distinct IDENTICAL W8 restore; 15min shared watch. ADOPT CANDIDATE never changes production: the '
        'coordinator promotes separately.')


def plan(group: str = 'split') -> str:
    if group == REPLAY:
        return replay_plan()
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


def post_close(result: dict[str, Json], out: Path, tag: str) -> None:
    """Watch even a stopped experiment after verified restoration; never an open HOLD. A chain abort ran its own watch."""
    if result.get('end_action') == 'abort' or not str(record(result['restore']).get('status', '')).startswith('RESTORED'):
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


def close_chain(result: dict[str, Json]) -> None:
    """WIN2.md 'restore' end: the chain (if any) ends with chain.clear() only after the window's OWN strict W8 restore
    verified AND its post-close watch passed. chain.abort clears by itself; a chained end keeps the marker."""
    if (result.get('end_action') == 'restore' and str(record(result['restore']).get('status', '')).startswith('RESTORED')
            and record(result.get('post_close_watch', {})).get('status') == 'PASS'):
        chain.clear()


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
        ck.require(budget_minutes(args.group) <= 85,
            f'UP3 matrix needs ESTIMATED {budget_minutes(args.group):.2f}min >85; redesign required before any host access')
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
    close_chain(result)
    result.update({'production': b.PRODUCTION, 'experiment': args.expected_experiment_commit, 'group': args.group,
        'shared_owner_sha256': pf.digest(b.ROOT / 'offload/launch/windows/scoped_remote.py'),
        'budget_minutes_estimated': budget_minutes(args.group), 'budget_sha256': pf.digest(b.HERE / 'budget.json'),
        'restore_sha256': pf.RESTORE_SHA256, 'qualification_policy': 'Round4 config-only',
        'chain_context': {'enabled': chain.enabled(), 'last': chain.last(), 'mode': live.chain_mode},
        'artifact_sha256': {str(p.relative_to(out)): pf.digest(p) for p in out.rglob('*.json*') if p.is_file()}})
    (out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'stop': result.get('stop'), 'decision': result['decision'], 'restore': result['restore']}))
    chained = str(record(result['restore']).get('status', '')).startswith('CHAINED')
    return 2 if record(result['restore']).get('status') == 'FAILED' else (
        0 if result['status'] == 'COMPLETE' and (chained or record(result.get('post_close_watch', {})).get('status') == 'PASS') else 1)


if __name__ == '__main__':
    raise SystemExit(main())
