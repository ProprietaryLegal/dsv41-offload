"""Fail-closed GB10 qualification contract. CPU evidence never satisfies a GB10 gate.

A separately reviewed in-engine qualifier must produce this receipt from actual inputs
on each rank. The window invokes that pinned program before timing, not a boolean approval.
An absent/partial qualifier is an explicit preparation gap, never implicit clearance.
"""
from __future__ import annotations

import math

from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record
from offload.up3 import build as b

CASES: dict[str, tuple[str, ...]] = {
    'split': ('m33', 'm42', 'm43', 'm63', 'dispatch32', 'dispatch64', 'teacher-forced-rounding'),
    'vor': ('valid', 'lens', 'indices', 'padded', 'invalid', 'replay', 'ctx327680',
            'mixed-stock', 'capture-stock', 'noncausal-stock', 'microbatch-stock', 'actual-groups', 'launch'),
    'k5': ('five-proposals', 'b4-24-rows', 'noncausal-stock'),
    'dual-ring': ('gid-device0', 'gid-device1', 'rank-log', 'collective-bytes', 'traffic-device0', 'traffic-device1'),
    'cpu-pin': ('all-thread-affinities', 'poller14', 'async-proxy', 'engram'),
}


def validate(receipt: dict[str, Json], *, arm: str, rank: int, run: str,
             container: str, since: float, now: float, program_sha: str) -> None:
    ck.require(arm in CASES, 'unknown GB10 gate arm')
    ck.require(receipt.get('schema') == 'up3-gb10-gates-v1' and receipt.get('arm') == arm
        and receipt.get('rank') == rank and receipt.get('run') == run and receipt.get('container') == container,
        'GB10 gate identity mismatch')
    ck.require(receipt.get('production') == b.PRODUCTION and receipt.get('program_sha256') == program_sha
        and receipt.get('manifest_sha256') == b.digest((b.HERE / 'manifests' / (arm + '.json')).read_bytes()),
        'GB10 gate source identity mismatch')
    ck.require(receipt.get('device_capability') == [12, 1] and receipt.get('scope') == 'actual-engine-inputs',
        'GB10 actual in-engine evidence required, not CPU/synthetic-only')
    stamp = number(receipt.get('time', 0))
    ck.require(math.isfinite(stamp) and since <= stamp <= now + 2, 'stale/future GB10 gate receipt')
    cases = record(receipt['cases'])
    ck.require(set(cases) == set(CASES[arm]), 'GB10 gate cases incomplete/extra')
    for name, value in cases.items():
        row = record(value)
        ck.require(row.get('pass') is True and row.get('samples', 0) != 0, 'GB10 gate failed: ' + name)
        samples = number(row['samples'])
        ck.require(math.isfinite(samples) and samples > 0, 'invalid gate sample count')
    if arm == 'split':
        for m in (33, 42, 43, 63):
            row = record(cases[f'm{m}'])
            ck.require(row.get('comparison') == 'every-real-output-byte-before-TP'
                and row.get('rows') == m and row.get('capacity') == 42
                and row.get('oracle') == ('single-smoe' if m <= 42 else 'per-row-decode')
                and row.get('matched_inputs_weights_ids_perm_pad_limit') is True,
                '0064 exact S17 gate incomplete')
        ck.require(all(record(cases[f'dispatch{m}']).get('unchanged') is True for m in (32, 64)),
                   '0064 dispatch boundary changed')
        rounding = record(cases['teacher-forced-rounding'])
        ck.require(rounding.get('same_prefix_and_batch_shape') is True
            and number(rounding.get('captured_mass', 0)) >= .99
            and number(rounding.get('kl', 1)) <= max(.005, 2 * number(rounding.get('cc_kl', 0)))
            and rounding.get('task_review') == 'pass', '0064 rounding/distribution gate failed')
    elif arm == 'vor':
        for name in ('valid', 'lens', 'indices', 'padded', 'invalid', 'replay', 'ctx327680'):
            ck.require(record(cases[name]).get('comparison') == 'every-byte-and-index', '0067 byte/index proof absent')
        group = record(cases['actual-groups'])
        ck.require(number(group.get('count', 0)) > 0 and group.get('capacity_mode') is True
            and group.get('derived_from_loaded_engine') is True, '0067 actual target groups absent')
        launch = record(cases['launch'])
        ck.require(launch.get('once_per_eligible_step') is True and launch.get('request_time_compile') is False
            and launch.get('exception_disabled') is False and launch.get('replay_copy_only_on_change') is True,
            '0067 launch/copy/exception gate failed')
    elif arm == 'k5':
        ck.require(record(cases['five-proposals']).get('proposals') == 5
            and record(cases['b4-24-rows']).get('rows') == 24
            and record(cases['noncausal-stock']).get('byte_identical') is True, 'K5 dimension/noncausal gate failed')
    elif arm == 'dual-ring':
        for index, name in enumerate(('rocep1s0f1', 'roceP2p1s0f1')):
            gid = record(cases[f'gid-device{index}'])
            ck.require(gid.get('device') == name and gid.get('gid_index') == 3 and gid.get('ipv4_roce_v2') is True,
                       'dual-ring GID gate failed')
            ck.require(number(record(cases[f'traffic-device{index}']).get('bytes_delta', 0)) > 0,
                       'dual-ring device carried no traffic')
        line = str(record(cases['rank-log']).get('line', ''))
        ck.require(line.startswith(f'EXL3 RDMA all-reduce: rank {rank} of 2 on rocep1s0f1,roceP2p1s0f1')
            and receipt.get('nccl_only_fallback') is False
            and record(cases['collective-bytes']).get('byte_identical') is True, 'dual-ring byte/log gate failed')
    elif arm == 'cpu-pin':
        affinity = record(cases['all-thread-affinities'])
        ck.require(affinity.get('every_tid') is True and number(affinity.get('thread_count', 0)) > 0
            and affinity.get('allowed_cpus') == '5-9,15-19'
            and record(cases['poller14']).get('cpu') == 14, 'cpu affinity/poller gate failed')
    # Do not accept NaN-based numerical comparisons, or a nonliteral pass marker.
    ck.require(all(record(v).get('pass') is True for v in cases.values()), 'gate not passed')
