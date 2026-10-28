"""Cat-only process discovery in the existing container's PID namespace.

No engine imports, remote interpreter, affinity writes or reliance on pin logs.
See HOST-PARSE-AUDIT.md Round 7 for observed argv and source provenance.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import cast

from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number
from offload.up3 import poller_identity

MAX_PID = 16384
BATCH = 512
PROXY_PACKAGE = '/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/remote_dspark'
# Exact BOOT in production 596ba1c:remote_dspark/proxy_proc.py:433-443.
PROXY_BOOT = '''import importlib.util, sys
pkg, fd = sys.argv[1], int(sys.argv[2])
spec = importlib.util.spec_from_file_location("remote_dspark", pkg + "/__init__.py", submodule_search_locations=[pkg])
mod = importlib.util.module_from_spec(spec)
sys.modules["remote_dspark"] = mod
spec.loader.exec_module(mod)
from remote_dspark.proxy_proc import main
sys.exit(main(fd))'''

Cat = Callable[[list[str], bool], str]


def proc_path(path: str) -> bool:
    if path == '/proc/sys/kernel/ns_last_pid':
        return True
    match = re.fullmatch(r'/proc/([1-9][0-9]*)(?:/task/([1-9][0-9]*))?/(status|stat|cmdline)', path)
    return match is not None and all(int(pid) <= MAX_PID for pid in match.groups()[:2] if pid is not None)


def batches(paths: list[str], cat: Cat, *, sparse: bool = False) -> str:
    return ''.join(cat(paths[i:i+BATCH], sparse) for i in range(0, len(paths), BATCH))


def cpu_list(text: str) -> list[int]:
    ck.require(re.fullmatch(r'\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*', text) is not None, 'invalid /proc CPU list')
    cpus: set[int] = set()
    for part in text.split(','):
        a, _, b = part.partition('-')
        low, high = int(a), int(b or a)
        ck.require(0 <= low <= high < 4096, 'invalid /proc CPU range')
        cpus.update(range(low, high+1))
    return sorted(cpus)


def statuses(text: str) -> dict[int, dict[str, Json]]:
    ck.require(text.startswith('Name:\t'), 'missing /proc status records')
    out: dict[int, dict[str, Json]] = {}
    for block in text.split('Name:\t')[1:]:
        fields: dict[str, str] = {}
        for line in ('Name:\t'+block).splitlines():
            key, sep, value = line.partition(':')
            ck.require(sep == ':' and key not in fields, 'malformed/duplicate /proc status field')
            fields[key] = value.strip()
        ck.require(all(key in fields for key in ('Name', 'State', 'Pid', 'Tgid', 'PPid', 'Threads', 'Cpus_allowed_list')),
            'incomplete /proc status record')
        ck.require(all(fields[key].isdigit() for key in ('Pid', 'Tgid', 'PPid', 'Threads')), 'invalid /proc status IDs')
        tid = int(fields['Pid'])
        ck.require(0 < tid <= MAX_PID and tid not in out, 'duplicate/out-of-range /proc TID')
        out[tid] = {'tid': tid, 'pid': int(fields['Tgid']), 'ppid': int(fields['PPid']),
            'comm': fields['Name'], 'state': fields['State'].split()[0], 'thread_count': int(fields['Threads']),
            'cpus': cast(Json, cpu_list(fields['Cpus_allowed_list']))}
    return out


def snapshot(cat: Cat) -> dict[int, dict[str, Json]]:
    upper = cat(['/proc/sys/kernel/ns_last_pid'], False).strip()
    ck.require(upper.isdigit() and 0 < int(upper) <= MAX_PID, 'PID namespace exceeds bounded census; refuse')
    # /proc/TID/status exists for nonleaders too, although readdir(/proc) hides them.
    # cat's own new TID is above this boundary and never treated as an engine process.
    return statuses(batches([f'/proc/{tid}/status' for tid in range(1, int(upper)+1)], cat, sparse=True))


def identify(rows: dict[int, dict[str, Json]], rank: int, cat: Cat) -> list[dict[str, Json]]:
    ck.require(rank in (0, 1), 'census rank differs')
    ck.require(sum(row['pid'] == tid for tid, row in rows.items()) <= 8,
        'excessive process leaders in engine namespace; refuse before argv reads')
    found: list[dict[str, Json]] = []
    for tid, row in rows.items():
        if tid != row['pid']:
            continue
        argv = cat([f'/proc/{tid}/cmdline'], False).rstrip('\0').split('\0')
        role = ''
        if tid == 1:
            ck.require(argv[:3] == ['/usr/bin/python3', '/usr/local/bin/vllm', 'serve']
                and argv.count('--node-rank') == 1 and argv[argv.index('--node-rank')+1] == str(rank),
                'required vllm launcher argv/rank differs')
            role = 'launcher'
        elif argv == ['VLLM::EngineCore'] and row['comm'] == 'VLLM::EngineCor':
            role = 'engine'
        elif argv == [f'VLLM::Worker_TP{rank}'] and row['comm'] == 'VLLM::Worker_TP':
            role = 'worker'
        elif (len(argv) == 6 and argv[:3] == ['/usr/bin/python3', '-I', '-c']
                and argv[3].strip() == PROXY_BOOT and argv[4] == PROXY_PACKAGE and argv[5].isdigit()):
            role = 'proxy'
        elif (len(argv) == 3 and argv[:2] == ['/usr/bin/python3', '-c']
                and re.fullmatch(r'from multiprocessing.resource_tracker import main;main\(\d+\)', argv[2])):
            continue  # interpreter bookkeeping, not an engine/plugin/proxy process
        elif argv and argv[0] == 'cat' and all(proc_path(path) for path in argv[1:]):
            continue  # concurrent read-only census reader
        else:
            raise ValueError(f'STOP: unidentified process in engine namespace pid {tid}, comm {row["comm"]!r}')
        ck.require(row['state'] not in ('Z', 'X'), f'required {role} process is not live')
        found.append({**row, 'role': role, 'argv': cast(Json, argv)})
    by_role = {str(row['role']): row for row in found}
    expected = {'launcher', 'engine', 'worker', 'proxy'} if rank == 0 else {'launcher', 'worker'}
    ck.require(len(found) == len(expected) and set(by_role) == expected, 'missing/duplicate required engine/worker/proxy process')
    worker_parent = by_role['engine']['pid'] if rank == 0 else 1
    ck.require(by_role['worker']['ppid'] == worker_parent
        and (rank == 1 or by_role['engine']['ppid'] == 1 and by_role['proxy']['ppid'] == by_role['worker']['pid']),
        'required engine/worker/proxy process tree differs')
    return sorted(found, key=lambda row: number(row['pid']))


def stats(text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in text.splitlines():
        first, sep, rest = line.partition(' (')
        _, end, tail = rest.rpartition(') ')
        fields = tail.split()
        ck.require(first.isdigit() and sep == ' (' and end == ') ' and len(fields) >= 20
            and fields[19].isdigit() and int(first) not in out, 'invalid/duplicate /proc stat identity')
        out[int(first)] = fields[19]  # starttime, field 22
    return out


def collect(rank: int, cat: Cat, *, receipts: Callable[[int, str], str] | None = None) -> dict[str, Json]:
    before = snapshot(cat)
    processes = identify(before, rank, cat)
    pids = {int(number(p['pid'])) for p in processes}
    tids = sorted(tid for tid, row in before.items() if row['pid'] in pids)
    paths = [f'/proc/{before[tid]["pid"]}/task/{tid}/stat' for tid in tids]
    starts = stats(batches(paths, cat))
    proofs: dict[int, dict[str, Json]] = {}
    if receipts is not None:
        for process in processes:
            if process['role'] == 'worker':
                pid = int(number(process['pid']))
                proofs[pid] = poller_identity.decode(receipts(pid, starts[pid]))
    after = snapshot(cat)
    current = identify(after, rank, cat)
    def stable(row: dict[str, Json]) -> dict[str, Json]:
        return {key: value for key, value in row.items() if key != 'state'}
    ck.require(list(map(stable, processes)) == list(map(stable, current)),
        'required process identity/count/leader affinity changed during census')
    relevant = {tid: row for tid, row in before.items() if row['pid'] in pids}
    ck.require({tid: stable(row) for tid, row in relevant.items()}
        == {tid: stable(row) for tid, row in after.items() if row['pid'] in pids},
        'required thread census/affinity changed during census')
    ends = stats(batches(paths, cat))
    ck.require(set(starts) == set(tids) and starts == ends, 'required TID missing/reused during census')
    out: list[Json] = []
    for process in processes:
        threads = [{**row, 'starttime': starts[tid]} for tid, row in relevant.items() if row['pid'] == process['pid']]
        ck.require(len(threads) == process['thread_count'] and bool(threads), 'incomplete required process thread census')
        item = {**process, 'status': 'OBSERVED', 'threads': cast(Json, threads)}
        if process['role'] == 'worker' and receipts is not None:
            item['pin_receipt'] = proofs[int(number(process['pid']))]
        out.append(item)
    return {'pinned_pids': [p['pid'] for p in processes if p['role'] != 'proxy'],
        'proxy_pids': [p['pid'] for p in processes if p['role'] == 'proxy'], 'processes': out,
        'discovery': 'OBSERVED: two bounded PID-namespace sweeps; exact argv/tree; per-TID status/stat identity',
        'poller_identity': ('OBSERVED: native-start receipt plus independently stable TID/comm/affinity census; host validation required'
            if receipts is not None else 'UNMEASURED: native RDMA std::thread inherits shared worker comm; no unique source-defined name'),
        'attribution': 'UNMEASURED: native TID births/Engram attribution require resident hooks'}
