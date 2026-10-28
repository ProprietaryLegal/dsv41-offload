"""Read-only config-arm evidence. No engine imports, worker hooks or CUDA calls.

Round 4 deliberately replaces resident-worker qualification for config-only
arms. These observations do not claim native collective-byte or TID-birth tests.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import shlex
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import cast

from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record
from offload.up3 import proc_census, poller_identity, rdma_pin

CONFIG_ARMS = ('k5', 'dual-ring', 'cpu-pin')
DEVICES = ('rocep1s0f1', 'roceP2p1s0f1')
CPUS = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
RDMA_SOURCE = '/usr/local/lib/python3.12/dist-packages/vllm_exl3_moe/rdma.py'
# Exact logger message bodies; logging prefixes are outside the match.
# src-image/vllm_exl3_moe/rdma.py:108-110 and __init__.py:53.
RDMA_INIT = re.compile(r'EXL3 RDMA all-reduce: rank (?P<rank>\d+) of (?P<world>\d+) on '
    r'(?P<devices>[^,\s]+(?:,[^,\s]+)*), gid (?P<gid>\d+), proxy on cpu (?P<cpu>\d+), '
    r'up to \d+ KB \(slots \d+ KB, all-gathers (?:up to \d+ KB|off)\)$')
PIN_INIT = re.compile(r'vllm_exl3_moe: pid (?P<pid>\d+), \d+ threads pinned to CPUs '
    r'\[\d+(?:, \d+)*\] \(EXL3_CPU_PIN\)$')
# This module is a production bind mount, NOT present in the src-image mirror.
# 596ba1c:offload/dspark/spark/remote_dspark/speculator.py:152-155;
# byte-verified by docker exec cat on BOTH ranks during the Round 6 audit.
PROXY_INIT = re.compile(r'remote DSpark \([^()]+ policy\) rank 0 -> \S+ through proxy process pid (?P<pid>\d+); '
    r'rank-0 buffers: \d+\.\d MiB device, \d+\.\d MiB shared region$')


def log_evidence(lines: Iterable[str]) -> dict[str, Json]:
    rdma: list[Json] = []
    pinned: set[int] = set()
    proxy: set[int] = set()
    routing = 0
    for raw in lines:
        line = raw.rstrip()
        match = RDMA_INIT.search(line)
        if 'EXL3 RDMA all-reduce: rank ' in line:
            ck.require(match is not None, 'malformed RDMA initialization log')
            ck.require(len(rdma) < 8, 'ambiguous/excessive RDMA initialization records')
            assert match is not None
            rdma.append(match.group(0))
        routing += "EXL3 RDMA all-reduce: vLLM's TP all-reduce routed" in line
        ck.require('RDMA all-reduce off' not in line and 'NCCL stays' not in line, 'NCCL-only fallback')
        match = PIN_INIT.search(line)
        if 'threads pinned to CPUs' in line or '(EXL3_CPU_PIN)' in line:
            ck.require(match is not None, 'malformed CPU-pin initialization log')
            assert match is not None
            pinned.add(int(match.group('pid')))
        match = PROXY_INIT.search(line)
        if 'through proxy process pid' in line:
            ck.require(match is not None, 'malformed proxy initialization log')
            assert match is not None
            proxy.add(int(match.group('pid')))
    return {'rdma': rdma, 'pinned_pids': cast(Json, sorted(pinned)), 'proxy_pids': cast(Json, sorted(proxy)),
        'rdma_routing_lines': routing}


def serve_arguments(script: str) -> list[str]:
    """Parse the mounted launcher, never Docker's ['/serve.sh'] wrapper argv."""
    lines = re.sub(r'\\\r?\n', ' ', script).splitlines()
    commands = [shlex.split(line, comments=True) for line in lines]
    launches = [tokens for tokens in commands if tokens[:3] == ['exec', 'vllm', 'serve']]
    ck.require(len(launches) == 1, 'host gate requires one exec vllm serve in /serve.sh')
    argv = launches[0]
    ck.require(sum(token == '--port' or token.startswith('--port=') for tokens in commands for token in tokens) == 1
        and '--port' in argv, 'host gate requires exactly one --port argument in /serve.sh')
    return argv


def serve_port(script: str) -> str:
    argv = serve_arguments(script)
    index = argv.index('--port')
    ck.require(index + 1 < len(argv) and argv[index + 1] in ('8000', '8001'),
        'host gate requires literal port 8000/8001 in /serve.sh')
    return argv[index + 1]


def admit(arm: str) -> None:
    ck.require(arm in CONFIG_ARMS,
        f'{arm}: parked code arm; reviewed resident-worker exactness integration unavailable; refuse BEFORE HOLD')


def gid(root: Path, device: str, *, read: Callable[[Path], str] = Path.read_text) -> dict[str, Json]:
    port = root / device / 'ports/1'
    address = ipaddress.IPv6Address(read(port/'gids/3').strip()).ipv4_mapped
    kind = read(port/'gid_attrs/types/3').strip()
    netdev = read(port/'gid_attrs/ndevs/3').strip()
    return {'device': device, 'index': 3, 'type': kind, 'ipv4': str(address), 'netdev': netdev}


def validate_gid(row: dict[str, Json], index: int) -> None:
    ip = ipaddress.IPv4Address(str(row['ipv4']))
    ck.require(row.get('device') == DEVICES[index] and row.get('index') == 3
        and row.get('type') == 'RoCE v2' and bool(row.get('netdev'))
        and ip in ipaddress.IPv4Network(f'10.202.{26+index}.0/24'), 'dual-ring GID3 fabric differs')


def collect(arm: str, rank: int, container: str) -> dict[str, Json]:
    admit(arm)
    inspect_running(subprocess.check_output(['docker', 'inspect', 'dsv41-serve'], text=True), container)
    script = subprocess.check_output(['docker', 'exec', 'dsv41-serve', 'cat', '/serve.sh'], text=True)
    ck.require(serve_port(script) == '8001', 'host gate requires experiment8001')
    proc = subprocess.Popen(['docker', 'logs', 'dsv41-serve'], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, errors='replace')
    assert proc.stdout is not None
    observed = log_evidence(proc.stdout)
    ck.require(proc.wait() == 0, 'host gate docker logs failed')
    result: dict[str, Json] = {'arm': arm, 'rank': rank, 'container': container,
        'policy': 'Round4 config-only; six experimental greedy checks required separately'}
    if arm == 'dual-ring':
        result.update(rdma=observed['rdma'], gids=[gid(Path('/sys/class/infiniband'), dev) for dev in DEVICES])
    elif arm == 'cpu-pin':
        reads = ExperimentReads()
        evidence = proc_census.collect(rank, reads.proc, receipts=reads.pin_receipt)
        evidence['log_corroboration'] = {key: observed[key] for key in ('pinned_pids', 'proxy_pids')}
        result['affinity'] = evidence
        inspect_running(subprocess.check_output(['docker', 'inspect', 'dsv41-serve'], text=True), container)
    return result


def validate(result: dict[str, Json], arm: str, rank: int, container: str) -> None:
    admit(arm)
    ck.require(result.get('arm') == arm and result.get('rank') == rank and result.get('container') == container,
        'host evidence arm/rank/container differs')
    if arm == 'dual-ring':
        lines = ck.strings(result['rdma'])
        match = RDMA_INIT.fullmatch(lines[0]) if len(lines) == 1 else None
        ck.require(match is not None and match.group('rank') == str(rank) and match.group('world') == '2'
            and match.group('devices') == ','.join(DEVICES) and match.group('gid') == '3',
            'second ring absent or ambiguous exact rank/GID3 log')
        rows = ck.rows(result['gids'])
        ck.require([row.get('device') for row in rows] == list(DEVICES), 'both dual-ring functions required')
        for i, row in enumerate(rows):
            validate_gid(row, i)
    elif arm == 'cpu-pin':
        evidence = record(result['affinity'])
        pinned = cast(list[Json], evidence['pinned_pids'])
        proxy = cast(list[Json], evidence['proxy_pids'])
        processes = ck.rows(evidence['processes'])
        unavailable: list[Json] = []
        if [p.get('pid') for p in processes] != sorted(set(pinned + proxy), key=number):
            unavailable.append('affinity census omits a known process or contains unexpected/duplicate processes')
        if not pinned:
            unavailable.append('required engine/worker PIDs unavailable in direct census')
        if rank == 0 and not proxy:
            unavailable.append('proxy PID unavailable')
        if sum(process.get('role') == 'worker' for process in processes) != 1:
            unavailable.append('exactly one required worker with startup poller receipt is required')
        for process in processes:
            if process.get('status') == 'UNMEASURED':
                ck.require(bool(process.get('reason')), 'unmeasured affinity needs a reason')
                unavailable.append(process)
                continue
            threads = ck.rows(process['threads'])
            ck.require(process.get('status') == 'OBSERVED' and bool(threads), 'empty affinity evidence')
            tids = [t['tid'] for t in threads]
            ck.require(len(set(map(str, tids))) == len(tids), 'duplicate affinity TID')
            leader = next((t for t in threads if t.get('tid') == process['pid']), None)
            ck.require(leader is not None and leader.get('cpus') == CPUS,
                f'observed worker/proxy affinity differs: process leader {process["pid"]} missing or not on registered mask')
            poller = poller_identity.validate(process, threads) if process.get('role') == 'worker' else None
            ck.require(process.get('role') == 'worker' or 'pin_receipt' not in process,
                'proven RDMA poller identity receipt belongs only to the required worker')
            for thread in threads:
                is_poller = poller is not None and thread.get('tid') == poller
                ck.require(bool(thread.get('starttime')) and (is_poller or
                    thread.get('cpus') == CPUS and thread.get('comm') != rdma_pin.COMM),
                    'observed worker/proxy affinity differs: CPU14 exception requires proven RDMA poller identity; '
                    'every other thread must carry the registered mask')
        corroboration = record(evidence.get('log_corroboration', {}))
        for key, known in (('pinned_pids', pinned), ('proxy_pids', proxy)):
            ck.require(set(map(str, cast(list[Json], corroboration.get(key, [])))) <= set(map(str, known)),
                'optional CPU-pin log contradicts direct process census')
        evidence['status'] = 'PARTIAL' if unavailable else 'OBSERVED'
        evidence['unmeasured'] = unavailable
        evidence['native_tid_births'] = 'RDMA: startup receipt checked against stable census; other native/Engram attribution UNMEASURED'
        ck.require(not unavailable,
            'incomplete required CPU-pin affinity census: ' + json.dumps(evidence, allow_nan=False))


class ProductionReads:
    """Finite inspect/logs/exec-cat/sysfs allowlist; no remote Python or port request."""

    def __init__(self, host: str | None = None) -> None:
        ck.require(host in (None, 'sparkhead', 'sparkworker-wired'), 'probe host must be a registered Spark')
        self.host = host

    def output(self, argv: list[str]) -> str:
        files = {str(Path('/sys/class/infiniband') / dev / 'ports/1' / suffix)
            for dev in DEVICES for suffix in ('gids/3', 'gid_attrs/types/3', 'gid_attrs/ndevs/3')}
        ck.require(argv in (['docker', 'inspect', 'dsv41-serve'], ['docker', 'logs', 'dsv41-serve'],
            ['docker', 'exec', 'dsv41-serve', 'cat', '/serve.sh'],
            ['docker', 'exec', 'dsv41-serve', 'cat', RDMA_SOURCE])
            or len(argv) == 2 and argv[0] == 'cat' and argv[1] in files,
            'production probe refused a command outside inspect/logs/exec cat/sysfs allowlist')
        command = argv if self.host is None else [
            'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', self.host, shlex.join(argv)]
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, errors='replace', timeout=15)

    def path(self, path: Path) -> str:
        return self.output(['cat', str(path)])

    def proc(self, paths: list[str], sparse: bool = False) -> str:
        ck.require(0 < len(paths) <= proc_census.BATCH and all(proc_census.proc_path(path) for path in paths),
            'production probe refused /proc path outside bounded cat allowlist')
        ck.require(not sparse or all(re.fullmatch(r'/proc/[1-9][0-9]*/status', path) for path in paths),
            'sparse reads permitted only for PID namespace discovery')
        argv = ['docker', 'exec', 'dsv41-serve', 'cat', *paths]
        command = argv if self.host is None else [
            'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', self.host, shlex.join(argv)]
        result = subprocess.run(command, capture_output=True, text=True, errors='strict', timeout=15)
        missing = result.stderr.splitlines()
        permitted = {f'cat: {path}: No such file or directory' for path in paths} if sparse else set()
        ck.require(result.returncode == (1 if missing else 0) and set(missing) <= permitted,
            f'required /proc read failed rc{result.returncode}: {result.stderr.strip()}')
        return result.stdout


class ExperimentReads(ProductionReads):
    """Receipt reader used only by --arm cpu-pin inside the existing experiment.

    No SSH mode, directory discovery, shell or production-probe receipt reads.
    The container ID is already checked before and after the full collection.
    """

    def pin_receipt(self, pid: int, starttime: str) -> str:
        ck.require(self.host is None and 0 < pid <= proc_census.MAX_PID
            and starttime.isascii() and starttime.isdigit(), 'experiment receipt identity/path differs')
        return subprocess.check_output(['docker', 'exec', 'dsv41-serve', 'cat',
            str(rdma_pin.receipt_path(pid, starttime))], stderr=subprocess.STDOUT, text=True, errors='strict', timeout=15)


def inspect_running(text: str, container: str | None = None) -> dict[str, Json]:
    rows = ck.rows(cast(Json, json.loads(text)))
    ck.require(len(rows) == 1, 'host gate requires exactly one inspected container')
    info = rows[0]
    state = record(info['State'])
    ck.require(state.get('Running') is True and (container is None or info.get('Id') == container),
        'host gate container changed')
    return info


def production_overlay_absence(info: dict[str, Json], source: str) -> dict[str, Json]:
    """Prove current installed startup bytes/mounts, without reading receipts."""
    inputs = record(ck.read(Path(__file__).with_name('inputs.json'))['inputs'])
    expected = inputs['image:vllm_exl3_moe/rdma.py']
    actual = hashlib.sha256(source.encode()).hexdigest()
    ck.require(actual == expected, 'production RDMA source differs: startup overlay must be absent')
    targets = (Path(RDMA_SOURCE), Path(RDMA_SOURCE).with_name('up3_rdma_pin.py'), rdma_pin.RECEIPTS)
    mounts = ck.rows(info['Mounts'])
    affected: list[Json] = []
    for mount in mounts:
        destination = mount.get('Destination')
        ck.require(isinstance(destination, str) and destination.startswith('/'), 'invalid production mount destination')
        path = Path(str(destination))
        if any(path == target or path in target.parents for target in targets):
            affected.append(destination)
    ck.require(not affected, 'production has a startup-overlay/receipt mount: ' + str(affected))
    return {'status': 'ABSENT: installed rdma.py equals pinned image; no startup-overlay or receipt mounts',
        'installed_rdma_sha256': actual, 'image_rdma_sha256': expected, 'overlay_mounts': affected,
        'receipts': 'NOT READ: private experiment receipts are outside production-probe scope'}


def probe_production(rank: int, reads: ProductionReads) -> dict[str, Json]:
    """Exercise the real readers/parsers, without experiment admission or qualification."""
    ck.require(rank in (0, 1), 'probe requires rank 0 or 1')
    ck.require(reads.host is None or reads.host == ('sparkhead', 'sparkworker-wired')[rank], 'probe host/rank differs')
    info = inspect_running(reads.output(['docker', 'inspect', 'dsv41-serve']))
    config = record(info['Config'])
    ck.require(config.get('Cmd') == ['/serve.sh'] and config.get('Entrypoint') == ['/bin/bash'],
        'production probe launcher wrapper differs from audited shape')
    env: dict[str, str] = {}
    for entry in ck.strings(config['Env']):
        key, equal, value = entry.partition('=')
        ck.require(equal == '=' and key not in env, 'duplicate/malformed full container environment')
        env[key] = value
    script = reads.output(['docker', 'exec', 'dsv41-serve', 'cat', '/serve.sh'])
    port = serve_port(script)
    ck.require(port == '8000', 'production probe requires production8000; experiment qualification uses --arm')
    argv = serve_arguments(script)
    ck.require(argv.count('--node-rank') == 1 and argv[argv.index('--node-rank') + 1] == str(rank),
        'production probe launcher node-rank differs')
    overlay = production_overlay_absence(info, reads.output(['docker', 'exec', 'dsv41-serve', 'cat', RDMA_SOURCE]))
    observed = log_evidence(reads.output(['docker', 'logs', 'dsv41-serve']).splitlines())
    dual_prefix = f'EXL3 RDMA all-reduce: rank {rank} of 2 on ' + ','.join(DEVICES) + ', gid 3,'
    dual_observed = any(line.startswith(dual_prefix) for line in ck.strings(observed['rdma']))
    gids: list[Json] = []
    errors: list[Json] = []
    for device in DEVICES:
        try:
            row = gid(Path('/sys/class/infiniband'), device, read=reads.path)
            index = DEVICES.index(device)
            validate_gid(row, index)
            gids.append(row)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            errors.append({'device': device, 'error': f'{type(exc).__name__}: {exc}'})
    try:
        affinity: Json = proc_census.collect(rank, reads.proc)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        affinity = {'status': 'UNMEASURED', 'error': f'{type(exc).__name__}: {exc}'}
        errors.append({'field': 'affinity', 'error': str(record(affinity)['error'])})
    inspect_running(reads.output(['docker', 'inspect', 'dsv41-serve']), str(info['Id']))
    return {'mode': 'probe-production', 'status': 'PARSE FAILURE' if errors else 'PARSE COMPLETE',
        'qualification': 'NOT QUALIFIED: read-only production plumbing probe; no HOLD or experiment gates',
        'rank': rank, 'host': reads.host or 'local Spark', 'container': info['Id'],
        'inspect': {'status': 'PARSED', 'Cmd': config['Cmd'], 'Entrypoint': config['Entrypoint'],
            'Running': record(info['State'])['Running'], 'identity_rechecked': True},
        'serve': {'status': 'PARSED', 'port': port, 'node_rank': rank},
        'config': {key: env.get(key) for key in ('EXL3_CPU_PIN', 'EXL3_RDMA_DEVICES')},
        'logs': observed, 'gids': gids, 'errors': errors,
        'startup_overlay': overlay,
        'arm_lines': {
            'rdma_init': 'OBSERVED' if observed['rdma'] else 'ABSENT: initialization not present in production log; routing is not ring-init',
            'dual_ring': 'OBSERVED' if dual_observed else (
                'ABSENT: dual-ring configured but initialization unmeasured' if env.get('EXL3_RDMA_DEVICES') == ','.join(DEVICES)
                else 'EXPECTED ABSENT: dual-ring arm is off in production'),
            'cpu_pin': 'OBSERVED' if observed['pinned_pids'] else (
                'ABSENT: CPU-pin configured but init unmeasured' if env.get('EXL3_CPU_PIN')
                else 'EXPECTED ABSENT: CPU-pin arm is off in production'),
            'proxy': 'OBSERVED' if observed['proxy_pids'] else (
                'EXPECTED ABSENT: proxy belongs to rank 0' if rank == 1 else 'ABSENT: rank-0 proxy init unmeasured')},
        'affinity': affinity,
        'affinity_expectation': 'CPU pin is off in production: default masks expected; census does not qualify the arm',
        'limits': 'No arm correctness, collective bytes/traffic, restored identity or performance qualification'}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=CONFIG_ARMS)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--container')
    parser.add_argument('--probe-production', action='store_true')
    parser.add_argument('--host', choices=('sparkhead', 'sparkworker-wired'), help='probe over read-only SSH; omit when on the Spark')
    args = parser.parse_args(argv)
    if args.probe_production:
        if args.arm is not None or args.container is not None:
            parser.error('--probe-production cannot be combined with --arm/--container')
        result = probe_production(args.rank, ProductionReads(args.host))
        print(json.dumps(result, allow_nan=False))
        if result['status'] != 'PARSE COMPLETE':
            raise SystemExit(1)
        return
    if args.arm is None or args.container is None or args.host is not None:
        parser.error('experiment collection requires --arm and --container, without --host')
    result = collect(args.arm, args.rank, args.container)
    validate(result, args.arm, args.rank, args.container)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
