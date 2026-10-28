"""Coordinator-only frozen decode copy. Run on x86host; standard library only.

A receipt certifies an unchanged nonce-bound snapshot, not sequence usability.
Errored sequences and newest-writer row validation remain dataset assembly gates.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

from offload.dspark import copy_to_x86host as transfer

Record = dict[str, object]
Inventory = dict[str, Record]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def scan(root_name: str, capture: bool, nonce: str) -> Inventory:
    """Runs locally in fake hosts or as self-contained stdlib code on source host.

    Spark payloads are NEVER read through the page cache, including metadata.
    Stat signatures detect same-size/new tagged writes as well as size changes.
    """
    root = Path(root_name)
    require(root.is_dir() and not root.is_symlink(), 'source is not a real directory')
    result: Inventory = {}
    metadata: dict[str, bytes] = {}
    paths = sorted(root.rglob('*'))
    for path in paths:
        require(not path.is_symlink(), 'source symlink refused: '+str(path))
        if path.is_dir():
            continue
        require(path.is_file(), 'nonregular source refused')
        relative = str(path.relative_to(root))
        st = path.stat()
        entry: Record = {'size': st.st_size, 'mtime_ns': st.st_mtime_ns,
                         'ctime_ns': st.st_ctime_ns, 'inode': st.st_ino}
        # Payload stream hashes are acquired by the capped transfer, not an
        # uncapped second large-file read on a serving Spark.
        payload = capture and path.name in ('taps.bin', 'ids.bin')
        if not payload:
            if capture:
                data = subprocess.check_output(['dd', 'if='+str(path), 'iflag=direct',
                                                'bs=4096', 'status=none'])
                entry['sha256'] = hashlib.sha256(data).hexdigest()
                metadata[relative] = data
            else:
                with path.open('rb') as stream:
                    entry['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        result[relative] = entry
    if capture:
        flushed = json.loads(metadata['FLUSHED.json'])
        request = json.loads(metadata['FLUSH.json'])
        require(flushed['nonce'] == nonce and request['nonce'] == nonce, 'flush nonce mismatch')
        errors = [json.loads(row) for row in metadata.get('ERRORS.jsonl', b'').splitlines()]
        count = json.loads(metadata['ERROR_COUNT.json'])
        require(type(flushed['errors']) is int and flushed['errors'] == len(errors)
                and count['errors'] == len(errors), 'flush error-count mismatch')
        require(flushed.get('schema') == 'd2.decode-capture.v1', 'wrong flush schema')
        sequences = flushed['sequences']
        require(isinstance(sequences, dict) and bool(sequences), 'empty or malformed flush sequences')
        expected = {'FLUSH.json', 'FLUSHED.json', 'ERROR_COUNT.json'}
        if 'ERRORS.jsonl' in metadata:
            expected.add('ERRORS.jsonl')
        for seq, row in sequences.items():
            require(re.fullmatch(r'[A-Za-z0-9_-]+', seq) is not None, 'unsafe sequence')
            for name in ('CAPTURE.json', 'ledger.jsonl', 'taps.bin', 'ids.bin'):
                expected.add(seq+'/'+name)
            require(result[seq+'/ledger.jsonl']['sha256'] == row['ledger_sha256'],
                    'ledger changed after flush: '+seq)
            require(result[seq+'/taps.bin']['size'] == row['taps_bytes']
                    and result[seq+'/ids.bin']['size'] == row['ids_bytes'], 'flush size drift: '+seq)
            header = json.loads(metadata[seq+'/CAPTURE.json'])
            require(header.get('schema') == 'd2.decode-capture.v1' and header['seq'] == seq
                    and header['taps_source'] == 'decode_capture', 'capture identity mismatch')
        require({str(p.relative_to(root)) for p in paths if p.is_dir()} == set(sequences),
                'new or unexpected capture directory after flush')
        require(set(result) == expected, 'unexpected files/new tagged writes after flush')
    require(bool(result), 'empty source tree')
    return result


# Keep the remote scan implementation identical to the tested local function.
SCAN_CODE = ('from __future__ import annotations\nimport hashlib,json,re,subprocess,sys\n'
             'from pathlib import Path\nRecord = dict[str, object]\nInventory = dict[str, Record]\n'
             + inspect.getsource(require)+'\n'+inspect.getsource(scan)
             + '\nprint(json.dumps(scan(sys.argv[1], sys.argv[2]=="capture", sys.argv[3])))\n')


class ProductionBusy(transfer.CopyInterrupted):
    """Valid metrics prove demand; only this condition is resumable."""


def production_idle() -> None:
    with urllib.request.urlopen(transfer.METRICS, timeout=3) as response:
        body = response.read().decode()
    busy = False
    for name in ('running', 'waiting'):
        values = re.findall(r'^vllm:num_requests_'+name+r'(?:\{[^\n]*\})?\s+(\S+)\s*$', body, re.M)
        require(bool(values), 'missing production gauges; copy aborted')
        numbers = [float(value) for value in values]
        require(all(math.isfinite(value) and value >= 0 for value in numbers),
                'invalid production gauges; copy aborted')
        busy = busy or any(value > 0 for value in numbers)
    if busy:
        raise ProductionBusy('production busy; transfer paused')


def destination_memory() -> None:
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    require(available >= 16*1024**3, 'x86host has less than 16 GiB available; copy aborted')


T = TypeVar('T')


class PauseBudget:
    """Cumulative wait budget shared by every file/inventory in this invocation."""
    def __init__(self, cap_seconds: float = 4*3600, idle_seconds: float = 30,
                 check: Callable[[], None] = production_idle,
                 memory: Callable[[], None] = destination_memory,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        require(math.isfinite(cap_seconds) and cap_seconds > 0, 'positive finite wait cap required')
        require(math.isfinite(idle_seconds) and idle_seconds > 0, 'positive finite idle interval required')
        self.cap_seconds = cap_seconds
        self.idle_seconds = idle_seconds
        self.check = check
        self.memory = memory
        self.clock = clock
        self.sleep = sleep
        self.waited = 0.0
        self.events: list[Record] = []

    def wait(self) -> None:
        start = self.clock()
        stable: float | None = None
        event: Record = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        self.events.append(event)
        print('COPY PAUSED: waiting for continuous production idle', flush=True)
        try:
            while True:
                elapsed = self.clock()-start
                require(self.waited+elapsed < self.cap_seconds, 'production pause total wait cap exhausted')
                self.memory()  # memory failures remain fatal even while paused
                try:
                    self.check()
                except ProductionBusy:
                    stable = None
                else:
                    if stable is None:
                        stable = self.clock()
                    if self.clock()-stable >= self.idle_seconds:
                        print('COPY RESUMING: continuous idle verified; rechecking snapshot', flush=True)
                        return
                self.sleep(min(0.5, self.cap_seconds-self.waited-elapsed))
        finally:
            duration = self.clock()-start
            self.waited += duration
            event['wait_seconds'] = duration

    def run(self, action: Callable[[], T], validate: Callable[[], None] | None = None) -> T:
        while True:
            try:
                return action()
            except transfer.CopyInterrupted as error:
                # The transport preserves the monitor's cause. Missing metrics,
                # HTTP errors, memory and all non-demand failures are hard aborts.
                if not isinstance(error, ProductionBusy) and not isinstance(error.__cause__, ProductionBusy):
                    raise
            while True:
                self.wait()
                try:
                    if validate is not None:
                        validate()
                    break
                except ProductionBusy:
                    continue


class Remote:
    def __init__(self, ssh: list[str], capture: bool) -> None:
        self.ssh = ssh
        self.capture = capture
        self.idle_check: Callable[[], None] = transfer.idle
        self.restart_partial = False

    def inventory(self, root: str, nonce: str) -> Inventory:
        self.idle_check()  # before ANY Spark NIC traffic
        command = shlex.join(['python3', '-c', SCAN_CODE, root,
                              'capture' if self.capture else 'state', nonce])
        if not self.restart_partial:
            return cast(Inventory, json.loads(subprocess.check_output(self.ssh+[command], text=True)))
        # Inventories may scan many ledgers: monitor their traffic too. Only the
        # SSH child started here is terminated when a demand/error guard trips.
        process = subprocess.Popen(self.ssh+[command], stdout=subprocess.PIPE, text=True)
        stop = threading.Event()
        errors: list[Exception] = []

        def monitor() -> None:
            while not stop.wait(0.5):
                try:
                    self.idle_check()
                except Exception as error:
                    errors.append(error)
                    process.terminate()
                    return

        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        try:
            output, _ = process.communicate()
            if errors:
                raise errors[0]
            require(process.returncode == 0, 'inventory read failed; copy aborted')
            return cast(Inventory, json.loads(output))
        finally:
            stop.set()
            watcher.join()
            if process.poll() is None:
                process.terminate()
                process.wait()
            if process.stdout is not None:
                process.stdout.close()

    def copy(self, source: str, destination: Path, entry: Record) -> Record:
        # Reuse the reviewed rate cap, continuous production-idle and source
        # memory monitors, stream hash + independent destination re-read.
        previous = transfer.SSH
        transfer.SSH = self.ssh
        try:
            # Only this invocation's demand-interrupted file may be restarted.
            try:
                return transfer.copy(source, destination, cast(str | None, entry.get('sha256')),
                                     cast(int, entry['size']), idle_check=self.idle_check)
            except transfer.CopyInterrupted as error:
                if self.restart_partial and (isinstance(error, ProductionBusy)
                                             or isinstance(error.__cause__, ProductionBusy)):
                    destination.with_name(destination.name+'.s2-partial').unlink(missing_ok=True)
                raise
        finally:
            transfer.SSH = previous


def frozen_copy(capture: Remote, state: Remote, source: str, generator: str,
                destination: Path, nonce: str, top20: str | None = None,
                reread: Callable[[Path], str] = transfer.sha_file,
                pause: PauseBudget | None = None) -> Record:
    started = time.monotonic()
    require(not destination.exists(), 'destination exists; NEW tree required')
    require(bool(re.fullmatch(r'[A-Za-z0-9_-]{1,120}', nonce)), 'unsafe nonce')
    groups = [('capture', capture, source), ('generator', state, generator)]
    if top20 is not None:
        groups.append(('top20', state, top20))
    budget = pause or PauseBudget()
    for _, remote, _ in groups:
        remote.idle_check = budget.check
        remote.restart_partial = True
    before: dict[str, Inventory] = {}

    def validate_snapshot() -> None:
        for name, remote, root in groups:
            if name in before:
                require(remote.inventory(root, nonce) == before[name],
                        'snapshot changed during pause: '+name+'; copied tree INVALID')

    for name, remote, root in groups:
        before[name] = budget.run(lambda: remote.inventory(root, nonce), validate_snapshot)
    os.umask(0o077)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipts: list[Record] = []
    for name, remote, root in groups:
        for relative, entry in before[name].items():
            require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                    'unsafe inventory path')
            target = destination/name/relative
            file_receipt = budget.run(lambda: remote.copy(root+'/'+relative, target, entry), validate_snapshot)
            require(target.stat().st_size == entry['size'], 'destination size mismatch')
            require(reread(target) == file_receipt['sha256'], 'destination re-read hash mismatch')
            receipts.append({'tree': name, 'relative': relative, **file_receipt})
    for name, remote, root in groups:
        require(budget.run(lambda: remote.inventory(root, nonce), validate_snapshot) == before[name],
                'snapshot changed during copy: '+name+'; copied tree INVALID')
    receipt: Record = {'schema': 'd2.decode-copy.v1', 'nonce': nonce, 'frozen': True,
                       'source': source, 'generator': generator, 'top20': top20,
                       'inventories': before, 'files': receipts,
                       'elapsed_seconds_including_snapshots_and_verification': time.monotonic()-started,
                       'rate_cap_bytes_per_second': transfer.RATE,
                       'pause_policy': {'continuous_idle_seconds': budget.idle_seconds,
                                        'total_wait_cap_seconds': budget.cap_seconds,
                                        'total_wait_seconds': budget.waited, 'events': budget.events,
                                        'partial_file_policy': 'restart interrupted file from byte zero'},
                       'assembly_required': 'newest-writer row hashes, committed IDs, error attribution'}
    # Marker written LAST; failed partial trees can never pass as a completed copy.
    with (destination/'COPY_RECEIPT.json').open('x') as output:
        json.dump(receipt, output, indent=2)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    return receipt


FLUSH_CODE = r'''
import json,os,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); nonce=sys.argv[2]
for control in ('FLUSH.json', 'FLUSHED.json'):
    if (p/control).exists() and json.loads((p/control).read_text()).get('nonce') == nonce:
        raise SystemExit('nonce already requested/acknowledged; choose a NEW nonce')
with (p/'FLUSH.json.tmp').open('x') as f:
    json.dump({'nonce':nonce}, f); f.flush(); os.fsync(f.fileno())
(p/'FLUSH.json.tmp').replace(p/'FLUSH.json')
deadline=time.monotonic()+30
while time.monotonic()<deadline:
    if (p/'FLUSHED.json').exists() and json.loads((p/'FLUSHED.json').read_text()).get('nonce')==nonce:
        break
    time.sleep(.2)
else: raise SystemExit('capture flush timed out')
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--nonce', required=True)
    parser.add_argument('--flush', action='store_true')
    parser.add_argument('--run')
    parser.add_argument('--generator-root')
    parser.add_argument('--top20-root')
    parser.add_argument('--pause-wait-cap-seconds', type=float, default=4*3600)
    args = parser.parse_args()
    require(os.environ.get('D2_COORDINATOR') == 'Deepseek/coordinator', 'coordinator execution only')
    require(bool(re.fullmatch(r'[A-Za-z0-9_-]{1,120}', args.nonce)), 'unsafe nonce')
    capture = Remote(transfer.SSH, True)
    if args.flush:
        transfer.idle()
        subprocess.run(transfer.SSH+[shlex.join(['python3', '-c', FLUSH_CODE,
                                               args.source_root, args.nonce])], check=True)
        capture.inventory(args.source_root, args.nonce)
        print('FLUSH VERIFIED '+args.nonce)
        return
    require(bool(args.run and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', args.run)), 'safe run required')
    require(bool(args.generator_root), 'frozen generator path required')
    state = Remote(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', 'controller'], False)
    destination = Path('/opt/dsv41/x86host-data/d2')/('decode-'+args.run)
    frozen_copy(capture, state, args.source_root, args.generator_root, destination, args.nonce, args.top20_root,
                pause=PauseBudget(cap_seconds=args.pause_wait_cap_seconds))
    print('COPY VERIFIED '+str(destination/'COPY_RECEIPT.json'))


if __name__ == '__main__':
    main()
