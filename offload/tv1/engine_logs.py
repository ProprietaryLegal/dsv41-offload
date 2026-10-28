"""Read-only, bounded experiment log preservation before Docker removal.

The emitted program uses stdlib only and never imports the engine or CUDA. Docker
logs are read by immutable ID after checking the experiment-only profile mount.
No production log or container environment is exported.
"""
from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import re
import selectors
import subprocess
import time
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, record

LIMIT = 8 * 1024**2  # first/last 4MiB if larger; full stream still hashed
SECONDS = 20.
TYPE = 'tv1-engine-logs-v1'


def _remote(limit: int, seconds: float) -> None:
    """Standalone remote body; keep imports in script() synchronized."""
    packet: dict[str, object] = {'type': 'tv1-engine-logs-v1', 'name': 'dsv41-serve'}
    try:
        query = subprocess.run(['docker', 'inspect', '--format',
            '{{.Id}}\n{{json .State}}\n{{json .Mounts}}', 'dsv41-serve'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, check=True, text=True)
        identity, state, mounts = query.stdout.splitlines()
        if re.fullmatch('[0-9a-f]{64}', identity) is None:
            raise ValueError('invalid experiment container ID')
        if not any(m.get('Destination') == '/sp1-profile' for m in json.loads(mounts)):
            raise ValueError('named container lacks experiment-only profile mount; refuse production logs')
        packet.update(container_id=identity, state=json.loads(state))
        proc = subprocess.Popen(['docker', 'logs', '--timestamps', identity],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        head, tail = bytearray(), bytearray()
        digest = hashlib.sha256()
        total = 0
        half = limit // 2
        try:
            assert proc.stdout is not None
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                end = time.monotonic() + seconds
                while True:
                    left = end - time.monotonic()
                    if left <= 0 or not selector.select(left):
                        raise TimeoutError('bounded docker log read expired')
                    chunk = os.read(proc.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    digest.update(chunk)
                    take = min(half - len(head), len(chunk))
                    head.extend(chunk[:take])
                    tail.extend(chunk[take:])
                    if len(tail) > half:
                        del tail[:-half]
                rc = proc.wait(timeout=max(.01, end - time.monotonic()))
        except Exception as exc:
            packet['error'] = f'{type(exc).__name__}: {exc}'
            rc = -1
        finally:
            if proc.poll() is None:
                proc.terminate()  # this program's read-only Docker client only
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1)
            if proc.stdout is not None:
                proc.stdout.close()
        data = bytes(head + tail)
        packet.update(returncode=rc, total_bytes=total, stored_bytes=len(data),
            stream_sha256=digest.hexdigest(), stored_sha256=hashlib.sha256(data).hexdigest(),
            truncated=total > limit, stream_complete=rc == 0 and 'error' not in packet,
            complete=rc == 0 and total <= limit and 'error' not in packet,
            data_b64=base64.b64encode(data).decode('ascii'))
        if rc and 'error' not in packet:
            packet['error'] = f'docker logs exited {rc}'
    except Exception as exc:
        packet['error'] = f'{type(exc).__name__}: {exc}'
    print(json.dumps(packet))


def script(*, limit: int = LIMIT, seconds: float = SECONDS) -> str:
    if limit < 2 or limit % 2 or seconds <= 0:
        raise ValueError('invalid TV1 log bounds')
    return ('from __future__ import annotations\n'
            'import base64,hashlib,json,os,re,selectors,subprocess,time\n'
            + inspect.getsource(_remote) + f'\n_remote({limit!r}, {seconds!r})\n')


def save(text: str, output: Path) -> dict[str, Json]:
    """Preserve bounded bytes and receipt, including explicit incomplete evidence."""
    packet = record(cast(Json, json.loads(text)))
    if packet.get('type') != TYPE or packet.get('name') != 'dsv41-serve':
        raise ValueError('TV1 engine log receipt identity mismatch')
    payload = packet.pop('data_b64', '')
    if not isinstance(payload, str):
        raise ValueError('TV1 log payload is not base64 text')
    data = base64.b64decode(payload, validate=True)
    if len(data) > LIMIT or (payload and (
            packet.get('stored_bytes') != len(data)
            or packet.get('stored_sha256') != hashlib.sha256(data).hexdigest())):
        raise ValueError('TV1 bounded engine log bytes/hash mismatch')
    if packet.get('complete') is True and (
            packet.get('returncode') != 0 or packet.get('truncated') is not False
            or packet.get('total_bytes') != len(data)
            or packet.get('stream_sha256') != hashlib.sha256(data).hexdigest()):
        raise ValueError('TV1 falsely complete engine log receipt')
    with output.open('xb') as handle:
        handle.write(data)
    packet['file'] = output.name
    with output.with_suffix('.json').open('x') as handle:
        json.dump(packet, handle, indent=2)
        handle.write('\n')
    return packet
