"""Deterministic synthetic taps using observed public worker STEP shapes.

No captured tensors or text. B1/two rows then B4/28 rows are DIAG2's recorded
transition; B2/B3 five-row increments, resets and fresh connections extend it.
All wire checks use the staged worker namespace and production sparse-q codec.
"""
from __future__ import annotations

import faulthandler
import hashlib
import importlib
import json
import os
import platform
import signal
import struct
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    faulthandler.dump_traceback_later(10., repeat=True, exit=False)

import numpy as np

BATCHES = (1, 4, 2, 3)
MODES = ('positive', 'zero', 'mixed')
ROUNDS = 10
SESSION = 0x4104


def save(path: Path, value: object) -> None:
    tmp = path.with_suffix('.new')
    tmp.write_text(json.dumps(value, allow_nan=False)+'\n')
    tmp.replace(path)


def bound(out: Path, phase: str, seconds: float = 30.0) -> None:
    save(out/'bound.json', {'phase': phase, 'until': time.monotonic()+seconds})


def signature(*values: Any) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def masks(batch: int, mode: str) -> list[int]:
    return [int(mode == 'positive' or (mode == 'mixed' and slot % 2 == 0)) for slot in range(batch)]


def rows(first: int, count: int, slot: int, row_bytes: int) -> bytes:
    # Finite synthetic BF16; each position's taps are independent of batching,
    # window chunks, process and transport. No private model state is persisted.
    values = []
    for pos in range(first, first+count):
        rng = np.random.default_rng(4104000+pos+slot*1000000)
        fp32 = rng.standard_normal(row_bytes//2).astype('<f4') * .1
        values.append((fp32.view('<u4') >> 16).astype('<u2').tobytes())
    return b''.join(values)


def encode_step(step_id: int, entries: list[Any], mask: list[int], row_bytes: int) -> bytes:
    p = importlib.import_module('worker.protocol')
    batch = len(entries)
    end = p.STEP_HDR.size + batch*(p.REQ_ENTRY.size+p.SAMPLE_META.size)
    offset = (end+63)//64*64
    header = p.STEP_HDR.pack(step_id, batch, p.STEP_SAMPLED, sum(e.n_rows for e in entries), row_bytes, offset)
    body = header + b''.join(p.REQ_ENTRY.pack(e.slot, e.req_flags, e.k_window, e.n_out, e.generation,
        e.first_pos, e.n_rows, e.anchor_token, e.anchor_pos) for e in entries)
    body += b''.join(p.SAMPLE_META.pack(4103+e.slot, bit) for e, bit in zip(entries, mask, strict=True))
    body += bytes(offset-len(body))
    body += b''.join(rows(e.first_pos, e.n_rows, e.slot, row_bytes) for e in entries)
    return bytes(body)


def step_body(batch: int, iteration: int, mode: str, row_bytes: int, depth: int) -> bytes:
    p = importlib.import_module('worker.protocol')
    increment = 2 if batch == 1 else 7 if batch == 4 else 5
    anchor = 128 + increment*iteration
    count = 128 if iteration == 0 else increment
    first = anchor-count
    entries = [p.ReqEntry(slot, p.REQ_NEED_DRAFT | (p.REQ_NEW_GEN if iteration == 0 else 0),
                         depth, 1, first, count, 100+slot+iteration, anchor, n_out=depth) for slot in range(batch)]
    return encode_step(iteration+1, entries, masks(batch, mode), row_bytes)


def cold_bodies(row_bytes: int, depth: int) -> tuple[bytes, bytes, bytes]:
    """Window-only setup, then B1/two rows -> B4/28 rows without resync.

    Slot0 advances by two before widening; its next seven rows must start at
    130, while slots1..3 still start at128. All retain generation1/full windows.
    """
    p = importlib.import_module('worker.protocol')
    setup = [p.ReqEntry(slot, p.REQ_NEW_GEN, depth, 1, 0, 128, 100+slot, 128, n_out=depth)
             for slot in range(4)]
    b1 = [p.ReqEntry(0, p.REQ_NEED_DRAFT, depth, 1, 128, 2, 101, 130, n_out=depth)]
    b4 = [p.ReqEntry(slot, p.REQ_NEED_DRAFT, depth, 1, 130 if slot == 0 else 128, 7,
                     102+slot, 137 if slot == 0 else 135, n_out=depth) for slot in range(4)]
    return (encode_step(1, setup, [0]*4, row_bytes), encode_step(2, b1, [1], row_bytes),
            encode_step(3, b4, [1]*4, row_bytes))


def checked_reply(frame: bytes, op: int, seq: int, session: int) -> bytes:
    p = importlib.import_module('worker.protocol')
    header = p.AppHeader.decode(frame)
    if (header.op, header.seq, header.session, header.status, header.body_len) != (op, seq, session, p.OK, len(frame)-32):
        raise RuntimeError('frame identity/status failure')
    return frame[32:]


def step_signature(body: bytes, batch: int, mode: str, depth: int, row_bytes: int) -> str:
    p = importlib.import_module('worker.protocol')
    from offload.dspark.spark.remote_dspark import sampled_wire as qwire
    end = p.REPLY_HDR.size + batch*p.REQ_STATUS.size + batch*depth*8
    original = bytearray(body[:end])
    struct.pack_into('<I', original, 12, struct.unpack_from('<I', original, 12)[0] & ~p.REPLY_SAMPLED)
    reply = p.decode_step_reply(original)
    if reply.n_req != batch or reply.num_spec != depth or any(st[0] != p.R_OK for st in reply.statuses):
        raise RuntimeError('STEP request failed or partial')
    mask, ids, q = qwire.unpack_q(body, end, batch, depth)
    qwire.validate(mask, ids, q, reply.draft_ids, np.ones(batch, dtype=np.bool_), 129280,
                   np.asarray(masks(batch, mode), dtype=np.int32))
    return signature(reply.draft_ids, reply.conf, mask, ids, q)


def exercise(call: Callable[[bytes], bytes], hello: Any, out: Path, transport: str, connection: int) -> list[str]:
    p = importlib.import_module('worker.protocol')
    seq = 0
    session = SESSION+connection
    timings: list[dict[str, object]] = []
    signatures = []
    def request(op: int, body: bytes, phase: str) -> bytes:
        nonlocal seq
        seq += 1
        frame = p.AppHeader(p.MAGIC, p.VERSION, op, seq, session, len(body), 0, 0).encode()+body
        bound(out, phase)
        started = time.perf_counter_ns()
        reply = call(frame)
        elapsed = time.perf_counter_ns()-started
        if elapsed >= 30_000_000_000:
            raise TimeoutError('frame exceeded existing 30s freshness bound')
        parsed = checked_reply(reply, op, seq, session)
        timings.append({'phase': phase, 'elapsed_ns': elapsed, 'request_bytes': len(frame), 'reply_bytes': len(reply),
                        'pid': os.getpid(), 'python_thread': threading.get_ident(), 'native_thread': threading.get_native_id()})
        save(out/f'{transport}-{connection}-timings.json', timings)
        return parsed
    request(p.OP_HELLO, hello.encode(), f'{transport}/connection{connection}/HELLO')
    # Setup performs insertion only, with its own unchanged 30s frame bound.
    # The first two draft calls reproduce DIAG2 before any warm matrix/reset.
    setup, b1, b4 = cold_bodies(hello.row_bytes, hello.num_spec)
    for body, batch, mode, phase in ((setup, 4, 'zero', 'window-only-setup'),
                                    (b1, 1, 'positive', 'B1-two-rows'), (b4, 4, 'positive', 'B4-28-rows')):
        answer = request(p.OP_STEP, body, f'{transport}/connection{connection}/cold/{phase}')
        signatures.append(step_signature(answer, batch, mode, hello.num_spec, hello.row_bytes))
    # Preserve the repeated width/mask matrix and reset coverage afterwards.
    for mode in MODES:
        for batch in BATCHES:
            request(p.OP_RESET, struct.pack('<I', 15), f'{transport}/{mode}/B{batch}/RESET')
            for iteration in range(ROUNDS):
                body = step_body(batch, iteration, mode, hello.row_bytes, hello.num_spec)
                answer = request(p.OP_STEP, body, f'{transport}/{mode}/B{batch}/round{iteration}')
                signatures.append(step_signature(answer, batch, mode, hello.num_spec, hello.row_bytes))
    request(p.OP_RESET, struct.pack('<I', 15), f'{transport}/final-RESET')
    request(p.OP_PING, b'', f'{transport}/final-PING')
    return signatures


def direct(core: Any, out: Path) -> list[str]:
    def call(frame: bytes) -> bytes:
        reply = core.handle(frame)
        if not isinstance(reply, bytes):
            raise TypeError('missing direct reply')
        return reply
    return [sig for connection in (0, 1) for sig in exercise(call, core.expected, out, 'direct', connection)]


def client(out: Path, port: int) -> dict[str, object]:
    p = importlib.import_module('worker.protocol')
    ofl = importlib.import_module('worker.server')._load_ofl()
    if os.environ.get('PP4_SERVER_RUNTIME'):
        runtime = json.loads(Path(os.environ['PP4_SERVER_RUNTIME']).read_text())
        for label, actual in (('ofl_python', Path(ofl.__file__)), ('ofl', Path(ofl.lib()._name))):
            row = runtime['files'][label]
            if actual.resolve() != Path(row['path']).resolve() or hashlib.sha256(actual.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('loaded OFL runtime differs: '+label)
    hello = p.Hello(**json.loads((out/'hello.json').read_text()))
    hello.drafter_identity = bytes.fromhex(str(hello.drafter_identity))
    hello.markov_identity = bytes.fromhex(str(hello.markov_identity))
    signatures: list[str] = []
    for connection in (0, 1):
        bound(out, 'OFL connection/reconnect')
        oc = ofl.Client(ofl.BACKEND_TCP, '10.202.28.1', port, '10.202.28.1', service='dspark-v1',
                        max_req=16 << 20, max_resp=64 << 10, inflight=1)
        response = bytearray(64 << 10)
        region = oc.reg(response)
        try:
            def call(frame: bytes, oc: Any = oc, region: Any = region, response: bytearray = response) -> bytes:
                seq = p.AppHeader.decode(frame).seq
                result = oc.call(p.AppHeader.decode(frame).op, frame, [(region, 0, len(response))],
                                 timeout_s=30., generation=seq)
                return bytes(response[:result.resp_len])
            signatures.extend(exercise(call, hello, out, 'ofl', connection))
        finally:
            oc.close()
    return {'signatures': signatures, 'complete': True, 'connections': 2, 'transport': 'real OflServer TCP on X86host'}


def hello_record(hello: Any) -> dict[str, Any]:
    data = asdict(hello)
    for key in ('drafter_identity', 'markov_identity'):
        data[key] = data[key].hex()
    return data


if __name__ == '__main__':
    if (os.environ.get('WINDOW_COORDINATOR') != 'Deepseek/coordinator'
            or os.environ.get('PP4_SERVER_PROBE') != '1'
            or sys.executable != '/home/operator/miniconda3/bin/python3'
            or platform.node().split('.')[0].lower() not in ('x86host', 'x86host')):
        raise ValueError('coordinator-only X86host-local client')
    probe_out = Path(os.environ['PP4_SERVER_OUT'])
    if not probe_out.resolve().is_relative_to('/opt/dsv41/x86host-data/pp4'):
        raise ValueError('client output outside PP4 scratch')
    port = int(os.environ['PP4_SERVER_PORT'])
    if not 18640 <= port <= 18699:
        raise ValueError('allocated diagnostic port required')
    save(probe_out/'ofl.json', client(probe_out, port))
