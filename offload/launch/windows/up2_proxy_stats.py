"""Coordinator experiment-only stats flush; emits no model data and imports no engine.

Uses the existing proxy housekeeping FLUSH_REQ/ACK control words. No tensor bytes
are read or written. Never use this against production or an unreviewed layout.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import time

PAGE = 65536
MAGIC = 0x31565344
FLUSH_REQ = 448
FLUSH_ACK = 512


def flush(fd: int, *, timeout: float = 2) -> None:
    def word(offset: int) -> int:
        return int(struct.unpack('<I',os.pread(fd,4,PAGE+offset))[0])
    if word(0) != MAGIC or word(4) != 1 or word(192) != 1 or word(256) != 0:
        raise ValueError('proxy control layout/state mismatch')
    request = (word(FLUSH_REQ)+1) & 0xffffffff
    # The engine is idle and this runner is the sole coordinator/flush requester.
    os.pwrite(fd,struct.pack('<I',request),PAGE+FLUSH_REQ)
    end = time.monotonic()+timeout
    while word(FLUSH_ACK) != request:
        if time.monotonic() >= end:
            raise TimeoutError('proxy stats flush ACK not received')
        time.sleep(.01)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pid',type=int,required=True)
    args = ap.parse_args()
    process = Path('/proc')/str(args.pid)
    maps = (process/'maps').read_text()
    if any(name in maps for name in ('libtorch','libc10','libcuda','libcudart')):
        raise ValueError('proxy loaded torch/CUDA library')
    candidates = [p for p in (process/'fd').iterdir() if os.readlink(p).startswith('/memfd:dsv41-remote-dspark')]
    if not candidates or len({os.stat(p).st_ino for p in candidates}) != 1:
        raise ValueError('expected one distinct proxy memfd (duplicate handles allowed)')
    fd = os.open(candidates[0],os.O_RDWR)
    try:
        if struct.unpack('<I',os.pread(fd,4,PAGE+320))[0] != args.pid:
            raise ValueError('proxy PID/control mismatch')
        flush(fd)
    finally:
        os.close(fd)
    data = json.loads(Path('/tmp/dsv41-remote-dspark.json').read_text())
    if data.get('async_proxy_process',{}).get('pid') != args.pid:
        raise ValueError('stats belong to another proxy')
    print(json.dumps(data,allow_nan=False))


if __name__ == '__main__':
    main()
