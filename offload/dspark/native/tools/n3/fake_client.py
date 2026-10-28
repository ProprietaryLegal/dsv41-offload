"""Local TCP-stub protocol soak; CPU only, never contacts a Spark."""
from __future__ import annotations

import argparse
import json
import socket
import struct
from pathlib import Path

import numpy as np
import torch

from ....worker.server import expected_hello
from .soak import rounds


class RemoteCore:
    def __init__(self, sock: socket.socket, drafter: str, markov: str):
        self.sock = sock
        self.expected = expected_hello(bytes.fromhex(drafter), bytes.fromhex(markov), max_model_len=1048576)

    def _recv(self, n: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < n:
            b = self.sock.recv(n-len(chunks))
            if not b:
                raise ConnectionError('worker disconnected')
            chunks.extend(b)
        return bytes(chunks)

    def handle(self, frame: bytes) -> bytes:
        self.sock.sendall(struct.pack('<I', len(frame))+frame)
        (n,) = struct.unpack('<I', self._recv(4))
        if n > 65536:
            raise RuntimeError(f'oversized worker reply: {n}')
        return self._recv(n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--address', default='127.0.0.1:18639')
    ap.add_argument('--drafter', required=True)
    ap.add_argument('--markov', required=True)
    ap.add_argument('--rounds', type=int, default=2000)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    host, port = a.address.rsplit(':', 1)
    if host not in ('127.0.0.1', '10.202.28.1', '10.202.29.1'):
        raise ValueError('fake client is local to x86host only')
    if a.rounds < 2000:
        raise ValueError('at least 2000 rounds required')
    # Synthetic, fixed CPU inputs; never initializes CUDA in this client.
    rng = np.random.default_rng(7)
    tokens = rng.integers(0, 120000, size=1600)
    taps = torch.from_numpy(rng.normal(0, 100, size=(1600, 15360)).astype(np.float32))
    raw = taps.to(torch.bfloat16).view(torch.uint8).numpy()
    with socket.create_connection((host, int(port)), timeout=30) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        rep = rounds(RemoteCore(sock, a.drafter, a.markov), raw, tokens, a.rounds, 0x4e330003)
    a.out.write_text(json.dumps(rep, indent=2)+'\n')
    print('DONE local fake client', rep['rounds'], rep['digest'], flush=True)


if __name__ == '__main__':
    main()
