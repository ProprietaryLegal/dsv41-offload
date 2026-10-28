"""Typed protocol frame driver shared by in-process and TCP fake clients."""
from __future__ import annotations

from typing import Protocol

from ....worker import protocol as P


class Core(Protocol):
    def handle(self, frame: bytes) -> bytes | None: ...


class Client:
    def __init__(self, core: Core):
        self.core = core
        self.seq = 0

    def call(self, op: int, body: bytes = b'', session: int = 0x4e330001) -> tuple[P.AppHeader, bytes]:
        self.seq += 1
        frame = P.AppHeader(P.MAGIC,P.VERSION,op,self.seq,session,len(body),0,0).encode()+body
        reply = self.core.handle(frame)
        if reply is None:
            raise ConnectionError('worker dropped the frame')
        return P.AppHeader.decode(reply),reply[P.APP_HDR.size:]
