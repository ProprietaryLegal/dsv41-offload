# SPDX-License-Identifier: Apache-2.0
"""Blocking request/response transports for the remote DSpark proxy.

API (the sketch in the O3 brief, to be satisfied by lane O1's library as well):

    client.connect(timeout_s)                   # raises TransportError
    n = client.request(msg, resp_buf, timeout_s) # msg: one contiguous buffer
                                                 # (AppHeader + body); the reply frame is
                                                 # written into resp_buf[:n]
    n = client.request_parts(parts, resp_buf, timeout_s)
                                                 # the message is the concatenation of
                                                 # parts (contiguous buffers, in order),
                                                 # sent without being joined in Python
    client.close()

Errors are explicit: TransportTimeout (deadline passed; the connection is then unusable
and must be closed) and TransportError (anything else). Nothing is retried here; the
proxy decides (CONTRACT 7.4/7.5).

TCP: each message is ``u32 length`` + ``length`` bytes, TCP_NODELAY, explicit local bind
(the direct-link address), no listener on the Spark side.
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

_LEN = struct.Struct("<I")
MAX_FRAME = 64 << 20


class TransportError(RuntimeError):
    pass


class TransportTimeout(TransportError):
    pass


@dataclass
class Endpoint:
    scheme: str            # "tcp" | "o1tcp" | "o1rdma"
    host: str
    port: int
    options: dict[str, str]

    def __str__(self) -> str:
        q = "&".join(f"{k}={v}" for k, v in sorted(self.options.items()))
        return f"{self.scheme}://{self.host}:{self.port}" + (f"?{q}" if q else "")


def parse_endpoint(text: str) -> Endpoint:
    """``tcp://10.202.28.1:47100[?bind=10.202.28.2]`` (this module's TCP stub), or
    ``o1tcp://...`` / ``o1rdma://...?dev=rocep1s0f0&gid=3`` (lane O1's library)."""
    u = urlparse(text.strip())
    if u.scheme not in ("tcp", "o1tcp", "o1rdma"):
        raise ValueError(f"endpoint {text!r}: scheme must be tcp, o1tcp or o1rdma")
    if not u.hostname or not u.port:
        raise ValueError(f"endpoint {text!r}: host and port required")
    opts = {k: v[-1] for k, v in parse_qs(u.query).items()}
    return Endpoint(u.scheme, u.hostname, u.port, opts)


class RpcClient:
    def connect(self, timeout_s: float) -> None:
        raise NotImplementedError

    def request(self, msg, resp_buf: bytearray, timeout_s: float) -> int:
        raise NotImplementedError

    def request_parts(self, parts, resp_buf: bytearray, timeout_s: float) -> int:
        """request() of the concatenation of parts. The round sends its STEP as [headers from a private buffer, rows
        from the gather staging] so that it never writes into memory the engine stream may be refilling (S11 P-1);
        TcpClient and O1Client override this without the join below."""
        return self.request(bytearray().join(memoryview(p).cast("B") for p in parts), resp_buf, timeout_s)

    def close(self) -> None:
        raise NotImplementedError

    @property
    def connected(self) -> bool:
        raise NotImplementedError


class TcpClient(RpcClient):
    def __init__(self, host: str, port: int, bind: str | None = None):
        self.host, self.port, self.bind = host, port, bind
        self._sock: socket.socket | None = None
        self._lenbuf = bytearray(4)

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self, timeout_s: float) -> None:
        self.close()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if self.bind:
                s.bind((self.bind, 0))
            s.settimeout(timeout_s)
            s.connect((self.host, self.port))
        except socket.timeout as e:
            s.close()
            raise TransportTimeout(f"connect {self.host}:{self.port}: timeout") from e
        except OSError as e:
            s.close()
            raise TransportError(f"connect {self.host}:{self.port}: {e}") from e
        self._sock = s

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def request(self, msg, resp_buf: bytearray, timeout_s: float) -> int:
        return self.request_parts((msg,), resp_buf, timeout_s)

    def request_parts(self, parts, resp_buf: bytearray, timeout_s: float) -> int:
        s = self._sock
        if s is None:
            raise TransportError("not connected")
        deadline = time.monotonic() + timeout_s
        views = [memoryview(p).cast("B") for p in parts]
        try:
            s.settimeout(_remaining(deadline))
            self._send_all(s, views, deadline)
            n = self._recv_frame(s, resp_buf, deadline)
        except socket.timeout as e:
            self.close()
            raise TransportTimeout(f"request to {self.host}:{self.port}: deadline") from e
        except TransportTimeout:
            self.close()
            raise
        except OSError as e:
            self.close()
            raise TransportError(f"request to {self.host}:{self.port}: {e}") from e
        return n

    def _send_all(self, s: socket.socket, views: list[memoryview], deadline: float) -> None:
        """The length prefix and the parts as one frame (one sendmsg per partial write, no join)."""
        bufs = [memoryview(_LEN.pack(sum(len(v) for v in views)))] + [v for v in views if len(v)]
        sent = s.sendmsg(bufs)
        while True:
            while bufs and sent >= len(bufs[0]):
                sent -= len(bufs[0])
                bufs.pop(0)
            if not bufs:
                return
            if sent:
                bufs[0] = bufs[0][sent:]
            s.settimeout(_remaining(deadline))
            sent = s.sendmsg(bufs)

    def _recv_frame(self, s: socket.socket, resp_buf: bytearray, deadline: float) -> int:
        _recv_exact(s, memoryview(self._lenbuf), deadline)
        (n,) = _LEN.unpack(self._lenbuf)
        if n > MAX_FRAME:
            raise TransportError(f"reply frame of {n} bytes exceeds {MAX_FRAME}")
        if n > len(resp_buf):
            resp_buf.extend(bytes(n - len(resp_buf)))
        _recv_exact(s, memoryview(resp_buf)[:n], deadline)
        return n


def _remaining(deadline: float) -> float:
    r = deadline - time.monotonic()
    if r <= 0:
        raise TransportTimeout("deadline passed")
    return r


def _recv_exact(s: socket.socket, mv: memoryview, deadline: float) -> None:
    got = 0
    while got < len(mv):
        s.settimeout(_remaining(deadline))
        k = s.recv_into(mv[got:])
        if k == 0:
            raise TransportError("connection closed by peer")
        got += k


# ------------------------------------------------------------------ server side helpers


def recv_frame_blocking(conn: socket.socket, buf: bytearray) -> int | None:
    """Server helper: read one length-prefixed frame into buf; None on clean EOF."""
    lb = bytearray(4)
    got = 0
    while got < 4:
        k = conn.recv_into(memoryview(lb)[got:])
        if k == 0:
            if got == 0:
                return None
            raise TransportError("connection closed inside a frame header")
        got += k
    (n,) = _LEN.unpack(lb)
    if n > MAX_FRAME:
        raise TransportError(f"frame of {n} bytes exceeds {MAX_FRAME}")
    if n > len(buf):
        buf.extend(bytes(n - len(buf)))
    mv = memoryview(buf)[:n]
    got = 0
    while got < n:
        k = conn.recv_into(mv[got:])
        if k == 0:
            raise TransportError("connection closed inside a frame")
        got += k
    return n


def send_frame_blocking(conn: socket.socket, payload) -> None:
    mv = memoryview(payload).cast("B")
    conn.sendall(_LEN.pack(len(mv)))
    conn.sendall(mv)


def make_client(ep: Endpoint) -> RpcClient:
    if ep.scheme == "tcp":
        return TcpClient(ep.host, ep.port, bind=ep.options.get("bind"))
    # Lane O1's library (offload/transport/API.md). Imported lazily so the TCP stub has
    # no dependency on it; see o1_adapter.py.
    from .o1_adapter import O1Client

    return O1Client(ep)
