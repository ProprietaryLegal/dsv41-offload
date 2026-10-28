# SPDX-License-Identifier: Apache-2.0
"""Adapter to lane O1's transport ``ofl`` (offload/transport/API.md, commit 41cf2f7+).

Endpoint forms (``bind`` = this side's direct-link IPv4; it also selects the RDMA device
and GID, API.md "Selecting device / port / GID"):

    o1rdma://10.202.28.1:18620?bind=10.202.28.2      RDMA RC (the fast path)
    o1tcp://10.202.28.1:18620?bind=10.202.28.2       ofl's TCP backend

Mapping onto the contract (CONTRACT 5): one contract message = one ofl request.
  ofl op         = AppHeader.op (1..5; ofl's op 0 is its built-in echo)
  ofl generation = AppHeader.seq (echoed by ofl; the contract's own echo is checked too)
  ofl payload    = the whole contract message (AppHeader + body)
  ofl response   = the whole contract reply frame, RDMA-written into the registered
                   reply buffer (one range)
The ofl library and its ctypes binding are loaded from VLLM_DSPARK_REMOTE_OFL_DIR (a
directory holding libofl.so and ofl.py, bind-mounted into the container), else from
OFL_LIB's directory. Timeouts break the ofl connection by design (the QP goes to error
before the call returns, so late data never lands); the proxy then reconnects.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import struct
import sys

from .transport import Endpoint, RpcClient, TransportError, TransportTimeout

SERVICE = "dspark-v1"
DEFAULT_MAX_REQ = 16 << 20     # >= 32 + 192 + 512 rows x 30,720 B = 15.73 MB (CONTRACT 8.1)
DEFAULT_MAX_RESP = 64 << 10    # STEP replies are < 1 KiB; STATS JSON fits
_OP_SEQ = struct.Struct("<HQ")  # AppHeader.op (offset 6) and .seq (offset 8)

_ofl = None


def load_ofl():
    """Import O1's ofl.py (and through it libofl.so) once."""
    global _ofl
    if _ofl is not None:
        return _ofl
    d = os.environ.get("VLLM_DSPARK_REMOTE_OFL_DIR") or ""
    if not d and os.environ.get("OFL_LIB"):
        d = os.path.dirname(os.environ["OFL_LIB"])
    if not d:
        raise TransportError(
            "o1 endpoint: set VLLM_DSPARK_REMOTE_OFL_DIR to the directory with libofl.so "
            "and ofl.py (lane O1, offload/transport)")
    py = os.path.join(d, "ofl.py")
    so = os.path.join(d, "libofl.so")
    if not os.path.exists(py) or not os.path.exists(so):
        raise TransportError(f"o1 endpoint: {py} or {so} missing")
    os.environ.setdefault("OFL_LIB", so)
    spec = importlib.util.spec_from_file_location("dsv41_ofl", py)
    if spec is None or spec.loader is None:
        raise TransportError(f"cannot import {py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dsv41_ofl"] = mod
    spec.loader.exec_module(mod)
    mod.lib()  # load + declare now: fail at startup, not on the first round
    _ofl = mod
    return mod


def _addr(buf) -> tuple[int, int]:
    """Address and length of a contiguous writable buffer (bytearray, numpy, memoryview)."""
    if hasattr(buf, "ctypes") and hasattr(buf, "nbytes"):
        if not buf.flags.c_contiguous:
            raise TransportError("buffer must be contiguous")
        return int(buf.ctypes.data), int(buf.nbytes)
    mv = memoryview(buf)
    if mv.readonly:
        raise TransportError("buffer must be writable (ctypes needs its address)")
    if not mv.contiguous:
        raise TransportError("buffer must be contiguous")
    c = (ctypes.c_char * mv.nbytes).from_buffer(mv.cast("B"))
    return ctypes.addressof(c), mv.nbytes


class O1Client(RpcClient):
    def __init__(self, ep: Endpoint):
        if ep.scheme not in ("o1tcp", "o1rdma"):
            raise TransportError(f"O1Client cannot serve {ep.scheme}://")
        self.ofl = load_ofl()
        self.ep = ep
        self.backend = self.ofl.BACKEND_RDMA if ep.scheme == "o1rdma" else self.ofl.BACKEND_TCP
        self.local_ip = ep.options.get("bind")
        if not self.local_ip:
            raise TransportError(
                f"endpoint {ep}: ?bind=<this side's direct-link IPv4> is required "
                "(it selects the RDMA device and GID)")
        self.max_req = int(ep.options.get("max_req", DEFAULT_MAX_REQ))
        self.max_resp = int(ep.options.get("max_resp", DEFAULT_MAX_RESP))
        self.gid_index = int(ep.options.get("gid", -1))
        self.rdma_device = ep.options.get("dev")
        self._c = None
        self._regions: dict[int, tuple[int, int, int]] = {}  # id(buf) -> (addr, len, region)
        self._keep: list[object] = []
        self._res = None
        self._seg = None

    @property
    def connected(self) -> bool:
        return self._c is not None

    def connect(self, timeout_s: float) -> None:
        self.close()
        try:
            self._c = self.ofl.Client(
                self.backend, self.ep.host, self.ep.port, self.local_ip, service=SERVICE,
                max_req=self.max_req, max_resp=self.max_resp, inflight=1,
                rdma_device=self.rdma_device, gid_index=self.gid_index,
                connect_timeout_ms=max(1, int(timeout_s * 1000)))
        except self.ofl.OflError as e:
            self._c = None
            if e.code == -3:
                raise TransportTimeout(f"ofl connect {self.ep}: {e}") from e
            raise TransportError(f"ofl connect {self.ep}: {e}") from e
        self._regions.clear()
        self._keep.clear()
        self._res = self.ofl.Result()
        self._seg = (self.ofl._Seg * 1)()

    def _region_for(self, buf) -> tuple[int, int]:
        addr, n = _addr(buf)
        hit = self._regions.get(id(buf))
        if hit is not None and hit[0] == addr and hit[1] == n:
            return hit[2], n
        if n < self.max_resp:
            raise TransportError(f"reply buffer of {n} B < max_resp {self.max_resp}")
        try:
            region = self._c.reg(buf)
        except self.ofl.OflError as e:
            raise TransportError(f"ofl reg: {e}") from e
        self._regions[id(buf)] = (addr, n, region)
        self._keep.append(buf)
        return region, n

    def request(self, msg, resp_buf, timeout_s: float) -> int:
        if self._c is None:
            raise TransportError("not connected")
        addr, n = _addr(msg)
        if n > self.max_req:
            raise TransportError(f"message of {n} B > ofl max_req {self.max_req}")
        op, seq = _OP_SEQ.unpack_from(memoryview(msg).cast("B"), 6)
        return self._call(addr, n, op, seq, resp_buf, timeout_s)

    def request_parts(self, parts, resp_buf, timeout_s: float) -> int:
        """The parts are copied straight into ofl's send slot (ofl_client_payload_buf), and the call then skips its own
        payload copy: the same single copy request() costs, with no joined buffer in between."""
        c = self._c
        if c is None:
            raise TransportError("not connected")
        spans = [_addr(p) for p in parts]
        n = sum(k for _, k in spans)
        if n > self.max_req:
            raise TransportError(f"message of {n} B > ofl max_req {self.max_req}")
        if not spans or spans[0][1] < _OP_SEQ.size + 6:
            raise TransportError("the first part must hold the AppHeader")
        op, seq = _OP_SEQ.unpack_from(memoryview(parts[0]).cast("B"), 6)
        pl, cap = c.payload_buf()
        if not pl:
            self.close()
            raise TransportError("ofl connection is broken (no payload slot)")
        if n > cap:
            raise TransportError(f"message of {n} B > ofl payload slot {cap} B")
        off = 0
        for a, k in spans:
            ctypes.memmove(pl + off, a, k)
            off += k
        return self._call(pl, n, op, seq, resp_buf, timeout_s)

    def _call(self, addr: int, n: int, op: int, seq: int, resp_buf, timeout_s: float) -> int:
        c = self._c
        region, cap = self._region_for(resp_buf)
        seg = self._seg
        seg[0].region, seg[0].offset, seg[0].len = region, 0, min(cap, self.max_resp)
        L = self.ofl.lib()
        rc = L.ofl_client_call(c.h, op, seq, addr, n, seg, 1, ctypes.byref(self._res),
                               int(timeout_s * 1e9))
        if rc != 0:
            msg_txt = (L.ofl_client_error(c.h) or b"").decode(errors="replace")
            name = self.ofl.ERRORS.get(rc, str(rc))
            if rc == -3:  # ETIMEDOUT: ofl already broke the connection
                self.close()
                raise TransportTimeout(f"ofl call: {name}: {msg_txt}")
            self.close()
            raise TransportError(f"ofl call: {name}: {msg_txt}")
        if self._res.generation != seq:
            self.close()
            raise TransportError(f"ofl generation echo {self._res.generation} != {seq}")
        return int(self._res.resp_len)

    def close(self) -> None:
        if self._c is not None:
            try:
                self._c.close()
            finally:
                self._c = None
                self._regions.clear()
                self._keep.clear()
