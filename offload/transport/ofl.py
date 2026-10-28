"""ctypes binding of libofl.so (ofl.h) for tests and tools.

    from ofl import Client, Server, BACKEND_RDMA
    c = Client(BACKEND_RDMA, "10.202.28.1", 18600, "10.202.28.2", service="ofl-pingpong", max_req=4096,
               max_resp=1 << 20)
    buf = bytearray(1 << 20)
    region = c.reg(buf)
    res = c.call(0, b"hello", [(region, 0, 5)])        # op 0 = built-in echo; res.resp_len == 5, buf[:5] == b"hello"

The library path: OFL_LIB, else libofl.so next to this file, else ~/dsv41-offload/o1/bin/libofl.so.
Errors raise OflError(code, message); a timeout breaks the connection (reconnect() before the next call).
"""

from __future__ import annotations

import ctypes
import os

BACKEND_TCP, BACKEND_RDMA = 1, 2
OP_ECHO = 0
MAX_SEGS = 16
ERRORS = {
    0: "OK", -1: "EINVAL", -2: "ESYS", -3: "ETIMEDOUT", -4: "EBROKEN", -5: "EPROTO", -6: "ETOOBIG", -7: "ENOMEM",
    -8: "ERDMA", -9: "EREMOTE", -10: "EBUSY", -11: "EREJECT",
}


class OflError(RuntimeError):
    def __init__(self, code: int, msg: str):
        super().__init__(f"{ERRORS.get(code, code)}: {msg}")
        self.code = code


class _ClientConfig(ctypes.Structure):
    _fields_ = [("backend", ctypes.c_int), ("server_ip", ctypes.c_char_p), ("server_port", ctypes.c_uint16),
                ("local_ip", ctypes.c_char_p), ("rdma_device", ctypes.c_char_p), ("rdma_port", ctypes.c_int),
                ("gid_index", ctypes.c_int), ("service", ctypes.c_char_p), ("max_req_bytes", ctypes.c_uint32),
                ("max_resp_bytes", ctypes.c_uint32), ("max_inflight", ctypes.c_uint32),
                ("connect_timeout_ms", ctypes.c_int64)]


class _Seg(ctypes.Structure):
    _fields_ = [("region", ctypes.c_uint32), ("reserved", ctypes.c_uint32), ("offset", ctypes.c_uint64),
                ("len", ctypes.c_uint64)]


class Result(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int32), ("resp_len", ctypes.c_uint32), ("req_id", ctypes.c_uint64),
                ("generation", ctypes.c_uint64), ("elapsed_ns", ctypes.c_int64), ("server_ns", ctypes.c_int64)]


class _ServerConfig(ctypes.Structure):
    _fields_ = [("bind_ip", ctypes.c_char_p), ("port", ctypes.c_uint16), ("backends", ctypes.c_int),
                ("rdma_device", ctypes.c_char_p), ("rdma_port", ctypes.c_int), ("gid_index", ctypes.c_int),
                ("service", ctypes.c_char_p), ("max_req_bytes", ctypes.c_uint32), ("max_resp_bytes", ctypes.c_uint32),
                ("max_inflight", ctypes.c_uint32), ("max_conns", ctypes.c_uint32), ("spin_us", ctypes.c_int64)]


HANDLER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p)
_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.environ.get("OFL_LIB"), os.path.join(here, "libofl.so"),
                 os.path.expanduser("~/dsv41-offload/o1/bin/libofl.so")):
        if path and os.path.exists(path):
            break
    else:
        raise OSError("libofl.so not found (set OFL_LIB)")
    L = ctypes.CDLL(path)
    vp, u32, u64, i64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int64
    L.ofl_client_connect.restype = vp
    L.ofl_client_connect.argtypes = [ctypes.POINTER(_ClientConfig), ctypes.c_char_p, ctypes.c_size_t]
    L.ofl_client_reg.argtypes = [vp, vp, ctypes.c_size_t, ctypes.POINTER(u32)]
    L.ofl_client_call.argtypes = [vp, ctypes.c_uint16, u64, vp, u32, ctypes.POINTER(_Seg), u32,
                                  ctypes.POINTER(Result), i64]
    L.ofl_client_submit.restype = i64
    L.ofl_client_submit.argtypes = [vp, ctypes.c_uint16, u64, vp, u32, ctypes.POINTER(_Seg), u32]
    L.ofl_client_wait.argtypes = [vp, i64, ctypes.POINTER(Result), i64]
    L.ofl_client_broken.argtypes = [vp]
    L.ofl_client_reconnect.argtypes = [vp, ctypes.c_char_p, ctypes.c_size_t]
    L.ofl_client_payload_buf.restype = vp
    L.ofl_client_payload_buf.argtypes = [vp, ctypes.POINTER(u32)]
    L.ofl_client_error.restype = ctypes.c_char_p
    L.ofl_client_error.argtypes = [vp]
    L.ofl_client_describe.restype = ctypes.c_char_p
    L.ofl_client_describe.argtypes = [vp]
    L.ofl_client_close.argtypes = [vp]
    L.ofl_server_create.restype = vp
    L.ofl_server_create.argtypes = [ctypes.POINTER(_ServerConfig), HANDLER, vp, ctypes.c_char_p, ctypes.c_size_t]
    L.ofl_server_start.argtypes = [vp]
    L.ofl_server_stop.argtypes = [vp]
    L.ofl_server_destroy.argtypes = [vp]
    L.ofl_server_describe.restype = ctypes.c_char_p
    L.ofl_server_describe.argtypes = [vp]
    L.ofl_server_stats.argtypes = [vp, ctypes.POINTER(u64 * 4)]
    for name in ("ofl_req_op",):
        getattr(L, name).restype = ctypes.c_uint16
    for name in ("ofl_req_id", "ofl_req_generation", "ofl_req_conn_id"):
        getattr(L, name).restype = u64
    for name in ("ofl_req_op", "ofl_req_id", "ofl_req_generation", "ofl_req_conn_id", "ofl_req_backend"):
        getattr(L, name).argtypes = [vp]
    L.ofl_req_payload.restype = vp
    L.ofl_req_payload.argtypes = [vp, ctypes.POINTER(u32)]
    L.ofl_req_resp_buf.restype = vp
    L.ofl_req_resp_buf.argtypes = [vp, ctypes.POINTER(u32)]
    L.ofl_req_ready.argtypes = [vp, u32, u32]
    L.ofl_req_set_len.argtypes = [vp, u32]
    _lib = L
    return L


def _addr_len(buf) -> tuple[int, int]:
    """Address and size of a writable buffer: (addr, len) tuple, bytearray, ctypes array, numpy array, memoryview."""
    if isinstance(buf, tuple):
        return int(buf[0]), int(buf[1])
    if hasattr(buf, "ctypes") and hasattr(buf, "nbytes"):  # numpy
        return buf.ctypes.data, buf.nbytes
    mv = memoryview(buf)
    if mv.readonly:
        raise ValueError("destination buffer must be writable")
    c = (ctypes.c_char * mv.nbytes).from_buffer(mv)
    return ctypes.addressof(c), mv.nbytes


class Client:
    def __init__(self, backend: int, server_ip: str, port: int, local_ip: str, *, service: str, max_req: int,
                 max_resp: int, inflight: int = 1, rdma_device: str | None = None, gid_index: int = -1,
                 connect_timeout_ms: int = 5000):
        L = lib()
        self._keep = []  # buffers whose addresses were registered
        cfg = _ClientConfig(backend, server_ip.encode(), port, local_ip.encode(),
                            rdma_device.encode() if rdma_device else None, 1, gid_index, service.encode(), max_req,
                            max_resp, inflight, connect_timeout_ms)
        err = ctypes.create_string_buffer(512)
        self.h = L.ofl_client_connect(ctypes.byref(cfg), err, 512)
        if not self.h:
            raise OflError(-4, err.value.decode())

    def describe(self) -> str:
        return lib().ofl_client_describe(self.h).decode()

    def reg(self, buf) -> int:
        addr, n = _addr_len(buf)
        self._keep.append(buf)
        region = ctypes.c_uint32()
        rc = lib().ofl_client_reg(self.h, addr, n, ctypes.byref(region))
        if rc:
            raise OflError(rc, lib().ofl_client_error(self.h).decode())
        return region.value

    def call(self, op: int, payload: bytes, segs, timeout_s: float = 2.0, generation: int = 0) -> Result:
        arr = (_Seg * max(1, len(segs)))(*[_Seg(r, 0, o, n) for r, o, n in segs])
        res = Result()
        pl = ctypes.create_string_buffer(bytes(payload), max(1, len(payload)))
        rc = lib().ofl_client_call(self.h, op, generation, pl, len(payload), arr, len(segs), ctypes.byref(res),
                                   int(timeout_s * 1e9))
        if rc:
            raise OflError(rc, lib().ofl_client_error(self.h).decode())
        return res

    def payload_buf(self) -> tuple[int, int]:
        """(address, capacity) of the next request's payload area (stable across reconnects with inflight 1)."""
        cap = ctypes.c_uint32()
        addr = lib().ofl_client_payload_buf(self.h, ctypes.byref(cap))
        return int(addr or 0), cap.value

    def broken(self) -> bool:
        return bool(lib().ofl_client_broken(self.h))

    def reconnect(self) -> None:
        err = ctypes.create_string_buffer(512)
        rc = lib().ofl_client_reconnect(self.h, err, 512)
        if rc:
            raise OflError(rc, err.value.decode())

    def close(self) -> None:
        if self.h:
            lib().ofl_client_close(self.h)
            self.h = None


class Request:
    """The request a Python handler gets (valid only during the call)."""

    def __init__(self, h: int):
        L = lib()
        self.h = h
        self.op = L.ofl_req_op(h)
        self.id = L.ofl_req_id(h)
        self.generation = L.ofl_req_generation(h)
        n = ctypes.c_uint32()
        p = L.ofl_req_payload(h, ctypes.byref(n))
        self.payload = ctypes.string_at(p, n.value) if n.value else b""
        cap = ctypes.c_uint32()
        self._resp = L.ofl_req_resp_buf(h, ctypes.byref(cap))
        self.capacity = cap.value

    def respond(self, data: bytes) -> None:
        if len(data) > self.capacity:
            raise ValueError(f"response {len(data)} > capacity {self.capacity}")
        ctypes.memmove(self._resp, data, len(data))
        lib().ofl_req_set_len(self.h, len(data))


class Server:
    def __init__(self, bind_ip: str, port: int, *, service: str, handler, backends=(BACKEND_TCP, BACKEND_RDMA),
                 max_req: int = 1 << 20, max_resp: int = 1 << 24, inflight: int = 2, spin_us: int = 200000,
                 rdma_device: str | None = None, gid_index: int = -1):
        L = lib()
        mask = 0
        for b in backends:
            mask |= 1 << b

        def _cb(user, req):  # exceptions must not cross into C: they become status -1
            try:
                r = handler(Request(req))
                return int(r or 0)
            except Exception as exc:  # noqa: BLE001
                print(f"ofl.py handler error: {exc!r}")
                return -1

        self._cb = HANDLER(_cb)
        cfg = _ServerConfig(bind_ip.encode(), port, mask, rdma_device.encode() if rdma_device else None, 1, gid_index,
                            service.encode(), max_req, max_resp, inflight, 8, spin_us)
        err = ctypes.create_string_buffer(512)
        self.h = L.ofl_server_create(ctypes.byref(cfg), self._cb, None, err, 512)
        if not self.h:
            raise OflError(-1, err.value.decode())

    def describe(self) -> str:
        return lib().ofl_server_describe(self.h).decode()

    def start(self) -> None:
        lib().ofl_server_start(self.h)

    def stats(self) -> dict:
        out = (ctypes.c_uint64 * 4)()
        lib().ofl_server_stats(self.h, ctypes.byref(out))
        return dict(zip(("conns", "requests", "bytes", "errors"), list(out)))

    def stop(self) -> None:
        if self.h:
            lib().ofl_server_destroy(self.h)
            self.h = None
