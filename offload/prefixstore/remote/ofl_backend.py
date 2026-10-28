"""CPU prototype using existing ofl transport. No GPU and no vLLM integration."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib
import logging
from pathlib import Path
import sys
import time
from typing import Protocol, cast

from store import DIRECT_IPS, GET, MAX_PACKET, PUT, Key, Namespace, PrefixStore, Refused, request, response

SERVICE = 'prefixstore-rp1-v1'
LOG = logging.getLogger(__name__)


class OflResult(Protocol):
    resp_len: int


class Wire(Protocol):
    def reg(self, buffer: bytearray) -> int: ...
    def call(self, op: int, payload: bytes, segs: list[tuple[int, int, int]],
             timeout_s: float = 2.0, generation: int = 0) -> OflResult: ...
    def close(self) -> None: ...


class WireRequest(Protocol):
    payload: bytes
    def respond(self, data: bytes) -> None: ...


class WireServer(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...


class OflModule(Protocol):
    Client: Callable[..., Wire]
    Server: Callable[..., WireServer]
    OflError: type[RuntimeError]


def ofl() -> OflModule:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'transport'))
    return cast(OflModule, importlib.import_module('ofl'))


def memory_guard(capacity: int, max_entries: int) -> int:
    """Conservative bound includes Python objects, fixed buffers and handler copies."""
    estimate = capacity + max_entries * 4096 + 256 * (1 << 20)
    if estimate > 64 << 30:
        raise Refused('cache plus metadata/transport budget exceeds lane 64 GiB')
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(values['MemAvailable'].split()[0]) * 1024
    if available < estimate * 2 + (32 << 30):
        raise Refused('not enough MemAvailable for bounded cache plus 32 GiB host reserve')
    return estimate


class PrefixServer:
    def __init__(self, bind_ip: str, port: int, secret: bytes, namespaces: frozenset[Namespace], *,
                 capacity: int = 64 << 20, max_entries: int = 4096, rdma: bool = False):
        if bind_ip not in DIRECT_IPS:
            raise Refused('listener must bind a x86host direct-link IP')
        if len(secret) < 32:
            raise Refused('authentication key must have at least 32 bytes')
        self.estimated_bytes = memory_guard(capacity, max_entries)
        self.store = PrefixStore(namespaces, capacity, max_entries)
        self.secret = secret
        self.refusals = 0
        self._wire = ofl().Server(bind_ip, port, service=SERVICE, handler=self._handle,
                                  backends=(1, 2) if rdma else (1,), max_req=MAX_PACKET,
                                  max_resp=MAX_PACKET, inflight=1, spin_us=0)
        # ofl.py sets max_conns=8. Never raise it: this bounds registered memory.

    def _handle(self, req: WireRequest) -> int:
        try:
            req.respond(self.store.handle(self.secret, req.payload))
            return 0
        except Refused as exc:
            self.refusals += 1
            LOG.warning('prefix request refused: %s', exc)
            return -100

    def start(self) -> None:
        self._wire.start()

    def stop(self) -> None:
        self._wire.stop()


@dataclass(frozen=True)
class Load:
    data: bytes | None
    reason: str


class PrefixClient:
    """One client/thread; missing/refused/timeout all mean recompute, never reuse buffer."""
    def __init__(self, server_ip: str, port: int, local_ip: str, secret: bytes, *,
                 backend: int = 1, timeout_s: float = 0.1, cooldown_s: float = 1.0):
        if server_ip not in DIRECT_IPS or local_ip not in DIRECT_IPS | {'10.202.28.2', '10.202.29.2'}:
            raise Refused('client must use direct-link addresses')
        if len(secret) < 32 or not 0 < timeout_s <= 1.0 or cooldown_s < 0:
            raise Refused('invalid key/deadline/cooldown')
        self.server_ip, self.port, self.local_ip = server_ip, port, local_ip
        self.secret, self.backend = secret, backend
        self.timeout_s, self.cooldown_s = timeout_s, cooldown_s
        self._wire: Wire | None = None
        self._buffer = bytearray(MAX_PACKET)
        self._region = 0
        self._retry_at = 0.0
        self.last_error = ''

    def _connect(self, budget: float) -> Wire:
        if time.monotonic() < self._retry_at:
            raise Refused('circuit open after prior refusal/transport failure')
        if self._wire is None:
            self._wire = ofl().Client(self.backend, self.server_ip, self.port, self.local_ip,
                                      service=SERVICE, max_req=MAX_PACKET, max_resp=MAX_PACKET,
                                      inflight=1, connect_timeout_ms=max(1, int(budget * 1000)))
            self._region = self._wire.reg(self._buffer)
        return self._wire

    def _call(self, op: int, key: Key, data: bytes = b'', *, budget: float | None = None) -> tuple[str, bytes]:
        deadline = time.monotonic() + min(self.timeout_s, budget if budget is not None else self.timeout_s)
        nonce, encoded = request(self.secret, op, key, data)
        wire = self._connect(max(0.001, deadline - time.monotonic()))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise Refused('deadline expired connecting')
        result = wire.call(op, encoded, [(self._region, 0, MAX_PACKET)], timeout_s=remaining)
        if time.monotonic() > deadline:
            raise Refused('deadline expired receiving')
        if not 0 < result.resp_len <= MAX_PACKET:
            raise Refused('invalid response length')
        return response(self.secret, bytes(self._buffer[:result.resp_len]), op, nonce, key)

    def _failed(self, exc: Exception) -> None:
        self.last_error = f'{type(exc).__name__}: {exc}'
        self.close()
        self._retry_at = time.monotonic() + self.cooldown_s
        LOG.warning('prefix tier fallback to compute: %s', self.last_error)

    def put(self, key: Key, data: bytes) -> bool:
        try:
            status, _ = self._call(PUT, key, data)
            return status == 'stored'
        except (Refused, OSError, ofl().OflError) as exc:
            self._failed(exc)
            return False

    def load(self, key: Key, *, budget: float | None = None) -> Load:
        try:
            status, data = self._call(GET, key, budget=budget)
            return Load(data, 'hit') if status == 'hit' else Load(None, 'miss: compute')
        except (Refused, OSError, ofl().OflError) as exc:
            self._failed(exc)
            return Load(None, self.last_error)

    def load_many(self, keys: list[Key], *, total_deadline_s: float = 0.5) -> Load:
        """At most 16 MiB staged; no partial prefix is published on any miss/failure."""
        if not keys or len(keys) > 16 or sum(key.nbytes for key in keys) > 16 << 20:
            return Load(None, 'prefix exceeds bounded request batch; compute')
        if not 0 < total_deadline_s <= 1:
            raise Refused('prefix deadline must be in (0, 1] seconds')
        deadline = time.monotonic() + total_deadline_s
        chunks: list[bytes] = []
        for key in keys:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return Load(None, 'prefix deadline expired; compute')
            loaded = self.load(key, budget=remaining)
            if loaded.data is None:
                return Load(None, loaded.reason)
            chunks.append(loaded.data)
        if time.monotonic() > deadline:
            return Load(None, 'prefix deadline expired; compute')
        return Load(b''.join(chunks), 'hit')

    def close(self) -> None:
        if self._wire is not None:
            self._wire.close()
            self._wire = None
