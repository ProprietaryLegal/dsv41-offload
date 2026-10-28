"""CPU-only bounded immutable prefix chunks; no vLLM/torch/model imports."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import secrets
import struct
import threading
import zlib

MAX_CHUNK = 1 << 20
MAX_META = 2048
HEADER = struct.Struct('<4sB3x16sIII')
MAC_BYTES = 32
MAX_PACKET = HEADER.size + MAX_META + MAX_CHUNK + MAC_BYTES
GET, PUT = 1, 2
DIRECT_IPS = frozenset(('10.202.28.1', '10.202.29.1'))


class Refused(ValueError):
    """An untrusted, mismatched or corrupt chunk must never reach the engine."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex(value: str) -> None:
    if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise Refused('identity must be a lowercase SHA256 digest')


@dataclass(frozen=True)
class Namespace:
    model: str
    layout: str
    scope: str
    rank: int
    world_size: int = 2

    def __post_init__(self) -> None:
        for value in (self.model, self.layout, self.scope):
            _hex(value)
        if type(self.rank) is not int or type(self.world_size) is not int:
            raise Refused('rank and world size must be integers')
        if not 0 <= self.rank < self.world_size <= 256:
            raise Refused('invalid rank/world size')


@dataclass(frozen=True)
class Key:
    namespace: Namespace
    group: int
    token_chain: str
    payload_sha256: str
    nbytes: int

    def __post_init__(self) -> None:
        _hex(self.token_chain)
        _hex(self.payload_sha256)
        if type(self.group) is not int or not 0 <= self.group < 1024:
            raise Refused('invalid group')
        if type(self.nbytes) is not int or not 1 <= self.nbytes <= MAX_CHUNK:
            raise Refused('invalid chunk size')

    def encode(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()

    @classmethod
    def decode(cls, raw: bytes) -> Key:
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict) or set(obj) != {
                'namespace', 'group', 'token_chain', 'payload_sha256', 'nbytes'
            }:
                raise Refused('invalid key fields')
            ns = obj['namespace']
            if not isinstance(ns, dict) or set(ns) != {'model', 'layout', 'scope', 'rank', 'world_size'}:
                raise Refused('invalid namespace fields')
            return cls(Namespace(**ns), obj['group'], obj['token_chain'], obj['payload_sha256'], obj['nbytes'])
        except (TypeError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
            raise Refused(f'invalid key: {exc}') from exc


@dataclass(frozen=True)
class Packet:
    op: int
    nonce: bytes
    meta: bytes
    data: bytes


def pack(secret: bytes, magic: bytes, packet: Packet) -> bytes:
    if len(secret) < 32:
        raise Refused('authentication key must have at least 32 bytes')
    if len(packet.meta) > MAX_META or len(packet.data) > MAX_CHUNK or len(packet.nonce) != 16:
        raise Refused('packet exceeds protocol limits')
    raw = HEADER.pack(magic, packet.op, packet.nonce, len(packet.meta), len(packet.data),
                      zlib.crc32(packet.data)) + packet.meta + packet.data
    return raw + hmac.digest(secret, raw, 'sha256')


def unpack(secret: bytes, magic: bytes, raw: bytes) -> Packet:
    if not HEADER.size + MAC_BYTES <= len(raw) <= MAX_PACKET:
        raise Refused('invalid packet length')
    if not hmac.compare_digest(hmac.digest(secret, raw[:-MAC_BYTES], 'sha256'), raw[-MAC_BYTES:]):
        raise Refused('authentication failed')
    got_magic, op, nonce, nmeta, ndata, crc = HEADER.unpack_from(raw)
    if got_magic != magic or nmeta > MAX_META or ndata > MAX_CHUNK:
        raise Refused('invalid protocol header')
    if len(raw) != HEADER.size + nmeta + ndata + MAC_BYTES:
        raise Refused('truncated/extra packet bytes')
    data = raw[HEADER.size + nmeta:-MAC_BYTES]
    if zlib.crc32(data) != crc:
        raise Refused('CRC mismatch')
    return Packet(op, nonce, raw[HEADER.size:HEADER.size + nmeta], data)


@dataclass(frozen=True)
class Entry:
    data: bytes
    crc32: int


class PrefixStore:
    """LRU bounds payload bytes AND entry count; no reserved/whole-cache pinning."""

    def __init__(self, namespaces: frozenset[Namespace], capacity: int = 64 << 20,
                 max_entries: int = 4096):
        if not namespaces or not 1 <= capacity <= 64 << 30 or not 1 <= max_entries <= 1_000_000:
            raise ValueError('invalid namespace/capacity/entry bound')
        self.namespaces = namespaces
        self.capacity = capacity
        self.max_entries = max_entries
        self._entries: OrderedDict[Key, Entry] = OrderedDict()
        self._lock = threading.Lock()
        self.used = 0
        self.evictions = 0
        self.corruptions = 0

    def _check(self, key: Key) -> None:
        if key.namespace not in self.namespaces:
            raise Refused('model/layout/scope/rank namespace is not allowed')
        if key.nbytes > self.capacity:
            raise Refused('chunk exceeds store capacity')

    def put(self, key: Key, data: bytes) -> None:
        self._check(key)
        if len(data) != key.nbytes or digest(data) != key.payload_sha256:
            raise Refused('immutable payload hash/size mismatch')
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                if existing.data != data:
                    raise Refused('immutable key collision or corrupt resident entry')
                self._entries.move_to_end(key)
                return
            while self._entries and (self.used + len(data) > self.capacity or
                                     len(self._entries) >= self.max_entries):
                _, old = self._entries.popitem(last=False)
                self.used -= len(old.data)
                self.evictions += 1
            self._entries[key] = Entry(data, zlib.crc32(data))
            self.used += len(data)

    def get(self, key: Key) -> bytes | None:
        self._check(key)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if (len(entry.data) != key.nbytes or zlib.crc32(entry.data) != entry.crc32 or
                    digest(entry.data) != key.payload_sha256):
                del self._entries[key]
                self.used -= len(entry.data)
                self.corruptions += 1
                raise Refused('resident chunk corruption; discarded')
            self._entries.move_to_end(key)
            return entry.data

    def handle(self, secret: bytes, raw: bytes) -> bytes:
        request = unpack(secret, b'RP1Q', raw)
        key = Key.decode(request.meta)
        if request.op == PUT:
            self.put(key, request.data)
            data, status = b'', b'stored'
        elif request.op == GET and not request.data:
            found = self.get(key)
            data, status = (b'', b'miss') if found is None else (found, b'hit')
        else:
            raise Refused('invalid operation or GET payload')
        # Bind each authenticated response to the operation, nonce AND full requested key.
        meta = status + b'\n' + key.encode()
        return pack(secret, b'RP1R', Packet(request.op, request.nonce, meta, data))


def request(secret: bytes, op: int, key: Key, data: bytes = b'') -> tuple[bytes, bytes]:
    nonce = secrets.token_bytes(16)
    return nonce, pack(secret, b'RP1Q', Packet(op, nonce, key.encode(), data))


def response(secret: bytes, raw: bytes, op: int, nonce: bytes, key: Key) -> tuple[str, bytes]:
    reply = unpack(secret, b'RP1R', raw)
    status, separator, encoded_key = reply.meta.partition(b'\n')
    if reply.op != op or reply.nonce != nonce or separator != b'\n' or encoded_key != key.encode():
        raise Refused('stale response or wrong key/operation')
    if status == b'hit' and op == GET:
        if len(reply.data) != key.nbytes or digest(reply.data) != key.payload_sha256:
            raise Refused('response payload identity mismatch')
    elif (status, op) not in ((b'miss', GET), (b'stored', PUT)) or reply.data:
        raise Refused('invalid response status/data')
    return status.decode('ascii'), reply.data
