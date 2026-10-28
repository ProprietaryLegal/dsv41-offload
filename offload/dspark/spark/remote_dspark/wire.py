# SPDX-License-Identifier: Apache-2.0
"""Wire codec for the remote DSpark contract (offload/dspark/CONTRACT.md, v1.0).

Pure Python + numpy, no torch and no vLLM: shared by the Spark proxy, the mock worker
and the offline tests. Every layout here mirrors a table in CONTRACT.md section 6; keep
them in step (and bump CONTRACT_VERSION / the change log) on any change.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

try:  # numpy is needed for STEP bodies; HELLO/PING/STATS framing works without it
    import numpy as np  # (the probe tool runs on Spark hosts whose python3 has no numpy)
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

CONTRACT_VERSION = "1.8"

MAGIC = 0x31505344  # bytes "DSP1" little-endian
VERSION = 1

OP_HELLO = 1
OP_STEP = 2
OP_RESET = 3
OP_STATS = 4
OP_PING = 5
OP_NAMES = {OP_HELLO: "HELLO", OP_STEP: "STEP", OP_RESET: "RESET",
            OP_STATS: "STATS", OP_PING: "PING"}

# Frame status (CONTRACT 9.1)
ST_OK = 0
ST_BAD_VERSION = 1
ST_IDENTITY_MISMATCH = 2
ST_CONFIG_MISMATCH = 3
ST_NO_SESSION = 4
ST_STALE_SEQ = 5
ST_MALFORMED = 6
ST_CAPACITY = 7
ST_INTERNAL = 8
ST_UNSUPPORTED_OP = 9
FRAME_STATUS_NAMES = {
    ST_OK: "OK", ST_BAD_VERSION: "BAD_VERSION", ST_IDENTITY_MISMATCH: "IDENTITY_MISMATCH",
    ST_CONFIG_MISMATCH: "CONFIG_MISMATCH", ST_NO_SESSION: "NO_SESSION",
    ST_STALE_SEQ: "STALE_SEQ", ST_MALFORMED: "MALFORMED", ST_CAPACITY: "CAPACITY",
    ST_INTERNAL: "INTERNAL", ST_UNSUPPORTED_OP: "UNSUPPORTED_OP",
}
# Frame errors after which the remote must not be retried (configuration errors).
FATAL_FRAME_STATUS = frozenset(
    {ST_BAD_VERSION, ST_IDENTITY_MISMATCH, ST_CONFIG_MISMATCH, ST_UNSUPPORTED_OP}
)

# Per-request status (CONTRACT 9.2)
RS_OK = 0
RS_PARTIAL_WINDOW = 1
RS_STALE_GENERATION = 2
RS_BAD_ENTRY = 3
RS_INTERNAL = 4
REQ_STATUS_NAMES = {RS_OK: "OK", RS_PARTIAL_WINDOW: "PARTIAL_WINDOW",
                    RS_STALE_GENERATION: "STALE_GENERATION", RS_BAD_ENTRY: "BAD_ENTRY",
                    RS_INTERNAL: "INTERNAL"}
# Per-request statuses whose drafts may be used.
USABLE_REQ_STATUS = frozenset({RS_OK, RS_PARTIAL_WINDOW})

# ReqEntry.req_flags
RF_NEW_GEN = 0x1
RF_NEED_DRAFT = 0x2
RF_RESYNC = 0x4

# StepHeader.step_flags / ReplyHeader.reply_flags
SF_WANT_MARGIN = 0x1
RPF_HAS_MARGIN = 0x1

PAYLOAD_TAPS_BF16 = 1
PAYLOAD_MAINX_BF16 = 2
SAMPLE_GREEDY = 1

CAP_TAPS_BF16 = 0x1
CAP_MAINX_BF16 = 0x2
CAP_HAS_MARGIN = 0x4

APP_HEADER = struct.Struct("<IHHQIIiI")          # 32 B
HELLO_REQ = struct.Struct("<20I32s32s48s")       # 192 B
HELLO_REP = struct.Struct("<6I32s32s40s")        # 128 B
STEP_HEADER = struct.Struct("<QHHIII8x")         # 32 B
REQ_ENTRY = struct.Struct("<BBBBIqIiq")          # 32 B
REPLY_HEADER = struct.Struct("<QHHIII8x")        # 32 B
REQ_STATUS = struct.Struct("<HHI")               # 8 B
RESET_BODY = struct.Struct("<I")                 # 4 B

assert APP_HEADER.size == 32
assert HELLO_REQ.size == 192
assert HELLO_REP.size == 128
assert STEP_HEADER.size == 32
assert REQ_ENTRY.size == 32
assert REPLY_HEADER.size == 32
assert REQ_STATUS.size == 8

APP_HEADER_SIZE = APP_HEADER.size
U32_MASK = 0xFFFFFFFF


class WireError(ValueError):
    """A malformed or inconsistent message."""


def align64(n: int) -> int:
    return (n + 63) & ~63


def rows_offset_for(n_req: int) -> int:
    """Offset of the row block from the start of a STEP body (multiple of 64)."""
    return align64(STEP_HEADER.size + REQ_ENTRY.size * n_req)


def gen_newer(a: int, b: int) -> bool:
    """Serial-number arithmetic on u32 generations: is a newer than b?"""
    d = (a - b) & U32_MASK
    return d != 0 and d < 0x80000000


# --------------------------------------------------------------------------- headers


@dataclass
class AppHeader:
    op: int
    seq: int
    session: int
    body_len: int
    status: int = 0
    flags: int = 0
    magic: int = MAGIC
    version: int = VERSION

    def pack_into(self, buf, off: int = 0) -> None:
        APP_HEADER.pack_into(buf, off, self.magic, self.version, self.op, self.seq,
                             self.session, self.body_len, self.status, self.flags)

    def pack(self) -> bytes:
        b = bytearray(APP_HEADER_SIZE)
        self.pack_into(b)
        return bytes(b)

    @classmethod
    def unpack_from(cls, buf, off: int = 0) -> "AppHeader":
        if len(buf) - off < APP_HEADER_SIZE:
            raise WireError(f"short frame: {len(buf) - off} < {APP_HEADER_SIZE}")
        magic, version, op, seq, session, body_len, status, flags = APP_HEADER.unpack_from(
            buf, off)
        h = cls(op=op, seq=seq, session=session, body_len=body_len, status=status,
                flags=flags, magic=magic, version=version)
        return h

    def check(self, frame_len: int) -> None:
        if self.magic != MAGIC:
            raise WireError(f"bad magic 0x{self.magic:08x}")
        if self.version != VERSION:
            raise WireError(f"unsupported version {self.version}")
        if APP_HEADER_SIZE + self.body_len != frame_len:
            raise WireError(
                f"body_len {self.body_len} does not match frame length {frame_len}")


# --------------------------------------------------------------------------- HELLO


@dataclass
class Hello:
    num_spec: int = 5
    window: int = 128
    hidden: int = 5120
    n_taps: int = 3
    tap_layer_ids: tuple[int, ...] = (37, 38, 39)
    payload_kind: int = PAYLOAD_TAPS_BF16
    row_bytes: int = 3 * 5120 * 2
    noise_token_id: int = 128799
    vocab_size: int = 129280
    max_model_len: int = 327680
    max_slots: int = 4
    max_rows_per_step: int = 4 * 128
    sample_mode: int = SAMPLE_GREEDY
    tp_size: int = 2
    drafter_identity: bytes = bytes(32)
    markov_identity: bytes = bytes(32)
    client_name: str = ""
    proto_min: int = 1
    proto_max: int = 1

    def pack(self) -> bytes:
        ids = list(self.tap_layer_ids) + [0] * (4 - len(self.tap_layer_ids))
        if len(ids) != 4:
            raise WireError("at most 4 tap layers")
        return HELLO_REQ.pack(
            self.proto_min, self.proto_max, self.num_spec, self.window, self.hidden,
            self.n_taps, *ids, self.payload_kind, self.row_bytes, self.noise_token_id,
            self.vocab_size, self.max_model_len, self.max_slots, self.max_rows_per_step,
            self.sample_mode, self.tp_size, 0,
            _b32(self.drafter_identity), _b32(self.markov_identity),
            self.client_name.encode("ascii", "replace")[:48],
        )

    @classmethod
    def unpack(cls, body) -> "Hello":
        if len(body) != HELLO_REQ.size:
            raise WireError(f"HELLO body {len(body)} != {HELLO_REQ.size}")
        v = HELLO_REQ.unpack(bytes(body))
        n_taps = v[5]
        return cls(
            proto_min=v[0], proto_max=v[1], num_spec=v[2], window=v[3], hidden=v[4],
            n_taps=n_taps, tap_layer_ids=tuple(v[6:6 + n_taps]), payload_kind=v[10],
            row_bytes=v[11], noise_token_id=v[12], vocab_size=v[13], max_model_len=v[14],
            max_slots=v[15], max_rows_per_step=v[16], sample_mode=v[17], tp_size=v[18],
            drafter_identity=v[20], markov_identity=v[21],
            client_name=v[22].rstrip(b"\0").decode("ascii", "replace"),
        )

    def config_tuple(self) -> tuple:
        """Fields that client and worker must agree on (CONFIG_MISMATCH otherwise)."""
        return (self.num_spec, self.window, self.hidden, self.n_taps,
                tuple(self.tap_layer_ids), self.payload_kind, self.row_bytes,
                self.noise_token_id, self.vocab_size, self.sample_mode)


@dataclass
class HelloReply:
    proto: int = 1
    max_slots: int = 4
    max_rows_per_step: int = 512
    capabilities: int = CAP_TAPS_BF16
    window: int = 128
    num_spec: int = 5
    drafter_identity: bytes = bytes(32)
    markov_identity: bytes = bytes(32)
    worker_name: str = ""

    def pack(self) -> bytes:
        return HELLO_REP.pack(self.proto, self.max_slots, self.max_rows_per_step,
                              self.capabilities, self.window, self.num_spec,
                              _b32(self.drafter_identity), _b32(self.markov_identity),
                              self.worker_name.encode("ascii", "replace")[:40])

    @classmethod
    def unpack(cls, body) -> "HelloReply":
        if len(body) != HELLO_REP.size:
            raise WireError(f"HELLO reply body {len(body)} != {HELLO_REP.size}")
        v = HELLO_REP.unpack(bytes(body))
        return cls(proto=v[0], max_slots=v[1], max_rows_per_step=v[2], capabilities=v[3],
                   window=v[4], num_spec=v[5], drafter_identity=v[6], markov_identity=v[7],
                   worker_name=v[8].rstrip(b"\0").decode("ascii", "replace"))


def _b32(b: bytes) -> bytes:
    if len(b) != 32:
        raise WireError(f"identity must be 32 bytes, got {len(b)}")
    return b


# --------------------------------------------------------------------------- STEP


@dataclass
class ReqEntry:
    slot: int
    generation: int
    first_pos: int
    n_rows: int
    anchor_token: int
    anchor_pos: int
    flags: int = 0
    k_window: int = 0
    n_out: int = 0   # CONTRACT 1.3: draft positions the Spark consumes (0 = all N)

    def pack_into(self, buf, off: int) -> None:
        REQ_ENTRY.pack_into(buf, off, self.slot, self.flags, self.k_window, self.n_out,
                            self.generation & U32_MASK, self.first_pos, self.n_rows,
                            self.anchor_token, self.anchor_pos)

    @classmethod
    def unpack_from(cls, buf, off: int) -> "ReqEntry":
        slot, flags, k_window, n_out, gen, first_pos, n_rows, anchor_token, anchor_pos = (
            REQ_ENTRY.unpack_from(buf, off))
        return cls(slot=slot, generation=gen, first_pos=first_pos, n_rows=n_rows,
                   anchor_token=anchor_token, anchor_pos=anchor_pos, flags=flags,
                   k_window=k_window, n_out=n_out)


@dataclass
class StepRequest:
    """A decoded STEP body. ``rows`` is a [n_rows, row_bytes] uint8 view (no copy)."""
    step_id: int
    step_flags: int
    row_bytes: int
    entries: list[ReqEntry]
    rows: np.ndarray

    @property
    def n_req(self) -> int:
        return len(self.entries)


def step_body_len(n_req: int, n_rows: int, row_bytes: int) -> int:
    return rows_offset_for(n_req) + n_rows * row_bytes


def pack_step_prefix(buf, off: int, step_id: int, entries: list[ReqEntry],
                     n_rows: int, row_bytes: int, step_flags: int = 0) -> int:
    """Write StepHeader + entries + zero padding at ``buf[off:]`` (``off`` = start of the
    STEP body). Returns the offset of the row block relative to the body start. The
    caller places the rows at ``buf[off + rows_offset:]`` (e.g. a pinned D2H target)."""
    n_req = len(entries)
    ro = rows_offset_for(n_req)
    STEP_HEADER.pack_into(buf, off, step_id, n_req, step_flags, n_rows, row_bytes, ro)
    p = off + STEP_HEADER.size
    total = 0
    for e in entries:
        e.pack_into(buf, p)
        p += REQ_ENTRY.size
        total += e.n_rows
    if total != n_rows:
        raise WireError(f"entries carry {total} rows, header says {n_rows}")
    pad_end = off + ro
    if pad_end > p:
        memoryview(buf).cast("B")[p:pad_end] = bytes(pad_end - p)
    return ro


def encode_step(step_id: int, entries: list[ReqEntry], rows: np.ndarray | None,
                row_bytes: int, step_flags: int = 0) -> bytes:
    """Convenience (tests, mock): full STEP body as bytes. ``rows`` is [n, row_bytes]
    uint8 (or any array whose bytes are the rows in order)."""
    n_rows = sum(e.n_rows for e in entries)
    body = bytearray(step_body_len(len(entries), n_rows, row_bytes))
    ro = pack_step_prefix(body, 0, step_id, entries, n_rows, row_bytes, step_flags)
    if n_rows:
        raw = np.ascontiguousarray(rows).view(np.uint8).reshape(-1)
        if raw.size != n_rows * row_bytes:
            raise WireError(f"rows hold {raw.size} bytes, expected {n_rows * row_bytes}")
        body[ro:ro + raw.size] = raw.tobytes()
    return bytes(body)


def decode_step(body, *, max_slots: int, max_rows: int, expected_row_bytes: int,
                window: int) -> StepRequest:
    mv = memoryview(body)
    if len(mv) < STEP_HEADER.size:
        raise WireError("STEP body shorter than its header")
    step_id, n_req, step_flags, n_rows, row_bytes, ro = STEP_HEADER.unpack_from(mv, 0)
    if row_bytes != expected_row_bytes:
        raise WireError(f"row_bytes {row_bytes} != {expected_row_bytes}")
    if not 1 <= n_req <= max_slots:
        raise WireError(f"n_req {n_req} outside 1..{max_slots}")
    if n_rows > max_rows:
        raise WireError(f"n_rows {n_rows} > max_rows_per_step {max_rows}")
    if ro != rows_offset_for(n_req):
        raise WireError(f"rows_offset {ro} != {rows_offset_for(n_req)}")
    if len(mv) != ro + n_rows * row_bytes:
        raise WireError(f"STEP body {len(mv)} != {ro + n_rows * row_bytes}")
    entries = [ReqEntry.unpack_from(mv, STEP_HEADER.size + i * REQ_ENTRY.size)
               for i in range(n_req)]
    seen = set()
    total = 0
    for e in entries:
        if e.slot >= max_slots or e.slot in seen:
            raise WireError(f"slot {e.slot} invalid or repeated")
        seen.add(e.slot)
        total += e.n_rows
    if total != n_rows:
        raise WireError(f"entries carry {total} rows, header says {n_rows}")
    rows = np.frombuffer(mv, dtype=np.uint8, count=n_rows * row_bytes, offset=ro)
    rows = rows.reshape(n_rows, row_bytes) if n_rows else rows.reshape(0, row_bytes)
    return StepRequest(step_id=step_id, step_flags=step_flags, row_bytes=row_bytes,
                       entries=entries, rows=rows)


@dataclass
class StepReply:
    step_id: int
    num_spec: int
    status: np.ndarray        # [n_req] uint16
    window_rows: np.ndarray   # [n_req] uint16
    generation: np.ndarray    # [n_req] uint32
    draft_ids: np.ndarray     # [n_req, N] int32
    confidence: np.ndarray    # [n_req, N] float32
    margin: np.ndarray | None = None  # [n_req, N] float32
    worker_us: int = 0
    recv_to_send_us: int = 0
    reply_flags: int = 0

    @property
    def n_req(self) -> int:
        return int(self.status.shape[0])

    def pack(self) -> bytes:
        n, N = self.n_req, self.num_spec
        flags = self.reply_flags | (RPF_HAS_MARGIN if self.margin is not None else 0)
        parts = [REPLY_HEADER.pack(self.step_id, n, N, flags, self.worker_us,
                                   self.recv_to_send_us)]
        st = np.zeros(n, dtype=np.dtype([("s", "<u2"), ("w", "<u2"), ("g", "<u4")]))
        st["s"], st["w"], st["g"] = self.status, self.window_rows, self.generation
        parts.append(st.tobytes())
        parts.append(np.ascontiguousarray(self.draft_ids, dtype="<i4").reshape(n, N).tobytes())
        parts.append(np.ascontiguousarray(self.confidence, dtype="<f4").reshape(n, N).tobytes())
        if self.margin is not None:
            parts.append(np.ascontiguousarray(self.margin, dtype="<f4").reshape(n, N).tobytes())
        return b"".join(parts)

    @classmethod
    def unpack(cls, body, *, expect_n_req: int | None = None,
               expect_num_spec: int | None = None) -> "StepReply":
        mv = memoryview(body)
        if len(mv) < REPLY_HEADER.size:
            raise WireError("STEP reply shorter than its header")
        step_id, n, N, flags, worker_us, r2s = REPLY_HEADER.unpack_from(mv, 0)
        if expect_n_req is not None and n != expect_n_req:
            raise WireError(f"reply n_req {n} != {expect_n_req}")
        if expect_num_spec is not None and N != expect_num_spec:
            raise WireError(f"reply num_spec {N} != {expect_num_spec}")
        has_margin = bool(flags & RPF_HAS_MARGIN)
        need = REPLY_HEADER.size + 8 * n + 8 * n * N + (4 * n * N if has_margin else 0)
        if len(mv) != need:
            raise WireError(f"STEP reply {len(mv)} bytes, expected {need}")
        o = REPLY_HEADER.size
        st = np.frombuffer(mv, dtype=np.dtype([("s", "<u2"), ("w", "<u2"), ("g", "<u4")]),
                           count=n, offset=o)
        o += 8 * n
        ids = np.frombuffer(mv, dtype="<i4", count=n * N, offset=o).reshape(n, N)
        o += 4 * n * N
        conf = np.frombuffer(mv, dtype="<f4", count=n * N, offset=o).reshape(n, N)
        o += 4 * n * N
        margin = None
        if has_margin:
            margin = np.frombuffer(mv, dtype="<f4", count=n * N, offset=o).reshape(n, N)
        return cls(step_id=step_id, num_spec=N, status=st["s"].copy(),
                   window_rows=st["w"].copy(), generation=st["g"].copy(),
                   draft_ids=ids.copy(), confidence=conf.copy(),
                   margin=None if margin is None else margin.copy(),
                   worker_us=worker_us, recv_to_send_us=r2s, reply_flags=flags)


# --------------------------------------------------------------------------- frames


def frame(op: int, seq: int, session: int, body: bytes = b"", status: int = 0) -> bytes:
    return AppHeader(op=op, seq=seq, session=session, body_len=len(body),
                     status=status).pack() + body


def split_frame(buf) -> tuple[AppHeader, memoryview]:
    mv = memoryview(buf)
    h = AppHeader.unpack_from(mv)
    h.check(len(mv))
    return h, mv[APP_HEADER_SIZE:]


# --------------------------------------------------------------------------- identity


def drafter_identity(file_hashes: dict[str, str]) -> bytes:
    """CONTRACT 6.7: sha256 over "<basename> <sha256-hex>\\n" sorted by basename."""
    lines = []
    for name in sorted(file_hashes):
        h = file_hashes[name].strip().lower()
        if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise ValueError(f"{name}: not a sha256 hex digest: {h!r}")
        if "/" in name:
            raise ValueError(f"{name}: basenames only")
        lines.append(f"{name} {h}\n")
    return hashlib.sha256("".join(lines).encode()).digest()


def parse_hex32(text: str, what: str) -> bytes:
    t = (text or "").strip().lower()
    if len(t) != 64:
        raise ValueError(f"{what}: expected 64 hex characters, got {len(t)}")
    return bytes.fromhex(t)


@dataclass
class Counters:
    """Small helper for per-reason counters (shared by client and mock worker)."""
    values: dict[str, int] = field(default_factory=dict)

    def inc(self, key: str, n: int = 1) -> None:
        self.values[key] = self.values.get(key, 0) + n
