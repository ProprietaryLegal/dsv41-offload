"""Wire codec of offload/dspark/CONTRACT.md v1.3 (lane O3): little-endian, offsets as in the contract's tables.

Pure Python + numpy views; no torch. Every decode checks lengths and raises ProtocolError (-> frame MALFORMED)."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = 0x31505344            # b"DSP1"
VERSION = 1
OP_HELLO, OP_STEP, OP_RESET, OP_STATS, OP_PING = 1, 2, 3, 4, 5

# frame status (AppHeader.status in replies), CONTRACT 9.1
OK, BAD_VERSION, IDENTITY_MISMATCH, CONFIG_MISMATCH, NO_SESSION, STALE_SEQ, MALFORMED, CAPACITY, INTERNAL, \
    UNSUPPORTED_OP = range(10)
FRAME_STATUS = {0: "OK", 1: "BAD_VERSION", 2: "IDENTITY_MISMATCH", 3: "CONFIG_MISMATCH", 4: "NO_SESSION",
                5: "STALE_SEQ", 6: "MALFORMED", 7: "CAPACITY", 8: "INTERNAL", 9: "UNSUPPORTED_OP"}
# per-request status (ReqStatus.status), CONTRACT 9.2
R_OK, R_PARTIAL_WINDOW, R_STALE_GENERATION, R_BAD_ENTRY, R_INTERNAL = range(5)

# HELLO fields
PAYLOAD_TAPS_BF16, PAYLOAD_MAINX_BF16 = 1, 2
SAMPLE_GREEDY = 1
CAP_TAPS_BF16, CAP_MAINX_BF16, CAP_HAS_MARGIN = 1, 2, 4
# STEP flags
STEP_WANT_MARGIN = 1
REQ_NEW_GEN, REQ_NEED_DRAFT, REQ_RESYNC = 1, 2, 4
REPLY_HAS_MARGIN = 1

APP_HDR = struct.Struct("<IHHQIIiI")          # 32 B
HELLO_REQ = struct.Struct("<6I4I10I32s32s48s")    # 192 B: 6 scalars, tap ids, 10 scalars (last reserved), ids, name
HELLO_REP = struct.Struct("<6I32s32s40s")         # 128 B
STEP_HDR = struct.Struct("<QHHIII8x")          # 32 B
REQ_ENTRY = struct.Struct("<BBBBIqIiq")        # 32 B
REPLY_HDR = struct.Struct("<QHHIII8x")         # 32 B
REQ_STATUS = struct.Struct("<HHI")             # 8 B

assert APP_HDR.size == 32 and HELLO_REQ.size == 192 and HELLO_REP.size == 128
assert STEP_HDR.size == 32 and REQ_ENTRY.size == 32 and REPLY_HDR.size == 32 and REQ_STATUS.size == 8


class ProtocolError(ValueError):
    pass


@dataclass
class AppHeader:
    magic: int
    version: int
    op: int
    seq: int
    session: int
    body_len: int
    status: int
    flags: int

    @classmethod
    def decode(cls, buf) -> "AppHeader":
        if len(buf) < APP_HDR.size:
            raise ProtocolError(f"frame of {len(buf)} B is shorter than the 32 B header")
        return cls(*APP_HDR.unpack_from(buf, 0))

    def encode(self) -> bytes:
        return APP_HDR.pack(self.magic, self.version, self.op, self.seq, self.session, self.body_len, self.status,
                            self.flags)


def reply(req: AppHeader, status: int, body: bytes = b"") -> bytes:
    return AppHeader(MAGIC, VERSION, req.op, req.seq, req.session, len(body), status, 0).encode() + body


@dataclass
class Hello:
    proto_min: int
    proto_max: int
    num_spec: int
    window: int
    hidden: int
    n_taps: int
    tap_layer_ids: tuple[int, int, int, int]
    payload_kind: int
    row_bytes: int
    noise_token_id: int
    vocab_size: int
    max_model_len: int
    max_slots: int
    max_rows_per_step: int
    sample_mode: int
    tp_size: int
    drafter_identity: bytes
    markov_identity: bytes
    client_name: str

    @classmethod
    def decode(cls, body) -> "Hello":
        if len(body) != HELLO_REQ.size:
            raise ProtocolError(f"HELLO body {len(body)} B, want {HELLO_REQ.size}")
        v = HELLO_REQ.unpack_from(body, 0)
        return cls(v[0], v[1], v[2], v[3], v[4], v[5], tuple(v[6:10]), v[10], v[11], v[12], v[13], v[14], v[15],
                   v[16], v[17], v[18], v[20], v[21], v[22].rstrip(b"\0").decode("ascii", "replace"))

    def encode(self) -> bytes:
        return HELLO_REQ.pack(self.proto_min, self.proto_max, self.num_spec, self.window, self.hidden, self.n_taps,
                              *self.tap_layer_ids, self.payload_kind, self.row_bytes, self.noise_token_id,
                              self.vocab_size, self.max_model_len, self.max_slots, self.max_rows_per_step,
                              self.sample_mode, self.tp_size, 0, self.drafter_identity, self.markov_identity,
                              self.client_name.encode()[:48])


def hello_reply(proto: int, max_slots: int, max_rows: int, caps: int, window: int, num_spec: int,
                drafter_identity: bytes, markov_identity: bytes, name: str) -> bytes:
    return HELLO_REP.pack(proto, max_slots, max_rows, caps, window, num_spec, drafter_identity, markov_identity,
                          name.encode()[:40])


@dataclass
class ReqEntry:
    slot: int
    req_flags: int
    k_window: int
    generation: int
    first_pos: int
    n_rows: int
    anchor_token: int
    anchor_pos: int
    n_out: int = 0                # CONTRACT 1.3: draft positions the Spark consumes, 0 = all N (byte 3, was reserved)
    row_start: int = 0            # index of this entry's first row in the row block (filled by decode_step)


@dataclass
class Step:
    step_id: int
    n_req: int
    step_flags: int
    n_rows: int
    row_bytes: int
    rows_offset: int
    entries: list[ReqEntry]
    rows: np.ndarray              # uint8 [n_rows, row_bytes] view into the frame (no copy)


def decode_step(body, expected_row_bytes: int, max_slots: int, max_rows: int) -> Step:
    """STEP request body -> Step. Raises ProtocolError for anything structurally wrong (frame MALFORMED);
    CAPACITY problems raise CapacityError."""
    if len(body) < STEP_HDR.size:
        raise ProtocolError("STEP body shorter than its 32 B header")
    step_id, n_req, step_flags, n_rows, row_bytes, rows_offset = STEP_HDR.unpack_from(body, 0)
    if n_req < 1:
        raise ProtocolError("STEP with no request")
    if n_req > max_slots or n_rows > max_rows:
        raise CapacityError(f"STEP with {n_req} requests / {n_rows} rows (limits {max_slots} / {max_rows})")
    if row_bytes != expected_row_bytes:
        raise ProtocolError(f"row_bytes {row_bytes}, HELLO said {expected_row_bytes}")
    entries_end = STEP_HDR.size + n_req * REQ_ENTRY.size
    if rows_offset % 64 or rows_offset < entries_end:
        raise ProtocolError(f"rows_offset {rows_offset} (entries end at {entries_end}, must be a multiple of 64)")
    if len(body) != rows_offset + n_rows * row_bytes:
        raise ProtocolError(f"STEP body {len(body)} B, header implies {rows_offset + n_rows * row_bytes}")
    entries = []
    start = 0
    slots = set()
    for i in range(n_req):
        v = REQ_ENTRY.unpack_from(body, STEP_HDR.size + i * REQ_ENTRY.size)
        e = ReqEntry(v[0], v[1], v[2], v[4], v[5], v[6], v[7], v[8], n_out=v[3])
        if e.slot in slots:
            raise ProtocolError(f"slot {e.slot} twice in one STEP")
        slots.add(e.slot)
        e.row_start = start
        start += e.n_rows
        entries.append(e)
    if start != n_rows:
        raise ProtocolError(f"entries carry {start} rows, header says {n_rows}")
    mv = memoryview(body)[rows_offset: rows_offset + n_rows * row_bytes]
    rows = np.frombuffer(mv, dtype=np.uint8).reshape(n_rows, row_bytes) if n_rows else np.empty((0, row_bytes),
                                                                                                np.uint8)
    return Step(step_id, n_req, step_flags, n_rows, row_bytes, rows_offset, entries, rows)


class CapacityError(ValueError):
    pass


def encode_step(step_id: int, entries: list[ReqEntry], rows: bytes, row_bytes: int, step_flags: int = 0) -> bytes:
    """Client side (tests, tools): build a STEP body; rows = the row block, entries in order."""
    n_rows = sum(e.n_rows for e in entries)
    entries_end = STEP_HDR.size + len(entries) * REQ_ENTRY.size
    rows_offset = (entries_end + 63) // 64 * 64
    out = bytearray(STEP_HDR.pack(step_id, len(entries), step_flags, n_rows, row_bytes, rows_offset))
    for e in entries:
        out += REQ_ENTRY.pack(e.slot, e.req_flags, e.k_window, e.n_out, e.generation, e.first_pos, e.n_rows,
                              e.anchor_token, e.anchor_pos)
    out += b"\0" * (rows_offset - len(out))
    assert len(rows) == n_rows * row_bytes
    return bytes(out) + rows


def encode_step_reply(step_id: int, num_spec: int, statuses: list[tuple[int, int, int]], draft_ids: np.ndarray,
                      conf: np.ndarray, margin: np.ndarray | None, worker_us: int, recv_to_send_us: int) -> bytes:
    n = len(statuses)
    flags = REPLY_HAS_MARGIN if margin is not None else 0
    out = bytearray(REPLY_HDR.pack(step_id, n, num_spec, flags, worker_us & 0xFFFFFFFF,
                                   recv_to_send_us & 0xFFFFFFFF))
    for st, wr, gen in statuses:
        out += REQ_STATUS.pack(st, wr, gen & 0xFFFFFFFF)
    out += np.ascontiguousarray(draft_ids, dtype="<i4").tobytes()
    out += np.ascontiguousarray(conf, dtype="<f4").tobytes()
    if margin is not None:
        out += np.ascontiguousarray(margin, dtype="<f4").tobytes()
    return bytes(out)


@dataclass
class StepReply:
    step_id: int
    n_req: int
    num_spec: int
    reply_flags: int
    worker_us: int
    recv_to_send_us: int
    statuses: list[tuple[int, int, int]]
    draft_ids: np.ndarray
    conf: np.ndarray
    margin: np.ndarray | None


def decode_step_reply(body) -> StepReply:
    step_id, n, N, flags, wus, rts = REPLY_HDR.unpack_from(body, 0)
    off = REPLY_HDR.size
    sts = [REQ_STATUS.unpack_from(body, off + i * REQ_STATUS.size) for i in range(n)]
    off += n * REQ_STATUS.size
    ids = np.frombuffer(body, dtype="<i4", count=n * N, offset=off).reshape(n, N)
    off += 4 * n * N
    conf = np.frombuffer(body, dtype="<f4", count=n * N, offset=off).reshape(n, N)
    off += 4 * n * N
    margin = None
    if flags & REPLY_HAS_MARGIN:
        margin = np.frombuffer(body, dtype="<f4", count=n * N, offset=off).reshape(n, N)
        off += 4 * n * N
    if off != len(body):
        raise ProtocolError(f"STEP reply {len(body)} B, parsed {off}")
    return StepReply(step_id, n, N, flags, wus, rts, sts, ids, conf, margin)


def serial_newer(a: int, b: int) -> bool:
    """a is newer than b in u32 serial-number arithmetic (RFC 1982)."""
    d = (a - b) & 0xFFFFFFFF
    return d != 0 and d < 0x80000000
