# SPDX-License-Identifier: Apache-2.0
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from remote_dspark import wire

CONTRACT = Path(__file__).resolve().parents[2] / "CONTRACT.md"


def test_struct_sizes_match_contract():
    assert wire.APP_HEADER.size == 32
    assert wire.HELLO_REQ.size == 192
    assert wire.HELLO_REP.size == 128
    assert wire.STEP_HEADER.size == 32
    assert wire.REQ_ENTRY.size == 32
    assert wire.REPLY_HEADER.size == 32
    assert wire.REQ_STATUS.size == 8


def test_contract_version_matches_doc():
    text = CONTRACT.read_text()
    m = re.search(r"Contract version \*\*([0-9.]+)\*\*", text)
    assert m and m.group(1) == wire.CONTRACT_VERSION


def test_app_header_offsets():
    h = wire.AppHeader(op=wire.OP_STEP, seq=0x0102030405060708, session=0xA1B2C3D4,
                       body_len=77, status=-5, flags=9)
    b = h.pack()
    assert b[:4] == b"DSP1"
    assert int.from_bytes(b[4:6], "little") == 1
    assert int.from_bytes(b[6:8], "little") == wire.OP_STEP
    assert int.from_bytes(b[8:16], "little") == 0x0102030405060708
    assert int.from_bytes(b[16:20], "little") == 0xA1B2C3D4
    assert int.from_bytes(b[20:24], "little") == 77
    assert int.from_bytes(b[24:28], "little", signed=True) == -5
    h2 = wire.AppHeader.unpack_from(b)
    assert h2 == h


def test_hello_offsets_and_roundtrip():
    h = wire.Hello(drafter_identity=bytes(range(32)), markov_identity=bytes(32),
                   client_name="spark tp0")
    b = h.pack()
    u32 = lambda off: int.from_bytes(b[off:off + 4], "little")  # noqa: E731
    assert (u32(8), u32(12), u32(16), u32(20)) == (5, 128, 5120, 3)
    assert (u32(24), u32(28), u32(32), u32(36)) == (37, 38, 39, 0)
    assert (u32(40), u32(44), u32(48), u32(52)) == (1, 30720, 128799, 129280)
    assert (u32(56), u32(60), u32(64), u32(68)) == (327680, 4, 512, 1)
    assert b[80:112] == bytes(range(32))
    assert b[144:153] == b"spark tp0"
    assert wire.Hello.unpack(b) == h


def test_req_entry_offsets():
    e = wire.ReqEntry(slot=3, generation=0xFFFFFFFE, first_pos=123456789012, n_rows=6,
                      anchor_token=128799, anchor_pos=123456789018,
                      flags=wire.RF_NEW_GEN | wire.RF_NEED_DRAFT, k_window=5)
    b = bytearray(32)
    e.pack_into(b, 0)
    assert b[0] == 3 and b[1] == 3 and b[2] == 5
    assert int.from_bytes(b[4:8], "little") == 0xFFFFFFFE
    assert int.from_bytes(b[8:16], "little") == 123456789012
    assert int.from_bytes(b[16:20], "little") == 6
    assert int.from_bytes(b[20:24], "little") == 128799
    assert int.from_bytes(b[24:32], "little") == 123456789018
    assert wire.ReqEntry.unpack_from(b, 0) == e


def _entries():
    return [wire.ReqEntry(slot=0, generation=1, first_pos=10, n_rows=2, anchor_token=7,
                          anchor_pos=12, flags=wire.RF_NEED_DRAFT),
            wire.ReqEntry(slot=2, generation=4, first_pos=0, n_rows=3, anchor_token=9,
                          anchor_pos=3)]


def test_step_roundtrip_and_alignment():
    rb = 12
    rows = np.arange(5 * rb, dtype=np.uint8).reshape(5, rb)
    body = wire.encode_step(42, _entries(), rows, rb, wire.SF_WANT_MARGIN)
    assert wire.rows_offset_for(2) == 128 and len(body) == 128 + 5 * rb
    req = wire.decode_step(body, max_slots=4, max_rows=512, expected_row_bytes=rb, window=128)
    assert req.step_id == 42 and req.step_flags == wire.SF_WANT_MARGIN
    assert req.entries == _entries()
    assert np.array_equal(req.rows, rows)
    for n in range(1, 9):
        assert wire.rows_offset_for(n) % 64 == 0
        assert wire.rows_offset_for(n) >= 32 + 32 * n


@pytest.mark.parametrize("mutate,err", [
    (lambda b: b.__setitem__(slice(16, 20), (9).to_bytes(4, "little")), "row_bytes"),
    (lambda b: b.__setitem__(slice(8, 10), (0).to_bytes(2, "little")), "n_req"),
    (lambda b: b.__setitem__(slice(20, 24), (64).to_bytes(4, "little")), "rows_offset"),
    (lambda b: b.__setitem__(32 + 32, 0), "slot"),            # second entry slot = 0 again
    (lambda b: b.__setitem__(slice(32 + 16, 32 + 20), (3).to_bytes(4, "little")), "rows"),
])
def test_step_malformed(mutate, err):
    rb = 12
    body = bytearray(wire.encode_step(1, _entries(), np.zeros((5, rb), np.uint8), rb))
    mutate(body)
    with pytest.raises(wire.WireError, match=err):
        wire.decode_step(bytes(body), max_slots=4, max_rows=512, expected_row_bytes=rb,
                         window=128)


def test_step_too_many_rows():
    rb = 4
    e = [wire.ReqEntry(0, 1, 0, 3, 1, 3)]
    body = wire.encode_step(1, e, np.zeros((3, rb), np.uint8), rb)
    with pytest.raises(wire.WireError, match="max_rows"):
        wire.decode_step(body, max_slots=4, max_rows=2, expected_row_bytes=rb, window=128)


def test_reply_roundtrip():
    n, N = 3, 5
    rep = wire.StepReply(step_id=9, num_spec=N, status=np.array([0, 1, 3], np.uint16),
                         window_rows=np.array([128, 7, 0], np.uint16),
                         generation=np.array([1, 2, 0xFFFFFFFF], np.uint32),
                         draft_ids=np.arange(n * N, dtype=np.int32).reshape(n, N),
                         confidence=np.linspace(0, 1, n * N, dtype=np.float32).reshape(n, N),
                         worker_us=1234, recv_to_send_us=1300)
    b = rep.pack()
    assert len(b) == 32 + 8 * n + 8 * n * N
    got = wire.StepReply.unpack(b, expect_n_req=n, expect_num_spec=N)
    assert got.step_id == 9 and got.margin is None
    for f in ("status", "window_rows", "generation", "draft_ids", "confidence"):
        assert np.array_equal(getattr(got, f), getattr(rep, f))
    rep.margin = np.ones((n, N), np.float32)
    got = wire.StepReply.unpack(rep.pack())
    assert got.reply_flags & wire.RPF_HAS_MARGIN and np.array_equal(got.margin, rep.margin)
    with pytest.raises(wire.WireError):
        wire.StepReply.unpack(b + b"x")
    with pytest.raises(wire.WireError):
        wire.StepReply.unpack(b, expect_n_req=2)


def test_frame_checks():
    f = wire.frame(wire.OP_PING, 5, 77)
    h, body = wire.split_frame(f)
    assert h.op == wire.OP_PING and h.seq == 5 and len(body) == 0
    with pytest.raises(wire.WireError, match="magic"):
        wire.split_frame(b"XXXX" + f[4:])
    with pytest.raises(wire.WireError, match="body_len"):
        wire.split_frame(f + b"\0")
    with pytest.raises(wire.WireError, match="short"):
        wire.split_frame(f[:10])


def test_gen_newer_wraps():
    assert wire.gen_newer(2, 1)
    assert not wire.gen_newer(1, 2)
    assert not wire.gen_newer(5, 5)
    assert wire.gen_newer(0, 0xFFFFFFFF)
    assert not wire.gen_newer(0xFFFFFFFF, 0)


def test_drafter_identity_definition():
    files = {"b.safetensors": "b" * 64, "a.json": "A" * 64}
    expect = hashlib.sha256(f"a.json {'a' * 64}\nb.safetensors {'b' * 64}\n".encode()).digest()
    assert wire.drafter_identity(files) == expect
    with pytest.raises(ValueError):
        wire.drafter_identity({"x": "zz"})
    with pytest.raises(ValueError):
        wire.drafter_identity({"dir/x": "a" * 64})
