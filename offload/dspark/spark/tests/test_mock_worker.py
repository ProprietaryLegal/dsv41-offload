# SPDX-License-Identifier: Apache-2.0
"""The contract's worker-side rules (CONTRACT 6, 7.2, 9) on the reference WorkerCore."""

import json

import numpy as np

from remote_dspark import wire
from remote_dspark.mock_worker import WorkerCore, fake_draft

ID = bytes(range(32))
H = 4            # small hidden for tests: row = 3 taps x 4 x 2 B = 24 B
RB = 3 * H * 2


def hello_cfg(**kw):
    base = dict(hidden=H, row_bytes=RB, drafter_identity=ID)
    base.update(kw)
    return wire.Hello(**base)


def mk_core():
    return WorkerCore(hello_cfg())


def call(core, op, seq, session, body=b""):
    rep = core.handle(wire.frame(op, seq, session, body))
    h, b = wire.split_frame(rep)
    return h, bytes(b)


def rows_for(positions, salt=0):
    out = np.zeros((len(positions), RB), np.uint8)
    for k, p in enumerate(positions):
        out[k] = (np.arange(RB) + p * 7 + salt) % 251
    return out


def step(core, seq, session, entries, rows, flags=0, step_id=1):
    body = wire.encode_step(step_id, entries, rows, RB, flags)
    h, b = call(core, wire.OP_STEP, seq, session, body)
    if h.status != wire.ST_OK:
        return h, None
    return h, wire.StepReply.unpack(b, expect_n_req=len(entries), expect_num_spec=5)


def expect(positions_rows, anchor_token, anchor_pos):
    return fake_draft([(p, r.tobytes()) for p, r in positions_rows], anchor_token, anchor_pos,
                      5, 129280)


def test_hello_session_and_identity():
    core = mk_core()
    h, b = call(core, wire.OP_HELLO, 1, 0xAA, hello_cfg().pack())
    assert h.status == wire.ST_OK
    rep = wire.HelloReply.unpack(b)
    assert rep.drafter_identity == ID and rep.capabilities & wire.CAP_TAPS_BF16
    h, _ = call(core, wire.OP_HELLO, 2, 0xAA,
                hello_cfg(drafter_identity=bytes(32)).pack())
    assert h.status == wire.ST_IDENTITY_MISMATCH
    h, _ = call(core, wire.OP_HELLO, 3, 0xAA, hello_cfg(window=64).pack())
    assert h.status == wire.ST_CONFIG_MISMATCH
    h, _ = call(core, wire.OP_HELLO, 4, 0xAA, hello_cfg(noise_token_id=1).pack())
    assert h.status == wire.ST_CONFIG_MISMATCH


def test_no_session_stale_seq_unsupported():
    core = mk_core()
    e = [wire.ReqEntry(0, 1, 0, 1, 5, 1, flags=wire.RF_NEED_DRAFT)]
    h, _ = step(core, 1, 0xAB, e, rows_for([0]))
    assert h.status == wire.ST_NO_SESSION
    call(core, wire.OP_HELLO, 10, 0xAB, hello_cfg().pack())
    h, _ = step(core, 10, 0xAB, e, rows_for([0]))
    assert h.status == wire.ST_STALE_SEQ
    h, rep = step(core, 11, 0xAB, e, rows_for([0]))
    assert h.status == wire.ST_OK and rep.status[0] == wire.RS_OK
    h, _ = step(core, 12, 0xAC, e, rows_for([0]))
    assert h.status == wire.ST_NO_SESSION
    h, _ = call(core, 99, 13, 0xAB)
    assert h.status == wire.ST_UNSUPPORTED_OP
    h, _ = call(core, wire.OP_PING, 1, 0)            # PING needs no session
    assert h.status == wire.ST_OK


def test_window_drafts_follow_contract():
    core = mk_core()
    S = 0x55
    call(core, wire.OP_HELLO, 1, S, hello_cfg().pack())
    truth = {}
    seq = 1
    # prefill chunk: trailing 128 rows of positions [0, 300)
    pos = list(range(172, 300))
    r = rows_for(pos)
    truth.update(zip(pos, r))
    seq += 1
    e = [wire.ReqEntry(1, 7, 172, 128, 42, 300, flags=wire.RF_NEED_DRAFT | wire.RF_NEW_GEN)]
    h, rep = step(core, seq, S, e, r)
    want = expect([(p, truth[p]) for p in range(172, 300)], 42, 300)
    assert rep.status[0] == wire.RS_OK and rep.window_rows[0] == 128
    assert np.array_equal(rep.draft_ids[0], want[0]) and np.allclose(rep.confidence[0], want[1])
    # decode: 4 valid rows starting at the anchor position
    pos = list(range(300, 304))
    r = rows_for(pos, salt=3)
    truth.update(zip(pos, r))
    seq += 1
    h, rep = step(core, seq, S, [wire.ReqEntry(1, 7, 300, 4, 9, 304, flags=wire.RF_NEED_DRAFT)], r)
    want = expect([(p, truth[p]) for p in range(304 - 128, 304)], 9, 304)
    assert np.array_equal(rep.draft_ids[0], want[0])
    # rollback (re-add at 290): rows >= 290 replaced
    pos = list(range(290, 295))
    r = rows_for(pos, salt=9)
    truth.update(zip(pos, r))
    for p in range(295, 304):
        truth.pop(p)
    seq += 1
    h, rep = step(core, seq, S, [wire.ReqEntry(1, 7, 290, 5, 11, 295, flags=wire.RF_NEED_DRAFT)], r)
    # rows 167..175 left the window at the decode step (lo = 304-128 = 176): partial
    want = expect([(p, truth[p]) for p in range(176, 295)], 11, 295)
    assert np.array_equal(rep.draft_ids[0], want[0])
    assert rep.status[0] == wire.RS_PARTIAL_WINDOW and rep.window_rows[0] == 295 - 176


def test_generation_rules_and_partial_window():
    core = mk_core()
    S = 0x77
    call(core, wire.OP_HELLO, 1, S, hello_cfg().pack())
    h, rep = step(core, 2, S, [wire.ReqEntry(0, 5, 0, 3, 1, 3, flags=wire.RF_NEED_DRAFT)],
                  rows_for(range(3)))
    assert rep.status[0] == wire.RS_OK                  # anchor 3: expected rows = 3
    h, rep = step(core, 3, S, [wire.ReqEntry(0, 4, 3, 1, 1, 4, flags=wire.RF_NEED_DRAFT)],
                  rows_for([3]))
    assert rep.status[0] == wire.RS_STALE_GENERATION and rep.draft_ids[0].sum() == 0
    # newer generation: window restarts -> only 2 rows near position 500: partial
    h, rep = step(core, 4, S, [wire.ReqEntry(0, 6, 498, 2, 1, 500, flags=wire.RF_NEED_DRAFT)],
                  rows_for([498, 499]))
    assert rep.status[0] == wire.RS_PARTIAL_WINDOW and rep.window_rows[0] == 2
    # inconsistent entry
    h, rep = step(core, 5, S, [wire.ReqEntry(0, 6, 500, 2, 1, 503, flags=wire.RF_NEED_DRAFT)],
                  rows_for([500, 501]))
    assert rep.status[0] == wire.RS_BAD_ENTRY
    # append-only entry (no NEED_DRAFT) returns zeros but keeps rows
    h, rep = step(core, 6, S, [wire.ReqEntry(2, 1, 0, 2, 1, 2)], rows_for([0, 1]))
    assert rep.status[0] == wire.RS_OK and rep.draft_ids[0].sum() == 0
    assert core.slots[2].rng.hi == 1


def test_reset_stats_margin():
    core = mk_core()
    S = 0x99
    call(core, wire.OP_HELLO, 1, S, hello_cfg().pack())
    h, rep = step(core, 2, S, [wire.ReqEntry(3, 1, 0, 2, 1, 2, flags=wire.RF_NEED_DRAFT)],
                  rows_for([0, 1]), flags=wire.SF_WANT_MARGIN)
    assert rep.margin is not None
    h, _ = call(core, wire.OP_RESET, 3, S, wire.RESET_BODY.pack(1 << 3))
    assert h.status == wire.ST_OK and core.slots[3].rng.empty
    h, b = call(core, wire.OP_STATS, 4, S)
    st = json.loads(b)
    assert st["steps"] == 1 and st["resets"] == 1


def test_new_session_drops_state_rehello_keeps():
    core = mk_core()
    call(core, wire.OP_HELLO, 1, 1, hello_cfg().pack())
    step(core, 2, 1, [wire.ReqEntry(0, 1, 0, 2, 1, 2)], rows_for([0, 1]))
    call(core, wire.OP_HELLO, 3, 1, hello_cfg().pack())     # reconnect, same session
    assert not core.slots[0].rng.empty
    call(core, wire.OP_HELLO, 1, 2, hello_cfg().pack())     # new session (engine restart)
    assert core.slots[0].rng.empty


def test_n_out_truncates_to_zeros_and_keeps_prefix():
    core = mk_core()
    S = 0x31
    call(core, wire.OP_HELLO, 1, S, hello_cfg().pack())
    pos = list(range(0, 10))
    r = rows_for(pos)
    full = step(core, 2, S, [wire.ReqEntry(0, 1, 0, 10, 77, 10, flags=wire.RF_NEED_DRAFT)], r)[1]
    core2 = mk_core()
    call(core2, wire.OP_HELLO, 1, S, hello_cfg().pack())
    cut = step(core2, 2, S, [wire.ReqEntry(0, 1, 0, 10, 77, 10, flags=wire.RF_NEED_DRAFT,
                                           n_out=4)], r)[1]
    assert np.array_equal(cut.draft_ids[0, :4], full.draft_ids[0, :4])
    assert np.array_equal(cut.confidence[0, :4], full.confidence[0, :4])
    assert cut.draft_ids[0, 4] == 0 and cut.confidence[0, 4] == 0.0
    b = bytearray(32)
    wire.ReqEntry(0, 1, 0, 1, 1, 1, n_out=4).pack_into(b, 0)
    assert b[3] == 4 and wire.ReqEntry.unpack_from(b, 0).n_out == 4
