"""DW1 A2: DraftCore routes a drafting STEP through backend.flush_draft when the backend has one (LeanGpuBackend), and
a STEP without drafts through flush(); replies equal the flush + draft path's. CPU only (fake backends)."""

import numpy as np

from worker import protocol as P
from worker.server import DraftCore, FakeBackend, expected_hello

HIDDEN = 4
ROW_BYTES = 3 * HIDDEN * 2
SESSION = 0x5E56


def fake_draft(win, tok, apos, N, vocab):
    return np.array([(tok + j + 1 + len(win)) % vocab for j in range(N)], np.int32), np.full(N, 0.25, np.float32)


def expected() -> P.Hello:
    e = expected_hello(bytes(32), bytes(32))
    e.hidden, e.row_bytes = HIDDEN, ROW_BYTES
    return e


class Counting(FakeBackend):
    def __init__(self, exp):
        super().__init__(exp.window, exp.row_bytes, exp.max_slots, fake_draft, exp.vocab_size)
        self.calls: list[str] = []

    def flush(self):
        self.calls.append("flush")

    def draft(self, reqs, N, want_margin):
        self.calls.append("draft")
        return super().draft(reqs, N, want_margin)


class CountingLean(Counting):
    def flush_draft(self, reqs, N, want_margin):
        self.calls.append("flush_draft")
        return FakeBackend.draft(self, reqs, N, want_margin)


def run(backend_cls):
    exp = expected()
    be = backend_cls(exp)
    core = DraftCore(exp, be)
    seq = [0]

    def call(op, body=b""):
        seq[0] += 1
        rep = core.handle(P.AppHeader(P.MAGIC, P.VERSION, op, seq[0], SESSION, len(body), 0, 0).encode() + body)
        return P.AppHeader.decode(rep), rep[P.APP_HDR.size:]

    assert call(P.OP_HELLO, P.Hello(**{**exp.__dict__, "client_name": "t"}).encode())[0].status == P.OK
    replies = []
    for ents in ([P.ReqEntry(0, P.REQ_NEED_DRAFT, 0, 3, 0, 4, 9, 4)],
                 [P.ReqEntry(0, 0, 0, 3, 4, 2, 9, 6)],                                # rows only, no draft
                 [P.ReqEntry(0, P.REQ_NEED_DRAFT, 0, 3, 6, 1, 11, 7),
                  P.ReqEntry(1, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT, 0, 5, 0, 3, 12, 3)]):
        rows = b"".join(np.full(e.n_rows * ROW_BYTES, e.slot + 1, np.uint8).tobytes() for e in ents)
        h, b = call(P.OP_STEP, P.encode_step(seq[0] + 1, ents, rows, ROW_BYTES))
        assert h.status == P.OK
        r = P.decode_step_reply(b)
        replies.append((r.statuses, r.draft_ids.tolist(), r.conf.tolist()))
    return be.calls, replies


def test_flush_draft_used_for_drafting_steps_only_and_replies_equal():
    calls_ref, rep_ref = run(Counting)
    calls_lean, rep_lean = run(CountingLean)
    assert calls_ref == ["flush", "draft", "flush", "flush", "draft"]
    assert calls_lean == ["flush_draft", "flush", "flush_draft"]
    assert rep_lean == rep_ref
