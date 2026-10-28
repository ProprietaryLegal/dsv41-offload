# SPDX-License-Identifier: Apache-2.0
"""S5 #3: the client refuses STEP replies whose values the target must not see.

A reply passes the shape, step and generation echo checks and still carries values the
target cannot take as input ids (outside [0, vocab_size)) or that the depth policy cannot
price (non-finite / out-of-range confidences), or contradicts CONTRACT 6.3 / 7.2 / 9.2
(unknown status, window bounds, n_out zero tail). Such a reply is a failed round
(bad_reply): the connection is dropped and every slot resyncs, never a usable draft.
"""

import time

import numpy as np
import pytest
import torch

from remote_dspark import wire
from remote_dspark.client import RemoteDraftClient
from remote_dspark.config import RemoteConfig
from remote_dspark.mock_worker import TcpStubServer, WorkerCore, fake_draft
from remote_dspark.round import STATUS_OK, STATUS_ROUND_FAILED, RemoteRound

import sim

ID = bytes(range(32))
N = 5
V = 129280
HELLO = wire.Hello(hidden=sim.H, row_bytes=sim.N_TAPS * sim.H * 2, drafter_identity=ID)


def cfg(**kw):
    env = {"VLLM_DSPARK_REMOTE": "tcp://127.0.0.1:1",
           "VLLM_DSPARK_REMOTE_IDENTITY": ID.hex(),
           "VLLM_DSPARK_REMOTE_POLICY": "capacity",
           "VLLM_DSPARK_REMOTE_LOG_EVERY": "0",
           "VLLM_DSPARK_REMOTE_STATS_FILE": ""}
    env.update(kw)
    return RemoteConfig.from_env(env)


class CannedWorker:
    """An RpcClient whose STEP reply is built by ``make(entries)`` (a StepReply)."""

    def __init__(self, make):
        self.make = make

    def request(self, msg, buf, timeout_s):
        h, body = wire.split_frame(msg)
        req = wire.decode_step(body, max_slots=HELLO.max_slots, max_rows=HELLO.max_rows_per_step,
                               expected_row_bytes=HELLO.row_bytes, window=HELLO.window)
        rep = self.make(req.entries)
        raw = wire.frame(wire.OP_STEP, h.seq, h.session, rep.pack())
        buf[:len(raw)] = raw
        return len(raw)

    def close(self):
        pass


def reply_for(entries, *, status=None, window_rows=None, ids=None, conf=None, margin=None):
    """A contract-conforming reply (OK, full window, valid drafts, zero tails), then the
    given overrides: dict {(i, j): value} for ids/conf/margin, {i: value} for the rest."""
    n = len(entries)
    st = np.zeros(n, np.uint16)
    wr = np.zeros(n, np.uint16)
    d = np.zeros((n, N), np.int32)
    c = np.zeros((n, N), np.float32)
    m = None if margin is None else np.zeros((n, N), np.float32)
    for i, e in enumerate(entries):
        wr[i] = min(HELLO.window, e.anchor_pos)
        if e.flags & wire.RF_NEED_DRAFT:
            k = N if e.n_out == 0 else e.n_out
            d[i, :k] = 1000 + np.arange(k)
            c[i, :k] = np.linspace(0.9, 0.1, k)
            if m is not None:
                m[i, :k] = 0.5
    for arr, over in ((st, status), (wr, window_rows)):
        for i, v in (over or {}).items():
            arr[i] = v
    for arr, over in ((d, ids), (c, conf), (m, margin)):
        for (i, j), v in (over or {}).items():
            arr[i, j] = v
    return wire.StepReply(step_id=1, num_spec=N, status=st, window_rows=wr,
                          generation=np.array([e.generation for e in entries], np.uint32),
                          draft_ids=d, confidence=c, margin=m)


def entry(slot=0, *, first=0, rows=1, flags=wire.RF_NEED_DRAFT, n_out=0, gen=1):
    return wire.ReqEntry(slot=slot, generation=gen, first_pos=first, n_rows=rows,
                         anchor_token=3, anchor_pos=first + rows, flags=flags, n_out=n_out)


def run_client(entries, **over):
    cli = RemoteDraftClient(cfg(), HELLO, background=False)
    cli._ready = True
    cli._client = CannedWorker(lambda es: reply_for(es, **over))
    n_rows = sum(e.n_rows for e in entries)
    buf = bytearray(wire.APP_HEADER_SIZE + wire.rows_offset_for(len(entries))
                    + n_rows * HELLO.row_bytes)
    return cli, cli.step(buf, entries, n_rows, step_id=1)


# --------------------------------------------------------------------- S5 reproductions


@pytest.mark.parametrize("token,confidence", [(-1, .9), (V, .9), (7, float("nan"))])
def test_s5_invalid_reply_values_refused(token, confidence):
    """S5 test_proxy_rejects_invalid_reply_values, verbatim inputs."""
    class Reply:
        def request(self, msg, buf, timeout_s):
            h, _ = wire.split_frame(msg)
            reply = wire.StepReply(1, 5, np.array([0]), np.array([1]), np.array([1]),
                                   np.full((1, 5), token, dtype=np.int32),
                                   np.full((1, 5), confidence, dtype=np.float32))
            raw = wire.frame(wire.OP_STEP, h.seq, h.session, reply.pack())
            buf[:len(raw)] = raw
            return len(raw)

        def close(self):
            pass
    h = wire.Hello(hidden=4, row_bytes=24)
    c = RemoteConfig.from_env({"VLLM_DSPARK_REMOTE": "tcp://127.0.0.1:1",
                               "VLLM_DSPARK_REMOTE_IDENTITY": bytes(32).hex(),
                               "VLLM_DSPARK_REMOTE_POLICY": "capacity",
                               "VLLM_DSPARK_REMOTE_DRAFT_MS": "4"})
    cli = RemoteDraftClient(c, h, background=False)
    cli._ready, cli._client = True, Reply()
    out = cli.step(bytearray(256), [wire.ReqEntry(0, 1, 0, 1, 3, 1, flags=2)], 1, step_id=1)
    assert not out.ok
    assert out.reason == "bad_reply"


# --------------------------------------------------------------------- client boundary


def test_valid_replies_pass():
    es = [entry(0, rows=1), entry(1, first=300, rows=6), entry(2, rows=40, flags=0)]
    cli, out = run_client(es)
    assert out.ok, cli.last_error
    # n_out: positions >= n_out are zeros and pass
    cli, out = run_client([entry(0, rows=3, n_out=4), entry(3, first=500, rows=2, n_out=4)])
    assert out.ok, cli.last_error
    # margins (finite, >= 0) pass; token V-1 and confidences 0.0 / 1.0 pass
    cli, out = run_client([entry(0, rows=2)], margin={}, ids={(0, 0): V - 1},
                          conf={(0, 0): 1.0, (0, 1): 0.0})
    assert out.ok, cli.last_error
    # PARTIAL_WINDOW with fewer rows than expected passes
    cli, out = run_client([entry(0, first=300, rows=6)], status={0: wire.RS_PARTIAL_WINDOW},
                          window_rows={0: 6})
    assert out.ok, cli.last_error
    # failed entries with zero drafts pass (their drafts are never consumed)
    for rs in (wire.RS_STALE_GENERATION, wire.RS_BAD_ENTRY, wire.RS_INTERNAL):
        cli, out = run_client([entry(0, rows=2)], status={0: rs}, window_rows={0: 0},
                              ids={(0, j): 0 for j in range(N)},
                              conf={(0, j): 0.0 for j in range(N)})
        assert out.ok, (rs, cli.last_error)


BAD = [
    # draft ids outside the vocabulary at a consumed position
    ("id_negative", [entry(0, rows=2)], dict(ids={(0, 2): -1})),
    ("id_vocab", [entry(0, rows=2)], dict(ids={(0, 4): V})),
    ("id_huge", [entry(1, first=9, rows=4)], dict(ids={(0, 0): 2**31 - 1})),
    # confidences that are not probabilities
    ("conf_nan", [entry(0, rows=2)], dict(conf={(0, 1): float("nan")})),
    ("conf_inf", [entry(0, rows=2)], dict(conf={(0, 1): float("inf")})),
    ("conf_neg", [entry(0, rows=2)], dict(conf={(0, 0): -0.25})),
    ("conf_gt1", [entry(0, rows=2)], dict(conf={(0, 3): 1.5})),
    # margins that are not top1 - top2
    ("margin_nan", [entry(0, rows=2)], dict(margin={(0, 0): float("nan")})),
    ("margin_neg", [entry(0, rows=2)], dict(margin={(0, 1): -1.0})),
    # unknown per-request status
    ("status_unknown", [entry(0, rows=2)], dict(status={0: 9})),
    # window bounds: more rows than the anchor can read, or than exist below it
    ("window_gt_W", [entry(0, first=1000, rows=6)], dict(window_rows={0: 129})),
    ("window_gt_anchor", [entry(0, rows=3)], dict(window_rows={0: 4})),
    ("window_gt_anchor_nodraft", [entry(0, rows=3, flags=0)], dict(window_rows={0: 7})),
    # status / window_rows disagree (CONTRACT 7.2 step 5)
    ("ok_but_short", [entry(0, first=1000, rows=6)], dict(window_rows={0: 100})),
    ("partial_but_full", [entry(0, first=1000, rows=6)],
     dict(status={0: wire.RS_PARTIAL_WINDOW})),
    # n_out zero tail (CONTRACT 1.3)
    ("tail_id", [entry(0, rows=2, n_out=3)], dict(ids={(0, 3): 17})),
    ("tail_conf", [entry(0, rows=2, n_out=3)], dict(conf={(0, 4): 0.5})),
    ("tail_conf_nan", [entry(0, rows=2, n_out=3)], dict(conf={(0, 4): float("nan")})),
    ("tail_margin", [entry(0, rows=2, n_out=4)], dict(margin={(0, 4): 0.5})),
    # entries whose drafts are not consumed must carry zeros
    ("nodraft_ids", [entry(0, rows=2, flags=0)], dict(ids={(0, 0): 5})),
    ("failed_entry_conf", [entry(0, rows=2)],
     dict(status={0: wire.RS_INTERNAL}, ids={(0, j): 0 for j in range(N)},
          conf={(0, 0): 0.0, (0, 1): 0.3, (0, 2): 0.0, (0, 3): 0.0, (0, 4): 0.0})),
    # the second entry is the bad one
    ("second_entry", [entry(0, rows=2), entry(2, first=50, rows=1)], dict(ids={(1, 0): V})),
]


@pytest.mark.parametrize("name,entries,over", BAD, ids=[b[0] for b in BAD])
def test_invalid_values_are_bad_reply(name, entries, over):
    cli, out = run_client(entries, **over)
    assert not out.ok and out.reason == "bad_reply" and out.unknown_apply, name
    assert out.reply is None
    assert not cli.ready                       # the existing path: drop + reconnect + resync
    assert "bad reply" in cli.last_error
    assert cli.stats.snapshot()["last_bad_reply"]


# --------------------------------------------------------------------- the whole round


@pytest.fixture
def server():
    srv = TcpStubServer(WorkerCore(HELLO), "127.0.0.1", 0).start()
    if not hasattr(srv.core.inject, "reply_hook_once"):   # another worker core (O2 conformance)
        srv.stop()
        pytest.skip("needs the reference worker's reply hook")
    yield srv
    srv.stop()


@pytest.mark.parametrize("corrupt", ["id_vocab", "conf_nan", "tail_nonzero"])
def test_round_refuses_corrupt_reply_then_resyncs_exactly(server, corrupt):
    """A worker reply with unsafe values fails that round (zero drafts on every rank) and
    takes the existing resync path: after the reconnect every usable draft is exact."""
    c = cfg(VLLM_DSPARK_REMOTE=f"tcp://127.0.0.1:{server.port}",
            VLLM_DSPARK_REMOTE_TIMEOUT_MS="200", VLLM_DSPARK_REMOTE_BACKOFF_MS="2",
            VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS="4")
    cap = [0, 5, 5, 4, 4] if corrupt == "tail_nonzero" else None
    rnd = RemoteRound(c, HELLO, device=torch.device("cpu"), tp_rank=0, broadcast_fn=None,
                      background=True, n_out_by_reqs=cap)
    t = time.monotonic() + 3
    while not rnd.client.ready and time.monotonic() < t:
        time.sleep(0.002)
    assert rnd.client.ready

    def hook(rep, req):
        for i, e in enumerate(req.entries):
            if not e.flags & wire.RF_NEED_DRAFT:
                continue
            if corrupt == "id_vocab":
                rep.draft_ids[i, 0] = V
            elif corrupt == "conf_nan":
                rep.confidence[i, 1] = np.nan
            elif e.n_out:
                rep.draft_ids[i, e.n_out] = 1
            else:
                continue
            return True
        return False

    eng = sim.Engine(seed=21)
    for s in range(sim.M):
        eng.new_request(s, prompt_len=150)
    corrupted_at = None
    exact_after = 0
    for step in range(160):
        if step >= 20 and corrupted_at is None:
            server.core.inject.reply_hook_once = hook
        sd = eng.step()
        res = rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected, sd.last_sampled,
                      sd.next_prefill_tokens)
        st = res.read_status()
        drafts, conf, ok = res.drafts.numpy(), res.conf.numpy(), res.ok_mask.numpy()
        if corrupted_at is None and server.core.inject.reply_hook_fired:
            corrupted_at = step
            assert st == STATUS_ROUND_FAILED
            assert not drafts.any() and not conf.any() and not ok.any()
            assert np.isfinite(conf).all()
        if st != STATUS_OK:
            time.sleep(0.01)
            continue
        for i, r in enumerate(sd.reqs):
            if not sd.need_draft[i]:
                continue
            assert ok[i]
            want_ids, want_conf = fake_draft(eng.expected_window(r), r.anchor, r.computed,
                                             sim.N, sim.VOCAB)
            k = rnd.n_out_for(sd.batch.num_reqs) or sim.N
            assert np.array_equal(drafts[i, :k], want_ids[:k])
            assert np.array_equal(conf[i, :k], want_conf[:k])
            if corrupted_at is not None:
                exact_after += 1
    assert corrupted_at is not None
    cnt = rnd.stats.snapshot()["counters"]
    assert cnt["fallback_bad_reply"] == 1
    assert cnt["resync_rounds"] >= 1 and cnt["hellos"] == 2
    assert exact_after > 20
    rnd.client.close()
