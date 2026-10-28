"""D4 self-test of the real worker (GPU backend) through the contract, on x86host.

1. HELLO: the live identity is accepted, a wrong one refused (IDENTITY_MISMATCH), a wrong config refused.
2. Exactness of the server plumbing: a simulated engine drives slot 0 through a 300-token prompt (trailing 128 rows,
   NEED_DRAFT) and 40 decode rounds (1 + accepted rows each, accepted drawn at random), while the same inserts and
   drafts are made directly on the model in a shadow slot. Server drafts and confidences must equal the shadow's bit
   for bit.
3. Resync: slot 1 receives the same history as one NEW_GEN|RESYNC window; its drafts are compared with slot 0's
   (different insert batching, so equality is reported, not required).
4. Batching: 4 requests in one STEP vs the same requests one per STEP (reported).
5. Rollback (n_rows = 0 to an earlier anchor) and PARTIAL_WINDOW on a fresh short window.
6. Latency over the wire: an ofl server on 10.202.28.1:<port> (RDMA + TCP backends) and an ofl client on x86host
   (10.202.28.1 -> itself), decode STEPs of 1 and 4 requests: round trip p50 / p99 vs in-process handle time.
Writes argv[2] (JSON). Usage: python -m worker.tools.server_selftest ROOT OUT.json [port] [tp]
tp = 2 (lane E): the TP = 2 worker on the NVLink pair (or, with DSPARK_WORKER_TP_EMULATE=1, both ranks on one card);
the shadow calls then go through TPDrafter's host-input graph API.
"""

from __future__ import annotations

import json
import os
import random
import struct
import sys
import time

import numpy as np
import torch

from .. import protocol as P
from ..server import OflServer, _load_ofl, build_core

SESSION = 0x5E1F7E57


class Client:
    """Frames for an in-process core (handle) or an ofl connection."""

    def __init__(self, core=None, ofl_client=None, region=None, resp_buf=None):
        self.core, self.c, self.region, self.resp = core, ofl_client, region, resp_buf
        self.seq = 0

    def call(self, op: int, body: bytes = b"", session: int = SESSION):
        self.seq += 1
        frame = P.AppHeader(P.MAGIC, P.VERSION, op, self.seq, session, len(body), 0, 0).encode() + body
        if self.core is not None:
            rep = self.core.handle(frame)
        else:
            res = self.c.call(op, frame, [(self.region, 0, len(self.resp))], timeout_s=2.0, generation=self.seq)
            rep = bytes(self.resp[: res.resp_len])
        h = P.AppHeader.decode(rep)
        return h, rep[P.APP_HDR.size:]


class Shadow1:
    """Direct calls on the one-card model (model.py)."""

    def __init__(self, m):
        self.m, self.dev, self.embed = m, m.dev, m.embed

    def reset(self, slot):
        self.m.reset_slots(torch.tensor([slot], device=self.dev))

    def insert(self, slot, p0, n, taps):
        sr = torch.full((n,), slot, dtype=torch.int64, device=self.dev)
        pr = torch.arange(p0, p0 + n, device=self.dev)
        tt = torch.from_numpy(taps[p0:p0 + n].copy()).to(self.dev).view(torch.bfloat16).view(n, -1)
        self.m.insert_taps_graphed(sr, pr, tt)

    def draft(self, slot, anchor_tok, anchor_pos, lo, N):
        s = torch.tensor([slot], device=self.dev)
        ids, conf = self.m.draft_graphed(s, torch.tensor([anchor_tok], device=self.dev),
                                         torch.tensor([anchor_pos - 1], device=self.dev), N,
                                         ctx_lo=torch.tensor([lo], device=self.dev))
        return ids[0].cpu().numpy().astype(np.int32), conf[0].cpu().numpy()


class ShadowTP:
    """Direct calls on the TP = 2 model (tp2.TPDrafter): host (pinned) inputs."""

    def __init__(self, m):
        self.m, self.dev, self.embed = m, m.dev, m.ranks[0].embed

    def reset(self, slot):
        self.m.reset_slots([slot])

    def insert(self, slot, p0, n, taps):
        sr = torch.full((n,), slot, dtype=torch.int64).pin_memory()
        pr = torch.arange(p0, p0 + n).pin_memory()
        tt = torch.from_numpy(taps[p0:p0 + n].copy()).view(torch.bfloat16).view(n, -1).pin_memory()
        self.m.insert_taps_graphed(sr, pr, tt)

    def draft(self, slot, anchor_tok, anchor_pos, lo, N):
        inp = torch.tensor([[slot], [anchor_tok], [anchor_pos - 1], [lo]], dtype=torch.int64).pin_memory()
        ids, conf, _ = self.m.draft_graphed(inp, N)
        return ids[0].cpu().numpy().astype(np.int32), conf[0].cpu().numpy()


def taps_for(model, toks: torch.Tensor, gen: torch.Generator) -> np.ndarray:
    e = model.embed[toks].float()
    t = torch.cat([e * 40.0, e * 70.0, e * 110.0], -1) + torch.randn(e.shape[0], 3 * e.shape[1], generator=gen,
                                                                   device=e.device) * 0.5
    return t.to(torch.bfloat16).contiguous().view(torch.uint8).cpu().numpy()       # [n, 30720] little-endian bf16


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "/opt/dsv41/x86host-data/dspark"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "d4_server_selftest.json"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 18629
    tp = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    rep: dict = {"tp": tp}
    core, m = build_core(root, graphs=True, tp=tp)
    sh = ShadowTP(m) if tp == 2 else Shadow1(m)
    exp = core.expected
    dev = m.dev
    rb, N, W = exp.row_bytes, exp.num_spec, exp.window
    cl = Client(core=core)

    # ---- 1. HELLO
    h, b = cl.call(P.OP_HELLO, P.Hello(**{**exp.__dict__, "drafter_identity": bytes(32)}).encode())
    rep["hello_wrong_identity"] = P.FRAME_STATUS[h.status]
    h, b = cl.call(P.OP_HELLO, P.Hello(**{**exp.__dict__, "num_spec": 4}).encode())
    rep["hello_wrong_config"] = P.FRAME_STATUS[h.status]
    h, b = cl.call(P.OP_HELLO, P.Hello(**{**exp.__dict__, "client_name": "selftest"}).encode())
    rep["hello_live"] = P.FRAME_STATUS[h.status]
    assert h.status == P.OK, rep

    # ---- 2. exactness against shadow model calls
    gen = torch.Generator(device=dev).manual_seed(7)
    rnd = random.Random(7)
    L = 1600
    toks = torch.randint(0, 120000, (L,), generator=gen, device=dev)
    taps = taps_for(sh, toks, gen)
    SH = 4                                   # shadow slot (model slots 4..7 are outside the contract's 0..3)

    def shadow_insert(slot, p0, n):
        sh.insert(slot, p0, n, taps)

    def shadow_draft(slot, anchor_pos, lo):
        return sh.draft(slot, int(toks[anchor_pos]), anchor_pos, lo, N)

    def step(entries, rows, flags=0):
        body = P.encode_step(1, entries, rows, rb, flags)
        h, b = cl.call(P.OP_STEP, body)
        assert h.status == P.OK, P.FRAME_STATUS[h.status]
        return P.decode_step_reply(b)

    sh.reset(SH)
    gen0 = 1
    a_pos = 300
    first = a_pos - W
    r = step([P.ReqEntry(0, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT, 0, gen0, first, W, int(toks[a_pos]), a_pos)],
             taps[first:a_pos].tobytes())
    shadow_insert(SH, first, W)
    lo = first
    same = []
    sd = shadow_draft(SH, a_pos, max(lo, a_pos - W))
    same.append(bool(np.array_equal(r.draft_ids[0], sd[0]) and np.array_equal(r.conf[0], sd[1])))
    rep["prefill_status"] = r.statuses[0]
    for rd in range(40):
        acc = rnd.randint(0, N)                      # bonus + acc accepted rows
        n = 1 + acc
        p0 = a_pos
        a_pos = p0 + n
        r = step([P.ReqEntry(0, P.REQ_NEED_DRAFT, N, gen0, p0, n, int(toks[a_pos]), a_pos)], taps[p0:a_pos].tobytes())
        shadow_insert(SH, p0, n)
        lo = max(lo, a_pos - 1 - W + 1)
        sd = shadow_draft(SH, a_pos, max(lo, a_pos - W))
        same.append(bool(np.array_equal(r.draft_ids[0], sd[0]) and np.array_equal(r.conf[0], sd[1])
                         and r.statuses[0][0] == P.R_OK and r.statuses[0][1] == W))
    rep["exact_vs_shadow"] = {"rounds": len(same), "identical": sum(same)}
    last_ids0 = r.draft_ids[0].copy()

    # ---- 3. resync of the same history into slot 1
    first = a_pos - W
    r1 = step([P.ReqEntry(1, P.REQ_NEW_GEN | P.REQ_RESYNC | P.REQ_NEED_DRAFT, 0, 77, first, W, int(toks[a_pos]),
                          a_pos)], taps[first:a_pos].tobytes())
    rep["resync"] = {"status": r1.statuses[0], "drafts_equal_slot0": bool(np.array_equal(r1.draft_ids[0], last_ids0)),
                     "slot0": last_ids0.tolist(), "slot1": r1.draft_ids[0].tolist()}

    # ---- 4. batching: four requests with distinct histories, one STEP vs one STEP each
    starts = [500, 650, 800, 950]
    ents, rows = [], []
    for s_, st in enumerate(starts):
        ents.append(P.ReqEntry(s_, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT, 0, 100 + s_, st - W, W, int(toks[st]), st))
        rows.append(taps[st - W:st])
    rb4 = step(ents, np.concatenate(rows).tobytes())
    singles = []
    for s_, st in enumerate(starts):
        r_ = step([P.ReqEntry(s_, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT, 0, 200 + s_, st - W, W, int(toks[st]), st)],
                  taps[st - W:st].tobytes())
        singles.append(r_.draft_ids[0])
    rep["batch4_vs_single"] = {"statuses": rb4.statuses, "positions_equal": int(sum(
        int((rb4.draft_ids[i] == singles[i]).sum()) for i in range(4))), "positions": 4 * N}

    # ---- 4b. CONTRACT 1.3 n_out: the same 4-request window with n_out = 4 (new generation, same rows, same batch).
    # Generations only ever increase per slot (CONTRACT 7.2: an older one is stale even with NEW_GEN).
    ents = [P.ReqEntry(s_, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT, 0, 250 + s_, st - W, W, int(toks[st]), st, n_out=4)
            for s_, st in enumerate(starts)]
    rn4 = step(ents, np.concatenate(rows).tobytes())
    rep["n_out4_vs_full"] = {
        "statuses": rn4.statuses,
        "prefix_bit_identical": bool(np.array_equal(rn4.draft_ids[:, :4], rb4.draft_ids[:, :4])
                                     and np.array_equal(rn4.conf[:, :4], rb4.conf[:, :4])),
        "tail_zero": bool((rn4.draft_ids[:, 4:] == 0).all() and (rn4.conf[:, 4:] == 0).all())}
    assert rep["n_out4_vs_full"]["prefix_bit_identical"] and rep["n_out4_vs_full"]["tail_zero"], rep["n_out4_vs_full"]

    # ---- 5. rollback and partial window
    st = starts[0]
    rr = step([P.ReqEntry(0, P.REQ_NEED_DRAFT, 0, 250, 0, 0, int(toks[st - 3]), st - 3)], b"")
    rep["rollback_n0"] = {"status": rr.statuses[0], "window_rows": rr.statuses[0][1]}
    # rolling back 3 positions leaves W - 3 rows (history below the window is not restored): PARTIAL_WINDOW
    assert rr.statuses[0][0] == P.R_PARTIAL_WINDOW and rr.statuses[0][1] == W - 3, rr.statuses
    rp = step([P.ReqEntry(2, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT, 0, 300, 1000, 10, int(toks[1010]), 1010)],
              taps[1000:1010].tobytes())
    rep["partial_window"] = {"status": rp.statuses[0]}
    assert rp.statuses[0][0] == P.R_PARTIAL_WINDOW and rp.statuses[0][1] == 10, rp.statuses
    rbad = step([P.ReqEntry(3, P.REQ_NEED_DRAFT, 0, 253, 5, 3, int(toks[9]), 9)], taps[5:8].tobytes())
    rep["bad_entry"] = {"status": rbad.statuses[0]}
    assert rbad.statuses[0][0] == P.R_BAD_ENTRY, rbad.statuses
    rstale = step([P.ReqEntry(3, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT, 0, 103, 800 - W, W, int(toks[800]), 800)],
                  taps[800 - W:800].tobytes())
    rep["stale_new_gen"] = {"status": rstale.statuses[0]}
    assert rstale.statuses[0][0] == P.R_STALE_GENERATION, rstale.statuses
    rbig = step([P.ReqEntry(3, P.REQ_NEED_DRAFT, 0, 253, exp.max_model_len, 1, int(toks[9]), exp.max_model_len + 1)],
                taps[5:6].tobytes())
    rep["beyond_max_model_len"] = {"status": rbig.statuses[0]}
    assert rbig.statuses[0][0] == P.R_BAD_ENTRY, rbig.statuses
    h, b = cl.call(P.OP_STATS)
    rep["stats_in_process"] = json.loads(b)

    # in-process timing of decode STEPs (handle() only)
    def decode_body(nreq, a0, n_out=0, flags=0):
        ents, rows = [], []
        for s_ in range(nreq):
            ap = a0[s_]
            ents.append(P.ReqEntry(s_, P.REQ_NEED_DRAFT, N, 500 + s_, ap - 6, 6, int(toks[ap]), ap, n_out=n_out))
            rows.append(taps[ap - 6:ap])
        return P.encode_step(9, ents, np.concatenate(rows).tobytes(), rb, flags)

    def warm_slots(nreq):
        for s_ in range(nreq):
            ap = starts[s_]
            step([P.ReqEntry(s_, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT, 0, 500 + s_, ap - W, W, int(toks[ap]), ap)],
                 taps[ap - W:ap].tobytes())

    timing = {}
    for nreq, n_out, flags in ((1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (3, 4, 0), (4, 4, 0),
                               (1, 0, P.STEP_WANT_MARGIN), (4, 4, P.STEP_WANT_MARGIN)):
        warm_slots(nreq)
        a0 = list(starts)
        times = []
        for it in range(60):
            a0 = [x + 6 for x in a0]
            body = decode_body(nreq, a0, n_out, flags)
            t0 = time.perf_counter()
            h, b = cl.call(P.OP_STEP, body)
            times.append((time.perf_counter() - t0) * 1e6)
            assert h.status == P.OK
            sts = P.decode_step_reply(b).statuses
            assert all(x[0] == P.R_OK for x in sts), sts
        times = sorted(times[10:])
        timing[f"in_process_{nreq}req{'_nout4' if n_out else ''}{'_margin' if flags else ''}_us"] = {
            "p50": round(times[len(times) // 2], 1), "p99": round(times[int(len(times) * 0.99)], 1)}
    rep["timing"] = timing

    # ---- 6. over ofl (RDMA and TCP backends), x86host -> its own direct-link address
    ofl = _load_ofl()
    srv = OflServer(core, "10.202.28.1", port, rdma=True)
    rep["ofl_server"] = srv.desc
    for backend_name, backend in (("rdma", ofl.BACKEND_RDMA), ("tcp", ofl.BACKEND_TCP)):
        oc = ofl.Client(backend, "10.202.28.1", port, "10.202.28.1", service="dspark-v1", max_req=16 << 20,
                        max_resp=64 << 10, inflight=1)
        buf = bytearray(64 << 10)
        region = oc.reg(buf)
        wc = Client(ofl_client=oc, region=region, resp_buf=buf)
        wc.seq = cl.seq + 1000 * (1 + backend)
        h, b = wc.call(P.OP_HELLO, P.Hello(**{**exp.__dict__, "client_name": f"selftest-{backend_name}"}).encode())
        assert h.status == P.OK, P.FRAME_STATUS[h.status]
        cl.seq = wc.seq
        for nreq, n_out in ((1, 0), (4, 0), (4, 4)):
            for s_ in range(nreq):
                ap = starts[s_]
                body = P.encode_step(1, [P.ReqEntry(s_, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT, 0, 600 + s_, ap - W, W,
                                                    int(toks[ap]), ap)], taps[ap - W:ap].tobytes(), rb)
                h, b = wc.call(P.OP_STEP, body)
                assert h.status == P.OK
            a0 = list(starts)
            rtt, wus = [], []
            for it in range(80):
                a0 = [x + 6 for x in a0]
                ents, rows = [], []
                for s_ in range(nreq):
                    ap = a0[s_]
                    ents.append(P.ReqEntry(s_, P.REQ_NEED_DRAFT, N, 600 + s_, ap - 6, 6, int(toks[ap]), ap,
                                           n_out=n_out))
                    rows.append(taps[ap - 6:ap])
                body = P.encode_step(9, ents, np.concatenate(rows).tobytes(), rb)
                t0 = time.perf_counter()
                h, b = wc.call(P.OP_STEP, body)
                rtt.append((time.perf_counter() - t0) * 1e6)
                assert h.status == P.OK
                sr_ = P.decode_step_reply(b)
                assert all(x[0] == P.R_OK for x in sr_.statuses), sr_.statuses
                wus.append(sr_.worker_us)
            rtt, wus = sorted(rtt[10:]), sorted(wus[10:])
            rep["timing"][f"ofl_{backend_name}_{nreq}req{'_nout4' if n_out else ''}_us"] = {
                "rtt_p50": round(rtt[len(rtt) // 2], 1), "rtt_p99": round(rtt[int(len(rtt) * 0.99)], 1),
                "worker_us_p50": wus[len(wus) // 2], "request_bytes": len(body) + 32}
        h, b = wc.call(P.OP_PING)
        rep[f"ping_{backend_name}"] = P.FRAME_STATUS[h.status]
        oc.close()
    srv.stop()
    rep["memory"] = {k: v for k, v in m.memory_table().items()}
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("stats_in_process", "memory")}, indent=1,
                     default=str))


if __name__ == "__main__":
    logging_ = __import__("logging")
    logging_.basicConfig(level=logging_.INFO)
    main()
