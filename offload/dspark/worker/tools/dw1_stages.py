"""DW1 item 1: per-stage timing of one worker STEP on one RTX 3090 (the one-card GpuBackend), plus a TCP loopback round
trip that uses no NIC.

Stages per decode STEP (live shapes: 1-2 requests at n_out 5, 3-4 requests at n_out 4; 6 rows per request):
  handle      DraftCore.handle wall time (what the Spark waits for, minus the wire)
  flush       GpuBackend.flush wall (row staging copy, H2D, insert graph launch)
  draft       GpuBackend.draft wall (input H2D, draft graph launch, D2H + sync, dtype conversion)
  insert_gpu  insert graph GPU time (CUDA events around the replay)
  draft_gpu   draft graph GPU time (CUDA events around the replay)
  parse_reply handle - flush - draft (frame decode, slot bookkeeping, reply encode)
  host_extra  handle - insert_gpu - draft_gpu (everything that is not one of the two graphs)
TCP loopback (127.0.0.1, DSPARK_WORKER_ALLOW_LOOPBACK=1): u32 length + frame both ways through TcpStubServer.

Instruments by wrapping methods on the live objects; the worker code itself is unchanged, so the same file runs on the
fb68c52 tree (baseline) and on a candidate tree. Usage (from offload/dspark, one GPU visible):
  python -m worker.tools.dw1_stages ROOT OUT.json [iterations]
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
import time

import numpy as np
import torch

from .. import protocol as P
from ..server import DraftCore, GpuBackend, TcpStubServer, expected_hello
from .server_selftest import SESSION, Client, Shadow1, taps_for

SHAPES = ((1, 0), (2, 0), (3, 4), (4, 4))        # (requests, n_out); 0 = all N (the live shapes, CONTRACT 1.3)
ROWS = 6


def pct(xs: list[float]) -> dict:
    s = sorted(xs)
    n = len(s)
    return {"p10": round(s[n // 10], 1), "p50": round(s[n // 2], 1), "p90": round(s[(n * 9) // 10], 1), "n": n}


class Timers:
    """Wall and CUDA-event timers wrapped around the backend and model methods of one live core."""

    def __init__(self, core, m) -> None:
        self.wall: dict[str, float] = {}
        self.ev: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        be = core.backend
        for name in ("flush", "draft"):
            setattr(be, name, self._wall(name, getattr(be, name)))
        for name, tag in (("insert_taps_graphed", "insert_gpu"), ("draft_graphed", "draft_gpu"),
                          ("insert", "insert_gpu"), ("main_x", "insert_gpu")):
            if hasattr(m, name):
                setattr(m, name, self._events(tag, getattr(m, name)))

    def _wall(self, name, fn):
        def run(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.wall[name] = self.wall.get(name, 0.0) + (time.perf_counter() - t0) * 1e6
        return run

    def _events(self, tag, fn):
        def run(*a, **k):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*a, **k)
            e.record()
            self.ev.append((tag, s, e))
            return out
        return run

    def take(self) -> dict[str, float]:
        torch.cuda.synchronize()
        out = dict(self.wall)
        for tag, s, e in self.ev:
            out[tag] = out.get(tag, 0.0) + s.elapsed_time(e) * 1e3
        self.wall, self.ev = {}, []
        return out


def build_one_card(root: str):
    """server.build_core(tp=1) without its GPU-c4e7a371 pin: DW1 runs on the granted card (GPU-bc62df2a), never on the
    production pair. Same model, capture set and backend as build_core; CUDA_VISIBLE_DEVICES must name one GPU."""
    from ..checkpoint import Checkpoint
    from ..identity import compute as compute_identity
    from ..model import DSparkDrafter

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not vis.startswith("GPU-bc62df2a"):
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={vis!r}: DW1 runs only on GPU-bc62df2a (coordinator grant)")
    ck = Checkpoint(root)
    ident = compute_identity(root, ck.manifest)
    dev = torch.device("cuda", 0)
    model = DSparkDrafter(ck, dev, slots=8, max_model_len=327680)
    exp = expected_hello(bytes.fromhex(ident["drafter_identity"]), bytes.fromhex(ident["markov_identity"]),
                         max_model_len=327680)
    model.capture(batch_sizes=tuple(range(1, exp.max_slots + 1)), ks=(exp.num_spec,),
                  insert_rows=(8, 16, 32, 64, 128), n_outs=tuple(range(1, exp.num_spec + 1)))
    backend = GpuBackend(model, exp.max_rows_per_step, exp.row_bytes, graphs=True)
    return DraftCore(exp, backend), model


def main() -> None:
    root = sys.argv[1]
    out_path = sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    os.environ["DSPARK_WORKER_ALLOW_LOOPBACK"] = "1"
    rep: dict = {"device": torch.cuda.get_device_name(0), "iterations": iters, "rows_per_request": ROWS,
                 "tree": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))}
    core, m = build_one_card(root)
    sh = Shadow1(m)
    exp = core.expected
    rb, N, W = exp.row_bytes, exp.num_spec, exp.window
    cl = Client(core=core)
    h, _b = cl.call(P.OP_HELLO, P.Hello(**{**exp.__dict__, "client_name": "dw1"}).encode())
    assert h.status == P.OK, P.FRAME_STATUS[h.status]
    gen = torch.Generator(device=m.dev).manual_seed(11)
    L = 128 + 6 * (iters + 40) + 8
    toks = torch.randint(0, 120000, (L,), generator=gen, device=m.dev)
    taps = taps_for(sh, toks, gen)
    starts = [W + 1] * 4

    def warm(nreq: int, call=None) -> None:
        for s_ in range(nreq):
            ap = starts[s_]
            body = P.encode_step(1, [P.ReqEntry(s_, P.REQ_NEW_GEN | P.REQ_NEED_DRAFT, 0, 700 + s_, ap - W, W,
                                                int(toks[ap]), ap)], taps[ap - W:ap].tobytes(), rb, 0)
            if call is None:
                h, b = cl.call(P.OP_STEP, body)
                assert h.status == P.OK, P.FRAME_STATUS[h.status]
            else:
                assert P.AppHeader.decode(call(body)).status == P.OK

    def body_for(nreq: int, a0: list[int], n_out: int) -> bytes:
        ents, rows = [], []
        for s_ in range(nreq):
            ap = a0[s_]
            ents.append(P.ReqEntry(s_, P.REQ_NEED_DRAFT, N, 700 + s_, ap - ROWS, ROWS, int(toks[ap]), ap,
                                   n_out=n_out))
            rows.append(taps[ap - ROWS:ap])
        return P.encode_step(9, ents, np.concatenate(rows).tobytes(), rb, 0)

    tm = Timers(core, m)
    stages: dict = {}
    for nreq, n_out in SHAPES:
        warm(nreq)
        tm.take()
        a0 = list(starts)
        rec: dict[str, list[float]] = {}
        for it in range(iters + 20):
            a0 = [x + ROWS for x in a0]
            body = body_for(nreq, a0, n_out)
            t0 = time.perf_counter()
            h, b = cl.call(P.OP_STEP, body)
            handle = (time.perf_counter() - t0) * 1e6
            assert h.status == P.OK
            sts = P.decode_step_reply(b).statuses
            assert all(x[0] == P.R_OK for x in sts), sts
            t = tm.take()
            if it < 20:
                continue
            t["handle"] = handle
            t["parse_reply"] = handle - t.get("flush", 0.0) - t.get("draft", 0.0)
            t["host_extra"] = handle - t.get("insert_gpu", 0.0) - t.get("draft_gpu", 0.0)
            for k_, v in t.items():
                rec.setdefault(k_, []).append(v)
        stages[f"{nreq}req_nout{n_out or N}"] = {k_: pct(v) for k_, v in rec.items()}
    rep["stages_us"] = stages

    # TCP loopback round trip (no NIC): the same frames through TcpStubServer on 127.0.0.1
    srv = TcpStubServer(core, "127.0.0.1", 0)
    sock = socket.create_connection(("127.0.0.1", srv.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def rpc(frame: bytes) -> bytes:
        sock.sendall(struct.pack("<I", len(frame)) + frame)
        hdr = b""
        while len(hdr) < 4:
            hdr += sock.recv(4 - len(hdr))
        (n,) = struct.unpack("<I", hdr)
        buf = bytearray()
        while len(buf) < n:
            buf += sock.recv(n - len(buf))
        return bytes(buf)

    seq = [10_000_000]

    def frame(op: int, body: bytes) -> bytes:
        seq[0] += 1
        return P.AppHeader(P.MAGIC, P.VERSION, op, seq[0], SESSION, len(body), 0, 0).encode() + body

    rtt: dict = {}
    for nreq, n_out in SHAPES:
        warm(nreq, lambda body: rpc(frame(P.OP_STEP, body)))
        a0 = list(starts)
        xs = []
        for it in range(iters + 20):
            a0 = [x + ROWS for x in a0]
            f = frame(P.OP_STEP, body_for(nreq, a0, n_out))
            t0 = time.perf_counter()
            r = rpc(f)
            dt = (time.perf_counter() - t0) * 1e6
            assert P.AppHeader.decode(r).status == P.OK
            tm.take()
            if it >= 20:
                xs.append(dt)
        rtt[f"{nreq}req_nout{n_out or N}"] = pct(xs)
    sock.close()
    srv.stop()
    rep["tcp_loopback_rtt_us"] = rtt
    with open(out_path, "w") as fh:
        json.dump(rep, fh, indent=1)
    for k_, v in stages.items():
        print(k_, {s: v[s]["p50"] for s in ("handle", "flush", "draft", "insert_gpu", "draft_gpu", "parse_reply",
                                            "host_extra") if s in v}, "tcp", rtt[k_]["p50"], flush=True)


if __name__ == "__main__":
    main()
