#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-flight probe of the remote DSpark worker from a Spark (or Controller against the mock).

Default: PING round trips only (PING needs no session and changes no worker state, so it
is safe while an engine is connected). ``--hello`` also checks HELLO (identity / config)
under a NEW session - the worker then drops any engine session's windows (the engine
recovers by itself: NO_SESSION -> HELLO -> resync), so use it before the engine starts.
``--step-rows R`` (needs --hello) sends one synthetic STEP of R zero rows for slot 0 and
RESETs it afterwards, timing the STEP round trip.

Refuses to send anything unless the serve's /metrics shows 0 running and 0 waiting
requests (COMMON.md rule 2), unless --no-metrics-check.

  python3 probe.py --endpoint o1rdma://10.202.28.1:18620?bind=10.202.28.2 --identity <hex>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from remote_dspark import wire  # noqa: E402
from remote_dspark.transport import make_client, parse_endpoint  # noqa: E402

METRICS = "http://HEAD_IP:8000/metrics"


def serve_idle(url: str) -> tuple[bool, str]:
    try:
        txt = urllib.request.urlopen(url, timeout=3).read().decode()
    except OSError as e:
        return False, f"metrics unreadable: {e}"
    vals = {}
    for key in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        m = re.findall(rf"^{re.escape(key)}(?:{{[^}}]*}})? ([0-9.eE+-]+)$", txt, re.M)
        vals[key] = sum(float(x) for x in m) if m else None
    ok = all(v == 0.0 for v in vals.values())
    return ok, json.dumps(vals)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--identity", default="", help="64-hex drafter identity (for --hello)")
    ap.add_argument("--markov-identity", default="0" * 64)
    ap.add_argument("--pings", type=int, default=200)
    ap.add_argument("--hello", action="store_true")
    ap.add_argument("--step-rows", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--timeout-ms", type=int, default=1000)
    ap.add_argument("--metrics", default=METRICS)
    ap.add_argument("--no-metrics-check", action="store_true")
    args = ap.parse_args()

    if not args.no_metrics_check:
        ok, detail = serve_idle(args.metrics)
        if not ok:
            raise SystemExit(f"refusing: serve not idle or unreadable ({detail})")
    ep = parse_endpoint(args.endpoint)
    cli = make_client(ep)
    t = args.timeout_ms / 1000.0
    cli.connect(t)
    resp = bytearray(1 << 16)
    session = secrets.randbits(32) or 1
    seq = 0
    rtts = []
    for _ in range(args.pings):
        seq += 1
        msg = bytearray(wire.frame(wire.OP_PING, seq, session))
        t0 = time.perf_counter()
        n = cli.request(msg, resp, t)
        rtts.append((time.perf_counter() - t0) * 1e6)
        h, _ = wire.split_frame(memoryview(resp)[:n])
        if h.status != wire.ST_OK or h.seq != seq:
            raise SystemExit(f"PING reply status {h.status} seq {h.seq}")
    out = {"endpoint": str(ep), "ping_us": {"p50": round(pct(rtts, .5), 1),
                                             "p99": round(pct(rtts, .99), 1),
                                             "min": round(min(rtts), 1)}}
    if args.hello:
        hello = wire.Hello(hidden=args.hidden, row_bytes=3 * args.hidden * 2,
                           drafter_identity=wire.parse_hex32(args.identity, "--identity"),
                           markov_identity=wire.parse_hex32(args.markov_identity,
                                                            "--markov-identity"),
                           client_name="dsv41 probe")
        seq += 1
        n = cli.request(bytearray(wire.frame(wire.OP_HELLO, seq, session, hello.pack())),
                        resp, t)
        h, body = wire.split_frame(memoryview(resp)[:n])
        out["hello_status"] = wire.FRAME_STATUS_NAMES.get(h.status, h.status)
        if len(body) == wire.HELLO_REP.size:
            rep = wire.HelloReply.unpack(body)
            out["worker"] = {"name": rep.worker_name, "identity": rep.drafter_identity.hex(),
                             "markov": rep.markov_identity.hex(), "max_slots": rep.max_slots,
                             "max_rows_per_step": rep.max_rows_per_step,
                             "capabilities": rep.capabilities}
        if h.status == wire.ST_OK and args.step_rows:
            R = args.step_rows
            rb = 3 * args.hidden * 2
            e = [wire.ReqEntry(0, 1, 0, R, 0, R, flags=wire.RF_NEW_GEN | wire.RF_NEED_DRAFT)]
            body = bytearray(wire.step_body_len(1, R, rb))      # zero rows, no numpy needed
            wire.pack_step_prefix(body, 0, 1, e, R, rb)
            seq += 1
            msg = bytearray(wire.frame(wire.OP_STEP, seq, session, body))
            t0 = time.perf_counter()
            n = cli.request(msg, resp, t)
            out["step_us"] = round((time.perf_counter() - t0) * 1e6, 1)
            h, body = wire.split_frame(memoryview(resp)[:n])
            out["step_status"] = wire.FRAME_STATUS_NAMES.get(h.status, h.status)
            if h.status == wire.ST_OK and len(body) >= wire.REPLY_HEADER.size + 8:
                _sid, _n, _N, _f, worker_us, _r2s = wire.REPLY_HEADER.unpack_from(body, 0)
                st, wrows, _g = wire.REQ_STATUS.unpack_from(body, wire.REPLY_HEADER.size)
                out["step_worker_us"] = worker_us
                out["step_req_status"] = wire.REQ_STATUS_NAMES.get(st, st)
                out["step_window_rows"] = wrows
            seq += 1
            cli.request(bytearray(wire.frame(wire.OP_RESET, seq, session,
                                             wire.RESET_BODY.pack(1))), resp, t)
        if h.status == wire.ST_OK or args.step_rows:
            seq += 1
            n = cli.request(bytearray(wire.frame(wire.OP_STATS, seq, session)), resp, t)
            h, body = wire.split_frame(memoryview(resp)[:n])
            if h.status == wire.ST_OK:
                out["worker_stats"] = json.loads(bytes(body) or b"{}")
    cli.close()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
