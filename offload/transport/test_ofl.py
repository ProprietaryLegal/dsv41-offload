#!/usr/bin/env python3
"""ofl transport tests through the ctypes binding (both backends, one host, loopback over one direct-link port).

    OFL_LIB=~/dsv41-offload/o1/bin/libofl.so python3 test_ofl.py --ip 10.202.28.1 --port 18601

Checks: echo; a Python handler op; a handler error status (EREMOTE, connection stays usable); service and size
rejection; and the timeout rule: a response that arrives after the client gave up must not land in the destination,
the connection is broken until reconnect(), and the next call after reconnect() is correct.
"""

import argparse
import sys
import time

import ofl

ap = argparse.ArgumentParser()
ap.add_argument("--ip", required=True)
ap.add_argument("--port", type=int, default=18601)
args = ap.parse_args()

SERVICE = "ofl-test"


def handler(req: ofl.Request) -> int:
    if req.op == 7:  # reverse
        req.respond(req.payload[::-1])
        return 0
    if req.op == 8:  # slow: answer after 0.5 s
        time.sleep(0.5)
        req.respond(b"L" * len(req.payload))
        return 0
    if req.op == 9:
        return -42
    return -1


srv = ofl.Server(args.ip, args.port, service=SERVICE, handler=handler, max_req=4096, max_resp=1 << 16)
srv.start()
print("server:", srv.describe())
fails = 0


def check(name, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + name)
    fails += 0 if cond else 1


for be, bname in ((ofl.BACKEND_TCP, "tcp"), (ofl.BACKEND_RDMA, "rdma")):
    c = ofl.Client(be, args.ip, args.port, args.ip, service=SERVICE, max_req=4096, max_resp=1 << 16)
    print(bname, "client:", c.describe())
    buf = bytearray(1 << 16)
    r = c.reg(buf)
    res = c.call(ofl.OP_ECHO, b"hello world", [(r, 0, 11)])
    check(f"{bname} echo", res.resp_len == 11 and bytes(buf[:11]) == b"hello world")
    # two ranges: 5 bytes at 100, rest at 1000
    res = c.call(7, b"abcdefghij", [(r, 100, 5), (r, 1000, 5)])
    check(f"{bname} reverse into two ranges", bytes(buf[100:105]) == b"jihgf" and bytes(buf[1000:1005]) == b"edcba")
    try:
        c.call(9, b"x", [(r, 0, 1)])
        check(f"{bname} handler error -> EREMOTE", False)
    except ofl.OflError as e:
        check(f"{bname} handler error -> EREMOTE", e.code == -9)
    res = c.call(ofl.OP_ECHO, b"still ok", [(r, 0, 8)])
    check(f"{bname} usable after EREMOTE", bytes(buf[:8]) == b"still ok")
    try:
        c.call(ofl.OP_ECHO, b"z" * 5000, [(r, 0, 16)])
        check(f"{bname} oversize payload refused", False)
    except ofl.OflError as e:
        check(f"{bname} oversize payload refused", e.code == -6)
    pb_before = c.payload_buf()[0]
    # timeout: the late answer ("LLLL...") must never reach buf
    buf[2000:2064] = b"S" * 64
    t0 = time.time()
    try:
        c.call(8, b"q" * 64, [(r, 2000, 64)], timeout_s=0.1)
        check(f"{bname} timeout raised", False)
    except ofl.OflError as e:
        check(f"{bname} timeout raised", e.code == -3 and time.time() - t0 < 0.3)
    check(f"{bname} broken after timeout", c.broken())
    time.sleep(0.8)  # the server's late response has been sent by now
    check(f"{bname} late response did not land", bytes(buf[2000:2064]) == b"S" * 64)
    try:
        c.call(ofl.OP_ECHO, b"x", [(r, 0, 1)])
        check(f"{bname} call on broken connection refused", False)
    except ofl.OflError as e:
        check(f"{bname} call on broken connection refused", e.code == -4)
    c.reconnect()
    check(f"{bname} payload_buf address stable across reconnect", pb_before != 0 and c.payload_buf()[0] == pb_before)
    res = c.call(7, b"123", [(r, 3000, 3)])
    check(f"{bname} correct after reconnect", bytes(buf[3000:3003]) == b"321")
    c.close()
    try:
        ofl.Client(be, args.ip, args.port, args.ip, service="wrong-service", max_req=16, max_resp=16)
        check(f"{bname} service mismatch refused", False)
    except ofl.OflError as e:
        check(f"{bname} service mismatch refused", "service" in str(e))
    try:
        ofl.Client(be, args.ip, args.port, args.ip, service=SERVICE, max_req=1 << 20, max_resp=16)
        check(f"{bname} over-limit client refused", False)
    except ofl.OflError as e:
        check(f"{bname} over-limit client refused", "limits" in str(e))

time.sleep(0.2)
print("server stats:", srv.stats())
srv.stop()
print("RESULT:", "ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
