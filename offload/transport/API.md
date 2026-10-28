Status 2026-10-23: TCP diagnostic and RDMA production transport; checkpoint identity, bounded host buffers and exact replies remain the contract. See [live roadmap](../../documents/plans/roadmap.md) and [measured record](../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# ofl — the DSV41-OFFLOAD request/response transport (API v1)

Owner: lane O1 (branch `offload/o1-engram`). Status 2026-10-22: **TCP and RDMA both pass ping-pong with every byte
checked, on both direct links** (Spark A ↔ x86host 10.202.28.x, Spark B ↔ x86host 10.202.29.x); the Python test suite
(`test_ofl.py`, 24 checks incl. timeout/late-data/reconnect) passes on both backends. Lanes O2/O3 can build on it now.

Files: `ofl.h` (the API, the reference), `ofl.cpp` (both backends, client + server, ~1.1k lines, no deps beyond
libibverbs + pthreads), `ofl.py` (ctypes binding), `ofl_pingpong.cpp` (self-test + microbench), `ofl_probe.c`
(read-only RDMA discovery), `test_ofl.py`, `Makefile`.

## What it is

One client ⇄ one server connection carrying requests `(op, generation, payload)` and responses `(status, bytes)`,
with the response **placed directly into caller-owned destination ranges** (up to 16 ranges of registered memory).

| | TCP backend | RDMA backend |
|---|---|---|
| use | correctness / debugging / fallback | the fast path |
| request | header + payload on the socket | one SEND (header + range table + payload) from a registered client slot into a pre-posted server receive slot |
| response | header + bytes; client `recv`s straight into the ranges | server RDMA-WRITEs the bytes into the ranges (pipelined while the handler is still producing them), then one RDMA WRITE WITH IMMEDIATE of the 64-byte response header into a client header slot; RC ordering ⇒ when the client sees that completion every byte is in place |
| wait | `ppoll` against the deadline | client busy-polls its CQ against the deadline |
| liveness | the socket | the TCP bootstrap socket stays open; server sees EOF ⇒ tears the QP down |

No GPUDirect anywhere: all memory is host memory (on a Spark, the engine's pinned staging / registered malloc pages).
No relaxed ordering on any MR (the header must land after the data).

## Selecting device / port / GID (live discovery)

Pass the **local IPv4 of the direct link** (`local_ip` / `bind_ip`). The library scans the verbs devices and picks the
device + port whose GID table has a **RoCE v2, IPv4-mapped GID equal to that address**; that index is the GID used.
If you also name `rdma_device` / `gid_index`, they are checked against the address (mismatch = error, not a silent
override). `ofl_probe` prints the tables. Live values 2026-10-22 (ofl_probe on all four ends):

| end | netdev | verbs device | IPv4 | GID (RoCE v2) | active MTU |
|---|---|---|---|---|---|
| x86host → Spark A | enp130s0f1np1 | rocep130s0f1 | 10.202.28.1 | 3 | 4096 |
| x86host → Spark B | enp130s0f0np0 | rocep130s0f0 | 10.202.29.1 | 3 | 4096 |
| Spark A → x86host | enp1s0f0np0 | rocep1s0f0 | 10.202.28.2 | 3 | 4096 |
| Spark B → x86host | enp1s0f0np0 | rocep1s0f0 | 10.202.29.2 | 3 | 4096 |

(Spark A also has rocep1s0f1 10.202.26.2 / roceP2p1s0f1 10.202.27.2 = the TP2 fabric; `local_ip` keeps you off it.)

## C API (see `ofl.h` for the exact declarations)

```c
// client
ofl_client_config cfg = {.backend = OFL_BACKEND_RDMA, .server_ip = "10.202.28.1", .server_port = 18620,
                         .local_ip = "10.202.28.2", .gid_index = -1, .service = "dspark-v1",
                         .max_req_bytes = 1 << 20, .max_resp_bytes = 1 << 16, .max_inflight = 2};
char err[512];
ofl_client* c = ofl_client_connect(&cfg, err, sizeof err);           // NULL + err on failure
uint32_t region; ofl_client_reg(c, buf, buf_len, &region);           // once; buf stays valid until close
ofl_seg seg = {.region = region, .offset = 0, .len = 4096};
ofl_result res;
int rc = ofl_client_call(c, MY_OP, generation, payload, payload_len, &seg, 1, &res, 5000000 /* ns */);
// rc == OFL_OK: res.resp_len bytes are in buf[0..]; res.status >= 0 is your handler's status
// async: t = ofl_client_submit(...); ... ofl_client_wait(c, t, &res, timeout_ns);  (up to max_inflight outstanding)
// zero-copy request: p = ofl_client_payload_buf(c, &cap); fill p; ofl_client_call(c, op, gen, p, n, ...)
```

```c
// server
int32_t handler(void* user, ofl_req* r) {
  uint32_t n; const void* p = ofl_req_payload(r, &n);
  uint32_t cap; uint8_t* out = ofl_req_resp_buf(r, &cap);         // registered; cap = min(client ranges, max_resp)
  ... produce bytes; optionally ofl_req_ready(r, off, len) for finished pieces (any thread) ...
  ofl_req_set_len(r, produced);
  return 0;                                                        // < 0: client gets OFL_EREMOTE, no bytes
}
ofl_server_config sc = {.bind_ip = "10.202.28.1", .port = 18620,
                        .backends = (1 << OFL_BACKEND_TCP) | (1 << OFL_BACKEND_RDMA), .gid_index = -1,
                        .service = "dspark-v1", .max_req_bytes = 1 << 20, .max_resp_bytes = 1 << 16,
                        .max_inflight = 2, .spin_us = 2000000};
ofl_server* s = ofl_server_create(&sc, handler, user, err, sizeof err);
ofl_server_start(s); ... ofl_server_destroy(s);
```

Python (`ofl.py`, tests/tools): `Client(backend, server_ip, port, local_ip, service=, max_req=, max_resp=)`,
`.reg(buffer)`, `.call(op, payload, [(region, offset, len)], timeout_s)`, `.reconnect()`; `Server(bind_ip, port,
service=, handler=fn(Request) -> status)`, `.start()`, `.stats()`, `.stop()`; errors raise `OflError(code, msg)`.

## Semantics you can rely on

- **Op 0 is a built-in echo** on every server (health check / ping-pong). Your ops start at 1.
- **Ordering:** one connection = one server thread; requests are handled and answered in submission order.
  `ofl_client_wait(t)` completes earlier tickets first (their results are kept until waited for).
- **Timeouts are fatal for the connection, by design.** On `OFL_ETIMEDOUT` (and `ERDMA`/`EPROTO`/`ESYS`) the client
  moves its QP to the error state *before returning* and closes the socket, so a late response can never write into
  your buffers (tested: `test_ofl.py` "late response did not land", both backends). Later calls return
  `OFL_EBROKEN` until `ofl_client_reconnect()` (new QP, regions re-registered with new rkeys, same region ids).
- **Handler errors are not fatal:** status < 0 ⇒ `OFL_EREMOTE`, `res.status` = your code, connection stays up.
- **Limits are negotiated at connect** (`max_req_bytes`, `max_resp_bytes`, `max_inflight`); a client asking more than
  the server allows, or naming another `service`, is refused at connect (`OFL_EREJECT`, reason in the message).
- **Pipelining:** `ofl_req_ready(r, off, len)` posts those response bytes immediately (thread-safe); whatever is not
  marked ready goes out after the handler returns, then the header. Bytes beyond the final length may already have
  been written into the client's ranges (inside its declared capacity): only trust `res.resp_len`.
- **Memory:** server per connection: `max_inflight × (max_resp_bytes + 64)` response buffers +
  `max_inflight × (320 + max_req_bytes)` receive slots, all registered. Client: `max_inflight × (320 + max_req_bytes)`
  send slots + `max_inflight × 64` header slots + your regions. Nothing is pinned beyond that.
- **CPU:** client waits busy-poll (one core while waiting). Server connection threads busy-poll for `spin_us` after the
  last request, then sleep on the completion channel (+ the bootstrap socket) and wake on the next request.
- **Binding:** servers refuse `0.0.0.0`; bind the direct-link address only (COMMON.md rule 9). Clients bind
  `local_ip` before connecting. Ports in use: 18600–18601 ofl self-tests, **18610 Engram** (lane O1). Pick 18620+
  for the drafter. (perftest's default 18515 is S1's.)
- **Zero-copy requests:** `ofl_client_payload_buf()` is the payload area of the next request's send slot. The
  slots are allocated once per client (page-aligned) and survive `ofl_client_reconnect()` (only their MRs are
  re-registered), so with `max_inflight = 1` the address is **stable for the client's lifetime** — you may
  `cudaHostRegister` it and D2H straight into it (tested: `test_ofl.py` "payload_buf address stable across
  reconnect"). With `max_inflight = k` it rotates over k fixed slots (`(next ticket - 1) % k`). It returns NULL while
  the connection is broken. Pass that pointer as `payload` and the copy is skipped.
- Thread safety: one client per thread at a time. Servers: handlers run concurrently across connections.

## Wire format v1 (little-endian; `ofl.h`)

`ofl_hdr` (64 B): magic `OFL1`, version 1, kind (HELLO/HELLO_ACK/REQ/RESP), op, nseg, status, req_id (per connection
from 1; RDMA immediate = low 32 bits), generation (echoed), payload_len, resp_max, rhdr_addr/rkey (RDMA: where the
response header goes), t_server_ns. RDMA request = header + nseg × `ofl_wire_seg` {addr, rkey, len} + payload.
Bootstrap: client HELLO {backend, limits, qpn, psn, gid, mtu, service, host} → server creates its QP, posts receives,
goes RTR/RTS, answers HELLO_ACK {status, granted limits, qpn, psn, gid, mtu}; client goes RTR/RTS. Path MTU =
min(both active MTUs). RC: timeout 14, retry 7, RNR retry 7.

## Build

```
x86host (x86_64; verbs headers extracted from libibverbs-dev_50.0-2ubuntu0.2_amd64.deb, not installed):
  make VERBS_INC=$HOME/dsv41-offload/o1/sysroot/usr/include OUT=$HOME/dsv41-offload/o1/bin
Spark host or the serve container (aarch64; libibverbs-dev 50.0 present):
  make OUT=/tmp/o1/bin /tmp/o1/bin/libofl.so /tmp/o1/bin/ofl_pingpong     (name the $(OUT)/ targets explicitly)
Inside a torch JIT extension: add ofl.cpp to `sources`, the dir to `extra_include_paths`, `-libverbs` to
`extra_ldflags` (as offload/engram/spark/ does).
```

## Measured (2026-10-22, idle serve, `offload/results/o1/pingpong_spark{A,B}_20261022.jsonl`)

Client on the Spark host, server on x86host, `ofl_pingpong`, every response byte checked (bad = 0 everywhere).
"rows R" = Engram shape: request R × 8 B, response R × 264 B in two ranges.

| shape | RDMA p50 A / B (µs) | RDMA p99 A / B | TCP p50 A / B (µs) |
|---|---|---|---|
| echo 64 B | 5.6 / 6.6 | 17.1 / 9.1 | 34.7 / 30.5 |
| rows 288 (75 KB) | 32.9 / 32.3 | 42.2 / 37.8 | 113 / 148 |
| rows 4096 (1.08 MB) | 371 / 358 | 393 / 375 | 1301 / 1018 |
| rows 16384 (4.3 MB) | 1086 / 1041 | 1109 / 1058 | 5269 / 3227 |
| rows 49152 (13.0 MB) | 3027 / 2866 | 3082 / 2893 | 11395 / 13615 |

The 13 MB RDMA case moves ~4.3–4.5 GB/s including the server's pattern fill (1.5 ms, overlapped by `ofl_req_ready`
pipelining) — ~86–90 % of the 40 Gb/s line rate. Loopback on x86host (same port): echo 6.1 µs.
