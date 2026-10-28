// ofl: a small request/response RPC transport for the DSV41-OFFLOAD lanes (Engram row server, remote DSpark drafter).
//
// One API, two backends:
//   TCP  - TCP_NODELAY, length-prefixed frames, explicit local bind; the correctness/debug path.
//   RDMA - reliable-connected (RC) queue pairs over RoCE v2. Bootstrap over a TCP connection on the same direct link
//          (it stays open as the liveness channel). A request is one SEND from a pre-registered client send slot into
//          one of the server's pre-posted receive slots. The response is placed by the server with RDMA WRITEs straight
//          into caller-owned, caller-registered destination memory (any number of ranges, pipelined while the handler
//          is still producing it), followed by one RDMA WRITE WITH IMMEDIATE of the 64-byte response header into a
//          client header slot. RC places writes in order, so when the client sees that completion every response byte
//          is in its memory. The client busy-polls its completion queue against a deadline.
// No GPUDirect: every buffer is host memory (on a DGX Spark, the engine's pinned staging buffers).
//
// Device, port and GID come from live discovery: give the local IPv4 address of the direct link and the library finds
// the verbs device and the RoCE v2 GID index that carry it (or name them explicitly; they are then checked against the
// address). Errors are negative return codes plus a message (ofl_*_error); nothing is retried silently.
//
// Thread safety: one client is used by one thread at a time (the caller serializes). A server runs one thread per
// connection; a handler may call ofl_req_ready() from several threads at once.
#ifndef OFL_H
#define OFL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OFL_MAGIC 0x314C464Fu  // "OFL1" little-endian
#define OFL_VERSION 1
#define OFL_MAX_SEGS 16        // destination ranges per request
#define OFL_OP_ECHO 0          // built in: every server answers op 0 with the request payload (ping-pong, health)

// return codes (negative = error)
enum {
  OFL_OK = 0,
  OFL_EINVAL = -1,     // bad argument / config
  OFL_ESYS = -2,       // a system call failed (message has errno text)
  OFL_ETIMEDOUT = -3,  // deadline passed; the connection is now broken (no late data can land)
  OFL_EBROKEN = -4,    // connection is broken; call ofl_client_reconnect
  OFL_EPROTO = -5,     // malformed or unexpected message
  OFL_ETOOBIG = -6,    // request or response exceeds negotiated limits
  OFL_ENOMEM = -7,
  OFL_ERDMA = -8,      // a verbs call or a work completion failed
  OFL_EREMOTE = -9,    // the server handler returned an error status (see ofl_result.status)
  OFL_EBUSY = -10,     // too many requests in flight
  OFL_EREJECT = -11,   // the server refused the connection (service name / limits / backend)
};

enum { OFL_BACKEND_TCP = 1, OFL_BACKEND_RDMA = 2 };

// ---------------------------------------------------------------------------------------------------------------
// wire format (little-endian on both x86_64 and aarch64; checked at compile time in ofl.cpp)
typedef struct ofl_hdr {  // 64 bytes
  uint32_t magic;         // OFL_MAGIC
  uint16_t version;       // OFL_VERSION
  uint16_t kind;          // OFL_K_*
  uint16_t op;            // application op (OFL_OP_ECHO built in)
  uint16_t nseg;          // RDMA request: destination ranges after the header
  int32_t status;         // response: handler status (>= 0 ok), or an OFL_E* transport error
  uint64_t req_id;        // per-connection, starts at 1; RDMA immediate = low 32 bits
  uint64_t generation;    // caller-defined, echoed in the response
  uint32_t payload_len;   // request: payload bytes; response: response bytes
  uint32_t resp_max;      // request: destination capacity (sum of ranges)
  uint64_t rhdr_addr;     // RDMA request: client address the response header goes to
  uint32_t rhdr_rkey;
  uint32_t reserved;
  uint64_t t_server_ns;   // response: server time from request arrival to response posted
} ofl_hdr;

enum { OFL_K_HELLO = 1, OFL_K_HELLO_ACK = 2, OFL_K_REQ = 3, OFL_K_RESP = 4 };

typedef struct ofl_wire_seg {  // 16 bytes, RDMA request only
  uint64_t addr;
  uint32_t rkey;
  uint32_t len;
} ofl_wire_seg;

// ---------------------------------------------------------------------------------------------------------------
// client

typedef struct ofl_client ofl_client;

typedef struct ofl_client_config {
  int backend;                // OFL_BACKEND_TCP or OFL_BACKEND_RDMA
  const char* server_ip;      // e.g. "10.202.28.1"
  uint16_t server_port;
  const char* local_ip;       // address to bind (the direct link's local end), required
  const char* rdma_device;    // NULL/"" = the device whose GID table holds local_ip
  int rdma_port;              // 0 = 1
  int gid_index;              // -1 = the RoCE v2 GID of local_ip; else checked to be that GID
  const char* service;        // must match the server's service name (e.g. "engram-v1")
  uint32_t max_req_bytes;     // largest request payload this client will send
  uint32_t max_resp_bytes;    // largest response this client will ask for
  uint32_t max_inflight;      // requests outstanding at once (1..64); 0 = 1
  int64_t connect_timeout_ms; // 0 = 5000
} ofl_client_config;

typedef struct ofl_seg {      // a destination range inside a registered region
  uint32_t region;
  uint32_t reserved;
  uint64_t offset;
  uint64_t len;
} ofl_seg;

typedef struct ofl_result {
  int32_t status;             // the handler's status (>= 0), when the call returned OFL_OK or OFL_EREMOTE
  uint32_t resp_len;          // response bytes placed into the ranges (in order)
  uint64_t req_id;
  uint64_t generation;        // echoed
  int64_t elapsed_ns;         // submit to completion, client side
  int64_t server_ns;          // server-reported handling time
} ofl_result;

// Connect (TCP connect from local_ip; for RDMA also the QP handshake). NULL on failure, message in err.
ofl_client* ofl_client_connect(const ofl_client_config* cfg, char* err, size_t errlen);
// Register caller-owned memory as a response destination (RDMA: ibv_reg_mr; TCP: bookkeeping). The memory must stay
// valid until ofl_client_close. Region ids are stable across ofl_client_reconnect (which re-registers them).
int ofl_client_reg(ofl_client* c, void* addr, size_t len, uint32_t* region);
// Payload area of the next request (ofl_client_call/submit with this pointer as `payload` skips the copy).
void* ofl_client_payload_buf(ofl_client* c, uint32_t* capacity);
// Send one request and wait for its response. On OFL_ETIMEDOUT / OFL_ERDMA / OFL_EPROTO / OFL_ESYS the connection is
// broken (later calls return OFL_EBROKEN) and, for RDMA, its queue pair is in the error state before the call returns,
// so a late response can no longer write into the destination.
int ofl_client_call(ofl_client* c, uint16_t op, uint64_t generation, const void* payload, uint32_t payload_len,
                    const ofl_seg* segs, uint32_t nseg, ofl_result* res, int64_t timeout_ns);
// Asynchronous form: submit returns a ticket (> 0); wait completes tickets in submission order (waiting for a later
// ticket completes the earlier ones first; their results are kept until waited for).
int64_t ofl_client_submit(ofl_client* c, uint16_t op, uint64_t generation, const void* payload, uint32_t payload_len,
                          const ofl_seg* segs, uint32_t nseg);
int ofl_client_wait(ofl_client* c, int64_t ticket, ofl_result* res, int64_t timeout_ns);
int ofl_client_broken(const ofl_client* c);
int ofl_client_reconnect(ofl_client* c, char* err, size_t errlen);
const char* ofl_client_error(const ofl_client* c);  // last error message
// "tcp 10.202.28.2 -> 10.202.28.1:18515" / "rdma rocep1s0f0 port 1 gid 3 mtu 4096 qpn ..."
const char* ofl_client_describe(const ofl_client* c);
void ofl_client_close(ofl_client* c);

// ---------------------------------------------------------------------------------------------------------------
// server

typedef struct ofl_server ofl_server;
typedef struct ofl_req ofl_req;  // one request being handled (valid only inside the handler call)

// Return a status >= 0 on success (set the response length with ofl_req_set_len) or a negative status (sent to the
// client, which gets OFL_EREMOTE; no response bytes).
typedef int32_t (*ofl_handler_fn)(void* user, ofl_req* req);

typedef struct ofl_server_config {
  const char* bind_ip;        // one listener on this address (direct link), never 0.0.0.0
  uint16_t port;
  int backends;               // bit mask of 1 << OFL_BACKEND_*: which backends clients may ask for
  const char* rdma_device;    // NULL/"" = the device holding bind_ip
  int rdma_port;              // 0 = 1
  int gid_index;              // -1 = the RoCE v2 GID of bind_ip
  const char* service;        // clients must name it
  uint32_t max_req_bytes;     // per-request payload limit
  uint32_t max_resp_bytes;    // per-request response limit (each connection registers max_inflight buffers this big)
  uint32_t max_inflight;      // per connection (receive slots / response buffers); 0 = 2
  uint32_t max_conns;         // 0 = 8
  int64_t spin_us;            // RDMA connection threads busy-poll this long after a request, then block (0 = 2000000)
} ofl_server_config;

ofl_server* ofl_server_create(const ofl_server_config* cfg, ofl_handler_fn fn, void* user, char* err, size_t errlen);
int ofl_server_start(ofl_server* s);  // accept thread + connection threads in the background
void ofl_server_stop(ofl_server* s);  // closes the listener and every connection, joins threads
void ofl_server_destroy(ofl_server* s);
const char* ofl_server_describe(const ofl_server* s);
// counters: connections accepted, requests, response bytes, errors (handler < 0 + protocol)
void ofl_server_stats(const ofl_server* s, uint64_t out[4]);

uint16_t ofl_req_op(const ofl_req* r);
uint64_t ofl_req_id(const ofl_req* r);
uint64_t ofl_req_generation(const ofl_req* r);
int ofl_req_backend(const ofl_req* r);
uint64_t ofl_req_conn_id(const ofl_req* r);  // stable per connection (for per-connection state in the handler)
const void* ofl_req_payload(const ofl_req* r, uint32_t* len);
// Response buffer (registered for RDMA). Capacity = min(the client's destination capacity, max_resp_bytes).
void* ofl_req_resp_buf(ofl_req* r, uint32_t* capacity);
// Optional pipelining: bytes [off, off + len) of the response are final and may go out now. Any thread; any order.
int ofl_req_ready(ofl_req* r, uint32_t off, uint32_t len);
void ofl_req_set_len(ofl_req* r, uint32_t len);  // response length (bytes not marked ready go out after the handler)

#ifdef __cplusplus
}
#endif
#endif
