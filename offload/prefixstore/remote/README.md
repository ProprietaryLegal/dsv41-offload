# Stage B: x86host RAM prefix tier — CPU prototype

The synthetic x86host server and fake clients work over existing `ofl` TCP and optional RDMA. This is a bounded,
authenticated chunk store with exact-byte returns and compute fallback. It is **not connected to vLLM**. No GPU,
Spark request, production restart, model data or service change is required or was used.

## Placement and connector boundary

First deploy/test Stage A. A later adapter can implement the file-tier load/store boundary in
`vllm/v1/kv_offload/tiering/fs/io.py` or a registered tier behind the same CPU-chunk interface. It must retain
Stage A's scheduler load cap, backpressure, completion accounting and rank-0-to-rank-1 copy. A remote tier does not
make a 256 MiB Spark CPU landing buffer able to load an unlimited prefix: the coordinator's 4-request conservative
Stage A cap is 29,568 tokens at that buffer size. Loading the full 150K would require a separately validated larger
landing budget or streaming adapter; neither was implemented here.

Start with **one rank-0 listener** and one RAM store. The 0077 layout certifies identical one-head latent bytes,
stores one copy on rank 0, and synchronizes rank 1 before GPU loads. Do not open an independent unfilled rank-1
slot. For truly sharded groups use separate rank keys and obtain both ranks' bytes; never reuse rank-0 bytes merely
because this model currently has one latent head. A future dual-listener service should share one `PrefixStore`
instance and one global capacity budget, rather than instantiate two full caches. Tests instantiate one listener
at a time, on each allowed address. They use x86host-local source/destination addresses throughout.

## Bytes and bandwidth from the actual allocator

Sources: K1 `offload/kvpack/kv_arith.py`, packed `attention.py`, image v5
`vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py:build_offloading_config`,
`vllm/v1/kv_offload/cpu/spec.py`, `shared_offload_region.py`, and upstream 0077/0081.

- K1 packed FP8 group stride: **138,240 bytes/block/rank**. The offloading translator takes the whole backing
  allocation stride (`kv_cache_tensors[0].size // num_blocks`), not a nominal compressed-record size.
- Prefix-cacheable groups cover **128 and 192 prompt tokens/block**. Bounded sliding-window replay and compressor
  ring state are not persisted as full-prefix blocks. Asymptotic raw bytes/token/rank:
  `138240 * (1/128 + 1/192) = 1800`.
- Shared CPU chunks align to 4,096 bytes: `ceil(138240/4096)*4096 = 139264`. An unchanged file-tier-compatible
  backend therefore budgets **1,813.333 aligned bytes/token/rank**. Padding may be omitted on the wire only by an
  adapter whose versioned layout digest explicitly specifies how to restore it; this prototype makes no such choice.
- 150,000 tokens: `ceil(150000/128)+ceil(150000/192) = 1172+782 = 1954` chunks. Raw **270,120,960 bytes**
  (257.607 MiB); aligned **272,121,856 bytes** (259.516 MiB). The last partial prefix blocks may remain compute;
  these ceiling figures are conservative payload sizing, not a promise that all tokens are cache hits.
- One 40 Gb/s link has a theoretical 5,000,000,000 bytes/s: **54.0 ms raw / 54.4 ms aligned** for 150K.
  Existing O1 large-message RDMA results of 4.3–4.5 GB/s imply **60–63 ms** for aligned bytes. This excludes
  request setup, gather/hash, staging, contention, GPU copy, replay and rank-1 synchronization. The recorded 0077
  ~1.7 GB/s gloo copy adds about **160 ms** if it copies all aligned chunks sequentially. These are projections.
- One-copy replicated layout means this data is stored once on x86host. Sharded/uncertified ranks would require
  separate payloads and double capacity; two links are not assumed to double useful throughput.

The 6 GiB GPU pool's reported ~2,574 B/token is an admission/pool figure, including window/ring reservations. It is
not the remote wire-byte rate. The Python prototype's local numbers below are also not physical-link bandwidth.

## Identity and poisoning

`Namespace` carries SHA256 digests for the exact checkpoint, KV layout and privacy/cache scope, plus rank and world
size. The server accepts an explicit frozen allowlist. Layout identity must cover engine/overlay revisions, cache
format/scales, block sizes/packing, selected layers/groups/order, canonical/direct mapping, alignment and replicated
owner policy. Model identity must cover legal-v2a checkpoint manifests and serving transformations, not a display
alias. Scope must preserve the engine's per-matter/salt isolation. Only synthetic identities are used in these tests.

Each immutable `Key` adds group, full token-prefix-chain SHA256, exact byte length and payload SHA256. A successful
GET validates namespace, CRC32, payload SHA256 and length before returning data. Corrupt resident chunks are deleted
and refused. PUT of different bytes under the same full key is refused; an identical PUT is idempotent. Authentication
uses HMAC-SHA256 over every bounded request and response. Responses also bind the full key, operation and a random
128-bit request nonce, preventing an old response or another requested chunk from being accepted. The wire CRC is
checked independently of authentication; the cache stores its own CRC and rechecks on GET.

The client must obtain payload hashes from a **trusted catalogue** mapping token-chain/group/namespace to immutable
content keys. The fake clients compute them from their synthetic bytes. Implementing that catalogue and connector
publication is a remaining integration task; model/token labels alone cannot prove correct KV. HMAC keys must be
provisioned in host-local memory/credentials and never logged or committed; the deterministic test key is public
synthetic fixture data. HMAC authenticates and detects tampering; it does not encrypt traffic. Trusted endpoint
compromise remains outside this prototype's correctness guarantee. Replayed valid PUT/GET operations do not change
immutable content, while replayed responses fail the nonce check.

## Bounds, eviction and failure

- Default payload capacity 64 MiB, default 4,096 entries; tests use 4 MiB / 64 entries. Global LRU evicts unpinned
  oldest immutable entries under both byte and entry limits. A GET retains a Python immutable byte reference while
  sending, so concurrent eviction cannot turn a reply into another chunk. Metadata is bounded separately.
- One chunk is at most 1 MiB; metadata at most 2,048 B; exact packet size checked before parsing. `ofl` negotiates
  these limits at connect. One request per connection, at most eight connections. Registered transport memory is
  about 2.1 MB/connection; only bounded communication buffers are registered, never the RAM cache.
- `load_many` accepts at most 16 chunks and 16 MiB, with a total deadline at most one second (default 0.5 s).
  Any failed/missing chunk discards the staged batch. Large engine loads must split into bounded batches and publish
  completion only after all required groups/ranks succeed. The prototype does not claim a 150K prefix fits one batch.
- Single calls default to 100 ms, including connection setup budget; clients use a one-second circuit cooldown after
  failures. Deadline/miss/corruption/namespace/authentication/refusal/link-down returns `Load(None, reason)` and logs
  a reason. `put` returns false on failure. No successful prior response buffer is ever a fallback. No automatic
  transport retry occurs in the failed request. RDMA may be selected explicitly; TCP is the correctness alternative.
- `ofl` destroys the QP/closes the connection on timeout before returning; tests verify a delayed reply cannot later
  write into the client buffer. Reconnect uses fresh transport state. This is what permits safe compute fallback.
- RAM restart loses this tier and yields misses. Stage A's persistent SSD tier can remain the durable tier; RAM
  restart survival is not claimed. A scheduler lookup result is only a hint: an eviction racing GET still falls back.
- `memory_guard` checks `/proc/meminfo` and rejects capacities whose payload + conservative metadata/temporary-buffer
  estimate exceeds 64 GiB, or whose current MemAvailable is below twice that estimate plus a 32 GiB host reserve.
  Check free memory before launch. These are admission bounds, not reservations against later competing processes.
  Future long-running use needs resource ownership and pressure monitoring, with stores disabled/entries evicted
  when headroom falls; no daemon or service has been installed here.

X86host has 503 GiB RAM. Reserve **215 GiB for Engram**, **32 GiB for drafter host state/staging as an initial planning
allowance**, and **64 GiB for OS/other workloads/headroom** before assigning any production prefix capacity. That
leaves a theoretical 192 GiB, but this lane is capped at 64 GiB total and uses only megabytes. A 63 GiB payload cache
would hold roughly 248 independent aligned 150K prefixes (before metadata and reuse), rather than extending the
GPU context or avoiding compute for never-seen tokens. The drafter's actual host footprint and current MemAvailable
must replace the planning allowance before deployment. Engram is already resident during these tests; do not
subtract its reserve a second time from MemAvailable. Both features share CPU, memory, NIC and PCIe resources.

## Reproduce on x86host, CPU only

Code staged in `~/dsv41-offload/rp1/src/`; the following commands create no services, import no torch and contact
no Spark. Run from this worktree on Controller after copying these three Python files plus `offload/transport/ofl.py`,
`ofl.cpp`, `ofl.h`, `Makefile` to matching paths under that x86host directory:

```bash
source ~/.config/ndm/paths.sh
"$HARNESS_PYTHON" offload/prefixstore/remote/test_remote.py
"$HARNESS_PYTHON" -m mypy --strict offload/prefixstore/remote/{store,ofl_backend,test_remote}.py
ssh x86host 'source ~/.config/ndm/paths.sh
free -g
mkdir -p ~/dsv41-offload/rp1/bin
cd ~/dsv41-offload/rp1/src/offload/transport
make VERBS_INC=$HOME/dsv41-offload/o1/sysroot/usr/include OUT=$HOME/dsv41-offload/rp1/bin $HOME/dsv41-offload/rp1/bin/libofl.so'
ssh x86host 'source ~/.config/ndm/paths.sh
cd ~/dsv41-offload/rp1/src
OFL_LIB=$HOME/dsv41-offload/rp1/bin/libofl.so CUDA_VISIBLE_DEVICES="" timeout 60s python3 offload/prefixstore/remote/test_remote.py --network --backend tcp --ip 10.202.28.1 --port 18640'
```

Repeat the last command with `--backend rdma`, and separately with `--ip 10.202.29.1`. The process owns its own
server/listener and closes it at test completion. No background daemon remains. The library reuses the already
available O1 verbs headers; no downloads/installations were needed. Initial private rebuild failed because the
`bin` output directory did not exist; creating it fixed the build. No upstream launcher was run.

## Measured 2026-10-23 and remaining gates

All **13 tests pass** for each of the four combinations (TCP/RDMA × 10.202.28.1/10.202.29.1), with all clients on
x86host. Four concurrent fake clients check exact bytes. Tests cover immutable writes, CRC/authentication corruption,
rank/model/layout/scope isolation, LRU byte/entry bounds, partial-batch refusal, absent/restarted store, timeout,
late data and server/link-down. The strict type check passes all three Python files.

| X86host-local path | 1 MiB GET median | 16 GETs aggregate | Forced 20 ms timeout returned |
|---|---:|---:|---:|
| TCP 10.202.28.1 (private rebuilt ofl) | 4.213 ms | 216.7 MiB/s | 20.5 ms |
| TCP 10.202.29.1 (private rebuilt ofl) | 4.167 ms | 216.3 MiB/s | 20.5 ms |
| RDMA 10.202.28.1 (private rebuilt ofl) | 4.435 ms | 219.6 MiB/s | 22.1 ms |
| RDMA 10.202.29.1 (private rebuilt ofl) | 4.039 ms | 245.9 MiB/s | 22.1 ms |

Peak measured RSS across the final timed runs **61,264 KiB**, zero major faults and zero swaps; x86host MemAvailable was 299–303 GiB before
runs. These Python/hash/copy/loopback timings validate operation, not a speedup over the actual Spark tier. At
~250 MiB/s a full 150K transfer alone projects to ~1 second, before 1,954 RPC overheads; a C++ batched/gathered
adapter and fewer copies are required to approach the physical-link bound. Do not claim the 54 ms ideal as a result.

Coordinator gates after Stage A passes: implement trusted catalogue + tier adapter and completion publication;
prove rank-1 fresh bytes, partial-load invalidation, namespace refusal and corruption fail-closed in-engine; match
fresh-compute codes/logits and first-token KL; measure 150K with actual load caps and streaming/landing memory;
measure Engram + drafter contention and decode cost; confirm each Spark MemAvailable stays at least 5 GiB. Only
then consider an operational service or production RAM budget. Nothing from this CPU prototype is activated by a
launch config or mounted in the production engine.
