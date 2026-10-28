# 0078 adapted to O1 remote rows

Draft overlay: `patches/0078-remote.patch`, pinned O1/v5 bases in `0078-bases.json`.
Donor HeNryous afe4d5e41ec6fa4db17c5cc88758af439a106913 (Apache-2.0), only host hashing semantics reused.
Affected op: next known prompt chunk's immutable Engram rows on layers 1/14, both TP ranks.
Mechanism: bounded exact row cache filled by O1's EXISTING RemoteProxy/connection, never a second reader/table copy.
Switch `VLLM_X1_ENGRAM_LOOKAHEAD=0` default; screening value 512 tokens (maximum 512, intentionally smaller than donor).

The patch modifies O1 engram.py + its C++ extension, adds `row_lookahead.h` beside the extension, adds
`vllm/v1/worker/gpu/x1_engram_lookahead.py`, and hooks v5 model_runner after forward on every real step.
The plain `lookahead_bridge.py` and header are reviewable originals mirrored in the patch. K1 P1 composes runner
changes; O1 stage manifest/JIT hashes MUST include the new header and bridge before it is loadable. The draft also
updates O1's explicit staging allowlist for the header and includes it in the JIT module hash. The bridge and runner
are engine files mounted through K1 P1. This is still an unbuilt source draft, not a staged deployment.
Do not enable in production. Full JIT/GB10 execution is an outstanding qualification gate, not a CPU result.

Bounds and ordering:
- Hash only prompt tokens already present in host-visible history. Copy <=515 token ids per request while the
  engine owns the slots; hash synchronously before returning. Never read generated future ids or retain a mutable
  history view. Host hash is independently checked against a scalar reference; GB10 row-id parity still required.
- One replacement snapshot/table, <=8192 unique local row ids, <=512 rows per non-preemptible network burst.
  8192*264 = 2,162,688 payload bytes/table; map/list overhead is additional and must be measured (budget 4 MiB/table).
  Four tables across two Sparks: <16 MiB cache budget + existing staging, no extra table copy on x86host.
- Every real forward replaces queued hints; decode-only steps submit empty snapshots. Abort/slot reuse/preemption
  cancels at the next engine step. A replaced in-flight generation cannot publish. No inference after an abort:
  at most the current bounded snapshot remains, not an unbounded continuing job. Cached rows are immutable and
  request independent; no token history/client salt is stored in the row cache. Table identity scopes each cache.
- Poll ALL foreground tables before ONE hint burst; layer-1's layer_hash_index=0 wins idle hint service over 14.
  A foreground arrival during a burst waits at most its bounded RPC deadline (2 ms plus teardown, to measure).
  No hint disk fallback, new connection, reader pool or background polling thread. Existing fallback remains exact.
- Lookahead forces O1's private bounce buffers. Hints write only those private registered destinations, then cache;
  foreground hits copy exact weights/scales into the live staging before publishing done. Partial hits split
  missing ids into a compact RPC and scatter its private-bounce reply back to original positions; duplicates and
  ordering are preserved. Thus a bounded8192-row cache can help a49152-row foreground chunk. Any RPC failure
  overwrites ALL requested output rows through the original exact fallback. Without private destinations,
  NIC lookahead could overwrite staging still being read over UVA.
  Completion is never published with only part of the output filled. Registered bounds unchanged. Reconnect/identity re-OPEN clears cache.
- Destructor leaves proxy membership before closing/freeing destinations. Cache mutations are locked; foreground
  and hints share `serve_mu_`, connection owner retains `client_mu_`; no Python callbacks on proxy/GPU threads.

Measurement plan (coordinator, after fixed-B baseline):
1. Build new extension detached from serving; stage/header hash and diff verified, both ranks, B0/B1/B0 interleaved,
   32K/150K then 256K/300K if useful. Same prefill shapes/effort75/prio10, cold and real prefix hits separately.
2. Instrument paired monotonic timestamps per table/lookup generation: stream writes request flag (issue), proxy
   observes request (host_observe), RPC submit/complete, cache copy complete and release-store done (ready). A
   bounded pinned marker written on the GPU stream before issue avoids conflating earlier target queueing time.
   CUDA events before first `_ready_rows` wait and immediately after it measure first-consumer wait, resolved later.
   Join by table identity + step + generation, never subtract medians. Report issue-to-ready, issue-to-observe,
   RPC, copy, first-consumer wait p50/p95/p99/max separately; existing `serve_ns` is observation-to-ready only.
3. Count hint rows, exact-hit rows, dedup, overwritten snapshots, canceled publications, wasted bytes, hint errors,
   foreground fallbacks, per-layer/rank waits, RSS/pinned bytes and x86host CPU load alongside remote drafter traffic.
   Current draft exports hint rows/hits; fine phase markers and cancellation counters need instrumentation before
   performance attribution. Do not infer exposed wait from the RTT histogram.
4. Stress cancel during RPC, cancel then same slot/new request, prefix hits, restart with different identity,
   lookup partial hits + duplicate ids + invalid ids, both table bursts, concurrent decoder arrival, timeout, TCP
   and RDMA, direct/bounce reference. Byte/scales identity before distribution tests at matched shapes.
Stop on any corrupt/stale byte, visibility/order fault, changed identity accepted, delayed decode >1% repeatedly,
foreground fallbacks in timed runs, or cache/RSS over budget. If residual layer-1 waits are already hidden, reject
lookahead rather than importing upstream's disk-reader +8–9% claim. This port's benefit is unmeasured.

Review update: the first draft had all-or-miss cache consumption, which cannot help a normal4096-token chunk
with an8192-row cache. The corrected draft fetches/scatters only misses, with a CPU-native partial-hit regression.
K1 P1 is now present at87b08b5; see `patches/K1-composition.json` and `compose_check.py` for verified zero-fuzz
composition against its real manifest. This supersedes the initial not-yet-present observation.
