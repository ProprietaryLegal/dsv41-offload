# PP1 q transport and scheduling design

Design only, OFF. No v1 wire/serving files changed. Correctness reference:
[reference.py](reference.py); deployment boundary: [AUDIT.md](AUDIT.md).

## Distribution contract and precision

The wire must describe the **same categorical q that sampled each token**,
conditioned on the committed prefix and all earlier tokens of that proposal
chain. Confidence/margin and q(x) alone cannot reconstruct a residual.

The CPU oracle accepts full or sparse normalized weights. Its FP32 helper first
rounds weights, then defines q as their exact rational normalization at both
ends. It proves the semantics; arbitrary FP32 weights would require a carefully
matched GPU normalization/sampling implementation. Do not send independently
rounded probabilities after sampling from unrounded logits. Do not silently
renormalize a different q on the verifier.

Recommended bounded GPU prototype: **dyadic categorical q**, with total mass
M=2^24. Define integer counts c_v≥0 whose exact sum is M; q(v)=c_v/M.
Transmit q(v) as little-endian FP32. Every such value is exactly representable;
all partial sums of nonnegative counts up to M are also exact in FP32 when
scaled by M. Both sides can recover c_v by multiplying by M, require it to be
an integer, and validate the sum using uint64. There is no receiver-only
renormalization. Zero counts have exactly zero probability.

Proposal construction (new implementation still required): obtain globally
normalized draft weights after draft temperature and Markov bias, reject NaN,
+inf, negative probabilities and zero effective mass; −inf masked logits are
allowed before softmax. Allocate integer counts using floor plus largest
remainders, ties by lowest global token ID. Use FP64 allocation arithmetic and
assert 0≤remaining<M and exact final count sum; refuse a numerically inconsistent
allocation rather than clipping or silently changing totals. Tiny entries may
become zero. That changes the proposal only, not the legal target.

For sparse top-k, select the global support first, normalize within it, then
allocate M counts on that support. Drop zero-count entries and declare zero
everywhere else. Sampling uses a uniform 24-bit integer and exact integer
cumulative counts (first cumulative count strictly greater than the draw).
Record the semantic ID `dyadic24-standard-v1`; do not call this unrounded
softmax sampling. q can approximate the draft arbitrarily poorly and still
preserve p under correct verification; approximation can reduce acceptance.

The full target p is the distribution after the **unchanged target** processors,
penalties, grammar, T=1 and top_p=1. Ratio tests and full-vocabulary residual
sampling must use this p and the defined q. GPU finite-precision p/ratio/log
residual calculations still need numerical qualification against the rational
oracle; the design is not a claim of bit-identical real-arithmetic GPU sampling.
The existing local verifier caches pre-temperature logits and reuses the target
temperature; it cannot be pointed at a probability buffer. Supporting independent
draft temperature requires a new q-aware standard-verification interface.
Passing log(q) through the old head dtype/cache would round it and redefine q.

## Versioned v2 message design

Keep v1 greedy production intact; negotiate v2 capabilities before allocating
or accepting sampled requests. No auto-upgrade or environment-only enablement.

| Boundary | Required fields / invariant |
|---|---|
| HELLO | wire version, q semantic ID, standard-verifier ID, vocab 129280, global ID map/hash, draft checkpoint/native-or-adapted identity, target identity, maximum slots/depth/support, byte lengths, RNG ABI, precision |
| STEP request | session ID, slot, generation, round ID, absolute anchor/first-draft position, committed-prefix/state digest, fixed depth, proposal temperature/support policy ID; target processor/grammar/stop context version |
| Per-position reply | sampled global int32 token ID, representation dense/sparse, count of entries, q values and sparse global IDs; probability checksum; q conditioned on that row's preceding sampled tokens |
| Completion | matching session/generation/round/position/context, exact reply size and byte checksum, explicit status; whole row becomes consumable only after validation |
| Commit/reset | accepted draft count, correction/bonus IDs, new committed position; cancel/reset invalidates generation, every probability slot and all speculative Markov/KV/grammar state |

Bounds are negotiated and checked before memory access. Reject duplicate sparse
IDs (even duplicate zeros), unsorted/noncanonical support, out-of-range/padded
IDs, missing q, non-finite values, negative/non-dyadic values, wrong total mass,
zero q at the proposed token, wrong identities, unexpected sizes and stale or
duplicate generations/rounds. Dense padding beyond vocabulary is always zero.
Checksums detect corrupt transport, not proof that the worker sampled correctly;
that requires independent CPU/GPU distribution and RNG tests.

One RNG owner draws each proposal: worker leader, keyed by request generation,
round, absolute predicted position, domain and draw counter. The peer receives
the chosen token for the next Markov step; it does not independently choose one.
The Spark sampler owns separate acceptance and correction/bonus RNG domains.
Never reuse proposal noise for correction. Request seed overrides and reset
semantics are explicit. Reconnection invalidates old in-flight RNG/state keys;
retries either reproduce the identical reply or reset the proposal generation.
Do not redraw under a reused key and consume an earlier q.

Missing/corrupt/timed-out q discards the **entire unverified proposal** on both
Spark ranks. The capacity path may then run target-only from the committed
prefix using its ordinary target sampler, after synchronized invalidation.
It must never return q=None alongside sampled IDs. No partial commit on one
rank and fallback on the other. Exposed failures have counters and errors.
No new resilience/local-drafter allocation is assumed.

## Size and critical path

All sizes below are **ESTIMATED, exact shape arithmetic**, excluding headers,
alignment, checksums, padding and pool bookkeeping. MB=10^6 bytes.

| Layout, depth 5 | Bytes / request | Bytes / four requests |
|---|---:|---:|
| Dense FP32, 129280 tokens | 2,585,600 (2.586 MB) | 10,342,400 (10.342 MB) |
| Sparse k=16, int32 ID + FP32 q | 640 | 2,560 |
| Sparse k=32 | 1,280 | 5,120 |
| Sparse k=64 | 2,560 | 10,240 |
| Sparse k=256 | 10,240 | 40,960 |
| Dense TP2 shard, 64640 tokens | 1,292,800 per rank | 5,171,200 per rank |

At nominal 40 Gb/s, dense serialization alone has an ideal lower bound of
0.517 ms/request or 2.068 ms/four-request batch on one link. These are **not
measured latency**: actual D2H, host staging, RDMA/TCP, GPU H2D, synchronization
and graph boundaries add cost. Sending only q(x) is smaller but incorrect.

First correctness integration: leader returns full q to Spark rank 0; validate
and TP-broadcast to the rank(s) that own sampled requests in the installed
vocabulary/DP sampling layout. The installed runner can shard requests, which
is different from vocabulary sharding. Do not assume that exposing a TP2 head
means its current sampler consumes vocabulary shards. Full replication costs
another 10.342 MB/batch on the Spark TP fabric plus each rank's buffers.

Two in-flight dense batch slots require 20,684,800 bytes (19.73 MiB) per full
copy on each endpoint, in addition to today's buffers. Preflight actual ofl
slot/reply limits: the tiny v1 reply pools are not sized for this. Bound at four
requests and fixed depth five; refuse incompatible allocation/graph shapes.
Use preallocated registered host staging, not assumed GPUDirect support.
Publish q-ready only after all D2H/network/H2D work and validation complete;
release a slot only after every sampler/rank consumer is done. Timeout watchdogs
must invalidate the generation before buffers can be reused. Preserve current
process-proxy isolation and late-reply protections.

## Sparse support and vocabulary-sharded TP2 follow-up

Sparse k=32 is the initial bounded candidate, k=16/64 only a development screen.
Local top-k on each worker shard produces at most 2k candidates; merge these
with global IDs and deterministic ties to obtain exact global top-k. This is
not independent local normalization. Compute one global support/count vector;
leader samples once and broadcasts the sampled ID before computing the next
Markov-conditioned row. A correct fixed-depth chain requires that dependency
even when backbone rows were computed in parallel.

For dense vocabulary sharding, each rank computes its logit max and shifted sum;
all-reduce max/sum yields global draft normalization. Count allocation must
coordinate remaining tickets globally. Store global shard ranges in the header.
For true rank-to-rank links, each worker rank could send only its q shard to the
corresponding Spark rank. Concurrent links give an ideal 0.259 ms/request/shard
serialization bound; shared uplinks and transport ownership must be measured.
This requires a different target sampler, not a switch in today's proxy.

A sharded exact sampler chooses the shard by its global count mass using one
leader-owned integer draw, then the token within it. Each Spark rank constructs
max(p−q,0) for **every** token it owns, including tokens outside sparse q support.
Global residual mass selects the correction shard; local residual CDF chooses
the token; broadcast its global ID. Shard selection and within-shard draws must
form the intended categorical, with zero-mass shards impossible to choose.
Global target normalization and grammar masks must match target-only sampling.
No full-vocab gather is necessary mathematically, but the current serving
sampler must first gain and qualify this ownership model.

Candidate kernels, in dependency order: global logsumexp and counts, global
top-k merge, integer categorical proposal, sparse q lookup by global ID,
full-target residual mass and sampling. Test each against the CPU oracle and
small-vocabulary sharding emulator before using CUDA graphs. Profile them only
after source-faithful native qualification, counting exposed whole-round cost.

## Depth admission must not inspect its own future draws

Initial PP1 always fixes depth **before** sampling (five, or a separately named
fixed-depth arm). No confidence pruning, adaptive acceptance window, margin
masking, global best-depth search, batch-dependent retrospective prefix cut,
or full-block confidence scheduling is enabled for sampled proposals.

[DSpark v1 appendix A](https://arxiv.org/html/2607.05147v1#A1) gives an explicit
counterexample: first-prefix throughput 0.9 loses to target-only 1.0, but a
later confidence depending on the first sampled token can raise the two-token
score to 1.134. Global selection then admits the first token for some values
of that token and excludes it for others. An exact verifier alone cannot undo
that selection. This is a PUBLISHED counterexample, not deployment evidence.

Any later adaptive policy must make the event “position i will be verified”
measurable before x_i is sampled. It may use committed history, load measured
before drawing, prior rounds, and x_1..x_(i−1); it cannot examine x_i or any
later confidence that encodes x_i and then revoke i. A sequential stopping rule
decides admit/stop, samples only if admitted, and never reopens a stopped prefix.
Even if a backbone computes all head features at once, later Markov confidences
are not automatically eligible inputs to earlier admission decisions.

The CPU suite includes a separate toy selection-bias regression: when q=p is
uniform, selectively admitting the first draw only when it is token 0 and
otherwise resampling target-only makes its output mass 3/4 instead of 1/2.
The fixed-depth reference never performs this selection. Adaptive scheduling
requires a separate proof and conditional-frequency suite before hardware.
