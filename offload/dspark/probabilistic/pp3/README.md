# PP3 sampled proposal experiment

OFF, unqualified for serving. `DSPARK_SAMPLED_TOP32=1` admits the worker's
explicit `TPDrafter.draft_sampled` method. The ordinary draft/graph path remains
greedy. The sampled entry uses no proposal RNG for T0 rows. Proposal depth is fixed before drawing; only positive target-temperature
rows sample, at independent drafter tau .7. q is sparse top32 with exactly 2^24
tickets: FP32 exponentials, FP64 normalization, floor counts and remaining
tickets to the largest-logit token. This differs from PP1's suggested largest
remainder allocation. Its semantic ID is `dyadic24-floor-first-v1`.

Two GPU launches per position select tile candidates, then fuse global top32,
temperature, normalization, integer ticket allocation and Philox sampling. Keys
are request-generation-owned seed, uint64 round and uint64 absolute position,
with a separate proposal domain. Support is sorted by ID. Ties choose lower ID.
Negative infinity masks are allowed; NaN/+infinity/zero mass fail the whole row.

MEASURED synthetic solo results: [initial](producer-initial.json),
[partial top-k](producer-topk.json). The latter takes .155–.209 ms at B1–B4,
depth5, **missing the absolute .05 ms goal**. These graph-replayed selector
microbenchmarks exclude Markov, pair communication, H2D/D2H and verification.
Torch argmax is a microbenchmark comparator, not the production fused selector.
The eager model integration still computes the original greedy selector and
materializes Markov logits, stages both vocabulary shards onto the leader GPU,
and copies the chosen token to its peer. It is a correctness prototype; neither
candidate-only pair exchange nor the real pair is qualified here. No exposed
live-pair speedup is claimed. Default paths import no sampled CUDA code.

The production EXL3 Python binaries target CPython3.12, while the required
`/home/operator/miniconda3/bin/python3` is CPython3.13. The new producer runs
with that required interpreter (Torch2.12.1/cu130, Triton3.7.1). Production-K2
whole-model replay under this interpreter is a remaining qualification gate;
PP2 used a separate cached CPython3.12 process. No binaries/environments changed.

Typed Triton boundary stubs in `typings/` describe the JIT API, not GPU arithmetic.
Use `MYPYPATH=offload/dspark/probabilistic/pp3/typings` for strict mypy. Kernel
scalar/vector values are explicitly `Any` because Triton constructs their types
at specialization; host buffer and switch APIs are typed.
