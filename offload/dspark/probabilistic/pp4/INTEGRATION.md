# PP4 integration (default OFF)

`DSPARK_SAMPLED_TOP32=1` selects the new K2 worker path; the experimental
engine still requires `VLLM_DSPARK_SAMPLED_TOP32=1` and PP3's guarded standard
verifier ABI. No production launcher enables either switch.

`TPDrafter.heads` branches before the original selector. A sampled position
launches one `sampled_fused._kernel` per physical rank, or one kernel containing
both logical vocabulary shards in solo emulation. It reads the existing BF16
head logits and the matching Markov cache/weights directly. Cached rows add
their bias; uncached rows retain the per-column FP32 product/tree reduction
and BF16 bias/add rounding. No additional conditioned full-vocabulary tensor,
greedy selection, full-logit peer copy or full-logit host copy occurs.

Each tile writes only its top32 packed value/ID candidates. A GPU release/acquire
retirement chain selects the last retiring CTA to finish the row; it never
spins for unscheduled blocks. That CTA reduces candidates, normalizes top32
at tau0.7, allocates exactly2^24 tickets and draws with the existing request-owned
Philox key/round/absolute-position domain. IDs, emitted FP32 q and sampled
tokens retain `dyadic24-floor-first-v1` semantics.

On the physical pair, each final CTA sends32 packed candidates (256B/request/
position), signals with a system release and waits with a bounded system
acquire. Both ranks reduce the same union, draw from the same q/key and
acknowledge consumption before a subsequent position can reuse the inbox.
Epoch exhaustion, timeout and bad logits fail closed with sticky peer errors.
The host discards the entire reply. An invalid intermediate uses an in-range
internal continuation0 while retaining its error flag, so later Markov
positions cannot dereference-1. This internal value is never a successful
proposal or a substitute for q. A sticky peer failure requires worker teardown.

Peer addresses are runtime arguments, not JIT constants. Both rank
specializations are compiled before either kernel launches; otherwise a GPU
could timeout while its peer was still compiling. Capture refuses an unprimed
paired specialization. Physical-pair execution/timeout visibility has not been
qualified by solo compilation and remains a coordinator gate.

The worker captures separate sampled graphs alongside original graphs when
the switch is explicitly enabled. Graph inputs include per-rank request keys
and T>0 masks; outputs/q/error state have fixed lifetime per(B,k,n_out).
Missing sampled graphs on a graphed worker refuse eager fallback. Fixed
depth4/5 is supported, including production query-block5/n_out4 and5; unused
output/confidence/probability padding is zero. The normal server uses its
negotiated fixed block depth.

OFF uses the original selector and graphs. An all-T0 request batch with the
switch ON reuses the original graph and output objects. Mixed batches use
the fused selector with a per-row T0 argmax and no Philox draw; q is zero on
those rows. Mixed T0 equality therefore needs numerical validation in addition
to the all-T0 branch test. Native, partial-rank, margin and traced sampled
head execution are outside this path and refuse.

PP3's transport, generation/slot mapping, probability validation, standard
min(1,p/q) acceptance, full-target residual and failed-proposal invalidation
remain unchanged. Frozen heldout replay consumes the actual new producer's q.
Solo qualification and whole-worker timing do not establish physical-pair,
GB10, grammar/parser or live throughput qualification. The separate PP4
coordinator runner refuses missing hash-bound admission evidence.


## Round2 review corrections (CPU only)

S52's four fixes are implemented in the PP4 launcher/preparation layer. The
nested production-imported remote_dspark package is replaced exactly once.
Verifier preparation starts from the reviewed full K1 runner682eb895…;
verifier-k1-receipt.json records refreshed source/output hashes. Both MTP None
guards remain. Receiving-host ownership includes nested experimental/recovery
SSH/SCP, with persistent fences, leases and deferred exact-HOLD close. Failed
rounds gate growth over an immutable post-warmup/post-greedy baseline across
headline, profiled and final snapshots.

Physical qualification has a concrete coordinator-only pair_qualify entrypoint,
called first inside each HOLD under receiving-host registration and both GPU
locks. It cannot synthesize qualification from these solo receipts. It executes
both peer kernels, changed-input graphs, mixed T0, isolated timeout/sticky
teardown, and physical whole-worker identity against the archived solo N4
receipt. It was written and CPU rehearsed, **not executed on hardware**.

The whole_worker script now exposes a guarded physical qualification-only
mode. full-worker.json remains the historical56f2515 solo measurement; its
runner hash refers to that historical source, not the new wrapper. Worker
numerical code and historical measured receipts were not changed or rerun.

The live comparison is explicitly CONTROLLED fixed5 common PP4 runtime OFF/ON,
not a production-exact performance claim. Measurement is split A1/B1 then
B2/A2 across two restored HOLDs; dedicated measured whole-block evidence is
required before either can contact hosts. Missing budget evidence refuses.
See ../../../launch/windows/PP4-COORDINATOR.md for the90min bound, pair-only
qualification path and outstanding engine/GB10 gates.
