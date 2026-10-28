# CONTRACT v1.8 bounded-replay audit

Pins: O2/O3 at X1 base 21bc3ab; v5 engine 6e517b15.dsv41gb10; official DeepInfer
9bd9cf243fc53ef9afbf8f04d391d989117d6ab9 (Apache-2.0), HeNryous 0075 at afe4d5e.
Affected op: reconstructing proposal context KV after prefix hit, rollback, final prefill, reconnect and RESET.
Verdict: 128 REAL taps suffice for the current DSpark worker's full forward-only context window. They do NOT
prove full target causal-state equivalence, identical proposals across batch shapes, or arbitrary rollback recovery.

Trace the dependency, not the layer count:
1. `worker/model.py:main_x` projects/normalizes cat(tap37,tap38,tap39). Target taps already encode the target history.
2. `context_kv` iterates all three stages but EVERY stage projects the SAME main_x row with its own wkv, kv_norm,
   RoPE and record quantization. `insert` has the same fused logic. No context layer consumes a previous context
   layer's outputs. Thus no 3*128 raw-history requirement arises in this DSpark context path.
3. Query path is different: `forward_block` runs the complete five-query noncausal block through all three stages;
   each stage reads its own rebuilt context KV and all five current queries. Query outputs are disposable and never
   committed to the context ring. n_out truncates head/Markov/confidence only. Sequential Markov still depends on
   previous sampled draft ids within the returned prefix.
4. Block queries read [max(anchor-128,0),anchor) for EVERY query, plus all five queries. Causal 127-token formulas
   are wrong here (CONTRACT1.7 corrected this). Worker context position predicates and generation gate stale slots.

| Lifecycle | What the current source actually guarantees | Qualification / counterexample |
|---|---|---|
| Forward decode + rejection | Only target-verified accepted rows enter each ring. Rejected proposal rows never committed. Window slides forward as normal. | Rejection within the just-verified block is not rollback of previously committed history; no extra suffix needed for that case. |
| Multiple partial chunks | Last 128 real taps overwrite obsolete context; gap starts new contiguous range. | With 0075, propagate actual one-row provenance (item1). Never draft before suffix restored. |
| Final chunk >128 | Normal final replay delivers the entire window, recomputes every stage's context KV. | Target source-layer KV writes must remain full; match all taps and final distribution at same shapes. |
| Final chunk <=128 | Accumulates with previous chunk to cover full window. | Previous chunk cannot be skipped if remainder<=128; scheduler min-chunk condition binds running AND waiting paths. |
| Prefix hit at h | Scheduler rewinds to h-128; target recomputes taps from replay_start. Current contract can admit complete suffix. | This is equivalence to v5's bounded-replay path, not proof that multilayer target replay equals full untruncated causal recomputation. Prefix h<=128 is recomputed fully. |
| Lower-position same-generation update | `WindowRange.apply` drops rows >=first_pos. Entries outside contiguous [lo,hi] cannot be read. | A window [872,999], rolled back to anchor995, has only 123 rows; needed positions867..871 were evicted. Same-session RESYNC resends only those123, so cannot repair the missing five. PARTIAL_WINDOW persists until fresh real rows/refill cover the window. |
| RESET / NEW_GEN / worker restart | Worker clears ranges; NEW_GEN older than current remains stale. Proxy backup can restore current real range on reconnect under correct session/generation. | RESET then draft-with-zero rows is invalid unless a real full suffix is resent. New owner must refill; old cache data must never be certified by generation alone. |
| Resync after transport failure | Rank-0 ring can resend every row it actually retains. No query output is needed. | EXL3 rounding changes with replay batch shape; proposals/confidences can change near ties. Cannot claim bit-identical proposals. Partial backup stays partial. |

Required CONTRACT wording correction at integration: §4.3's induction assumes forward-only positions or a fresh
full target replay after backward moves. §7.4's “exactly ... with no failure” concerns retained raw rows; derived KV
and proposal identity are qualified by batch rounding. Arbitrary same-generation backtracking is supported as a
PARTIAL_WINDOW proposal approximation, not exact reconstruction. Do not silently broaden those claims.

For exact proposals under a maximum rollback B, retain W+B source rows (and include every state lane needing older
positions), or force fresh target suffix replay at rollback. Our wire HELLO/window=128 and fixed 15MiB backup do not
support that extension today; changing retention entails protocol/buffer/worker changes, not copying a constant.
A simpler integration gate: classify real lower-anchor events in engine logs; if any lack full target replay,
record partial-window proposals or suppress proposals until refilled. Target verification remains authoritative.

DeepInfer comparison (read-only source): README's decoder YoCo compact mode uses keep_tokens=140 for its default
configuration, and `deepinfer/worker/model_runner.py:108` asserts keep_tokens >= window_size + configured max
backtracking. `examples/yoco/yoco_equiv.py:385` also checks that bound and SWA storage size. This is retained TARGET
suffix state with allowed rollback; it is not our row-local DSpark conditioning transform. Its example140 is not
portable. Batch-invariant/deterministic diagnostics also constrain speculation policy; changing head/path/shape can
change proposal schedules even when a fixed verify kernel is batch invariant. Use matched teacher forcing (item6).

Explicit approximations/inherited differences:
- v5 target decoder bounded replay limits late layers' history at chunk/prefix boundaries; full causal-stack
  equivalence is not established by this audit. 0075 preserves its final consumer path subject to scheduling gates.
- Cross-GPU EXL3 accumulation, insert batch shape and kernel shape rounding can alter draft tokens/confidence;
  existing O2 parity evidence, not a theorem of ring size, qualifies them.
- PARTIAL_WINDOW intentionally drafts with fewer context rows (verifier keeps target decisions authoritative).
- Greedy worker proposals replace no target distribution; “target remains exact” assumes correct verifier/RNG and
  sampler integration, and is not proof of identical stochastic output text or absence of later buffer races.

Measurement: CPU regression `tests/test_state_audit.py` reproduces backtracking insufficiency, reset/reuse and
checks actual row-local worker source; existing O3/O2 suites cover stale NEW_GEN, range, reply validation, failure.
Coordinator capture: final 128 taps + every stage KV before/after fresh insert vs resync, q/context/graph shapes,
forward vs backtrack anchors, arbitrary rejection patterns, prefix replay and disconnect/cancel/new owner. Full
fixed-prefix target distributions at matched shapes, accepted proposals/confidence histograms, all PARTIAL_WINDOWs.
Stop on any stale-generation read, missing required row certified OK, unexplained non-near-tie proposal change,
rank disagreement, or target distribution shift above matched A-A floor. No GPU test run by X1.
