# 0075 on v5 with the remote drafter

Draft only. Donor: HeNryous afe4d5e41ec6fa4db17c5cc88758af439a106913, Apache-2.0.
Affected operation: eager intermediate prefill replay of target layers 20–39. No weights/record formats change.
Mechanism: retain one late-layer row instead of 128 when q > 128 and remaining_after > 128; constrain BOTH
running/waiting scheduler paths so the successor cannot be <=128 rows. No 0072/0074 scheduler policies imported.

`patches/0075-v5.patch` is a hand-port against the pinned v5 reference plus this branch's O3 gather. Bases and donor
are in `0075-bases.json`. It needs no 0067 import. P1 is K1-owned: feed these file diffs into its mechanism when
available, hash the resulting composed files, and mount on both ranks. Do not mount a second model_state over K1's
0067 file: compose the two hunks; they affect separate regions. No P1 existed on K1's branch at initial inspection.

The upstream patch alone violates O3's tap assumption: only one row has real taps on a skipped step, but O3 gathers
128. The draft adds a typed `InputBatch.replay_tap_lens_np` handoff, reset on every graph/eager preparation. Model state
publishes actual kept suffix lengths, and BOTH fused and torch gathers use them. A drafting request with fewer
than its required candidate taps fails loudly. Wire protocol stays v1: partial-prefill STEP carries one real row,
NEED_DRAFT off. A gap resets the worker/rank-0 contiguous range. No zero-filled non-taps are certified as accepted taps.
CONTRACT §2/4.3 must be amended by O3/coordinator at integration to describe this optional provenance rule.

State trace:
- Target layers before 20 run on all scheduled rows. `DecoderReplayLayers.__call__` still invokes `write_batch_kv`
  before trimming: L20 source/compressed KV and SWA KV writes for ALL rows are retained. L20's own SWA metadata keeps
  the original replay bound (`first_swa_prefix`); only L21–39's disposable intermediate suffix is shortened.
- A skipped chunk has no requested prompt logits or sampled final token. Its one-row late-layer result/taps may
  differ; no later target consumer reads them: the successor trims, with its late-layer `kv_start >= seq_len-kept`.
- A non-skipped chunk >128 refills all 128 real taps and late-layer KV. A final chunk <=128 is safe because the
  preceding chunk could not skip when remaining_after <=128. Final replay/logits and decode use the baseline path.
- Worker context KV for each of three layers is a row-local transform of SAME main_x, not a causal stack of context
  layers. The final 128 real raw taps rebuild each layer; intermediate draft query outputs never enter context KV.
- Prefix hit: scheduler rewinds h-128; all required suffix rows pass normal replay. A newly admitted request/slot
  gets NEW_GEN. Cancel removes no required shared state; slot reuse changes generation. Preemption starts again
  through bounded replay; failed STEP resync uses only rank-0's real contiguous suffix. A one-row partial window
  can be resynced during prefill but cannot request drafting; the first drafting step must have a full window.

CPU gate: `python -m pytest -q offload/spark_leads/tests/test_replay_overlay.py`. The actual patch is applied with
zero fuzz, every base hash checked, Python syntax compiled; scheduler equivalence, 1,500 randomized long schedules,
real CPU O3 gather/ring/resync with mixed rejection, logprobs/MM exclusions and invalid provenance are checked.
This is not numerical proof for GB10 or the Triton gather.

Coordinator GB10 plan (150K, 256K, ~300K actual occupied; reserve output space):
1. Baseline D+K1 vs same +0075, switch off/on/off interleaved, identical legal-v2a, FP8, effort75, priority10,
   sampling, KV budget, chunks/graph policy; record exact overlays/sha/donor, per-request q/kept/NEED_DRAFT.
2. Cold, warm prefix hit, final remainder 1/127/128/129; budget leftovers <=128 from mixed decoding; 1–4 requests.
   Include prompt-logprobs and MM exclusion probes, cancel after a skipped step, reuse slot, prefix/preempt/restart,
   worker disconnect before final chunk and after it; both TP ranks must make the same decisions.
3. Assert no draft needs a partial window; no trimmed zero tap sent; final KV source writes equal baseline count;
   count resync/fallback/breaker events. Fixed-prefix/full-logit A-A vs A-B at matched shapes (item 6), needles + legal
   full task. Profile late-layer rows and first-consumer wait, not only reported prefill tps.
4. Measure cold TTFT, prefill time, per-decoder delivery, long-request progress, accepted tokens/round, peak RAM.
Stop on stale/missing final KV, provenance error, rank disagreement, deadlock, any unexplained distribution shift,
or repeatable decode regression >1%. Promote only repeatable useful gain (~5% end-to-end or justified component gain).

Approximation: intermediate one-row activations differ and are intentionally unconsumed. Existing v5 bounded replay
and cross-hardware/batch EXL3 rounding are inherited, not declared exact full causal recomputation by this patch.

K1 follow-up: its actual P1/packing overlay landed during X1 at87b08b5c486e68458223ea38dca6bb911b678400.
`compose_check.py` reads that pinned git tree, checks K1/X1 base/overlay hashes and applies both X1 patches with
zero fuzz in scratch. All composed Python/shell files compile/check; no K1 files/worktrees were edited. Its runner
0076a additions and the X1 hook compose. The resulting hashes are in `patches/K1-composition.json`.
