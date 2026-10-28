Status 2026-10-23: Conditional design only, used if real replay equivalence fails; calibrated amendments 1–2 apply. Real G1 data and matched-pilot gates remain pending. See [live roadmap](../../../documents/plans/roadmap.md) and [measured record](../../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# D2 decode-capture fallback: design and option A implementation

Lane D2-DATA, 2026-10-23. Written at the coordinator's request ("write that fallback's design now; no code needed
unless cheap"). It applies only if the prefill-vs-decode equivalence gate (`d2/equiv.py report`) returns verdict
`fail`. Then training waits for taps captured during generation, as below.

Implementation 2026-10-24: option A is implemented in the replay overlay and
`capture/decode_capture.py`, with the reviewed generator tag, decode assembler,
CPU tests and coordinator-only dry-run-default smoke/apply package. See
[DECODE-CAPTURE-APPLY.md](DECODE-CAPTURE-APPLY.md). The W9 gate failed; no new
collection, live qualification or training was performed by the implementation
lane. The calibrated gate/design below remains the pre-registered contract.

## 1. When it applies

The rule as `equiv.py report` implements it (coordinator, 2026-10-23; amended 14:59Z and again 16:30Z, both before
any replay data, in offload-x86host `documents/plans/2026-10-23-d2-plan.md`, 4dd6f74 and 89a2098). `equiv drafts` runs
at `--every 1`: all 542 blocks, about 90 min of CPU in 3 shards (about 30 s per block per process at 8 threads for the
three arms, measured on Controller), after the window and off the critical path.

- **pass:** at least 99 % of the gate blocks are equal between the arms D (decode taps) and R (prefill taps). The
  gate blocks are those with at least one decode row in the 128-row window; the identical-input blocks are reported
  apart.
- **pass_bf16_noise:** otherwise, judged against a same-run noise control. The control arm N is D with Gaussian
  relative noise sigma = 2^-8 on every decode-captured row (seed 20261023), drafted through the same FP32 path on
  the same blocks. All three must hold:
  - position-1 agreement D vs R >= the control's position-1 agreement - 0.05 (absolute; amendment 2: a gross-error
    backstop, since first-token flips happen at the drafter's near-ties and measure the drafter, not the taps);
  - at every position, the median KL(D || R) <= max(1e-3, 2 x the control's median KL(D || N) at that position);
  - at every position, the median KL(D || R) <= 0.05 nats (a hard ceiling, at least 13x below the smallest
    drafter-to-target KL measured, 0.65 nats).
  Prefill taps are then used, and the result is recorded as BF16-level noise. The mean accepted length of D, R, N
  and the engine against the realized tokens is recorded, not gating.
- **fail:** anything else. Training waits for this fallback (option A below).

Why the control (D2-DATA flag 9, resolved by the amendment): 0.4 % elementwise noise on the window-1 decode taps,
which is the size of BF16 rounding, through the V1 FP32 reference drafts path, gave only 4 of 8 differing blocks equal
and a median per-position KL of 0.008–0.020 nats. The first rule's fixed 1e-3-nat bound would have failed on rounding
noise alone. The calibrated rule passes that case and still fails a real divergence (10x the control's KL, or
position-1 disagreements the control does not show); `d2/tests/test_equiv_control.py` checks both.

## 2. The prefill zero-taps question (resolved: replay is safe; guarded anyway)

CONTRACT.md line 34 says this about eager prefill steps: layers ≥ 20 run only on each request's last min(q, 128)
rows (patch 0034, decoder SWA bounded replay), and the trimmed rows' taps are zero.

The image source makes one exception, and CONTRACT.md does not mention it. In `nvidia/model_state.py`:

- `add_request` sets `_keeps_rows_np` for any request whose sampling params carry `prompt_logprobs`.
- `_prepare_replay_batch` then uses `kept_lens = query_lens` for that request. It does not use `min(q, 128)`.

Every replay request carries `prompt_logprobs: 20`, so every replay row runs every layer. Phase 0 is consistent with
this: prompt logprobs come from the last layer, and the committed token was in the top-20 at 99.5–100 % of span
positions. Trimmed rows would have produced garbage there.

The guard does not depend on this reading. Commit e0760ef added it:

- The overlay counts every row in which any layer's 5120 taps are all zero. It records them in DONE.json as
  `zero_rows` and `first_zero_row`, and in ERRORS.jsonl as kind `zero_taps`.
- `replay.py run` (its first-sequence check) refuses a sequence with `zero_rows != 0`, and so do `replay.py copy`,
  `equiv.load_prefill` and `dataset.py build`. A DONE.json with no `zero_rows` count is refused too.
- `shards.check_record` (DATASET.md rule 3) rejects a zero layer row.
- `test_zero_taps.py::test_engine_keeps_prompt_logprobs_rows` pins the engine lines this relies on.

CONTRACT.md is owned by lane O3. It should gain the exception: "requests with `prompt_logprobs` keep every row".

## 3. Option A (recommended): capture during generation (Spark overlay, decode mode)

Add a second mode to `capture/speculator_replay.py`, enabled by `VLLM_DSPARK_DECODE_CAPTURE_DIR`. It is inert unless
set, and the image prefix stays byte-identical.

### Requests and tags

The generation driver already tags its requests `d2g.<pid>.<seg>`. The tag would become `d2c.<pid>.<seg>.<r0>`, with
`r0 = max(0, prompt_len - 128)` taken from the driver's `prompt_ids`. That is a one-line change in `generate.py`.
Untagged requests are never copied off the GPU, so NDM's requests and anything else stay in device memory only. The
per-round cost for them is one regex over `req_ids`.

### Rows per round

For each tagged request, save the rows the target has just validated. These are the rows the x86host worker receives
(CONTRACT "context rows per step"): the first `num_scheduled − num_rejected` rows of the request's span, at their
positions. Keep only rows with `p >= r0`.

- **Decode steps run in CUDA graphs and compute every row.** In a round with m accepted drafts, rows 0..m carry the
  true tokens at P+1..P+1+m, so each position's row is computed exactly once with its correct input. These are the
  taps the drafter sees in production, so no equivalence question arises.
- **The prompt tail r0..prompt_len−1 comes from the last prefill chunk(s).** Each chunk keeps its last min(q, 128)
  rows. If the last chunk is shorter than 128, the previous chunk's kept tail covers the rest, so the last 128 prompt
  positions are always computed. The zero-row count catches any gap.
- **Position L−1 (the final sampled token) is never fed back,** so it has no row. The assembler sets the shard's L to
  `L_gen − 1`, which drops one span position per sequence. `top_ids`/`top_lp` still cover that range.

### Files

Rows go by pwrite to `<dir>/<pid>/taps.bin` at `(p − r0) * 30720`, with ids to `ids.bin`, as in replay mode.

- **Resumes.** After an abort, the driver resumes with prompt + partial as a new request (a new seg). Its prefill
  recomputes the prefix hit's last window. Those rows are already on file, so the first row (the decode one) is kept.
  A re-sent row whose bytes differ is counted as `dup_rows_differing`, as information.
- **Completeness.** No DONE.json is possible: the overlay never sees a request finish. A ledger records
  (round, seg, first position, rows). The assembler checks coverage r0..L−2 against the generation state's final L,
  plus `ids.bin == tokens[r0:L−1]` (rule 2, which proves the keying).
- **Write path.** Pinned host staging with `non_blocking` copies and a writer thread, so the serving step never waits
  on disk. This differs from replay mode, which can afford synchronous copies at max_tokens 1. The writer keeps
  `fdatasync` + `POSIX_FADV_DONTNEED` every 256 MB, because GB10 CUDA cannot reclaim page cache.

### Top-20 logprobs

There are two sources:

- (i) The replay window as planned, using only `prompt_logprobs`. Its taps can be discarded, or kept as the R arm of
  a full-dataset equivalence check. Its cost is roughly the same as the current replay.
- (ii) `logprobs: true, top_logprobs: 20` on the generation requests themselves. This removes the replay window.
  However, vLLM support for logprobs together with DSpark speculative decoding has not been verified on this image.
  It needs a 3-prompt check on :8001 in a coordinator window before anyone relies on it.

### Volume and cost

- About 30 KB per generated token: about 31 GB for 1M tokens, written to the Spark at about 3 MB/s with 3 streams.
- The Spark has room: D2-DATA.report.md §3 records 1.8 TB free on Spark A's `/`. Re-check this with `df` before
  the run. Controller's disk, by contrast, is 95 % full.
- The copy to x86host reuses `replay.py copy` (copy_to_x86host over the direct link, O_DIRECT reads, sha256
  verification).

### Where it runs

The overlay has to be mounted on the serving container. That means generation runs either:

- on production with a D config plus the decode-capture overlay. That takes two coordinator restarts (in and out),
  and the overlay is inert for everyone else; or
- on :8001 in a window, which blocks NDM for 3–4 h. Not recommended.

Production with the overlay is the practical choice. It is the coordinator's call, because it changes production.

### Effort

About 150 lines in the overlay, 1 line in `generate.py`, and an assembler in `dataset.py` (`--taps decode_capture`;
the equivalence gate does not apply to it). Tests reuse the fake overlay harness (`test_replay_overlay.py`) with
multi-segment, resume and zero-row cases. Estimate: 2–3 h of lane time, plus a 3-prompt gated smoke in a window.

### Hedge

If the coordinator wants no risk of a second generation pass, tonight's generation could already run under this
overlay. That would require the code above before the generation starts, which does not fit tonight's 2 h target.

If the fallback triggers later, D2 needs a fresh generation pass under the overlay. Tonight's text cannot be
regenerated bit-exactly, because T=1 sampling and batching are not reproducible. Its prompt_logprobs and replay taps
would still be valid data for the prefill-tap arm.

## 4. Option B: capture on the x86host worker

The production worker already receives every validated decode row and each prefill tail, at most 128 rows per
request per step, over the direct link. Capturing there writes nothing on the Sparks and uses x86host's NVMe.

The problem is identity. The wire carries slot, positions, session and seq, but no request id (CONTRACT §6). The
worker therefore cannot tell D2 requests from NDM's without a change to the contract (v1.9: a request tag or hash in
the per-request record) and to the Spark proxy that fills it.

It would also put capture code into the production worker's hot path. A dropped or slow write must never delay a
reply, so it needs the same pinned-staging and writer-thread design as option A, but on a service that is never
restarted lightly.

Not recommended unless Spark-side writes are ruled out. Its estimated effort is about a day across the O2 and O3
lanes.

## 5. Gate for fallback data

- The zero-row and coverage checks of §3.
- G1(a) plumbing.
- G1(b) on the `acap` split, unchanged.

There is no equivalence gate, because the taps come from decode itself. `dataset.py build` gets
`--taps decode_capture`, which does not require `--equiv-gate`. The MANIFEST records the tap source.
