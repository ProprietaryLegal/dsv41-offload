# PF1-CAUSE item 3: no source-only fix; what the window must show first, and the fix each answer leads to

## Plain language

- The source does not show the cause. Nothing in the MoE half depends on max_model_len, so there is no line to
  change that removes "the max_model_len dependence" with confidence. Writing a patch now would be a guess.
- The window in WINDOW.md answers it in one HOLD. Its optional third arm **is** the leading fix candidate, and it
  needs **no code**: K1's existing switches `VLLM_EXL3_TF_PROMPT_TRANSIENT=0` and `VLLM_EXL3_TF_PROMPT_ALIAS=0` hold
  the routed-expert prompt scratch from start-up instead of allocating it per MoE call. It costs about 1 GiB more
  memory per Spark. If it removes the extra 0.63 s per chunk, the cause is the per-call scratch (H1) and the remaining
  work is a memory decision, not a code change.

## 1. What the window must show before any fix is written

| # | Required observation (rank 1 unless stated; 524,288 − 327,680, per full 4,092-row chunk, matched prefixes) | Purpose |
|---|---|---|
| W1 | `moe.local:4092` ≈ +600 ms on rank 1 and ≈ 0 on rank 0; `attn.total:4092` ≈ 0 on both | confirms item 1 with direct timings |
| W2 | the split of that +600 ms across `tf.*:4092`, `mhc.ffn_pre`, and the remainder (router + shared expert) | names the component (H1/H3 vs H2 vs H4) |
| W3 | `tf.*:128` (replay tail) unchanged | separates per-call scratch (H1) from persistent placement (H3) |
| W4 | allocator counters and scratch pointers on `moe.routed.tf` / `tf.rot_in` / `tf.down_combine`: equal per-chunk counts at both caps, or a step per call at 524,288 only | separates fixed placement from per-call mapping churn |
| W5 (arm 3) | `moe.local:4092` and the 150K traced TTFT with `PF1C_TF_TRANSIENT=0` at 524,288 | tests the leading candidate directly |
| W6 | host-parity reads (kernel, driver, THP, buddyinfo) on both Sparks | explains "why rank 1" if placement is the cause |

## 2. Paths after the window

| Window answer | Path | Change (all default-off or profile-scoped; none touches production until a promotion) |
|---|---|---|
| W2 = `tf.*`, W5 removes it | **A: H1 confirmed** | Candidate A1, env only, K1 profile 524,288: `VLLM_EXL3_TF_PROMPT_TRANSIENT=0`, `VLLM_EXL3_TF_PROMPT_ALIAS=0` (+1,014 MiB held per rank). If its memory rows fail, candidate A2 (a small default-off K1 patch, written only then): keep one persistent prompt scratch of the ALIAS layout (750 MiB) allocated at profile time, so placement is fixed early but the 264-MiB z/rot split saving of 0076e stays |
| W2 = `tf.*`, W5 does not remove it, W3 shows `tf.*:128` growing too | B: H3 (persistent buffers / weights) | next window: nsys of one full layer on rank 1 at both caps (SP1's recipe) with the kernel names these spans give; no fix yet |
| W2 = router + shared expert | C: H2 | spans one level deeper (image `vllm_exl3_moe/moe_method.py` `_apply_with_shared` and `dense.py` reconstruct), same overlay style |
| W2 = `mhc.ffn_pre` | C': H4 | spans in the tilelang mHC dispatch |
| W1 fails (no local growth on rank 1) | D: H5 | an nsys window on both ranks; the all-reduce wait would then not be local work |

## 3. Candidate A1: invariants, and how a window verifies greedy identity and the gain

- **Invariant (source).** TRANSIENT and ALIAS change only buffer lifetime and storage location; K1 states "Same
  kernels, same arguments, same bits" for both (K1 `vllm_exl3_moe/tf_prompt.py:27-39`, `tf_experts.py:28-35`), and
  every element a prompt call reads is written earlier in the same call. With TRANSIENT=0 the buffers are the original
  pre-0076d persistent scratch (`tf_prompt.py:82-90`, `tf_experts.py:384-417` `else` branches). No arithmetic,
  scheduling, chunk plan or kernel choice depends on the switch.
- **CPU evidence today.** `test_overlay.py` runs the real K1 `routed()` with TRANSIENT=ALIAS=1 against a recording stub;
  the same harness can assert the T0 launches (CPU, before any window) — the kernels then read/write the persistent
  buffers instead of the per-call ones; launch list and scalar arguments are unchanged except buffer identity. This is
  boundary equivalence, not native parity.
- **Greedy identity in a window.** On the T0 arm, before its cold request: the existing short-prompt greedy check
  (W5's `greedy_check.py` short prompts, temperature 0, the strict-identity option of S22 FLAG C) against the production
  reference outputs, plus the 8K/32K compare against the engine's A-A floor (S18 `check_w6.py` method: top-20 delta
  ≤ 1.5 × floor). Bit-identity is expected; anything above the floor stops the candidate.
- **TTFT gain in a window.** A later OFF HOLD (the window-1 design: `off-327`, `off-524`, `off-524-t0`, 150K and 300K
  cold ×2, same prompts/salts) with the existing gate: success = `off-524-t0` vs `off-327` gap < 5 % at both lengths
  with spread ≤ 3 % (the NOT_REPRODUCED criterion, now applied to the candidate), and decode ms/round unchanged.
- **Memory.** +1,014 MiB held per rank (K1 README). S18 R5 lowest MemAvailable was 5.64 / 6.40 GiB and window 2's
  524,288 prefill 6.00 / 6.77 GiB (sparkhead / sparkworker, MEASURED; `offload/prefill_maxlen/PF1-WINDOW-RECONCILE.md` §5,
  WINDOW.md §2) → about 4.6 / 5.4 GiB and 5.0 / 5.8 GiB (ESTIMATED): equal to S18's lowest reading overall (4.60 GiB,
  during a model load) and about 0.6 GiB above the 4 GiB window stop line on sparkhead under R5's 500K prompts. A1 therefore needs the K1-PROMOTE memory rows re-run at 524,288 before any
  promotion; A2 exists for the case it fails. K1's promotion package and gates are not changed by this lane.

## 4. Why no patch is offered now

Every source-visible max_model_len dependence sits in phases the traces show unchanged (CAUSE.md §4). The only
cap → MoE link is memory placement, which no CPU test can reproduce and which a patch could only move, not remove,
without first knowing which buffer is slow. The env-only A1 tests the most likely buffer without code.
