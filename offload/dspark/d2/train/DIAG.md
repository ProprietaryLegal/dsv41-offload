# D2-DIAG: why lower mean KL gave lower acceptance (pilot d2pass2-r1)

Lane D2-DIAG, 2026-10-25, Controller CPU only. Brief: `documents/plans/lanes/D2-DIAG.md`. No GPU, no training, no
evaluation and no run of the diagnostic tool by this lane. Every number is labelled MEASURED (with its source) or
ESTIMATED. Paths are relative to `offload/dspark/d2/` unless stated. Line numbers are at the brief commit d2a179a.

The pilot result being diagnosed (MEASURED, `PILOT-d2pass2-r1/evaluation.json`, sample of 1,000 fixed-K2 rounds per
split): E[A] baseline dev 1.505 / heldout_legal 1.855; REF 20K 1.527 / 1.796; mean per-position KL fell about 13 %,
median per-position KL rose 20–60 % (legal position 0: 0.0894 → 0.1398 at REF 20K), already at 20 updates.

## Audit

### (a) The training target

**Where the teacher numbers come from.** The pilot's shards are the second pass d2pass2-r1 (decode capture, option ii):
the top-20 rows were recorded by the generation request itself, not by a later replay.

- The generator's request body is `temperature 1.0, top_p 1.0`, plus `logprobs 20, return_tokens_as_token_ids` for
  option ii (`generate.py:163-168`; settings note `generate.py:9`). No `top_k`, `min_p`, penalties, stop ids or seed are
  sent. MEASURED from the frozen generator state (`/opt/dsv41/data/d2/gen-decode-07fc7d6-d2pass2-r1/seqs/*.json`,
  `segments[].sampling`, read only): every segment records exactly `max_tokens 16384, temperature 1.0, top_p 1.0,
  priority 20, logprobs 20, return_tokens_as_token_ids true`.
- Server defaults: S36 (`S36.report.md` "generation_config.json is ABSENT on both ranks… config.json has no generation
  or sampling key… neutral defaults top_k 0, min_p 0.0") and S32 (`S32.report.md` "Raw vs processed mode": default
  `raw_logprobs`, no `--logprobs-mode` override in the launch path). The exporter binds `logprobs_mode: raw_logprobs`
  into VERIFIED.json and refuses anything else (`generation_export.py:117`, `:132`).
- So the teacher row is log_softmax of the target's raw logits, and with T = 1 and no processors the raw distribution IS
  the distribution the committed token was sampled from and the one the rejection sampler verifies against.
- **MEASURED calibration (this lane, CPU, aggregates only, all 974,303 generated positions of the 131 sequences in the
  generator state):** committed token == teacher argmax in 0.82959 of positions; the mean teacher top-1 probability is
  0.82969. The reliability curve (mean top-1 probability vs hit rate, by top-1-probability decile) agrees within 0.004
  in every decile with ≥ 20,000 rows (e.g. 0.3: 0.3537 vs 0.3543; 0.5: 0.5504 vs 0.5497; 0.9: 0.9890 vs 0.9890) and
  within 1.3 binomial SE in the 0.1 decile (3,559 rows: 0.173 vs 0.181), and per split (train 0.8282/0.8280,
  dev 0.8195/0.8198, heldout_legal 0.8417/0.8414). A temperature or processing mismatch between the logged
  distribution and the sampling law would show here as a systematic gap; there is none. Also MEASURED: mean top-20
  coverage 0.99809, committed token outside the top-20 in 0.193 % of positions, mean teacher bucket entropy 0.493 nats
  (top-20 + one tail bucket, a lower bound on the full entropy). 58.9 % of positions have teacher top-1 ≥ 0.9.
- Row parsing (`generation_top20.py:51-74`): the 20 most likely ids sorted by (−logprob, id), the sampled token as a
  21st entry when outside the top-20 (`tok_rank` 21 then), `resid = max(0, 1 − Σ exp(top_lp))` (`:74`). The decode
  dataset copies `top_ids, top_lp, tok_lp, tok_rank, resid` unchanged (`decode_dataset.py:293`, `:313`).

**How a block becomes a target** (`train/data.py:225-262`, DATASET.md §4). For round P: window taps rows
max(0, P−127)..P; anchor `tokens[P+1]`; draft i (0..4) targets position `t_i = P+2+i`: teacher row of `t_i`
(`top_ids`, `top_logp`), `tail_mass = resid` of that row, label `tokens[t_i]` (the committed sample), teacher-forced
Markov input `previous[i] = tokens[t_i − 1]` (`previous[0]` = the anchor). Positions past the sequence end are masked
(`mask = target_pos < L`, `:243`); `present = position_mask = mask` (`:261-262`), so on D2 shards every unmasked
position has a top-20 row and the 0.1 × sampled-CE fallback (`loss.py:131-141`; FAST `fast_loss.py:64`) never fires.
The sampled label is NOT used by the distillation loss when the top-20 is present.

**The loss** (`train/loss.py:90-148`, REF; FAST `fast_loss.py` computes the same quantity with an explicit backward).
For drafter logits z over the whole vocabulary: log q_i = z_i − logZ for the 20 teacher ids; the tail bucket
log q_T = logZ(complement of the 20 ids) − logZ, from a complementary logsumexp, never 1 − Σq (`:48-87`, `:126-129`).
Per position the divergence is the coarse forward KL

    KL_bucket(p || q) = Σ_{i∈top20} p_i (log p_i − log q_i) + p_T (log p_T − log q_T)        (loss.py:39-43)

(no renormalisation of either side; within-tail allocation is unobserved, so it is a lower bound on the full-vocabulary
KL). Positions are weighted (1, .8, .65, .5, .4) (`loss.py:12`) and the batch loss is Σ w·KL / Σ w over unmasked
positions of the batch (`:142-145`). It is a mass-covering objective: it charges p_i log(p_i/q_i) wherever the teacher
has mass, and nothing for drafter mass placed where the teacher has none except through normalisation.

**What is trained** (`train/model.py`, `loop.py`). Trainable parameters: rank-32 LoRA (A kaiming, B zero,
`ops.py:157-159` in class Dense) on `attn.wq_a, wq_b, wkv, wo_b`, the three shared-expert matrices and the eight `wo_a`
groups of each of the 3 stages (`model.py:105-111`); the RMS-norm weights `attn_norm, ffn_norm, q_norm, kv_norm` per
stage (`:112-113`), `main_norm` (`:263`) and the final `norm` (`:267-268`); the Markov head `markov_head.embed.weight`
and `markov_head.head.weight` (`:267-268`; shapes [129280, 256] each, so about 66.2 M parameters, ESTIMATED from Config
`vocab 129280, markov 256` and `evaluate.py:197`); `confidence_head.proj.weight` is registered trainable but frozen in
the distill stage (`loop.py:122-123`). Frozen: `main_proj`, vocabulary head, embeddings, routers, experts, attention
sinks and hc mixes. Optimizer (`loop.py:124-135`, `:198`; pilot run `pilot.py:98-99`): AdamW lr 2e-4, weight decay
0.01 on every trainable tensor (norms and Markov head included), default betas/eps, cosine decay from the first update
to 0 over 159 updates with **no warm-up**, global grad-norm clip 1.0, batch 128 blocks (one block = 5 positions), one
epoch over the first 20,000 shuffled train blocks (`pilot.py:107`). Under Adam the first updates move every coordinate
with a non-negligible gradient by about lr regardless of its gradient size; the Markov head's 129,280 output rows all
receive softmax gradient at every step.

**Population.** Training blocks are every round P of every train sequence (`data.py:225-233`), shuffled; each
position counts with weight w_i whether or not the drafter's earlier positions would have been accepted.

### (b) How pilot_eval computes E[A]

`train/pilot_eval.py:132-191` (`score`), reference math only (`:135-136`).

- **Population:** the amendment-4 sample (seed 20261024, ≤ 1,000 rounds per split) of complete rounds on the fixed K2
  trajectory: `complete_rounds` walks each sequence from P = gen_start − 1, takes the K2 drafter's 5 drafts at P
  (`g1.py k2` output, the production native drafter replica), counts its accepted prefix against the committed tokens
  and jumps to P + accepted + 1; only rounds with all 5 targets present (P + 6 < L) are eligible
  (`pilot_sample.py:89-105`). Candidate drafts never choose which rounds are scored (`evaluate.py:160-162` note).
- **Drafts:** `model.greedy(h, hx, anchor, chunk=8192)` (`pilot_eval.py:162-163`; `model.py:345-361`): the block hidden
  states come once from the window taps plus [anchor, NOISE × 4] (`model.py:304-326`), independent of any draft; then
  for positions 0..4 the argmax over the full vocabulary of base logits + Markov(prev), with prev = the anchor at
  position 0 and the drafter's OWN previous pick afterwards; ties keep the lower id. **Fixed depth 5**, no confidence
  or adaptive depth.
- **Scoring:** `match_length(picks, labels)` = number of leading drafts equal to the committed tokens `tokens[P+2..P+6]`
  (`evaluate.py:35-41`). E[A] = mean over rounds; bonus/correction token excluded. Whose tokens: the committed T = 1
  samples of the target, i.e. one sample per position from exactly the teacher distribution (calibration above).
- **KL reported alongside:** `bucket_kl` in float64 (`evaluate.py:104-109`) on logits computed with the TEACHER-FORCED
  previous token (`pilot_eval.py:167-169`) at all 5 positions of every round; `kl_by_position` = mean,
  `kl_median_by_position` = median over rounds (`:188-190`); `loss` = the training loss on the round (`:170`).

### (c) Train/eval mismatches

1. **Objective vs metric (the main one).** The loss is forward bucket-KL to the full teacher distribution; E[A] scores
   only the drafter's argmax, against one sample of that distribution. For a deterministic draft d at position i the
   acceptance probability is p_teacher(d | committed prefix) (calibration above), so the only thing E[A] rewards is
   putting the argmax on a high-teacher-probability token. KL can fall by matching the shape (entropy, secondary
   tokens, tail) while the argmax moves among near-ties or does not move at all. Mean KL is dominated by rounds where
   the drafter puts little mass on the teacher's mode (heavy tail: baseline legal position 0 mean 0.764 vs median 0.089,
   MEASURED); those rounds pull the gradient.
2. **Conditioning (Markov input): no mismatch on scored positions.** Training and the KL use previous = committed
   token; greedy uses the drafter's own pick. match_length stops at the first mismatch, so every position that adds to
   E[A] has own picks equal to the committed tokens before it, i.e. identical conditioning. The block hidden states
   use NOISE ids at positions 1..4 in both paths. The tool measures this identity explicitly (greedy pick vs
   teacher-forced argmax on prefix-matched positions must agree 100 %).
3. **Population weighting.** Position i ≥ 1 contributes to E[A] only on rounds whose positions 0..i−1 were accepted;
   the loss weights every round's position i by w_i regardless (and the eval KL is an unconditional mean too). So
   most of the KL gradient at positions ≥ 1 comes from rounds that E[A] never scores at that position.
4. **Round population.** Training uses every P; E[A] uses the K2-trajectory rounds (P just after a K2 round ended).
   Same sequences' distribution in kind, but not the same weighting (minor; not separately tested).
5. **Splits.** heldout_legal's task families (chronology, motion_outline) never occur in train; dev shares train's
   families on disjoint sources (DATASET.md §5). The regression is on heldout_legal; dev is flat/noisy.
6. **No hidden mismatch in the teacher:** temperature/processing match (calibration), the label is the T = 1 sample,
   top-20 coverage 0.998 (MEASURED above).

**MEASURED, in-distribution online check (from `evaluation.json` `step_agreement`, information only):** the pilot
logs, before each update, the greedy accepted prefix on that step's fresh training batch (128 never-seen train
blocks, all P, the model after s−1 updates). REF mean accepted prefix: steps 1–20 1.927, 21–40 1.860, 41–80 1.892,
81–120 1.889, 121–159 1.939 (FAST-B, independent order: 1.905, 1.878, 1.890, 1.919, 1.916); the training loss on the
same batches fell 1.679 → 1.557 (REF). Batch-to-batch SD is 0.152, so the step-1 single batch (1.703) is not a usable
baseline; but over 159 updates the online loss falls about 7 % with no visible trend in acceptance even on the
training distribution (REF slope +0.026 per 100 steps). So the missing gain is not only a train → held-out domain
shift.

## Hypotheses (pre-registered 2026-10-25 before the tool exists and before any GPU run)

Written and committed by D2-DIAG before `pilot_diag.py` was written. The coordinator reads the tool's output against
the signatures below exactly as written; no threshold is changed after the run.

### Variants (all on the pilot's frozen sample, reference path, same model load as pilot_eval)

| id | weights | shares a receipt value |
|---|---|---|
| V0 | unadapted 1017 drafter (zero-init LoRA B) | baseline |
| R2.5 | REF checkpoint 2,500 blocks (20 updates) | REF 2500 |
| R20 | REF checkpoint 20,000 blocks (159 updates) | REF 20000 |
| F20 | FAST checkpoint 20,000 blocks | FAST 20000 |
| L20 | R20 with the Markov head (`markov_head.embed.weight`, `markov_head.head.weight`) reset to its loaded initial value: "LoRA-only" = trained LoRA + trained norms | none |
| H20 | the initial state with ONLY the two Markov-head tensors taken from R20: "head-only" (LoRA B = 0, norms initial) | none |

The confidence projection is identical in all variants (frozen in the distill stage) and affects neither drafts nor KL.
L20 and H20 partition the trained tensors; any tensor name outside {LoRA a/b, norm weights, Markov head, confidence
projection} makes the tool fail.

### Measurements (per split dev / heldout_legal, per draft position i = 0..4, per variant)

Notation: g_i = greedy pick (pilot_eval's own-draft path), d_i = argmax of the TEACHER-FORCED logits (pilot_eval's KL
logits), y_i = committed token, p_i = teacher top-20 + tail at position i, q_i = drafter softmax.

- **EA**: E[A] exactly as pilot_eval (must equal the receipt for V0, R2.5, R20, F20; built-in check, fails loudly).
- **EA_RB**: Rao-Blackwellised E[A] = mean over rounds of Σ_i [Π_{j<i} 1(g_j = y_j)] · p_i(g_i). Same expectation as EA
  (y_i is a T = 1 sample of p_i given the committed prefix, which equals the drafted prefix whenever the product is 1),
  without the 0/1 sampling noise of y on the scored position. Lower bound with p_i(g) = 0 when g is outside the
  top-20, upper bound with min(p_T, p_20); the outside fraction is reported.
- **reach_i**, **acc_given_reach_i**: fraction of rounds whose positions 0..i−1 were accepted; acceptance at i among them.
- **m_y_i** = rate(d_i = y_i); **m_t_i** = rate(d_i = teacher argmax); **pp_i** = mean p_i(d_i) (teacher probability of
  the drafter's teacher-forced pick; lower bound as above).
- **q1_i** drafter top-1 probability, **H_i** drafter entropy (full vocabulary), **gap_i** top-1 − top-2 logit margin
  (means and medians).
- Teacher (variant independent, recomputed every pass as a consistency check): top-1 probability, bucket entropy,
  top-20 coverage, calibration rate(teacher argmax = y_i).
- **KL_i** = bucket_kl (bit-identical code path to pilot_eval), split into **KLtop_i** (top-20 terms) and **KLtail_i**
  (tail term); mean and median.
- **Against V0** (paired, same rounds): Δ of EA, EA_RB, m_y, pp, q1, H, KL (mean) with a cluster-robust standard
  error (clusters = the source families pilot_eval records, `data.request_families`); fractions of rounds whose KL_i
  improved / worsened, whose 1(d_i = y_i) flipped 0→1 / 1→0, whose accepted length rose / fell; for rounds whose d_i
  differs from V0's: count, mean Δ p_i(pick) (new − old) and V0's median gap on them vs all rounds; mean ΔKL_i on rounds
  where V0's d_i equals the teacher argmax ("right") and where it does not ("wrong").
- **Temperature scan:** mean and median KL_i of logits/τ for τ ∈ {0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0}; τ* = grid
  argmin of the mean.
- **Identity check:** on every position with Π_{j<i} 1(g_j = y_j) = 1, g_i = d_i (must hold 100 %; any miss is a bug and
  fails the run).

**Resolved:** a Δ is resolved when |Δ| ≥ 2 × its cluster-robust SE. **Primary split:** heldout_legal (where the
pilot regressed); dev is read the same way and reported, and a hypothesis that holds on legal but contradicts on dev
is reported as split-specific. "Pooled" = positions 0–4 averaged with equal weight.

### H0 — the regression is sampling noise in the committed tokens (precondition)
- **Confirmed** if on heldout_legal Δ EA_RB (R20 − V0, lower bound) is NOT resolved negative and |Δ EA_RB| < ½ |Δ EA|.
- **Ruled out** if Δ EA_RB (lower and upper bound) is resolved negative.
- If confirmed, the diagnosis becomes "no gain" rather than "loss"; H1–H7 are then read on pp/m_t and KL medians.

### H1 — mass-covering objective (forward KL + tail spreads mass)
- **Signature:** R20 vs V0 on legal: Δ H (pooled) resolved > 0 AND Δ q1 (pooled) resolved < 0 AND Δ pp_0 < 0
  (resolved, or Δ EA_RB resolved < 0) while Δ KL (mean, pooled) < 0; and mean ΔKL_0 > 0 on "right" rounds with
  mean ΔKL_0 < 0 on "wrong" rounds.
- **Confirmed** if all hold. **Ruled out** if Δ H ≤ 0 or Δ q1 ≥ 0 (pooled, point estimates): the drafter did not
  become flatter.
- Tail share: if the pooled median KLtail rises by more than the pooled median KLtop, the tail bucket is named as the
  carrier (reported, not a separate verdict).

### H2 — teacher flatter/sharper than the target's own sampling distribution
- **Signature:** on the sample, |calibration rate − mean teacher top-1 probability| resolved (> 2 SE) at any position
  (sign: top-1 prob above the hit rate = teacher sharper than the sampling law; below = flatter).
- **Ruled out** if within 2 SE at all positions on both splits. The audit already measured agreement within 0.0001 on
  all 974,303 generated positions and established raw logprobs at T = 1 with no processors (§Audit (a)); the tool
  re-checks it on the exact sample.

### H3 — one trainable component causes it (LoRA trunk vs Markov head)
Let D(v) = Δ EA_RB(v − V0) on legal (lower bound); if H0 is confirmed use D(v) = Δ pp_0 instead.
- **Head confirmed** if D(H20) ≤ ⅔·D(R20) (reproduces ≥ ⅔ of the damage) and D(L20) ≥ ⅓·D(R20) (≤ ⅓ of it).
- **Trunk confirmed** symmetrically (L20 reproduces ≥ ⅔, H20 ≤ ⅓).
- **Ruled out (no single component)** if both reproduce ≥ ⅓, or neither reproduces ≥ ⅓ (an interaction).
- Reported with each component's Δ H, Δ q1, Δ KL mean/median, so H1's mechanism can be located in one component.
- Only read when D(R20) < 0 is resolved; otherwise UNRESOLVED.

### H4 — conditioning / scored-population mismatch (damage only at positions ≥ 1)
By construction (§Audit (c)2) the Markov conditioning is identical on every scored position; the identity check proves
it on the real sample. What remains is the population mismatch of §Audit (c)3.
- **Signature:** Δ m_y_0 and Δ pp_0 (R20 − V0, legal) not resolved negative, while Δ acc_given_reach_i for some i ≥ 1 is
  resolved negative, or Δ pp pooled over i ≥ 1 is resolved negative.
- **Ruled out** if Δ pp_0 is resolved negative and |Δ pp_0| ≥ |mean Δ pp_{1..4}| (position 0 is hurt at least as much).
- Identity-check failure is a tool/code bug, reported as such, not as evidence for H4.

### H5 — learning-rate / early-update shock (damage at 20 updates)
Let D as in H3.
- **Signature:** D(R2.5) ≤ ⅔·D(R20) (≥ ⅔ of the final damage at 20 updates) AND on rounds where R2.5's d_0 differs
  from V0's, the mean Δ p_0(pick) ≤ 0 (the early flips do not move the argmax toward the teacher), AND R2.5's Δ q1 /
  Δ H have the same sign as R20's.
- **Confirmed** if all hold. **Ruled out** if D(R2.5) ≥ ⅓·D(R20) (the damage builds over training) or the R2.5 flips
  have resolved positive Δ p_0(pick).
- Only read when D(R20) < 0 is resolved.

### H6 — the KL gain is calibration (temperature), which greedy acceptance ignores (added by D2-DIAG)
- **Signature:** τ*(V0) ≠ 1.0 on the grid at position 0 and pooled, and V0's mean KL at its τ* recovers ≥ 50 % of
  R20's mean-KL reduction (pooled), while τ*(R20) is closer to 1.0 than τ*(V0).
- **Confirmed** if all hold: the measured KL improvement is mostly a global sharpness change, invisible to argmax
  acceptance, so mean KL is not a usable progress signal for this objective. **Ruled out** if τ*(V0) = 1.0 or V0 at τ*
  recovers < 25 %.

### H7 — held-out family shift (legal task families never in train) (added by D2-DIAG)
- **Signature:** Δ EA_RB (R20 − V0) on dev ≥ 0 while on legal it is resolved negative, and the dev − legal difference of
  Δ EA_RB exceeds 2 × the root-sum-square of the two SEs.
- **Confirmed** if it holds (the adapter helps train-like families and hurts unseen ones). **Ruled out** if dev's
  Δ EA_RB is also resolved negative, or dev's Δ EA_RB ≤ 0 and the difference is not resolved (no gain anywhere; the
  online train-batch check in §Audit (c) already shows no in-distribution trend).

### Clarifications (written after an independent review of the tool, before any run)
- H1: "Δ pp_0 < 0 (resolved, or Δ EA_RB resolved < 0)" is read literally: the point estimate Δ pp_0 must be < 0, and
  its resolution may come from Δ pp_0 itself or from a resolved negative Δ EA_RB.
- H6: "Ruled out if τ*(V0) = 1.0" means at position 0 or pooled (either one fails the signature definitively, τ* being
  a deterministic grid choice). "Closer to 1.0" stays the absolute distance on the grid as written.
- These narrow ambiguous wording only; no threshold changes.

### Reading
Hypotheses are not exclusive (e.g. H1 + H5: a mass-covering shift that happens within the first updates). Each is
reported CONFIRMED, RULED OUT or UNRESOLVED per split with the numbers that decided it; FAST F20 is reported next to
R20 only to confirm FAST ≈ REF (no verdict depends on it). A second pilot is justified only by a CONFIRMED cause with
a matching recipe in §Second-pilot options, and only on NDM's word.

## Tool (`pilot_diag.py`)

`python -m offload.dspark.d2.train.pilot_diag --plan --receipt --evaluation --root --manifest --ckpt --k2-dev
--k2-legal --split {dev,heldout_legal} --out` (GPU lease required, same as pilot_eval); `--merge-parts DEV LEGAL --out`
joins the two parts on CPU and adds the cross-split readings (H2 on both splits, H7).

What it does, in order, and where it stops:
1. **Binding (before any model load).** The receipt's `plan_sha256` and `evaluation_sha256` must equal the files;
   the receipt embeds the same evaluation and plan; the evaluation is pilot_eval's (`EVALUATOR`), bound to the plan and
   the frozen sample. Every module hash the pilot plan recorded must be byte-unchanged; the only permitted new source
   file is `d2/train/pilot_diag.py` (the plan's `source_sha256` predates it, so the tool validates the plan with
   `validate_spec` against the current sources only after that check). The tool therefore runs on the pilot's source
   commit b6f79e9 plus `pilot_diag.py` (see §Run); on today's branch tip it refuses, because `loop.py` changed and
   `full_*.py` were added after the pilot. MEASURED on Controller (CPU): `git archive b6f79e9` + `pilot_diag.py` gives
   0 changed and exactly 1 new module against the real plan's `source_sha256`, and the tool's tests plus
   `test_pilot_eval.py` pass on that tree (34 passed). Then, as pilot_eval: `verify_sample_draw`
   (both splits re-drawn from seed 20261024 and compared), `verify_units`, `read_arm` for REF and FAST, and the arm
   checkpoint records must equal the receipt's evaluated checkpoints.
2. **Model.** `torch.manual_seed(23)`, `load(ckpt, "cuda", rank=32, reference_math=True)`, frozen-identity and
   initialization hash checks, `apply_trainables` per variant: exactly pilot_eval's path. L20/H20 are built from the
   loaded initial tensors and the REF 20K checkpoint by tensor group; an unclassified tensor name stops the run.
3. **Passes** in the order V0, R2.5, R20, F20, L20, H20 over the split's frozen keys. Each round runs pilot_eval's
   greedy + `match_length` and its teacher-forced logits + `bucket_kl` unchanged (the loss term is not recomputed); the
   extra statistics are derived from those same logits on CPU in float64.
4. **Sanity check (fails loudly).** After each of V0, R2.5, R20, F20: E[A], rounds, requests, the population hash and
   every per-request accepted sum must EQUAL the receipt's values (exact float equality); KL mean/median are recorded
   (`kl_bit_identical`) and a difference > 1e-6 also stops the run. A mismatch raises before the next pass, so a wrong
   checkpoint, sample or code path cannot produce a diagnosis. Teacher statistics must be identical in every pass.
5. **Identity check (conditioning).** On every prefix-matched position the greedy pick must equal the teacher-forced
   argmax. *Clarification written with the tool, before any run:* the greedy path computes each position's logits in
   its own GEMM while the KL path batches five positions, so an exact near-tie can differ by float rounding. A miss
   whose teacher-forced logit gap is ≤ 1e-3 is counted as an arithmetic near-tie (`identity.near_ties`, reported);
   any larger miss stops the run as a conditioning bug. This does not change any hypothesis threshold.
6. **Output** (mode 600, exclusive create): identities (plan/receipt/evaluation/sample/K2/manifest hashes, checkpoint
   paths and hashes, source commit and tool hash), per-variant aggregates (means, medians, per-position lists,
   temperature scan, identity counts, the same per-request accepted totals pilot_eval writes), paired V0 comparisons
   with cluster-robust SEs (clusters = `data.request_families`), and `readings`: the §Hypotheses rules applied
   mechanically, for the coordinator to confirm. No row, token id or text is written; picks stay in memory.

ESTIMATED extra cost per round over pilot_eval's scoring: 0.042 s (MEASURED on Controller, 8 threads, synthetic
[5, 129,280] logits: `round_stats` incl. the 7-temperature scan; x86host's CPU may differ), about +9 % on the pilot's
MEASURED 0.484 s per round per pass; the loss term pilot_eval computed is not repeated.

CPU tests: `tests/test_pilot_diag.py` (dense-oracle check of every per-round metric, KL parts, temperature scan and
Rao-Blackwell value; near-tie vs conditioning-miss; exact reproduction of `pilot_eval.score` on the tiny synthetic
drafter; all six variants and ablation composition; the receipt check stopping at V0 or R20; request totals and KL
in the check; tensor-group coverage; plan/receipt/source binding incl. the one permitted new module; refusal before
model load; every reading branch; merge H2/H7 and part identity).

## Run (coordinator only, locally on x86host; NOT run by D2-DIAG)

Preconditions: both solo cards idle (no `d2pilot-*` session, no flock holder on
`~/dsv41-offload/locks/gpu-{240876ad,bc62df2a}.lock`); `tmux has-session -t =dsv41-powerlog` (the lease refuses without
it); `free -g` has room for the same two jobs the pilot's evaluation ran in parallel (same model load, plus three
checkpoint tensor sets of the trainable parameters), no swap (the lease stops a child above 128 GiB RSS); production pair
untouched. The commit to run is the reviewed one containing `pilot_diag.py`, transferred by the coordinator's existing
review/transfer process into a clean clone (this lane does not touch x86host).

```bash
set -euo pipefail
source ~/.config/ndm/paths.sh
umask 077
DIAG_COMMIT=REVIEWED_FULL_40_HEX_COMMIT                      # contains offload/dspark/d2/train/pilot_diag.py
SRC=$HOME/dsv41-offload/d2-diag-src-${DIAG_COMMIT:0:7}       # clean clone at DIAG_COMMIT (existing transfer)
SNAP=$HOME/dsv41-offload/d2/diag/d2pass2-r1                  # NEW snapshot (never the pilot's)
OUT=/opt/dsv41/x86host-data/d2/diag/d2pass2-r1               # NEW output directory (never the pilot's)
PILOT=/opt/dsv41/x86host-data/d2/pilot/decode-d2pass2-r1     # pilot output: READ ONLY
G1=/opt/dsv41/x86host-data/d2/g1-d2pass2-r1                  # K2 drafts: READ ONLY
MANIFEST=/opt/dsv41/x86host-data/d2/decode-d2pass2-r1-c2/shards/MANIFEST.json
CKPT=/opt/dsv41/x86host-data/native-1017
PY=$HOME/dsv41-offload/e/venv/bin/python                     # the pilot's interpreter (requirements-x86host.lock)
PILOT_COMMIT=b6f79e9de44ceb581820b8df5c3d779ff79cb26b         # the pilot's SOURCE_COMMIT (LANES 2026-10-25 13:21)
[ "$(cat ~/dsv41-offload/d2/pilot/decode-d2pass2-r1/SOURCE_COMMIT)" = "$PILOT_COMMIT" ] || { echo "pilot source differs"; exit 1; }
[ "$(git -C "$SRC" rev-parse HEAD)" = "$DIAG_COMMIT" ] || { echo "clone not at DIAG_COMMIT"; exit 1; }
[ -z "$(git -C "$SRC" status --porcelain)" ] || { echo "clone not clean"; exit 1; }
git -C "$SRC" merge-base --is-ancestor "$PILOT_COMMIT" "$DIAG_COMMIT" || { echo "pilot commit not in history"; exit 1; }
[ ! -e "$SNAP" ] || { echo "$SNAP exists"; exit 1; }
[ ! -e "$OUT" ] || { echo "$OUT exists"; exit 1; }
# Inputs must be the pilot's (values: receipt/evaluation/plan copies in PILOT-d2pass2-r1/, LANES 2026-10-25 13:15).
# Cross-check the four input paths against the pilot's own eval script: grep -o -- '--[a-z0-9-]* [^ ]*' \
#   ~/dsv41-offload/d2/pilot/decode-d2pass2-r1/eval-dev.sh
sha256sum -c - <<SUMS
8637ba67e85c7a3b87ac334ebd43e6fa6a0eabffeafe57f26499f64c7fcf6340  $PILOT/receipt.json
2d5ba29d3f3c20dc2035c85a2dd5ca00d8c7603ad817dfcd7fba8ec2b82f132d  $PILOT/evaluation.json
c4732bc224944af7df7de1d1fe67dbf6efd2d6bcd9d49bdc94af584085074402  $PILOT/plan.json
450bb71653f3def68cd95ee9579a456c74c469d08a016940cceda24de48b4f30  $MANIFEST
6777ccb8f51f404874766ce6a39300ce1a05f22ed80685fb24ca33ab18489d42  $G1/dev.k2_drafts.jsonl
89ad1170865c3ea554970b1af11b917f5449b3401f4a622237d6c1475a80e58a  $G1/heldout_legal.k2_drafts.jsonl
SUMS
mkdir -p -m 700 "$SNAP/repo" "$SNAP/logs" "$OUT"
# The pilot's exact source tree plus ONLY the diagnostic module: the tool refuses any other changed or added module
# (since b6f79e9, loop.py changed and full_*.py were added; a plain DIAG_COMMIT archive would be refused by design).
git -C "$SRC" archive "$PILOT_COMMIT" | tar -x -C "$SNAP/repo"
git -C "$SRC" show "$DIAG_COMMIT:offload/dspark/d2/train/pilot_diag.py" > "$SNAP/repo/offload/dspark/d2/train/pilot_diag.py"
printf 'pilot %s\ndiag %s (offload/dspark/d2/train/pilot_diag.py only)\n' "$PILOT_COMMIT" "$DIAG_COMMIT" > "$SNAP/SOURCE_COMMIT"
ARGS="--plan $PILOT/plan.json --receipt $PILOT/receipt.json --evaluation $PILOT/evaluation.json --root $PILOT/arms"
ARGS="$ARGS --manifest $MANIFEST --ckpt $CKPT --k2-dev $G1/dev.k2_drafts.jsonl --k2-legal $G1/heldout_legal.k2_drafts.jsonl"
for job in "dev GPU-240876ad-ea42-47de-1b8d-f6153810b7e1 dev" "legal GPU-bc62df2a-59cf-9828-7d43-558d1355bab7 heldout_legal"; do
  set -- $job
  cat > "$SNAP/diag-$1.sh" <<SCRIPT
set -euo pipefail
source ~/.config/ndm/paths.sh
umask 077
export D2_TRAIN_ROOT=$SNAP D2_GIT_SHA=$PILOT_COMMIT+pilot_diag@$DIAG_COMMIT D2_PYTHON=$PY
export PYTHONPATH=$SNAP/repo:$SNAP/repo/offload/dspark OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
cd $SNAP/repo
D2_PILOT_GPU=$2 bash $SNAP/repo/offload/dspark/d2/train/run_pilot_locked.sh diag-$1 \
  -m offload.dspark.d2.train.pilot_diag $ARGS --split $3 --out $OUT/$1.part.json
SCRIPT
done
tmux new-session -d -s d2diag-d2pass2-r1-dev   "bash $SNAP/diag-dev.sh;   echo DIAG-DEV-EXIT=\$? >> $SNAP/logs/diag-dev.coordinator.log"
tmux new-session -d -s d2diag-d2pass2-r1-legal "bash $SNAP/diag-legal.sh; echo DIAG-LEGAL-EXIT=\$? >> $SNAP/logs/diag-legal.coordinator.log"
```

Progress: `$SNAP/logs/diag-{dev,legal}.log` (one JSON line per finished pass: variant, E[A], EA_RB, reproduced;
nvidia-smi lines from the lease every 5 s). A `DiagnosticError` (receipt mismatch, changed module, identity miss) ends
the job non-zero: report it as is; do not edit and rerun. When both exit 0, merge on CPU (no lease):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=$SNAP/repo:$SNAP/repo/offload/dspark "$PY" -m offload.dspark.d2.train.pilot_diag \
  --merge-parts "$OUT/dev.part.json" "$OUT/legal.part.json" --out "$OUT/diagnostic.json"
```

Outputs (aggregates and identities only, mode 600): `$OUT/dev.part.json`, `$OUT/legal.part.json`,
`$OUT/diagnostic.json`. They may be copied to a NEW campaign directory (e.g. `DIAG-d2pass2-r1/`) for reading. Read
`splits.<split>.readings` and `cross_split_readings` against §Hypotheses as written; the reading in the files is
mechanical and the coordinator confirms it.

**Wall time (ESTIMATED from MEASURED rates).** The pilot evaluation took 104.9 min per card for 13 passes × 1,000
rounds = 0.484 s per round per pass (MEASURED, includes its setup and the loss term). The diagnostic runs 6 passes ×
1,000 rounds per card: 6,000 × 0.484 s = 48.4 min, plus the extra statistics 6,000 × 0.042 s = 4.2 min (Controller
micro-benchmark), plus setup similar to the pilot's (sample re-draw, train-shard hashing, 8 checkpoint reads in
`read_arm`), already inside the 0.484 s average. ≈ 53–60 min per card; the two splits run in parallel on the two solo
cards, so ≈ 1 h wall, within the 1.5 h target. GPU use is the pilot evaluation's (reference math: CPU activations,
GPU vocabulary-head GEMMs), at the unchanged 225 W solo caps.

## Second-pilot options (written only; not run, none chosen)

The coordinator chooses after reading the diagnostic; NDM approves any second pilot (amendment 3). Common to every
option below:
- **Pass rule unchanged** (amendments 6/6a/6b): at 10,000 and 20,000 blocks on dev and dev_legal, REF and FAST each
  gain ≥ 0.05 E[A] over the zero-adapter baseline with the 90 % family-clustered lower bound > 0 (else INCONCLUSIVE if
  REF shows no useful gain); FAST − REF lower bound ≥ −0.03; FAST not below baseline by > 0.01; FAST median per-position
  KL not worse than REF's by > 0.01 nats. E[A] stays the 0/1 fixed-K2 metric; the diagnostic's EA_RB is information
  only.
- Any recipe change touches modules the pilot plan hashes, so it is a NEW pilot: new source snapshot, new plan
  (`prepare`), new output namespace. The amendment-4 sample is a pure function of the manifest, the K2 files and seed
  20261024, so `sample` redraws the identical key list; the baseline E[A] (1.505 / 1.855) is reproduced, not re-used.
- FAST must implement the same change (and pass the FAST-vs-REF non-inferiority), or the option is REF-only and the
  full run would then need REF throughput (≈ 44 s per update; MEASURED: REF 6,999 s for 159 updates).
- Code sizes are ESTIMATED lines of production code (tests extra), from the current module structure.

| hypothesis (if CONFIRMED) | recipe change it implies | code | how amendment 6 would judge it |
|---|---|---|---|
| H0 noise only | none: the KL recipe gives no measurable gain; a second pilot of the same recipe would re-measure ≈ 0 | 0 | Expect INCONCLUSIVE (REF useful gain < 0.05). Only a scale change (more data/updates: Phase 2b, needs NDM's word and its Spark cost) is a different experiment. |
| H1 mass covering | (a) existing bucket TV (`Run(divergence="tvd")`, already in loss.py and fast_loss.py; OBJ1 §2's single bounded arm); (b) hard CE on the teacher argmax (the acceptance-optimal pick, = the committed token's mode) or on the committed token, alone or as a small mixture; (c) reverse bucket KL | (a) ≈ 5 (pilot_run setting + plan identity); (b) ≈ 40–60 (REF term + FAST explicit backward); (c) ≈ 60–80 (both paths) | Same rule. KL is no longer the objective, so mean KL may rise vs baseline; the rule's KL clause compares FAST with REF only, so it stays judgeable. |
| H2 teacher temperature (ruled out by the audit unless the sample disagrees) | rescale the teacher (log p / τ, renormalised with the tail bucket) or recollect processed logprobs | ≈ 15 (data.py target transform) | Same rule; needs the τ source stated before data. |
| H3 one component | freeze the harmful component: e.g. trunk-only (Markov head frozen) or head-only (LoRA/norms frozen) via a component filter on `requires_grad` | ≈ 15–25 (loop.py gate + plan field + FAST parity) | Same rule; the frozen component's tensors must be bit-identical to the initial ones in every checkpoint (one extra check). |
| H4 scored-population mismatch | weight position i by the drafter's own prefix-survival under the teacher, w_i ∝ Π_{j<i} p_j(d_j) (detached), or position-0-only (1, 0, 0, 0, 0) as a bounded probe | ≈ 20–40 (both loss paths) | Same rule; E[A] already scores exactly that population. |
| H5 early-update shock | lower LR (e.g. 5e-5) with linear warm-up over ~10 % of updates; optionally no weight decay on norms/Markov head | ≈ 5–10 (Run field + scheduler lambda in loop.py) | Same rule; risk: 159 updates at a lower LR may not move E[A] by the required 0.05 (→ INCONCLUSIVE). |
| H6 calibration-only KL gain | as H1 (an objective whose optimum is the argmax, not the shape), and select checkpoints on dev E[A]/EA_RB instead of KL | as H1 | Same rule. |
| H7 held-out family shift | not fixable inside the pilot: training on chronology/motion_outline would consume the dev_legal split; options are broader train families (new prompts + generation, Spark time) or a smaller/regularised update | data: new generation pass (NDM approval); code ≈ 0 | dev_legal stays a judged split, so a recipe that only helps train-like families still FAILS on legal regression. |

Combinations (e.g. H1 + H5) map to one arm combining the two changes only if the diagnostic confirms both; OBJ1's rule
of changing one thing per arm otherwise stands.
