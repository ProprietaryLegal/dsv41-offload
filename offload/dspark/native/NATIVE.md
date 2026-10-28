Status 2026-10-23: Native representation/semantics record; later N4 official arithmetic and S20 assessment govern readiness. Old timing/acceptance tables are historical evidence. See [live roadmap](../../../documents/plans/roadmap.md) and [measured record](../../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# Native 1017 DSpark draft on the NVLink pair (N1-NATIVE)

**In short.** The official DeepSeek-V4.1-Flash-1017 draft (`mtp.0-2`, 2,401 tensors) is stored as MXFP4 routed experts,
FP8 E4M3 dense matrices with E8M0 32x32 block scales, and BF16 / FP32 small tensors; it shares the target's BF16
embedding and BF16 head. This directory serves those official values on x86host's two RTX 3090s without any lossy
re-quantization: either packed (the official bytes, decoded in the GEMM) or expanded (decoded once to BF16, exact).
`NATIVE_MANIFEST.json` lists every tensor (official format, bytes, sha256, value checks, what the deployed worker
holds instead). Source: `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`,
shards 44-46 (draft), 2 (embed), 43 (head + final norm), sha256-verified (`/opt/dsv41/x86host-data/native-1017/
RECEIPT.txt` on x86host); official inference code alongside (`inference/`, SHA256SUMS).

## 1. Tensors: native vs decoded vs EXL3 vs shared

| tensor family (per stage s = 0..2 unless noted) | official | supply | deployed worker (TENSORS.json) | native arms |
|---|---|---|---|---|
| `embed.weight` [129,280 x 5,120] | BF16 | shared with target | official bytes (HeNryous copy; sha256 compared in the manifest) | unchanged |
| `head.weight` [129,280 x 5,120] | BF16 | shared with target | EXL3 K6 | K6 (control) / official BF16 per vocab shard (head arm) |
| `mtp.s.ffn.experts.{0..127}.w1/w3` [2,304 x 5,120] | MXFP4 (I8 [2,304 x 2,560] + E8M0 [2,304 x 160]) | draft-only | EXL3 K2 | packed MXFP4 / expanded BF16 (exact) |
| `mtp.s.ffn.experts.{0..127}.w2` [5,120 x 2,304] | MXFP4 (I8 [5,120 x 1,152] + E8M0 [5,120 x 72]) | draft-only | EXL3 K2 | packed / expanded |
| `mtp.s.ffn.shared_experts.w1/w3` [2,304 x 5,120], `w2` [5,120 x 2,304] | FP8 E4M3 + E8M0 32x32 | draft-only | EXL3 K5 | FP8 packed / BF16 expanded (exact) |
| `mtp.s.attn.wq_a` [1,280 x 5,120], `wkv` [512 x 5,120], `wq_b` [32,768 x 1,280], `wo_b` [5,120 x 8,192] | FP8 + E8M0 | draft-only | EXL3 K5 | FP8 / BF16 expanded |
| `mtp.s.attn.wo_a` [8,192 x 4,096] | FP8 + E8M0 (the official runtime dequantizes it to BF16: convert.py) | draft-only | official FP8 bytes, scales decoded to fp32 at load | same, or BF16 expanded (exact) |
| `mtp.0.main_proj` [5,120 x 15,360] | FP8 + E8M0 | draft-only | EXL3 K5 (fp32-accumulating build) | FP8 / BF16 expanded |
| `mtp.s.ffn.gate.weight` [128 x 5,120] | BF16 (used as fp32) | draft-only | official BF16 | unchanged |
| `mtp.s.ffn.gate.bias` [128] | FP32 | draft-only | official FP32 | unchanged, never through fp16 |
| `mtp.s.ffn.gate.bias_vl` [128] | FP32 | draft-only | not loaded (vision-only bias; drafts are text) | not loaded |
| norms (`attn_norm`, `ffn_norm`, `q_norm`, `kv_norm`, `mtp.0.main_norm`, `mtp.2.norm`) | BF16 | draft-only | official | unchanged |
| `mtp.s.hc_{attn,ffn}_{fn,base,scale}`, `mtp.s.attn.attn_sink` | FP32 | draft-only | official | unchanged |
| `mtp.2.markov_head.{embed,head}.weight` [129,280 x 256] | BF16 | draft-only | official (+ an 8,192-row bias cache) | official; cache / factorized / hot-row arms |
| `mtp.2.confidence_head.proj.weight` [1 x 5,376] | BF16 (used as fp32) | draft-only | official | unchanged |

"Shared" means the draft reads the target's own tensor (`Transformer.__init__`: `self.mtp[-1].embed = self.embed;
self.mtp[-1].head = self.head`; convert.py drops any `mtp.*.embed/head` copy). The Spark target's embed / head and
sampler never change; the draft keeps its own copy on the 3090s (it must compute logits locally). The per-tensor
truth (sha256 of the official bytes vs the worker's bytes) is in `NATIVE_MANIFEST.json` (`worker` field).

## 2. Official semantics (inference/model.py, kernel.py)

**Quantized linear** (`model.linear`): a weight of dtype float4_e2m1fn_x2 or float8_e4m3fn takes an FP8 activation:
`act_quant(x, 32, "ue8m0", e8m0)` per 32 consecutive inputs: `amax = max(max|x|, 1e-4)` (fp32),
`s = 2^ceil(log2(fp32(amax * fp32(1/448))))` by exponent bits, `q = e4m3_rne(clamp(x / s, ±448))`. The GEMM forms,
per 32-wide K block, the exact FP8 x FP8 (FP4 weights are cast to FP8 through fp32, losslessly) products summed in fp32,
multiplies the block sum by `s_a * s_w` and accumulates in fp32; the output is **bf16**. FP4 weight scales: one E8M0
per 32 inputs of each output row (`[out, in/32]`); FP8 weight scales: one per 32x32 block (`[ceil(out/32),
ceil(in/32)]`). Packed FP4: byte j of a row holds input 2j in the low nibble, 2j+1 in the high nibble; E2M1 table
`0, .5, 1, 1.5, 2, 3, 4, 6` with bit 3 the sign (0b1000 = -0). E8M0 byte b = 2^(b-127); 255 = NaN.

On SM86 (no FP8 tensor cores) the same arithmetic is reproduced exactly: a QDQ'd bf16 activation equals `q * s`
exactly (4 significant bits, power-of-two scale), a decoded weight `code * 2^(b-127)` is exact in bf16 (range checked
for every tensor in the manifest), and bf16 x bf16 products are exact in the fp32 accumulator. A native kernel
therefore differs from the official one only in fp32 summation order. **The activation QDQ is part of the model**
(trained with it): it must run before every FP8/FP4 linear, never be dropped as "lossy".

**Other QDQ points** (`act_quant(..., inplace=True)`, bf16 in place): the window KV after `kv_norm` + RoPE (both
the context rows `main_kv` and the block rows `kv`), and the attention output `o` per 32 before the `wo_a` einsum.

**`wo_a`**: dequantized to bf16 at load (convert.py); `o` (QDQ'd) `einsum("bsgd,grd->bsgr")` in bf16 (fp32 acc), per
group g of 8, then `wo_b` (FP8 linear).

**RMSNorm**: fp32 `x * rsqrt(mean(x^2) + 1e-20)`, times the bf16 weight (fp32 product), cast back to bf16.

**Gate** (text tokens): `scores = linear(x.float(), W.float())` (fp32, no TF32) / gate_temp (1.0) ->
`sqrt(softplus(scores))`; `ids = topk(scores + bias, 3)` (fp32 bias; it only selects); `w = scores[ids]`,
`w /= sum(w) + 1e-20`, `w *= 1.5`.

**Expert** (routed: FP4; shared: FP8): `gate = w1(x).float()`, `up = w3(x).float()` (each linear's bf16 output),
`up = clamp(up, -10, 10)`, `gate = clamp(gate, max=10)`, `h = silu(gate) * up` (fp32), routed: `h = w * h` (fp32),
`w2(h.to(bf16))` (QDQ of h per 32, bf16 output).

**MoE**: `y = 0` (fp32); for expert ids in ascending order `y[rows] += expert(x[rows], w)` (bf16 output added in
fp32); `y += shared(x)`; `out = y.to(bf16)`.

**Block / mHC**: `hc_mixes(stream, fn, scale, base)`: flatten the 4 copies (fp32), `F.linear(x, fn) * rsqrt(mean(x^2)
+ 1e-20)`, `hc_split_sinkhorn` (20 iterations, eps 1e-6) -> pre / post / comb; `hc_pre` collapses with the previous
sub-block's pre-mix (fp32 sum, bf16 out); `hc_post = post * x + sum(comb * residual)`. Attention uses the pre-mix
the previous FFN produced, the FFN the one this attention produced.

**DSpark stage**: stage 0 `main_x = main_norm(main_proj(cat(taps 37, 38, 39)))`; block input ids `[anchor, noise x 4]`
(noise 128799) embedded and replicated over the 4 copies; attention over the 128-row window of main_x KV plus the
block's own 5 rows (sink per head); after stage 2: `logits = head(norm(hc_pre(x)))` in fp32 (bf16 weight used as
fp32); position i: `logits[:, i] += markov.head(markov.embed(prev_id))` (rank 256), `prev = argmax` (temperature 0);
`confidence = proj(cat(hidden, markov_embed).float())`.

`reference.py` implements the linear / expert / gate / MoE semantics as oracles (`acc="f64"`: exact products summed
in float64; `acc="blocked"`: the official block order).

## 3. TP2 split of the native tensors (rank r = 0, 1)

The cuts are the deployed TP2 worker's (TP2.md), now on 32-element scale groups:

| tensor | rank r holds | scale groups |
|---|---|---|
| routed w1 / w3 | output rows [1152r, 1152r + 1152) | rows' own scales `[1152, 160]` |
| routed w2 | inputs [1152r, 1152r + 1152): packed bytes [576r, 576r + 576), scales [36r, 36r + 36) | 36 whole groups; h's QDQ blocks never straddle the cut |
| shared w1 / w3 | output rows [1152r, ...): FP8 scale rows [36r, 36r + 36) | whole 32x32 blocks |
| shared w2 | inputs [1152r, ...): scale cols [36r, ...) | whole blocks |
| wq_b | output rows (heads 32r..32r+31) [16384r, ...): scale rows [512r, ...) | whole blocks |
| wo_a | groups 4r..4r+3: rows [4096r, ...) | whole blocks |
| wo_b | inputs [4096r, ...): scale cols [128r, ...) | whole blocks; o's QDQ blocks never straddle |
| head (all arms) | tile-aligned vocabulary split: rank 0 [0, 64768), rank 1 [64768, 129280), via `vocab_split()` | — |
| everything else | replicated | — |

Where the official single-device model rounds a FULL sum to bf16 (each routed expert's w2 output, the shared w2,
wo_b), a rank holds only a partial sum. Two combine modes, both measured: **exact-official** — all-reduce the fp32
partials per (row, expert slot) and per shared expert, then round each to bf16 and add in ascending expert id as the
official MoE does; **fused** — each rank sums its slots in fp32, one fp32 all-reduce, one bf16 rounding (fewer
roundings than the official model; differs from it by at most the per-expert bf16 roundings it skips).

## 4. Kernel contract (KX-MXFP4, KX-GROUPED)

See the "From N1" sections of `KX-MXFP4.progress.md` / `KX-GROUPED.progress.md` (campaign folder) for the
lane-facing copy; this is the binding text.

Routed experts, one stage, one rank:
* inputs: `x` bf16 [T, 5120], **already act_qdq'd** (one QDQ per row, shared by all three experts and the shared
  expert; `formats.act_qdq` / N1's device op), `ids` int32 [T, 3] (0..127, device-resident, never read back),
  `rw` fp32 [T, 3]; T = 5 x requests (5, 10, 15, 20; up to 32 supported).
* packed weights (KX-MXFP4): `w13` uint8 [128, 2, 1152, 2560] (w1 rows then w3 rows of the rank), `s13` uint8
  [128, 2, 1152, 160]; `w2` uint8 [128, 5120, 576], `s2` uint8 [128, 5120, 36]. Low nibble = even input. E8M0 bias 127.
* expanded weights (KX-GROUPED): `w13` bf16 [128, 2, 1152, 5120], `w2` bf16 [128, 5120, 1152], values exactly
  `code * 2^(b-127)`.
* math per (t, j), e = ids[t, j]: `g = bf16(sum_k x·w1)`, `u = bf16(sum_k x·w3)` (fp32 accumulation, ROUNDED TO BF16:
  the official linear returns bf16); `u = clamp(u, -10, 10)`, `g = min(g, 10)`; `h = (g / (1 + expf(-g))) * u` (fp32,
  full-precision expf, as torch's silu); `h = rw[t, j] * h`; `h16 = bf16(h)`; `hq = act_qdq(h16)` per 32 over the
  rank's 1152 (36 blocks); `y = sum_k hq·w2` fp32 (the rank's partial over its 1152 inputs).
* outputs (preallocated, graph-safe): mode SLOT: fp32 [T, 3, 5120] per-slot partials (exact-official combine is done
  after the all-reduce by N1's epilogue); mode COMBINE: fp32 [T, 5120] = slot partials summed in fp32 in ascending
  expert-id order per row (+ optional fp32 `add` input, e.g. the shared expert's partial). No host sync, no
  allocation, no data-dependent launch dimensions.
* exactness test: against `reference.py` on the real tensors (`native-1017`), every element within fp32
  summation-order noise of `acc="f64"`; bf16 roundings of g / u bit-identical to the reference's for > 99.9 % of
  elements (the rest: fp32 order flipping a bf16 tie, listed).

Dense (KX-GROUPED "dense native"): `y = bf16(sum_k act_qdq(x)·W)` with W FP8 + 32x32 E8M0 (packed) or exact BF16
(expanded); shapes (rank-local, N x K): wq_a 1280 x 5120, wkv 512 x 5120, wq_b 16384 x 1280, wo_b 5120 x 4096 (fp32
partial out), shared w1/w3 1152 x 5120, shared w2 5120 x 1152 (fp32 partial out), main_proj 5120 x 15360 (insert
rows 8-128 only), head 64640 x 5120 bf16 weight with fp32 logits out (the official head is fp32 x fp32; bf16 x bf16 ->
fp32 is exact here because the normed hidden is bf16), wo_a 4 groups x (1024 x 4096) bf16 einsum.

## 5. Memory per card (TP2), computed from the real shapes; measured values replace these in N1-NATIVE.report.md

Weights per rank in MiB (rank 0; rank 1 holds 0.9 MiB less head). Shared by every arm, as lane E measured them (TP2.md):
embedding 1,262.5 (bf16, replicated), Markov embedding 63.1 + head shard 31.6 + bias cache 1,012.5, RoPE 80.5 at
327,680 positions (256.5 at 1,048,576), norms / mHC / router 18.8, windows 3.4: **2,472.4**.

| part | K2 control | native packed | native expanded | source of the native number |
|---|---|---|---|---|
| routed experts (3 stages x 128) | 1,633.8 (EXL3 K2) | 3,442.5 | 12,960.0 | per stage: w13 128x2x1152x2560 B + s13 128x2x1152x160 B + w2 128x5120x576 B + s2 128x5120x36 B = 1,147.5 MiB; expanded 128x3x1152x5120x2 B = 4,320 MiB |
| dense (wq_a, wkv repl.; wq_b, wo_a, wo_b, shared shards; main_proj repl.) | 218.5 (K5 + FP8 wo_a) | 321.1 (FP8 + fp32 scale values) | 639.8 (bf16) | 335,413,248 FP8 bytes per rank |
| head shard | 237.3 (K6) | 237.3 (K6) or 632.5 (bf16 arm) | same | 64,768 x 5,120 x 2 B |

Totals (weights + fixed state, before activations / graph pool ~210 MiB and the CUDA context ~410 MiB, E's measured
overheads): control 4,562; native-experts-packed (K5 dense) 6,371; native-packed 6,473; native-hybrid (packed
experts, bf16 dense) 6,792; **native-expanded 16,310 MiB (15.9 GiB)** -> ~17.0 GiB per card in nvidia-smi, ~7 GiB
under the 24 GiB card. BF16 head arm +395 MiB; the exact factorized Markov drops the 1,012.5 MiB bias cache. 1M-token
RoPE +176 MiB.

Load peak: the loader streams one matrix at a time (expanded: one decode, ~0.15 GB of temporaries; packed: one
6 MB copy), so the device peak is the serving footprint + ~0.15 GB plus whatever torch's caching allocator keeps
reserved from those temporaries; the host peak is lane E's (identity verification of the K2/K5/K6 delivery,
6.9 GB RSS) plus one matrix; the official shards are memory-mapped (page cache, not process memory).

## 6. FP16 or BF16 (decided from the real values)

* Weights: every decoded FP4 / FP8 value of the draft is exact in BOTH fp16 and bf16 (manifest: |w| 2^-21..6).
* Activations: the QDQ'd activation `q * s` must be **bf16** (or the kernel keeps the e4m3 code q, exact in fp16, and
  applies s in fp32). The captured taps (main_proj's input, 2,749 rows of A-capture-20261022) reach |x| = 280,576 with
  per-32 scales up to 2^10, so QDQ'd values reach 448 x 2^10 = 458,752: an fp16 overflow (max 65,504). main_x (after
  main_norm) stays below 0.38. Downstream activations are normed (attention / FFN inputs) or clamped (SwiGLU: |h| <=
  silu(10) x 10 x route weight); measured per-layer ranges come with the GPU parity runs.
* Choice: bf16 operands everywhere (both exact for the weights; bf16 has the range for the activations); fp16 only for
  code-domain operands with fp32 scale application. Router bias, gate scores and confidence projection: fp32. In the official BF16-head arms,
  base logits and Markov bias/sum are fp32. The four K6 serving arms retain the inherited engine policy:
  base logits, Markov bias and final sum round to BF16 (see §9); the K6 label is not an FP32-logit claim.

## 7. Measured (2026-10-23 04:19-04:55Z, GPU-7744b44b, 325 W owner cap, SM 1695-1980 MHz; results in offload/results/n1/)

Per card, one TP2 rank alone (solo rank 0: the rank's real shards and kernels, collectives replaced by same-shape
stand-ins), 8 slots, 327,680 positions, graphs for B 1-4 x n_out 4/5 + inserts 8-128:

| arm | torch peak during load (MiB) | nvidia-smi after load / serving (MiB) | graphed step p50 B1 / B2 / B3 / B4 (us) |
|---|---|---|---|
| k2-tp2 (control) | 4,604 | 4,950 / 5,218 | 1,752 / 2,243 / 2,520 / 3,341 |
| native-hybrid-kx (MXFP4 experts, KX regfed; bf16 dense, cuBLAS) | 6,833 | 7,182 / 7,578 | 2,143 / 2,502 / 2,790 / 3,394 |
| native-packed-kx (MXFP4 experts KX; FP8 dense, N1 Triton) | 6,509 | 6,870 / 7,136 | 2,134 / 2,532 / 2,776 / 3,661 |
| native-expanded (bf16 experts + dense, N1 Triton experts) | 16,344 | 16,686 / 17,068 | 2,864 / 3,858 / 4,669 / 5,677 |
| native-hybrid (N1 Triton packed experts) | 6,836 | 7,176 / 7,560 | 5,584 / 9,600 / 12,963 / 17,158 |
| native-hybrid-head (+ bf16 head, exact fp32 Markov, no cache) | 6,218 | 6,560 / 6,950 | 6,157 / 10,242 / 13,648 / 17,700 |
| native-hybrid-head-hot (+ 4,096 exact fp32 hot rows) | 7,230 | 7,570 / 7,948 | 6,082 / 10,248 / 13,615 / 17,575 |

Host: max RSS during start-up 14.8 GB (native-hybrid / -expanded) and 15.3 GB (-kx, host shards per stage), including
the file-backed pages of the memory-mapped official shards and lane E's checkpoint verification. Load: 9-11 s (N1
kernels), 135-140 s with KX regfed (per-expert admission + fragment prepack).

Fidelity on the live captures (A-capture-20261022, 453 rounds, 542 live blocks; TP2 emulated on one card; windows A =
captured main_x, B = main_x from taps): blocks exact vs the engine's (K2) drafts: control 92.6 / 93.4 % (= lane E),
native arms 58-63 % (they are different weights, as intended). Realized acceptance against the TARGET's own tokens (538
live blocks where it is determined; temperature 1.0: P(match) = p_target(draft), an unbiased per-position estimate):
engine 1.810; k2-tp2 1.801 / 1.805; native-hybrid-kx 1.823 / 1.825 (vs control +0.022 +- 0.021 / +0.020 +- 0.027);
native-hybrid 1.820 / 1.825; native-packed 1.823 / 1.820; native-experts-packed 1.822 / 1.818; k2-head 1.803 / 1.809;
native-hybrid-head 1.807 / 1.805. Native arms trend +0.01-0.02 tokens per block over the control, within ~1 standard
error: not established on 538 blocks; the in-engine A/B decides.
Exactness checks: KX regfed vs N1 kernels 99.47 % identical blocks (fp32 order near-ties), hot-row vs factorized Markov
100 % identical, graph replay == eager for native-hybrid and native-hybrid-kx (B 1-4, n_out 4/5).

## 8. N2 native-hybrid-kx2 (experimental, opt-in)

`native-hybrid-kx2` preserves native-hybrid-kx's official QDQ, FP32 router,
post-TP per-slot BF16 rounding and ascending-expert combination, and deployed
K6 draft head. It uses `KXFusedBank` with KX-FUSEDMOE's published
`select_variant(stage, rows)` / `dispatch.json`: mode 131072, except stage 1
at ten rows uses qualified mode 256. Workspaces and SLOT buffers are owned
by each rank, allocated before capture and reused on its ordered capture
stream. They cannot be shared across simultaneous independent streams.

`KXSelectiveDense` uses the delivered FP8 `configuration(M)` for the
measured draft operations at M5/M10/M15 and main_proj M8. The final preset
selects `dense_profile="measured-small"`; M20 and unlisted shapes use BF16. Both source representations are verified lossless at load;
activation QDQ remains the caller's responsibility. Candidate outputs and
split scratch are preallocated. Existing graph-captured PyTorch operations
elsewhere retain their previous allocation behavior. This is not a claim
that the entire eager drafter has become allocation-free. No head epilogue
is added; the grouped native head epilogue's FIX1 FP32 default is unchanged.

N2 tools live in `native/tools/n2/`. On X86host, stage the source into
`~/dsv41-offload/n2/repo`, then run `queue.sh` in detached tmux. Every job
uses `run_locked.sh`, pins only GPU-7744b44b, retains the inherited exclusive
lock and times out at 830s with at most 5s teardown. Logs include clocks,
power, temperature and the unchanged owner power limit. `dense_screen`
compares same-shard FP8/BF16 paths; `inmodel_check` audits actual model-call
activations against decoded-value FP64 and the prior register pipeline.
Detailed differences and captured parity dumps stay outside git.

`pair.sh` is for the coordinator only: it refuses execution unless PAIR_OK
names N2 or N2_PAIR_BY_COORD=1, locks both cards separately for every job,
and runs bench plus capture parity for K2, kx and kx2. This lane never runs
it. Solo timings do not establish the pair/engine performance objective.
Serving integration still needs worker.server native-arm construction,
V1-schema kv_window/s*.attn_out/main_x_P dumps and classification gates,
RX pair-safety qualification and the coordinator's E2/E3 engine window.

N2's selected `dense_profile="measured-small"` extends FP8 to every
measured draft dense operation at M5/M10/M15, including four-group wo_a. It
retains BF16 M20, main_proj M32/M128 and unlisted shapes. At T20 the shared
w1/w3 BF16 matrices are concatenated by output row and projected together;
each output still rounds to BF16. This removes a poor cuBLAS shape choice,
without changing weight values, scale alignment, activation QDQ or FP32
accumulation. The extra verified BF16 copy is 67.5 MiB per rank across stages.

The experiment also uses allocation-free shared SwiGLU+QDQ and post-TP
ascending-ID combine epilogues under kernels/fp8dense/ and kernels/fusedmoe/.
Both preserve the original arithmetic and rounding points and have exact
old-pipeline/changed-input graph tests. Shared paired-GEMM rounding can differ
from separate GEMMs; the model oracle lists those projection/HQ differences
separately and checks the same BF16/FP64 gates. No FP16 narrowing, approximate
exponential, atomic accumulation, host routing readback or missing QDQ is added.

For reproducible timing, native_bench now settles each B/n_out shape for 0.5s
without changing clocks, then uses the original 200 event samples. It records
per-case telemetry and all kernel names for both output counts. Historical
first-case values with only ten warmups can include automatic-boost ramping;
keep them separate from the settled repeated comparison. The N2 in-model oracle
streams all detailed element differences to a local JSONL and stores aggregate
metrics plus record references in its JSON. Both files remain outside git.

The final `native-hybrid-kx2` preset selects `measured-small`. The original
wq_b5/main8-only experiment remains reproducible with an explicit
`ArmConfig` replacement `dense_profile="initial"`; native-hybrid-kx remains
the unchanged comparison/rollback arm. Final gates: 171 component tests,
all-shape full-emulated model FP64/N1 dense and expert-projection oracles,
and all eight graph/eager cases. Pair/engine qualification remains separate.

Final N2 repeated solo result (three interleaved repetitions, each 200 events):
all 16 rank/B/output-count targets pass, 3.45–10.91% below K2 by median p50.
For every case, the slowest observed kx2 run is below the fastest observed K2
run. See `tools/n2/evidence/repeated-summary.json` for all values/ranges, full
kernel-name medians, telemetry, memory and frozen-source hashes, and that
directory's README for reproduction. These are solo channel-shard timings;
physical pair, V1 causal/state and E2/E3 serving results remain unqualified.

## 9. N3 native serving and the V1 entry gate

`worker.server --tp 2` accepts `DSPARK_WORKER_ARM=native-packed|native-expanded|native-hybrid-kx|native-hybrid-kx2`.
K2 remains the default. Native identities cover official shard provenance, exact arm config, retained K6 head provenance
and runtime/kernel source hashes; `spark/tools/identity.py --arm ARM RECEIPT` computes the matching Spark value.
The registered four active slots, negotiated max length, graph prewarm, status and RX poison/failure handling remain.
Missing shards, identity mismatch and failed prepack refuse; there is no implicit K2 fallback.

All four native serving arms keep the **target-shared EXL3 K6 head**. N4 corrects the inherited arithmetic: head
outputs, Markov bias and their sum remain FP32. Native arms do not use the incompatible BF16 Markov cache.
Attention uses the official online 64-key schedule, per-block BF16 probabilities, FP32 running state and sink last.
It reads physical ring order, compacting short windows before the draft keys, into preallocated rank output storage.
The pinned official TileLang CUDA uses full-precision expf/division and a contracted denominator update; the native
port uses libdevice.exp, div_rn and explicit FMA. K2 retains its deployed arithmetic. K6 weights remain a representation
difference from the official BF16 head; "native" does not imply universal bit identity across GPU reduction schedules.

N3 adds opt-in serving dumps (`DSPARK_V1_DUMP_DIR`, selected `DSPARK_V1_ROUNDS`) with actual insertion-graph main_x,
live window KV, routing and layer/head outputs. Each selected traced eager replay must equal its graph reply. The
independent V1 classifier remains unchanged, comparing mirror/mirror64 with strict state, attention and KL gates.
N3's historical 12-key packed pilot failed: two own-logit flags, s0 attention median excess600.402x and KL3.41x band.
Its BF16 policy reproduced those drafts; official FP32 recomposition caused the flags. N4's corrected packed/kx/kx2 pilots
clear state flags, unexplained disagreements and KL, but still fail strict stage-0 attention excess. A component
control using unmodified official GPU attention at stage0 and the official CPU reference elsewhere also fails that
attention gate (55.34x on these 12 samples); this is a calibration question, not a worker waiver. The real worker has
additional residual arithmetic differences. Do not classify the whole discrepancy as harmless or promote on graph
parity alone. A passing V1 verdict and physical-pair qualification remain prerequisites before GB10 promotion.
See the campaign N4 report for current verdicts and same-card cost measurements; historical N2/N3 timings used a
different GPU and power cap.

Current N4 solo/reproduction commands and memory scopes: `tools/n4/README.md`. Historical N3 serving/fake-client/V1/pair commands and depth recipe: `tools/n3/README.md`. The N3 solo wrapper pins the former development card; N4 uses its reassigned bc wrapper.
The pair script is coordinator-only and never changes a service or Spark. Expanded full TP2 emulation does not fit a
24GiB solo card; measure each real shard separately, then run the full serving/dump equality checks on the physical pair.
