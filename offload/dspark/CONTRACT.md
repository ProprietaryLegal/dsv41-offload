Status 2026-10-23: Normative wire/state contract v1.8; capacity + async production W5. prompt_logprobs row retention note is current; no new wire version is introduced by this docs pass. See [live roadmap](../../documents/plans/roadmap.md) and [measured record](../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# Remote DSpark drafter — wire and state contract (DSV41-OFFLOAD)

Contract version **1.8** (2026-10-22; documentation note 2026-10-23). Owner: lane O3 (Spark side; closed, edits
assigned by the coordinator). Lane O2 implements the x86host worker against it.
Every change is listed in the change log (§13) with a version bump; the wire `version` field changes only on an
incompatible change. Propose changes in `O2.progress.md` (the coordinator relays them).

This contract comes from the proposer that runs in the live serve. File references point to the read-only copy of the
image at `/opt/dsv41/data/src-image/` (vLLM `0.31.1.dev0+g6e517b15.dsv41gb10`, image `sha256:435a5236`). It
does not come from the upstream remote-dspark mod (vLLM 0.29, GLM/Kimi), which uses a different state model.

## 1. Plain summary

- Each engine step, Spark rank 0 sends the worker the **target tap rows** for the positions the target has just
  accepted. Rows are only sent for those positions, and never more than the last 128 per request. Rank 0 also sends
  one **anchor token** per request.
- The worker keeps a **128-position window** of those rows per request slot. It drafts a block of **5 tokens**, with
  a **confidence** for each, by greedy argmax.
- Rank 0 broadcasts the drafts and confidences to rank 1 over the existing TP group. The verifier, rejection sampler,
  depth policy and k schedule stay unchanged on the Sparks.
- Nothing a draft produces is ever committed to the window. Only context rows that the target has verified are stored.
- Prefix-cache hits, preemption, resume and reconnection need no special worker logic. The engine already recomputes
  the last 128 tokens of a prefix hit (SWA bounded replay), and rank 0 keeps a copy of every live window, so it can
  re-send one at any time.

## 2. Facts from the running proposer (verified in source)

| fact | value | source |
|---|---|---|
| speculator | `DSparkSpeculator(DFlashSpeculator)`; `method="dspark"` → `init_speculator` | `v1/worker/gpu/spec_decode/__init__.py:35-40` |
| block / k | `num_speculative_tokens = 5` = `dspark_block_size`; anchor-as-first: N = 5 query tokens per request (`[anchor, noise×4]`), every query position predicts the next token | `dspark/speculator.py:47-56`, `dflash/speculator.py:624-671` (`SAMPLE_FROM_ANCHOR`) |
| noise token | `dspark_noise_token_id = 128799` (`parallel_drafting_token_id`) | config.json; `spec_decode/utils.py:85` |
| conditioning | `main_x = main_norm(main_proj(cat(aux_hidden_states, -1)))`, `main_proj` 15,360 → 5,120 (`ReplicatedLinear`, EXL3 K5 in this checkpoint), `main_norm` RMSNorm eps 1e-20 | `nvidia/dspark.py:104-112, 163-168`; `dflash/speculator.py:376-382`; config `quantization_config.exl3_moe.dense.drafter.main_proj = 5` |
| taps | `aux_hidden_states` = 3 tensors `[T, 5120]` bf16, in the order of `dspark_target_layer_ids = [37, 38, 39]`. Tap L = mean over the 4 hc copies of the residual stream **entering** decoder layer L (0-based), i.e. the output of layer L−1 | `eagle3_utils.py:113-120` ("v4.1 reads the attention inputs"), `nvidia/model.py:540-541, 1119-1138` |
| tap validity | in eager prefill steps, layers ≥ 20 (`max(kv_source_layer_ids)`) run only on each request's **last min(q, 128) rows** of the step (decoder SWA bounded replay, patch 0034). The trimmed rows' taps are **zeros**. **Exception: a request that asks for prompt logprobs keeps every row** (`add_request` sets `_keeps_rows_np` when `sampling_params.prompt_logprobs` is set; `_prepare_replay_batch` then uses `kept_lens = query_lens` for it), so all its rows run every layer and carry real taps; lane D2-DATA's replay requests rely on this (`prompt_logprobs: 20`), with the overlay's zero-row count as the backstop (commit e0760ef: DONE.json `zero_rows`, refused by replay run/copy, equiv and the dataset build). CUDA-graph (decode) steps compute every row | `nvidia/model.py:793-848`, `decoder_replay_layers.py:65-85`, `nvidia/model_state.py:237-243` (`_keeps_rows_np`), `:427-431` (`kept_lens = where(keeps_rows, query_lens, min(query_lens, window))`) |
| tap placement over TP | replicated on both ranks. The live config has no sequence parallel (no `--enable-expert-parallel`); with SP the engine all-gathers aux before returning. **Rank 0 alone holds complete taps** | `nvidia/model.py:245-255, 1134-1138` |
| context KV | each draft layer: `kv = kv_norm(wkv part of fused_wqa_wkv(main_x))`, RoPE + quant + insert at the context positions | `nvidia/dspark.py:170-201, 256-279` |
| context rows per step | the target's rows minus the rejected suffix: `num_valid_ctx = num_scheduled − num_rejected`, positions = the target's positions of those rows | `dflash/speculator.py:570-622` |
| anchor | `last_sampled[req]` if `num_sampled > 0`, else `next_prefill_tokens[0, req]` (partial prefill). Anchor position = last valid context position + 1 | `dflash/speculator.py:578-585, 625` |
| draft window | the draft layers are pure sliding-window attention (`compress_ratios[40:43] = 0`), window W = 128, but the DSpark draft block does **not** use the causal per-token window (`[max(P−127, 0), P]`, `sparse_swa.py:1060-1066`, the target's SWA layers). Its non-causal kernel anchors one window to the whole block: with the context ending at L = anchor_pos − 1, **every** query token of the block attends to the 128 context positions `[max(anchor_pos − 128, 0), anchor_pos − 1]` plus all N query tokens of the block (the earlier and the later ones) | config `sliding_window = 128`; `v1/attention/backends/mla/sparse_swa.py:1253-1255` (`ComputeDSparkNoncausalSWAIndicesKernel`: `start_pos = max(prefix_len − window_size, 0)`, `end_pos = seq_len`), `:550-562` + `compressor_utils.py:21-27` (index width W + N, padded); `nvidia/dspark.py:9-10` |
| query clamp | query positions `min(pos, max_model_len − 1)` with max_model_len = 327,680 | `dflash/speculator.py:656-657, 679-681` |
| sampling (live) | greedy drafts, `draft_logits is None`. `VLLM_DSPARK_LOCAL_ARGMAX=1` gives the argmax of (base logits + Markov bias) and resolves ties to the lower vocabulary index (= `torch.argmax`). The sequential Markov step uses the previous **sampled** token (step 0: the anchor). Markov bias rows come from the cache `VLLM_DSPARK_MARKOV_CACHE=/root/.cache/markov_top_tokens.json` | `dspark/speculator.py:173-230, 120-130`; `nvidia/dspark.py:369-450`; live container env |
| confidence | `sigmoid(confidence_head(cat(head_hidden, markov_embed)))` per draft position. Consumed by the depth policy `VLLM_DSPARK_DEPTH_POLICY=conf` (live) | `dspark/speculator.py:223-230`; `nvidia/dspark.py:466-471`; `depth_policy.py` |
| k actually verified | decided **on the Spark only**: 0040 schedule `[[1,2,5],[3,4,4]]` (k = 5 at 1–2 requests, 4 at 3–4), 0047 adaptive window in the scheduler (`VLLM_DSV41_ADAPTIVE_K=1`, from accepted-draft EMAs), then per-request depth from the confidences (rows past the depth are dead and never accepted) | `v1/core/sched/scheduler.py:1491-1500, 2027-2036`; `model_runner.py:1528-1540, 1693-1698`; `depth_policy.py` |
| prefix-cache hit | `CacheConfig.swa_bounded_replay = True` (default; V2 runner, no PCP, not ROCm): sliding-window groups (the draft layers included) are **not** prefix-cacheable. On a hit of h tokens the scheduler rewinds to `replay_start = h − 128` and the target **recomputes those 128 tokens**, so their taps reach the drafter like any prefill rows. Hits ≤ 128 tokens are discarded and fully recomputed | `config/cache.py:240`; `models/deepseek_v41/attention.py:529-570`; `v1/kv_cache_interface.py:950-955`; `v1/core/sched/scheduler.py:1002-1012, 1055-1065, 3126-3135` |
| preemption | the V2 runner removes preempted requests (`finish_requests` unions `preempted_req_ids`) and re-adds them as new requests (prefix hit + replay as above) | `model_runner.py:1187-1199, 1216-1258` |
| request slot | `idx_mapping[i]` = request-state index 0..`max_num_seqs`−1 (4). Slots are reused | `model_runner.py:1341-1342`; `states.py` |

## 3. Conditioning layout: send the raw taps (decided with numbers)

Two candidates: `main_x` (5,120 bf16 = **10,240 B/row**, computed on rank 0) or the raw taps (3 × 5,120 bf16 =
**30,720 B/row**, `main_proj` + `main_norm` on the worker).

| item | main_x on Spark | taps to worker |
|---|---|---|
| Spark work per round | `main_proj` GEMM on rank 0. EXL3 K5 weights are 46.9 MiB; at the 273 GB/s LPDDR5x floor that is ≥ 0.18 ms per round, ~0.2–0.3 ms in practice | none (row gather only) |
| Spark memory (capacity mode) | `main_proj` + `main_norm` must stay loaded on rank 0 (≈ 47 MiB) | **no drafter tensor on either Spark** |
| wire per decode round (at most 1+k rows/request) | 1 req ≤ 61 KiB; 4 req ≤ 200 KiB | 1 req ≤ 180 KiB; 4 req ≤ 600 KiB |
| extra wire time for taps at 4.6 GB/s effective (40 Gb/s) | — | +26 µs (1 req, 6 rows) … +89 µs (4 req, 20 rows) |
| worker `main_proj` on the 3090 | — | ≈ 0.05–0.08 ms (47 MiB at ~900 GB/s) |
| prefill (≤ 128 rows/request/step, §4.3) | 1.25 MiB per request-chunk | 3.75 MiB per request-chunk (+0.5 ms against a ~1 s chunk) |

**Decision: payload kind 1 = TAPS_BF16.** The taps cost 25–90 µs more on the wire but save ≥ 0.18 ms on rank 0,
and they keep both Sparks free of drafter weights. The result: about 0.05–0.15 ms better per round at 1–4 requests,
plus a clean capacity mode. Kind 2 = MAINX_BF16 is reserved in HELLO. Revisit it only if the measured link (lane S1)
or the worker's `main_proj` time (lane O2) reverses the sign.

Row layout: `row = cat(tap37, tap38, tap39)` as little-endian bf16. That is exactly row r of
`torch.cat(aux_hidden_states, dim=-1)`, the tensor `combine_hidden_states` consumes, so the worker computes
`main_x = main_norm(main_proj(row))` with the same weights.

## 4. Per-round data flow

### 4.1 Who talks to whom
- **Rank 0** (sparkhead) is the only client. It uses one connection to the worker over its direct link
  (10.202.28.2 → 10.202.28.1) through lane O1's transport (TCP or RDMA RC). It sends one request and waits for the
  reply.
- **Rank 1** (sparkworker) never talks to x86host. It receives drafts, confidences and a status word in one NCCL broadcast
  from rank 0 on the TP group. Both ranks must feed identical draft tokens to the target, because TP ranks diverge
  otherwise. The broadcast also makes every fallback decision identical on both ranks.
- Why not have both ranks receive from x86host? It would save one ~20 µs NCCL hop. But the two links can fail
  independently, so the ranks would still need an agreement collective, and then nothing is saved. Rejected for v1.
  Revisit only if the coordinator measures that the TP broadcast dominates.

### 4.2 Sequence (sync mode, all policies)
1. The target forward and the rejection sampler finish on both ranks, as today.
2. Rank 0, on its GPU stream: gathers the candidate rows of the taps, drops rejected rows (device-side compaction
   using `num_rejected`, no host sync), and updates its device ring (§7.3). It then copies rows and metadata into a
   pinned staging buffer and waits for that copy.
3. Rank 0 sends `STEP` and waits for the reply, with a deadline (§10).
4. Rank 0 uploads `[status, drafts, confidences]` and broadcasts them on the TP group. Rank 1 receives the same
   tensor. Both ranks write `draft_tokens` and `draft_token_confidence_probs`.
5. The engine continues unchanged: the depth policy records confidences, the scheduler takes the drafts, the next
   verify runs.

An optional async variant exists for capacity mode (`VLLM_DSPARK_REMOTE_ASYNC=1`). The engine stream waits on a
device flag (`cuStreamWaitValue32`) that the proxy thread's side stream writes after the reply is in device memory,
so rank 0's CPU never blocks. It is transparent to the worker: the same messages, the same order.

### 4.3 Which rows are sent (the trailing-window rule)
For request i in the step, with q_i scheduled rows and `num_rejected_i` rejected:
- the candidate rows are the **last `n_i = min(q_i, W)` rows** of the request's span, with W = 128;
- the valid rows are the first `n_i − num_rejected_i` of those. `num_rejected_i > 0` only for decode rows, where
  `q_i ≤ 1 + k ≤ 6 < W`, so every candidate row is a decode row;
- valid rows are sent in position order: `first_pos` … `first_pos + n_rows − 1`, and `anchor_pos = first_pos + n_rows`.

Why this is complete:
- Within a step, only the last min(q, 128) rows have real taps (decoder bounded replay).
- The draft block reads exactly the W = 128 context positions before the anchor, `[anchor_pos − 128, anchor_pos − 1]`
  (fewer near position 0), for every query token of the block (§2, block-anchored window). It never reads further
  back, and it needs all 128: the window is 128 rows, not the 127 of a causal per-token window.
- By induction over steps, the worker always holds every position in `[max(lo, anchor_pos − W), anchor_pos − 1]`,
  where `lo` is where the request's rows begin. That is position 0 for a fresh prompt, or `replay_start` after a
  prefix hit. Bounded replay recomputes the 128 tokens before the hit (`replay_start = h − 128`), so once the first
  chunk is done every one of the 128 rows the anchor reads exists.

## 5. Transport framing

The contract is transport-independent. One request message gets one reply message. Requests on a connection are
processed in order, with one outstanding request per connection in v1.
- **TCP stub:** each message is a little-endian `u32 length` followed by `length` bytes (AppHeader + body).
  `TCP_NODELAY`, explicit local bind to the direct-link address.
- **O1 transport `ofl`** (lane O1, `offload/transport/API.md`, commit 41cf2f7 or later): one contract message is one
  ofl request.
  - ofl `op` = AppHeader.op (1–5; ofl's op 0 is its built-in echo); ofl `generation` = AppHeader.seq (ofl echoes
    it).
  - Payload = the whole contract message (AppHeader + body). The reply is the whole contract reply frame,
    RDMA-written into one registered 64 KiB range.
  - service `dspark-v1`, `max_inflight` 1, `max_req_bytes` 16 MiB (worst case STEP: 32 + 192 + 512 × 30,720 B
    = 15.7 MB), `max_resp_bytes` 64 KiB.
  - Ports: **18620** = drafter on the Spark A link (server 10.202.28.1, client binds 10.202.28.2); 18621 is
    reserved for the Spark B link. O1 uses 18610 (Engram) and 18600–18601 (self-tests).
  - An ofl timeout breaks the connection before returning (the QP goes to error), so a late reply never lands. The
    proxy then reconnects (same session) and resyncs (§7.5).

  Endpoints, as the Spark proxy writes them: `o1rdma://10.202.28.1:18620?bind=10.202.28.2` (RDMA RC) and
  `o1tcp://…` (ofl's TCP backend). The contract's own TCP stub is `tcp://…` (§5 first bullet).

## 6. Messages (little-endian; all offsets in bytes)

### 6.1 AppHeader (32 B, every message)
| off | type | field | value |
|---|---|---|---|
| 0 | u32 | magic | `0x31505344` (bytes `D S P 1`) |
| 4 | u16 | version | 1 |
| 6 | u16 | op | 1 HELLO, 2 STEP, 3 RESET, 4 STATS, 5 PING |
| 8 | u64 | seq | client sequence number, strictly increasing within a session (gaps allowed); the reply echoes it |
| 16 | u32 | session | nonzero random id chosen when the Spark proxy starts; the reply echoes it |
| 20 | u32 | body_len | bytes after the header |
| 24 | i32 | status | 0 in requests; frame status in replies (§9) |
| 28 | u32 | flags | 0 (reserved) |

### 6.2 HELLO (op 1): request body 192 B
| off | type | field | live value |
|---|---|---|---|
| 0 | u32 | proto_min | 1 |
| 4 | u32 | proto_max | 1 |
| 8 | u32 | num_spec (N) | 5 |
| 12 | u32 | window (W) | 128 |
| 16 | u32 | hidden | 5120 |
| 20 | u32 | n_taps | 3 |
| 24 | u32[4] | tap_layer_ids | 37, 38, 39, 0 |
| 40 | u32 | payload_kind | 1 = TAPS_BF16 (2 = MAINX_BF16 reserved) |
| 44 | u32 | row_bytes | 30720 |
| 48 | u32 | noise_token_id | 128799 |
| 52 | u32 | vocab_size | 129280 |
| 56 | u32 | max_model_len | 327680 |
| 60 | u32 | max_slots | 4 (`max_num_seqs`) |
| 64 | u32 | max_rows_per_step | 512 (= max_slots × W) |
| 68 | u32 | sample_mode | 1 = GREEDY (argmax of base + Markov bias, lowest index on ties, confidence head on) |
| 72 | u32 | tp_size | 2 (informative) |
| 76 | u32 | reserved | 0 |
| 80 | u8[32] | drafter_identity | sha256 (§6.7) |
| 112 | u8[32] | markov_identity | sha256 of the bytes of `markov_top_tokens.json` (zeros if no Markov cache) |
| 144 | u8[48] | client_name | ASCII, NUL-padded |

HELLO reply body (128 B), status 0 or an error from §9:
| off | type | field |
|---|---|---|
| 0 | u32 | proto (chosen) |
| 4 | u32 | max_slots |
| 8 | u32 | max_rows_per_step |
| 12 | u32 | capabilities (bit0 TAPS_BF16, bit1 MAINX_BF16, bit2 HAS_MARGIN) |
| 16 | u32 | window |
| 20 | u32 | num_spec |
| 24 | u8[32] | drafter_identity (the worker's) |
| 56 | u8[32] | markov_identity (the worker's) |
| 88 | u8[40] | worker_name (ASCII) |

The worker rejects the HELLO in these cases:
- `IDENTITY_MISMATCH` if either identity differs;
- `CONFIG_MISMATCH` if N, W, hidden, n_taps, tap ids, row_bytes, noise id, vocab or sample_mode differ, or the
  payload kind is unsupported.

A HELLO with a new session id makes the worker drop all state of any previous session. The worker serves one client.
A HELLO with the current session id (a reconnect) keeps the windows, but the client resyncs every live slot anyway
(§7.4).

### 6.3 STEP (op 2): the per-round message
Request body:
```
StepHeader (32 B)
  0  u64 step_id      rank-0 engine step counter (monotonic; echoed; informative)
  8  u16 n_req        1..max_slots
 10  u16 step_flags   bit0 WANT_MARGIN (return top1-top2 margins, §6.3 reply), others 0
 12  u32 n_rows       total rows in the row block
 16  u32 row_bytes    = HELLO row_bytes
 20  u32 rows_offset  offset of the row block from the start of the body, multiple of 64
 24  u32 reserved[2]
ReqEntry[n_req] (32 B each), batch order
  0  u8  slot         request-state index 0..max_slots-1 (unique within a STEP)
  1  u8  req_flags    bit0 NEW_GEN  discard the slot's window before applying rows
                      bit1 NEED_DRAFT  return drafts/confidences (0 = partial prefill chunk: append only)
                      bit2 RESYNC  the rows are a re-send of the window from the Spark's ring (informative)
  2  u8  k_window     drafts the target verified for this request in the step that produced these rows
                      (0 for prefill; informative, for worker acceptance stats)
  3  u8  n_out        draft positions the Spark will consume, 0 = all N (v1.3; sent as 0 before). The worker
                      still runs the full N-token block, so positions < n_out are bit-identical to a full-block
                      draft; it may skip head / Markov / confidence for positions >= n_out, which it returns
                      as draft 0, confidence 0.0, margin 0.0
  4  u32 generation   per-slot generation (§7.1)
  8  i64 first_pos    position of this request's first row (ignored when n_rows == 0)
 16  u32 n_rows       rows of this request in the row block: valid context rows only, 0 <= n_rows <= W
 20  i32 anchor_token token at anchor_pos
 24  i64 anchor_pos   last valid context position + 1 (= first_pos + n_rows when n_rows > 0)
zero padding up to rows_offset
Row block: n_rows x row_bytes; entries in ReqEntry order, positions ascending within an entry
```
Reply body:
```
ReplyHeader (32 B)
  0  u64 step_id        echo
  8  u16 n_req          echo
 10  u16 num_spec       N
 12  u32 reply_flags    bit0 HAS_MARGIN
 16  u32 worker_us      apply + draft compute time, µs
 20  u32 recv_to_send_us  µs from request fully received to reply handed to the transport
 24  u32 reserved[2]
ReqStatus[n_req] (8 B): u16 status (§9.2), u16 window_rows, u32 generation (echo)
draft_ids  : i32 [n_req][N]   target vocabulary ids (full-vocab drafter; no remap)
confidence : f32 [n_req][N]   sigmoid outputs in [0, 1]
margin     : f32 [n_req][N]   only if HAS_MARGIN: top1 − top2 of (base + Markov) logits at each step
```
For entries without NEED_DRAFT, and for entries with a nonzero per-request error, the worker writes draft ids 0 and
confidences 0.0. The same applies to positions ≥ n_out (§6.3 ReqEntry).

**Reply values the Spark refuses (v1.4).** The drafts become the target's input ids on both ranks and the confidences
feed the depth policy, so rank 0 checks every reply before anything is broadcast. Besides the shape and the echoes
(`step_id`, `n_req`, `num_spec`, per-entry `generation`), a reply is refused when, for any entry:
- `status` is not a §9.2 code;
- `window_rows > min(W, anchor_pos)`; or, for a NEED_DRAFT entry, status `OK` with `window_rows != min(W, anchor_pos)`
  or `PARTIAL_WINDOW` with `window_rows ≥ min(W, anchor_pos)` (§7.2 step 5);
- a consumed position (NEED_DRAFT, status OK or PARTIAL_WINDOW, position < n_out) has a draft id outside
  `[0, vocab_size)`, a confidence that is not a finite value in [0, 1], or (HAS_MARGIN) a margin that is not finite
  and ≥ 0;
- any other position has a nonzero draft id, confidence or margin.

A refused reply is a failed round (counter `fallback_bad_reply`, `last_bad_reply` in the stats): no draft of that
reply is used, rank 0 drops the connection, and after the reconnect every live slot resyncs (§7.4).

**n_out (v1.3).** The Spark sends `n_out = cap[n]` when a dynamic speculation schedule is set and the depth policy runs
a mode that masks past the cap (`conf`, `confb` or `confr`, with no `VLLM_DSPARK_DEPTH_FILE`, which could switch the
mode at runtime). All of these are live, with schedule `[[1,2,5],[3,4,4]]`, so n_out is 4 at 3–4 requests. Otherwise
it sends 0 (all N): `fixed:<d>` and `off` verify by the scheduler window alone, which can exceed `cap[n]` when the
next step has fewer requests. The shadow policy always sends 0, so parity covers every position.
`VLLM_DSPARK_REMOTE_N_OUT=0` turns n_out off. In the cases where n_out is sent, nothing reads the positions ≥ cap:
- `DSparkDepthPolicy.record` zeroes the confidences at positions ≥ `cap[num_reqs]` of the current step before
  choosing depths (`depth_policy.py:432-434`).
- Rows past a request's depth are dead and never accepted (`model_runner.py:1528-1540, 1693-1698`), even when the
  next step has fewer requests.
- Calibration only uses positions below the verified depth.

Measured by lane O2: at 4 requests × n_out 4 the head runs 16 instead of 20 rows (a 16-row instead of a 32-row EXL3
tile), saving ≈ 0.55 ms per round.

### 6.4 RESET (op 3)
Body `u32 slot_mask` (bit i = slot i; `0xFFFFFFFF` = all). The worker discards those windows. Reply: empty body.

### 6.5 STATS (op 4)
Empty body. Reply body: UTF-8 JSON of worker counters. Suggested keys: `steps`, `rows`, `drafted_requests`,
`partial_windows`, `stale_generation`, `bad_entries`, `worker_us_p50/p99`, `accepted_hist` (from n_rows−1 against
k_window), `gpu_mem_bytes`, `session`, `uptime_s`.

### 6.6 PING (op 5)
Empty body. Reply: empty body, status 0. Used for health checks and RTT probes. A PING never changes state.

### 6.7 drafter_identity
`sha256` over the lines `"<basename> <sha256-hex>\n"`, sorted by basename, for every source file the worker reads
tensors or configuration from. For the live build these are:
- `config.json`, `exl3_moe_plan.json`;
- `model-00002`, `model-00043` … `model-00046-of-00046.safetensors`;
- `model-exl3-head.safetensors`, `model-exl3-mtp.safetensors`, `model-exl3-mtp-experts.safetensors`.

Each file's sha256 is the **Spark-side source hash** that lane S2 recorded in `DSPARK_COPY_DONE` (`files[].sha256`;
x86host `/opt/dsv41/x86host-data/dspark/DSPARK_COPY_DONE`). For the eight tensor files this equals the deployment's
pinned `manifest.json`. config.json and exl3_moe_plan.json are legal-v2a's own and are pinned by the receipt.
`offload/dspark/spark/tools/identity.py` computes the digest from the receipt; the coordinator passes it to the Spark
as `VLLM_DSPARK_REMOTE_IDENTITY`.

The worker computes the same digest from the same receipt. It must also check that the extracted files it actually
loads match the receipt's `extracted_files[].sha256`, which is how its tensors are tied to these sources. A worker
built from the baseline checkpoint, or any other one, is refused at HELLO.

**Live values (legal-v2a, 2026-10-22):**
- `drafter_identity = be2c1436fa9f9e66b398e1b1da2e140e6fddc82a11939e2ac4240a0ca79543bd`
- `markov_identity = 0b6e71e638abd5abce2db433965bb5ff523cbbe92ed7ae54324a50ccaeffa19a`, the sha256 of the live
  `markov_top_tokens.json`. It is byte-identical on both Sparks and on x86host.

## 7. State and lifecycle

### 7.1 Slot, generation and the Spark's bookkeeping (rank 0)
- The worker keeps per slot: `generation`, the contiguous position range `[lo, hi]` of the rows it holds (or empty),
  and the row data keyed by `pos mod W`.
- Rank 0 keeps per slot: `owner` (request id), `generation`, the same `[lo, hi]`, the device ring rows (§7.3) and a
  `dirty` bit.
- Each round, for each request in the batch:
  - If its request id differs from the slot's `owner`, the slot was reused (finish, abort, preemption or a new
    request). Rank 0 sets `owner`, increments `generation` (u32, wraps; compared by serial-number arithmetic), clears
    the ring range and sends NEW_GEN.
  - Nothing else changes a slot's generation on rank 0 (failures, resyncs and reconnects keep it), so within a
    session rank 0 never sends a slot a generation older than one it sent before, with or without NEW_GEN.
  - If the same request comes back at a lower position (re-add after preemption, streaming-input update, prefix
    replay), `first_pos ≤ hi`. Both sides roll back: rows at positions ≥ `first_pos` are dropped (§7.2). Rows below
    stay valid because a causal hidden state depends only on earlier tokens. The generation does not change.
- No CLOSE message exists. A finished request's window stays on the worker until its slot is reused, which takes at
  most `max_slots` × (W × row_bytes + KV) of worker memory. Cancellation and abort are therefore implicit. An
  explicit RESET is available.

### 7.2 Worker apply rule (per ReqEntry, in order)
1. If `generation` is older than the slot's: status `STALE_GENERATION`; ignore the entry (the window is kept).
   This check comes first and holds **whatever the flags**: NEW_GEN never makes an older generation acceptable.
   If it is newer, or NEW_GEN is set (with the slot's own generation: a resync): empty the window and adopt the
   generation.
2. Validate: `0 ≤ n_rows ≤ W`; `first_pos ≥ 0`; if `n_rows > 0` then `anchor_pos == first_pos + n_rows`; the rows
   fit in the block. On failure: `BAD_ENTRY`.
3. Apply rows `[p0, p0+n)`:
   - if the window is empty, or `p0 > hi + 1` (a gap): `lo = p0`;
   - else (`p0 ≤ hi + 1`): drop rows ≥ `p0`, i.e. roll back, and set `lo = min(lo, p0)`;
   - then `hi = p0 + n − 1` and `lo = max(lo, hi − W + 1)`.

   If `n_rows == 0`: require `hi + 1 ≥ anchor_pos`, then roll back to `hi = anchor_pos − 1`.
4. After applying, require `hi == anchor_pos − 1`, else `BAD_ENTRY`. Then
   `window_rows = hi − max(lo, anchor_pos − W) + 1` and `expected = min(W, anchor_pos)`.
5. If NEED_DRAFT:
   - draft with queries `[anchor_token, noise × (N−1)]` at positions
     `min(anchor_pos + j, max_model_len − 1)`, j = 0..N−1, over the window rows;
   - status `PARTIAL_WINDOW` if `window_rows < expected`. This is non-fatal: the drafts are produced but degraded,
     and the verifier keeps the output exact. Otherwise status `OK`.
6. Draft (query) rows and draft tokens are **never** inserted into the window. The next round's rows replace them.

### 7.3 Rank 0's device ring (the window's backup copy)
- Rank 0 writes every valid row it sends into `ring[slot, pos mod W]` with the same `[lo, hi]` rule, on the GPU.
  Size: 4 slots × 128 × 30,720 B = **15 MiB**, rank 0 only.
- This ring is rank 0's own copy of every live window. The worker's copy is a cache that rank 0 can rebuild at any
  time. In capacity mode the 15 MiB counts as retained memory.

### 7.4 Resync (failure recovery without a local drafter)
Rank 0 marks a slot `dirty` when:
- a STEP that carried rows for it failed, timed out or got no reply (rank 0 cannot know whether the worker applied
  it);
- after a reconnect, `NO_SESSION` or `STALE_SEQ`;
- the slot got a per-request status other than `OK`.

The next time a dirty slot appears in a batch, rank 0 sends `NEW_GEN | RESYNC` with the slot's whole ring window.
That means rows `[max(lo, anchor_pos − W), hi]`, already including this round's new rows. The worker's window then
holds exactly the rows it would have held with no failure. Its derived KV is recomputed in a different batch shape
(≤128 rows at once instead of 1–6 per step), and EXL3 kernel rounding depends on the batch size, so drafts after a
resync can differ at near-ties (lane O2 observed 2 of 5 positions in one test). The verifier keeps the output
exact; only proposals can change. No content-addressed prefix store is needed while SWA bounded replay is on (§2). If a future config turns bounded replay off, a hit at h leaves
`[h − 128, h − 1]` without rows: the worker reports `PARTIAL_WINDOW` and rank 0 counts it. A `PREFIX_RESTORE` op is
reserved (op 6) for that case and is not part of v1.

### 7.5 Sessions, sequence numbers, connections
- `session`: a random nonzero u32 per Spark proxy start. It also changes after an engine restart.
- `seq`: starts at 1 and strictly increases. The worker records the last applied seq and rejects `seq ≤ last` with
  `STALE_SEQ`, without applying the frame. The client never retries a STEP; after a failure it sends a new STEP (with
  RESYNC) under a new seq.
- **Timeout:** the client closes the connection and later reconnects with the same session, sending HELLO again.
  Frames from the dead connection that the worker processes later are either already older than the RESYNC (the
  worker processes frames in the order received) or rejected by seq.
- The worker accepts a new connection on a listener by closing the previous one on that listener. Newest wins.

## 8. Budgets

### 8.1 Bytes per round (TAPS_BF16, 30,720 B/row; headers excluded except where shown)
| case | rows sent (max) | request | reply |
|---|---|---|---|
| decode, 1 request, k=5 | 1 + accepted ≤ 6 | ≤ 184,320 B + 96 B headers | 32 + 8 + 40 = 80 B |
| decode, 2 requests, k=5 | ≤ 12 | ≤ 368,640 B | 32 + 16 + 80 = 128 B |
| decode, 3 requests, k=4 | ≤ 15 | ≤ 460,800 B | 32 + 24 + 120 = 176 B |
| decode, 4 requests, k=4 | ≤ 20 | ≤ 614,400 B | 32 + 32 + 160 = 224 B |
| one 4,096-token prefill chunk (1 request) | 128 | 3,932,160 B | 80 B |
| a step holding r prefill requests (r ≤ 4) | 128 each | r × 3.93 MB | |
| without the trailing-window rule (for contrast) | 4,096 | 125.8 MB | |

At the nominal 40 Gb/s (≈ 4.6 GB/s effective) these come to:
- 40 µs for one 6-row decode request, 134 µs for 4 requests;
- 0.85 ms per prefill request-chunk.

Lane S1's measured latency and throughput replace these figures. With main_x the numbers would be one third (§3).

### 8.2 Memory
| where | item | size |
|---|---|---|
| Spark rank 0 | device ring (§7.3) | 15 MiB |
| Spark rank 0 | device staging (gather + ring-resync stage) | 2 × 512 rows × 30 KiB = 30 MiB, bounded by max_rows_per_step |
| Spark rank 0 | pinned message buffer (rows sent zero-copy over TCP; copied once into the ofl send slot) | 15 MiB |
| Spark rank 0/1 | broadcast tensor | < 1 KiB |
| worker | context rows | 4 × 128 × 30 KiB = 15 MiB (or the derived KV, worker's choice) |
| worker | per-step input | ≤ 512 rows × 30 KiB ≈ 15 MiB |

## 9. Status codes

### 9.1 Frame status (AppHeader.status in replies)
| code | name | client action |
|---|---|---|
| 0 | OK | |
| 1 | BAD_VERSION | fatal configuration error: log, disable remote (policy fallback) |
| 2 | IDENTITY_MISMATCH | fatal: log both digests, disable remote |
| 3 | CONFIG_MISMATCH | fatal: log, disable remote |
| 4 | NO_SESSION | HELLO, then resync all live slots |
| 5 | STALE_SEQ | the frame was not applied: HELLO, resync |
| 6 | MALFORMED | round failed: resync the slots in that STEP; count it |
| 7 | CAPACITY | round failed (too many rows or slots): resync; count it |
| 8 | INTERNAL | round failed: resync; count it |
| 9 | UNSUPPORTED_OP | fatal: disable remote |

### 9.2 Per-request status (ReqStatus.status)
| code | name | meaning | client action |
|---|---|---|---|
| 0 | OK | drafts valid | use them |
| 1 | PARTIAL_WINDOW | drafts valid, window had fewer than expected rows | use them; count it; mark the slot dirty (resync next round) |
| 2 | STALE_GENERATION | entry ignored | zero drafts for this request this round; dirty. Rank 0 never sends an older generation (§7.1), so the worker's state does not match rank 0's: rank 0 starts a **new session** (a HELLO with a new session id drops every window, §6.2) and resyncs every live slot (v1.5) |
| 3 | BAD_ENTRY | entry rejected (positions or sizes) | zero drafts this round; dirty; log the entry |
| 4 | INTERNAL | the worker failed on this entry | zero drafts this round; dirty |

## 10. Failure policies (Spark side; `VLLM_DSPARK_REMOTE_POLICY`)

The per-round deadline is `VLLM_DSPARK_REMOTE_TIMEOUT_MS`, default 50 ms. A normal round is a few ms; one verify
step is 33–48 ms.

| policy | local drafter | on a failed round | memory |
|---|---|---|---|
| `capacity` | **not allocated**: no draft weights, graphs or draft KV layers | drafts = 0, confidences = 0.0. Depth policy `conf` then picks depth 0 (target-only cost). Counted. The circuit breaker below applies. `VLLM_DSPARK_REMOTE_ON_FAIL=raise` raises instead (engine-fatal; for benchmarks) | freed: drafter weights, Markov cache, draft graphs and workspace, draft SWA KV layers. Retained: 15 MiB ring + staging |
| `resilience` | kept, its context KV updated every round (eager precompute of the new rows) | that round runs the local drafter (the decision is broadcast, so both ranks run it); the circuit breaker below applies | nothing freed; the local drafter is retained and reported as not freed |
| `shadow` | kept, full local propose every round | n/a | remote and local proposals compared and logged (per-position agreement, first-mismatch histogram, margins). Local drafts are used unless `VLLM_DSPARK_REMOTE_SHADOW_USE=remote` |

**Circuit breaker (rank 0, every policy; v1.6).** A worker that answers but keeps failing would otherwise cost every
engine round its round trip (frame errors, per-request errors) or, after each reconnect, one full deadline
(timeouts). Rank 0 therefore counts consecutive **failed attempts**: rounds that sent a STEP and got a timeout, a
transport error, a refused reply (§6.3), a frame error, or per-request errors for every request that needed drafts.
- After `VLLM_DSPARK_REMOTE_TRIP` failed attempts in a row (default 3), the breaker opens. The next
  `VLLM_DSPARK_REMOTE_COOLDOWN` engine rounds (default 256) send nothing. They fail fast like a round without a
  connection: capacity zero drafts, resilience the local drafter, shadow no comparison. They are counted as
  `fallback_cooldown`.
- After the cooldown, the next round that sends a STEP *with a request that needs drafts* is a probe. Success closes
  the breaker; failure opens it again for another cooldown.
- The cooldown counts engine rounds, connected or not. These rounds are neither failures nor successes (v1.8):
  - rounds that send nothing (no connection, nothing to send, or a connection newer than the one the round's entries
    were built for);
  - rounds whose STEP succeeds but has no request that needs drafts (prefill appends only). They never exercise
    drafting, so they neither reset the count nor close a probe. If such a STEP fails, it is a failure.
- A success resets the count. A round where only some requests failed is a success.
- Rank 0 keeps its ring exact while the breaker is open. Every slot of a skipped round is marked dirty, so the probe
  resyncs those windows (§7.4).
- Transport failures also keep the client's wall-clock reconnect backoff: 1 s, doubling up to 30 s, reset by a
  successful HELLO.
- `VLLM_DSPARK_REMOTE_TRIP=0` turns the breaker off: every round tries again, as before v1.6.
- Steady-state cost with a worker that times out on every STEP: one 50 ms deadline per 257 rounds (≈ 9 s at
  35 ms/round, ≈ 0.5 %). Without the breaker it is one deadline per reconnect cycle (≈ 1 s, ≈ 5 %).

A failure in the middle of a request is safe in every policy:
- `capacity` only loses proposals for that round. The target's KV and the verifier are untouched, and the worker's
  window is rebuilt from rank 0's ring (§7.4).
- `resilience` has the local context KV in sync every round, so the local drafter can take over at any round
  boundary.

## 11. What the worker must match numerically (lane O2)
- Same tensors: the identity in §6.7. Same `main_proj` (EXL3 K5) and `main_norm`.
- Context KV computed exactly like `precompute_and_store_context_kv`: wkv from the fused `wq_a|wkv`, `kv_norm`,
  RoPE at the context positions. RoPE parameters are those of the drafter's attention layers (target-config RoPE;
  `compress_ratio = 0` layers).
- Query block: `[anchor, noise×4]`, non-causal inside the block, window 128.
- The draft MoE: 128 experts, top-3, plus the shared expert.
- mHC with hc_mult 4, ending in `hc_collapse` with the last layer's FFN pre-mix.
- Base logits = the target's head (`model-exl3-head`, EXL3 K6) over `norm(head_hidden)`.
- Sequential Markov bias from the previous sampled token, using the cached rows of `markov_top_tokens.json`
  (`markov_cache.py`) where they exist and the GEMM path elsewhere, like `markov_bias_for_ids`.
- Greedy argmax with lowest-index ties.
- Confidence from `cat(head_hidden, markov_embed)` per position.
- Representation changes are allowed only on the drafter side, and must be recorded (e.g. bf16 window KV instead of
  the Spark's fp8 record).

## 12. Optional extensions (reserved, not v1)
- `PREFIX_RESTORE` (op 6): content-addressed window restore, for configurations without SWA bounded replay.
- A split round, `STEP_CTX` + `STEP_ANCHOR`: send the taps before sampling finishes, so the worker can precompute
  context KV while the Spark samples. Worth ≈ the tap transfer time per round.
- Payload kind 2 (MAINX_BF16), if measurements reverse §3.
- Probabilistic drafting (draft probabilities over the 129,280-entry vocabulary are ~2.5 MB per request per round)
  is out of scope. The remote speculator refuses to start unless `draft_sample_method == "greedy"`, with no
  `dspark_draft_topk` and no watermarking.

## 13. Change log
| version | date | change |
|---|---|---|
| 1.0 | 2026-10-22 | first version, derived from the live proposer (lane O3) |
| 1.1 | 2026-10-22 | wire unchanged (`version` stays 1). §5: O1 `ofl` mapping, limits and ports (18620). §6.7: identity from Spark-side source hashes; the worker must tie its extracted files to them; live digests recorded. |
| 1.2 | 2026-10-22 | wire unchanged. §7.4: after a resync the window rows are exact, but drafts may differ at near-ties because KV is recomputed in another batch shape (lane O2). §8.2: rank-0 buffers corrected (45 MiB device, 15 MiB pinned). §4.2: async variant described. |
| 1.3 | 2026-10-22 | ReqEntry byte 3 = `n_out` (proposed by lane O2). Backward compatible both ways (old senders send 0 = all N; old workers ignore it), so the wire version stays 1. Positions ≥ n_out return zeros; positions < n_out equal the full-block draft. |
| 1.4 | 2026-10-22 | Wire format unchanged (`version` stays 1). §6.3: the Spark now refuses replies with values the target must not see (draft ids outside the vocabulary, confidences outside [0, 1] or not finite, bad margins, unknown status, window_rows out of bounds or inconsistent with the status, a nonzero n_out tail or nonzero drafts for unconsumed entries) as a failed round + resync (S5 #3). A worker that follows §6.3 and §7.2 is unaffected (lane O2's DraftCore passes O3's suites unchanged). |
| 1.5 | 2026-10-22 | Wire format unchanged. §7.2 step 1 made explicit: an older generation is STALE_GENERATION even with NEW_GEN (S5 #2; the reference worker exempted NEW_GEN, now fixed; lane O2 fixes its worker). §7.1: rank 0's slot generation changes only on a new owner, so it never sends an older one. §9.2: on STALE_GENERATION rank 0 starts a new session and resyncs, since a same-session resync would stay stale. |
| 1.6 | 2026-10-22 | Wire format unchanged. §10: the circuit breaker 1.0–1.5 promised (`VLLM_DSPARK_REMOTE_TRIP` 3, `VLLM_DSPARK_REMOTE_COOLDOWN` 256 rounds) is now implemented as specified (S5 #5: it did not exist; frame errors were retried every round). The text now states exactly which rounds are failed attempts, which are neutral, and how the probe and the resync work; "reconnect + HELLO" after the cooldown is replaced by "one probe round" (transport failures reconnect on their own backoff). |
| 1.7 | 2026-10-22 | Wire format unchanged; explanation corrected (S5 optional). §2 / §4.3 described the draft window with the causal per-token formula (127 context rows). The DSpark block kernel (`sparse_swa.py:1253-1255`) anchors one 128-row context window `[anchor_pos − 128, anchor_pos − 1]` to the whole block, plus the full query block. The rules themselves (W = 128 rows kept and sent, `expected = min(W, anchor_pos)`) were already 128-row; lane O2's worker matches the non-causal kernel. |
| 1.8 | 2026-10-22 | Wire format unchanged. §10 refined after an independent review of the 1.6 breaker: a successful STEP that carries no request needing drafts (prefill appends only) is neutral, so it cannot reset the failure count or close a probe without drafting having worked. A STEP whose connection is newer than the epoch its round was built for is refused on rank 0 (nothing sent, neutral) and the next round resyncs. |
| 1.8 note | 2026-10-23 | Documentation only; wire and behaviour unchanged, so the version stays 1.8 (it must equal `wire.CONTRACT_VERSION`, which is deployed Spark code; `spark/tests/test_wire.py` pins the two). Coordinator, assigned to lane D2-DATA. §2 tap validity: requests with `prompt_logprobs` keep every row in eager prefill (`model_state.py _keeps_rows_np`), so their taps are real on every row; the trim to the last min(q, 128) rows applies to the other prefills. The zero-row guard e0760ef (overlay `zero_rows`, refused downstream) is the backstop. |
