# AUD1 item 2: resident prefix reuse in production, before RP1

**Short answer (from source and earlier measurements; production itself not yet probed):**
- Automatic prefix caching is **on** in production. A hit covers the shared prefix rounded down to a **384-token**
  boundary (K1 layout). The engine then recomputes the hit's last 128 tokens to rebuild sliding-window KV, plus
  everything after the hit.
- The remote DSpark drafter **cannot** curtail or misalign target hits. It has no KV on the Sparks, draft rows are
  never cached, and its 128-row window is refilled by the same 128-token replay. That is the invariant SGLang #44673
  had to add.
- The DeepSeek V4.1 chat encoding is **byte-stable** for the same document under two different trailing questions,
  provided the question comes after the document and the reasoning setting, system text, tools, response format and
  cache salt are the same in both requests. The encoder inserts no date or random text.
- The GB10 window S18 already MEASURED this mechanism on the K1 layout, including a 200K-token partial hit. It has
  not been measured on today's exact production (21bed31: K1-327 + TP2 worker + async proxy). The probe below
  answers that with two requests.

Sources: image copy `/opt/dsv41/data/src-image/` (cited `path:line`); production 21bed31 overlays (K1 =
`offload/kvpack/overlay` at 559c5867, `remote_dspark`); earlier lanes S8 (`S8.report.md` N-4), S18
(`S18.report.md:68,71,93`), EV1 (`offload/results/COMPARISON.md:90`), CONTRACT (`offload/dspark/CONTRACT.md`).

## 1. Is automatic prefix caching on, at what granularity, and what does a hit cover?

| Question | Answer | Evidence |
|---|---|---|
| APC enabled? | Yes. `CacheConfig.enable_prefix_caching = True` by default. Production's launcher passes no `--no-enable-prefix-caching` and no `--prefix-caching-*` flag. None of the image's disable paths (RISC-V, encoder-only, non-causal layers) applies | `config/cache.py:140`; 21bed31 `offload/launch/dsv41_tp2_offload.sh:78-88`, `serve.sh` EXTRA; `engine/arg_utils.py:2987-3003`, `config/vllm.py:1413-1424`, `v1/engine/core.py:275-286`; RUNBOOK.md:108 "the prefix cache is on" |
| Hash | Chained SHA-256 over 64-token blocks (`prefix_caching_hash_algo="sha256"`), keyed per (hash, group) | `config/cache.py:142`; S8 N-4 |
| Groups that take part (K1, capacity, 6 GiB) | Groups (S8 table, GB10 log + engine allocator): 14 sliding-window groups of 64-token blocks, compressed groups of 128 and 192 tokens (K1 packs three 64-token pages into the 192 group), and the compressor ring (block 8). Only the compressed (MLA/indexer) groups are prefix-cacheable. The SWA groups are not, under `swa_bounded_replay=True`: `SlidingWindowMLASpec.prefix_cacheable = not bounded_replay`. The ring is not: `CircularBufferSpec.prefix_cacheable = False`. The coordinator skips both in hit lookup | `config/cache.py:240`; K1 overlay `v1/kv_cache_interface.py:975,1004`; `v1/core/kv_cache_coordinator.py:748`; `models/deepseek_v41/attention.py:529-578` |
| Hit granularity | Scheduler block = lcm(64, 128, 8, 192) = **384**. Hits are rounded down to 384 (no fine-grained partial hits: no Mamba groups). cached_tokens = floor((n−1)/384)×384 | `kv_cache_coordinator.py:702-735` (alignment = scheduler block unless Mamba partial hits); `single_type_kv_cache_manager.py:608` (`find_longest_cache_hit`); K1 README:61-64; S8 N-4. **MEASURED** S18 (K1 on GB10, port 8001): 35,712 / 99,840 / 249,600 cached for 36K/100K/250K prompts; partial hit (same first 200K tokens, different tail) 199,680; 500K 499,584; fresh-salt controls 0; hit-vs-cold output within the engine's floor |
| What a hit covers | A hit of h tokens (h on a 384 boundary) reuses the compressed and indexer KV for [0, h). The scheduler rewinds by 128 tokens (`replay_start = h − 128`) and recomputes them so the sliding-window KV (and the drafter's taps) are rebuilt; their cached KV is left alone. Everything from h onward is computed. Hits of 128 tokens or fewer are recomputed in full. Uncomputed work after a hit: (n − h) + 128 tokens, at most 383 + 128 + question tokens | `v1/core/sched/scheduler.py:1100-1123, 3123-3135`; `config/cache.py:240-245`; CONTRACT.md:44 |
| Counters | `vllm:prefix_cache_queries` (+ request tokens per admitted lookup), `vllm:prefix_cache_hits` (+ local hit tokens, recorded at admission before the replay rewind), `vllm:prompt_tokens_cached`, usage `prompt_tokens_details.cached_tokens` (`--enable-prompt-tokens-details` is on in production) | `v1/metrics/loggers.py:636-652, 747`; `v1/core/kv_cache_manager.py:253-262`; `sched/scheduler.py:1298-1301`; 21bed31 `serve.sh` EXTRA |
| Pool | 2,502,600 tokens (K1 at 6 GiB/rank, MEASURED S8/S18). Freed blocks stay cached until LRU eviction. A 150K document stays resident while live plus cached blocks fit. Under pressure it is evicted: S18 R1b, a 300K A-B-A under pool pressure, cached only 59,136 of 299,904 the second time | S18.report.md:71 |

Earlier MEASURED reuse on the pre-K1 layout: EV1 36,016-token repeat → 35,968 cached under both shadow and capacity
drafting (`COMPARISON.md:90`), i.e. the 128-token v5 granularity.

## 2. Can the remote DSpark draft window curtail or misalign target prefix hits? (SGLang #44673's invariant)

SGLang #44673 (merged 2026-10-16) fixed this: with encoder SWA bounded replay the target kept SWA out of its radix
tree, but the DSpark draft's paged SWA still put an SWA component in the tree. Evicting the draft's SWA then cut hits
on Main KV the target could still reuse ("re-hits 8448 of its 8565 tokens instead of 0", UPSTREAM CLAIM). The fix
moves the draft SWA into a per-request ring that starts at the hit.

Production already satisfies the invariant, for four reasons:
- **No draft KV on the Sparks.** The capacity speculator (`VLLM_DSPARK_REMOTE_POLICY=capacity`) loads no draft model
  and allocates no draft KV groups. S8's allocator table shows 14 SWA groups in capacity mode against 15 with the
  local drafter (`S8.report.md:166-168`; K1 README:61). Nothing draft-only exists that could gate or evict a target
  hit. The target's own SWA groups are kept out of hit lookup (section 1).
- **Draft rows are never cached.** `allocate_slots` caches at most `request.num_tokens`, the finalized tokens; rows
  of unverified drafts are excluded (`kv_cache_manager.py:603-612`). Hashes cover committed tokens only, so the draft
  depth (k = 5/4, 0047 window) cannot shift a hit's start or alignment. Hits always end on committed 384-token
  boundaries.
- **The drafter's context after a hit is rebuilt by the same replay.** The x86host worker attends to exactly the 128
  context rows before the anchor (CONTRACT.md:39, 108-110). The target recomputes the hit's last 128 tokens, and
  their taps reach the worker like any prefill rows (CONTRACT.md:44, 111-114). After a hit or re-add at a lower
  position, both sides drop rows at positions ≥ `first_pos` (CONTRACT.md:321-322). So the worker never reads rows
  left by an earlier request in the slot. This is the equivalent of #44673's `swa_req_ring_valid_from`.
- **A failed or late remote round changes drafts only** (zero drafts, target-only rounds), never cache state.

Residual cost, not a curtailment: a hit gives the drafter only its 128-row window, which is all it reads. No
acceptance loss follows from the hit itself.

## 3. Is the server's chat encoding byte-stable for the same document under two questions?

Path: `DeepseekV41ForCausalLM` → tokenizer mode `deepseek_v41` (`config/model.py:742-743`; base config architectures
= `DeepseekV41ForCausalLM`, `/opt/dsv41/data/ref-ds41/config.json`) → `tokenizers/deepseek_v41.py:57-103` →
`tokenizers/deepseek_v41_encoding.py` `encode_messages` (`:547-591`). The prompt is:

`BOS` + [thinking mode only: system token + `Reasoning Effort: N (range 1-100, …)\n\n`] + [system message text +
`## Tools …` + response-format schema] + `<｜User｜>` + user content + `<｜Assistant｜>` + `<think>` or `</think>`

| Item | Stable for the same document with two questions? | Source |
|---|---|---|
| Document then question in one user message | Yes: the shared bytes run from BOS to the end of the document and separator; reuse = that, rounded down to 384. Only a BPE boundary token can differ | `:323-420`, `user_msg_template = "{content}"` |
| Question before the document | No reuse past the question | same |
| System text | Rendered verbatim at index 0. Identical text → identical bytes; any change breaks everything after it | `:367-371` |
| Reasoning setting | Thinking on adds the effort line at the very start. thinking=false (or `reasoning_effort: "none"`) omits it. Mixing on/off, or low (50) / high (75) / max (100) between the two requests, gives **zero** reuse. `high` and `xhigh` both map to 75 and render identically | `:186-217`, `deepseek_v41.py:73-87` |
| Tools | Placed in the system message (a system message is created if absent), each tool as `json.dumps` with insertion order (`to_json`, `:49-54`) and no sorting. Same tool JSON → same bytes; adding, removing or reordering a tool breaks reuse after BOS. Tools also turn `drop_thinking` off | `deepseek_v41.py:64-69`; `:249-262, 573-576` |
| response_format | JSON schema in the system block: same rule as tools | `:374-377` |
| Dates, times, random ids | None: the encoder adds no date, time or nonce | whole file (no date/time use) |
| Multi-turn | In thinking mode earlier assistant reasoning is dropped (`drop_thinking=True`). A follow-up turn's prompt is not "turn-1 prompt + turn-1 output", so reuse covers turn 1's prompt only | `:525-545, 579-591` |
| cache_salt | Mixed into the first block's hash only, which changes every chained hash. Same salt (or none) shares; a different salt shares nothing | K1 overlay `v1/core/kv_cache_utils.py:637-640` |
| Text-part lists | Parts joined with `\n\n` in order: deterministic | `deepseek_v41.py:21-45` |

Practical rule for reuse: one fixed system prompt and tool list, the same `chat_template_kwargs`, the document first
and the question last, and the same salt (or none).

## 4. The probe the coordinator runs: `offload/audit/prefix_reuse_probe.py` (+ `.sh`)

Dry by default (`python3 offload/audit/prefix_reuse_probe.py` prints the plan and sends nothing).
`--execute --base-url URL --out FILE` sends **exactly two** sequential streaming chat requests:
- the same document: the pinned public corpus `opinions_cl.jsonl` (sha256 `80461203…b6a1`), records in file order up
  to 525,000 chars, about 150K tokens. Document sha256 `5133a34a…f160fa`;
- two different short questions after the document;
- `max_tokens` 8, thinking off, T=1/top_p=1, priority 10 (background class);
- one fresh salt shared by both, so request 1 is cold whatever ran before.

Before each request it reads `/metrics` and refuses unless `num_requests_running` and `num_requests_waiting` are
both 0 (COMMON rule 2). It reads `/metrics` again after each request, and touches no other endpoint. It records
numbers and hashes only:
- prompt, cached and completion tokens; client TTFT and total time;
- deltas of `prefix_cache_queries/hits`, `prompt_tokens(_cached)`, external prefix cache, preemptions, server TTFT
  sum/count and prefill KV-computed tokens;
- sha256 of corpus, document, prompts, salt and the generated text.

CPU tests (`offload/audit/tests/test_prefix_reuse_probe.py`, loopback fake server; 12 pass) cover:
- the dry run sends nothing;
- exactly 2 POSTs plus 4 `/metrics` GETs;
- reuse-works and no-cache verdicts;
- the replay-window gap between the metrics hit counter and usage;
- a busy server refused before any POST;
- other traffic flagged CONTAMINATED;
- a failed first request stops before the second;
- the corpus pin, no overwrite, and a check that no document, question or generated text appears in the output.

Coordinator run (production idle, nothing else in a window; about 75-90 s, mostly the cold 150K prefill; the cold
request holds ~150K tokens of the 2.5M pool until evicted):
```bash
# from a host with the corpus (Controller: default path; sparkhead: --data ~/dsv41-deploy/data/opinions_cl.jsonl)
offload/audit/prefix_reuse_probe.sh --execute --base-url http://HEAD_IP:8000 \
    --out /opt/dsv41/campaigns/DSV41-OFFLOAD/AUD1-prefix-probe-$(date -u +%Y%m%dT%H%M%SZ).json   (absolute: the wrapper cds first)
```

### What the result means

- **RESIDENT_REUSE_WORKS**: all of
  - request 2's cached_tokens ≥ floor((min(prompt₁, prompt₂) − 96)/384)×384 − 384;
  - cached_tokens is a multiple of 384;
  - the `/metrics` hit delta agrees within the 128-token replay window;
  - TTFT₂ ≤ 0.2 × TTFT₁.

  ESTIMATED expectation: cached ≈ prompt − 400 to 600; TTFT₂ about 1 s against a cold TTFT₁ of ~65-75 s (RUNBOOK
  ~2,000-2,300 tok/s prefill). Then resident reuse works today for "same document, new question" and needs no new
  code.
- **RESIDENT_REUSE_FAILS**: request 2 cached less than half of the shared prefix. Read with sections 1-3: APC off on
  the running container, encoding drift between the two prompts (the hashes in the output let the coordinator
  compare), the K1 hit path, or eviction (check `num_preemptions`, KV usage). RP1 would not fix an encoding or
  hit-path cause, because its tiers sit behind the same hashes and lookup.
- **PARTIAL**: some reuse but short of the shared prefix, or no TTFT drop. Inspect the cached count against the 384
  rule and the replay gap.
- **NOT_COLD / CONTAMINATED / INCOMPLETE**: the run is void (salt collision or stale state / other traffic changed
  the counters / a request failed). Rerun when idle.

### What RP1 would still add if resident reuse works

Resident reuse lasts only while the blocks stay in the GPU pool. RP1 adds what the pool cannot:
- **Persistence across restarts and windows.** Every production restart, promotion or experiment window empties
  the pool. RP1 Stage A's file tier is designed to survive controlled restarts (`RP1-PREFIX-STORE.report.md`, GB10
  gates P0-P2 not yet run).
- **Capacity beyond the 2.5M-token pool under concurrency.** Four 327K contexts can evict a cached document (S18
  R1b). As built, Stage A's 256 MiB CPU tier caps a reload at 29,568 tokens per request, and a full 150K reload needs
  about 1.3 GiB per rank (RP1 report). Stage B's x86host RAM tier is a CPU prototype only.

RP1 adds neither faster reuse within a session nor a fix for encoding drift.
