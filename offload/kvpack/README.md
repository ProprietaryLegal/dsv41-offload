Status 2026-10-23: S8R2 fixes merged; W6/W6b under coordinator, S18 final pending. Current candidate is 524,288 / expected pool 2,820,756; historical 1M tests below are dropped by the evening amendment. See [live roadmap](../../documents/plans/roadmap.md) and [measured record](../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# K1: KV page packing and head-node memory for the DSV41 serve (overlay on image v5)

Lane K1, 2026-10-22. Donor: github.com/HeNryous/deepseek-v41-1017-dgx-spark @ afe4d5e41ec6fa4db17c5cc88758af439a106913,
patches 0051 → 0053 → 0057 → 0059 (code only) → 0060 → 0076 applied to the python tree of image v5
(`/opt/dsv41/data/src-image`). Every switch defaults to off. Mounted with no switch, the overlay runs the v5 code
paths (test R0 in `GB10-TEST-PLAN.md` proves this on GB10).

## What it buys (arithmetic from the v5 allocator; `kv_arith.py`, exact against 5 known figures)

| D-kv6 (capacity mode, KV 6 GiB/rank) | v5 | 0057 | 0060 (`K1_KV_PAIR=3`) |
|---|---:|---:|---:|
| tokens at max-model-len 327,680 | 1,604,926 (4.90 ×) | 2,195,668 (6.70 ×) | **2,502,600 (7.64 ×)** |
| tokens at max-model-len 1,048,576 | 1,850,243 (1.76 ×) | 2,682,188 (2.56 ×) | **3,154,934 (3.01 ×)** |
| bytes charged per token (327,680) | 4,014 | 2,934 | 2,574 |

Same pool bytes, so no memory change. Equivalently, D-kv6's 1,604,926 tokens fit in about 3.85 GiB/rank with packing,
which frees about 2.15 GiB/rank. MAX_SEQS stays 4, so beyond 4 × max-model-len the pool buys headroom and nothing more.

Head-node memory (0076 + config). Per rank, against today's D-kv6 at 327,680 (corrected after review S8, P-2):

| item | switch | effect |
|---|---|---|
| MTP residual buffer | `K1_DROP_MTP=1` (0076b) | −160 MiB never allocated (never read by DSpark; capacity mode reads taps only) |
| TensorFold prompt scratch | `K1_TF_TRANSIENT=1` (0076d) | our serve runs `VLLM_EXL3_MOE_TF_PROMPT=1`, so 1,014 MiB (xd 56.6 + z 503.3 + xg/xu 503.3 MB at 4,096 rows, top-6, D 5,120, I 1,152) is allocated from startup (the profile run at 4,096 rows) and held. With 0076d it is allocated per MoE call and handed back to torch's caching allocator (expandable segments, no `empty_cache`), where other phases' temporaries reuse it. That lowers the peak; it does **not** return the memory to the system, so MemAvailable gains far less than 1,014 MiB |
| down partial aliasing | `K1_TF_ALIAS=1` (0076e, with d) | call peak 1,014 → 750 MiB |
| indexer prefill buffer | `K1_INDEXER_FACTOR=4` (config) | rows = max_model_len × factor × 132 B: 330 → 165 MiB at 327,680; 1,056 → 528 MiB at 1M |
| RoPE caches at 1M | (0033a cap = max_model_len + 2048, two caches) | +0.34 GiB at 1M against 327,680 |

What the whole set (factor 4 + 0076b + 0076d + 0076e) is worth, from upstream's own A/B (0076 header, image v23,
max-model-len 524,288): **+714 MiB free after start, +813 MiB at the lowest point** under a 478K needle. Of that,
factor 4 is 264 MiB at 524,288 and 0076b 160 MiB, which leaves about 290–390 MiB for 0076d/e. For us:
- at 327,680: about **+0.6–0.7 GiB per rank** (factor 4 is worth only 165 MiB there);
- at 1M against today's 327,680 serve: RoPE +0.34 GiB and the indexer buffer 330 → 528 MiB are spent; 0076b and
  0076d/e give back about 0.45–0.55 GiB. Net: **about break-even (0 to −0.1 GiB)**. Without the switches it is
  about **−1.05 GiB**.
- My first estimate (+1.2 GiB freed, +0.6 GiB net at 1M) counted 0076d's 1,014 MiB as returned to the system; it is
  about 0.5 GiB too optimistic.

Upstream: "this is what makes max-model-len 1,048,576 fit"; its serve was killed at 1,333 MiB free without the
switches. R3/R4 may still approach the plan's 1.5 GiB MemAvailable stop line during 900K prefills; the stop condition
protects them.

## Files

| file | what |
|---|---|
| `overlay/` | the 13 patched files, container paths under `/usr/local/lib/python3.12/dist-packages/` |
| `MANIFEST.json` | donor sha, patch sha256s, base (v5) and overlay sha256 per file, switches |
| `build_overlay.sh` | rebuilds the overlay from v5 + donor; verifies (default) or `--write`s |
| `stage_overlay.sh` | `verify-checkout` (this checkout's overlay == MANIFEST), `check` (the image on a node is the overlay's base; `--pull never`), `stage` (versioned node-local copy, verified), `verify-running` (the running container serves the reviewed bytes; live K1 switches, `--expect`), `args` (production docker `-v`), `overlays` (start_exp pairs) |
| `make_configs.py` | writes / checks `offload/launch/configs/D-kv6-async-1m-k1mem.env`, `D-kv6-async-pack.env`, `D-kv6-async-pack-1m.env` (base `D-kv6-async-process-nap-lazy.env`, the async production candidate the GB10 plan runs) and the sync control `D-kv6-1m-k1mem.env`, `D-kv6-pack.env`, `D-kv6-pack-1m.env` (base `D-cal1-kv6.env`); `D-kv6-1m.env` stays the coordinator's plain 1M run |
| `kv_arith.py` | the pool arithmetic with its five checks |
| `needle.py` | needle recall and cross-request corruption (foreign codes); nominal ctx can undershoot, use saved server prompt usage |
| `prefix_identity.py` | prefix-cache hit vs cold output identity (T=0, top-20 logprobs, fixed salt), partial hits, A-B-A under pool pressure |
| `test_stage_overlay.sh` | 27 offline tests (staging, tamper refusals, verify modes, `--pull never`, rebuild, arithmetic incl. the pressure run, configs, memprobe guard, prefix-identity self-test, switch and KV-bytes defaults) |
| `GB10-TEST-PLAN.md` | the coordinator's runs, one row per switch, stop conditions, promotion and rollback |

## Compatibility with our other overlays

- **O3 remote DSpark, capacity mode** (D configs). The chain does not touch `_annotate_eagle_groups` or the
  `spec_decode` overlay. Capacity mode has no draft KV groups; packing changes only the ratio-1 compressed group (block
  64 → 192 tokens), so the scheduler block becomes lcm(64, 128, 8, 192) = 384 (today 128).
  - Prefix hits land on 384-token boundaries: up to 383 tokens re-prefilled per hit (today 127). The 64-token hashes
    are unchanged; the 128/192-token groups look them up through the last hash of each block (S8 N-4).
  - The EAGLE trailing-block drop is a no-op in our serve: the positional fallback flags the group of the last
    registered layer, an SWA group, and with `swa_bounded_replay` on SWA groups are not prefix-cacheable, so the
    coordinator never looks that group up (S8 N-4; capacity and shadow both cached 35,968). `K1_NODROP` therefore
    stays off in every config; it is kept only as a switch.
  - `remote_dspark`'s only block-size use is its local draft-KV path (`kernel_block_sizes` of the draft SWA groups,
    block 64, unchanged).
  - 0076b: capacity `propose` ignores `last_hidden_states`. The local-backed modes use the aux taps.
  - The one mechanism packing changes on the hit path is checked on GB10 by `prefix_identity.py` (hit vs cold, T=0).
- **O3 shadow / resilience** (local draft KV kept): draft groups are SWA (block 64) and unchanged. Upstream ran 0057/0060 with the local DSpark drafter (T=0 8/8, needles 6/6).
- **O1 remote Engram** (`engram.py` via sitecustomize): not in the chain, no shared file. 0076a's Engram fields are
  valid under B: `EngramRemoteTable.spin_stats` counts the remote reads (S8 N-7; my first note said otherwise).
- **Ungated hunk:** 0059's state-bucket split in `kv_cache_utils.py` is not switch-gated, but it is a provable no-op for
  V4.1 whatever the record format (S8 N-5). V4.1's ring bucket holds three compressor state caches of one page size
  (32 KiB: 8 rows × 1,024 fp32, layers 2/8/14; the V4.1 indexer has no compressor of its own), so the new branch's
  condition `balanced and len(page_size_layers) > 1` is false and the else branch is the v5 code verbatim. (My first
  explanation, "six rings of 32 + 8 KiB", repeated upstream's comment and does not match V4.1.) S8's offline run of
  the real allocator gives one ring group and results identical to v5 in all eight off-switch configurations; R0
  checks it on GB10.
- **NVFP4** (0059/0061) stays off: rejected upstream at the KV-KLD gate (KL 0.028, top-1 95.0 %). Its code is carried
  only because 0060 sits on it. Its FlashInfer imports are lazy, so they are never loaded with the switch off.
- `vllm_exl3_moe/display_kv.py` (0053) is not deployed; it is imported only with `VLLM_DSV41_DISPLAY_KV_MIB`.

## EV1 actual-count audit (2026-10-23)

[Saved R5 prompt counts](../evidence/K1-TOKEN-COUNTS.md) distinguish nominal needle/prefix labels from actual
chat tokens. R5-c500 already has three literal 500,000-token prompts (matching `/tokenize` and usage), each
501,500 total with 1,500 reasoning and zero visible-answer tokens. Literal occupancy is covered; completed-answer
usefulness is unproven. The partial-prefix full prompt lengths were not saved. 524,288 remains HOLD.
