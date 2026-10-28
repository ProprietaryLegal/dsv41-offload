# K1 GB10 test plan: KV page packing (0060) and the 0076 memory switches on D-kv6

Coordinator-only, inside an open window (`offload/launch/window.sh open`). Every start uses
`offload/launch/start_exp.sh <config>` on sparkhead; every measurement runs from Controller against
`http://HEAD_IP:8001`. Predictions come from `offload/kvpack/kv_arith.py`. That script reproduces 989,460 /
1,003,083 / 1,604,926 from today's GB10 logs and upstream's 1,346,928 / 1,531,258 exactly.

## 0. Before the first start

| step | command | pass |
|---|---|---|
| checkout provenance (on sparkhead, `~/dsv41-deploy-offload`: start_exp copies the overlay from there) | `git merge-base --is-ancestor 919adeb HEAD && offload/kvpack/stage_overlay.sh verify-checkout` (919adeb = offload/kvpack's tip, merged into offload-x86host as 5993820) | ancestor; "overlay == MANIFEST (559c5867ba42), 13 files" |
| image base | `offload/kvpack/stage_overlay.sh check spark sparkworker` (`docker run --pull never`: a missing tag fails, it never pulls) | both nodes: "image matches the overlay base (11 files, 2 new)" |
| overlay = rebuild (on **Controller**: SRC and DONOR exist only there, so on sparkhead `build_overlay.sh` stops with "donor is not at …") | `offload/kvpack/build_overlay.sh` and `offload/kvpack/test_stage_overlay.sh` | "verified: 13 overlay files and MANIFEST.json match a fresh rebuild", "27 tests passed" |
| 1M only: x86host worker | none: the live `dsv41-dspark-worker` unit already runs `--max-model-len 1048576` (checked 2026-10-23), and so does the TP2 production unit; do not restart it | HELLO from a 1,048,576 engine accepted (`hellos` +1, no refusal in the worker log) |
| memory logger | 5-s MemAvailable / SwapFree logger on both Sparks for every run below, from Controller (command below) | running |

Memory logger, from Controller in the checkout (one per run; `offload/results/K1/` exists only after the first suite run):

```
mkdir -p offload/results/K1
while sleep 5; do for n in sparkhead sparkworker-wired; do printf '%s %s ' "$(date -u +%T)" $n; ssh -o BatchMode=yes $n "awk '/MemAvailable|SwapFree/{printf \"%s %s \", \$1, \$2}' /proc/meminfo"; echo; done; done | tee offload/results/K1/mem-<run>.log
```

**After every start, before any measurement:** `offload/kvpack/stage_overlay.sh verify-running --expect "<the run's K1
switches>" spark sparkworker`. It checks two things on both nodes:
- `docker exec` sha256 of the 13 mounted paths equals MANIFEST `overlay_sha256` (the reviewed bytes are what runs);
- the K1 switches in the container environment are exactly the run's set (column "K1 switches" in §1; R0: `--expect ""`).
Production's own `VLLM_DSV41_AK_*` / `VLLM_DSV41_MHC_*` variables are not K1 switches and are not counted. Any mismatch:
stop the run and fix the checkout or the start.

Stop everything and restart production (`./stop.sh && ./start.sh`) if MemAvailable on either Spark falls below
**1.5 GiB**, or swap use grows by more than 1 GiB during a run. Upstream's watchdog killed its 1M serve at 1,333 MiB free
when the 0076 switches were off.

## 1. Runs, in order (about 6 min per start)

The column "K1 switches" is the exact `--expect` set for `stage_overlay.sh verify-running` after that start. Timing comes
only from starts without memprobe (R0, R1, R3, R4). Memory comes from R1p vs R2p, which are both probed (S8 R-4b).
`K1_NODROP` stays off everywhere: it has no effect in our serve (S8 N-4).

Base (S8R2 FIX-2): every row runs on the async production candidate. `make_configs.py` generates `D-kv6-async-pack.env`,
`D-kv6-async-pack-1m.env` and `D-kv6-async-1m-k1mem.env` from `D-kv6-async-process-nap-lazy.env` (window 4; R3's TP2 +
async package uses the same settings). Sourced as `start_exp.sh` sources them, they differ from the sync K1 configs only in
the base's five async variables (`VLLM_DSPARK_REMOTE_ASYNC=1 VLLM_DSPARK_REMOTE_ASYNC_PROXY=process
VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS=5000 VLLM_DSPARK_REMOTE_ASYNC_SPIN_US=50 VLLM_DSPARK_REMOTE_ASYNC_NAP_US=20`). The overlay
and the async drafter share no files or code paths, the KV lines do not depend on async, and R0 is the same-base control.
The sync configs `D-kv6-pack.env`, `D-kv6-pack-1m.env` and `D-kv6-1m-k1mem.env` (base D-cal1-kv6) stay generated as the
sync control: the same start lines without `-async` give the same K1 switch sets.

| run | start (sparkhead, `~/dsv41-deploy-offload`) | K1 switches | purpose |
|---|---|---|---|
| R0 | `K1_KV_PAIR= K1_DROP_MTP= K1_TF_TRANSIENT= K1_TF_ALIAS= K1_INDEXER_FACTOR= offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack.env` | `""` | Overlay mounted with every switch off: must equal its base D-kv6-async-process-nap-lazy (inertness). Also the v5-layout floor for the prefix-identity and needle rows. |
| R1 | `offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack.env` (defaults, no memprobe) | `VLLM_DSV41_KV_PAIR=3 VLLM_DSV41_DROP_MTP_BUFFER=1 VLLM_EXL3_TF_PROMPT_TRANSIENT=1 VLLM_EXL3_TF_PROMPT_ALIAS=1 VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4` | Packing + 0076 at 327,680: the timing gate and the correctness rows. |
| R1p | `K1_MEMPROBE=/root/.cache/dsv41-memprobe-R1p offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack.env` | `VLLM_DSV41_KV_PAIR=3 VLLM_DSV41_DROP_MTP_BUFFER=1 VLLM_EXL3_TF_PROMPT_TRANSIENT=1 VLLM_EXL3_TF_PROMPT_ALIAS=1 VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4 VLLM_DSV41_MEMPROBE=/root/.cache/dsv41-memprobe-R1p VLLM_DSV41_MEMPROBE_LAYERS=2,14,20` | Memory with every switch on (probed, never timed). |
| R2p | `K1_MEMPROBE=/root/.cache/dsv41-memprobe-R2p K1_DROP_MTP= K1_TF_TRANSIENT= K1_TF_ALIAS= K1_INDEXER_FACTOR= offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack.env` | `VLLM_DSV41_KV_PAIR=3 VLLM_DSV41_MEMPROBE=/root/.cache/dsv41-memprobe-R2p VLLM_DSV41_MEMPROBE_LAYERS=2,14,20` | Memory with the 0076 + factor switches off: R1p − R2p is their share. |
| R1b | `K1_KV_BYTES=1610612736 offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack.env` | `VLLM_DSV41_KV_PAIR=3 VLLM_DSV41_DROP_MTP_BUFFER=1 VLLM_EXL3_TF_PROMPT_TRANSIENT=1 VLLM_EXL3_TF_PROMPT_ALIAS=1 VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4` | Pool pressure, packed (S8 P-1): 1.5 GiB/rank = 11,650 blocks = 625,609 tokens. Forces preemption and prefix-block eviction. |
| R1b-v5 | `K1_KV_BYTES=1610612736 K1_KV_PAIR= offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack.env` | `VLLM_DSV41_DROP_MTP_BUFFER=1 VLLM_EXL3_TF_PROMPT_TRANSIENT=1 VLLM_EXL3_TF_PROMPT_ALIAS=1 VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4` | The same pressure in the v5 layout (401,205 tokens): R1b's floor. |
| R3 | `offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-1m-k1mem.env` | `VLLM_DSV41_DROP_MTP_BUFFER=1 VLLM_EXL3_TF_PROMPT_TRANSIENT=1 VLLM_EXL3_TF_PROMPT_ALIAS=1 VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4` | The 1M cap with 0076 + factor 4, no packing. Memory from the 5-s logger. |
| R4 | `offload/launch/start_exp.sh offload/launch/configs/D-kv6-async-pack-1m.env` | `VLLM_DSV41_KV_PAIR=3 VLLM_DSV41_DROP_MTP_BUFFER=1 VLLM_EXL3_TF_PROMPT_TRANSIENT=1 VLLM_EXL3_TF_PROMPT_ALIAS=1 VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4` | The finalist: 1M + packing. |

Optional: R3 with `K1_INDEXER_FACTOR=` (expect `VLLM_DSV41_DROP_MTP_BUFFER=1 VLLM_EXL3_TF_PROMPT_TRANSIENT=1 VLLM_EXL3_TF_PROMPT_ALIAS=1`) gives the
indexer row at 1M. Only run it if MemAvailable in R3 stayed at 3 GiB or more.

## 2. Measurements and pass criteria, one row per switch

| switch (variable) | run(s) | measure | expected | stop / fail |
|---|---|---|---|---|
| overlay mounted, all off | R0 vs the base's log (D-kv6-async-process-nap-lazy) | `GPU KV cache size`, `kv cache group sizes`, `kv lcm block sizes`, Model loading GiB, MemAvailable; `bench/suite.sh R0 … short c32`; `logit_ab.py run --label R0` against a D-kv6-async-process-nap-lazy run; `prefix_identity.py --block 128 --n 36016 100017 250000 --partial 250000:200000` (the floor for R1/R4) | 1,604,926 tokens; groups `[64×14, 128, 8, 64]`; lcm 128; ms/round within ±1 % of the base; greedy within the engine's A-A floor; prefix rows PASS with `cached` = floor((n − 1)/128) × 128 | any difference in the KV lines → the overlay is not inert: do not continue |
| `K1_KV_PAIR=3` (0057 + 0060) | R1 (timed) | KV lines; live bytes/token on a 150K prompt (`vllm:kv_cache_usage_perc` × pool ÷ prompt tokens); suite `short c32 c150` ms/round at 1 stream and the c32 3-stream group, against R0; `logit_ab.py` greedy against R0 | **2,502,600 tokens** (7.64 × 327,680); groups end in `192`; lcm 384; ms/round within +1 % of R0 (upstream −0.51 %); greedy identical to R0 within the floor | KV figure ≠ 2,502,600; any T=0 change beyond the floor; ms/round > +1 % |
| packing: prefix-hit identity (S8 R-4a) | R1 and R4; R0 = floor | `prefix_identity.py --url … --block 384 --n 36016 100017 250000 --partial 250000:200000 --out offload/results/K1/prefix-<run>.jsonl`; R4: `--n 250000 --partial 500000:400000`. Fixed salt per prompt: cold, then the hit, then a fresh-salt control | cold `cached` 0; hit `cached` = floor((prompt_tokens − 1)/384) × 384 (e.g. 35,712 where R0 gives 35,968); partial hit = the shared tokens rounded down to 384; hit text and top-20 logprobs PASS (identical to cold or no further apart than cold vs control) | FAIL (any `cached_tokens` off); a FLAG that R0's same row does not also show |
| packing: cross-request corruption | R1 and R4; R0 = floor | `needle.py --ctx 8000 30000 50000 100000 --concurrent 4 --needles 6` | 16/16 requests 6/6, **no foreign code** | any miss or foreign code: stop, back to R0 |
| pool pressure: preemption and eviction (S8 P-1) | R1b, R1b-v5 (do not go below 1.5 GiB: one 327,680-token request needs 9,515 v5 blocks) | KV line; `needle.py --ctx 240000 --concurrent 4 --reps 2 --out offload/results/K1/needle-<run>.jsonl` (4 × 240K: the full-attention blocks alone need 12,500 packed / 22,500 v5 blocks > 11,650), R1b-v5 with `--reps 1`; `prefix_identity.py --order ABA --fillers 2 --n 300000 --block 384` (R1b-v5: `--block 128`), i.e. A, B, C, then A again; both tools print the `vllm:num_preemptions` / prefix-cache counter deltas | R1b 625,609 / R1b-v5 401,205 tokens; preemptions > 0 in the needle run; every request 6/6, no foreign code; A's repeat `evicted: true` (cached < the full hit) and its output PASS against A's cold run | any miss, foreign code, FAIL; zero preemptions (the run did not exercise the path: raise `--concurrent` or lengths, never lower the KV below 1.5 GiB) |
| packing: needles | R1: `--ctx 250000 --reps 2`; R4: `--ctx 250000 500000 900000` | recall, TTFT | 6/6 at every length (upstream: 250K 6/6, 948K 3/3) | any miss at ≤ 250K; at ≥ 500K, record and compare with R3 |
| `K1_NODROP` | none required | (S8 N-4) with `swa_bounded_replay` on, SWA groups are not prefix-cacheable, so the group the positional fallback flags is never looked up; capacity and shadow both cached 35,968 | no effect; stays off | — |
| `K1_DROP_MTP=1` (0076b) | R1p vs R2p | worker log line `dsv41 0076b: no MTP residual buffer`; memprobe `allocated` at idle | −160 MiB/rank (4,096 × 20,480 bf16); greedy identical (R1 vs R0) | any T=0 change |
| `K1_TF_TRANSIENT=1` (0076d) | R1p vs R2p, memprobe `step` records | torch allocated at idle after startup, and between prefill steps of a 150K prompt | R2p: the 1,014 MiB/rank prompt scratch is allocated already after startup (the profile run at 4,096 rows). R1p: torch *allocated* lower by up to 1,014 MiB between calls, but the memory stays in torch's caching allocator (reserved), so MemAvailable rises far less. Whole set R1p − R2p: about +0.6–0.7 GiB MemAvailable per rank (upstream, factor 4 + b + d + e at 524,288: +714 MiB after start, +813 MiB lowest) | greedy change (R1 vs R0); prefill time > +1 % (R1 vs R0) |
| `K1_TF_ALIAS=1` (0076e, with 0076d) | R1p (layers 2,14,20 set by the config); optional R1p variant with `K1_TF_ALIAS=` | memprobe `phase` ffn peak | call peak 1,014 → 750 MiB/rank | greedy change |
| `K1_INDEXER_FACTOR=4` | R1p vs R2p (327,680); optional R3 variant (1M) | allocated at startup; prefill time at 150K (R1 vs R0: chunking must not change) | 330 → 165 MiB/rank at 327,680; 1,056 → 528 MiB/rank at 1M; prefill ±1 % | prefill > +1 % |
| `MAX_LEN_OVERRIDE=1048576` | R3 (D-kv6-async-1m-k1mem), R4 | KV lines; MemAvailable idle and lowest during a 900K needle; prefill time at 500K / 900K; remote-drafter stats (positions past 327,680 served, 0 failures) | R3: **1,850,243 tokens** (1.76 × 1M); R4: **3,154,934 tokens** (3.01 × 1M); RoPE caches +0.34 GiB/rank against 327,680. Free memory with all switches on: about the same as today's 327,680 serve (0 to −0.1 GiB); a 900K prefill may approach the 1.5 GiB stop line | MemAvailable < 1.5 GiB; any drafter failure at positions > 327,680 |
| `K1_MEMPROBE` (0076a) | R1p, R2p only | `~/.cache/dsv41-serve/dsv41-memprobe-<run>_r0.jsonl` on sparkhead, `_r1` on sparkworker (no directory is created: the prefix sits directly in `/root/.cache`; the configs refuse anything else); `VLLM_DSV41_MEMPROBE_LAYERS=2,14,20` comes with it | step and phase records; its Engram fields include O1's remote reads (`EngramRemoteTable.spin_stats`) | — (never in a timed run) |

Suites (timed starts only): R0 and R1 `offload/bench/suite.sh <run> http://HEAD_IP:8001 offload/results/K1 short c32 c150`,
plus `c256 c300 cap4` for R1. R1p/R2p: idle readings, then one cold 150K request (`suite.sh … c150` stage, 1 rep is
enough). Needles: `offload/results/K1/needle-<run>.jsonl`. T=0: `offload/engram/tests/logit_ab.py run --url
http://HEAD_IP:8001 --label <run> --out offload/results/K1/ab_<run>.jsonl --data
/opt/dsv41/data/bench-data/opinions_cl.jsonl`, then `compare` against R0.

## 3. Promotion

Production can take D-kv6-async-pack-1m (or D-kv6-async-pack) once:
- R0 is inert;
- R1 (and R1p/R2p for memory) meets every row above;
- R1's and R4's prefix-identity, corruption and needle rows pass;
- R1b and R1b-v5 pass the pool-pressure row (preemption and eviction exercised, outputs intact);
- MemAvailable never went below 1.5 GiB.

Production form: `offload/kvpack/stage_overlay.sh stage spark sparkworker`, then add `$(offload/kvpack/stage_overlay.sh args)` to the
docker arguments of the production launcher and the K1 variables to `OFFLOAD_DOCKER_ENV`.

Rollback: remove both, then `./stop.sh && ./start.sh`. The staged directory `~/dsv41-kvpack/<id>` can stay; nothing reads it
without the `-v` arguments.
