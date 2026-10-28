# PF2-LOWKV item 2: window design (one HOLD, ≤ 75 min ESTIMATED including restore)

Written 2026-10-26 by lane PF2-LOWKV, before any PF2 window exists. Pre-registered: the thresholds and the reading in §5
are decision rules, not results.

- Arithmetic: `ARITHMETIC.md`.
- Runner: `offload/launch/windows/pf2_window.py`.
- Analysis: `offload/pf2_lowkv/analyze.py`.
- Coordinator sheet: `offload/launch/windows/PF2-LOWKV.md`.

## 1. Question

At the 524,288 maximum request, does reserving less memory for the KV pool remove the intermittent slow 150K prefill?

- **Slow state (MEASURED).** 150K cold took 95.3 / 91.5 / 91.3 s in PF1 W1/W2/W3, against 69.2–70.8 s in CW/MATCH and
  68.8–71.2 s at 327K. PF1-CAUSE located the slow time in sparkworker's MoE work.
- **The change.** The reservation goes from 6 GiB to 3 GiB per rank (+3.0 GiB MemAvailable ESTIMATED). Weights,
  precision, attention, request maximum and code are unchanged.
- **What this window does not decide.** It is a cause test, not a promotion. Promotion gates are in `PROMOTION.md`.

## 2. Arms (same image, weights, K1 overlays, remote drafter, env; one HOLD)

| arm | max_model_len | `--kv-cache-memory-bytes` | expected engine pool | role |
|---|---:|---:|---:|---|
| H | 524,288 | 6,442,450,944 (production) | 2,820,756 tokens | capacity control = PF1's OFF arm |
| L | 524,288 | `--low-kv-bytes`, default 3,221,225,472 | 1,410,348 tokens (`arith.expected_tokens`) | LOW reservation |

- **Config.** Both arms start through `offload/launch/configs/PF2-lowkv.env`. It is PF1-window.env's OFF arm at 524,288
  (production's own docker env, drafter env and read-only mounts, port 8001, `VLLM_ENGRAM_REMOTE_REQUIRED=1`) plus three
  things:
  - `KV_BYTES_OVERRIDE=$PF2_KV_BYTES`, the only difference between H and L;
  - SP1's reviewed `--profiler-config` (identical JSON; idle until `/start_profile`);
  - the read-write trace mount `~/dsv41-offload-run/pf2-profile:/pf2-profile`, which differs from SP1's only in its
    directory name.
- **Runtime identity check.** Experiment == production except: port, cap 524,288, REQUIRED=1, the one KV value, and
  exactly one reviewed profiler config. Anything else STOPs the window.
- **`record_shapes` stays false.**
  - SP1's window MEASURED a prefill profiler overhead of 24.9 % with `record_shapes=false` and did not measure
    `record_shapes=true`. Its overhead is therefore not shown to be acceptable (brief).
  - Kernel identity and launch configuration (grid, block, registers, shared memory) are kineto kernel-event arguments
    and are recorded without it.
  - MoE "shape" is read from the launch configuration and the fixed 4,092-token chunk (§6).
- **No 327K arm.** A 5th start (≈ 13 min) does not fit in 75 min. The same-day 327K reference comes from the post-FW1
  SP1 rerun (unprofiled `ctl-150k-b1`, the analysis `--reference` input) and production's own restore check.

## 3. Sequence: H – L – H – L (4 starts, alternating)

- **Why this order.**
  - Every PF1 slow arm was the 2nd start of its HOLD, and the fast MATCH arm was the 1st.
  - Putting H first gives H the "first start after production" position, the one that has only been seen fast. That
    biases against the L-favouring reading, which is the conservative choice.
  - Positions 2–4 all follow a previous experiment start. Position is recorded and reported.

Per start, in order:

1. **Start** (`start_exp.sh` with `PF1_CAP=524288 PF1_TRACE=0 PF1_EAGER=0 PF2_KV_BYTES=<arm>`), then:
   - identity checks;
   - an engine-log KV pool equal to the arm's expected tokens;
   - 13 in-container pins;
   - proxy, Engram and x86host worker ready;
   - the profile directory empty and the profiler never started;
   - **host identity**: `uname -r` and driver on each Spark equal `--sparkhead-kernel/--sparkhead-driver/--sparkworker-kernel/
     --sparkworker-driver`; THP enabled/defrag recorded (never changed);
   - MemAvailable ≥ 5 GiB on both before the start.
2. **Per-process memory sampler.** One light `docker exec -i dsv41-serve python3 -u -` per Spark: 1 s
   `smaps_rollup` + `status` of the API server, engine core and workers, numbers only. It is owned, and stopped before
   the next start.
3. **`warmup`**: 32K, 64 tokens forced, discarded.
4. **`c150-a`, `c150-b`**: unprofiled cold 150K (the same public prompt, fresh cache salts, 1,500 output tokens,
   T=1/top_p=1/seed 42/priority 10, exactly PF1's cold cell).
   - Their TTFTs classify the start fast/slow (§5).
   - Their SSE decode rates are the start's **decode sample**.
5. **`p150`**: profiled cold 150K, a third salt.
   - **One** mid-prompt prefill capture on both ranks: `/start_profile` 35 s after submission (≈ chunk 14–19 of 37,
     prefix ≈ 57–78K tokens, fast or slow), held 5 s (2–3 chunks; the profiler also stops itself after 16 iterations),
     then `/stop_profile`.
   - Capture floor: MemAvailable ≥ 4.5 GiB on both Sparks, else SKIPPED_MEMORY (the request still runs).
   - Its TTFT is profile-affected and never classified.
6. **Collect.** Per node: one new rank file (≤ 128 MiB, directory ≤ 512 MiB, free disk ≥ 8 GiB), `scp`, O_DIRECT sha256
   on both sides, then remove the verified Spark copy. Memory stays guarded at ≥ 4 GiB through the trace flush, because
   the 1 s telemetry runs throughout.
7. **Last L start only: `needle-480k`.**
   - The existing `offload/kvpack/needle.py` builder and scorer: 6 needles, 480,000 tokens, T=0, thinking off, 300
     output tokens, at most 6 `/tokenize` calls. It is built on L, where the headroom is.
   - Recorded: recall, foreign codes, TTFT/elapsed, answer sha256. No text.
   - It runs last so that a memory trip costs only this request. It is skipped (recorded SKIPPED_TIME) if it cannot finish
     before the submission bound.
   - At 524,288 one 480K request needs 8,085 blocks (L pool: 23,301).

## 4. Telemetry (both Sparks, 1 s, owned ssh streams; numbers only)

| source | fields |
|---|---|
| `/proc/meminfo` | MemAvailable, MemFree, Cached, SwapFree, SwapTotal, AnonHugePages, Shmem (the MemAvailable floor guard reads this row) |
| `/proc/pressure/memory` | some/full avg10, avg60 and total (µs) |
| `/proc/vmstat` | compact_stall/fail/success, compact_migrate_scanned, compact_free_scanned, pgscan_kswapd/direct, pgsteal_kswapd/direct, allocstall_*, thp_fault_alloc, thp_fault_fallback, thp_collapse_alloc, pgmajfault, pgfault, pswpin, pswpout |
| nvidia-smi | PF1's reviewed FIELDS: SM/memory clocks, power, temperature, utilisation, throttle reasons |
| in-container sampler | per process (API server, EngineCore, Worker_TP*): Rss, Pss, Pss_Anon, Pss_File, Pss_Shmem, Swap from `smaps_rollup`; VmRSS, RssAnon, RssFile, VmHWM from `status`; read time in ms |

- **Torch allocated vs reserved: not recorded.** The existing memprobe (`dsv41_memprobe.py`) reports only when
  `VLLM_DSV41_MEMPROBE` is set. Setting it changes the arm's environment, so the brief's condition ("without changing the
  arm") excludes it.
- **Flag: `smaps_rollup` walks every mapping under the process's mmap lock.**
  - At 1 Hz this is a small perturbation, and it is the same in H and L.
  - Its read time is recorded per sample. If the engine-core/worker read exceeds 50 ms, the analysis flags it.

## 5. Pre-registered classification and reading

**Classification of each 150K cold request** (unprofiled `c150-a/b` only):

- ratio `r = TTFT / ref`;
- `ref` is the same-day 327K unprofiled 150K TTFT (the SP1 rerun's `ctl-150k-b1`, passed as `--reference`; default 70.0 s,
  the PF1/SP1 MEASURED 68.8–71.2 s);
- **fast** if `r ≤ 1.12`, **slow** if `r ≥ 1.20`, **ambiguous** in between.
- MEASURED separation today: fast 0.98–1.02, slow 1.31–1.37.

**A start** is fast if both its 150Ks are fast, slow if both are slow, and mixed otherwise.

**Reading over the window**, the first matching rule wins (order amended after S44 round 1 and recorded in the campaign register
before the 2026-10-26 window ran: a slow L start is checked first):

| reading | rule |
|---|---|
| NOT_HEADROOM | any classified L start slow ("both slow → not headroom"; L slow while H is fast also refutes it) |
| INCOMPLETE | fewer than 2 classified starts per arm |
| AMBIGUOUS | any ambiguous request or mixed start |
| HEADROOM_IMPLICATED | every L start fast and at least one H start slow, in this window |
| NO_CAUSE_SHOWN | every start fast: the slow state did not appear; L stays a candidate, no cause shown |

**Reported beside the reading, never changing it:**

- **Memory pressure and compaction around each prefill.** These are deltas from 10 s before submission to first token:
  compact_stall/fail, pgscan/pgsteal, allocstall, thp_fault_fallback, pgmajfault, PSI some/full total, and MemAvailable
  min/median. They are given per rank and per start, together with the per-process PSS/RSS change.
- **Position in the HOLD.**
- **Decode tok/s, H vs L.** An L decode regression above 2 % is flagged for promotion.

## 6. Kernel-dispatch comparison (fast vs slow captures)

For each captured `p150` (both ranks):

- the SP1 `trace_budget` per-chunk bucket budget (routed_moe, dense_gemm, attention, collectives, other, idle);
- a **dispatch signature table** per rank: (kernel name, grid, block, registers/thread, static + dynamic shared memory)
  → launches per complete chunk and mean/median µs per launch.

The diff between one fast and one slow capture of the same rank lists:

1. signatures present in only one capture;
2. the same signature with different launches per chunk;
3. the same signature with per-launch time changed by more than 10 %, ranked by ms per chunk.

Rank 1 (sparkworker) `routed_moe` is the focus (PF1-CAUSE). If every start is fast (or every start is slow), the diff runs H
against L and is labelled "no state contrast".

## 7. Guards and bounds

- **Memory.**
  - MemAvailable ≥ 4 GiB on both Sparks at every 1 s sample, through every trace flush. Any breach STOPs the window.
  - ≥ 5 GiB before each start.
  - Capture floor 4.5 GiB.
  - Swap growth ≤ 256 MiB after readiness.
- **Process and traffic.**
  - Telemetry liveness.
  - Foreign demand on :8001 above the owned in-flight requests STOPs the window.
  - STOP file.
  - Trace caps as SP1.
- **Deadlines from the HOLD timestamp.**
  - No step starts unless its estimate fits before **T+62 min**.
  - The supervisor aborts at **T+66 min**.
  - Restore start+check+close takes 7.4 min (MEASURED), so production is back by **≈ T+74 min** even after an abort.
- **Always: `W8/restore.sh start|check|close`.**
  - Required strings: "K1 START VERIFIED"; six "greedy K1-post vs restored: i IDENTICAL n" lines (i = 0..5, no literal
    "6/6"); "PRODUCTION HOLD OK"; "CHECK: DONE"; "CLOSE: window closed".
  - Then remove only the empty profile directories this window created.
- **Not changed:** THP, drop_caches, clocks, power, K1 code, the production checkout, `W8/restore.sh`.

## 8. Minute plan (ESTIMATED from MEASURED PF1/SP1 step times)

Measured step times, PF1 W1/W3/MATCH/CW and SP1 W ledgers and command logs:

- **Arms and restore.**
  - HOLD → first warm-up: 7.1 min.
  - Arm-to-arm restart: 6.5–6.8 min.
  - Restore: 7.37 min.
- **Requests.**
  - 32K warm-up: 0.3 min.
  - 150K cold incl. 1,500 tokens: 1.60–1.63 min fast, 1.97–2.03 min slow.
  - Profiled 150K: 1.75 min at 327K.
  - 300K prefill at 524K: 142 s fast / 193 s slow, so 480K ≈ 230 s / 312 s (×1.6 linear in tokens, PF1's per-token
    cost is flat).

| step | all fast | H slow, L fast | all slow |
|---|---:|---:|---:|
| HOLD → start 1 ready (snapshot, stop production, start, checks, sampler) | 7.3 | 7.3 | 7.3 |
| start 1 cells (build 0.5, warm-up 0.35, 2 × 150K, p150 + trace 0.1, probes 0.3) | 6.25 | 7.4 | 7.4 |
| starts 2–4 (restart 6.5 + cells without build) | 3 × 12.25 | 12.25 + 13.4 + 12.25 | 3 × 13.4 |
| needle-480k on start 4 (build 0.5, prefill, answer) | 4.6 | 4.6 | 5.9 |
| finish checks | 0.3 | 0.3 | 0.3 |
| restore start+check+close | 7.4 | 7.4 | 7.4 |
| **HOLD total** | **62.6** | **64.9** | **68.5** |
| production down (first stop at ≈ T+0.5 → restore start verified, ≈ HOLD − 0.85) | ≈ 61.8 | ≈ 64.0 | ≈ 67.7 |

- **Margin to the 75 min brief bound.** 6–12 min. The needle step must start by T+56 to pass the T+62 bound with its
  6-minute estimate. In the all-slow case it starts at ≈ T+54.9.
- **If the window runs late.** If the slow state runs longer than measured, the needle is skipped first. A 4th start that
  does not fit is skipped and recorded, and the reading then needs ≥ 2 starts per arm (else INCOMPLETE).
- **Contingency.** If a start fails or memory trips, the window ends in restore. A rerun is a new HOLD after review,
  never an automatic retry.
