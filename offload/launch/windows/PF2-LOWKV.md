# PF2-LOWKV coordinator sheet: 524,288 at a smaller KV reservation (H 6 GiB vs L 3 GiB), one HOLD

**For:** coordinator Deepseek / coordinator only. The lane wrote this sheet and has never run any step of it.

**Background:**

- design and pre-registered reading: `offload/pf2_lowkv/DESIGN.md`;
- arithmetic: `ARITHMETIC.md`;
- promotion gates (not run): `PROMOTION.md`.

**When:**

- after FW1 and FW2 (both Sparks DGX OS 7.6 / kernel 7.0.0-1019-nvidia / driver 580.178.04 since 2026-10-26 16:44Z);
- after the post-FW1 SP1 rerun, which gives the same-platform 327K reference;
- and, if NDM sets it, after sparkhead's 0.5 GB display carveout.

**Production impact:**

- about 62.6–68.5 min of HOLD, with production down about 61.8–67.7 min (ESTIMATED);
- hard bound: production is back by about T+74 even after an abort.

## 1. Files

| file | role |
|---|---|
| `offload/launch/configs/PF2-lowkv.env` | PF1-window.env's OFF arm at 524,288 plus three things: `KV_BYTES_OVERRIDE=$PF2_KV_BYTES`, SP1's reviewed `--profiler-config` byte for byte, and the read-write bind `~/dsv41-offload-run/sp1-profile:/sp1-profile`. It refuses any other cap/trace/eager, a KV value outside [2, 6] GiB or not a whole MiB, and any inherited setting |
| `offload/launch/windows/pf2_precheck.sh` | Read-only precheck. Prints PASS/FAIL/INFO and writes nothing |
| `offload/launch/windows/pf2_window.sh` | The runner. A dry plan by default; `--execute` runs H-L-H-L and ALWAYS restores |
| `offload/pf2_lowkv/cells.py` | Cell driver (warm-up, c150-a/b, p150, needle-480k), called by the runner inside the HOLD only |
| `offload/pf2_lowkv/analyze.py` | CPU analysis after the window: classification, reading, memory and dispatch diff |

## 2. Parameters (no defaults for the platform: read them, never assume)

| flag | value | where it comes from |
|---|---|---|
| `--sparkhead-kernel` / `--sparkhead-driver` | e.g. `7.0.0-…-nvidia` / `580.178.…` | `FW1/inventory-post.txt` line 1 (kernel) and line 5 (driver; line 3 is the OTA version), written by `fw1_sparkhead.sh post`. The precheck FAILs if the two differ |
| `--sparkworker-kernel` / `--sparkworker-driver` | `7.0.0-1019-nvidia` / `580.178.04` since FW2 (2026-10-26 16:44Z); read fresh | `ssh sparkworker 'uname -r; nvidia-smi --query-gpu=driver_version --format=csv,noheader'` (read-only) |
| `--low-kv-bytes` | `3221225472` (default) | `ARITHMETIC.md`: the smallest of 2/3/4 GiB holding 2 × 524K + two 32K seats with ≥ 25 % margin. The precheck and the runner check it with `arith.check_low` |
| `--reference` (precheck and analysis) | the post-FW1 SP1 rerun window directory | its unprofiled `cells/ctl-150k-b1.json` TTFT is the 327K reference for fast/slow |

## 3. Exact commands (Controller, reviewed clean checkout)

```bash
source ~/.config/ndm/paths.sh
cd <reviewed checkout>                  # HEAD = the reviewed commit (e.g. a pinned PF2-W/checkout-<commit>); no tracked changes
C=$(git rev-parse HEAD)
OUT=/opt/dsv41/campaigns/DSV41-OFFLOAD/PF2-W-$(date -u +%Y%m%dT%H%M%SZ)
REF=/opt/dsv41/campaigns/DSV41-OFFLOAD/<post-FW1 SP1 rerun dir>
P="--sparkhead-kernel <K> --sparkhead-driver <D> --sparkworker-kernel <K> --sparkworker-driver <D>"    # section 2

# 3a. read-only precheck, production serving (~1-2 min ESTIMATED); fix every FAIL before going on
bash offload/launch/windows/pf2_precheck.sh "$C" $P --out "$OUT" --reference "$REF"

# 3b. dry plan (no command, network or file)
bash offload/launch/windows/pf2_window.sh --dry-run

# 3c. the window. Add --stage-bundle if 3a said "run with --stage-bundle"
WINDOW_COORDINATOR=Deepseek/coordinator bash offload/launch/windows/pf2_window.sh --execute \
  --out "$OUT" --expected-experiment-commit "$C" $P

# 3d. analysis (CPU, Controller; after the window, production already restored)
CUDA_VISIBLE_DEVICES= "$HARNESS_PYTHON" -m offload.pf2_lowkv.analyze --window "$OUT" --reference "$REF"
```

**The precheck (3a)** runs the runner's preflight before any production time is spent:

- **Controller:** local checkout; W8 `restore.sh` pin (d1c4479d…); D2 paused or absent; public corpus; port 18001 free;
  `--out` new.
- **Parameters:**
  - the LOW reservation (blocks, pool tokens, margin);
  - the kernel/driver parameter format;
  - `FW1/inventory-post.txt` kernel/driver equal to the sparkhead parameters (if absent: INFO);
  - the reference TTFT.
- **sparkhead:** 21bed31/tag clean with no HOLD; the experiment checkout and the 6 staged pins incl. `PF2-lowkv.env`;
  `:8000` serving K1 327,680, and its load.
- **Both Sparks:**
  - production image running and `:8001` free;
  - the 13 in-container pins and image bases;
  - image `ProfilerConfig` fields and endpoints (source text only);
  - **`uname -r` and driver equal to the parameters**; THP recorded, never changed;
  - MemAvailable ≥ 5 GiB;
  - the extended telemetry: every vmstat counter and PSI some/full;
  - `~/dsv41-offload-run/sp1-profile` absent and ≥ 10 GiB free in home.
- **x86host:** TP2 worker fb68c52, and no foreign process on the drafter GPUs.

**Runner exit (3c)** prints one JSON line with `status`, `stop` and `restore`:

| Exit | Meaning |
|---|---|
| 0 | Window COMPLETE (a start skipped for time shows as SKIPPED_TIME in `summary.json`) and production RESTORED |
| 1 | A STOP or a refusal; production RESTORED, or no HOLD was written (`restore.status` says which) |
| 2 | **Production not verified; the HOLD is left open.** Read `$OUT/cmd/restore-*.log`, then run `bash /opt/dsv41/campaigns/DSV41-OFFLOAD/W8/restore.sh start`, `check` and `close` by hand |

**To abort,** run `touch "$OUT/STOP"` or press Ctrl-C. The supervisor STOPs, the in-container samplers stop, and the
restore runs; signals are ignored inside it.

**A restore counts only with all of these:**

- `K1 START VERIFIED`;
- six `greedy K1-post vs restored: i IDENTICAL n` lines, i = 0..5 (a literal "6/6" is not accepted);
- `PRODUCTION HOLD OK`;
- `CHECK: DONE`;
- `CLOSE: window closed`.

## 4. ESTIMATED minutes (from MEASURED PF1/SP1 step times; DESIGN.md section 8)

| Phase (T = HOLD written) | all fast | H slow, L fast | all slow |
|---|---:|---:|---:|
| HOLD, snapshot, stop production, `s1-H` start + checks + samplers | T+7.3 | T+7.3 | T+7.3 |
| `s1-H` cells (build, warm-up, c150-a/b, p150 + traces) | T+13.6 | T+14.7 | T+14.7 |
| `s2-L` restart + cells | T+25.8 | T+26.9 | T+28.1 |
| `s3-H` restart + cells | T+38.1 | T+40.3 | T+41.5 |
| `s4-L` restart + cells | T+50.3 | T+52.6 | T+54.9 |
| `needle-480k` on `s4-L` (build + ~230–312 s prefill) | T+54.9 | T+57.2 | T+60.8 |
| finish checks, profile directories removed | T+55.2 | T+57.5 | T+61.1 |
| W8 restore `start`, `check`, `close` (MEASURED 7.37 min) | **T+62.6** | **T+64.9** | **T+68.5** |

**Bounds enforced by the runner** (`pf2_window.py`):

- **Submission.** No cell, needle or start begins unless its estimate ends by **T+62**. A later start is skipped as a
  whole (SKIPPED_TIME); the needle needs 6 min.
- **Abort.** The supervisor aborts at **T+66**, and the restore then takes ~7.4 min, so production is back by ≈ T+74.
- **Memory.** MemAvailable ≥ 4 GiB on both Sparks at every 1 s sample, including trace flushes. ≥ 5 GiB before each
  start. Captures are SKIPPED_MEMORY below 4.5 GiB. Swap growth stays ≤ 256 MiB after readiness.
- **Trace caps (SP1).** ≤ 128 MiB per file, ≤ 512 MiB per directory, ≥ 8 GiB free disk after each capture.
- **Foreign demand.** Demand on :8001 above the owned in-flight requests STOPs the window; so does a stale telemetry
  stream.

## 5. Reading the result (pre-registered, `analyze.py`; DESIGN.md section 5)

| `reading.verdict` | meaning | next |
|---|---|---|
| HEADROOM_IMPLICATED | every L start fast, ≥ 1 H start slow | Report to NDM. A 524K-LOW profile becomes a candidate: `PROMOTION.md` G1–G9 in its own window, on NDM's word. The dispatch diff names what changed in sparkworker's MoE kernels |
| NO_CAUSE_SHOWN | every start fast | The slow state did not appear (as in CW/MATCH). L stays a candidate; no cause is shown. Do not promote on this alone |
| NOT_HEADROOM | an L start was slow | Headroom does not prevent the slow state. Use the fast-vs-slow dispatch diff and the compaction/PSI deltas to choose the next lead; L is not a fix |
| AMBIGUOUS | a request between 1.12× and 1.20×, or a mixed start | No reading. Report the numbers |
| INCOMPLETE | fewer than 2 classified starts per arm | No reading. A rerun is a new HOLD after review, never an automatic retry |

**Always reported beside the verdict:**

- decode L vs H (> 2 % slower is flagged);
- the needle (6/6 expected; any miss or foreign code FAILs promotion G2);
- per-prefill compaction, reclaim and PSI deltas and PSS/RSS changes;
- `smaps_rollup` reads over 50 ms (possible perturbation).

## 6. Contingencies

- **A start fails** (identity, KV pool, pins, kernel/driver drift). The window STOPs and restores. Fix the cause, re-review,
  run a new HOLD.
- **Memory trip.**
  - On H: it repeats PF1 MATCH's sparkhead dip. Record it. The L half of the evidence is still needed, so a rerun puts L
    first only after review.
  - On L: that alone is evidence against the headroom reading.
- **SKIPPED_MEMORY capture.** The 150K TTFTs still classify; only that start's dispatch row is missing.
- **Needle SKIPPED_TIME.** G2 in `PROMOTION.md` covers it later; the reading does not depend on it.
