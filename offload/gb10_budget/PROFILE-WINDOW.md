# SP1-WINDOW coordinator sheet: one profiling HOLD, then a decision

**For the coordinator (Deepseek / coordinator) only.** The lane wrote and CPU-tested everything here and ran none of it.

## In plain language

- **What it does.** One HOLD of about **28 min** (ESTIMATED; production is down about 27.6 min) runs today's production
  (K1 327,680 at 21bed31) as the PF1 OFF arm on port 8001. The only addition is vLLM's own torch profiler.
- **What it captures.** Four unprofiled control cells come first, then four profiled cells, each with short traces from
  both Sparks. The cells are decode at B1 short, B1 32K, B4 32K and B1 150K, plus prefill at 32K and 150K.
- **Restore.** The window always ends with the W8 restore (`start`, `check`, `close`), whatever happens.
- **What it decides.** A CPU tool turns the traces into a per-step time budget. The decision table in §5 then names the
  next lane: a kernel lane, a chunk grid, a depth A/B, or nothing.
- **What it never changes.** No attention change, 0075/UP2, SO2, clock or power change, K1 or cap change, and no Nsight.

Design and evidence:

- [PROFILE-WINDOW-RECONCILE.md](PROFILE-WINDOW-RECONCILE.md) (item 1): why it is safe, with file:line evidence.
- [PROFILE-WINDOW-DESIGN.md](PROFILE-WINDOW-DESIGN.md) (item 2): cells and the minute plan.

## 1. Files (all in the reviewed commit)

| File | Role |
|---|---|
| `offload/launch/configs/SP1-profile.env` | The PF1 OFF 327,680 arm plus exactly `--profiler-config '<reviewed JSON>'` and one read-write bind `~/dsv41-offload-run/sp1-profile:/sp1-profile`. Refuses any other cap, trace or eager setting, and any inherited extra argument |
| `offload/launch/windows/sp1_precheck.sh` | Read-only precheck: Controller, sparkhead, sparkworker-wired, x86host status file. Prints PASS/FAIL/INFO and writes nothing |
| `offload/launch/windows/sp1_window.sh` | The runner. A dry plan by default; `--execute` runs the window and always restores |
| `offload/gb10_budget/window_cells.py` | Cell driver, called by the runner inside the HOLD only |
| `offload/gb10_budget/trace_budget.py`, `buckets.json` | CPU analysis after the window. Buckets and decision thresholds were pre-registered on 2026-10-26, before any trace existed |

## 2. Exact commands (Controller, reviewed clean checkout)

```bash
source ~/.config/ndm/paths.sh
cd <reviewed checkout>                  # HEAD = the reviewed commit; no tracked changes
C=$(git rev-parse HEAD)
OUT=/opt/dsv41/campaigns/DSV41-OFFLOAD/SP1-W-$(date -u +%Y%m%dT%H%M%SZ)

# 2a. read-only precheck, production serving (~1 min ESTIMATED); fix every FAIL before going on
bash offload/launch/windows/sp1_precheck.sh "$C" --out "$OUT"

# 2b. dry plan (no command, network or file)
bash offload/launch/windows/sp1_window.sh --dry-run

# 2c. the window. Add --stage-bundle if 2a said "run with --stage-bundle"
WINDOW_COORDINATOR=Deepseek/coordinator bash offload/launch/windows/sp1_window.sh --execute \
  --out "$OUT" --expected-experiment-commit "$C"

# 2d. analysis (CPU, Controller; after the window, production already restored)
CUDA_VISIBLE_DEVICES= "$HARNESS_PYTHON" -m offload.gb10_budget.trace_budget --window "$OUT"
```

**The precheck (2a)** repeats the runner's own preflight before any production time is spent. Its checks are:

- local checkout, W8 `restore.sh` pin (d1c4479d…), D2 paused or absent, public corpus, port 18001 free, `--out` new;
- sparkhead ledger: 21bed31, the tag, clean, no HOLD;
- experiment checkout and the 6 staged pins, including `SP1-profile.env`;
- `:8000` serving K1 327,680, and its load;
- on both Sparks: production image running and `:8001` free; the 13 in-container pins and image bases;
- on both Sparks: the image's `ProfilerConfig` has every reviewed field, and `/start_profile` and `/stop_profile`
  exist. It reads the source text only and imports no vllm or torch;
- on both Sparks: MemAvailable ≥ 5 GiB with GPU telemetry; `~/dsv41-offload-run/sp1-profile` absent; ≥ 10 GiB free
  in home;
- x86host: TP2 worker fb68c52, and no foreign process on the drafter GPUs.

`nsys` is reported as INFO only.

**Runner exit (2c)** prints one JSON line with `status`, `stop` and `restore`:

| Exit | Meaning |
|---|---|
| 0 | Window COMPLETE and production RESTORED |
| 1 | A STOP or a refusal; production RESTORED, or no HOLD was written (`restore.status` says which) |
| 2 | **Production not verified; the HOLD is left open.** Read `$OUT/cmd/restore-*.log`, then run `bash /opt/dsv41/campaigns/DSV41-OFFLOAD/W8/restore.sh start`, `check` and `close` by hand |

**To abort**, run `touch "$OUT/STOP"` or press Ctrl-C. The supervisor STOPs and the restore runs; signals are
ignored inside the restore.

**A restore counts only with all four strings:**

- `K1 START VERIFIED`;
- six `greedy K1-post vs restored: i IDENTICAL n` lines;
- `PRODUCTION HOLD OK`;
- `CLOSE: window closed`.

## 3. ESTIMATED minutes and hard bounds

| Phase (T = HOLD written) | ESTIMATED |
|---|---|
| Preflight (production serving) | 1-2 min |
| HOLD, snapshot, start of `sp1-327` (production down from about T+0.2) | T+0.2 to T+6.7 |
| Identity, mount, pins, warm-up, prompt build | T+6.7 to T+7.8 |
| Controls: short, 32K B1, 32K B4, 150K | T+7.8 to T+13.1 |
| Profiled: short, 32K B1 (prefill-early + decode), 32K B4, 150K (prefill-late + decode); traces collected after each | T+13.1 to T+20.2 |
| Final checks, profile directories removed | T+20.2 to T+20.5 |
| W8 restore `start`, `check`, `close` (MEASURED 7.37-7.63 min in PF1) | T+20.5 to T+28.1 |

The bounds are enforced by the runner (`sp1_window.py`):

- **Submissions close at T+40.** A cell that does not fit is not started.
- **The supervisor aborts at T+45**, with the STOP text "45-minute SP1 abort bound". Restore is done by about T+53 even
  then.

These guards are active throughout:

- MemAvailable ≥ 4 GiB (1 s telemetry, both Sparks);
- no new swap after ready, and telemetry alive;
- no foreign demand on `:8001` and no STOP file;
- capture floor MemAvailable ≥ 4.5 GiB before every `/start_profile`; below it the capture is SKIPPED_MEMORY and the
  request still runs;
- per trace file ≤ 128 MiB, per profile directory ≤ 512 MiB, and ≥ 8 GiB free disk after each capture.

## 4. After the window

- **Where the output goes.**
  - `$OUT/summary.json` holds the status, the restore result and artifact hashes.
  - `$OUT/cells/*.json` holds client numbers and hashes only.
  - `$OUT/traces-manifest.json` lists every trace with its sha256, which was checked on both sides of the copy.
  - `$OUT/traces/<cell>/<host>/` holds the traces themselves: 12 files ESTIMATED, at most 128 MiB each.
- **Analysis.** The command in 2d writes `$OUT/trace-budget.json` and never overwrites it. It also prints one line per
  capture: period, bucket shares, graph visibility and coverage. Below those come the client-side profiler overhead
  and `next work` with any `limited:` reasons.
- **Reporting.** Report numbers only, labelled MEASURED (profile-affected) with `$OUT` as the source. The shares are
  attribution; they are not promotion timings.
- **Leftover traces.** If a trace is left on a Spark (a STOP during collection), the restore reports
  `profile_dir_<host>: LEFT n files`. The files stay as evidence. Remove `~/dsv41-offload-run/sp1-profile` by hand
  before any later SP1 window; the next precheck will FAIL until you do.

## 5. Decision table (pre-registered in `buckets.json`, read from `trace-budget.json`)

First check whether the result can be read at all:

| Result | Meaning | Next |
|---|---|---|
| `graph_visibility: invisible` (or `partial`) on a decode capture | CUPTI did not record kernels inside CUDA graphs on this image | No decode lane from this trace. Run **HOLD E** (eager, §6). Prefill rows stay usable |
| `decode_overhead` > 0.10 (SSE gaps inside vs outside a capture) | The profiler changed decode too much to rank decode shares | **HOLD R** (§6) before acting on decode shares |
| `prefill_overhead` > 0.10 | The same, for prefill | Read prefill shares as a bound only; rerun in HOLD R if a prefill lane depends on them |
| A capture `SKIPPED_MEMORY` | MemAvailable was below 4.5 GiB at that start | **HOLD M** for that cell only |
| `coverage` < 0.90 | Too much GPU time in kernels that no pattern matches (`unmatched_top` lists them) | Add patterns to `buckets.json` in a reviewed CPU change and rerun 2d. **No window** |
| `aligned: false` or one rank missing | Rank traces do not cover the same iterations | Use only captures that are aligned; report the gap |

If the result is readable, each SUPPORTED row is a candidate. `next_work.next` orders them: largest addressable share
first, then `experiments.json` rank.

| SUPPORTED when (any decode or prefill capture) | Experiment (`experiments.json`) | Justifies |
|---|---|---|
| decode `routed_moe` share ≥ 40 % | 2 `decode-smoe-bandwidth` | **kernel lane**: smoe PIPE/OBI/CTA-tail sweep on the measured K histogram |
| prefill `routed_moe` share ≥ 40 % | 1 `prefill-grouped-kernel` | **kernel lane**: TF prompt grouped kernel (SM121 tiling, scratch) |
| prefill `idle_share` ≥ 10 % (host gaps and waits inside chunk periods) | 3 `chunk-threshold-grid` | **chunk grid** window: 2048/4096; 8192 only after the headroom gate |
| decode `drafter` share ≥ 10 % (engine stream waiting on the drafter reply) | 8 `depth-policy-recalibration` | **depth A/B** window |
| decode `host_gap` ≥ 0.5 ms per step | 5 `remaining-host-gaps` | host-gap lane (launch path, `sample_tokens`, async scheduling) |
| decode `collectives` share ≥ 5 %, or rank imbalance ≥ 1 ms per step | 6 `collective-boundary-fusion` | collective fusion lane (RDMA ring boundaries) |
| decode `sampler` share ≥ 5 % | 7 `verify-coverage-and-width` | verify/rejection width lane |
| 32K B4 decode `routed_moe` share ≥ 50 % | 9 `expert-parallel-split` | EP **feasibility study only** (B1 regression possible) |
| none of the rows above | — | **Nothing**: record the budget in the roadmap; no kernel lane, grid or depth A/B from this window |

Some results are not decided by the trace:

- `clocks-and-cooling` is a telemetry question: compare clock, power and temperature in `$OUT/telemetry-*.jsonl`.
- `kv-dtype-change` stays excluded.
- A large `attention` share has no owning hypothesis, and attention changes are out of scope. Report it to NDM; it
  does not start a lane.

## 6. Contingencies (each is a new HOLD, a fresh `$OUT` and the same reviewed commit; never pool cells across HOLDs)

| HOLD | When | Change | ESTIMATED |
|---|---|---|---|
| **E** eager | decode graph kernels invisible | Add `--enforce-eager` to the arm (a reviewed one-flag change to `SP1-profile.env`). Shares are attributable but not production's. The start is unmeasured: measure it first and STOP if it exceeds 15 min | 30-40 min |
| **R** overhead | profiler overhead > 10 % at B1 | Decode captures with `max_iterations` 4; cells 5-8 only | ≈ 28 min |
| **M** memory | a capture SKIPPED_MEMORY | Rerun that cell only, after checking sparkhead's idle MemAvailable | 18-22 min |
| Nsight | wanted after the torch trace | Follow-up lane: a launcher change for `nsys profile` at process creation (SP1.report.md 571-591), with its own review. Not scheduled; the precheck only reports whether `nsys` exists | — |

## 7. Do not

- Run the window without a PASS from the precheck in the same session.
- Reuse `$OUT`.
- Run it while D2 generation is active.
- Change the profiler JSON outside `SP1-profile.env` and `sp1_window.PROFILER_CONFIG`; a CPU test keeps the two
  byte-identical.
- Read profiled timings as production speed, or compare them with unprofiled windows.
- Copy traces, prompts or generated text into git or reports. The tools record numbers, kernel names and hashes only.
