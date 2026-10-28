# SP1-WINDOW item 2: one HOLD that replaces SP1's estimated buckets with traced spans

Status: design. This lane produces no GB10 numbers. Cited inputs are **MEASURED** (source named). Every duration in the
plan is **ESTIMATED** from them. Item 1 (`PROFILE-WINDOW-RECONCILE.md`) fixes the arm, the profiler JSON and the guards
used here.

## The question

SP1 allocated a 40-ms B1 step and a 1.9-s prefill chunk to buckets by arithmetic (SP1.report.md "A constrained,
explicitly uncertain budget"): routed MoE about 18 ms, dense about 8 ms, and so on. Nothing has been traced at kernel
level. The window must deliver four things:

1. Observed per-step spans for decode: B1 short, B1 32K, B4 32K and B1 150K.
2. Observed per-chunk spans for prefill: 32K and 150K, in chunks of 4,092 tokens.
3. Both TP ranks for every capture.
4. An unprofiled matched control, so the profiler's own overhead is known before any span is believed.

The budget tool (item 4) turns the traces into named buckets and ranks `experiments.json`.

## Design: one experiment start, eight cells, always W8 restore

**Arm.** `sp1-327`:

- `SP1-profile.env`, which is `PF1-window.env` with `PF1_CAP=327680 PF1_TRACE=0 PF1_EAGER=0`, plus `--profiler-config`
  and the `/sp1-profile` mount.
- Port 8001, through the runner's tunnel 127.0.0.1:18001.
- The production TP2 K2 drafter, untouched.

**Prompts.** All prompts come from the public corpus (`window_requests.corpus`, seed-42 order, 11.8 M characters).
Each cell starts at its own character offset, so no two cells send the same text. Every request uses a fresh,
deterministic cache salt, so cached prompt tokens are 0 (the runner refuses a hit). The other settings match PF1 and
bench:

- temperature 1, top_p 1, seed 42, priority 10, thinking high;
- `max_tokens` 1,500, the same output budget as every historical step table (D, W4, W5, K1).

| Order | Cell | Requests | Profiler | Captures |
|---|---|---|---|---|
| 0 | warm-up | 1 × 32K, 64 tokens (EOS ignored), discarded | configured, never started | — |
| 1 | `ctl-short-b1` | 1 × ~1K | configured, never started | — (control) |
| 2 | `ctl-32k-b1` | 1 × 32K | configured, never started | — (control) |
| 3 | `ctl-32k-b4` | 4 × 32K, distinct prompts, released together | configured, never started | — (control) |
| 4 | `ctl-150k-b1` | 1 × 150K | configured, never started | — (control) |
| 5 | `prof-short-b1` | same prompt as 1, new salt | started | `decode` |
| 6 | `prof-32k-b1` | same prompt as 2, new salt | started twice | `prefill-early` (chunks 1-3), then `decode` |
| 7 | `prof-32k-b4` | same prompts as 3, new salts | started | `decode` (after all four streams emit) |
| 8 | `prof-150k-b1` | same prompt as 4, new salt | started twice | `prefill-late` (start 50 s after submission, prefix ~100K+), then `decode` |
| — | restore | — | — | `W8/restore.sh start`, `check`, `close`, always and on every STOP |

**Capture rules** (driver `window_cells.py`, runner `sp1_window.py`):

- **`decode`.** Every stream of the cell has emitted its first text, plus 2.0 s more, so the capture lands in steady
  decode and not on the first post-prefill steps. Then `/start_profile`. The profiler stops itself after 16 iterations
  (15 complete step periods). The driver sends `/stop_profile` after ≤ 3 s anyway.
- **`prefill-early`.** `/start_profile` is sent after prompt construction and the idle check, right before submission.
  `/stop_profile` follows 5 s later. 32K TTFT is ~14.5 s (MEASURED W5), so the capture holds 2-3 full chunks. If a
  first token arrives before the stop, the capture is labelled `mixed`, never `prefill`.
- **`prefill-late`.** `/start_profile` is sent 50 s after submission, and only if no token has arrived (SP1
  `capture.py` rule). It stops 5 s later. 150K TTFT is 68.9-71.2 s (MEASURED PF1 W1/W3 off-327).
- **Before every `/start_profile`.** The latest 1 s telemetry sample on both Sparks (≤ 3 s old) must show MemAvailable
  ≥ 4.5 GiB. Otherwise the capture is `SKIPPED_MEMORY`: the request still runs and is recorded, and no profiler starts.
- **After every profiled cell, on each node.** The runner requires:
  - one new rank file per capture (rank 0 on sparkhead, rank 1 on sparkworker);
  - each file ≤ 128 MiB, the directory ≤ 512 MiB, and ≥ 8 GiB free disk.

  It then copies the files, compares sha256 on both sides and removes the verified Spark copies. Any breach is a STOP.

### Why these cells and no others

**Controls first.** "Configured, never started" is a pure control. After the first capture the wrapper exists and
`step()` runs every iteration (cheap, but no longer zero). The controls also give the profile-free SSE step gaps and TTFT
of exactly the prompts the profiled cells reuse.

**Three overhead readings, all MEASURED and profile-affected.**

| Reading | Comparison | What it shows |
|---|---|---|
| (a) within the profiled request | SSE gaps inside the capture interval vs the same request's gaps outside it | the cost of recording |
| (b) cross-request | the control's decode gap p50 vs the profiled request's gaps outside the capture | whether "configured" alone costs anything |
| (c) prefill | profiled vs control TTFT, divided by the profiled chunk count | the per-chunk cost of recording, an upper bound (it includes the export stall) |

The design threshold is SP1's 10 %. Above it, a trace locates work but cannot rank small gains.

**One request carries both captures in 6 and 8.** A 150K prefill costs ~70 s. Running prefill and decode captures on
separate requests would add ~2.5 min, with no new information.

**One capture per cell.**

- Each decode capture yields 15 step periods per rank.
- Each prefill capture yields 2-3 chunk periods per rank.
- The 150K prefill is captured late, so the prefix-dependent indexer and attention cost shows against the 32K early
  chunks.
- Repeats would mostly buy run-to-run noise that the controls already measure.

**Excluded, with the reason for each.**

| Excluded | Reason |
|---|---|
| 150K B4 | four 150K prefills, the unmeasured memory case SP1 flagged; not in the brief |
| 300K | its prompt construction made the deepest MEASURED memory dip (4.17 GiB) |
| B3 | B4 covers the k4 schedule |
| warm and partial-cache modes | they remove the prefill being attributed |
| an `--enforce-eager` arm | item 1 §3 |
| Nsight | item 1 §3 |
| attention change, 0075/UP2, SO2, clock/power, K1, cap | the brief excludes them |

**Telemetry.** Both Sparks are sampled every 1 s through PF1's `Supervisor`, unchanged:

- SM and memory clocks, power draw and limit, temperature, utilization;
- `clocks_event_reasons.active` (throttle reasons);
- MemAvailable, SwapFree.

The tool slices this per capture interval.

**Production counters, per cell.**

- Process-proxy stats before and after (`up2_proxy_stats.py`). The usual checks apply: no new
  `fallback_*`/`provenance_error`, watchdog running, `rounds_ok` grows.
- vLLM `/metrics` speculative counters before and after, through the tunnel: accepted per round, per position.
- X86host worker steps, from the read-only status probe.

All of this is numbers only. No prompt or generated text is written anywhere. The runner keeps prompt sha256, cache-salt
sha256, character offsets and token counts.

## MEASURED inputs for the estimate

| Input | Value | Source |
|---|---|---|
| HOLD write → start begins | 0.16-0.22 min | PF1-W1/W3/MATCH `cmd/` (hold-read → `<arm>-hold`) |
| Start → `start_exp.sh` done, 327,680 | 6.5 min (0.17 → 6.67 min after HOLD, both W1 and W3) | PF1-W1/W3 `cmd/*-off-327-start.log` |
| Post-start identity, 13-pin verify, logs, worker | 0.15 min | PF1-W1/W3 (start log → worker probe) |
| Restore: reap → `start` → `check` → `close` | 7.37-7.63 min (`start` ≈ 7.0, `check` ≈ 0.33) | PF1-W1 44.70 → 52.33, W3 34.82 → 42.19, MATCH 11.39 → 18.76 min |
| 32K warm-up, 64 tokens | TTFT 16.9 / 17.4 s | PF1-W1/W3 `off-327-requests.jsonl` |
| Short B1, 1,500 tokens | TTFT 0.37 s; decode 52.8 tok/s → ≈ 28.8 s | COMPARISON W5 short, W8 C1 |
| 32K B1 | TTFT 14.5 s; decode 62.6 tok/s → ≈ 38.5 s | COMPARISON W5 / W8 C1 |
| 32K B4 | 6,000 tokens in 116.0 s, TTFT up to 66.0 s | COMPARISON W5 "32K, 4 concurrent" |
| 150K B1 | TTFT 68.9-71.2 s; decode 54.1 tok/s → ≈ 97 s total | PF1-W1/W3 off-327; COMPARISON W8 C1 |
| `/tokenize` cost | 19-33 s for a 20-23-call bisection → ≈ 1-1.5 s per call | S30 via UP2-LOW.md budget |
| Profiler stop/export stall | **not measured** | ESTIMATED ≤ 15 s per capture, plus ≤ 20 s collection |

## ESTIMATED minute-by-minute plan (T = HOLD written)

**Before the HOLD** (≈ 1-2 min, production serving): the runner's read-only preflight. This is PF1's preflight, plus:

- the profile directory is absent on both nodes;
- ≥ 10 GiB free in each home;
- the staged pins include `SP1-profile.env`;
- 1 s telemetry is running.

| T (min) | Step |
|---:|---|
| 0.0-0.2 | `window.sh open` (bounded drain ≤ 20 min, else no HOLD); production snapshot of both ranks → `rollback.json`; tunnel up |
| 0.2-6.7 | `sp1-327` start: `stop.sh`, so production is down from here; `mkdir` of the profile directory on both nodes before the start |
| 6.7-6.9 | PF1 `runtime()` identity, after removing the reviewed `--profiler-config`; `/sp1-profile` read-write mount on both ranks; 13-pin verify; KV 2,502,600; proxy, Engram and worker ready; "Profiler … enabled" log line; profile directory empty |
| 6.9-7.3 | warm-up 32K, 64 tokens |
| 7.3-7.8 | build the short, 4 × 32K and 150K prompts (≤ 4 `/tokenize` each, `window_cells.MAX_COUNT_CALLS`); write numeric cuts and hashes |
| 7.8-8.4 | `ctl-short-b1` |
| 8.4-9.2 | `ctl-32k-b1` |
| 9.2-11.3 | `ctl-32k-b4` |
| 11.3-13.1 | `ctl-150k-b1` |
| 13.1-14.0 | `prof-short-b1` (decode capture; collect) |
| 14.0-15.4 | `prof-32k-b1` (prefill-early + decode; collect two) |
| 15.4-17.9 | `prof-32k-b4` (decode; collect) |
| 17.9-20.2 | `prof-150k-b1` (prefill-late + decode; collect two) |
| 20.2-20.5 | clean checks: both ranks' logs, container unchanged, proxy counters, worker; profile directory removed on both nodes |
| 20.5-28.1 | `W8/restore.sh start`, `check`, `close` (production back ≈ 27.8) |

| Outcome | HOLD | Production not serving | Submissions | Prompt tokens |
|---|---:|---:|---:|---:|
| full window | ≈ **28 min** | ≈ 27.6 min | 15 requests (incl. warm-up) | ≈ 0.65 M |

**Deadline.**

| Bound | Time from the HOLD timestamp |
|---|---|
| Submissions close | T+40 min |
| Measurement abort (the supervisor STOPs, the runner restores) | T+45 min |
| Restore done, even after an abort at the bound | ≈ T+53 min |

That leaves more than 2× margin on the ESTIMATED 20-min measurement phase and stays well inside the brief's 75 min.

A cell is not started unless its ESTIMATED duration fits before the submission bound:

| Cell | ESTIMATED duration |
|---|---|
| control short / 32K B1 / 32K B4 / 150K | 1 / 1.5 / 3 / 3 min |
| profiled cell | the control's + 1 min |

The start itself is refused unless start plus warm-up (8 min) fits.

## Contingency HOLDs (only if this window calls for them)

| Trigger | HOLD | ESTIMATED |
|---|---|---|
| Decode `graph_visibility = invisible` | **E**: the same eight cells with `--enforce-eager` added to the arm (a reviewed one-flag change of `SP1-profile.env`). It attributes decode kernel time by bucket; exposed shares are not production's. Start time **unmeasured**, so first measure the start in E and stop if it exceeds 15 min | ≈ 30-40 min |
| Profiler overhead (a) > 10 % at B1 | **R**: repeat the decode captures with `max_iterations` 4 and CPU activity kept (annotations need it); same cells 5-8 | ≈ 28 min |
| A capture `SKIPPED_MEMORY` | **M**: rerun only the skipped cell after the coordinator checks sparkhead's idle MemAvailable; no other change | ≈ 18-22 min |
| Nsight wanted after the torch trace | follow-up lane: launcher change for `nsys profile` at process creation (SP1.report.md 571-591), its own review | not scheduled |

Each contingency uses a new HOLD, a fresh output directory and the same reviewed commit. Cells from different HOLDs are
never pooled.
