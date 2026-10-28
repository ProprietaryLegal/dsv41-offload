# PF1-WINDOW item 2: the smallest matched window that attributes the 524,288 prefill cost

Status: design. No GB10 numbers are produced by this lane. MEASURED inputs are cited; every duration in the
plan below is **ESTIMATED**.

## The question, and what answering it needs

S18 found a flat **+0.163 ms per prompt token** in cold prefill at max_model_len 524,288, compared with 327,680:

| Prompt | 327,680 | 524,288 |
|---|---|---|
| 150K | 0.4595 ms/token | 0.6224 ms/token |
| 32K | 0.4547 ms/token | 0.6175 ms/token |

Decode is unchanged. The cost is the same per token at both lengths. Per scheduled 4,096-token chunk that is about
+0.67 s on a ~1.88 s chunk (ESTIMATED, from the MEASURED per-token figures). The window must answer two things:

1. **Does the gap still exist on today's production?** Today's stack is TP2 drafter, K1 21bed31 and process proxy.
   S18 ran on the one-card worker on 2026-10-23. This needs an uninstrumented, matched OFF pair.
2. **Which prefill span carries the extra per-chunk time?** This needs the PF1-R2 trace at both caps at matched prefix,
   on both ranks (`fit_cost`: ≥ 3 common full-chunk prefixes per cap and rank).

One 150K cold request gives 36 full 4,096-token chunks at prefixes 0-143,360. That is far more than the fit's
3-prefix minimum, and it spans the range where S18 measured the flat cost. 300K adds within-cap slope leverage for
the headline TTFT. The OFF pair runs it (it is also W13's measured point). The traced pair does not need it.

## Design (one HOLD, automatic gate)

| Order | Arm | Cap | Trace | Requests (all B1, cold, cache_salt pairs fixed per window) |
|---|---|---|---|---|
| 1 | `off-327` | 327,680 | off | warm-up 32K (64 tokens, discarded); 150K ×2; 300K ×2; 1,500 output |
| 2 | `off-524` | 524,288 | off | same cells, same prompts (character cut reused, recounted), same salts |
| — | **gate** | — | — | CPU: REPRODUCED → continue; NOT_REPRODUCED / AMBIGUOUS → restore now |
| 3 | `on-327` | 327,680 | on | warm-up 32K (the first traced prefill: an eager/graph refusal fails here, cheaply); 150K ×1 |
| 4 | `on-524` | 524,288 | on | same |
| — | restore | — | — | `W8/restore.sh start`, `check`, `close` (always; also on every STOP) |

What every arm holds equal:

- production's mounted bytes and env (`PF1-window.env`);
- the remote TP2 drafter (positions ≤ 301,500);
- the public corpus, header and order, seed 42, T1, top_p 1, priority 10, thinking high;
- the 1,500-token output reserve (TTFT comparable with S18 and W13);
- KV bytes;
- clocks: the existing cap of 2,150 MHz, which `start_exp.sh` re-applies at the same value as production's own start.
  No clock or power change.

What is **not** in the window, and why:

- **400/450/500K points.** Not needed to attribute a per-token cost already flat from 32K to 150K. They would also
  push the TP2 worker past the 327,680 positions it has served (S22 FLAG D).
- **Warm and partial-warm modes.** Cache hits remove the prefill being attributed.
- **A second ON repetition.** Each 150K traced request already gives 36 per-chunk samples per rank.
- **The return arm `off-327-return`.** At 300K, W13's paired OFF-A/OFF-B start-to-start spread was 0.025-1.53 %
  (median 0.105 %), MEASURED in `UP2-screen-300K-20261025T131508Z/summary.json`. S18's gap is 35 %. Drift of that
  size cannot turn a ≥ 10 % gap into "no gap", nor the reverse. The return arm only matters when the gate is
  AMBIGUOUS, and then it runs in its own HOLD (below).
- Excluded by the brief: attention changes, 0075/UP2, SO2, clock or power changes.

### Gate (decided by the runner from the OFF cells; MEASURED TTFT only)

For n in {150,000, 300,000}:

- `gap_n = median(off-524 TTFT) / median(off-327 TTFT) − 1`
- `spread = |rep0 − rep1| / mean`, for each arm and context
- `excess_ms_per_token = (median524 − median327) / n`

The excess is reported alongside S18's 0.163.

| Gate | Condition | Next |
|---|---|---|
| REPRODUCED | gap ≥ 10 % at both n, every spread ≤ 3 % | ON pair in this HOLD |
| NOT_REPRODUCED | gap < 5 % at both n, every spread ≤ 3 % | restore. The 524,288 cost is absent today. Next: S18's R5 gate rows for promotion (coordinator) |
| AMBIGUOUS | anything else, including any noisy arm | restore. Next: HOLD 1b, a return-arm design (below) |

Why these thresholds:

- **10 %** is more than 6 × W13's largest measured start-to-start spread, and far below S18's 35 %.
- **5 %** is NDM's speed-loss rule (S18 verdict).
- **3 %** is about 2 × W13's largest paired spread.

These are window-design thresholds, not promotion gates. K1's promotion gates are unchanged.

## MEASURED inputs used for the estimate

| Input | Value | Source |
|---|---|---|
| Start to ready, 327,680 | 6.77, 6.82 min | W13 300K window file times (`UP2-screen-300K-…`: 0019→0020, 0110→0112) |
| Start to KV-verified, 524,288 | 6.7 min (ready after 361 s) | `offload/results/K1/W6b-runner.log` 18:08:13 → 18:14:55 |
| Post-start identity checks | ~13 s | W13 0020 → 0039 |
| 32K warm-up incl. prompt build | 33 s (32 tokens) | W13 0039 → 0041 |
| 150K cold, 327,680 | TTFT 68.6-69.0 s, full 96.0-98.0 s | REQUEST-EVIDENCE.md (R1, K1 production) |
| 150K cold, 524,288 | TTFT 93.34-93.37 s, full 120.2-120.6 s | REQUEST-EVIDENCE.md (R5) |
| 300K cold, 327,680 | TTFT 139.6-140.5 s, full 166.9-170.7 s; cell incl. checks ~190 s | W13 `off-a-cold300000-*.json`; R1 c300 140.2 s |
| 300K cold, 524,288 | **none measured**. ESTIMATED 140.2 + 300,000 × 0.163 ms = 189 s TTFT, ~216 s full | flat-cost extrapolation (S18) |
| Prompt build | 19-33 s per exact prompt (20-23 /tokenize calls) | S30 via UP2-LOW.md budget section |
| Restore (start + check + close) | 7.8 min | W13: runner end 14:08:12Z → closed 14:15:59Z (LANES.md 2026-10-25 14:16) |
| Trace overhead | **not measured** | ESTIMATED ≤ 20 % on traced requests |

## ESTIMATED minute-by-minute plan (T = HOLD written)

Before the HOLD, with production still serving, about 1-2 minutes of read-only work:

- runner preflight: checkouts, ledger, production health and idle, worker, D2 pause, corpus, nvidia-smi probe;
- memory and GPU telemetry starts;
- bundle staging, only if the remote checkout is not yet at the reviewed commit.

| T (min) | Step |
|---:|---|
| 0.0 | `window.sh open` (production idle: immediate; drain bounded at 20 min, otherwise no HOLD) |
| 0.0-0.3 | production identity snapshot, both ranks (mounted hashes, env, argv, Engram, image bases); tunnel up |
| 0.3-7.1 | `off-327` start (`stop.sh` → production down from here) |
| 7.1-7.4 | identity = production except port/cap/REQUIRED; 13-pin verify both ranks; KV 2,502,600; proxy, Engram, worker ready |
| 7.4-8.0 | warm-up 32K (64 tokens, drafted rounds > 0) |
| 8.0-11.9 | 150K build 0.5; cold ×2 at ~1.7 each |
| 11.9-18.2 | 300K build 0.5; cold ×2 at ~2.9 each |
| 18.2-18.4 | clean checks (logs, counters, container unchanged, worker) |
| 18.4-25.1 | `off-524` start |
| 25.1-26.0 | identity checks (KV 2,820,756, /v1/models 524288); warm-up (cuts reused: one /tokenize each) |
| 26.0-30.3 | 150K cold ×2 at ~2.1 |
| 30.3-38.0 | 300K cold ×2 at ~3.7; clean checks |
| 38.0 | **gate**. NOT_REPRODUCED / AMBIGUOUS → restore 38.0-45.8, HOLD ends ≈ **46 min** |
| 38.0-38.3 | stage hash-checked trace sources; per-run trace directories on both nodes |
| 38.3-45.1 | `on-327` start |
| 45.1-46.1 | 17-pin verify both ranks; mounted hashes = production with the 2 targets replaced + 4 added; warm-up (first traced prefill) |
| 46.1-48.6 | 150K traced cold ×1 (~1.9 with ≤ 20 % overhead); collect both ranks' trace files; event completeness |
| 48.6-55.4 | `on-524` start |
| 55.4-59.4 | checks, warm-up, 150K traced ×1 (~2.4), collect |
| 59.4-67.2 | restore start/check/close (production back ≈ 66.5) |

| Outcome | HOLD | Production not serving | Submissions | Prompt tokens |
|---|---:|---:|---:|---:|
| REPRODUCED (full design) | ≈ **67 min** | ≈ 66 min | 14 | 2,228,000 |
| gate stops after OFF pair | ≈ **46 min** | ≈ 45 min | 10 | 1,864,000 |

Bounds:

- **Submissions** close at T+75. The last planned submission is at ≈ T+56.5, leaving 18 min of margin.
- **Measurement abort** is at T+80: the supervisor STOPs, and the runner then restores.
- **Restore** needs about 8 min, so the HOLD ends by about T+90 even on an abort at the bound.
- **Arm starts** are refused unless the start plus warm-up (~8 min, ESTIMATED) fits before T+75.
- **Requests** keep their 900 s wall bound.

By comparison, the old recipe had:

- five arms;
- five points × three cache modes × 2 repetitions;
- about 88 submissions plus warm-ups;
- an ESTIMATED 150-190 min, with a 150/180-minute budget.

That exceeds one HOLD and would have needed two or three reservations.

### Optional extension (coordinator flag, not default)

`--on-contexts 150000 300000` adds a traced 300K request to each ON arm. That costs +8 min, for a HOLD of ≈ 75 min,
with the last submission at ≈ T+64. Use it only if the 150K trace shows a prefix-dependent step that needs the
150-300K range.

## Contingency HOLDs (only when the previous result calls for them)

| Trigger | HOLD | ESTIMATED |
|---|---|---|
| Gate AMBIGUOUS | **1b**: `off-327` → `off-524` → `off-327-return`, 150K/300K ×2, then decide | ≈ 64 min |
| ON warm-up refuses (graph-mode prefill) or the ON pair cannot be fitted | **2e**: matched eager `off-327`, `off-524`, `on-327`, `on-524` (`PF1_EAGER=1` on all four), 150K. Eager OFF (150K ×2) must reproduce the gap before its trace explains it | ≈ 55 min (eager decode is slower; 150K full ~+20 %, ESTIMATED) |
| ON arms in their own HOLD (an OFF-only HOLD already REPRODUCED) | **2**: `on-327` → `on-524` with `--phase on --gate-from <prior summary>` | ≈ 31 min |

Each contingency uses a new HOLD, a fresh output directory and the same reviewed commit. Arms or cells from
different HOLDs are never combined.
