# PF2-LOWKV: what a 524K-LOW production profile would have to pass (written, NOT run)

Written 2026-10-26 by lane PF2-LOWKV. This is a gate list for a later lane and window, and only if the PF2 window reads
HEADROOM_IMPLICATED or NO_CAUSE_SHOWN and NDM asks for it. Nothing here is promoted without NDM's word. Production stays
K1 @ 327,680 / 6 GiB (21bed31, prod-k1-327-20261023).

## Profile to be built (not built here)

- **Change.** `offload/promote/k1/` gains a third profile, `524288-low`, with these fields:
  - `K1_MAX_LEN=524288`;
  - `K1_KV_BYTES = <LOW>` (3,221,225,472 unless the window says otherwise);
  - `K1_KV_TOKENS = arith.expected_tokens(LOW)` (1,410,348).
- **Mechanism.** The same mechanism as the existing 327680 / 524288 profiles: `constants.sh` `k1_select_profile`,
  `profiles.py`, a `dsv41.env` patch, `check_runtime.py`.
- **What does not change.** Image, weights, overlays, drafter and env stay as production's.
- **Rollback.** `W8/restore.sh` returns to K1-327680 today. A `524288-low` promotion keeps that path and adds a profile
  switch back to `327680`, verified by `start_checked.sh` ("K1 START VERIFIED" with KV 2,502,600).

## Gates (all must PASS in one promotion window; MEASURED, never ESTIMATED)

| # | gate | method | pass rule |
|---|---|---|---|
| G1 | capacity | engine log | `GPU KV cache size` == `arith.expected_tokens(LOW)` on rank 0; max concurrency at 524,288 ≥ 2.0 |
| G2 | long needles | `offload/kvpack/needle.py --ctx 450000 500000 --reps 2` (6 needles, T=0) | 4/4 requests 6/6 recall, no foreign code; 0 preemptions in the `/metrics` delta |
| G3 | two concurrent long requests | `needle.py --ctx 500000 --concurrent 2` | both 6/6 recall, no foreign code, 0 preemptions; MemAvailable ≥ 4 GiB on both Sparks throughout (1 s telemetry) |
| G4 | corruption set | `needle.py --ctx 8000 30000 50000 100000 --concurrent 4` (K1's set) | every request full recall, no foreign code |
| G5 | greedy identity | the W5 greedy prompts as `W8/restore.sh check` runs them, against K1-327680 production | six "IDENTICAL" lines; a single divergence FAILs |
| G6 | regression vs same-day K1-327680 | unprofiled SP1 control cells (`window_cells` ctl-short-b1, ctl-32k-b1, ctl-150k-b1), ≥ 2 repetitions each, same HOLD or same day after FW1 | short TTFT and tok/s, 32K TTFT and tok/s, 150K TTFT and tok/s each within **2 %** of the 327K control (median of reps) |
| G7 | 524K prefill state | 150K cold ×2 at the start of the promotion arm and again after G2–G4 | both pairs classify "fast" by `analyze.classify` (ratio ≤ 1.12 vs the same-day 327K control) |
| G8 | memory | 1 s telemetry for the whole window | MemAvailable ≥ 4 GiB both Sparks always; idle MemAvailable ≥ the K1-327680 idle level; no swap growth after readiness |
| G9 | rollback | `start_checked.sh` with profile `327680`, then `W8/restore.sh check` | K1 START VERIFIED (2,502,600), six IDENTICAL, PRODUCTION HOLD OK |

## Notes

- **G6 is the binding cost gate.** The PF1 slow state cost +32–36 % at 150K and 300K. A 524K-LOW profile that is fast in
  one window and slow in another fails G7 the first time it is slow. Two windows on different days are needed before G7
  counts as stable.
- **The lower pool is the price.**
  - 1,410,348 tokens against 2,820,756 (H) and 2,502,600 (production 327K).
  - Prefix-cache retention for long documents (AUD1: resident reuse works) halves.
  - G2–G3 prove that two maximum-length requests still fit. Cache retention is reported, not gated.
- **Out of scope.** No 1M, UP2/0075, SO2 or attention change is part of this profile.
