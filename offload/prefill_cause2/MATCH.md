# PF1-CAUSE2 item 3: the matched window (`pf1_window.py --phase match`)

Status: CPU-rehearsed on Controller with fakes; **not run**. Coordinator (Deepseek / coordinator) execution only.
This reading is pre-registered: written 2026-10-26 before any match window. MEASURED / ESTIMATED as in DIFF.md.

## Plain language

- One HOLD, three arms at max_model_len 524,288, each a fresh start: **off-524** (production code, exactly W1/W3's
  OFF arm), **cause-524** (exactly CW's cause arm: PF1-R2 + MoE-half spans, TRANSIENT on), **fix-524** (production
  code + the item-2 fence switch, untraced). Each runs CW's timed protocol: the 32K warm-up, then the 150K cold
  request **twice** (1,500 output tokens, the same prompts and cache salts in every arm).
- After its timed requests, off-524 and fix-524 also answer the six W5 greedy prompts (answer hashes compared 6/6) and
  an 8,192-token cold prompt for the first-token top-20 distribution (off twice = the A-A floor, fix once).
- The runner reads the result by the fixed rule below and always restores production through W8 (start/check/close,
  greedy 6/6 against K1-post), whether the arms complete or any guard stops them.
- ESTIMATED **≈ 42-43 min HOLD**, production down ≈ 42 min.

## Coordinator command (after review and integration; never run by a lane)

```bash
WINDOW_COORDINATOR=Deepseek/coordinator python3 -m offload.launch.windows.pf1_window --execute --phase match \
  --cause-from /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-CW-20261026T024340Z \
  --out /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-MATCH-<UTC> \
  --expected-experiment-commit <reviewed full sha> [--stage-bundle]
python3 -m offload.launch.windows.pf1_window --phase match          # dry plan: no command, network or file
```

`--cause-from` (required with, and only with, `--phase match`) refuses unless window CW is COMPLETE and restored, its
evidence hashes still match, and the trace + cause pins about to be mounted equal the 10 hashes CW's cause-524 mounted on
**both** ranks (so cause-524 repeats CW's bytes). No prior OFF gate is used: off-524 is this HOLD's own control.

## What the runner adds (shared HOLD / guards / restore / reap; existing phases unchanged)

| Piece | Match behaviour |
|---|---|
| Arms | `MATCH_ARMS`: `off-524` = `OFF_ARMS[1]`, `cause-524` = `CAUSE_ARMS[1]` (same identity rules as their phases), `fix-524` (`Arm(fix=True)`, config `PF1-fix.env`) |
| Pins | preflight `MATCH_PIN_FILES` = cause pins + `PF1-fix.env`, `prefill_cause2/{BASES.json,OVERLAY.sha256}` |
| Staging | trace, cause and fix sources built locally, hash-checked, copied to the experiment checkout's `generated/` only if absent; an existing different tree STOPs |
| Identity (fix) | mounts = production + exactly the 4 fence pins (3 MoE files replaced, helper added); env = production + REQUIRED=1 + `VLLM_DSV41_MOE_FENCE=1` (its absence, any trace switch or a production value STOPs); argv port/cap only; in-container verifier **14** sources |
| Every arm | host-parity reads, MemAvailable ≥ 5 GiB before its start, per-arm minima kept in `arm-memory.json` even after a STOP |
| Checks | `offload.prefill_cause2.match_requests` after the timed cells of off/fix: identity row + 6 greedy rows + 2/1 logit rows, hashes and numbers only (no text) |
| Reading | `match_reading()` → `summary.json` `match` (verdict, speed, safety, TTFTs, medians, spreads, thresholds) |
| Restore | unchanged W8 path; reap matches `start_exp.sh` with any of `PF1-(window|cause|fix).env` and the window's own HOLD opener |

Guards unchanged: MemAvailable ≥ 4 GiB on both Sparks always, swap growth after readiness, telemetry liveness, foreign
demand on :8001, STOP file, 75-min submission / 80-min abort bounds, 8-min arm budget check before each start.

## Pre-registered reading (fixed; `MATCH_THRESHOLDS`)

Let `off`, `cause` be the median 150K TTFT of off-524 and cause-524 (2 reps each, same HOLD).

| Verdict | Condition (all reps) | Meaning | Next |
|---|---|---|---|
| **CONFIRMED** | every arm's rep spread ≤ 3 %; both cause reps ≤ 0.80 × off; both fix reps within ±5 % of cause **and** ≤ 0.80 × off; greedy 6/6 identical and logit floor PASS | the event records in the MoE half are the effective difference (item 1 D1); the switch removes the 524,288 gap | K1-PROMOTE memory/promotion rows at 524,288 with the switch; 300K points; then narrow the 25 sites |
| **REFUTED** | spreads ≤ 3 %; cause reproduces (as above); both fix reps ≥ 0.95 × off | the records alone do not do it; the trace runtime's host work (D2-D5) does | fix candidate 2: the host-side part (bookkeeping pace / allocator read) as its own switch |
| **CAUSE_NOT_REPRODUCED** | spreads ≤ 3 %; a cause rep > 0.80 × off | CW's fast 524,288 was state, not code | the alternating off/cause arm series (state dependence) |
| **UNSAFE** | greedy < 6/6 or logit floor fails (any timing) | the switch changed outputs: never adopt | diagnose; no promotion |
| **AMBIGUOUS** | anything else (incl. spread > 3 %, fix between the bands, checks not run) | no decision | repeat or redesign |

A check artifact that exists but is incomplete (not 6 greedy rows, not a/aa + b logit rows, different logit prompts)
raises: the window STOPs with no verdict, after the timed cells are already saved in `summary.json`, and restores.

Safety rule: greedy = the W5 prompts' answer SHA-256 and completion tokens equal for all six, fix vs off; logit =
`up2_checks.numerical(off_a, off_aa, fix_b)`: coarse KL(off ‖ fix) over the top-20 + OTHER ≤ max(0.005, 2 × KL(off_a ‖
off_aa)) and common captured mass ≥ 0.99, same exact 8,192-token prompt, every probe cold (cached_tokens 0).

These are experiment readings, not a promotion. Reference values (MEASURED, other HOLDs): off-524 150K 90.8-95.8 s
(W1, W3), cause-524 69.20 s and T0 69.77 s (CW), off-327 68.9-71.2 s; a CONFIRMED result looks like off ≈ 91 s,
cause ≈ fix ≈ 69-70 s.

## ESTIMATED timeline (MEASURED inputs: W1/W3/CW `cmd/` file times and request `total_s`)

| Step | MEASURED basis | ESTIMATED here |
|---|---|---:|
| preflight, HOLD, snapshot, idle | 0.26-0.32 min (W3, CW) | 0.3 min |
| stage 3 overlays + first stop/start | first arm start incl. staging 6.51 (W3) / 6.97 min (CW) | in arm starts |
| arm start (stop → ready), each | off 6.25-6.51, cause 6.68-6.97 min | 6.5 / 6.7 / 6.5 min |
| identity, pins, parity, logs, each | 0.15-0.17 min | 0.17 min × 3 |
| off-524 warm-up + 150K ×2 | 21.1 + 118.7 + 118.3 s total_s (W3) + ~5 s | 4.4 min |
| cause-524 warm-up + 150K ×2 | 17.1 + 97.6 s (CW, one rep) | 3.6 min |
| fix-524 warm-up + 150K ×2 | as cause (if fast) / as off (if not) | 3.6-4.4 min |
| checks off / fix | 1,800 greedy tokens at ~53 tok/s (CW decode: 1,500 in 28.4 s) + 3/2 short-to-8K prefills | ~1.0 min each |
| collect (+ cause trace copy) | 0.01-0.02 min | 0.05 min × 3 |
| restore start / check / close | 7.35 + 0.32 + 0.03 min (CW) | 7.7 min |
| **HOLD total** | | **≈ 42-43 min** (production down ≈ 42) |

Memory (MEASURED, CW `arm-memory.json`): cause-524 minima 6.22 GiB (sparkhead) / 7.53 GiB (sparkworker); the fence adds no
caching-allocator memory. Every guard line is unchanged.

## CPU rehearsal (`offload/launch/windows/tests/test_pf1_match.py`, 36 tests)

Pure reading (every verdict, unpaired/incomplete cells and checks refuse); fix identity (+ missing switch, trace env,
replaced/missing mount, production switch); 14-pin verifier; `--cause-from` against the real CW files and a tampered
copy; reap patterns; Backend-fake order and **every STOP point** (HOLD, snapshot, staging, each arm's start / requests /
collect) ending in restore (production stopped iff an arm started); restore failure reported; the real `Live` backend
through fake commands: full rehearsal (3 starts, 2 check drivers after the stats, trace copy for cause only, staging of
all three trees, arm memory, reap pattern, HOLD purpose, no answer text in artifacts) and fix-arm faults (check driver
fails, STOP file, interrupt, 80-min bound, 13-pin verify, < 5 GiB start floor, memory/swap/GPU-telemetry trips during
cause-524, different staged fix tree) — every one ends in W8 start/check/close (or check/close before any start);
the check driver itself against fake HTTP (W5 bodies, hashes only, ledger, cold-only refusal, dry run, refusals).

Not covered on CPU: native timing, the real W8 restore, real Spark guards, in-container imports of the fence helper.
