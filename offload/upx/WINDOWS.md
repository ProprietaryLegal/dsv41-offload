# R-UPX windows — 0064, CPU pin, 0085b, 0067 (rule win2-v1, chain mode)

Designs for the four UPAUDIT gaps without an owner (B13, C2, B33, B16; ranked fixes 4, 5, 9). The same designs are
data in `windows.py` (`python3 -m offload.upx.windows plan`); `tests/test_windows.py` checks them against WIN2's
limits and the brief. This lane builds no runner and contacts no host. Every figure below is ESTIMATED unless marked
MEASURED.

## Summary

| Window (chain order) | Starts | Candidate | Decision cells | ESTIMATED HOLD chained / last |
|---|---|---|---|---|
| `UPX-0064` | C1 / X-split / C2 | R-UP3 arm `split` (0064, `VLLM_EXL3_MOE_SPLIT_MAX_TOKENS=63`) | `band-32k-4user`, `band-150k-4user` | 83 / 106 min |
| `UPX-PIN` | C1 / X-pin / C2 | R-UP3 arm `cpu-pin` (`EXL3_CPU_PIN=5-9,15-19` + Round 8 receipt) | same band cells | 78 / 100 min |
| `UPX-0085B` | C1 / X-release / C2 | `UPX.env` `UPX_0085B=file` | `mem-r0`, `mem-r1` (MEASURE-0085B.md) | 66 / 89 min |
| `UPX-0067` | C1 / X-vor / C2 | R-UP3 arm `vor` (0067, `VLLM_DSV41_VOR_FUSED=1`) | `ctl-short-b1`, `ctl-32k-b1` | 58 / 81 min |

All four chained: **308 min** (83 + 78 + 66 + the last window's 81, which includes the MEASURED 7.4-min strict W8
restore and the 15-min watch). `UPX-W1` (C1 / split / C2 / pin / C3, one shared middle control) is 135 / 157 min and is
listed for comparison only: it saves 26 min against the first two windows but makes one 2.3-hour HOLD.

Controls: `C-prod` = R-UP3's `window.env` with `UP3_ARM=control` (production's own env, mounts and bytes, port 8001;
`offload/up3` @ `ac23d4f`, read here, never edited). `C-upx` = `UPX.env` with `UPX_BASE=prod UPX_0085B=0` =
`prod.env` = production's own env and mounts from `~/dsv41-deploy/dsv41.env`, port 8001 (tested against production's
`dsv41.env` at 596ba1c). The R-UP3 `split` bytes equal UP1's 0064 bytes (tested against `upstream_adopt/MANIFEST.json`).

## Why these decision cells (the K5 trap again)

UPAUDIT fix 4 asks for decision cells "whose steps contain 33–63 rows (mixed 150K + decoders, 4-user)". On our stack
the shared cells do not contain such steps, except by chance:

- **MEASURED:** the six W5/K1 greedy prompts are 17, 20, 20, 23, 24 and 27 tokens (`K1-production/greedy-post.jsonl`
  usage). A B1 decode step verifies at most 1 + 5 rows. No step of the standard greedy gate reaches 33 rows, so that
  gate cannot see 0064 at all.
- **Arithmetic (v5 scheduler, production SPEC `[[1,2,5],[3,4,4]]`):** a decode step at four slots has at most
  4 × (1 + 4) = 20 rows. A long prompt beside other requests runs `long_prefill_token_threshold` = 1,024-token chunks
  (the threshold applies with more than one eligible request), a lone one 4,096-token chunks. A step has 33–63 rows only
  when a prefill piece of at most 63 rows minus the decode rows beside it runs: a short or prefix-cached prompt, or a
  prompt's final remainder. `ctl-mixed150k-4user` (three 32K decoders, the 150K arriving late) has four prompts, so at
  most four such steps among ~300, and only when a remainder happens to land in 18–48 tokens (about 3 % per prompt).
  `ctl-32k-b4` is the same. Deciding 0064 on them would repeat the K5 mistake: a test on cells where the arm cannot act.

Upstream's +9 % came from 8 slots (8 × 4 = 32 decode rows, so almost any prefill piece beside them lands in 33–63) and
agentic arrivals. We run 4 slots. `band.py` therefore adds two cells that contain the band by construction:

| Cell | Decoders (`max_tokens`, `ignore_eos`) | Then | ESTIMATED bound |
|---|---|---|---|
| `band-32k-4user` | `32k-1`, `32k-2`, `32k-3` together (1,200 each) | once all three decode: token-id prompts of 36/40/44/48 tokens, `max_tokens` 1, one after another in the fourth slot, until the first decoder finishes | 150 s |
| `band-150k-4user` | `150k` alone first (1,800), then `32k-1`, `32k-2` beside its decode (1,200) | the same short prompts once all three decode | 210 s |

Each short prompt's prefill step has its own L rows plus 3–15 decode rows (three decoders, k = 0..4), 39–63 rows
whatever the decoders' k: one band step per short prompt. The cell counts them (`band_steps`: exact prompt length, no
cache hit, one output token, all three decoders alive) and **refuses below 30 band steps or below 10 % of the decoder
rounds**. ESTIMATED: one short prompt per 1–2 rounds, about 40 % of rounds. Primary metric: the decoders' pooled server
ms per round (`nbw_cells.summarize`'s definition; the short prompts add no decode time and no rounds), which includes
every band step. Guards: accepted per round, the short prompts' median latency.

Production share of band steps at 4 slots is **UNMEASURED**: the band cells measure the per-band-step effect, which
bounds the production effect from above. Legal traffic has few prompts under 64 uncached tokens; prefix-cached
follow-ups are the realistic source.

## Correctness screen (NDM's 0.9-nat gate, replacing the never-built resident-worker harness)

| Gate | Arms | Registration |
|---|---|---|
| `greedy` | every start | `nbw_cells.greedy(..., near_tie_nats=0.9 + 1e-3, record_distribution=True, probe_salt_prefix=<window tag>)` on the six K1 references. Blind to 0064 (above); required as the base identity |
| `band-probes` | 0064: C1, X, C2 | `band.kl_probes`: top-20 one-token probes whose final prefill step has 32, 33, 42, 43, 63, 64 rows, plus 8,232 and 32,808 tokens (two and eight full 4,096-row chunks, then a 40-row step). `band.kl_screen`: KL(C1 ‖ X) ≤ max(0.005, 2 × KL(C1 ‖ C2)) per probe, common mass ≥ 0.99 (UP2's estimator and registration). 33/42 = one exl3_smoe chunk, 43/63 = two; 32/64 = unchanged dispatch |
| `band-greedy` | 0064: C1 (reference), X, C2 (A/A) | `band.token_greedy`: six token-id prompts of 33/38/42/43/52/63 tokens, 300 tokens at T = 0 (`ignore_eos`). X against C1: identical or a first divergence that is a 0.9-nat near-tie (both tokens in the top-20, within 0.9 nats of each other and of the top). The prompt's KV comes from the band step, so every later token depends on 0064's rounding |
| `kl` | the control after each candidate | `greedy_kl.compare_recorded` over the candidate's recorded divergences (six-prompt and band greedy), bound max(0.005, 2 × the band probes' A/A floor) |
| `engagement` | 0064: X present, C1/C2 absent | 0064's once-per-process log line on BOTH ranks (`band.split_engaged`: the logged call's rows must lie in 33–63) |
| `census` | CPU pin: every start | R-UP3's Round 8 startup receipt + direct thread census (S84 cleared); in the controls it records the default masks |
| `long-greedy` | 0067: C1 (reference), X, C2 | `band.token_greedy` on 32,000- and 150,000-token prompts: decode at long context runs 0067's fused SWA metadata for every group and its replay-start copy |
| `needle-150k` | 0067: X | `up2_request.needle(target=150000)`: recall 6/6, no foreign codes, cached 0, finish `stop` |
| `release-census` | 0085b: X | MEASURE-0085B.md |

A failed gate stops that arm (the runner records why and keeps the controls); a failed C1 identity is a window stop.

## Decisions (decision_v2 unchanged; WIN2-R2 merges in)

- 0064, CPU pin, 0067: C1 + C2 = 4 control repeats pooled against 3 candidate repeats per decision cell, **unpaired**
  (Welch). `windows.series(group, cell, metric, control, candidate)`; `decide_window(series, group.decision_cells,
  groups=group.groups)` (groups: 32K vs 150K context; short vs 32K for 0067).
- 0085b: three ON/OFF battery pairs inside one engine, **paired** (adjacent pairs, as WIN2-R2 requires for pairing).
- `windows.arm_verdict`: every gate passes and status CANDIDATE → **CANDIDATE FOR PROMOTION**; INCONCLUSIVE → **NO GAIN
  SHOWN (stays off)** (these are speed or memory claims, not fixes); NOT A CANDIDATE; any failed gate → STOP.
- WIN2-R2 (coordinator, 2026-10-27): a guard is a LOSS only when its whole 95 % interval is below zero; paired loss
  tests use the spread of paired differences. Both arrive with the R2 merge; no change here.
- Archive: every control repeat of every decision cell (band cells under their own names with `max_tokens` = the
  largest decoder's, 1,200 / 1,800; B1 cells at 512; `battery-150k` at 16) with WIN2-R2's provenance (production
  commit, proven clock cap, date, start, repeat).

Expected outcomes (honest): 0064 is unknown per band step here; a WIN in the band cells shows the per-step gain, not
the production gain. CPU pin: RS1 ESTIMATED 0.5–1.5 %, may not clear 2 %. 0067: RS1 ESTIMATED 0–0.7 % per round, so
INCONCLUSIVE is the likely result and the window may be skipped when HOLD time is short. 0085b: memory gains far above
2 % if UPSTREAM holds.

## Budget (per start: MEASURED 6.8-min start + steps; 5 min fences per window)

| Step | Seconds | Basis |
|---|---|---|
| warm-up | 60 | catalogue |
| greedy | 60 | ESTIMATED: 6 × 300 tokens at MEASURED 51–57 tok/s B1 |
| regions | 30 | ESTIMATED: four `/tokenize` calls over 772K public characters |
| band-probes | 60 | ESTIMATED: eight probes, the long two prefill 8K and 33K tokens |
| band-greedy | 60 | ESTIMATED: 6 × 300 tokens at ~51 tok/s |
| long-greedy | 120 | MEASURED 32K TTFT 15 s + 150K TTFT 69 s, ~12 s decode |
| needle-150k | 120 | MEASURED 150K TTFT 69 s + answer |
| kl | 60 | ESTIMATED; zero work if no divergence |
| engagement / census | 15 / 30 | ESTIMATED / R-UP3's host-evidence reserve |
| `band-32k-4user` | 150 | MEASURED 32K B3 TTFT ≤ 51.7 s + ~60 s band phase (ESTIMATED ~20 tok/s per decoder) |
| `band-150k-4user` | 210 | MEASURED 150K TTFT 69 s + ESTIMATED 25 s for the two 32K prefills + ~60 s band phase |
| `ctl-short-b1` / `ctl-32k-b1` | 60 / 90 | catalogue |
| battery (0085b) | 200 | two MEASURED 69 s 150K prefills + 20 s + 15 s idle + file writes and log reads |

Per-start totals (`windows.plan()`): 0064 C1 23.6, X 29.6, C2 24.6 min; PIN 21.8 / 27.8 / 22.8; 0085B 15.5 / 29.1 /
16.5; 0067 15.8 / 20.3 / 16.8.

**Ceiling:** R-UP3's runner refuses a plan over 85 min and stops submissions 73 min after the HOLD opens. `UPX-0064`
(78 min of starts) does not fit that submit ceiling as written and `UPX-PIN` (72 min of starts) sits at it; in chain mode the
coordinator either lifts it for chained windows (R-UPF's chained designs are 142 min) or runs the compact variant
(C1 with one repeat per band cell: control repeats 1 + 2 = 3, the WIN2 minimum; saves 6 min).

## Chain mode (WIN2.md, WIN2-R2)

`WINDOW_CHAIN=1` for every window, `WINDOW_CHAIN_LAST=1` on the last. WIN2-R2 (2b48eec): `chain.preflight(...,
placeholders_present=...)` at each start; in `'chained'` mode the production identity is `chain.production_snapshot()`,
C1 is the measurement baseline, the admission HOLD opens with `chain.allow_unreachable(mode)` and both Sparks are proven
quiet before `start_exp.sh`. The end follows `chain.end_action`: `'chained'` = stop, quiet proof, `chain.CHAINED_CLOSE`
in the window's `-restore-close` scope + commit-close, `chain.finish_chained`, no W8; `'restore'` (any stop with an own
HOLD, or the last window) = the strict six-IDENTICAL W8, the watch and `chain.clear()`; `'abort'` (marker, no own HOLD)
= `chain.abort`.

**Placeholder finding (R-UPX, 2026-10-28; `tests/test_chain_placeholders.py`, real user scopes, no host contact).**
`scoped_remote`'s tmux shim puts the `dsv41-hold` placeholders that `hold_admission.open_script` makes on a private
socket inside the hold phase's scopes. The chain gate's probe (default socket) never sees them, `CHAINED_CLOSE` run in
the `-restore-close` scope checks yet another private socket and always refuses, and fencing the hold phase kills them.
Until the shared harness keeps the placeholders outside the phase scopes, a chained end cannot succeed for any scoped
runner; the UPX runner therefore checks both views first (`Live.chain_ready`) and, when either is missing, runs the
strict W8 instead (production back, chain ended cleanly). Chained HOLD estimates below assume the fix; until then every
UPX window costs its "last window" figure. Recommended order: `UPX-0064`, `UPX-PIN`, `UPX-0085B`, `UPX-0067` (UPAUDIT
priority; 0085b before 0067 because 0067 is likely INCONCLUSIVE). If 0064 is a CANDIDATE, a later window should run
CPU pin on top of 0064 (upstream measured them together); that window is not designed here because its base depends on
0064's promotion.

## What a runner must add (not done here: shared or R-UP3-owned modules)

1. R-UP3 `up3_window` (or the WIN2 chain runner): admit `split` and `vor` under a new registration (`upx-0p9`) whose
   gates are the table above, instead of the native adapter that parks them before HOLD (UP3 R4). Today both refuse at
   execute admission and "cannot be unparked with a qualifier flag".
2. The same runner: run `band.BAND_CELLS` (`band.run_band_cell`), the band probes / band greedy / long greedy, build
   `band.regions` once per window and verify them in every start (`band.verify_regions`), and read both ranks' docker
   logs for `band.split_engaged`.
3. ~~For `UPX-0085B`~~ **built here:** `battery_window.py` (runner) + `battery_cells.py` (request driver). It subclasses
   `nbw_window.Live` (shared HOLD admission, scoped ownership, quiet proof, strict W8, post-close watch) without the
   native pair, writes and reads back the threshold file on both ranks through owned scoped commands, keeps the 1 s
   telemetry, collects `docker logs --timestamps`, decides with `measure_0085b` + decision_v2, archives the control
   batteries with WIN2-R2's schema (per-start `clock_cap_mhz` from its start log, `YYYYMMDD` date, production commit,
   archive head before and after), and follows WIN2-R2's chain rules above (`end_action`, `CHAINED_CLOSE` after
   `chain_ready`, `chain.abort`). Proposed launch: `scratch/RUPX/launch/upx_0085b.sh`.
4. R-PC1 (M2 only): end `offload/pc1/window.env` with `[ "${UPX_0085B-0}" = 0 ] || source offload/upx/x0085b.env || return 1`
   and run three battery pairs in an extra chunk16384 start.
