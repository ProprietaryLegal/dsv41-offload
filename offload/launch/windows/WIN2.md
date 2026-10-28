# WIN2 — fairer, deeper measurement and back-to-back windows

NDM 2026-10-27: "Fairer win rule is essential", "Do better deeper testing and measuring", "go window after window".
Method source: Fable review (roadmap section "Independent review 2026-10-27 (Fable)", e8e3e3a).

## Rule `win2-v1`, revision 3 (`decision_v2.py`)

Revision 2 (WIN2-R2, S92 FIX 3) and revision 3 (WIN2-R3 item 0, coordinator decision on the R2 report: the primary LOSS
also needs its interval below zero) keep the name `win2-v1` so runner code keeps working; every decision records
`rule_revision: 3`.

| Question | Answer |
|---|---|
| What decides | At most **two decision cells** per window; every other cell is reported only. |
| Primary metric | First series per cell. Prefer **pooled server ms per round** (`server_decode_ms / spec_request_rounds`) for decode cells and **TTFT** for prefill cells. Accepted/round and client tok/s are guards (they can only cause a LOSS). |
| Repeats | ≥ 3 per arm in every decision cell (refused below). |
| WIN | point gain > 2 % **and** the 95 % interval excludes 0 (paired Student t; Welch t for unpaired arms; floored df). |
| Loss-repeat test | the candidate is worse than the loss threshold in ≥ ceil(2n/3) repeats (2 of 3). Control spread = **1.69 × the unbiased control SD (s / c4(n)) / mean**, the expected range of three values whatever the number of control repeats (rev 1 used the range of all controls, which widened with more repeats: a true 5 % loss was caught 74 % of the time at 3 v 3 but 28 % at 8 v 4). Threshold = that spread unpaired, **√2 × it paired** (a paired difference has √2 × the SD of one value). |
| Primary LOSS | the loss-repeat test fires **and** the whole 95 % interval is below zero (revision 3; revision 2 needed the loss-repeat test only). |
| Guard LOSS | the guard's loss-repeat test fires **and** its whole 95 % interval is below zero. |
| Window | any decision-cell LOSS → NOT A CANDIDATE; ≥ 1 WIN and no LOSS → CANDIDATE; else INCONCLUSIVE. Optional `groups` give separate single-user / multi-user verdicts. |

**When to pair.** `Series.paired` defaults to True, but pair only true adjacent pairs: K/I/K/I alternation, one control
and one candidate repeat per pair in run order (or a bracket where control repeat r is the mean of the two bracketing
starts' repeat r). C/X/C with all control repeats pooled, and any layout with unequal repeats, is unpaired
(`paired=False`, Welch). Unpaired, schedule ≥ 4 control repeats (C/X/C with 2 per C): at 3 v 3 the no-effect false
loss per cell is 8 %.

**No-effect rates** (the candidate has no real effect; two decision cells, primary ms/round + both guards; S92's
`sim_corr.py` model, seed 5, 10,000 windows: ms/round cv 1.5 %, acceptance cv 12 %, tok/s derived from both;
`scratch/WIN2R3/sims/sim_rev3.py`, revision 2 in `scratch/WIN2R2/sims/`). "Rejected" = NOT A CANDIDATE, "accepted" =
CANDIDATE:

| design | rev 3 primary only: rejected / accepted | **rev 3 with both guards** | rev 2 with both guards | rev 1 with both guards |
|---|---|---|---|---|
| paired 3 v 3 | 1.7 % / 2.0 % | **4.0 % / 1.9 %** | 11.4 % / 1.8 % | 32.5 % / 1.4 % |
| unpaired 4 v 3 (C/X/C) | 2.7 % / 2.1 % | **6.2 % / 2.0 %** | 14.6 % / 1.9 % | 16.0 % / 1.8 % |

A no-effect candidate is now about as likely to be accepted as rejected on the primary alone; the guards add the rest of
the rejections (they have their own two chances per window).

**Power per decision cell** (one cell, primary ms/round, cv 2 %, seed 9, 20,000 trials; LOSS % for a true loss, WIN %
for a true gain; revision 2's LOSS % in brackets; WIN is unchanged by revision 3):

| design | true loss 5 % | true loss 3 % | no effect: LOSS / WIN | true gain 3 % | true gain 5 % |
|---|---|---|---|---|---|
| unpaired 3 v 3 | 41.6 % [74.6] | 16.8 % [45.3] | 1.3 % [8.4] / 1.2 % | 17.3 % | 45.1 % |
| unpaired 4 v 3 | 53.3 % [76.7] | 22.2 % [43.6] | 1.4 % [5.8] / 1.4 % | 26.4 % | 62.1 % |
| unpaired 6 v 3 | 55.7 % [80.1] | 22.6 % [42.7] | 1.0 % [3.7] / 1.9 % | 35.7 % | 69.8 % |
| unpaired 8 v 3 | 55.3 % [81.5] | 21.7 % [42.0] | 0.7 % [2.8] / 2.1 % | 37.8 % | 70.0 % |
| unpaired 8 v 4 | 67.5 % [72.7] | 22.6 % [27.9] | 0.4 % [0.8] / 1.8 % | 51.6 % | 90.6 % |
| paired 3 v 3 | 29.3 % [55.4] | 11.3 % [30.0] | 0.8 % [4.8] / 1.8 % | 19.5 % | 42.6 % |
| paired 4 v 4 | 37.6 % [42.7] | 12.2 % [16.8] | 0.4 % [1.0] / 1.7 % | 32.3 % | 69.3 % |
| paired 6 v 6 | 44.4 % [44.7] | 12.6 % [13.6] | 0.2 % [0.3] / 1.3 % | 55.1 % | 93.8 % |
| paired 8 v 8 | 28.8 % [28.8] | 4.6 % [4.6] | 0.0 % [0.0] / 1.1 % | 71.8 % | 99.1 % |

What the change costs: at three candidate repeats a real 5 % loss is now caught about half the time (unpaired 4–8
controls) instead of about three quarters; the remainder is INCONCLUSIVE, not CANDIDATE (a WIN needs the interval above
zero, which a true loss essentially never gives: 0.0 % in every row). With ≥ 4 candidate repeats the interval condition
rarely binds (8 v 4: 67.5 % vs 72.7 %). More controls stop helping past 6 because the Welch interval is then limited by
the three candidate values; adding candidate repeats is what raises power. LOSS and WIN are now symmetric in their
evidence: both need the 95 % interval to exclude zero, so a no-effect cell's chances split about evenly (≈ 1–2 % each).
At paired 8 v 8 the 2-of-3 count needs six of eight differences beyond √2 × spread, which is why its LOSS power is
lower than at 6 v 6.

Why ms/round: across the control archive (WIN2's 540 control rows from 10 windows), ms/round moves **0.7–5.0 %** from day to
day within one cell definition. Client tok/s moves up to 11 % and accepted/round up to 23 %, because acceptance per
round depends on the sampled text (T = 1). Threshold2048's short cell is the example: ms/round was 3.4 % better while
tok/s was 13.5 % worse, because that run's acceptance happened to be 28 % lower.

## Seeded paired cells, rule `win2-paired-v1` (WIN2-R3; F-METHOD §3.1)

**Cells** (`window_cells.SEEDED_CELLS`, `run_seeded_cell`): one prompt per member, ONE cold prefill per prompt under the
cell's cache salt (proved cold: `cached_tokens == 0`), then n >= 6 seeded continuations of 256 tokens on the warm prefix
(proved warm: `cached_tokens > 0`; `ignore_eos`, so every request is exactly 256 tokens). Seeds are registered per cell
(`seed_base + 10 x pair + member`), identical in every arm. B1 cells scrape the server counters around EACH request
(rounds, accepted drafts, decode seconds per request; the request-success counter must move by exactly one, so a foreign
request refuses the cell); B >= 2 cells send groups of B concurrent requests and the group is the pair unit. Each pair
records decode tok/s, ms/round, accepted/round, TTFT and a sha256 of every output (no text).

| cell | members | profile | ESTIMATED time per arm (6 pairs) |
|---|---|---|---|
| `seeded-short-b1` | short | single | ~38 s |
| `seeded-32k-b1` | 32k-0 | single | ~56 s |
| `seeded-32k-b2` | 32k-1, 32k-2 | multi | ~88 s |
| `seeded-32k-b3` | 32k-1..3 | multi | ~120 s |

**Rule** (`decision_v2.decide_paired_cell` / `decide_paired_window`; `seeded_series` builds the inputs from C1, X, C2):
- d_i = oriented relative difference of candidate pair i against control pair i (C/X/C: the mean of C1_i and C2_i).
- Paired t 95 % interval AND the exact two-sided sign-flip p (n = 6: the smallest p is 0.031, so all six pairs must
  agree; 3 pairs can never reach 0.05). n >= 6 is refused below.
- Primary WIN: point > 2 %, lower bound > 0, p < 0.05. LOSS (primary or guard): upper bound < 0, p < 0.05. Guards never
  win. The rule is symmetric apart from the 2 % a WIN must also clear.
- Built-in A/A (C1 vs C2): A/A drift (interval excludes 0 with p < 0.05) makes the cell INCONCLUSIVE; a target-side lever
  (`expect_identical`) also needs identical token streams and round counts in both control starts
  (`window_cells.streams_identical`), else INCONCLUSIVE.
- **Start drift.** Every pair shares ONE candidate start, so a start offset looks exactly like an effect. Register the
  between-start SD the seeded A/A pilot measures (`start_sd={'<cell>/<metric>': sd}`); it widens the interval by
  start_sd^2 x (1 + 1/k). Until the pilot has run, the rule assumes 0.
- Profiles: every cell is `single` (B1) or `multi` (B >= 2); the window reports one verdict per profile (at most two
  decision cells per profile); promotions state which profile they rely on.
- Primary metric: decode tok/s for single-stream decode cells (seed luck cancels in the pairs), ms/round reported;
  TTFT for prefill cells (no seeds needed). PP4's drafter has its own RNG: its pairs are only partly correlated, so
  report the measured diff CV, never assume the target-side figures.

**Operating characteristics** (per decision cell, simulated: seed luck 7 % shared by the arms, per-request residual
1 % or 2 %; `scratch/WIN2R3/sims/sim_paired.txt`, `sim_paired_drift.txt`; WIN / LOSS %):

| design, residual, between-start SD | n | no effect | true +3 % | true +5 % | true -3 % |
|---|---|---|---|---|---|
| C/X/C, 1 %, 0 | 6 | 0.0 / 1.3 | 91.0 / 0 | 97.3 / 0 | 0 / 93.7 |
| C/X/C, 1 %, 0 | 12 | 0.0 / 2.2 | 95.1 / 0 | 95.3 / 0 | 0 / 95.3 |
| C/X/C, 2 %, 0 | 6 | 0.6 / 1.3 | 44.5 / 0 | 83.7 / 0 | 0 / 48.4 |
| C/X/C, 2 %, 0 | 12 | 0.2 / 2.2 | 87.0 / 0 | 95.3 / 0 | 0 / 92.7 |
| C/X/C, 1 %, 1 % (not registered) | 6 | **4.5 / 10.0** | 53.6 / 0.1 | 70.8 / 0 | 0 / 61.1 |
| C/X/C, 1 %, 1 % (registered start_sd 1 %) | 6 | 0.4 / 0.4 | 27.9 / 0 | 63.1 / 0 | — |
| C/X/C, 2 %, 1 % (registered start_sd 1 %) | 12 | 1.0 / 1.0 | 32.6 / 0 | 63.7 / 0 | — |
| C/X (one control start), 1 %, 1 % | 6 | **8.2 / 13.5** | 69.8 / 0.2 | 95.6 / 0 | 0 / 77.2 |

Read: with no start drift the rule is fair (false WIN <= 0.6 %, false LOSS 1.3-2.2 %) and resolves 3 % at n = 12;
the A/A check costs ~5 % power (it is itself a 5 % test). Unregistered start drift makes false calls common, and C/X
(no A/A) is worse: keep C/X/C and register start_sd from the pilot.

## Process rules as code (WIN2-R3; F-METHOD §3.2; adoption is NDM's decision 1)

1. **Smoke start** (`smoke_start.py`). Per candidate arm: start_exp.sh, /health + /v1/models, one greedy prompt, one
   64-token seeded request whose speculative rounds must move, then the arm is ALWAYS stopped (a stop failure raises:
   HOLD stays, restore). ~8.5 min per arm (6.8 min MEASURED start), bound 15 min. `chain_head(live, arms)` smokes every
   candidate at a chain's head and drops failures before their windows; `x_first(live, arm)` makes the smoke the
   window's first start, and FAIL ends the window before any control start. The host side is the runner's Live
   backend (`smoke_start.Backend`); `python -m offload.launch.windows.smoke_start <experiment> <arm> [--mode chain-head]`
   prints the plan, and `--execute --backend module:factory --out DIR` runs it (coordinator only). Smoke windows never
   count toward the cap and never write the archive. No false-call rate: it is a pass/fail plumbing check (as are the
   cap and the filter below: neither judges a measurement).
2. **Two-window cap** (`window_cap.py <LANE> [--prefix P]`; launchers call it; exit 0 ALLOW, 2 REFUSE). It counts the
   lane's windows since adoption (`<campaign>/WINDOW-CAP.json` {"adopted_utc": ...}, or `--since`) whose summary
   `status` is COMPLETE, INCONCLUSIVE, NOT A CANDIDATE or CANDIDATE. Windows that stopped (INCOMPLETE, whatever their
   pre-filled decision says), admission windows and smoke starts do not count. From the third window on, it needs a new
   hypothesis dated after the latest counted window: `experiments/<LANE>/hypothesis-<n>.md` with a UTC timestamp in its
   first lines, a register line `HYPOTHESIS <LANE> ... <UTC>` in LANES.md, or NDM's `WINDOW-CAP OVERRIDE <LANE> <UTC>`.
   On the 2026-10-27 history (since 2026-10-26), PC1 and UP3 would already need a hypothesis; NBW, PP4 and DEP1 would not.
3. **2x resolution filter** (`resolution.py --cell --metric --design unpaired:AvB|pairs:N|seeded:N --expected-gain
   --max-tokens --clock --commit [--diff-cv]`; exit 0 QUEUE, 2 REFUSE). The half-width comes from the archive per
   definition (cell, metric, max_tokens, clock cap, commit): the within-window CV of one repeat pooled over windows
   (df-weighted, all control repeats of a window). Seeded designs take the pilot's measured paired-difference CV
   (`--diff-cv`), else the ESTIMATE ms/round CV x sqrt(2). On today's archive (512 tokens, 2150 MHz, 596ba1c) the CVs are
   ms/round 2.55 % short-b1 and 1.07 % 32k-b1, decode tok/s 6.8 % and 6.7 % (F-METHOD's table used 2.1 % / 1.0 % /
   7.1 % / 6.5 %). So the half-widths are a little wider than F-METHOD's: seeded 6 pairs ±3.8 % short-b1 and ±1.6 %
   32k-b1. A +3 % lever on 32k-b1 needs 8 seeded pairs (±1.3 %), not 6.

## Distributional gate for candidates (WIN2-R3; F-METHOD §3.4; `dist_gate.py`)

- **Probe.** 64 registered public-corpus prompts (32 short, 24 at 32K, 8 at 150K; `dist_gate.DIST_PROMPTS`), cut once
  (`dist_gate cuts`), then the top-20 next-token logprobs at 8 fixed positions: the prompt's last 8 tokens are the fixed
  continuation (`/v1/completions`, token-id prefixes, max_tokens 1, logprobs 20, one cache salt per run). Only token ids,
  prefix hashes and logprob vectors are written, on Controller, never git.
- **Compare.** Per position: KL(control || candidate) over the shared top-20 + OTHER (`greedy_kl.top20_kl`; positions
  with common mass < 0.99 are counted) and top-1 agreement.
- **A/A band.** Register it BEFORE any candidate use (`dist_gate band`, sha256-sealed) from production-vs-production
  comparisons. The seeded A/A pilot provides one; register more as pilots accumulate.
- **Gate.** `dist_gate gate`, exit 2 on LOSS. Candidate quality LOSS when:
  - its mean KL exceeds `kl_ceiling` = 1.5 × the 99th percentile of the A/A mean KL (seeded bootstrap over the A/A
    positions); or
  - its top-1 disagreements exceed the A/A's by an exact conditional binomial test at one-sided p < 0.01.
  - it has significantly more positions with top-20 common mass < 0.99 than the A/A (same exact test; the coarse KL is
    not trustworthy there);
  - **Decision recorded (rule 27):** F-METHOD's literal top-1 rule ("agreement < A/A minimum − 1") is reported
    (`fmethod_top1_rule`) but does not decide. Against one A/A run it rejects a no-effect candidate 13–24 % of the time
    once the A/A shows ≥ 1 expected near-tie flip.
- **Greedy identity** stays the W8 restore gate, unchanged.
- **Simulated** (512 positions, band from one A/A run, `scratch/WIN2R3/sims/sim_dist.txt`):

  | expected A/A flips | no-effect false LOSS: this gate | no-effect false LOSS: F-METHOD literal | LOSS at 4× / 10× the flip rate |
  |---|---|---|---|
  | 0.25 | 0.1 % | 2.1 % | 0.2 % / 1.1 % |
  | 1 | 0.0 % | 13.7 % | 4.9 % / 54.8 % |
  | 3 | 0.0 % | 24.2 % | 38.8 % / — |

  The mean-KL part alone gives 0.0–0.2 % false LOSS. The top-1 test's power grows with the number of registered A/A
  comparisons (q = 1/(k+1)).
- **Cost** (ESTIMATED): one cold prefill per prompt per arm. That is ~15.5 min per arm for all 64 prompts, or ~6.5 min for
  the `short,32k` subset that the pilot registers. F-METHOD's "+2 min" covers the short prompts only. The band and the
  gate must cover the same subset.

## Seeded A/A pilot recipe (WIN2-R3; end-of-restore add-on, no extra HOLD; `aa_pilot.py`)

1. **Start A:** at the end of the window's last control start (production configuration, experiment port), after its
   own cells, run `aa_pilot run --execute --url http://127.0.0.1:18001 ...`: seeded short-b1, 32k-b1, 32k-b2 and 32k-b3,
   then the distributional probe on the `short,32k` subset.
2. **Start B:** the same on restored production, between W8 `check` (PASS) and W8 `close`, still inside the window's
   HOLD, so no legal traffic reaches it. W8's steps and its greedy-identity gate are unchanged; the pilot only reads.
3. **Compare:** `aa_pilot compare A B`.
   - **PASS** = every B1 seeded cell has identical output digests and per-request round counts in both starts. B ≥ 2
     cells are reported only.
   - The result also gives each cell and metric's measured paired-difference CV (`resolution.py --diff-cv`), a 1-df
     between-start SD estimate (accumulate over pilots before registering `start_sd`), and the distributional A/A
     comparison (`dist_gate band`).
   - **FAIL** means seeded pairing at B1 cannot assume identical streams: runners report measured diff CVs only, and no
     target-side lever uses `expect_identical`.
4. **Cost** (ESTIMATED): ~5 min of seeded cells plus ~6.5 min of probe per start, about 23 min in all.

## Control archive

`/opt/dsv41/campaigns/DSV41-OFFLOAD/control-archive.jsonl`: append-only and sha-chained, schema
`dsv41-control-archive-v3`. Use `archive_append` to write, and `archive_spread` / `archive_rows` / `archive_head` to read.
Every window appends every control repeat. **Every field below is required** (WIN2-R2, S92 FIX 2); a row without one
is refused, and a runner on older code refuses to append to the v3 file:

| field | value |
|---|---|
| `window`, `cell`, `metric`, `value` | as before |
| `max_tokens` | positive integer; part of the cell definition |
| `commit` | the 40-hex **production** commit. Take the summary's `production_commit`, then `production`, then `rollback.json` (`production_commit` or `commit`): `win2_history.production_commit(out)`. **Never** `experiment_commit` |
| `clock_cap_mhz` | the ONE cap of the row's **own start** (positive int): `cap, source = win2_history.start_clock_cap(out, start)`. Proof is that start's log `cmd/NNNN-<start>-start.log`: start_exp.sh prints `<host> clock cap N MHz` only after `nvidia-smi -lgc 0,N` succeeded; both Sparks must print it and agree, else the start is refused. A start with no start log is classified by its checkpoint times against `scratch/CLOCK.log` (2150, except 2300 from 13:27:18 to 13:36:27Z and from 19:23:25Z on); a start that straddles a change is refused. Write `clock_source` (`'start-log'` or `'clock-log'`) beside it |
| `clock_caps_mhz` | optional window-level **audit** field: `win2_history.clock_caps_mhz(out)` (list of MHz, or `'unknown'`) |
| `date` | `YYYYMMDD` of the window (`win2_history.window_date(out)`) |
| `start`, `repeat` | the control start's name and the repeat index inside it, so between-start variance can be watched |

**max_tokens is part of the cell.** NBW ran short-b1, 32k-b1, 32k-b2 and 32k-b3 at 1,500 tokens; every other runner
used 512. At B3 that alone moves ms/round from about 72 to about 81 ms (r = −0.93 over 23 repeats): shorter answers
spend more of their decode beside the other streams' 32K prefills. A runner whose cell differs in anything else
(prompts, batch, late arrival) gives it a new name.

**Spreads never pool definitions.** `archive_spread` refuses a cell/metric whose rows mix max_tokens, production commit
or clock cap unless `where=` picks one of each, e.g. `where={'max_tokens': 512, 'commit': <40-hex>,
'clock_cap_mhz': 2300}`; `archive_spreads(path, cell, metric)` reports one spread per (max_tokens, clock cap, commit).
The clock ceiling was 2150 MHz (dsv41.env default) for every start of every archived window through
PC1-threshold512-W-20261027T181314Z (each start's log proves it) and is 2300 MHz from 2026-10-27 19:23:25Z
(`GPU_CLOCK_CAP=2300` in /etc/environment; PP4-941-admission at 19:23:39Z is the first 2300 window, and its start log
says so). Busy clocks rose about 7 % (2125 → 2275 MHz), so a pooled spread would mostly measure the clock step. The PP4
promotion will change the commit.

Report `archive_spread` next to the in-window spread. Keep the 2 % required effect unless the archive shows larger
real variation; if it does, say so in the report and never raise the bar silently. Record `archive_head(path)` (row
count + head sha) in the window summary before and after appending: the chain detects an edited line, not a truncated
tail. To backfill history, run `python -m offload.launch.windows.win2_history backfill --archive <archive> <window
dirs>`. It is idempotent per (window, cell, metric) and reads a window named twice in one call once.

WIN2-R2 rebuilt the archive (596 rows: the 540 v1 rows with their production commit, per-start clock cap, start and
repeat filled in, plus PC1-threshold512's 56). The v1 file is kept at
`scratch/WIN2R2/control-archive.v1-before-win2r2.jsonl`; the proof (multiset equality) is in `WIN2-R2.report.md`.

## New shared cells (`gb10_budget/window_cells.py`, opt-in by name, not in ORDER)

- `ctl-mixed150k-4user`: three 32K decoders; the 150K document arrives only after all three are decoding and must be
  admitted beside them (running gauge ≥ 4 while every decoder is alive; else the cell refuses). ~300 s.
- `ctl-long-5k`: one B1 request with a 5,000-token answer. ~180 s.

## Back-to-back mode (`chain.py`)

- Coordinator sets `WINDOW_CHAIN=1` for every window of a chain and `WINDOW_CHAIN_LAST=1` on the last one.
- **Hold placeholders stay up for the whole chain** (coordinator rule, from R-PC1 / R-PP4; WIN2-R3 root fix of the
  defect R-NBW proved in `scratch/RNBW/placeholder-proof.txt`). Between chained windows there is no HOLD file, so the
  placeholders are what keep NVO drama3 and the local-law queue off the Sparks while production is down.
  - **Two kinds.** A window's own placeholder (made by `hold_admission.open_script` inside its `-hold` scope) lives on
    that scope's PRIVATE tmux socket (the scoped_remote tmux shim) and dies when the scope is fenced: it covers one
    window only. The **chain placeholder** is a tmux `dsv41-hold` session on the Spark's DEFAULT socket running
    `vllm serve [DSV41-OFFLOAD hold] owner=<chain owner>`; it outlives every window scope.
  - **Owner.** `owner = chain.chain_owner()` (the marker's while the window ran chained, else a new
    `chain-<UTC>-<8 hex>`); call it once per window and keep it. The owner is in the process title on each Spark and in
    the marker (`placeholder_owner`). Start, probe and stop refuse a placeholder of any other owner, including
    window.sh open's (no owner), and never touch it.
  - **Where they are looked for.** Every step uses `/usr/bin/tmux -L default` (`chain.PLACEHOLDER_TMUX`), never the
    PATH `tmux`, so it reaches the default socket from anywhere, including inside a scope.
  - **Start** `chain.placeholders_up(owner)`: from the runner's own unscoped process over plain ssh (refused inside any
    `dsv41w-*.scope`, where a new default server would die with the fence), at a `'chained'` end, BEFORE the first fence
    of the window's scopes. Raises unless both Sparks are UP for this owner. Idempotent: a chained window keeps the same
    processes.
  - **Close** `chain.chained_close(owner)` replaces R2's `CHAINED_CLOSE` (which looked at the scope's private socket and
    always failed): run inside the window's `-restore-close` scope; it proves this owner's placeholders on both Sparks
    through the scope's ssh shim, then removes only the HOLD (the rm shim records it and refuses another window's HOLD;
    commit-close removes it).
  - **Probe** `chain.placeholders_present()` (owner from the marker): the chained preflight and `chain.gate()` (so
    `chain_prev_ok.sh`) require both UP. A marker without an owner (written before this fix) stops the chain.
  - **Stop** `chain.end_chain(owner=None)`: the chain's last step, ONLY after the strict W8 restore verified AND the
    post-close watch PASSED. It takes this owner's placeholders down (from an unscoped process; refused inside a scope),
    then clears the marker; a foreign or unreachable placeholder is left alone and reported (`ENDED WITH ALERT`).
    `chain.clear()` refuses while the marker records an owner. `window.sh close` (W8 close under a scope) cannot
    remove them: its kill reaches the scope's private socket.
  - systemd ≥ 258 expands `${...}` in a `systemd-run --scope` command line (Controller runs 259), so the placeholder
    scripts and `chained_close` use no braced expansions.
- **Runner start** (S92 FIX 1). `mode = chain.preflight(production_serving, hold_exists,
  placeholders_present=chain.placeholders_present())` (or `chain.gate()`, which probes this chain's placeholders only
  while the marker is present) replaces the "production serving K1" preflight. A chained preflight without the placeholder
  result refuses. In mode `'chained'` only:
  - open the admission HOLD with `hold_admission.open_script(..., allow_unreachable=chain.allow_unreachable(mode))`.
    The default `require-serving` admission raises when :8000 does not answer, so a chained window would fail before
    its HOLD;
  - prove both Sparks quiet with `spark_quiet.command(<marker>, deadline=...)` on sparkhead and sparkworker-wired **before**
    `start_exp.sh` (the gate infers "production down" only from a failed `/v1/models` probe; it never proves quiet);
  - do not capture production (it is down): `path, rollback = chain.production_snapshot()` returns the chain's
    capture (the rollback.json of the window that started from serving production; its sha256 is checked against the
    marker), whatever runner wrote it. Copy it into the window's own `rollback.json` with `chained_from` /
    `chained_from_sha256` and take the window's own first control start as the measurement baseline.
- **Runner end.** Decide with `chain.end_action(stopped=<any stop>, hold_written=<own HOLD written>)`, never with
  `hold_written` alone:
  - `'chained'`: `owner = chain.chain_owner()`; `chain.placeholders_up(owner)` before the first fence; stop the
    experiment, prove the Sparks quiet, close the own HOLD with `chain.chained_close(owner)` inside the window's
    `-restore-close` scope (`owner.command('launch', f'{tag}-restore-close', ['bash', '-c', chain.chained_close(owner)],
    ...)`, require `chain.CHAINED_CLOSE_OK` in its output, then the usual fenced commit-close; `window.sh close` would
    refuse without :8000), `chain.finish_chained(out, quiet_proof, placeholder_owner=owner)` (the marker records the
    owner and the chain's production snapshot: inherited from the previous marker, or this window's own rollback.json),
    write
    `"chain": "CHAINED"` and `"restore": {"status": "CHAINED: production left down"}` into the summary, and **skip** the
    W8 restore and the post-close watch;
  - `'restore'`: the normal strict W8 restore + post-close watch inside the own HOLD, then `chain.end_chain()` (only
    after RESTORED + watch PASS; `chain.end_chain(owner)` when this window started chain placeholders and failed before
    `finish_chained`). Until then the placeholders stay up;
  - `'abort'` (no own HOLD, chain marker present: e.g. a pre-HOLD failure of a chained window; production is down from
    the previous window): `summary['restore'] = chain.abort(out / 'chain-abort')`. With no HOLD present it runs the strict
    W8 start|check|close under the reviewed `post_close_watch.ScopedRecovery` (its own HOLD, Sparks proved quiet, six
    distinct IDENTICAL greedy) plus the 15-minute post-close watch, and runs `end_chain()` (placeholders down, marker
    cleared) only on RESTORED + watch PASS.
    With a HOLD present it alerts and touches nothing. Never write "NOT NEEDED: production untouched" in this case;
  - `'none'`: no HOLD and no marker; production was not touched.
  `chain.restore_required(stopped, hold_written)` is `end_action(...) in ('restore', 'abort')`.
- **Coordinator gate**: `offload/launch/windows/chain_prev_ok.sh` replaces campaign `launch/prev-ok.sh` for chained windows.
  Exit 0 admits; exit 2 is a refusal with no chain marker (as prev-ok.sh); **exit 3 is `CHAIN STOPPED: PRODUCTION DOWN`**:
  any refusal (previous summary, gate, HOLD, stale container) while the marker is present. The gate then runs
  `chain_abort.sh` itself before exiting, so a stopped chain never leaves production down unattended. RUN scripts call
  the gate under `set -e`, so exit 3 also ends that RUN script; the abort's `summary.json` (in
  `$C/CHAIN-ABORT-<UTC>/`) is a valid PREV_FILE target for the next gate.
- **Coordinator abort by hand**: `offload/launch/windows/chain_abort.sh [OUT]` (= `python -m offload.launch.windows.chain
  abort --out OUT`, needs `WINDOW_COORDINATOR`). Exit 0 restored + watch PASS (placeholders down, marker cleared) or no
  marker; 3 failed (marker and placeholders kept, production down); 4 a HOLD exists (alert only). By hand:
  `python -m offload.launch.windows.chain placeholders [--owner X]` (probe both Sparks) and, after a verified restore +
  watch PASS, `WINDOW_COORDINATOR=... python -m offload.launch.windows.chain end [--owner X]`.
- The legal pipeline is down for the whole chain (NDM accepts this while building). The last window, or any stop,
  brings production back with the strict six-IDENTICAL W8.
- Window re-scoring (`python -m offload.launch.windows.win2_history reanalyze --decision C1 [--decision C2] <window>`)
  uses decide_window's cell rule, refuses more than two decision cells, treats roles `control`, `K`, `off` (TV1) and
  `H` (PF2) as controls, and judges every other role as its own arm (a two-arm window reports `arms`, never a pooled
  candidate).
- Test-only overrides (never set in production): `DSV41_CHAIN_MARKER`, `DSV41_CHAIN_PRODUCTION_URL` (chain.py),
  `DSV41_CHAIN_CAMPAIGN`, `DSV41_CHAIN_PY`, `DSV41_CHAIN_ABORT` (the two scripts). `chain status` prints the marker path
  and production URL in use.

## Greedy refinement (`greedy_kl.py`, opt-in)

Candidate arm: `nbw_cells.greedy(..., near_tie_nats=<registered>, record_distribution=True,
probe_salt_prefix=<window tag>)` keeps the top-20 and prefix at every divergence. On the command line:
`python -m offload.launch.windows.nbw_cells --execute --mode greedy ... --near-tie-nats <registered>
[--record-distribution --probe-salt-prefix <window tag>]` (defaults unchanged: 0.501 nat, no recording). Following control arm:
`greedy_kl.compare_recorded(api, candidate_rows, salt_prefix, bound=<registered>)` re-probes those prefixes with a fresh
cache salt and measures KL(control || candidate) over shared top-20 + OTHER (UP2's estimator); common mass must be
≥ 0.99. The bound is each runner's registration (UP2 used max(0.005, 2 × A/A floor)).

## Adoption checklist for each runner fork

1. `git merge offload/win2` (needs the "WIN2 API" commit ce1bcf2 or later).
2. Schedule ≥ 3 repeats per arm in the decision cells (C/X/C with ≥ 2 repeats per start gives ≥ 4 control and needs
   ≥ 3 candidate repeats: run the candidate start with 3 repeats or use C/X/C/X/C).
3. Build `decision_v2.Series` per decision cell, primary metric first; call `decide_window(series, decision_cells,
   groups=...)`; write its dict as the summary `decision` (rule `win2-v1`).
4. `archive_append` every control repeat at the end of the window, including after a later stop for the starts that
   finished. Each row carries every field of the archive table above (commit = production, `clock_cap_mhz` of its own
   start from `start_clock_cap`, `YYYYMMDD` date, start, repeat); record `archive_head` before and after.
5. Chain mode as above; summary `chain` field. Tests (fake transport, `tests/test_chain_stop.py` shows the shared
   parts): a chained non-last end (marker, no restore); a last or stopped window (strict restore, `chain.end_chain()`);
   **a chained start with production down** (admission with `allow_unreachable`, quiet proof before `start_exp.sh`);
   **a pre-HOLD failure in the chained state, which must restore** (`end_action == 'abort'` -> `chain.abort`); and
   **a gate refusal in the chained state**, which exits 3 and takes the abort path; **placeholders** (WIN2-R3): a
   chained end calls `placeholders_up(owner)` before its first fence and `chained_close(owner)` in its
   `-restore-close` scope, `finish_chained(..., placeholder_owner=owner)`; a chained start refuses without this
   chain's placeholders; a final restore ends with `end_chain()`; a chained start that follows another runner's window
   takes its production capture from `chain.production_snapshot()`. The real-scope proof is
   `tests/test_chain_placeholder_rehearsal.py` (real user scopes, scoped_remote shims, tmux, fake per-host HOME).
6. **WIN2-R3 (seeded cells; adopt after its independent check).** Swap the decode decision cells for seeded cells
   (`run_seeded_cell` on every arm, C/X/C; `seeded_series` + `decide_paired_window`), keep the existing gates, tag every
   cell's profile and state in the promotion which profile it relies on; call `window_cap` from the launcher and
   `resolution` on the pre-registration (`expected_gain` per decision cell); add the smoke start (`x_first` in a single
   window, `chain_head` at a chain's head); register `start_sd` / `--diff-cv` once the seeded A/A pilot has measured them.
7. Greedy: keep the runner's registered near-tie limit (0.9 general, 1.1 PP4 and DEP1 fixed2 per NDM) and, where the
   window has a following control start, the opt-in KL check.

## Rehearsal before the first Spark window (design note; not built)

The real target runs only on the two Sparks (RUNBOOK-offload.md lists no other configuration of it), so no rehearsal
off the Sparks can measure speed, acceptance or numerics. A rehearsal exists to catch runner faults (sequencing,
parsing, timeouts, refusals, chain and archive writes) before they cost a window. Three levels:

1. **Required for every new or changed runner: fake transport on Controller CPU.** Real runner code over inert recording
   remote commands and an in-process HTTP fake (`tests/test_nbw_rehearsal.py`, `tests/test_pf2_rehearsal.py`,
   `UP2-REHEARSAL.md`). Add the WIN2 cases of checklist step 5: a chained non-last end (marker, no restore), a stop
   inside a chained window (restore, `chain.end_chain()`), a chained start with production down, a pre-HOLD failure in the
   chained state (`chain.abort`), and a gate refusal in the chained state (exit 3).
2. **Proposed: a stub "target-only" HTTP server (CPU only; Controller or x86host CPU, loopback port, no GPU).** One small
   threaded process that answers the endpoints the runners touch with vLLM's formats: `/v1/models` (one model,
   `max_model_len`), `/tokenize` (`count` from the text length), `/v1/chat/completions` (SSE with reasoning/content
   deltas, final usage, `[DONE]`), `/v1/completions` (top-20 logprobs, token ids), `/metrics` (the
   `vllm:num_requests_running` and `vllm:num_requests_waiting` gauges and the speculative counters bench2
   `parse_metrics` reads), and `/start_profile` + `/stop_profile`. It needs a configurable admission limit and decode
   delay, so that `ctl-mixed150k-4user`'s late-arrival check passes at limit 4 and refuses at limit 3. The runner points
   at it via its target URL with the remote side still faked. This proves real sockets, timeouts, SSE parsing,
   concurrency and admission. It proves nothing about performance. Not built here: matching vLLM's SSE and metrics
   formats exactly is a lane of its own, and level 1 already gates every runner.
3. **Drafter/worker code changes only: an owned x86host window** (as the PP4 server probe did). The coordinator books it,
   because every x86host 3090 is assigned to DSV41 work. It checks that the worker loads and answers; Spark rounds are
   still needed for any number. Config-only Spark arms skip this level.

Then, in the first chained window, the coordinator runs the runner's `--plan` / dry-run against the live read-only probes
before the HOLD opens.
