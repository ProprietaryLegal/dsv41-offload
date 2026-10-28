# WIN2 — fairer, deeper measurement and back-to-back windows

NDM 2026-10-27: "Fairer win rule is essential", "Do better deeper testing and measuring", "go window after window".
Method source: Fable review (roadmap section "Independent review 2026-10-27 (Fable)", e8e3e3a).

## Rule `win2-v1`, revision 2 (`decision_v2.py`)

Revision 2 (WIN2-R2, S92 FIX 3) keeps the name `win2-v1` so runner code keeps working; every decision records
`rule_revision: 2`.

| Question | Answer |
|---|---|
| What decides | At most **two decision cells** per window; every other cell is reported only. |
| Primary metric | First series per cell. Prefer **pooled server ms per round** (`server_decode_ms / spec_request_rounds`) for decode cells and **TTFT** for prefill cells. Accepted/round and client tok/s are guards (they can only cause a LOSS). |
| Repeats | ≥ 3 per arm in every decision cell (refused below). |
| WIN | point gain > 2 % **and** the 95 % interval excludes 0 (paired Student t; Welch t for unpaired arms; floored df). |
| Loss-repeat test | the candidate is worse than the loss threshold in ≥ ceil(2n/3) repeats (2 of 3). Control spread = **1.69 × the unbiased control SD (s / c4(n)) / mean**, the expected range of three values whatever the number of control repeats (rev 1 used the range of all controls, which widened with more repeats: a true 5 % loss was caught 74 % of the time at 3 v 3 but 28 % at 8 v 4). Threshold = that spread unpaired, **√2 × it paired** (a paired difference has √2 × the SD of one value). |
| Primary LOSS | the loss-repeat test fires. |
| Guard LOSS | the guard's loss-repeat test fires **and** its whole 95 % interval is below zero. |
| Window | any decision-cell LOSS → NOT A CANDIDATE; ≥ 1 WIN and no LOSS → CANDIDATE; else INCONCLUSIVE. Optional `groups` give separate single-user / multi-user verdicts. |

**When to pair.** `Series.paired` defaults to True, but pair only true adjacent pairs: K/I/K/I alternation, one control
and one candidate repeat per pair in run order (or a bracket where control repeat r is the mean of the two bracketing
starts' repeat r). C/X/C with all control repeats pooled, and any layout with unequal repeats, is unpaired
(`paired=False`, Welch). Unpaired, schedule ≥ 4 control repeats (C/X/C with 2 per C): at 3 v 3 the no-effect false
loss per cell is 8 %.

**No-effect rates** (the candidate has no real effect; two decision cells, primary ms/round + both guards; S92's
`sim_corr.py` model, seed 5, 10,000 windows: ms/round cv 1.5 %, acceptance cv 12 %, tok/s derived from both; rerun in
`scratch/WIN2R2/sims/`). "Rejected" = NOT A CANDIDATE, "accepted" = CANDIDATE:

| design | primary only: rejected / accepted | with both guards: rejected / accepted | rev 1 with both guards |
|---|---|---|---|
| paired 3 v 3 | 9.2 % / 1.9 % | **11.4 % / 1.8 %** | 32.5 % / 1.4 % |
| unpaired 4 v 3 (C/X/C) | 11.4 % / 2.0 % | **14.6 % / 1.9 %** | 16.0 % / 1.8 % |

Per cell, no effect (cv 2 %, 20,000): false LOSS 4.5 % paired 3 v 3, 5.7 % unpaired 4 v 3, 3.7 % unpaired 6 v 3. Power
for a true 5 % loss no longer falls with control repeats: unpaired 3, 4, 6 controls v 3 → 75 %, 77 %, 80 %; 8 v 4 → 73 %
(rev 1: 75 %, 65 %, 50 %, 28 %). With more *candidate* repeats the 2-of-3 count needs more of them beyond the threshold
(paired 5 v 5 → 33 %). The rule is still easier to lose than to win: about 5 % of a no-effect cell's chances go to LOSS
against under 2 % to WIN, because the primary loss test has no interval condition.

Why ms/round: across the control archive (WIN2's 540 control rows from 10 windows), ms/round moves **0.7–5.0 %** from day to
day within one cell definition. Client tok/s moves up to 11 % and accepted/round up to 23 %, because acceptance per
round depends on the sampled text (T = 1). Threshold2048's short cell is the example: ms/round was 3.4 % better while
tok/s was 13.5 % worse, because that run's acceptance happened to be 28 % lower.

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
- **Hold placeholders stay up for the whole chain** (coordinator rule, from R-PC1 / R-PP4). Between chained windows
  there is no HOLD file, so the placeholders (tmux `dsv41-hold` running `vllm serve [DSV41-OFFLOAD hold]` on both
  Sparks) are what keep NVO drama3 and the local-law queue off the Sparks while production is down. A chained end
  removes **only** the HOLD file (`chain.CHAINED_CLOSE`), the next chained preflight requires both placeholders
  (`chain.placeholders_present()`), and only the final strict W8 restore (`window.sh close`) or `chain.abort` (whose
  W8 close runs `window.sh close`) takes them down. A chained window's own HOLD opening reuses them
  (`open_script` creates a placeholder only when it is missing).
- **Runner start** (S92 FIX 1). `mode = chain.preflight(production_serving, hold_exists,
  placeholders_present=chain.placeholders_present())` (or `chain.gate()`, which probes the placeholders only while the
  marker is present) replaces the "production serving K1" preflight. A chained preflight without the placeholder
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
  - `'chained'`: stop the experiment, prove the Sparks quiet, close the own HOLD with `chain.CHAINED_CLOSE` inside the
    window's `-restore-close` scope (`owner.command('launch', tag, ['bash', '-c', chain.CHAINED_CLOSE], ...)`, require
    `chain.CHAINED_CLOSE_OK` in its output, then the usual fenced commit-close; `window.sh close` would refuse without
    :8000 and would kill the placeholders), `chain.finish_chained(out, quiet_proof)` (the marker records the chain's
    production snapshot: inherited from the previous marker, or this window's own rollback.json), write
    `"chain": "CHAINED"` and `"restore": {"status": "CHAINED: production left down"}` into the summary, and **skip** the
    W8 restore and the post-close watch;
  - `'restore'`: the normal strict W8 restore + post-close watch inside the own HOLD, then `chain.clear()` (only after
    RESTORED + watch PASS);
  - `'abort'` (no own HOLD, chain marker present: e.g. a pre-HOLD failure of a chained window; production is down from
    the previous window): `summary['restore'] = chain.abort(out / 'chain-abort')`. With no HOLD present it runs the strict
    W8 start|check|close under the reviewed `post_close_watch.ScopedRecovery` (its own HOLD, Sparks proved quiet, six
    distinct IDENTICAL greedy) plus the 15-minute post-close watch, and clears the marker only on RESTORED + watch PASS.
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
  abort --out OUT`, needs `WINDOW_COORDINATOR`). Exit 0 restored + watch PASS (marker cleared) or no marker; 3 failed
  (marker kept, production down); 4 a HOLD exists (alert only).
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
   parts): a chained non-last end (marker, no restore); a last or stopped window (strict restore, `chain.clear()`);
   **a chained start with production down** (admission with `allow_unreachable`, quiet proof before `start_exp.sh`);
   **a pre-HOLD failure in the chained state, which must restore** (`end_action == 'abort'` -> `chain.abort`); and
   **a gate refusal in the chained state**, which exits 3 and takes the abort path; **placeholders**: a chained end
   leaves them up (CHAINED_CLOSE), a chained start refuses without them; a chained start that follows another runner's
   window takes its production capture from `chain.production_snapshot()`.
6. Greedy: keep the runner's registered near-tie limit (0.9 general, 1.1 PP4 and DEP1 fixed2 per NDM) and, where the
   window has a following control start, the opt-in KL check.

## Rehearsal before the first Spark window (design note; not built)

The real target runs only on the two Sparks (RUNBOOK-offload.md lists no other configuration of it), so no rehearsal
off the Sparks can measure speed, acceptance or numerics. A rehearsal exists to catch runner faults (sequencing,
parsing, timeouts, refusals, chain and archive writes) before they cost a window. Three levels:

1. **Required for every new or changed runner: fake transport on Controller CPU.** Real runner code over inert recording
   remote commands and an in-process HTTP fake (`tests/test_nbw_rehearsal.py`, `tests/test_pf2_rehearsal.py`,
   `UP2-REHEARSAL.md`). Add the WIN2 cases of checklist step 5: a chained non-last end (marker, no restore), a stop
   inside a chained window (restore, `chain.clear()`), a chained start with production down, a pre-HOLD failure in the
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
