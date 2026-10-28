# WIN2 — fairer, deeper measurement and back-to-back windows

NDM 2026-10-27: "Fairer win rule is essential", "Do better deeper testing and measuring", "go window after window".
Method source: Fable review (roadmap section "Independent review 2026-10-27 (Fable)", e8e3e3a).

## Rule `win2-v1` (`decision_v2.py`)

| Question | Answer |
|---|---|
| What decides | At most **two decision cells** per window; every other cell is reported only. |
| Primary metric | First series per cell. Prefer **pooled server ms per round** (`server_decode_ms / spec_request_rounds`) for decode cells and **TTFT** for prefill cells. Accepted/round and client tok/s are guards (they can only cause a LOSS). |
| Repeats | ≥ 3 per arm in every decision cell (refused below). |
| WIN | point gain > 2 % **and** the 95 % interval excludes 0 (paired Student t; Welch t for unpaired arms; floored df). |
| LOSS | worse than the cell's own control spread in ≥ ceil(2n/3) repeats (2 of 3). |
| Window | any decision-cell LOSS → NOT A CANDIDATE; ≥ 1 WIN and no LOSS → CANDIDATE; else INCONCLUSIVE. Optional `groups` give separate single-user / multi-user verdicts. |

Why ms/round: across the control archive (540 control rows from 10 windows), ms/round moves **0.7–5.0 %** from day to
day within one cell definition. Client tok/s moves up to 11 % and accepted/round up to 23 %, because acceptance per
round depends on the sampled text (T = 1). Threshold2048's short cell is the example: ms/round was 3.4 % better while
tok/s was 13.5 % worse, because that run's acceptance happened to be 28 % lower.

## Control archive

`/opt/dsv41/campaigns/DSV41-OFFLOAD/control-archive.jsonl`: append-only and sha-chained. Use `archive_append`
to write, and `archive_spread` / `archive_rows` to read. Every window appends every control repeat with window, cell,
metric, value and **max_tokens** (all required), plus commit, clock caps and date.

**max_tokens is part of the cell.** NBW ran short-b1, 32k-b1, 32k-b2 and 32k-b3 at 1,500 tokens; every other runner
used 512. At B3 that alone moves ms/round from about 72 to about 81 ms (r = −0.93 over 23 repeats): shorter answers
spend more of their decode beside the other streams' 32K prefills. `archive_spread` refuses to pool a cell name across
different max_tokens; pass `where={'max_tokens': N}`. A runner whose cell differs in anything else (prompts, batch,
late arrival) gives it a new name.

Report `archive_spread` next to the in-window spread. Keep the 2 % required effect unless the archive shows larger
real variation; if it does, say so in the report and never raise the bar silently. To backfill history, run
`python -m offload.launch.windows.win2_history backfill --archive <archive> <window dirs>`. It is idempotent per
(window, cell, metric).

## New shared cells (`gb10_budget/window_cells.py`, opt-in by name, not in ORDER)

- `ctl-mixed150k-4user`: three 32K decoders; the 150K document arrives only after all three are decoding and must be
  admitted beside them (running gauge ≥ 4 while every decoder is alive; else the cell refuses). ~300 s.
- `ctl-long-5k`: one B1 request with a 5,000-token answer. ~180 s.

## Back-to-back mode (`chain.py`)

- Coordinator sets `WINDOW_CHAIN=1` for every window of a chain and `WINDOW_CHAIN_LAST=1` on the last one.
- Runner end: `if chain.skip_restore(stopped=<any stop>)`: stop the experiment, prove the Sparks quiet (existing
  spark_quiet proof), `chain.finish_chained(out, quiet_proof)`, write `"chain": "CHAINED"` and
  `"restore": {"status": "CHAINED: production left down"}` into the summary, close the HOLD, and **skip** the W8 restore
  and the post-close watch. Otherwise run the normal strict W8 restore + watch, then `chain.clear()`.
- Runner start: replace the "production serving K1" preflight with `chain.preflight(production_serving, hold_exists)`
  (or `chain.gate()`); in mode `'chained'` skip the production identity snapshot that needs a running server and take
  the window's own first control start as the baseline.
- Coordinator gate: `offload/launch/windows/chain_prev_ok.sh` replaces campaign `launch/prev-ok.sh` for chained windows.
- The legal pipeline is down for the whole chain (NDM accepts this while building). The last window, or any stop,
  brings production back with the strict six-IDENTICAL W8.

## Greedy refinement (`greedy_kl.py`, opt-in)

Candidate arm: `nbw_cells.greedy(..., near_tie_nats=<registered>, record_distribution=True,
probe_salt_prefix=<window tag>)` keeps the top-20 and prefix at every divergence. Following control arm:
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
   finished. Each row carries `max_tokens` (required).
5. Chain mode as above; summary `chain` field; tests for both chained and last/failed paths.
6. Greedy: keep the runner's registered near-tie limit (0.9 general, 1.1 PP4 and DEP1 fixed2 per NDM) and, where the
   window has a following control start, the opt-in KL check.

## Rehearsal before the first Spark window (design note; not built)

The real target runs only on the two Sparks (RUNBOOK-offload.md lists no other configuration of it), so no rehearsal
off the Sparks can measure speed, acceptance or numerics. A rehearsal exists to catch runner faults (sequencing,
parsing, timeouts, refusals, chain and archive writes) before they cost a window. Three levels:

1. **Required for every new or changed runner: fake transport on Controller CPU.** Real runner code over inert recording
   remote commands and an in-process HTTP fake (`tests/test_nbw_rehearsal.py`, `tests/test_pf2_rehearsal.py`,
   `UP2-REHEARSAL.md`). Add two WIN2 cases: a chained non-last window ends with the marker and no restore, and a stop
   inside a chained window runs the restore and `chain.clear()`.
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
