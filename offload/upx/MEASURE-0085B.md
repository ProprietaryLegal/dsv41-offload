# 0085b measurement design: idle memory and the next 150K prefill's low point

Arithmetic: `measure_0085b.py` (tests: `tests/test_measure_0085b.py`). Window data: `windows.py` group `UPX-0085B`.
Runner: `battery_window.py` (`python3 -m offload.upx.battery_window` prints the plan; `--execute` is coordinator only),
requests: `battery_cells.py`; CPU rehearsal: `tests/test_battery_window.py`.
Nothing here has run on a Spark. Figures marked UPSTREAM are HeNryous's (same hardware class, local drafter, 300K
prompts); ESTIMATED figures are arithmetic; nothing below is MEASURED on our stack yet.

## The questions

1. **Idle memory.** After a long prefill, does each rank give back its allocator cache within the idle period?
   UPSTREAM: free memory at idle 5.7 vs 3.9 GiB (head) and 6.5 vs 4.7 GiB (worker), one release of 1,748 MiB per rank.
2. **The next long prefill's low point.** Does the lowest MemAvailable during the *next* 150K prefill rise?
   UPSTREAM: head node 3,804 vs 3,269 MiB (+535 MiB) during the next 300K prefill; first token unchanged (130-132 s).
3. **The 16K-chunk floor question.** PC1 promotes chunk16384 with NDM's 3 GiB floor. PC1's admission estimate is
   3.6 GiB on sparkhead at 150K (`offload/pc1/headroom.py` @ offload/pc1-win2: MEASURED 150K minimum 6,749,280 kB minus
   885 MiB per extra 4,096 rows minus 256 MiB contingency; ESTIMATED).

The point behind question 3: PC1's minima come from fresh starts and first prefills. In production the allocator keeps
the high-water mark of earlier prefills, and the next prefill can grow new segments beside the cached ones instead of
reusing them; the UPSTREAM gap above (535 MiB) is that effect. So the steady-state low point without 0085b can sit
about 0.5 GiB below what a fresh-start control shows (UPSTREAM figure, our size unknown). With 0085b, an idle period of
at least 5 s returns the engine to the fresh-start state before the next prefill. The battery below measures exactly
that steady state (B follows A), with and without the release.

## The battery (one unit of measurement)

| Phase | What | Recorded |
|---|---|---|
| set | write the threshold file on BOTH ranks (`configs.release_file_command`: 131072 = ON, 0 = OFF), atomically | the command, both exit codes |
| A | one cold 150K B1 request (public prompt `150k`, fresh cache salt, `max_tokens` 16) | send / done times |
| idle | `IDLE_WAIT_S` = 20 s with no request: 5 s grace, then (ON) the release, then settle | — |
| B | the next cold 150K B1 request (another fresh salt, same prompt) | send, first token, done |
| after | `POST_B_IDLE_S` = 15 s with no request (ON releases again, so the next battery starts from the same state) | — |

Per rank the arithmetic takes, from the runner's 1 s telemetry (`pf1_window.TELEMETRY` rows) and `docker logs
--timestamps` of each rank's container:

- `idle_kb`: median MemAvailable over [A done + 10 s, A done + 20 s) (at least 8 samples);
- `next_low_kb`: minimum MemAvailable over [B sent, B first token] (the prefill);
- `next_ttft_s` (head only): B's first token, the guard for the re-mapping cost after a release;
- the release census: ON = exactly one 0085b line per rank in A's idle (at least 5 s after A, after >= 131,072 prefill
  tokens) and exactly one after B, none while a request runs; OFF and image = none. Any `release failed` line stops.

A rank below 3 GiB during B stops the window (NDM floor), in every mode.

## M1 — window `UPX-0085B` (production chunking, production-exact base; ESTIMATED HOLD 66 min chained, 89 min as the last window)

| Start | Arm (`UPX.env`) | Content |
|---|---|---|
| C1 | `C-upx` (production-exact `prod.env`, no overlay: image bytes) | warm-up · six greedy (0.9) · 2 image batteries |
| X | `release-file` (`UPX_BASE=prod UPX_0085B=file`) | warm-up · six greedy (0.9, record distribution) · 3 ON/OFF pairs in strict alternation (`measure_0085b.schedule(3)`) · release census over the whole start |
| C2 | `C-upx` | warm-up · six greedy + greedy_kl re-probe of X's divergences · 2 image batteries |

- **Decision (rule win2-v1, `decision_v2` unchanged):** cells `mem-r0` (sparkhead) and `mem-r1` (sparkworker-wired); primary
  `next_low_kb` (higher is better), guards `idle_kb` and (head) `next_ttft_s`. ON_i and OFF_i are adjacent batteries in
  one engine, so the series are **paired** (WIN2-R2: true adjacent pairs only; C/X/C stays unpaired). CANDIDATE = a WIN
  on at least one rank, no LOSS anywhere. Expected from UPSTREAM: idle +1.8 GiB (~27 %), next low +0.5 GiB (~8 %),
  both far above the 2 % bar; TTFT unchanged.
- **Gates:** six greedy at 0.9 nats (0085b cannot change numerics; this proves the overlay changed nothing else), the
  release census, no failure line, the floor.
- **Reported:** mounted-but-off (OFF) against the image batteries of C1/C2 (`measure_0085b.summary`
  `off_minus_image_kb`): the switch-off identity in memory terms, next to the code-level proof (tests: reverse port
  patch = image bytes; switch-off call-for-call equal to the image).
- **Archive:** the C1/C2 image batteries are control rows of cell `battery-150k` (`max_tokens` 16), metrics
  `idle_kb`, `next_low_kb`, `next_ttft_s`, with WIN2-R2's schema: production commit, the start's own `clock_cap_mhz`
  and `clock_source` (`win2_history.start_clock_cap` on its start log, both Sparks agreeing), `YYYYMMDD` date, start,
  repeat. An unproven clock blocks only the archive (`archive_error`), never the window. Pooled spreads need
  `where={'max_tokens': 16, 'clock_cap_mhz': cap, 'commit': production}`.
- **Decision rule revision 2 (WIN2-R2):** the TTFT guard can only make a cell LOSS when its loss-repeat test fires AND
  its whole 95 % interval is below zero (`tests/test_windows.py` pins both sides on the head cell).

## M2 — the 16K-chunk floor (composed with PC1's chunk16384; needs PC1's runner)

One extra start at the end of PC1's chunk16384 window (chained, after its own C/X/C), ESTIMATED 29 min:
`PC1_ARM=chunk16384` plus `UPX_0085B=file` (PC1's `window.env` ends with `source offload/upx/x0085b.env`; chunk16384
changes only `--max-num-batched-tokens`, so no mount collides). Warm-up, six greedy, 3 ON/OFF pairs as in M1.
Output: `summary()['hosts'][h]['floor_margin_kb']` with and without the release, next to PC1's fresh-start minimum
from the same window. This answers "does chunk16384 keep 3 GiB in steady state, and by how much does 0085b widen the
margin". It does not need a WIN: it is a floor measurement.

## Clocks and timestamps

Telemetry `utc` and docker log stamps come from the Sparks, request times from the runner on Controller; all are NTP wall
clocks. Every window the arithmetic uses is at least 5 s from the event it brackets (grace 5 s, idle sample window
starting 10 s after A), so sub-second skew cannot move a sample across a boundary.

## Promotion, if M1 is a CANDIDATE

Production would mount the two overlay files and set `VLLM_DSV41_RELEASE_FILE` with the file at 131072 on both ranks:
the file is a live kill switch (write 0, no restart). Cost in file mode with the file at 0: the engine core waits the
5 s grace and sends one no-op RPC per idle period. Composition: `x0085b.env` appends to any base and refuses a base that
already mounts `engine/core.py` or `worker/gpu_worker.py`; no current package does (tests).
