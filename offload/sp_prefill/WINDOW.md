# R-SP window design (WIN2 rule win2-v1, chain mode)

Coordinator-only. Registered in `window.py` (`python -m offload.sp_prefill.window --plan` prints it); arms in
`offload/launch/configs/RSP-window.env`. What is built: plan, budget, decision adapter, archive rows, chain close and
preflight, arm config, microbenchmark — all CPU-tested. What is not built: the executing runner (HOLD, starts, cell
driver, telemetry, greedy call, W8). Its cells, metrics and budgets are PC1's chunk-window cells (`offload/pc1/cells.py`
CHUNK_PLAN on `offload/pc1-win2`), so the cheapest executor is PC1's runner with this window's arm config and plan, or
a thin runner on the same shared modules (`pf1_window`, `scoped_remote`, `spark_quiet`, `post_close_watch`,
`nbw_cells.greedy`); either way the level-1 fake-transport rehearsal of WIN2.md runs on Controller first.

## Shape

| | |
|---|---|
| Arms | C/X/C: `c1` control (production launch, `RSP_SP=0`), `x1` candidate (`RSP_SP=1`: R-SP build + `VLLM_DSV41_PREFILL_SP=1`), `c2` control |
| Decision cells (×3 per start) | `32k-b1` and `150k-b1`, primary metric `ttft_s` (lower is better); candidate repeat r paired with mean(c1 r, c2 r) |
| Reported (×1 per start) | `short-b1` (committed tok/s, ms/round: SP must not touch decode), `mixed150k-2decode` (completion time, long-doc TTFT) |
| Rule | `decision_v2.decide_window(series, ('32k-b1', '150k-b1'), groups={'single-user': both})`: WIN = gain > 2 % and the 95 % paired-t interval excludes 0; LOSS = worse than own control spread in 2 of 3 |
| Greedy | six W5 prompts on `x1` via `nbw_cells.greedy(..., near_tie_nats=0.901)` (policy `rsp-near-tie-0p9`, NDM 0.9 nats). Expected IDENTICAL by construction; anything but IDENTICAL / NEAR-TIE makes the window NOT A CANDIDATE |
| Memory | 1 s MemAvailable on both Sparks; shared 4 GiB floor (production chunk 4,096); report each arm's minimum during 150K prefill (ESTIMATED SP gain +80 MiB per rank) |
| Archive | `archive_append(control-archive.jsonl, window.archive_rows(...))`: every c1 and c2 repeat of every cell and metric with `max_tokens` 512, `start`, `repeat`, the 40-hex production commit, the date and `clock_cap_mhz` of that row's own start (S92 FIX 2 / WIN2-R2); report `window.spreads(archive, commit=..., clock_cap_mhz=...)` next to the in-window bracket spreads |
| Chain | start: `window.preflight(chain.production_serving(), chain.hold_exists())` (chained: c1 is the baseline); end: `window.close(stopped)` = `chain.skip_restore` (CHAINED: quiet proof, `chain.finish_chained`, no W8, no watch; otherwise strict W8 restore + watch + `chain.clear()`) |

## Sequence

1. Before the HOLD (Controller): in the experiment checkout `python -m offload.sp_prefill.build_overlay --output
   offload/sp_prefill/generated`; `sha256sum -c offload/sp_prefill/OVERLAY.sha256` inside it; CPU suite
   `pytest offload/sp_prefill`; `window.py --plan`; runner rehearsal (WIN2 level 1).
2. Gate: `chain_prev_ok.sh` (chained) or the usual prev-ok; HOLD opens.
3. **Microbenchmark** (Sparks quiet, ~5 min, below). `STOP SP ARM` → run c1 only, report, close (the controls still
   go to the archive).
4. `c1`: start (`RSP_SP=0`), warm-up, cells in PC1's order (short, 32K ×3, 150K ×3, mixed).
5. `x1`: start (`RSP_SP=1`). Both ranks' logs must show `dsv41 R-SP: prefill sequence parallelism from 3073 tokens`
   (an `off:` line or another number is a STOP: the arm would measure K1). Greedy six before the cells; then cells.
6. `c2`: as c1.
7. `window.decide(measured, greedy)` → summary `decision`; archive rows; spreads.
8. `window.close(stopped)`.

ESTIMATED HOLD (PC1's measured-derived budgets: start-up 404 s, warm-up 90 s, transition 60 s, greedy 60 s, cells
35 / 100 / 20 / 145 s): one start ≈ 19.7 min; **≈ 68 min as a chained (non-last) window, ≈ 93 min as the last window**
(W8 restore 12 min + 15-min post-close watch).

## Microbenchmark (in-window, `rs_bench.py`)

Why: the CPU proof assumes NCCL's reduce-scatter adds exactly what its all-reduce adds, and SP's cost on the RoCE
link is two more collective launches per full-row layer. Both are cheap to settle with the Sparks quiet before an arm
is spent.

- One process per Spark in the experiment image (torch only), NCCL with production's NCCL variables (the serve's
  `OFFLOAD_DOCKER_ENV` `NCCL_*` entries), the checkout mounted read-only, rendezvous on the TP address:
  `torchrun --nnodes 2 --nproc-per-node 1 --node-rank {0,1} --master-addr <sparkhead TP address> --master-port <free>
  -m offload.sp_prefill.rs_bench --out <run dir>/rs_bench.json` (docker invocation as `start_exp.sh` runs the image;
  `--plan` prints the sizes only).
- Sizes: T = 4,096 (production chunk), 8,192, 16,384 rows × 5,120 bf16 (Tp-row reduce-scatter, L-row all-gather) and
  the replay-row fetch (5 all-gathers of 128 / 512 rows); 5 warm-up + 20 timed iterations each, CUDA events.
- Output per T: median / p90 ms of all-reduce, reduce-scatter, all-gather; per-layer K1 (2 AR) vs SP (2 AG + 2 RS);
  per-chunk delta over the 20 full-row layers; `rs_equals_ar_bitwise`.
- Gate: every T bit-identical → `SP ARM MAY RUN`; otherwise `STOP SP ARM` (exit 3). The timings are reported, never
  decide: the TTFT cells decide.

## What the first window must prove

1. Start-up with the switch on, the threshold line on both ranks (3,073 tokens), no `off:` reason.
2. RS == AR bitwise on NCCL over RoCE at 4K–16K rows.
3. Greedy six IDENTICAL (registered tolerance 0.9-nat near-tie; a near-tie would point at a row-count-dependent kernel
   such as DeepGEMM's mega mHC).
4. 32K and 150K TTFT under win2-v1 (ESTIMATED +1–3 %: INCONCLUSIVE is a likely, acceptable outcome).
5. MemAvailable ≥ 4 GiB throughout; the per-arm minimum difference (ESTIMATED +80 MiB per rank at chunk 4,096).
6. No decode change: short-b1 ms/round and committed tok/s within the control spread (reported).
