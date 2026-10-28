# PF1-WINDOW coordinator sheet: attributing the 524,288 cold-prefill cost

For the coordinator (Claude session Deepseek / coordinator) only. The PF1-WINDOW lane wrote and CPU-rehearsed everything
here. It never ran the precheck, the runner or any command below against sparkhead, sparkworker-wired or x86host. Every duration is
**ESTIMATED** (from the MEASURED inputs in `PF1-WINDOW-DESIGN.md`); every threshold is a window-reading rule, not a K1
promotion gate (those are unchanged).

## In plain language

- **What the window answers.** K1 at max_model_len 524,288 makes cold prefill about 35 % slower than at 327,680 (S18:
  +0.163 ms per prompt token). This window checks two things on today's production stack:
  - whether that gap is still there;
  - if it is, which part of the prefill step carries it.
- **How much production time it takes.** One HOLD of about **67 minutes** (about 46 if the gap is gone or unclear).
  Production is down for about 66 (45) of those minutes. The runner always ends with the W8 restore (start, check,
  close) and refuses to call the window done unless production is verified.
- **Your steps, in order:**
  1. review and stage;
  2. run the read-only precheck (about 1-2 min, production keeps serving);
  3. dry-run, then `--execute`;
  4. read `summary.json` (gate) and, if the ON pair ran, `fit.json` (spans);
  5. pick the next step from the table at the end.

## 1. Before the window (once per reviewed commit)

1. **Review.** Review branch `offload/pf1-window`. Its head becomes the reviewed commit. Run every command below from a
   clean Controller checkout at exactly that commit; the runner refuses a dirty or different tree.
2. **Reserve 90 minutes** with no other producer. The runner's own bounds from the HOLD timestamp:
   - submissions close at +75 min;
   - measurement aborts at +80;
   - restore is done by about +90 even after an abort.
3. **D2.**
   - `D2-GEN.PAUSE` must exist in the campaign directory, or no `d2-gen` tmux session may be running.
   - No D2 GPU job may run on the two drafter GPUs on x86host during the HOLD. Every arm uses production's TP2 drafter;
     a co-tenant would load the arms unequally. The precheck checks the GPU processes.
4. **Experiment checkout.** sparkhead `~/dsv41-deploy-offload` is reused, never `~/dsv41-deploy`. If it is not at the
   reviewed commit, add `--stage-bundle`. The runner then bundle-stages the commit, as W13 did; the checkout must have
   no tracked changes.

```bash
source ~/.config/ndm/paths.sh
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export PF1_REVIEWED_COMMIT=$(git rev-parse HEAD)          # in the clean reviewed checkout
export PF1_OUT=/opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-W1-$(date -u +%Y%m%dT%H%M%SZ)
```

## 2. Read-only precheck (the coordinator runs it; production keeps serving)

```bash
bash offload/launch/windows/pf1_precheck.sh "$PF1_REVIEWED_COMMIT" --out "$PF1_OUT"
```

It runs every check, even after a failure. Each prints `PASS`/`FAIL`/`INFO`, and the last line is `PRECHECK PASS` or
`PRECHECK FAIL: n check(s)` (exit 1). It writes nothing anywhere; its only output is stdout. It never touches :8001 or
:18001 and never sends a request to the x86host worker. These are the runner's own preflight and snapshot checks, run
before the HOLD so that a failure costs no production time (code `offload/launch/windows/pf1_precheck.py`).

**Controller (local):**

| Check | Why |
|---|---|
| local HEAD = reviewed commit, no tracked changes | the runner refuses otherwise |
| `W8/restore.sh` sha256 = `d1c4479d…` | the runner checks its exact STOP strings |
| D2 paused or absent | brief and LANES rule |
| public corpus readable (`opinions_cl`, `filings_recap`) | prompts are cut from it |
| 127.0.0.1:18001 not listening | the runner owns the tunnel |
| `--out` is new and directly in the campaign directory | the runner refuses otherwise |

**sparkhead:**

| Check | Why |
|---|---|
| `~/dsv41-deploy` at 21bed31 = tag `prod-k1-327-20261023`, `dsv41.env`/`serve.sh`/`start.sh` clean, no `HOLD` | production ledger |
| `~/dsv41-deploy-offload` clean; at the reviewed commit (else INFO: use `--stage-bundle`) | experiment checkout |
| the 7 staged pin files equal the reviewed tree (when already staged) | `PF1-window.env`, launchers, trace pins, K1 manifest |
| `offload/prefill_trace/generated` absent or equal to `OVERLAY.sha256` | otherwise the ON phase would STOP after the OFF arms |
| :8000 `/health`; `/v1/models` = one model, 327,680, `/srv/dsv41/legal-v2a` | production is K1 327,680 |
| :8000 `/metrics` running + waiting (INFO) | `window.sh open` drains; the runner waits at most 20 min |

**sparkhead and sparkworker-wired (each):**

| Check | Why |
|---|---|
| `dsv41-serve` running image `sha256:435a5236…`; :8001 not listening; no `dsv41-hold` session | no stale experiment or window |
| in-container 13-pin verify; image trace bases = `BASES.json` (hash-only `docker exec`) | same check as the runner's snapshot, but before the HOLD |
| nvidia-smi fields readable; MemAvailable ≥ 5 GiB | start floor (guard: ≥ 4 GiB always) |
| ≥ 10 GiB free in home; old `pf1-trace` run directories counted | trace output space |

**x86host (status file, /proc and nvidia-smi only):**

| Check | Why |
|---|---|
| worker fb68c52, TP2, `--max-model-len 1048576`, GPU identity, status fresh, no TP errors | the drafter in every arm |
| the two drafter GPUs are used only by the worker's process tree | no D2 or other co-tenant |

## 3. The window

```bash
bash offload/launch/windows/pf1_window.sh --dry-run        # the full plan; no command, network or file

export WINDOW_COORDINATOR=Deepseek/coordinator
bash offload/launch/windows/pf1_window.sh --execute \
  --expected-experiment-commit "$PF1_REVIEWED_COMMIT" \
  --out "$PF1_OUT"                                          # add --stage-bundle if the precheck said so
```

What it does (`PF1-WINDOW-DESIGN.md` has the minute-by-minute plan):

1. Preflight (read-only), then `window.sh open`.
2. Snapshot production on both ranks into `rollback.json`.
3. `off-327` → `off-524`. Each arm runs a warm-up of 32K, then 150K and 300K cold ×2, with 1,500 output tokens; prompts
   and salts are matched.
4. Gate.
5. If the gate says REPRODUCED: stage the hash-checked trace sources, then `on-327` → `on-524` (150K traced ×1 each),
   collecting both ranks' trace files.
6. **Always** run `W8/restore.sh start` (only if production was stopped), then `check`, then `close`. It requires these
   lines:
   - `K1 START VERIFIED`;
   - six `greedy K1-post vs restored: i IDENTICAL n` lines;
   - `PRODUCTION HOLD OK`;
   - `CHECK: DONE`;
   - `CLOSE: window closed`.

The last stdout line is JSON: `{"status", "stop", "gate", "restore"}`.

| Exit | Meaning | Coordinator action |
|---|---|---|
| 0 | complete, production restored and verified | read the results (section 4) |
| 1, `STOP before any remote action` | nothing was touched | fix the named cause; rerun with a new `--out` |
| 1, restore `RESTORED…` or `NOT NEEDED…` | the window stopped; production verified (or never stopped) | read `summary.json` `stop`; fix; precheck again; new `--out` |
| 2, `PRODUCTION NOT VERIFIED; HOLD left open` | the restore failed a check | read `$PF1_OUT/cmd/restore-*.log`, then run the steps below by hand |

For exit 2, run by hand, in order:

```bash
bash /opt/dsv41/campaigns/DSV41-OFFLOAD/W8/restore.sh start   # only if production is not serving
bash /opt/dsv41/campaigns/DSV41-OFFLOAD/W8/restore.sh check
bash /opt/dsv41/campaigns/DSV41-OFFLOAD/W8/restore.sh close
```

Common STOPs and their remedies:

| STOP | Remedy |
|---|---|
| `experiment checkout is not at the reviewed commit` | add `--stage-bundle` |
| `D2 generation active without pause` | pause D2 by its own runbook |
| `MemAvailable below 5 GiB` | wait; the memory owner is not this window |
| `HOLD is not this window's…` | never adopt or renew another HOLD |
| supervisor: `MemAvailable … below 4 GiB`, `new swap use after readiness`, `foreign demand on the experiment`, `stale Spark telemetry`, `GPU telemetry failing` | investigate the cause before any rerun |
| `80-minute measurement bound … reached` / `not enough time before the submission bound` | the run was slower than ESTIMATED; use HOLD 1-off plus HOLD 2 |

To abort a running window yourself, `touch "$PF1_OUT/STOP"`. The supervisor stops within its poll interval, and the
runner then restores.

An ON-arm warm-up failure (the engine raises `PF1 trace requires CUDAGraphMode.NONE for prefill`) means the traced
prefill ran in a captured graph. The OFF results remain valid. The next step is HOLD 2e (below).

## 4. Results

The run directory `$PF1_OUT` holds numbers and hashes only (no prompt or generated text):

- `summary.json`: gate, cells, checks, restore, artifact hashes;
- `checkpoints.jsonl`, `deadlines.json`, `rollback.json`;
- `*-identity.json`, `*-requests.jsonl`, `requests-ledger.jsonl`, `proxy-stats.jsonl`;
- telemetry; `cmd/*.log`;
- for the traced arms: `traces/<run>/<host>/prefill-r{0,1}-p*.jsonl` and `on-*-trace-check.json`.

**Gate** (`summary.json` → `gate`):

- `status`: REPRODUCED, NOT_REPRODUCED or AMBIGUOUS.
- For each of 150K and 300K, `points[]` gives:
  - `median_ttft_s` per arm;
  - `gap` (median 524 / median 327 − 1);
  - `excess_ms_per_prompt_token`, against S18's 0.163.

**Spans** (only if the ON pair ran). Check `arms.on-*.checks.trace.correlation`:

- `HASH`: run the command below as written.
- `BY_SIZE_ONLY…`: the engine's request id is not the SSE id; add `--match-by-size`. This maps the one traced request
  of each cap and size. It refuses if that request is not unique or not identical on both ranks.

```bash
"$HARNESS_PYTHON" -m offload.prefill_trace.fit_cost \
  --traces "$PF1_OUT"/traces/*/*/prefill-r*-p*.jsonl \
  --requests "$PF1_OUT"/on-327-requests.jsonl "$PF1_OUT"/on-524-requests.jsonl \
  --output "$PF1_OUT"/fit.json            # + --match-by-size when correlation is BY_SIZE_ONLY
```

`fit.json`, per rank:

- `ranks[].models_common_support.cuda_ms`: the chunk-level model.
  - `high_cap_extra_ms` (c) is the flat per-chunk extra at 524,288.
  - `high_cap_prefix_extra_ms_per_100K` (d) is its growth with prefix.
- `spans.rankN`:
  - `chunk_total_median_delta_ms` (**Δchunk**): the matched-prefix median of the per-chunk CUDA delta between caps.
  - `labels[<span>]`: `median_delta_ms` and `share_of_chunk_total_delta` for each instrumented span. Spans nest and
    overlap, so read each share alone and never add shares.
  - `indexer_plan`: counts per cap of workspace overflow, group splits and query subchunking.

Expected size of the effect (ESTIMATED from MEASURED S18):

- 0.163 ms × 4,096 tokens ≈ **0.67 s per full chunk**;
- the window's own figure is `excess_ms_per_prompt_token` × 4,096.

**Coverage** = Δchunk / (window excess × 4,096). Trace overhead (≤ 20 %, ESTIMATED, never measured) affects both caps.

## 5. Which result justifies which next step

The design numbers (≥ 10 % gap, < 5 %, 3 % spread, coverage and share cut-offs) are reading rules for this window. K1's
promotion gates and NDM's 5 % speed-loss rule are unchanged.

| Result | Reading | Next step |
|---|---|---|
| Gate **NOT_REPRODUCED** (gap < 5 % at 150K and 300K, spreads ≤ 3 %) | the S18 cost is absent on today's stack (TP2 drafter, 21bed31, process proxy) | **Promote path, no cost to accept**: the coordinator runs K1's existing 524,288 promotion gates (S18's R5 rows) unchanged. No trace work |
| Gate **AMBIGUOUS** (gap between 5 % and 10 % at either point, or a spread > 3 %) | drift or noise cannot be excluded | No promote/abandon decision. Run **HOLD 1b** (return arm; see the gaps below), or repeat HOLD 1 as `--phase off` on another night. Never combine cells from different HOLDs |
| **REPRODUCED**, coverage ≥ 0.8 on both ranks, one span family with share ≥ 0.5 | the extra time is inside the traced prefill step, in a named span | **Fix candidate**, by span family (rows below). It becomes a new K1 lane: CPU qualification first, then its own matched window. It must clear K1's unchanged gates before any promotion |
| … `indexer.gather` / `indexer.mqa` / `indexer.topk`, with more `workspace_exceeded` / `group_split` / `query_subchunked` at 524,288 | indexer work or workspace planning sized by max_model_len | candidate: size the indexer workspace and logits budget by the scheduled request, not the cap |
| … `metadata.prepare` / `metadata.indexer_chunk` / `sparse.metadata_scan` | the block-table scan bound grows with the cap (scan bound, not a proof of a native scan) | candidate: build metadata over the used block-table extent |
| … a `sparse.dispatch.*` label present at one cap only | the dispatch path changes with the cap | candidate: the dispatch choice at 524,288. FlashInfer's native choice needs its own timeline, not a guessed threshold |
| … `collective.*` | transfer **and wait**: usually a symptom | compare both ranks. A candidate only if an upstream span on the other rank grew; otherwise treat it as "no single span" (next row) |
| **REPRODUCED**, coverage ≥ 0.8, no family with share ≥ 0.5 | the cost is spread out or in uninstrumented native/fused code | No fix candidate from this trace. Decide between a profiler-timeline design (a new lane; not this recipe) and the promote/abandon rows below |
| **REPRODUCED**, coverage < 0.5 | the cost is outside `chunk.total`: input-batch preparation, scheduling, sampling, proxy or drafter (the trace excludes these) | No fix candidate yet; next: a host-side timing design (a new lane) |
| **REPRODUCED**, and no fix is within reach, or a fix fails its own window | the cost is real and stays | **Promote 524,288 with the cost accepted**: NDM's decision on the window's MEASURED medians (150K and 300K TTFT at each cap). It needs an explicit NDM waiver of the 5 % speed-loss rule. K1's other gates are unchanged |
| as above, and NDM declines the cost | — | **Abandon 524,288 for now**: production stays at 327,680; the PF1 line closes, with the window's evidence recorded |
| the same safety STOP twice (memory, swap, identity, foreign demand) | the apparatus is not safe as built | **Stop**: no rerun until the cause is understood and the runner is re-reviewed |

## 6. ESTIMATED minutes per HOLD

| HOLD | Command | HOLD | Production down | Built |
|---|---|---:|---:|---|
| precheck (no HOLD) | `pf1_precheck.sh` | 0 (about 1-2 min of reads) | 0 | yes |
| **1** (default) | `--execute` (`--phase all`) | ≈ **67**; ≈ **46** if the gate stops after the OFF pair | ≈ 66 / 45 | yes |
| 1, 300K traced too | `--on-contexts 150000 300000` | ≈ 75 (last submission ≈ T+64) | ≈ 74 | yes |
| 1-off | `--phase off` | ≈ 46 | ≈ 45 | yes |
| **2** (ON arms in their own HOLD) | `--phase on --gate-from <REPRODUCED, restored HOLD-1 dir>` | ≈ 31 | ≈ 30 | yes |
| **1b** (gate AMBIGUOUS) | `off-327` → `off-524` → `off-327-return`, 150K/300K ×2 | ≈ 64 | ≈ 63 | **no**: needs a return arm and its gate rule in the runner |
| **2e** (ON warm-up refused) | `PF1_EAGER=1` on four matched arms, 150K | ≈ 55 | ≈ 54 | **partly**: config and identity checks support eager; no CLI option or eager gate yet |

Each HOLD uses a new `--out` and the same reviewed commit. HOLD 2 checks that its `--gate-from` directory ran the
same commit and config, REPRODUCED, restored, and that its evidence hashes are unchanged.

For HOLD 2:

```bash
bash offload/launch/windows/pf1_window.sh --execute --phase on \
  --gate-from /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-W1-<stamp> \
  --expected-experiment-commit "$PF1_REVIEWED_COMMIT" \
  --out /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-W2-$(date -u +%Y%m%dT%H%M%SZ)
```

## 7. After the last PF1 window

These are coordinator housekeeping steps. They are not needed between HOLDs: each run uses fresh trace directory names,
and a matching `generated/` is reused.

- **Copied trace directories.** The runner copied them into `$PF1_OUT/traces/`. Confirm the copy, then remove the
  remote copies:

  ```bash
  ssh sparkhead 'rm -rf ~/dsv41-offload-run/pf1-trace/<tag>-on-*'
  ssh sparkworker-wired 'rm -rf ~/dsv41-offload-run/pf1-trace/<tag>-on-*'
  ```

- **Experiment checkout.** `~/dsv41-deploy-offload/offload/prefill_trace/generated` (untracked, hash-checked) can be
  removed, or left for the next run.

## 8. Cause window (PF1-CAUSE-WIRE, coordinator only)

CPU wiring and rehearsal add an opt-in phase; no cause window has been run by this lane. Review the full
[span design and pre-registered reading](../prefill_cause/WINDOW.md) before execution. Existing phases and K1 promotion
gates are unchanged. Reserve **ESTIMATED 27 minutes HOLD** for the two cause arms or **36.5 minutes with T0**
(production down approximately 26 / 36 minutes; based on MEASURED window-2 timings).

From a clean reviewed checkout, the coordinator alone runs:

```bash
source ~/.config/ndm/paths.sh
export PF1_REVIEWED_COMMIT=$(git rev-parse HEAD)
export PF1_CAUSE_OUT=/opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-CW-$(date -u +%Y%m%dT%H%M%SZ)
export WINDOW_COORDINATOR=Deepseek/coordinator
bash offload/launch/windows/pf1_window.sh --execute --phase cause --cause-t0 \
  --gate-from /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-W1-20261025T234254Z \
  --expected-experiment-commit "$PF1_REVIEWED_COMMIT" \
  --out "$PF1_CAUSE_OUT" --stage-bundle
```

Omit `--cause-t0` for the two-arm sequence. Cause always uses PF1-cause.env and instrumented spans (including T0),
32K warm-up / 64 tokens then 150K cold ×1 / 1,500 output per arm. The existing precheck remains a check of the
original PF1-window sources; it does **not** certify the cause-specific source tree. The cause runner itself checks the
additional config/pins, stages both overlay trees, refuses different existing generated sources, and verifies all
18 selected engine paths and exact allowed env/mounted-byte changes on both ranks. Its cause-only ancestor allowance
admits reviewed cause additions plus inert documentation/D2 training/analysis changes; launch/K1/PF1-R2 runtime/build
inputs and the identity-capture script must remain unchanged from window 1. The ordinary ON rule is unchanged.

The same HOLD, production-idle, deadline, telemetry, swap, demand, owned-process reaping, JIT wait, signal handling and
W8 restore paths apply. Cause additionally records kernel/driver/THP/fragmentation facts, checks >=5 GiB before every
start, and records per-arm observed MemAvailable minima for each host (including failed arms in `arm-memory.json`).
The >=4 GiB stop guard is unchanged. D2 pause and exclusive drafter use are prerequisites just as above; a live pilot
is not authorization to run this window alongside it.

After successful restoration, CPU analysis:

```bash
python3 -m offload.prefill_cause.analyze \
  --cause-window "$PF1_CAUSE_OUT" \
  --off-window /opt/dsv41/campaigns/DSV41-OFFLOAD/PF1-W1-20261025T234254Z \
  --output "$PF1_CAUSE_OUT/cause-analysis.json"
```

The output reports both ranks' per-layer MoE-half deltas and a separate T0 reading, with input hashes. Window-1 OFF
arms have measured TTFT but **no chunk traces**; T0 chunk totals are compared with the measured cause-327 range,
explicitly labeled. The fixed 5% / chunk-range rules are in WINDOW.md §6. No T0 arm gives `NOT RUN`. These are
experiment readings; greedy/logit-floor checks and existing K1 memory/promotion gates still precede adoption.
