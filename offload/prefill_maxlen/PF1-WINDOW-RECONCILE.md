# PF1-WINDOW item 1: the PF1-R2 recipe reconciled with today's production

Lane PF1-WINDOW, 2026-10-25, CPU only on Controller. No Spark, sparkworker, x86host or API access by this lane. Every
"MEASURED" value below comes from a saved file named next to it. Live values the runner must re-read in the window
are marked **RUNNER CHECKS**.

## Plain language

- The window **can run tonight**, once the coordinator has reviewed and staged this branch. The old recipe cannot run
  as written.
- The old recipe would have written into the production checkout `~/dsv41-deploy` in five places.
- Worse, it would not have tested today's production code. Its config mounts the experiment checkout's copy of the
  remote-drafter client, and that copy has drifted from what production runs (one file changed, one file added).
- The fix is a new config, `offload/launch/configs/PF1-window.env`. It reads production's own `dsv41.env` in a subshell
  and mounts production's own staged files read-only. The OFF arms therefore run production's exact bytes. They differ
  from production only in the port, the max_model_len and one Engram safety switch.
- The TP2 drafter on x86host accepts a 524,288 engine by its source. It has never served positions past 327,680 on a GPU
  (S22 FLAG D). The design therefore keeps every request at or below 301,500 positions. The drafter stays on in every
  arm, matched; the window stops if the worker refuses the 524,288 HELLO.

## 1. Experiment checkout: separate, bundle-staged, never production

| Item | Evidence | Window rule |
|---|---|---|
| W13 house pattern | LANES.md 2026-10-25 06:43/07:50/12:06: reviewed commit bundle-staged into sparkhead `~/dsv41-deploy-offload`, clean; Controller pinned clone | same: runner refuses unless remote HEAD = full reviewed commit and `git status --porcelain --untracked-files=no` is empty |
| `start_exp.sh` acts on the checkout that contains it | `offload/launch/start_exp.sh:9` (`cd "$(dirname "$0")/../.."`), `:12` sources that checkout's `dsv41.env`, `:50-53` writes `logs/exp-*` there | run only from `~/dsv41-deploy-offload` |
| Production checkout reads only | `start_exp.sh:11` and `window.sh:10` read/write `~/dsv41-deploy/HOLD`, the designated HOLD path | HOLD is the only production-checkout write (window.sh, unchanged); `PF1-window.env` *reads* `~/dsv41-deploy/dsv41.env` in a subshell |
| Production ledger | `git show 21bed31:dsv41.env` (tag prod-k1-327-20261023); `up2_low.py:333-336` checks it | **RUNNER CHECKS** HEAD = 21bed31 = tag, `git diff --exit-code HEAD -- dsv41.env serve.sh start.sh` |

Every place the **old recipe** (`offload/prefill_maxlen/GB10-TEST-PLAN.md`) would write in production paths:

| Line | Old action | Replacement |
|---|---|---|
| 72-73 | `scp`/`tar -x` the trace package into `sparkhead:dsv41-deploy/offload/` | generated trace sources go to `~/dsv41-deploy-offload/offload/prefill_trace/generated` (untracked, hash-checked) |
| 74 | `scp PF1-R2-trace.env` into the production `offload/launch/configs/` | the config is tracked in the staged experiment commit |
| 89-91, 194-205 | `cd ~/dsv41-deploy` then `start_exp.sh`, which writes `logs/` into production (`start_exp.sh:50-53`) and runs production's older 21bed31 `start_exp.sh` | `cd ~/dsv41-deploy-offload` |
| 175 | telemetry runs `$HOME/dsv41-deploy/offload/prefill_trace/telemetry.py` (absent at 21bed31: `git ls-tree 21bed31 offload/` has no `prefill_trace`) and writes on the Spark | stdlib telemetry streamed over SSH stdin to Controller; nothing written on the Sparks |
| 220-223 | `cat >` a temporary eager config into the production checkout | `PF1_EAGER=1` switch in the tracked config (contingency only) |
| 309-315 | restore with bare `./stop.sh; ./start.sh` and `window.sh close` (no greedy and no x86host check) | `W8/restore.sh start|check|close` (unchanged) |

## 2. Mounted engine bytes and environment

MEASURED from the W13 300K window, captured 13:15Z on 2026-10-25 with production serving
(`UP2-screen-300K-20261025T131508Z/0009-sparkhead.log`, `0013-sparkworker-wired.log`, from
`offload/dspark/d2/capture_window_identity.sh`). Each rank has 37 mounted-code hashes:

- The 13 K1 files equal `offload/kvpack/MANIFEST.json` (sha 559c5867…).
- The 20 `spec_decode` files equal the **21bed31** tree.
- The rest are 3 files under `/opt/dsv41-ofl` and `/serve.sh`.
- The image is `sha256:435a5236…` on both ranks (also `offload/operations/profiles.v1.json`). The engine is
  `0.31.1.dev0+g6e517b15.dsv41gb10`.

| Source | Production (21bed31 `dsv41.env`) | Old recipe (`PF1-R2-trace.env:2` → `D-kv6-async-pack.env`) | PF1-window.env |
|---|---|---|---|
| K1 13 files | `-v $HOME/dsv41-kvpack/559c5867ba42/…:ro` | checkout `offload/kvpack/overlay` (bytes equal to MANIFEST) | production mounts, read-only |
| remote_dspark + `spec_decode/__init__.py` | `$HOME/dsv41-offload-prod/dspark/…:ro` = 21bed31 tree (MEASURED above) | checkout `offload/dspark/spark/remote_dspark`. At this branch `speculator.py` differs (b42c0e6 D2 capture hook) and `suffix_provenance.py` is extra (4f36892, 0075); `git diff 21bed31 HEAD -- offload/dspark/spark/remote_dspark` | production mounts, read-only |
| ofl | `$HOME/dsv41-offload-prod/dspark/ofl:ro` | `$HOME/dsv41-offload-run/ofl:ro` | production mount |
| Base `DOCKER_ENV` | `serve.sh:49-59` at 21bed31 | `serve_exp.sh`: identical except it appends `EXP_DOCKER_ENV` (diff of the two blocks) | same |
| Engram `REQUIRED` | 0 | 1 | 1 on every arm (a down link fails the start instead of silently reading locally) |
| `ASYNC_RESPAWNS=3`, `RESPAWN_RESET_S=3600` | explicit | absent (defaults) | production's values |
| K1 switches, KV bytes, cap | 5 switches; 6,442,450,944; 327680 | same | read from production; refused unless `6442450944:327680:1:327680` |

Because of the drift, an OFF arm on the old recipe would not have been "today's production". The new config's test
(`offload/launch/windows/tests/test_pf1_config.py`, 18 cases) sources the real `git show 21bed31:dsv41.env`. It
proves three things:

- OFF env equals production token for token, except `REQUIRED=1`.
- OFF mounts equal production's `OFFLOAD_DOCKER_ARGS` byte for byte.
- ON drops exactly the two trace targets and adds the six hash-checked trace sources and an `rw` trace directory.

It also refuses inherited K1/spec overrides, a changed production cap/profile/factor/proxy, a non-`:ro` production
mount and a tampered trace source. **RUNNER CHECKS** (OFF arms): both ranks' mounted-code hashes equal the production
snapshot taken at the window's start, `/serve.sh` excepted. Argv equals production except `--port` and
`--max-model-len`.

## 3. K1 packing, factor, KV and trace pins

| Item | MEASURED value | Source |
|---|---|---|
| KV at 327,680 / 524,288 | 2,502,600 / 2,820,756 tokens | S18.report.md (R1, R5); K1-PROMOTE.report.md:74 |
| Trace base pins for K1 targets | BASES.json `model_runner.py` 682eb895…, `flashinfer_sparse.py` 0b45a382… equal the MANIFEST overlay hashes | this lane, CPU |
| Trace base pins for image files | `sparse_attn_indexer.py` 75d14da5…, `mla/indexer.py` 2da8f08a…, `parallel_state.py` a33fc846… equal `/opt/dsv41/data/src-image` | this lane, CPU |
| Rebuild | `build_overlay --enabled` → `sha256sum -c OVERLAY.sha256`: 6/6 OK | this lane, CPU (no forced hash) |
| src-image provenance | no provenance file beside the copy | **RUNNER CHECKS** the 3 image bases inside the production container on both ranks, O_DIRECT read, before any ON arm |
| Trace hash identity in-container | `emit_verify` OFF 13 pins, ON 17 pins | piped into both ranks after every start |

## 4. TP2 K2 drafter on the x86host pair

| Fact | Evidence |
|---|---|
| Running worker | revision fb68c52, `--tp 2 --max-model-len 1048576`, GPUs c4e7a371/7744b44b, `tp_comm_errors` 0. MEASURED 13:15Z (`UP2-screen-300K-…/0008-x86host.log`) |
| A 524,288 HELLO is accepted | `server.py:244` at fb68c52 (`1 <= req.max_model_len <= exp.max_model_len`); worker bound `server.py:781`; RoPE rows `max_model_len + 2048` (`model.py:166`) |
| Positions past 327,680 never GPU-served by TP2 | S22 FLAG D (`S22.report.md:43`); R5's 500K drafting used the one-card worker; `offload/promote/k1/README.md:195-198` |

**Decision: the remote drafter stays on in both caps, matched with production.** Every measured point is at most
300,000 prompt tokens plus 1,500 output, so no position exceeds 301,500. The only new input to the worker is the
524,288 HELLO, which its source accepts. A drafter-off pair would measure a different system from production, so it is
not used.

Three conditions stop the window and restore production:

- the worker rejects the HELLO;
- no `remote DSpark worker … ready` line appears;
- a warm-up or cold request shows no drafted rounds.

The runner also checks the worker read-only on x86host (revision, argv, GPUs, fresh status, zero `tp_comm_errors`), as
UP2 did (`up2_low.py:291-306`). It never changes the worker. The 400-500K points are out of scope for this window.
Testing them would also need the S22 FLAG-D stats check.

## 5. Memory, swap and demand guards

| Guard | Basis (MEASURED) | Window rule |
|---|---|---|
| MemAvailable floor | S18: lowest 4.60 GiB (sparkworker, R0 load); R5 minimum 5.64/6.40 GiB; R5 idle 0.95 GiB below R1 on sparkhead; sparkhead 3.09 GiB only under 2×500K (excluded: B1, ≤300K); production 6.56/7.55 GiB idle (LANES 2026-10-25 06:30) | STOP below **4 GiB** on either Spark, polled every 1 s from before HOLD to restore; ≥ 5 GiB required before each start |
| Swap | Every start swaps +3.3-4.5 GiB during model load, then stays flat (S18 runbook item 4) | load-time swap is recorded; after readiness, STOP if SwapFree falls > 256 MiB below that arm's ready baseline |
| Telemetry liveness | — | STOP on missing or stale (> 5 s) samples, monitor exit, or an unsupported nvidia-smi field (probed read-only in preflight) |
| Production demand | `window.sh:20-27` drains :8000 (≤ 60 min) before writing HOLD | the runner bounds the drain at 20 min, then refuses without HOLD |
| D2 generation | `up2_low.py:330-332` | requires `D2-GEN.PAUSE` or no `=d2-gen` tmux session |
| Foreign requests during the window | `window_requests.py:133-139` idle before each request | the supervisor polls :8001 metrics through the owned tunnel: running + waiting must not exceed the owned in-flight count (0 or 1); otherwise STOP |

## 6. Other corrections to the old recipe

- **Deadlines.** The old 150-minute submission / 180-minute hard budget (`GB10-TEST-PLAN.md:111, 282`) is replaced by
  HOLD-owned bounds: submissions until open + 75 min, measurement abort at open + 80 min, restore inside 90 min.
- **Warm-up.** `gb10_budget.capture` (`GB10-TEST-PLAN.md:144`) takes a different request path. It is replaced by a
  `--warmup` mode of `window_requests` (32,000 tokens, 64 output tokens with EOS ignored), whose usage and drafted
  rounds are checked.
- **Stop line.** The 1.5 GiB stop line (`GB10-TEST-PLAN.md:288`, `telemetry.py`) is raised to 4 GiB, per this brief.
- **Correlation risk (UNVERIFIED).** The trace hashes the engine's request id (`runtime.py:209`). The driver hashes
  the SSE response id (`window_requests.py:215`). If the installed API adds a suffix, `fit_cost` refuses. The runner
  records whether the hashes match. Each ON arm sends exactly one cold request per context, so chunks can still be
  assigned by their final `prefix + new` length. That fallback is reported as such, never relabelled as a match.

## 7. Conclusion

The window can run safely tonight once three things are true:

1. The coordinator has reviewed and staged this branch.
2. Production :8000 is idle, and D2 generation is paused or absent.
3. The x86host TP2 worker is healthy.

D2-DIAG uses the solo cards, not the drafter pair, and is unaffected. Nothing here edits production files, the K1
package, the restore script or K1's promotion gates.
