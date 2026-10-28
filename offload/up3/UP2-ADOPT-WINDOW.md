# UP2-ADOPT — 0075 replay skip qualification window (group `replay` of the UP3 runner)

NDM 2026-10-27: asked whether 0075 "should go through one qualification window under your 0.9-nat standard and into
production?" — "do this yes". Then (~19:10Z): "Fairer win rule is essential … go window after window". Brief:
`documents/plans/lanes/UP2-ADOPT.md`; decision rule: `documents/plans/lanes/WIN2-MEASURE-CHAIN.md` (win2-v1).

## What the arm is

CURRENT production 596ba1c / `prod-k1-327-ops1-20261026` (K1 327680, KV 2502600, TP2, x86host worker fb68c52) plus exactly:

* env `VLLM_DSV41_REPLAY_SKIP=1`;
* `offload/up3/overlay/replay/` (rebuilt and verified byte for byte by `python -m offload.up3.build`, pinned in `manifests/replay.json`):
  four vLLM files mounted per file (`vllm/v1/core/sched/scheduler.py`, `vllm/v1/worker/gpu/input_batch.py`,
  `vllm/models/deepseek_v41/nvidia/model_state.py`, new `vllm/models/deepseek_v41/replay_skip.py`) and the remote DSpark
  package mounted as ONE directory (production's 19 staged `*.py` with 0075 applied to gather/roundcore/round/speculator,
  plus the new `suffix_provenance.py`) in place of production's `remote_dspark` directory mount, which `window.env` removes
  exactly once (refusing 0 or 2+ occurrences).

Bytes come from production's own seeds (image files from `/opt/dsv41/data/src-image`, remote package from git
596ba1c) and UP1's four reviewed 0075 patches in UP1 order (`0075-process`, `-successor`, `-suffix`, `-ready-waiter`),
applied with `--fuzz=0` (offsets allowed exactly as UP1's builder allowed them; every output byte is pinned). Seven of the
files are byte-identical to the bytes UP1 measured on 2026-10-25; `model_state.py` lacks only UP1's dormant 0067 hunks and
`speculator.py` lacks only UP1's dormant D2 capture hook (production never had it). The scheduler equals UP2's reviewed
scheduler (`offload/upstream_adopt/MANIFEST.json`). K1's mounted files are untouched (none overlaps).

`start_exp.sh` runs `offload.upstream_adopt.replay_guard` for REPLAY=1; with `UP3_ARM=replay` it uses `verify_up3`:
exactly the UP3 replay scheduler and remote overlays, no duplicate overlay target, and no raw `-v` mount that replaces,
shadows or overlaps the scheduler, the remote package or any replay overlay.

## Gates (any failure is a STOP, then the strict restore)

* Every start: six greedy vs K1-post through the shared NBW checker at `--near-tie-nats 0.901` (0.9+1e-3; registration
  `up2-0075-adopt-0p9`, recorded in each greedy artifact); OPS1 warm-up; PF1 identity (`runtime()`: env differs by the one
  switch only, mounted sources = production + the 24 manifest files, argv = production with port 8001).
* First control: same-start A/A numeric probes at 32K and 150K (UP2's `up2_request --numeric`: T0, 64 tokens, first-token
  top-20 probabilities as token hashes). Replay arm: one probe per context within UP2's registered bound
  `max(.005, 2 × A/A KL)` with captured common mass ≥ .99 (`up2_checks.numerical`); 150K needle 6/6, no foreign codes.
* Provenance (UP2's definition): `ck.STOP_COUNTERS` (fallback_internal_error, replay_suffix_incomplete,
  provenance_errors, partial_window, …) are ZERO from engine start on every start (checked at the post-warm-up
  baseline, then never grow; S97 FIX 1), and both rank logs are free of `ck.STOP_LOG`
  (`invalid replay tap provenance`, `replay skip would remove taps`, `PARTIAL_WINDOW`, …).
* WIN2 greedy refinement (`greedy_kl.py`, opt-in, checklist 6): the replay arm's greedy records each divergence's
  top-20 and prefix (`--record-distribution`); the LAST control re-probes those prefixes with a fresh salt and requires
  KL(control || replay) ≤ max(.005, 2 × the window's worst A/A numeric KL) with common mass ≥ .99. Six IDENTICAL means
  nothing to re-probe.
* NOT RUN: UP2's native 12-check receipts (the apparatus does not exist, roadmap row 24). Every gate decision records it.

## Measurement and decision (win2-v1, `offload/launch/windows/decision_v2.py`)

| cell | repeats/start | role | metrics |
|-|-|-|-|
| `150k-b1` (one cold 150K request, 512 tokens) | 3 | DECISION | cold TTFT (primary), server ms/round (guard) |
| `32k-b1` | 4 | DECISION (no regression) | cold TTFT (primary), server ms/round (guard) |
| `short-b1` | 4 | reported | ms/round, accepted/round, decode tok/s |

Paired C/X/C: repeat r of the arm against the mean of repeat r of the two controls. WIN = point gain > 2 % and the 95 %
paired t interval excludes 0; LOSS = worse than the cell's own control spread in ≥ ⌈2n/3⌉ repeats; a guard can only cause
LOSS. The summary `decision` IS `decide_window`'s win2-v1 dict (`status` CANDIDATE / INCONCLUSIVE / NOT A CANDIDATE)
plus `adopt`. **`adopt` = ADOPT CANDIDATE iff every gate PASSED, neither decision cell is a LOSS, and neither decision
cell's cold TTFT is significantly slower (its whole 95 % interval below zero; `significantly_slower` lists them; S97
FIX 2: a consistent slowdown inside the control spread is INCONCLUSIVE, never adoptable).** No speed win is required: the gain is
the MEASURED 2026-10-25 evidence (`UP2-screen-300K-20261025T131508Z`: 300K cold request ON 130.3–130.9 s vs OFF
139.6–141.8 s, paired median 0.9317, spread 0.105 %, on the UP1 composition). A 150K WIN (win2 status CANDIDATE) is
reported as additional evidence (Fable: +1 to 4 % ESTIMATED at 150K). Every control repeat of a FINISHED control start
is appended to the WIN2 control archive `$C/control-archive.jsonl` (schema v3, WIN2-R2) with `max_tokens` 512, `start`,
`repeat`, the 40-hex production `commit`, the `clock_cap_mhz` both Sparks printed in that start's own start log
(`clock_source` start-log; a start log without ONE cap on both Sparks stops the window) and a `YYYYMMDD` date, kept even if
a later start stops; the start's `checks.archive` records the archive head before and after. The decision reports the
archive's day-to-day spread for that definition (max_tokens 512, production commit, the controls' clock cap) next to
the in-window one.
The window never changes production; promotion is the separate coordinator window (`offload/promote/up2`).

## Budget

ESTIMATED (runner formula): blocks 1439/1414/1179 s, HOLD 81.20 min ≤ 85 incl. 120 s setup + 720 s recovery
(`--dry-run --group replay` prints it). MEASURED basis on the same production (UP3-dual-ring-W-20261027T165101Z):
start-to-gates 420–440 s, short-b1 49–51 s, 32k-b1 106–109 s, restore 525 s; 150K B1 from the MEASURED 69.4 s cold TTFT
plus 512 tokens at 40–42 tok/s; numeric/needle probes from the MEASURED 150K/32K TTFTs (69.4 / 14.9 s). With those the
window is about 65 minutes.

## Launch (coordinator only; one HOLD; never concurrently with another window)

Stage the exact reviewed commit of `offload/up2-adopt` on BOTH Sparks at `~/dsv41-deploy-offload` with the coordinator's
reviewed staging procedure (as for UP3 run7). Then, from a clean local worktree at that same commit:

```bash
source ~/.config/ndm/paths.sh
: "${UP2A_REVIEWED_COMMIT:?full independently reviewed commit required}"
export WINDOW_COORDINATOR=Deepseek/coordinator
UP2A_OUT=/opt/dsv41/campaigns/DSV41-OFFLOAD/UP2-ADOPT-W-$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=$PWD:$PWD/offload/dspark /home/operator/miniconda3/bin/python3 -m offload.launch.windows.up3_window \
 --execute --group replay --out "$UP2A_OUT" \
 --remote-repo dsv41-deploy-offload \
 --expected-experiment-commit "$UP2A_REVIEWED_COMMIT"
```

To stop: create `$UP2A_OUT/STOP` and stay attended through restoration and the watch. Chain mode (`WINDOW_CHAIN=1`,
`WINDOW_CHAIN_LAST=1` on the last window) comes from the UP3 runner's WIN2 adoption (offload/up3-win2 1530c9c, merged):
a clean non-last replay window leaves production DOWN with the CHAINED marker; any stop or the last window runs the
strict six-IDENTICAL W8 restore (both tested for this group). A chained window compares against the chain's first UP3
capture (`--production-identity` when the previous window was another runner). A promotion never runs chained.
WIN2-R2 (offload/win2 2b48eec, merged): a chained preflight requires the hold placeholders on both Sparks; the chained
end closes only the HOLD file with the shared `chain.CHAINED_CLOSE` (numbered close transaction of the window's own
HOLD, then the fenced commit-close), so the placeholders stay up; the end is decided with `chain.end_action`, and a
window that owns no HOLD while the marker is present (any refusal before HOLD in a chain) runs `chain.abort` (strict
W8 under its own scoped HOLD + watch) instead of "NOT NEEDED"; the marker clears only after the window's own RESTORED
**and** post-close watch PASS. The coordinator orders every window and archive append.

## CPU verification

```bash
S=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/UP2ADOPT; unset TMUX TMUX_PANE
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 TMPDIR=$S/tmp TMUX_TMPDIR=$S/tm PYTHONPATH="$PWD:$PWD/offload/dspark"
/home/operator/miniconda3/bin/python3 -m offload.up3.build
/home/operator/miniconda3/bin/python3 -m offload.up3.qualification_lock
/home/operator/miniconda3/bin/python3 offload/up3/qualify.py --check-package
/home/operator/miniconda3/bin/python3 -m offload.launch.windows.up3_window --dry-run --group replay
/home/operator/miniconda3/bin/python3 -m pytest -q -p no:cacheprovider offload/up3/tests offload/upstream_adopt/tests \
 offload/launch/windows/tests/test_kl1_window.py offload/launch/windows/tests/test_nbw_window.py \
 offload/launch/windows/tests/test_spark_quiet.py offload/launch/windows/tests/test_production_health.py \
 offload/launch/windows/tests/test_post_close_watch.py offload/launch/windows/tests/test_decision_v2.py \
 offload/launch/windows/tests/test_chain.py offload/launch/windows/tests/test_greedy_kl.py \
 offload/launch/windows/tests/test_chain_stop.py offload/launch/windows/tests/test_s92_regressions.py \
 offload/launch/windows/tests/test_win2_history.py
/home/operator/miniconda3/bin/python3 -m mypy --strict --explicit-package-bases offload/up3/*.py offload/up3/tests \
 offload/launch/windows/up3_window.py offload/upstream_adopt/replay_guard.py offload/upstream_adopt/tests/test_replay_guard.py
bash -n offload/up3/window.env
```
