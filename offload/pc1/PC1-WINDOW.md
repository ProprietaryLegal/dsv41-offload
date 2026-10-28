# PC1 coordinator sheet

## WIN2 (2026-10-27; NDM "Fairer win rule is essential", "Do better deeper testing and measuring")

This section supersedes the Round 1-4 decision rule, cell table and budget below, which stay as the record of the
windows run before WIN2. Shared rule and checklist: `offload/launch/windows/WIN2.md` (WIN2-R2, merged at 2b48eec).

**Order (2026-10-27 expert review).** chunk16384 first, then chunk8192; a chunk16384 result is measured evidence only,
and its promotion waits for the MA1 16K memory-lifetime audit. threshold2048 / threshold512 are "Later" (the multi-user
prefill-threshold policy depends on how often 3+ users draft at once). WIN2-R3 (matched-seed paired cells, at least 6
pairs per decision cell) comes next; PC1 already runs one seeded repeat (seed 42+r) per driver call, which R3 can pair.

**Plans** (`cells.PLANS`; each start runs only its group's plan; decision cells x3, reported cells x1):

| Group | Decision cells (primary metric; guards) | Reported (x1) | Verdict groups |
|---|---|---|---|
| threshold2048, threshold512 | short B1 (ms/round; accepted/round, client tok/s) and mixed 150K + 2 decoders (ms/round; accepted/round, tok/s, 150K first token) | cold 32K, cold 150K | single-user = short; multi-user = mixed |
| chunk2048, chunk8192, chunk16384 | cold 32K and cold 150K (TTFT) | short B1, mixed | single-user = both |

Rule `win2-v1` revision 2 (shared `decision_v2`): the candidate's three repeats are compared unpaired (Welch t) with
the six repeats of the two bracketing controls; S92 found same-seed repeats are not true pairs (the two controls of the
threshold512 window ran short B1 repeat 0 at 1.51 and 1.18 accepted/round). WIN = point gain > 2 % and the 95 % interval
excludes 0. LOSS = the primary is worse than the control spread (1.69 x s / c4 of the controls) in 2 of 3 repeats; a guard
needs that AND its whole 95 % interval below zero. Any decision-cell LOSS = NOT A CANDIDATE, a WIN with no LOSS =
CANDIDATE, else INCONCLUSIVE. Starts whose start logs prove different clock caps decide nothing (`clock_caps_by_start`). Reported cells and reported metrics never decide. The summary also keeps
each metric's bracket (first control, candidate, last control) and, for COMPLETED control starts, appends every repeat
to `control-archive.jsonl` (schema v3: max_tokens 512, 40-hex production commit, that start's proven `clock_cap_mhz` and
`clock_source`, date YYYYMMDD, start, repeat; archive head before and after) and reports the archive spread of each
primary at the same max_tokens, clock cap and commit.

**Drafter faults** (lesson of PC1-threshold512-W-20261027T181314Z, which stopped on a drafter disconnect inside a
CONTROL start): every repeat is one driver call with the drafter counters read before and after it. A small fault
inside one repeat (at most one disconnect and one watchdog fire, <= 64 failed rounds, no breaker trip or respawn,
at most two per start) is logged in `drafter-faults.jsonl`, kept as evidence, and that repeat is re-run once with a
fresh salt if the remaining plan still fits the submit deadline. Anything larger, a third fault in a start, or growth
between repeats STOPs as before. The decision reports control-vs-arm attribution (NONE, CONTROL-ONLY, BOTH, ARM-ONLY-1,
ARM-SUSPECT); two or more arm-only faults with clean controls (p = 1/9 by chance) veto promotion.

**Chain mode** (`WINDOW_CHAIN=1`, last window also `WINDOW_CHAIN_LAST=1`; WIN2-R2 `chain` API):
- Start: `chain.preflight` with the placeholder probe (both Sparks' `dsv41-hold`) while the marker exists; a chained start
  admits its HOLD with `chain.allow_unreachable`, proves both Sparks quiet before the first `start_exp.sh`, and takes
  production from `chain.production_snapshot()` (copied into its `rollback.json` with `chained_from`).
- End, by `chain.end_action`: `chained` (stop-free, non-last) stops the experiment, proves quiet, runs the shared
  `chain.CHAINED_CLOSE` on sparkhead inside the window's `-restore-close` scope (HOLD file only; placeholders stay),
  commit-closes, writes the marker; summary `chain: CHAINED`, no W8, no watch. `restore` (own HOLD) runs the strict W8 and
  the watch, and clears the marker only on RESTORED + watch PASS. `abort` (no own HOLD, marker present) runs
  `chain.abort`, whose own strict W8 and watch the summary carries (`chain: ABORT`).
- Gate chained windows with `offload/launch/windows/chain_prev_ok.sh` (exit 3 = the chain stopped and the abort ran).

**Budget** (`budget.json` `win2`): admission allowances give ESTIMATED HOLDs of 76.7 min (threshold groups, chunk8192,
chunk16384) and 73.7 min (chunk2048). Projected from MEASURED PC1 starts and cells: threshold2048 67.7 min,
threshold512 71.8, chunk2048 63.0, chunk8192/16384 65.5 (larger-chunk arm cells and startup ESTIMATED). The mixed cell
measured 169-171 s per control repeat against its 145 s allowance; each start block's allowance still covers its
measured block (prepare is allowed 614 s against <= 436 s measured).

## Rounds 1-4 (before WIN2)


Round3: `chunk8192` is prepared for independent source/CPU review, subject to the
fresh-control reserve and continuous5GiB floor in [HEADROOM.md](HEADROOM.md).
`chunk16384` is registered but REFUSED: projected sparkhead3.59GiB/sparkworker 4.67GiB.
Do not schedule it. Each admitted arm still gets its own C/X/C HOLD.
New-arm cost is ESTIMATED78.20min, using measured353/337s PC1 readiness and the
existing404s basis plus180s candidate startup allowance; larger starts unmeasured.
Historical chunk2048 lost at32K; threshold2048 is NOT A CANDIDATE. This round
does not authorize repeating either arm or combining settings.

Prepared for source/CPU review. No performance winner, hardware qualification or
production promotion is claimed. Only Deepseek / coordinator runs a real window.
Source: PC1 lane brief; UP3 `06be7c0` lifecycle and Rounds4–7/S73 lessons.

The launcher is default OFF. `PC1_ARM` selects exactly one registered pair:

| Arm | max-num-batched-tokens | long-prefill-token-threshold |
|---|---:|---:|
| control |4096|1024|
| chunk2048 |2048|1024|
| threshold2048 |4096|2048|
| threshold512 |4096|512|
| chunk8192 |8192|1024|
| chunk16384 (REFUSED) |16384|1024|

Unregistered and combined changes refuse. A candidate requires `PC1_REVIEWED=1`, port8001
and a separate experiment checkout. `window.env` sets that marker after checking
the isolated arm, forbidden inherited overrides, and the actual production base.
It reads the production environment instead of using this development branch's
historical top-level launch settings. No production file is changed. `inputs.json`
pins44 production source/config inputs to596ba1c / prod-k1-327-ops1-20261026.
K1, target, fixed6GiB/rank KV budget,327680 cap, TP2 K2 fb68c52/process proxy, schedule, scheduler,
Engram policy, clocks, image, remote mounts and OPS1 are preserved. The only argv
changes are port8001 and the selected pair. Full environment/mounted-byte equality
is required. `hostflag.c` is a build input, never an installed-file requirement.

KV identity is arm-specific: control/threshold arms require exactly2502600;
chunk2048 requires the checked allocator's2933321;8192/16384 derive1934490/1330447.
[CAPACITY.md](CAPACITY.md) and [HEADROOM.md](HEADROOM.md) derive these from the fixed
pool and SWA admission formula. Every start records
expected and observed capacity in `summary.json`, including rejected observations;
rank1 normally logs no capacity line. Capacity is reported separately from the
unchanged TTFT decision. Cold salts continue to exclude prefix reuse.

Each HOLD has exactly OFF / selected-arm / OFF. Do not loop arms inside
one HOLD or combine winners. All three starts must complete. Round3 uses
`--group chunk8192` with a fresh matching output directory after independent review.

| Cell | Groups per start | Requests per group | Output tokens/request | Decision |
|---|---:|---:|---:|---|
| short B1 |2|1|512|decode no regression|
| cold32K B1 |2|1|512|TTFT target; decode no regression|
| cold150K B1 |2|1|512|TTFT target; decode no regression|
| mixed150K + two active32K decoders |2|3|512|each decoder p95 delivery gap no regression|

That is12 measured requests/start,36/HOLD, plus18 premeasurement greedy checks
and discarded OPS1 waves1–4. Public cuts, high reasoning, temperature1/top_p1,
priority10, seeds42/43, budgets, actual prompt counts and zero cache hits match.
Fresh salts differ across all requests/starts. No generated text is saved.

Mixed decoders emit before the long request is submitted; all three must be
admitted while both decoders remain alive. This proves arrival overlap, not that
512-token decoders span the entire150K prefill. p95 uses the complete decoder
stream's client delivery gaps, including zeros within speculative/SSE bursts;
it is not a GPU token clock. This screen makes no sustained mixed-throughput claim.
Threshold may be ignored for a lone eligible request by the current scheduler;
there is no claim that a configured threshold equals actual scheduled qlen.

For each matched repeat, the reference is the mean of the bracketing controls.
Either cold32K or cold150K B1 must reduce mean TTFT by strictly more than
`max(2% * reference TTFT, absolute difference of control-start mean TTFTs)`.
Both the point improvement and the paired95% lower bound must exceed that fixed
threshold. Every cell's per-stream TTFT, client decode rate and server pooled
committed-token proxy must avoid loss beyond their respective control spread;
both mixed decoders also gate p95. Per-repeat losses are checked separately.
Decode never supplies a winning objective. Two repeat groups permit exact
enumeration of four equiprobable paired bootstrap draws; the nearest-rank95%
endpoints are their min/max. This is a conditional screen on these prompts and
starts, not restart or task-family uncertainty. A pass says CANDIDATE FOR REVIEW.

Budget (`budget.json`): historical MEASURED startup388–404s, MEASURED NBW-W3
short/32K controls, and PF1-W1 cold150K controls with source hashes. Scale only
decode gaps511/1499; retain whole cold TTFT. Changed startup/greedy/host reads/mixed
are ESTIMATED. Start blocks are1214s OFF and1244s candidate. Estimated HOLD:
`120 + 1214 + 1244 + 1214 + 720 = 4512s = 75.20min`.
The120s setup reserve includes public cuts; startup/greedy/warmup/transition cost
is614/644s (404+90+60+60, plus30 candidate gate). These are allowances,
not measured PC1 durations. Admission reserves ALL remaining mandatory blocks.
The immutable clock is HOLD publication+73min submission/abort and+85min restore;
retries never renew it. The15-minute post-close watch is outside HOLD.

Every start verifies sources/full runtime, then six pinned W5/K1 greedy outputs
through the shared NBW first-divergence checker with explicit threshold
`0.9+1e-3`(0.901nats), policy **pc1-r3-near-tie-0p9**, registered by NDM's
2026-10-27 "You can loosen the nats to .9" instruction. Both tokens must be in
top20, within0.901nats and satisfy the same0.901nat top-choice-gap rule, else STOP.
Missing/prefix/length diagnosis still refuses. Evidence from each start must name
that exact policy and threshold. The shared NBW default stays0.501; W8 still
requires six exact IDENTICAL rows. No CPU-pin/poller exception
is needed. OPS1 waves count as owned demand up to4. Full counter baselines are
taken once after greedy/warmup; all subsequent snapshots retain sparse keys and
reject resets/growth of STOP/watchdog/failure counters.

Shared receiving-host cgroups, nested SSH/SCP ownership, atomic HOLD admission,
all-host fencing/reaping/absence, quiet proof and masked recovery signals follow
UP3. Each restore command has a serial exclusive log. The named experiment is
stopped and container/JIT/compute absence proved before unchanged W8. Exact full
restored identity and six distinct complete IDENTICAL0–5 rows are required;
near ties cannot satisfy W8. Deferred close is committed only after scopes are
sealed; ambiguity retains HOLD. Failed post-close watch invalidates the candidate.

CPU reproduction, from this reviewed worktree; set PC1_S to newly authorized
private scratch under the campaign and retain deny wrappers/network guard as
described in the lane report. No GPU, live preflight, HOLD or W8 is involved:

```bash
source ~/.config/ndm/paths.sh
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PC1_S/tmp" TMUX_TMPDIR="$PC1_S/t" XDG_CACHE_HOME="$PC1_S/cache"
export PYTHONPATH="$PC1_S/guard:$PWD:$PWD/offload/dspark"
export PATH="$PC1_S/bin:$PATH"
unset TMUX TMUX_PANE
/home/operator/miniconda3/bin/python3 -m offload.pc1.build
/home/operator/miniconda3/bin/python3 -m offload.pc1.window --dry-run --group chunk2048
/home/operator/miniconda3/bin/python3 -m pytest -q -p no:cacheprovider \
  offload/pc1/tests --basetemp="$PC1_S/tests"
/home/operator/miniconda3/bin/python3 -m mypy --strict --explicit-package-bases \
  --cache-dir "$PC1_S/mypy" offload/pc1/*.py offload/pc1/tests
bash -n offload/launch/serve_exp.sh offload/pc1/window.env
```

The separate finite read-only probe performs exactly inspect, exec-cat /serve.sh,
inspect on one named host,15s/read. It sends no port request, writes no host file,
imports no remote engine and cannot qualify experiment evidence. It refuses8001:

```bash
source ~/.config/ndm/paths.sh
PYTHONDONTWRITEBYTECODE=1 /home/operator/miniconda3/bin/python3 \
  -m offload.pc1.host_gates --probe-production --host sparkhead --rank 0
PYTHONDONTWRITEBYTECODE=1 /home/operator/miniconda3/bin/python3 \
  -m offload.pc1.host_gates --probe-production --host sparkworker-wired --rank 1
```

Future coordinator execution after independent source/CPU review and staged,
clean separate checkouts at the same reviewed commit on BOTH Sparks. Keep the
existing production Engram/source/JIT mounts and X86host workers intact; arrange
the existing D2 pause, public corpus and exclusive campaign ownership first.
Use a short fresh campaign directory name, no spaces or punctuation besides
underscores/hyphens. The runner rejects its local or remote reviewed-commit drift,
production pin drift and dirty tracked files before HOLD.

```bash
source ~/.config/ndm/paths.sh
# Set this explicitly to the reviewed integrated40-character commit, staged on BOTH Sparks.
: "${PC1_REVIEWED_COMMIT:?reviewed integrated commit required}"
export WINDOW_COORDINATOR=Deepseek/coordinator PYTHONDONTWRITEBYTECODE=1
/home/operator/miniconda3/bin/python3 -m offload.pc1.window --execute \
  --group chunk8192 --remote-repo dsv41-deploy-offload \
  --expected-experiment-commit "$PC1_REVIEWED_COMMIT" \
  --data-dir /opt/dsv41/data/bench-data \
  --out /opt/dsv41/campaigns/DSV41-OFFLOAD/PC1-chunk8192-W1
```

For another arm, use a separate reviewed invocation with its matching group and
fresh output name AFTER restoration/watch finishes. The lane did not execute
these commands or stage files remotely. A live timing/native numerical pass,
actual restore and post-close health evidence remain coordinator work.
