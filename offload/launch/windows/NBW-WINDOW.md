# NBW coordinator sheet — prepared, not run

Only Deepseek / coordinator runs these commands after review. NBW made no
Spark connection or serving request. Round 2 used X86host file-only searches
(`ls`/`find`/`sha256sum`) under the brief's narrow exception; no live state was changed. Do not use W8/native_ab.sh or its pre-K1 configs.
Read NBW-ARMS.md and NBW-PROMOTION.md first. This runner always restores K2;
it never promotes native. Failure to stage/start native is a STOP, not a reason
to change production's identity checks or permit compilation.

## OWN1 shared ownership revision (prepared, not executed)

Use the newly reviewed complete HEAD. NBW now uses `scoped_remote.py` for the
HOLD writer, native/pair operation, engine starts, pair restoration and all W8
phases, including nested SSH/SCP receiving commands. Every destination needs
cgroup v2, a reachable user systemd manager and `systemd-run --user --scope`.
The exact owner source travels inline. A permanent phase fence is written on
Sparkhead, sparkworker-wired, X86host and Controller before any scope is killed; every unit
and saved cgroup must then be absent/empty. Failed proof keeps HOLD.

The placeholder tmux server uses a private phase socket and empty config.
Only that server receives an unavailable private D-Bus address, which keeps
modern tmux panes from escaping into independent `tmux-spawn` scopes. The
owner and deadline guardian retain the working user bus. No shared tmux
server, systemd setting or production script is changed. Placeholder scopes
remain between measurement arms and are reaped before W8 like other prior
mutators. The HOLD file remains until final verification.

Recovery explicitly stops the named experiment containers, waits for zero
Spark compute apps, and inspects full Docker IDs, commands and configuration
for anonymous expert and remote-Engram JIT containers on both Sparks. Import
and build-lock waits count as JIT even before CUDA appears. Enumeration or
inspection failure refuses W8. Compilers finish within the existing bound;
they are never killed by cleanup. W8 itself stays unchanged.
Its final HOLD unlink is deferred until every phase on every host proves
absence again. Strict recovery still requires six distinct IDENTICAL checks;
measurement greedy near ties and failed-round growth gates are unchanged.
The original HOLD+90 ceiling remains immutable; the final 90 seconds are
reserved for commit/cancel/retention. Cancellation and restoring the original
HOLD after a lost reply remain available after mutation time expires.

After a successful close, the runner polls production for 15 minutes at
30-second intervals: models plus one greedy token at priority 20. Two
consecutive confirmed failures allow one new watchdog-purpose HOLD and scoped W8
start/check/close recovery. That new recovery has a separate immutable
30-minute admission bound; it never renews the measurement HOLD. Any failed
verification or repeat failure is loud and cannot trigger another recovery.
Active request gauges make an unadmitted probe busy/inconclusive and reset
the failure streak without spending the recovery allowance. Failed probes
require bounded idle/unreachable confirmation. Health is revalidated on the
receiving host immediately before stop; recovered/busy/unknown health refuses
stop and retains HOLD for coordinator review.
This adds post-close monitoring time, not measurement HOLD time.

Both callers use `hold_admission.py`: drain, then publish complete HOLD bytes
atomically and exclusively. An existing HOLD is never truncated. A private
per-caller nonce, saved inode and content digest prove admission, including
after a lost reply; a loser cannot adopt even an identically named winner.
The shared global admission lock covers the final ownership check through
the actual first stop. Production `window.sh` is unchanged and its old open
operation is no longer used by these callers. Stage/pin all three shared
helpers: `hold_admission.py`, `spark_quiet.py`, `production_health.py`.

## CPU plan and rehearsal on Controller

From the reviewed checkout (this branch, or the coordinator's reviewed integration):

```bash
set -euo pipefail
source ~/.config/ndm/paths.sh
NBW_REPO=$PWD
NBW_CAMPAIGN=/opt/dsv41/campaigns/DSV41-OFFLOAD
NBW_SCRATCH=$NBW_CAMPAIGN/scratch/NBW
mkdir -p "$NBW_SCRATCH/tmp" "$NBW_SCRATCH/tmux"
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export TMPDIR=$NBW_SCRATCH/tmp TMUX_TMPDIR=$NBW_SCRATCH/tmux
export PYTHONPATH=$NBW_REPO:$NBW_REPO/offload/dspark
unset TMUX
/home/operator/miniconda3/bin/python3 -m offload.launch.windows.nbw_window --dry-run
/home/operator/miniconda3/bin/python3 -m pytest -q -p no:cacheprovider \
  offload/launch/windows/tests/test_nbw_config.py \
  offload/launch/windows/tests/test_nbw_window.py \
  offload/launch/windows/tests/test_nbw_rehearsal.py \
  offload/launch/windows/tests/test_nbw_prerequisites.py \
  offload/launch/windows/tests/test_nbw_scoped.py \
  offload/launch/windows/tests/test_scoped_close.py \
  offload/launch/windows/tests/test_hold_admission.py \
  offload/launch/windows/tests/test_spark_quiet.py \
  offload/launch/windows/tests/test_s54_jit_scope.py \
  offload/launch/windows/tests/test_production_health.py \
  offload/launch/windows/tests/test_post_close_watch.py --basetemp="$TMPDIR/pytest"
/home/operator/miniconda3/bin/python3 -m mypy --strict --follow-imports=silent \
  --cache-dir="$TMPDIR/mypy" offload/launch/windows/nbw_window.py \
  offload/launch/windows/nbw_cells.py offload/launch/windows/nbw_prerequisites.py \
  offload/launch/windows/scoped_remote.py offload/launch/windows/post_close_watch.py \
  offload/launch/windows/hold_admission.py offload/launch/windows/spark_quiet.py \
  offload/launch/windows/production_health.py \
  offload/launch/windows/tests/test_nbw_config.py offload/launch/windows/tests/test_nbw_window.py \
  offload/launch/windows/tests/test_nbw_rehearsal.py offload/launch/windows/tests/test_nbw_prerequisites.py
```

Original MEASURED NBW baseline: 43 focused tests passed and strict mypy passed
five files. Round 2 adds the two S49 regressions and fourteen prerequisite tests;
strict mypy covers all seven NBW files.
House baseline before NBW changes: 153 pass, two historical PF1 cause-lineage
failures at 630f629. `NBW-baseline.log` preserves them; no assertion was weakened.

## Staging and solo prerequisite — coordinator only, BEFORE HOLD

The full review commit must be the final clean HEAD, including docs/tests. Freeze
it after review; no source edits during the window. The fixed NB1 item-2 source
commit is `dc6de03a8601acba039e972e69c8cee54560f4d7`. NB1's original scratch was
removed; do not assume it is still there. The following creates new separate
checkouts only. All commands are **written, not executed on real hosts by NBW**.
The bundle contains committed source only; no weights, caches or corpus.

```bash
set -euo pipefail
NBW_HEAD=$(git rev-parse HEAD)
NBW_REMOTE=dsv41-nbw-window
NBW_BUNDLE=nbw-reviewed-$NBW_HEAD.bundle
test -z "$(git status --porcelain --untracked-files=no)"
git bundle create "$NBW_SCRATCH/$NBW_BUNDLE" HEAD
scp "$NBW_SCRATCH/$NBW_BUNDLE" "sparkhead:$NBW_BUNDLE"
scp "$NBW_SCRATCH/$NBW_BUNDLE" "x86host:$NBW_BUNDLE"
ssh sparkhead "test ! -e ~/$NBW_REMOTE && git clone --no-checkout ~/$NBW_BUNDLE ~/$NBW_REMOTE && \
  cd ~/$NBW_REMOTE && git checkout --detach $NBW_HEAD && test \"\$(git rev-parse HEAD)\" = $NBW_HEAD"
/home/operator/miniconda3/bin/python3 -m offload.launch.windows.nbw_prerequisites \
  --emit stage > "$NBW_SCRATCH/stage-native.sh"
# Set the cache-alternative token ONLY after reviewing the Round 2 provenance section below.
ssh x86host "WINDOW_COORDINATOR=Deepseek/coordinator NBW_REVIEWED_CACHE_ALTERNATIVE=E-TP2-937e9633 \
  bash -s -- \"\$HOME/$NBW_BUNDLE\"" < "$NBW_SCRATCH/stage-native.sh"
```

The emitted stage script verifies all five S49/N6 prebuilt binary hashes and
both matching E/production tune cache hashes before creating anything. It
clones durable source at `dc6de03a8601acba039e972e69c8cee54560f4d7`, writes
SOURCE_COMMIT, and copies the alternative cache into its own
`~/dsv41-offload/nbw-source/exl3_tune_cache-reviewed.bin`. It records the cache
origin/qualification limit in PREREQUISITES.txt. Existing destinations, wrong
hashes, missing revisions or missing cache/binary prerequisites refuse. A
partially staged tree stays for inspection; it is never silently overwritten.
No production, unit, N4 cache or shared E file is changed.

Review host-local `nvidia-smi -L` and current allocation immediately before
selecting a solo GPU. **PP3 currently owns bc62df2a; do not start there until it
releases it.** There is no default GPU choice or automatic fallback. Frozen
dc6de03's `check_visible` permits only bc62df2a among X86host's solo cards. Passing
240876ad refuses during script generation: that card would require a separately
reviewed source guard change, which this package does not make. Never use the
production pair to bypass a busy solo. No power/clock change is prescribed.

The coordinator-selected supported solo command uses a nonblocking lock on
inherited FD 9 (the frozen source verifies the actual descriptor), verifies no
compute process exists on that UUID, and constructs the full TP2-emulated
native worker with graphs, level 2 and 1M RoPE. It uses the exact prebuilt-only
NBW bootstrap and private reviewed tune cache, exits without a listener or
request, and prints a fresh receipt. Missing Triton keys refuse without a build.
Run attended or in a coordinator-owned detached session.

```bash
set -euo pipefail
# Explicit coordinator selection after allocation/host-local UUID checks and PP3 release:
NBW_SOLO_GPU=GPU-bc62df2a-59cf-9828-7d43-558d1355bab7
/home/operator/miniconda3/bin/python3 -m offload.launch.windows.nbw_prerequisites \
  --emit solo --solo-gpu "$NBW_SOLO_GPU" > "$NBW_SCRATCH/solo-native.sh"
ssh x86host 'WINDOW_COORDINATOR=Deepseek/coordinator bash -s' \
  < "$NBW_SCRATCH/solo-native.sh" > "$NBW_CAMPAIGN/NBW-native-solo.log" 2>&1
# Require the SSH/startup exit status to be zero and review the COMPLETE log before this step.
NBW_SOLO_GPU="$NBW_SOLO_GPU" /home/operator/miniconda3/bin/python3 - <<'PYRECEIPT'
import json, os
from pathlib import Path
from offload.launch.windows.nbw_window import Native, TUNE_CACHE, TUNE_SHA256
from offload.launch.windows.nbw_prerequisites import NATIVE_COMMIT, NATIVE_TREE
root = Path('/opt/dsv41/campaigns/DSV41-OFFLOAD')
rows = [json.loads(s.removeprefix('NBW_SOLO_RECEIPT '))
        for s in (root/'NBW-native-solo.log').read_text().splitlines() if s.startswith('NBW_SOLO_RECEIPT ')]
assert len(rows) == 1 and rows[0]['solo_start'] == 'PASS'
r = rows[0]
assert r['batch_invariant'] == '2' and r['gpu_uuid'] == os.environ['NBW_SOLO_GPU']
assert r['native_tree'] == NATIVE_TREE and r['native_commit'] == NATIVE_COMMIT
assert r['tune_cache'] == TUNE_CACHE and r['tune_sha256'] == TUNE_SHA256
Native(r['native_tree'], r['native_commit'], r['native_identity'], r['native_markov_identity']).validate()
(root/'NBW-native-solo-reviewed.json').write_text(json.dumps(r, indent=2) + '\n')
PYRECEIPT
```

Solo success establishes only this prerequisite; it does not qualify physical
TP2 performance or resolve N4/N5 fidelity limits. An unsuccessful solo attempt
needs no HOLD. Record/remove only the coordinator's owned staging files after
use; retain reviewed source/cache receipts and do not remove shared N4 files.

## Attended serving window

Keep D2/local-law producers paused; one coordinator owns all endpoints. Ensure
no other window/HOLD exists. Public files `opinions_cl.jsonl` and
`filings_recap.jsonl` must already exist in `/opt/dsv41/data/bench-data`.
The six K1 references are read from `K1-production/greedy-post.jsonl`; native
outputs are compared in memory and only hashes/counts are saved.

```bash
set -euo pipefail
export WINDOW_COORDINATOR=Deepseek/coordinator
NBW_OUT=$NBW_CAMPAIGN/NBW-W-$(date -u +%Y%m%dT%H%M%SZ)
# The output must NOT already exist. Use the still-reviewed HEAD from staging.
/home/operator/miniconda3/bin/python3 - "$NBW_OUT" "$NBW_HEAD" "$NBW_REMOTE" <<'PY'
import json, sys
from pathlib import Path
from offload.launch.windows.nbw_window import main
r = json.loads(Path('/opt/dsv41/campaigns/DSV41-OFFLOAD/NBW-native-solo-reviewed.json').read_text())
raise SystemExit(main(['--execute', '--out', sys.argv[1], '--expected-experiment-commit', sys.argv[2],
 '--remote-repo', sys.argv[3], '--native-tree', r['native_tree'], '--native-commit', r['native_commit'],
 '--native-identity', r['native_identity'], '--native-markov-identity', r['native_markov_identity']]))
PY
```

Use a coordinator-owned tmux for the actual long window; the CPU tests above
intentionally unset TMUX and use private TMPDIR/TMUX_TMPDIR. Create STOP with
`touch "$NBW_OUT/STOP"` to trigger orderly recovery. Do not launch a second
restore while the supervised runner is restoring.

## Timing and reading

All serving times here are ESTIMATED, based on W8/N6 and PF1 K1 startup logs,
not NBW measurements. Per start: four cells once each, fixed 1500 delivered
output tokens, ignore_eos true, T=1/top_p=1, seed42, high reasoning, priority10,
fresh salts and public corpus. Actual 32K cuts are within 2% below 32000 and
are identical across starts; usage must show zero cached tokens. Short is a
1024-token public prompt, not a zero-context workload. No mixed/150K cells.

| Work inside HOLD | ESTIMATED minutes |
|---|---:|
| HOLD/setup, snapshot and one-time prompt cuts | 2 |
| K start, identity and four cells | 11.5 each ×2 =23 |
| I pair swap/prepack, Spark start, identity, six greedy checks and cells | 17.5 each ×2 =35 |
| Pair close, original K1 start, strict check and close (reserved) | 10–12 |
| Mandatory K I K I total including recovery | about70–72 |
| Optional closing K | +11.5, total about82–84 |

Before every start require its full conservative budget (K12 / I18 min) fits
before HOLD+72. The optional fifth K may skip; mandatory starts may not skip
silently. Abort at HOLD+75, bound every recovery command to finish by HOLD+90.
Unexpectedly slow startup/cells mean STOP and restore, never a relaxed deadline.
The pre-HOLD drain timeout is20 minutes and does not count toward HOLD age;
The shared admission operation publishes HOLD after drain. Preflight/solo/staging are outside HOLD.

The fixed verdict uses per-block median per-request decode tok/s, then medians
across starts. K spread=max−min of K block medians, no floor. I's short B1 and
32K B1 gains must be strictly greater than spread; B2/B3 may lose at most
spread. Require two complete starts per arm, fixed matching prompt hashes,
zero failed/some-failed rounds, every I greedy pass and successful restore.
No early verdict or promotion from a partial run.

Every cell saves per-stream and median TTFT/tok/s, acceptance per request-round,
server spec counters and server decode-duration sums. The reported ms/round is
1000×sum(request decode seconds)/sum(spec request-rounds). It includes terminal
and non-spec decode effects and is not a direct engine step wall timer. SSE
arrival times and proxy before/after counters are retained separately. Actual B,
raw confidence/chosen depth and exposed draft wait are not exported by unchanged
production; the summary says so rather than inventing them.

## Recovery

The runner masks subsequent signals, closes only its owned local processes,
fences/reaps its native launcher using the inherited KL1 ownership implementation
(unique NBW tags under kl1-control), and requires launcher absence before X86host
`prod_window.sh close`. Those persistent ABORT fences remain so late SSH arrivals
cannot launch after recovery. It then runs W8 start (if stopped), check, close,
counting exactly six distinct IDENTICAL indices0..5 and rejecting DIVERGES even
with rc0. It requires PRODUCTION HOLD OK, K1 START VERIFIED, CHECK: DONE and
CLOSE: window closed. No failed check may release HOLD.

Exit0 means complete and restored, not necessarily CANDIDATE. Exit1 means STOP
with restoration verified; exit2 means production is not verified and HOLD
remains. Inspect summary.json and cmd/restore-*.log. If manual recovery is needed,
first prove the owned native launcher fenced and absent and restore X86host's
production TP2 worker with `prod_window.sh close`. Only then:

```bash
set -euo pipefail
bash "$NBW_CAMPAIGN/W8/restore.sh" start
bash "$NBW_CAMPAIGN/W8/restore.sh" check
# Inspect THIS check output: six distinct IDENTICAL indices 0..5, no DIVERGES,
# PRODUCTION HOLD OK and CHECK: DONE. A zero return code alone is insufficient.
# Only after that strict gate:
bash "$NBW_CAMPAIGN/W8/restore.sh" close
```

Archive hashes and review the result before considering NBW-PROMOTION.md. Delete
only this task's scratch after tests/staging; keep measured artifacts and owned
remote ABORT fences. No push, merge, email or automatic production promotion.

## Round 2 — S49 pair-open fence (2026-10-26)

Do not run the original 0e6aafb runner. Pair-open now runs **inside** the same
registered, run-scoped launcher as native, before PAIR_OK validation. Registration
precedes any production worker stop. Recovery fences and proves that entire
pair-open/native process group absent before `prod_window.sh close`; permanent
ABORT also refuses a delayed registration and all retries after HOLD closes.
GPU locks remain nonblocking; ownership checks, signal masking, strict six-line
restore gate and HOLD+90 recovery deadline remain in force.

MEASURED CPU: the original 43 tests passed at 0e6aafb. Both added S49 regressions
failed there when independently owned pair-open sessions were released after
W8 close. With the fix, all 45 tests pass, including exact generated remote
shells, real pair dispatch with file-only GPU/unit fakes, before-registration
and active pair-open pauses, lost local replies, and post-restore retry refusal.
The existing delayed/active native-launch regressions are retained.

### Round 2 prerequisite search and reviewed alternative

**Original W8 tune cache bytes and recorded hash were not recovered.** Read-only
search covered campaign W8, N4/N5/N6 files and N6-final-results, local retained
N4/N5 data/manifests and W8-native results, and X86host's `~/dsv41-offload`,
`/opt/dsv41/x86host-data`, `~/.cache` and `~/.triton`. Only `ls`, `find` and
`sha256sum` were run on X86host. No process, GPU, unit or write command was run.
The configured `n4/results/exl3_tune_cache_w8.bin` is absent. References in the
old W8 launcher are paths, not byte/hash provenance.

MEASURED file-only: E runtime `~/dsv41-offload/e/run/exl3_tune_cache.bin` and
production `~/dsv41-offload/prod/run/exl3_tune_cache_tp2.bin` are each 1,712 bytes
and both SHA256:

```text
937e9633334be0a5a7a97557e83f962bfbc33dce84a76b93e331b8017b0d405e
```

The checked-in production staging recipe `worker/run/prod_tp2.sh:80–81` seeds
the production cache from E's cache; their current identical hashes support
that existing E/TP2 lineage. This **does not prove W8 or N4 tune-cache identity**.
The separately discovered default `~/.cache/exllamav3/autotune/coop_autotune_v1.bin`
has a different hash (`9d126f4bc3e2802d2c061643a94972f8d2fadecdc1405de9e16553e8fe3a5f3e`)
and is not substituted.

The smallest prepared alternative is a pinned private copy of the existing
E/TP2 bytes. The emitted stage script requires the explicit review token,
checks both originals and S49's five native binary hashes, then creates only
NBW's durable source/private cache/receipt. Runtime preflight and each native
start verify the private cache SHA256; no missing-cache fallback exists. The
solo command uses that same path/hash and the exact prebuilt-only bootstrap.
Nothing was actually staged or qualified here; the coordinator must review this
alternative before staging and obtain a fresh complete zero-exit solo receipt
before HOLD. A cache miss or changed hash means STOP outside HOLD; no rebuild
is authorized by this package. N4/N5 fidelity qualifications remain unchanged.

The prerequisite emitter never connects, starts a worker or writes a file;
it prints scripts for review. CPU tests execute them only inside private fake
X86host homes with synthetic source/cache/binary fixtures and fake GPU queries.
The real Git bundle, new-checkout/source receipt, hash refusals, repeated-stage
refusal, nonblocking FD9 inheritance, busy-lock refusal and GPU-query error/activity
refusal are exercised.
