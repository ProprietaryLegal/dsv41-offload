# UP3 coordinator sheet — Round 8

**CPU-pin requires independent recheck of Round 8 before a window.** Its
experiment-only RDMA overlay now brackets native start, proves the sole new
CPU14 thread, renames it `exl3-rdma-prx` and writes a private container receipt.
The stable direct census requires receipt PID/TID/starttimes, current comm and
current affinity to agree. Every other thread, including every leader and the
rank-0 proxy, must have exactly the registered mask. Missing/lazy/unreadable,
ambiguous or mismatched evidence refuses before candidate timing and uses the
unchanged recovery path. No production image or control-arm bytes change.
See [RDMA-PIN.md](RDMA-PIN.md) for exact native/lazy call sites and tests.
K5 and dual-ring behavior is unchanged; no hardware qualification is claimed.

Round 4 supersedes the older all-arm resident-worker prerequisite. **Admissible
arms: k5, dual-ring; CPU-pin is prepared subject to the Round8 recheck above.
Parked before HOLD: split/0064 and
vor/0067**, until reviewed resident-worker exactness integration exists.
Only coordinator Deepseek / coordinator may execute after independent review.
No hardware correctness, performance gain or production promotion is claimed.
Production remains596ba1cf4e3b11be29b41a57e7f69a5d8efa4799 /
prod-k1-327-ops1-20261026.

## Round 4 — S59 fixes

FIX 1 follows the coordinator's rule27 decision: config-only arms use the
NBW/DEP1 experimental greedy correctness gate instead of the native adapter.
K5 changes proposal count under exact target verification; two-rank dual-ring
reduction sums the same two values; pinning moves threads. They change no
arithmetic implementation. Round8 adds only CPU-pin startup identity
instrumentation and its receipt; no resident adapter is installed. Code arms refuse at execute admission before commands,
output creation, preflight or HOLD; they cannot be unparked with a qualifier flag.

Dual ring also requires both exact rank initialization lines naming
rocep1s0f1,roceP2p1s0f1 and GID3, plus host sysfs IPv4 RoCEv2 GID3 on the26/27
fabrics. Missing second ring, wrong rank/GID or NCCL-only fallback STOPs before
timing. `host_gates.py` reads only existing container identity/logs and sysfs.
CPU pin now discovers the launcher, engine, worker and rank-0 proxy from exact
observed argv and parent relationships, independent of pin/proxy logs. It reads
every required thread's /proc status and stat using only Docker exec cat.
Two bounded namespace sweeps, complete Threads counts and stable TID starttimes
are required. An observed wrong mask or inaccessible, missing or unstable census
STOPs before timing. Optional pin/proxy logs can corroborate or contradict the
direct census, but their absence cannot replace or invalidate it. Round8
proves the RDMA TID via the startup receipt; other native/Engram TID-birth
attribution remains UNMEASURED. This
round does not claim native collective-byte/traffic or birth-attribution gates.
K5 retains the unchanged noncausal code/source checks and exact runtime schedule.

FIX 2 ports DEP1 9b72534: every OFF and candidate start performs six ordered
experimental W5/K1 greedy checks before OPS1 warmup or timing. The unchanged
NBW checker compares in-memory output to K1 references. At first token divergence,
both tokens must be in top20, within0.501nats, and within0.501nats of the top
choice; missing logprobs, prefix/length divergence or larger gap STOPs before
timing. Only hashes and diagnosis numbers are saved. The candidate decision
requires this evidence from all three starts. W8 still separately requires six
exact IDENTICAL rows, with no near-tie substitution.

FIX 3 ports DEP1's full post-greedy/post-OPS1 snapshot as one immutable baseline
per start. Every before/after cell snapshot and the final snapshot compares
against it. Failed/some-failed, watched STOP counters and watchdog fires may have
unchanged startup totals; measurement growth/reset fails. All observed counters
must remain present and nondecreasing, including sparse keys first seen after
the baseline. Initially absent zero counters are allowed. Healthy process-proxy,
capacity, mlock, torch-free and watchdog checks still bind every snapshot. No
live counter is cleared. Full baseline, final stats and observed keys are saved.

## Governing sequence, metric and budget (Rounds 3–4)

One arm per fresh HOLD: **OFF/arm/OFF**, all3 starts mandatory. Executable choices k5,dual-ring,cpu-pin; split/vor remain parked. No combined arm. OFF preserves current production's
complete container env, argv, engine/target/mount/Engram identities;8001 is the
experiment port. Candidate changes only its registered arm bytes/switches.
Review `ARMS.md` and each manifest. No replay, K1, image, unit or X86host worker change.

Each start has24 measured requests: short B1 four repeats,32K B1 four repeats,
32K B3 two groups (six requests),32K B4 one group (four requests), and mixed150K
plus two32K decoders twice (six requests). Every request forces512 tokens,
T1/top_p1/high/priority10; seeds42+repeat, matched public cuts, fresh cold salts,
zero cached tokens, exact prompt/completion usage and length finish required.
The mixed request must be admitted with both decoders alive (three running
requests); decoders must overlap its prefill but need not span the entire150K
prefill. This is an arrival/overlap screen, not sustained mixed throughput.
Missing corpus, per-token SSE usage or overlap STOPs; no filler replacement.

Decision uses the S57-cleared DEP1 arithmetic in `metrics.py`. Pool raw
request-round counts R, accepted-token counts A and server request-decode
milliseconds D, then compute(R+A)/D per cell/start. Retain A/R,D/R and raw
counters separately. Pair whole simultaneous request groups by repeat slot
across starts; deterministic2,000-draw percentile bootstrap gives a95% paired
interval. Point gain AND lower bound must strictly exceed max(2% of the mean
bracketing control rate, absolute C-to-C spread). Every cell's point and lower
bound must avoid loss beyond its C spread. TTFT per stream must avoid loss
beyond its own C spread; mixed per-stream p99 delivery gap has the same separate
no-regression gate. Client decode tok/s remains diagnostic.

Winning targets: split=mixed,pin/K5/dual=32K B3,vor=32K B1. B4 has only one
simultaneous group: its conditional interval has no within-start group variance,
and it cannot drive a win. The interval is conditional on fixed prompts/starts,
not task-family or restart uncertainty. Partial/STOP/unverified restore or failed
post-close watch cannot qualify. Passing means CANDIDATE FOR REVIEW only;
combine passing arms solely in a later reviewed window. No automatic promotion.

P99 measures client inter-token DELIVERY gaps. Continuous SSE usage supplies
actual counts; speculative tokens delivered together share a timestamp and have
zero within-burst gap. This is not an internal GPU clock. Invalid counters,
missing completed-request counts, cache hits, wrong seeds/budgets or incomplete
streams fail closed.

Budget evidence is `budget.json`: MEASURED historical startup388–404s and
NBW-W3 cold TTFT/decode spans; only decode scales511/1499 for512 outputs. B4,
mixed and qualification costs are explicitly ESTIMATED. Per-group seconds:
short20,32K B135,32K B380,B4105,mixed145. No cold-prefill scaling. Each start reserves60s for all greedy checks and any near-tie diagnostic
requests (ESTIMATED; DEP1 reserved42s). The greedy subprocess is capped at60s
and the original premeasurement deadline. Control block1389s; candidate1419s
includes30s host-evidence reserve. Total120s setup+2*1389+1419+720s
recovery=5037s=**83.95min ESTIMATED**.
The15-minute post-close watch is outside HOLD. Changed startup, greedy diagnostics and host evidence time remain unmeasured;
submission slack is63s. This is a bounded screen, not a completion guarantee.

Before any host access, an over85min plan refuses. Before every start, admission
reserves ALL remaining mandatory blocks. Premeasurement startup/greedy/warmup/host evidence has its own block allowance;
each read-only collector is capped at15s and remaining allowance. An overrun stops and restores, never drops a
mandatory control. HOLD publication+73min is the immutable submit/abort ceiling;
all restoration is bounded by HOLD+85. Neither drain, retry nor phase can renew
those clocks. An estimated fit is not a hardware completion guarantee.

## Shared ownership, recovery and post-close watch

`offload-x86host`f4eeecd merged into offload/up3 via ca252c9. There is no UP3
private registry or SSH/SCP shim. `scoped_remote.py` owns every receiving-host
mutation: HOLD placeholders, nested SSH/SCP, experiment stop/start, qualification,
OPS1 warmup and unchanged W8. Fence/reap/absence proofs cover sparkhead,
sparkworker-wired,x86host,localhost, with all destinations fenced before any kill.
Only exact owned cgroups are signalled. Shared `hold_admission.py` publishes
complete post-drain HOLD bytes under the global lock, with nonce/inode ownership
checked again through the first stop. Tags and immutable leases are remembered
before ambiguous submission; never delete their persistent fences.

Recovery masks INT/TERM/HUP including second signals. It fences prior scopes,
stops the named experiment through an owned scope, and uses`spark_quiet.py`
to prove containers, anonymous JIT and compute absent on both Sparks. Cleanup
failure retains HOLD. W8 start/check/close each have their own fenced scope.
Require full saved production identity and six complete distinct IDENTICAL
indices0..5, rc0 and W8 completion markers. W8 SHA remains
`d1c4479dbfbd90bc2036c7bf1561db5b564c2b162a7cde4a121e46bfb9f86a49`.

W8 close uses the shared deferred unlink. Seal every scope before commit-close;
a lost commit is cancelled and retains the original HOLD. Deadline exhaustion
is FAILED, never RESTORED. Bounded recovery reads remain available after STOP or
dead telemetry. UP3 never stops the X86host drafter pair.

After verified close, shared`production_health.py`/`post_close_watch.py` watch
production for15min with priority20 one-token greedy probes. Busy/unknown traffic
is inconclusive and cannot authorize a stop. At most one separately scoped
watchdog recovery is admitted after confirmed failures. Failed watch invalidates
a candidate. This watch also runs after a stopped experiment is safely restored.

## Parked native qualification package

Round2 adapters and their source lock remain unchanged as preparation, with
CPU tests retained. The host collector cannot create resident-worker receipts.
The unavailable native lifecycle is no longer invoked by config arms. Split and
VOR require reviewed integration on both real workers and exact native gates
before any future admission change. No integration is built in Round4.

## CPU validation (Controller only)

All scratch must be under the named UP3 directory and removed after checking
that owned test processes ended. Private tmux directory`t`keeps Unix sockets
under108bytes. The deny-only PATH protects real network/Docker/GPU operations.
The shared-owner tests use real local CPU cgroups and need only their own user
scope list/kill operations; no production unit is permitted.

```bash
source ~/.config/ndm/paths.sh
S=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/UP3
mkdir -p "$S/tmp" "$S/t" "$S/bin"
for tool in ssh scp curl nvidia-smi systemctl docker; do
  printf '#!/bin/sh\necho "CPU rehearsal denies host/GPU access" >&2\nexit 97\n' > "$S/bin/$tool"
  chmod 700 "$S/bin/$tool"
done
ln -sf /home/operator/miniconda3/bin/python3 "$S/bin/python3"
unset TMUX TMUX_PANE
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$S/tmp" TMUX_TMPDIR="$S/t" PATH="$S/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/offload/dspark"
/home/operator/miniconda3/bin/python3 -m offload.up3.build
/home/operator/miniconda3/bin/python3 -m offload.up3.qualification_lock
/home/operator/miniconda3/bin/python3 offload/up3/qualify.py --check-package
/home/operator/miniconda3/bin/python3 -m offload.launch.windows.up3_window --dry-run
/home/operator/miniconda3/bin/python3 -m pytest -q -p no:cacheprovider \
 offload/up3/tests offload/upstream_adopt/tests \
 offload/launch/windows/tests/test_kl1_window.py \
 offload/launch/windows/tests/test_nbw_window.py \
 offload/launch/windows/tests/test_spark_quiet.py \
 offload/launch/windows/tests/test_production_health.py \
 offload/launch/windows/tests/test_post_close_watch.py --basetemp="$S/tests"
/home/operator/miniconda3/bin/python3 -m mypy --strict --explicit-package-bases \
 --cache-dir "$S/mypy" offload/up3/*.py offload/up3/tests \
 offload/launch/windows/up3_window.py
bash -n offload/up3/window.env
git diff --check
```

To run the shared real-cgroup suite in the same CPU environment, replace ONLY
its systemctl wrapper with this owned-fixture restriction:

```bash
cat > "$S/bin/systemctl" <<'SH'
#!/bin/bash
if [ "$1" = --user ]; then
  case "$2" in
    show-environment) exec /usr/bin/systemctl "$@" ;;
    list-units|kill)
      last=${!#}
      case "$last" in dsv41w-own1test-*) exec /usr/bin/systemctl "$@" ;; esac ;;
  esac
fi
echo "refuses systemctl outside owned CPU fixture scopes" >&2
exit 97
SH
chmod 700 "$S/bin/systemctl"
/home/operator/miniconda3/bin/python3 -m pytest -q -p no:cacheprovider \
 offload/launch/windows/tests/test_scoped_remote.py \
 offload/launch/windows/tests/test_scoped_close.py \
 offload/launch/windows/tests/test_hold_admission.py \
 offload/launch/windows/tests/test_spark_quiet.py \
 offload/launch/windows/tests/test_s54_jit_scope.py \
 offload/launch/windows/tests/test_production_health.py \
 offload/launch/windows/tests/test_post_close_watch.py --basetemp="$S/shared-tests"
```

## Conditional staging and attended execution (coordinator only)

Obtain independent review of the final applicable commit first (Round8 for
CPU-pin). Choose exactly one arm. Split/VOR remain parked; no qualifier flag bypasses admission.

Stage the exact reviewed commit in the separate clean experiment checkout on
BOTH Sparks at`~/dsv41-deploy-offload`using the coordinator's reviewed procedure.
No implicit fetch/install or real-host precheck mode exists. Verify source pins,
production tag/scripts/env, model, worker, host memory/kernel/driver and fresh
per-run control roots. No production checkout, K1 package, image, unit, W8 or
X86host pair change is part of this lane. Use a normal coordinator shell with no
CPU denial wrappers.

```bash
source ~/.config/ndm/paths.sh
: "${UP3_REVIEWED_COMMIT:?full independently reviewed commit required}"
export WINDOW_COORDINATOR=Deepseek/coordinator
UP3_GROUP=k5
UP3_OUT=/opt/dsv41/campaigns/DSV41-OFFLOAD/UP3-k5-W-$(date -u +%Y%m%dT%H%M%SZ)
/home/operator/miniconda3/bin/python3 -m offload.launch.windows.up3_window \
 --execute --group "$UP3_GROUP" --out "$UP3_OUT" \
 --remote-repo dsv41-deploy-offload \
 --expected-experiment-commit "$UP3_REVIEWED_COMMIT"
```

Use a fresh group/output for each HOLD; never run groups concurrently. To stop,
create`$UP3_OUT/STOP`and remain attended through restoration/watch. Failed restore
exit2 leaves HOLD and requires review of summary.json and cmd/restore-*.log.
Prove all previous mutators absent before manual W8 recovery. Do not clear
fences, renew HOLD, close after failed cleanup or bypass six IDENTICAL checks.

## Round 6 — hardware-faithful host gates

The host gate reads the container's actual `/serve.sh`, joins shell continuations,
and requires exactly one literal `--port 8001` on its `exec vllm serve` command.
Docker `Config.Cmd` is the real `['/serve.sh']` wrapper with entrypoint
`['/bin/bash']`; it does not carry vLLM flags. Production port 8000 refuses
experiment qualification. The captured rank-0 production launcher fixture contains
only launch configuration, with no prompt, weights or generated text.

The complete remote parser audit and captured/source-rendered fixture provenance
are in [HOST-PARSE-AUDIT.md](HOST-PARSE-AUDIT.md). Full source log bodies now
replace abbreviated RDMA fixtures. Logging prefixes are allowed; partial or
malformed initialization messages refuse. Both required sysfs device names,
including case, were confirmed on both hosts. Production's proxy module is a
verified bind mount absent from the src-image mirror.

Read-only production plumbing probe (no HOLD, experiment start, port request or
engine import). It now includes the direct process/thread census. From this
reviewed checkout on Controller:

```bash
source ~/.config/ndm/paths.sh
PYTHONDONTWRITEBYTECODE=1 /home/operator/miniconda3/bin/python3 -m offload.up3.host_gates \
 --probe-production --host sparkhead --rank 0
PYTHONDONTWRITEBYTECODE=1 /home/operator/miniconda3/bin/python3 -m offload.up3.host_gates \
 --probe-production --host sparkworker-wired --rank 1
```

On a Spark with the reviewed module already available, omit `--host` and supply
that Spark's rank. This lane used the Controller commands, without staging or writing
anything remotely. The command allowlist contains only Docker inspect/logs,
Docker exec cat /serve.sh and the fixed installed rdma.py path, bounded /proc status/stat/cmdline reads, and cat of
GID3/type/netdev on the two registered f1 devices. No remote Python, ls, shell
loop or affinity write runs. Every subprocess has a 15-second timeout. The
namespace ceiling is 16,384, with at most 512 paths per cat; larger namespaces
refuse. The full inspected container
ID is checked again after collection. Only the two relevant EXL3 config fields
are returned; arbitrary environment values are not printed.

Exit 0 / `PARSE COMPLETE` means the plumbing parsed; the output explicitly says
`NOT QUALIFIED`. Production's single-ring line can be present; the dual-ring
and CPU-pin arm evidence is expected absent. Rank-1 has no rank-0 proxy. GID field
failures retain their device/error and exit nonzero. A missing/unreadable census
is a PARSE FAILURE. Default production masks are reported without requiring the
experimental pin. The result remains NOT QUALIFIED and cannot be combined with experiment --arm or
--container arguments. Actual arm qualification still uses the experiment gate.

The earlier Round 6 probes completed on 2026-10-27 with stable container IDs, correct rank and
port 8000, real single-ring initialization, the rank-0 proxy, no pin-init, and
valid GID3/type/netdev on both f1 devices. No correctness/performance or live
restore result follows from this observation.

A final repeat encountered replacement containers, then refused on **both** hosts
because their mounted launchers used experiment port 8001:
`STOP: production probe requires production8000; experiment qualification uses --arm`.
The initial read-only production probes above succeeded before that external
transition. The refusal is expected isolation, not an arm result; this lane did
not change containers or wait for/interrupt the other window.

## Round 7 — historical direct census, no lost-log dependency

The implementation and source-grounded remaining blocker are documented in
[HOST-PARSE-AUDIT.md](HOST-PARSE-AUDIT.md#round-7-superseding-cpu-pin-census-and-remaining-blocker).
The actual first-registration regression executes the pinned plugin source with
only model imports and OS affinity operations mocked; its pin-init log is empty.
The actual collector/window regressions use that empty-log condition and the
captured process shapes. Complete exact-mask evidence passes in CPU fixtures;
missing/unreadable workers and a shared-comm CPU14 thread stop before candidate
timing and perform fake W8 restoration. These are CPU control-flow checks, not
hardware qualification.

The production probe commands above now collect and display the full direct
census, while retaining the production8000 admission requirement and NOT QUALIFIED
label. Round7 attempts refused the coordinator-managed8001 containers. Later
read-only attempts observed container replacement/removal; no successful Round7
production8000 probe was claimed at that checkpoint. Final bounded retries then
passed on BOTH hosts at production8000 with complete direct censuses and stable
container IDs; see the campaign report for exact receipts. Do not change/restart
a container to make a probe pass.

CPU-pin requires a reviewed way to distinguish the native poller from other
threads sharing its comm before a real window. No native name setter exists in
the checked source (`exl3_rdma.cu` SHA256
`61754d3f062a4da82066dc9908ab7a5766119e6459e821e78004b3d3068d5ed8`).
A name-only exemption for `VLLM::Worker_TP`, or permitting all nonleaders on14,
would not satisfy that requirement. This round does neither and does not change
the image. No change to the decision, budget, arithmetic arms or shared lifecycle.

## Round 8 — startup identity receipt

CPU-pin alone mounts the two pinned files listed in its manifest through
`window.env` and the existing read-only overlay machinery. Control mounts no
replacement source. The native extension and arithmetic/transport paths are
unchanged. The helper snapshots tasks around native start, waits at most2s for
its self-pin, requires one new TID, renames/rechecks it and exclusively writes
and fsyncs `/run/up3-rdma-pin/PID-STARTTIME.json`. This path belongs to the
fresh experiment container's private writable layer; it is never host-mounted
or reused across starts. The runner already requires a fresh container ID.

`ring_for` remains lazy. No extra request, model import or collective forces
setup. If greedy/OPS1 never initialize it, the absent receipt refuses before
timing. Native startup/identity exceptions rethrow instead of falling back.
The host reads the exact worker receipt with Docker exec cat between the two
existing stable process/thread sweeps. Receipt decoding rejects duplicate keys.
Its PID/TID/starttimes, unique comm and `[14]` affinity must match the current
census. Name-only and affinity-only exemptions, a reused TID, extra named
threads and all other wrong masks refuse. The leader's exact mask remains
mandatory. Suppressed pin-init logs remain optional corroboration.

CPU checks include a real Linux thread rename/receipt subprocess and actual
runner -> collector receipt regressions on each rank. They prove control flow,
not native GB10 operation or performance. The mandatory six greedy checks,
baseline/counter rules, timing decision,85-minute budget and all shared
ownership/restoration/watch paths are unchanged. Coordinator commands above
remain the exact entry points after independent recheck; choose
`UP3_GROUP=cpu-pin` and a fresh matching output directory for this arm only.


Round8 production-probe extension adds exactly one fixed source read:
`docker exec dsv41-serve cat /usr/local/lib/python3.12/dist-packages/vllm_exl3_moe/rdma.py`.
The source must equal the pinned original image hash, and inspect must show no
mount covering rdma.py, its identity helper or /run/up3-rdma-pin. The output
labels `startup_overlay` ABSENT and records both hashes. It does not read or
write receipts. The production8000 requirement, full direct census, final
container-ID check and NOT QUALIFIED label remain. Both final Round8 probes
passed on production8000 with the overlay absent; see the campaign report.

## WIN2 — chain mode and the A/B/C chain (R-UP3, 2026-10-27)

Chain mode (`WINDOW_CHAIN=1`, coordinator only; `WINDOW_CHAIN_LAST=1` on the last window) works for every group. A
read-only probe (HOLD file, `:8000/health`) feeds `chain.preflight`. In `'chained'` mode, production is down by design.
The window then does four things:

- loads a sha-pinned UP3 production identity: the marker's `quiet_proof.production_identity`, or
  `--production-identity <UP3 rollback.json of production 596ba1c>` after a non-UP3 window;
- proves both Sparks quiet before any HOLD;
- admits its HOLD with `allow_unreachable`;
- always restarts production through W8 when it has to restore.

A clean non-last window fences everything, stops the experiment, proves quiet, and removes its own HOLD under
hold_admission's ownership lock. It then writes the marker and sets summary `chain` to `CHAINED`. A stop before any
HOLD while the marker exists reports `FAILED: CHAIN STOPPED: PRODUCTION DOWN`, which needs the coordinator's chain
abort.

WIN2 groups (`win2_plan.py`; launch with `--group`):

| Group | Starts | Decision cells (512 tokens, repeats per start) | Reported | MEASURED-basis blocks | ESTIMATED HOLD |
|---|---|---|---|---|---|
| `ring-depth2` | C / dual-ring / fixed2 / C | short-b1 ×6, 32k-b3 ×4 | mixed150k-2decode ×1 | 981/981/1011/981 s | 79.90 min |
| `defer-shares` | C / defer (0051) / ring-defer / C | short-b1 ×6, 32k-b3 ×4 | mixed150k-2decode ×1 | 981 s ×4 | 79.40 min |
| `k5-batch` | C / k5 / C | 32k-b3 ×5, 32k-b4 ×5 (k 4→5 only at 3–4 requests) | none | 1256 s ×3 | 76.80 min |

- **Arms.** Every arm is config-only (`build.ENV`, `window.env`, manifests with no files). fixed2 is DEP1's
  `VLLM_DSPARK_DEPTH_POLICY=fixed:2 VLLM_DSV41_ADAPTIVE_K=0` plus the K5 schedule. defer is
  `VLLM_DSV41_INDEXER_DEFER=1`, which production's K1 attention.py carries but sets nowhere. ring-defer is both.
- **Host evidence.** ring-defer runs dual-ring's host gate. k5, fixed2 and defer have none.
- **Greedy near-tie limits.** 0.9+1e-3 nats, except fixed2 at 1.1+1e-3 (NDM 2026-10-27). They are passed to
  `offload.up3.cells --near-tie-nats`, and the result must carry the limit.
- **Decision.** win2-v1 per candidate arm against both bracketing control starts, unpaired.
  - dual ring, defer and ring-defer: pooled ms/round decides and accepted/round guards.
  - fixed2 and k5 change how many tokens a round verifies, so pooled committed tokens per decode second, (R+A)/D,
    decides. Their ms/round and accepted/round are reported.
- **Clock cap.** Every start records the cap that `start_exp.sh` printed. A window whose starts disagree, or that has
  none, is NOT A CANDIDATE.
- **Archive.** Control repeats go to `control-archive.jsonl` with commit, the start's own `clock_cap_mhz`,
  `max_tokens` 512, start, repeat and date. Spreads never pool other commits, clocks or token budgets.
- **A/A calibration.** At one request, K5 and production both run k = 5, so UP3-k5-W-20261027T063155Z's B1 cells
  were an A/A test.

## R-UPF — 0091 and the fixed top-k as single windows (R-UP3, 2026-10-27)

`upf_plan.py` runs R-UPF's designs (`offload/upfix` 42fe86a: `windows.py`, `configs.py`, `WINDOW.md`). It imports
them and copies nothing. Each arm is `offload/upfix/UPF.env` plus that arm's `configs.ARMS` environment.

Coordinator decisions (2026-10-27):
- UPF-W1 is refused before any host access. NDM's standing ceiling is production restored by HOLD+132, and UPF-W1
  would need HOLD+176.
- UPF runs as two SINGLE windows, `upf-topk` first and then `upf-0091`, off-hours only. There is no chain until WIN2-R3.
- Each group gets HOLD bounds sized from its own budget: submit/abort = ceil(1.15 × the design's in-HOLD estimate),
  and restore is 12 minutes later.

| Group | Pre-steps | Starts | Blocks | ESTIMATED HOLD | HOLD bounds |
|---|---|---|---|---|---|
| `upf-topk` | T0 stop + GPU idle, T1 SM121 build, T2 bit-compare | C1 / X-topk / C2 | 1251/1971/1251 s | 105.55 min | submit +108, restore +120 |
| `upf-0091` | none | C1 / X-0091 / C2 (soak in C1 and X) | 1851/2331/1251 s | 107.55 min | submit +110, restore +122 |

- **T0–T2.** These run after the HOLD and the production snapshot, before C1. All three are in the experiment
  checkout under the HOLD's `stop` admission, using WINDOW.md's commands.
  - T1 and T2 each run in `timeout -k 30 600`. The containers are named `upfix-t1` and `upfix-t2`, and either is
    removed on 124 or 137.
  - T1 checks that the image id is `pf1_window.IMAGE`.
  - After T2 the runner reads both receipts and runs `configs verify-topk`. The library's sha256 is then pinned among
    X-topk's installed bytes.
  - If T1 or T2 fails, crashes or times out, or verify-topk refuses, only X-topk is skipped. If T0 fails, the window
    stops.
- **Image bases.** In production mode, before T0, every UPF overlay's base file must equal its manifest
  `base_sha256` (S93 runner note). The production container shows the image bytes because K1 mounts none of these
  paths.
- **Identity relative to C1.**
  - C1 against production: the runner records a diff of env, mounts and argv. Image, engine, model target files and
    Engram identity must be equal.
  - C1's greedy must be 6/6 IDENTICAL to the K1 reference, or the window STOPs.
  - Every later start must equal C1 plus exactly its arm's dist-packages overlays (sha of the reviewed source) and
    `TOPK_ENV`.
- **Gates (registration upf-0p9).**
  - greedy: `offload.up3.upf_driver greedy`, at 0.9+1e-3 nats with distributions recorded.
  - needles: 150K, plus 300K for topk, run in qualify.
  - soak: 600 s with four streams, after the cells. RSS per process role is sampled on both Sparks, plus `/dev/shm`,
    and judged with `windows.soak_ok`.
  - C2's greedy step is the KL re-probe of each candidate's recorded divergences, at 0.005. C1 is 6/6 IDENTICAL by
    gate, so there are no A/A rows.
  - The OPS1 warm-up stands in for the catalogue's 64-token warm-up cell.
- **Arm-level stops.** A candidate's failed gate records `STOPPED` with the gate name and the window goes on. A
  control's failure, or any infrastructure failure, stops the window and runs the strict W8 restore.
- **Cells.** `ctl-150k-b1` and `ctl-32k-b4` run through `offload.up3.cells`. They use the shared catalogue's prompts
  and bounds at 512 tokens per request, with the design's repeats: C 2, X 3.
- **Verdict.** `windows.arm_verdict` over `decide_window`, rule win2-v1 revision 2, with C1 and C2 pooled against X,
  unpaired. The outcomes are PROMOTABLE (no loss), NOT A CANDIDATE, STOP: <gate> or SKIPPED: <reason>. A decision
  whose `rule_revision` differs from `windows.RULE_REVISION` is reported UNJUDGED.
- **Archive.** Every COMPLETE C1/C2 repeat goes to the archive as `windows.archive_rows`: 512 tokens, the production
  commit, the start's clock cap from its start log, and the date.
  - `main()` appends these rows after `summary.json` exists, because `win2_history.production_commit` reads it.
  - Spreads are reported under the `ctl-` names and the old `150k-b1` / `32k-b4` names.

Launch: `scratch/RUP3/launch/UP3W2-RUN.sh upf-topk <R> <PREV_TMUX> <PREV_FILE|-> single`, then the same for
`upf-0091`.
