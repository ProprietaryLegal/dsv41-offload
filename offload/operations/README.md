# Production profiles and operating ledger (OPS1, 2026-10-23)

The **recorded production profile is K1-327680 with TP2 + async**. This updates
the older roadmap/comparison snapshot that still describes one-card async.
It does not supersede the coordinator's campaign register or authorize a change.

`profiles.v1.json` is the versioned identity/configuration registry. It names
A, D, async, TP2+async, K1-327680, K1-524288 HOLD, native-source-faithful and
adapted separately, including config hashes, source refs/trees, drafter/Markov
identities, overlay manifests, evidence and rollback sections below. An unbuilt
adapter identity is null, never borrowed from the unadapted model. Historical
config files are references, not scripts to source over current production.
Native means the unadapted official-source backbone **with the declared
shared EXL3 K6 head**; it does not claim every tensor is official BF16.

## Read-only observation

From Controller (no GPU/import of torch, no Spark HTTP, no remote writes):

```bash
source ~/.config/ndm/paths.sh
CUDA_VISIBLE_DEVICES='' python3 offload/operations/live_profile.py --live
# Save aggregate output only under the campaign directory if a receipt is needed.
python3 offload/operations/live_profile.py --snapshot /path/to/saved-output.json
CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -s offload/operations -p test_live_profile.py -v
```

Only fixed `git log/status/diff`, `docker inspect/logs` on sparkhead and fixed
`systemctl --user status`, `readlink`, status-file `cat` on x86host are issued.
An inactive systemd unit's status code 3 is handled explicitly; other failed
reads fail. No credentials, arbitrary Docker environment entries, prompt/text
or tensor data are printed. Snapshot mode makes no remote call; its result is
as-of the saved observation, not a fresh health check. Exit 0 means one exact
**CONFIGURATION_MATCH**; exit 2 means DRIFT_OR_UNKNOWN (including live HOLD),
exit 1 means an observation/schema/read error.

Match includes image ID, startup cap/KV/sequence/chunk settings, expected env
and drafter/Markov identities, registered HEAD/dirty state/diff digest, worker
release, unit enablement/PID and fresh worker status, the complete K1 profile
mount set and K1 mount coverage/read-only/version.
The checker does not prove mounted file bytes, target manifest bytes, current
clock settings, GPU lock holders, adapter loading or the second Spark's state.
It never equates a healthy unit, a symbolic link or a container image label
with those checks. Observe both ranks with the existing coordinator runtime
verifier when a production qualification is needed. Separate SSH reads are
not atomic; a concurrent switch can yield a deliberate refusal. Rerun after
the coordinator's window closes. Failed/stale reads are never replaced by the
previous saved result.

Current registration intentionally expects the still-uncommitted K1 diff.
After the prepared commit/tag below, the checker must report drift until the
coordinator updates that profile's allowed HEAD, dirty state and diff digest,
reviewing the new observation. Future arbitrary commits do not inherit trust.
Native/adapted candidates have no automatic match rule until qualification.

## Closed operating ledger / evidence boundary

All numerical observations in this table are **MEASURED** in the cited records;
no new model requests or performance benchmarks were run by OPS1.

| Item | Disposition | Evidence / remaining gate |
|---|---|---|
| Async process isolation / recovery | Promoted; preserve | W5 in [COMPARISON](../results/COMPARISON.md); R3/S12 reports in campaign directory. No replay of solved fixes |
| TP2 production worker | Promoted 20:21Z; separately attributable to K1 | Campaign `LANES.md` 20:21 entry; W6-prod records; live unit active/current fb68c52 |
| K1 at 327680 | Promoted 21:19Z; source record not yet committed on sparkhead | Campaign K1-production start/post logs and S18/S22; live cap 327680, pool 2502600, bytes/rank 6442450944, seqs 4, batch 4096 |
| K1 at 524288 | HOLD | S18 final: cold prefill +35%; valid correctness does not waive speed gate. PF1-R2/coordinator own explanation and new qualification |
| Sparkhead source record/tag | Prepared, **not run** | HEAD ad2a4a0; only dsv41.env/start.sh dirty. Procedure below creates prod-k1-327-20261023 without restart |
| Native source-faithful arm | Not production qualified by N5; outcome qualification separately authorized | N4/N5; expert continuation rows 3/7. kx2 and packed identities recorded separately; stale DRAFT_MS must be calibrated |
| Adapted drafter | Not built/qualified | D2 pilot and serving receipts required; null identity is intentional |
| Old startup warm-up | Confirmed deficient; patch CPU-qualified, not applied | [Warm-up report](../launch/OPERATIONS.md), branch offload/ops1 item f4bc392; actual old EOS/reasoning/worker rounds unavailable |
| Concurrency 6/8 / clocks | Plans only | [Controlled plans](CONCURRENCY-CLOCKS.md); no setting changed |
| D2 fairness | Separate owner D2-FIX1 | Expert continuation row 5; legal priority unchanged. No scheduling change in OPS1 |

Live observation from OPS1 at 22:39Z: sparkhead container
`1ff67bf41da156b143abcbbc8071389d7d92652c5d593b74b792a72e143fbbdb`, started
20:46:08Z, running, image ID recorded in the registry. X86host TP2 worker pid
1339801, `current -> fb68c52`, one-card unit inactive, status time 22:39:08Z,
42,524 worker steps / 126,987 drafted requests / zero TP communication errors.
These are cumulative counters, not warm-up-only results or new performance
measurements. No lock inspection was authorized/performed by OPS1.

The container logs' long target-only interval was the **known native pair
window**, not a newly diagnosed failure: campaign `LANES.md` records opening
at 21:19:25Z, closing with PRODUCTION HOLD OK at 21:43Z. At 21:42:55Z the proxy
reconnected (epoch 2). Thus 29,624 cumulative failed rounds in the proxy logs
must not be reported as a new steady-state failure rate. Attributable worker
rounds for the old one-token warm-up still cannot be reconstructed.

## Prepared K1 source record (do not run from this lane)

Coordinator only, on sparkhead in `~/dsv41-deploy`. This records the already
applied files; **no start/stop or benchmark** is needed. Do not mix the warm-up
patch or any later changes into this commit. Preserve `logs/k1-promotion/`:
its saved original files, receipt/report and promoted hashes remain the guard
and rollback authority even after committing. Git commit/tag do not change
those file bytes. If any precheck differs, inspect; do not overwrite/restage.

```bash
source ~/.config/ndm/paths.sh
cd ~/dsv41-deploy
set -euo pipefail
test "$(git branch --show-current)" = main
test "$(git rev-parse HEAD)" = ad2a4a07f5da5d81d46a838882f1c02bc932e14b
git diff --cached --exit-code
python3 offload/promote/k1/manage.py verify --profile 327680
python3 - <<'PY'
import subprocess
expected = [' M dsv41.env', ' M start.sh']
actual = subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=normal'], text=True).splitlines()
if actual != expected:
    raise SystemExit('REFUSED: dirty paths differ: ' + repr(actual))
PY
git apply --reverse --check offload/promote/k1/dsv41.env.327680.patch offload/promote/k1/start.sh.patch
if git show-ref --verify --quiet refs/tags/prod-k1-327-20261023; then
  echo 'REFUSED: tag already exists' >&2
  exit 1
fi
git diff --check
git diff -- dsv41.env start.sh
# Coordinator reviews that exact diff against the saved promotion receipt.
git add -- dsv41.env start.sh
git commit -m 'production: record reviewed K1 packing at 327680 with TP2 async'
python3 offload/promote/k1/manage.py verify --profile 327680
git tag -a prod-k1-327-20261023 -m 'K1 327680; KV 2502600; async process proxy; x86host TP2 current fb68c52; S18/S22 qualified; 524288 HOLD'
git log -1 --format=fuller
git status --short
```

Record the resulting commit/tag object SHA in this registry and the campaign
ledger; refresh live observation and require only the expected clean-source
change. The tag pins Spark deployment sources, not x86host's service/link; those
identities must remain in the same receipt. Never push from this lane.

## Rollback commands (coordinator window only)

These are **prepared**, not executed or newly rehearsed by OPS1. Before any
rollback, pause producers/D2, drain both running/waiting gauges, use
`offload/launch/window.sh open 'reviewed profile rollback'` and preserve all
current diffs. Do not use a tag checkout to discard later edits. Check target
path/alias remain legal-v2a and retain exact local Engram fallback. Both nodes
must pass fresh runtime, request, semantics and hold checks before
`offload/launch/window.sh close`. An old tag is not proof of current health.

### K1

On sparkhead, after drain, rollback K1 to **the actual saved TP2+async state**:

```bash
timeout --kill-after=15s 45s bash ./stop.sh </dev/null
python3 offload/promote/k1/manage.py restore
timeout --kill-after=15s 600s bash ./start.sh </dev/null
```

The restore refuses later edits and verifies byte identity; keep HOLD on error.
Check old cap327680/pool1604926, process proxy, both-rank hashes, TP2 worker,
accepted drafts and greedy regression. Do not replace manage.py restore with
PROD_K1=0 or a tag checkout. See [K1 package](../promote/k1/README.md).

### TP2+async

K1 rollback above selects this profile, retaining x86host's TP2 unit/current
fb68c52. To roll the Spark configuration back from an unpromoted candidate,
use that candidate's saved environment/overlay transaction; no blanket checkout
of the experimental branch. [Async/TP2 package](../promote/async-tp2/README.md)
owns the separately reviewed worker switch and failure drills.

### async

After K1 restore, a separately reviewed worker rollback selects the one-card
worker. In a coordinator pair window on x86host (never OPS1):

```bash
systemctl --user disable --now dsv41-dspark-worker-tp2-prod.service
test -d "$HOME/dsv41-offload/prod/dspark-worker/d275f6b/offload/dspark/worker"
ln -sfn d275f6b "$HOME/dsv41-offload/prod/dspark-worker/current"
systemctl --user enable --now dsv41-dspark-worker.service
```

Before those commands, verify the staged d275f6b source/identity and preserve
the current link. `worker/run/prod_tp2.sh` documents this layout but deliberately
does not flip the link itself. The historical one-card unit has no ExecStart
GPU-flock wrapper: the coordinator must retain its one-card ownership/hold and
prevent lane admission through the whole handoff; do not assume this unit
provides the TP2 unit's two-lock guarantee. A later approved one-card runtime
gets a new registry entry. Check no pair consumer or waiting lane can race
the handoff. Existing [production window protocol](../launch/prod_window.sh)
remains authoritative; do not disable the unit outside its owner window.

### D

To remove async after restoring K1 and preserving the current diff, coordinator
on sparkhead selects the recorded synchronous D configuration files:

```bash
git restore --source=prod-d-20261023 -- dsv41.env start.sh serve.sh
./stop.sh
./start.sh
```

Use the reviewed one-card worker rollback above to match historical D's worker
identity. Current safety-hardened runtime code remains; this command restores
configuration files only. Confirm sync mode, KV1604926/max327680 and local-law
contract. Inspect the diff before restart; preserve any unrelated configuration
change rather than overwriting it blindly.

### A

After the same K1 restoration and diff preservation, the historical local
checkpoint drafter/local Engram configuration is:

```bash
git restore --source=baseline-20261022 -- dsv41.env start.sh serve.sh
./stop.sh
./start.sh
```

Require MODEL_DIR still `/srv/dsv41/legal-v2a`, cap327680, KV989460,
bytes/rank4026531840; verify target/source identities against the campaign
baseline before this older configuration is used. The unused x86host worker
can remain owned by its production unit; no lane releases its locks. Do not
load stock target weights to claim rollback.

### native/adapted

No native/adapted production rollback receipt exists yet. Each future candidate
must snapshot the qualified **K1-327680** target env/mounts, worker unit/current,
arm, drafter/Markov/source/adapter identities and existing GPU lock ownership
before switching. Its rollback uses the coordinator's `prod_window.sh` close
protocol and saved arm transaction, then verifies the registered K2 identities
and healthy process proxy. Until an actual candidate transaction exists,
there is deliberately no speculative command that overwrites worker links.

The registered worker max-model-len1048576 is a historical protocol upper bound,
not permission for >524288 target requests or new 1M experiments. The active
target cap is 327680 and K1-524288 remains HOLD.
