# Round 11 X86host-only server diagnostic — written, not run

Ready for independent source/CPU review. The lane has made no host contact,
GPU call, production mutation, staging operation or serving request this round.
This diagnostic cannot qualify Spark admission, performance or promotion.
The existing S78/S76 greedy-generation salt blocker is still open.

`server_probe` runs ON X86host, from a clean independently reviewed candidate
checkout. It contains no SSH, Spark API, HOLD or engine-launch operation.
Production's target keeps serving while `prod_window.sh pair COORD` temporarily
releases its drafter pair; the diagnostic listener uses an allocated port
18640–18699 on X86host's own 10.202.28.1. It never uses production's 18620.
The local client uses real OFL TCP callbacks; RDMA is outside this diagnostic.

## Inputs and identity

Stage two separate clean local checkouts under /opt/dsv41/x86host-data/pp4:

- Baseline: full `1b8122992fb440edc668ee9e628ee9624ae2faf2`.
- Candidate: full final independently reviewed Round 11 commit, containing the
  driver. It must contain exactly the graph override deletion and graph-mode
  readiness log in server.py; every other non-test worker Python/CUDA/C++ header
  and implementation byte must equal the baseline. The driver enforces this.

No staging is performed by the driver. Transfer a reviewed bundle to X86host
only and make these checkouts there; do not use pp4_admission_stage.sh, which
also stages a Spark. No source tree or shared runtime is changed during a run.
Delete staging clones after the coordinator has retained the reviewed evidence.

Supply the preserved raw physical-pair driver `receipt.json` from the baseline
1b81229 qualification, not an admission summary or solo receipt. Its exact
matrix, physical UUIDs, both-rank q/ID replay, isolated timeout/sticky teardown,
whole-K2/N4 checks, commit and original source hashes are validated. These
unchanged numerical sources remain the reference qualification; that receipt
DID NOT qualify the service builder. It is not relabelled or rehashed. This
avoids requiring another Spark HOLD merely to diagnose the builder mismatch.

The actual new graph reference is produced and preserved by a separate owned
candidate process before baseline and candidate run. All three processes use
identical reviewed runtime/checkpoint bytes. Candidate output signatures must
match the separately preserved reference exactly. Both variants' server.py
hashes, all worker Python source hashes, driver/bootstrap/frame/trace hashes,
runtime receipt SHA256 and physical-receipt SHA256 are recorded in identity.json.
Before imports the child rechecks its own run, staged directory, worker bytes,
qualified runtime manifest and inherited GPU lock inodes. Loaded extension and
OFL module paths/hashes must match the reviewed inputs.

The required miniconda interpreter is /home/operator/miniconda3/bin/python3;
Torch must be 2.12.1+cu130. The existing checkpoint is read and checked by the
real production builder at /opt/dsv41/x86host-data/dspark. No model download,
conversion or private capture is needed.

Prepare this local runtime manifest in PP4 scratch, with actual absolute paths
and SHA256 values, after independently reviewing the files. `files` must have
EXACTLY these six labels:

```json
{
  "schema": "pp4-server-probe-runtime-v1",
  "reviewed": true,
  "python": "/home/operator/miniconda3/bin/python3",
  "torch": "2.12.1+cu130",
  "staging_receipt": "ABSOLUTE_REVIEWED_STAGING_RECEIPT",
  "staging_sha256": "ACTUAL_RECEIPT_SHA256",
  "files": {
    "stock": {"path": "ABSOLUTE_STOCK_CPYTHON313_SO", "sha256": "ACTUAL_SHA256"},
    "facc": {"path": "ABSOLUTE_FACC_CPYTHON313_SO", "sha256": "ACTUAL_SHA256"},
    "p2p": {"path": "ABSOLUTE_P2P_ROOT/SOURCE_KEY/dspark_p2p.so", "sha256": "ACTUAL_SHA256"},
    "ofl": {"path": "ABSOLUTE_LIBOFL_SO", "sha256": "ACTUAL_SHA256"},
    "ofl_python": {"path": "ABSOLUTE_OFL_PY", "sha256": "ACTUAL_SHA256"},
    "tune": {"path": "ABSOLUTE_REVIEWED_TUNE_BIN", "sha256": "ACTUAL_SHA256"}
  }
}
```

The staging receipt must be REVIEWED for the candidate commit, with the existing
three-entry CPython3.13 `runtime_artifact_sha256`, runtime_python and
runtime_torch fields. Stock/facc/P2P hashes must equal it. The P2P source-key
folder and sibling dspark_p2p.so.sha256 must already exist; the actual loader
refuses missing or wrong bytes rather than compiling P2P. The OFL Python module
and shared library are separately pinned here. These are local consistency
and review gates; the driver does not authenticate invented review receipts.

Each process gets fresh private empty Triton, CUDA, Torch-extension, TorchInductor,
XDG, HF and temporary cache directories. The tune file starts from a private
byte-identical COPY; shared tune bytes are never written. Relevant resolved
CUDA/Triton/allocator/runtime settings and library versions are retained.
Inherited DSPARK/TRITON/CUDA/EXLLAMAV3 feature settings are removed before the
fixed experiment settings are installed; CUDA module loading uses the runtime
default, recorded as unset if not explicitly supplied. Cold here means empty
process-private compiler/driver caches, not evicted OS page cache. Warm means
repetition within that same process. No page-cache drop is attempted.

## Actual execution and evidence

The bootstrap arms a repeating ten-second native Python traceback timer and
SIGUSR1 stack dump before model imports. It executes the ACTUAL staged
`worker.server` as `__main__`, with `--tp 2`, sampled switch ON and the caller's
default graphs. A one-shot Python profile hook installs diagnostic host wrappers
at that module's real main entry, before its actual build_core_tp2 call.
No model or kernel implementation is copied into the probe.

The order inside each owned process is:

1. Real builder/loading/default capture. Resolved backend.graphs and the real
   readiness log are preserved. Baseline should show False; candidate/reference
   must show True. Startup is bounded at 600 seconds, the existing whole-worker
   qualification bound; this is not a measured new graph-startup duration.
2. From one ordinary Python thread, real encoded HELLO/STEP/RESET/PING frames
   enter actual DraftCore.handle. After HELLO, a labelled window-only setup
   STEP inserts 128 rows for each of four generation1 slots, with no draft
   flag and zero masks. It has the same 30-second frame bound. The FIRST
   positive draft is B1/two rows, IMMEDIATELY followed by B4/28 rows (seven per
   request), in the same session with no RESET, resync, other width or draft
   between them. Slot0 advances128->130->137; slots1..3 advance128->135.
   All windows remain full and generations unchanged. Only AFTER that pair,
   the existing reset/repetition matrix runs: B1/B4/B2/B3, ten changed-anchor/
   round/key iterations per width and mask, all positive/all T0/mixed. These
   warm resyncs still send full128-row windows; subsequent increments are
   two/seven/five rows per request at B1/B4/B2–3. A new session repeats setup,
   cold transition and matrix. Service depth is HELLO-negotiated fixed5.
   Synthetic finite BF16 taps are
   deterministic by slot/position and match regardless of insertion chunks.
   These are observed public shapes with synthetic values, not recorded private
   tensors and not a claim to reproduce an unknown full public request.
3. Actual model APIs first run consecutive positive B1/B4 calls at output
   depth4, then consecutive positive B1/B4 at depth5, without an intervening
   model draft or reset within either pair. The API consumes preloaded KV;
   row-delta shapes are exercised by the protocol steps above. Afterwards the
   existing B1/B4/B2/B3 matrix runs at output depths4/5, all three masks and ten
   changed keys/anchors using the deterministic synthetic N4 fixtures.
   T0 IDs and confidence must exactly equal the original greedy model path.
4. The real server main starts real OflServer. A separate X86host-local CPU client
   sends the same frames through its native connection-thread callbacks, closes
   and reconnects a new connection/session. Direct/OFL ID/confidence/q signatures
   must match exactly. OFL runs after direct/model setup, so this is explicitly
   warm model traffic with fresh native connections, not a cold OFL-start claim.

Each STEP checks frame identity/status, complete windows, fixed depth,
CRC/support/mask, exact dyadic24 mass and sampled-token support using the
production codec. Every frame must finish inside the existing 30-second status
freshness window; there is no cold-request exemption or timeout extension.
Server and client have independent flushed per-phase/request records and
Python/native thread IDs. Builder/capture, reset/stage/flush, whole forward,
selector warmup/load, each rank submission, result checks and Python/native
reply serialization are timed with perf_counter_ns. Logs contain hashes,
counts and timing only, without text or tensor payloads.

Rank submissions buffer their records in memory until BOTH return. There is
no log-file operation, print, lock, event or synchronization inserted between
paired submissions. Native traceback timers still name a blocked call; buffered
rank records from a call that never returns may not reach disk. Stream/device
handles and host submission durations are recorded on successful submissions.
After result drain ONLY, inspect both ranks' sticky/error/done/epochs, exact
support/q AND proposed IDs; require healthy state and advancing matching epochs.
No CUDA scheduling/overlap claim is made without a CUDA activity trace. Exact
healthy peer exchange is evidence of completed handshake, not a timeline.

Candidate/reference must use graph dispatch, have graph submission and healthy
peer evidence, make no positive eager-selector call or Triton warmup/handle-load
in real direct/OFL requests, complete all frames and match exact signatures.
Setup in the separately labelled model-level phase is not silently treated as
service request work. Every step duration is retained; a two-sample percentile
cannot conceal a slow wider batch.

## Ownership, deadlines and close

Only the coordinator may execute. Local preflight refuses off X86host, a dirty or
wrong tree, wrong source delta, missing/changed qualification/runtime, another
listener, insufficient48GiB RAM or any used swap. It opens no service/window.
The real run registers a X86host-local shared scoped_remote phase BEFORE
pair-open, writes intent before submission, and opens `prod_window.sh pair
COORD` INSIDE that owned scope. No lane-private registry is introduced.
It takes BOTH existing GPU locks nonblocking, verifies physical UUID inventory,
and passes the inherited locks to each new-session worker process. UUID order
and actual two-device properties are rechecked; emulation refuses.

Only this owned scope receives MemoryMax48G/MemorySwapMax0. Workers run
`taskset -c 12-15,28-31 nice -n 19`, eight compute threads/one interop thread.
No power/clock setting, installed unit, current link, shared tune or image is
changed. Production unit stop/start is exclusively the coordinator's existing
pair-window script, as explicitly requested for this diagnostic.

MEASURED driver/startup duration: UNAVAILABLE. ESTIMATED attendance allowance:
up to72minutes from immutable60-minute probe scope plus12-minute restoration
reserve. Three startups each have a600s bound; real graph startup is unknown.
The client also has a600s overall OFL-phase bound; every request independently
has30s. These bounds do not promise the matrix fits: the shared absolute probe
lease wins and truncates an overrun. Do not extend it without reviewed cold-build
measurements and a separately labelled setup design.

On success, error, lost pair-open reply, request/startup bound or ordinary
signal, stop/wait ONLY the processes this driver created, prove their groups
and allocated listener absent, then fence/reap/prove the probe scope absent.
Only then submit close in a SEPARATE shared owned scope. Close is bounded by
the ORIGINAL probe+12-minute ceiling, with90s reserved for final close-scope
absence proof. Second signals are masked during cleanup. Exact
`PRODUCTION HOLD OK` from prod_window.sh close is required. Failure of absence,
close or its final proof yields STOP/RESTORE FAILED and coordinator inspection;
it cannot claim a restored pair. The unsafe alternative of closing while an
old pair-open can still run is refused. No other process is killed by name.

Shared ownership fence/control receipts use the existing module's persistent
control path; all probe data, caches and stage trees stay in authorized PP4
scratch. Persistent ABORT records must not be deleted. This is an attended
coordinator diagnostic, not a replacement for W8, six-index greedy restoration,
full admission or the post-close watch of a later Spark HOLD.

## Coordinator command after independent review — not executed

On X86host, with both source trees staged, the reviewed runtime manifest and raw
baseline physical receipt, and a FRESH output directory:

```bash
source ~/.config/ndm/paths.sh
: "${PP4_PROBE_CANDIDATE:?absolute candidate checkout under X86host PP4 scratch}"
: "${PP4_PROBE_BASELINE:?absolute separate baseline checkout under X86host PP4 scratch}"
: "${PP4_REVIEWED_COMMIT:?full independently reviewed final Round10 commit}"
: "${PP4_PROBE_RUNTIME:?absolute runtime manifest}"
: "${PP4_BASELINE_PAIR_RECEIPT:?absolute original raw 1b81229 pair receipt}"
: "${PP4_PROBE_OUT:?fresh output beneath /opt/dsv41/x86host-data/pp4}"
: "${PP4_PROBE_RUN:?fresh PP4-server- tag, <=51 characters}"
cd "$PP4_PROBE_CANDIDATE"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/offload/dspark"
WINDOW_COORDINATOR=Deepseek/coordinator \
 /home/operator/miniconda3/bin/python3 -m offload.dspark.probabilistic.pp4.server_probe \
 --execute --baseline "$PP4_PROBE_BASELINE" --candidate "$PP4_PROBE_CANDIDATE" \
 --commit "$PP4_REVIEWED_COMMIT" --runtime "$PP4_PROBE_RUNTIME" \
 --pair-receipt "$PP4_BASELINE_PAIR_RECEIPT" --out "$PP4_PROBE_OUT" \
 --run "$PP4_PROBE_RUN" --port 18644 --weights /opt/dsv41/x86host-data/dspark
```

Default preview requires no arguments and makes no host or model call:
`python -m offload.dspark.probabilistic.pp4.server_probe` with the required
interpreter and worktree PYTHONPATH.

Interpretation is conditional. Baseline reproduction at a named phase plus
candidate clearance under the same cold/warm controls supports the graph
correction as the remedy. A cold setup that completes then repeats quickly
supports a readiness/setup diagnosis. Failure only in OFL prioritizes callback
or transport handling; failure in a common GPU phase means the one-line change
is insufficient. A candidate pass without baseline reproduction closes the
dispatch mismatch but leaves the original cause unresolved. Any probe failure
must stop before another Spark admission. No automatic promotion occurs.
