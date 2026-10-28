# Second-pass full training and full G2

This procedure follows the real pilot-2 ACE PASS only (Amendment 8, original source d1dd41a). It prepares a separate FAST
checkpoint and a full development-population G2 receipt. It does not package,
qualify a fresh TEST set, run a worker, access a Spark or deploy anything.
See [admission audit](FULL-ADMISSION.md) and [pilot staging](PILOT-STAGE.md).

## Cost and scope sheet

MEASURED inputs: coordinator's D2-FULL-PREP brief records FAST approximately
0.85 seconds per B128 optimizer update and REF approximately 40 seconds per
update, on unchanged 225 W solo cards. The pilot has 159 updates per arm; these
are historical first-pilot KL rates; ACE throughput is unmeasured here. Historical reference evaluation is approximately
0.5 seconds per round per pass (D2 plan amendment 4). These rates are inputs,
not full-run measurements by this lane. All costs below are ESTIMATED and omit
startup, I/O, validation, checkpoint writes, interruptions and queueing.

MEASURED metadata: S36-inputs/shards/MANIFEST.json SHA256
`450bb71653f3def68cd95ee9579a456c74c469d08a016940cceda24de48b4f30`:
65 training sequences / 496,416 captured rows; 30 dev / 233,500 rows;
24 heldout_legal / 220,458 rows; 12 heldout_general / 40,566 rows.
Captured rows include context and are NOT training blocks. The same manifest
reports 990,940 total rows and 974,172 generated span positions; their difference
is 131 * 128, so every sequence has all 128 initial context rows. D2Blocks
makes L-gen_start-1 overlapping training blocks per sequence. This yields
488,096 training span positions and **488,031 unique training blocks**.
The coordinator must confirm len(D2Blocks(manifest, 'train')) against actual
sidecars before launch; Controller's receipt-only copies cannot do that read.

Keep PILOT.md's **three epochs**, fresh from the native base and zero LoRA B;
do not continue a 20K pilot checkpoint under a new cosine schedule. B128,
chunk8192, rank32, raw taps, AdamW .0002/.01, clipping 1 and seed23 retain
the qualified ACE recipe from the admitted receipt. No recipe CLI override exists.
Other supported receipts use their own qualified recipe: TVD or KL at 2e-4,
LR at 5e-5 with ceil(.1 * total_updates) warm-up followed by cosine.
These cost estimates extrapolate historical KL rates, not measured full ACE rates. Three epochs present 1,464,093 blocks in
3 * ceil(488031 / 128) = **11,439 updates**. Overlapping five-target windows
and repeated epochs do not create more unique data.

| Work | ESTIMATED solo GPU-hours | ESTIMATED wall time |
|---|---:|---:|
| Full FAST, three epochs, one solo | 2.701 | 2.701 h |
| Same full run in REF (cost comparison only; not scheduled) | 127.100 | 127.100 h |
| Full dev baseline + candidate, 90,837 complete rounds each | 25.233 | 25.233 h on solo A |
| Full legal baseline + candidate, 77,949 complete rounds each | 21.653 | 21.653 h on solo B |
| General baseline + candidate, at most 14,624 rounds each | <=4.062 | <=4.062 h on solo B after legal |
| Full G2 including general | <=50.947 | about 25.715 h with two solos |
| FAST followed by full G2 | <=53.648 | about 28.416 h with two solos for G2 |

MEASURED complete dev/legal round counts come from the real pilot-plan header's
`sample.population_rounds`, not the 1,000-round samples. General's 14,624 comes
from its G1 receipt and includes possible incomplete tails; the full scorer
excludes those, so it is a planning upper count. Each full G2 request is scored
twice, once unadapted and once adapted, through reference arithmetic. The
historical 0.5 seconds/round is not a guarantee on these longer real contexts.
A pilot evaluation cost must not be substituted for this approximately day-long
full G2. Request-boundary journals allow interruptions without discarding
completed requests. Repeating a partially scored request adds overhead.

**Data sufficiency:** this is enough unique training data for the approved
20K matched pilot and an evidence-gated Phase 2a adapter run. The plan approves
approximately 1M generated tokens across the dataset; it does not impose a
1M-unique-training-block floor. The pass generated MEASURED 974,303 tokens in
3.0 Spark-hours (coordinator brief), close to that dataset-scale intent.
However, **the available unique training population is only about 0.488M**;
it is not enough to claim training on 1M unique blocks. Three epochs do not
fix that distinction. There is no basis to promise adaptation from the size:
the actual pilot, full G2, fresh reserved TEST and real-worker/GB10 G3 decide.

If 1M unique TRAIN blocks are required as a larger data objective, the deficit
is approximately 511,969. At the pass-2 rate 974303/3 = 324,768 generated tokens
per Spark-hour, additional generation costs ESTIMATED **1.58 Spark-hours** if
all new eligible source families enter train and current heldouts remain frozen,
or approximately **3.15 Spark-hours** at the current approximately 50.1% train
allocation. These are optimistic generation-only costs: new prompts/families,
context rows, incomplete tails, top20 verification, copying, assembly and G1
add work. Merely reaching 1M total generated tokens instead needs approximately
25,697 more tokens, **0.079 Spark-hours**; that does not yield 1M training blocks.
Spark-hours here use exactly the brief's 974,303/3.0 accounting unit; do not
silently confuse deployment wall-hours with two-rank GPU-card-hours.
No extra generation is launched or newly authorized by this sheet. Phase 2b's
3–5M tokens/full-dense or expert-LoRA training still needs a costed NDM decision.

## Coordinator procedure (local x86host only)

Do not run these execute commands in a lane. No script connects to another host.
Review/transfer the integration commit through the coordinator's existing process;
keep that clean checkout and the original pilot snapshot available unchanged.
The full snapshot must be NEW and separate from the pilot. Production units,
production-pair cards, power/clock settings and live requests are outside this
procedure. The existing interpreter and requirements-x86host.lock are sufficient.

In the reviewed full checkout on x86host, render first:

```bash
set -euo pipefail
source ~/.config/ndm/paths.sh
export PYTHONPATH="$PWD:$PWD/offload/dspark"
export CUDA_VISIBLE_DEVICES=''
D2_FULL_PYTHON=$HOME/dsv41-offload/e/venv/bin/python
full_args=(
  --source "$PWD" --commit REVIEWED_FULL_40_HEX_COMMIT
  --snapshot "$HOME/dsv41-offload/d2/full/decode-d2pass2-r1-p2-ace-full"
  --output /opt/dsv41/x86host-data/d2/full/decode-d2pass2-r1-p2-ace-full
  --assembly /opt/dsv41/x86host-data/d2/decode-d2pass2-r1-c2
  --pilot-source "$HOME/dsv41-offload/d2/pilot/decode-d2pass2-r1-p2-ace/repo"
  --pilot-receipt /opt/dsv41/x86host-data/d2/pilot/decode-d2pass2-r1-p2-ace/receipt.json
  --g1 /opt/dsv41/x86host-data/d2/g1-d2pass2-r1/dev.json
  --k2-dev /opt/dsv41/x86host-data/d2/g1-d2pass2-r1/dev.k2_drafts.jsonl
  --k2-legal /opt/dsv41/x86host-data/d2/g1-d2pass2-r1/heldout_legal.k2_drafts.jsonl
  --k2-general /opt/dsv41/x86host-data/d2/g1-d2pass2-r1/heldout_general.k2_drafts.jsonl)
"$D2_FULL_PYTHON" -m offload.dspark.d2.train.full_stage "${full_args[@]}"
# Confirm receipt.json reports a genuine PASS; FAIL/INCONCLUSIVE stops here.
# LOCAL CPU admission, no GPU or detached job:
"$D2_FULL_PYTHON" -m offload.dspark.d2.train.full_stage "${full_args[@]}" --check-inputs
```

Dry default does not read missing inputs, invoke git/tmux or initialize CUDA.
Explicit CPU admission rechecks pilot_stage's genuine -c2 assembly/provenance,
G1(a), exact dev G1(b), VALID hash-bound live sources, parity and K2. It pins the complete d1dd41a pilot-2 map and verifies
the original pilot source and invokes its unchanged require_pilot on CPU to
recompute real PASS, intervals, sample/unit identities and every checkpoint,
then redraws the frozen sample from both exact K2 files in that original namespace.
It also checks full general K2 coverage. Synthetic, missing, FAIL/INCONCLUSIVE,
modified-source, withdrawn acap or mismatched receipts stop admission.

Check free RAM and the existing exact `dsv41-powerlog` session before launch.
Only the coordinator, on local x86host with the exact assembly path, can execute:

```bash
export D2_COORDINATOR=Deepseek/coordinator
"$D2_FULL_PYTHON" -m offload.dspark.d2.train.full_stage "${full_args[@]}" --execute --phase stage
"$D2_FULL_PYTHON" -m offload.dspark.d2.train.full_stage "${full_args[@]}" --execute --phase full-train --job full-fast
# Wait for full-fast's detached process to exit successfully and inspect the final checkpoint/log.
# full-g2 admission independently verifies all three epochs and the exact PASS identity.
"$D2_FULL_PYTHON" -m offload.dspark.d2.train.full_stage "${full_args[@]}" --execute --phase full-g2 --job g2-dev
"$D2_FULL_PYTHON" -m offload.dspark.d2.train.full_stage "${full_args[@]}" --execute --phase full-g2 --job g2-legal
# After legal finishes and releases solo B (do not launch both on that card):
"$D2_FULL_PYTHON" -m offload.dspark.d2.train.full_stage "${full_args[@]}" --execute --phase full-g2 --job g2-general
# After dev.json, heldout_legal.json and heldout_general.json all exist:
"$D2_FULL_PYTHON" -m offload.dspark.d2.train.full_stage "${full_args[@]}" --execute --phase finish --job g2-finish
```

Stage uses git archive, SOURCE_COMMIT and STAGED.json in the NEW full snapshot;
records include input hashes, the admitted recipe and exact rendered commands.
Snapshot/output basenames must equal the pilot receipt namespace plus -full;
these commands use decode-d2pass2-r1-p2-ace-full. A different qualified recipe
requires its own original receipt/snapshot and correspondingly distinct namespace. Every subsequent
phase revalidates source bytes/file set, inputs and the command contract before
starting a session. Inherited GIT_* variables are stripped from child git calls.
Session names are `d2full-decode-d2pass2-r1-p2-ace-full-{full-fast,g2-dev,g2-legal,g2-general,g2-finish}`;
existing sessions/scripts refuse. Only these new sessions belong to this run.
No session is killed. Snapshot scripts and logs are `NAME.sh` and
`logs/NAME.coordinator.log`; leased child logs append in `logs/full-fast.log`
and `logs/full-g2-SPLIT.log`.

Full FAST runs on 240876ad; G2 dev uses 240876ad, legal then general use bc62df2a.
All GPU commands go through unchanged run_pilot_locked.sh: UUID-only pinning,
nonblocking per-solo flock, unchanged 225 W, power logger, child RSS <=128 GiB,
no swap, supervised cancellation. Full initialization uses seed23 before model
construction; it never resumes a pilot checkpoint. Fresh initialization must match the pilot initialization hash. Numerical source
files and the pilot verdict implementation remain byte-identical to pilot 2 d1dd41a.

The final train file is `full-fast.pt`, with immutable
`full-fast.checkpoints/step-NNNNNNNNN.pt` boundaries. The loop saves every 300 s
and on its supervised stop. Resume binds run/data/precision/base/backend/pilot
hash, explicit full_recipe and RNG/scheduler/optimizer state. Inspect an interrupted job's saved
boundary; resume only that job in its existing namespace:

```bash
# Example only after an actual incomplete full-fast interruption:
"$D2_FULL_PYTHON" -m offload.dspark.d2.train.full_stage "${full_args[@]}" --execute --phase full-train --job full-fast --resume --attempt 1
# Example only after an incomplete dev scoring interruption:
"$D2_FULL_PYTHON" -m offload.dspark.d2.train.full_stage "${full_args[@]}" --execute --phase full-g2 --job g2-dev --resume --attempt 1
```

If interrupted again, increment --attempt for a fresh session/script name.
Never rerun a completed training/scoring job. G2 creates SPLIT.requests/contract.json
and one aggregate receipt per completed request. Resume requires unchanged
manifest, all source hashes, adapter, pilot, recipe and K2/population identities, and
reuses only completed requests; the interrupted request restarts. It never
samples/limits the complete-round population. Final SPLIT.json creation is
exclusive. Keep the journals private alongside the full outputs.

`g2.json` binds all three part receipts to the same recipe, manifest, source, pilot and
completed adapter. Accepted length is the committed-K2 trajectory proxy, bonus
excluded. Full dev/legal checks retain legal gain >=.15 accepted draft tokens,
request-bootstrap 95% lower bound >0, every-request gain >=-.2, strictly lower
mean teacher KL at each position, and amendment-6 no-native-regression floor
of -.01 on each development split. Dev must contain all 30 requests and legal
at least 20; the scorer enumerates the full actual manifest populations.
All splits refuse NaN/inf or incomplete summaries. heldout_general keeps the
D2-TRAIN G2 gates of `evaluate.g2` (amendment 4: final G2 unchanged): at least
10 requests, gain >=-.03, every-request gain >=-.2 and strictly lower KL at each
position; it has no native-regression floor and cannot rescue a failing
dev/legal verdict. The old
four-acap requirement is replaced by the amendment-7 dev population; no acap
receipt or old `evaluate g2 --acap` command is used.

On G2 FAIL, preserve the receipt and report it; do not retune on heldouts or
redraw the pilot sample. G2 PASS is full DEVELOPMENT evidence only. Fresh reserved
public TEST families, isolated real-worker qualification, the floating native
adapter package/profile (WORKER-EVAL.md), V1 and coordinator GB10 G3/depth work
remain required. No checkpoint, package or service is automatically promoted.

## CPU validation

Use Controller's HARNESS_PYTHON. No new dependencies or installs. The private tmux
base must already exist and be short enough for a Unix socket; the lane scratch
path below plus `/t/s33tXXXXXXXX/tmux-1000/default` is 96 bytes.

```bash
source ~/.config/ndm/paths.sh
D2_FULL_SCRATCH=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/D2-FULL2-ADMIT
mkdir -p "$D2_FULL_SCRATCH/t" "$D2_FULL_SCRATCH/tmux"
env -u TMUX OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH="$PWD:$PWD/offload/dspark" TMPDIR="$D2_FULL_SCRATCH" \
  TMUX_TMPDIR="$D2_FULL_SCRATCH/tmux" S33_SHORT_DIR="$D2_FULL_SCRATCH/t" \
  "$HARNESS_PYTHON" -m pytest -q offload/dspark/d2/tests \
  offload/dspark/d2/train/tests offload/dspark/d2/s34 offload/dspark/d2/s35 offload/dspark/d2/s36 offload/dspark/d2/s37 \
  --basetemp "$D2_FULL_SCRATCH/pytest" -o tmp_path_retention_policy=failed
"$HARNESS_PYTHON" -m mypy --strict --explicit-package-bases --follow-imports=silent \
  offload/dspark/d2/train/{loop,full_admission,full_stage,full_train,full_eval}.py \
  offload/dspark/d2/train/tests/{test_full_admission,test_full_stage,test_full_recipes,snapshot_git}.py \
  offload/dspark/d2/s37/test_s37_{admission,numerics,stage,g2,cost}.py
```

Successful fixture cleanup prevents the broad suite retaining multiple GB.
CPU tests verify dry/no-access behavior, source lineage and unqualified pilot
refusals, UUID/namespace protections, full populations, the original G2 bootstrap,
threshold failures/nonfinite/mismatched coverage, and interruption/resume using
tiny real reference arithmetic. They cannot establish real pilot PASS, GPU
throughput, adaptation, serving equivalence or production readiness.
