# Amendment-7 G1(b): coordinator procedure

The coordinator must use this order (amendment 7b / S35 FIX-2):

1. Run `decode_close.sh pause` and wait for its return with `D2-GEN2-EXIT` seen.
2. Immediately capture the final `/metrics` snapshot as `metrics-<UTC>.prom`
   and save the generator's last `rate.jsonl` row (the forced exit row) as
   `rate-<same UTC>.json`. Record the pause time. Capture within 120 seconds of
   the generator's exit; the existing 120-second freshness check is unchanged.
3. Collect the attestation evidence with `pass2_collect.sh --execute` while the
   derivative still serves, so the mounts reflect the captured deployment.
4. Flush with a NEW nonce, then copy and record that nonce at copy time.
5. Roll back. Only then resolve/review the evidence and stage attestation/context,
   transfer/hash-check, assemble/G1(a), build the live reference, score K2/G1(b),
   and stage the pilot after the K2 holds release their locks.

Do not delay the final snapshot or exit row until after collection, flush, copy
or rollback: the paused generator writes no periodic rate rows, and a later
snapshot cannot recover this measurement.

Prepared and rehearsed on Controller, CPU only. This lane did not connect to any
host, run CUDA, touch generator state or write into W12. Commands below are for
coordinator review and later **local x86host** execution. Production changes,
pause/capture/rollback operations and transfers are outside this procedure.

`python -m d2.g1 procedure` only prints commands. It has no execute option and
reads no data. From the reviewed checkout on x86host:

```bash
source ~/.config/ndm/paths.sh
export PYTHONPATH="$PWD/offload/dspark:$PWD"
export CUDA_VISIBLE_DEVICES=''
D2_G1_PYTHON=$HOME/dsv41-offload/o2/venv/bin/python
D2_G1_SHARDS=/opt/dsv41/x86host-data/d2/decode-d2pass2-r1-c2/shards
D2_G1_WORK=/opt/dsv41/x86host-data/d2/g1-d2pass2-r1
# Private output directory, distinct from inputs; coordinator creates it locally.
umask 077
mkdir -p "$D2_G1_WORK"
```

First the coordinator must supply the final metrics snapshot captured AFTER the
second-pass generator pause and BEFORE the rollback engine restart, together
with its matching generator rate row. Do not use a post-rollback snapshot or
an intermediate snapshot as final. Preserve the original timestamped filenames.
Use local copies of all chosen snapshots and matching rate rows; no endpoint
is fetched by these tools. The existing first snapshot can be included even
if it has no matching rate; the final snapshot MUST have one. A rate row's `t`
must precede its snapshot by no more than 120 seconds. These are cumulative
counters since the 02:55Z start, not differences between the first/final files.

Substitute the actual UTC pause and rollback times and local file paths; the
placeholders deliberately cannot produce a receipt:

```bash
"$D2_G1_PYTHON" -m d2.g1 live-reference \
  --snapshots /LOCAL/metrics-FIRST.prom /LOCAL/metrics-FINAL.prom \
  --rates /LOCAL/rate-FINAL.json \
  --paused-at ACTUAL_UTC_PAUSE_TIME --rollback-at ACTUAL_UTC_ROLLBACK_TIME \
  --out "$D2_G1_WORK/live-reference.json"
```

Receipt creation refuses an existing output. Counter creation times must belong
to the 02:55Z restart, stay identical across snapshots, and no counter may drop.
Final snapshot time must be >= pause and < rollback. Coordinator-supplied times
are an explicit attestation; file contents cannot independently prove the pause.
All raw source SHA256 hashes and absolute paths are bound in the receipt. Keep
these sources at those paths throughout pilot staging/training. If transferred,
build the reference receipt locally against the transferred byte-identical files;
its receipt identity includes paths. Never edit an old receipt to relocate it.

Inspect `status`, `foreign_share`, `position0_acceptance`,
`per_position_acceptance`, `accepted_per_round`, `mean_depth` and the counters.
Foreign share is `(engine generation_tokens_total - generator gen_tokens) /
engine generation_tokens_total`. Above 2% produces INCONCLUSIVE; report it to
NDM. Per-position acceptance is accepted at each position divided by all draft
rounds; positions >0 are censored by adaptive depth and are information only.
The API process start may predate the engine restart; engine counter creation
series identify the engine instance. Never substitute process start for that.

Render the exact holds/scores (still dry):

```bash
"$D2_G1_PYTHON" -m d2.g1 procedure \
  --shards "$D2_G1_SHARDS" --work "$D2_G1_WORK" \
  --live-reference "$D2_G1_WORK/live-reference.json"
```

Default card is `GPU-bc62df2a-59cf-9828-7d43-558d1355bab7`, lock
`$HOME/dsv41-offload/locks/gpu-bc62df2a.lock`. The only alternate in g1.py's
GPU_ALLOWED is `GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc`, with
`gpu-7744b44b.lock`. The coordinator must confirm current allocation before
using that alternate; its presence in the allowlist is not a new allocation.
Never use a production card. No power, service or allocation changes are made.

Execute each rendered K2 hold independently in a coordinator-owned detached
session, only while its prescribed card is available. Each command uses a
nonblocking flock, **780-second internal deadline**, and **820-second outer
TERM timeout**, with a 20-second kill grace (840 seconds maximum per lease). K2 preserves complete sequences in JSONL;
if the deadline stops it, repeat that split's identical hold in a later lease.
Do not run multiple holds on one card. Confirm all MANIFEST sequences finished
before CPU scoring. Incomplete/missing draft coverage refuses scoring/admission.
K2 root remains `/opt/dsv41/x86host-data/dspark`; no weights are downloaded.

Run the rendered CPU score for `dev` first, then `heldout_legal` and
`heldout_general` as information. Outputs per split are
`SPLIT.k2_drafts.jsonl` and `SPLIT.json` in D2_G1_WORK. G1 score binds the exact
MANIFEST hash, K2 JSONL hash, absolute live receipt path and live receipt hash.
Binding PASS requires a VALID reference and the amendment-7b gross-error
tripwire: `proxy position0 >= 0.5 * live position0` on **dev**. Comparison uses unrounded rates. K2 committed-trajectory
simulation is unchanged. Reports include proxy/live position-0 rates, accepted
per round and per-position values, and name depth-0 censoring, window, split
coverage and foreign-share biases. The former ±0.03 rule is withdrawn. The historical `--mode historical-acap` path uses 1.82 +/-
0.1 accepted/round only for history; it cannot admit a real pilot.

Pass `dev.json`, `dev.k2_drafts.jsonl` and `heldout_legal.k2_drafts.jsonl` to
[PILOT-STAGE.md](train/PILOT-STAGE.md). Pilot sampling remains seed 20261024,
at most 1,000 complete rounds per split, once before training. Amendments 3–6
verdict formulas and the reserved TEST set are unchanged. Neither successful
command rendering nor synthetic CPU fixtures establish real G1 or pilot PASS.
