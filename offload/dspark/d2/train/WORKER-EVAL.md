# D2 adapted native worker qualification — isolated GPU plan

This is an evaluation harness, off by default. It is never selected by the
production worker CLI or by `DSPARK_WORKER_ARM`. No service/unit/Spark changes
are required. `build_adapted_evaluation_core` loads the exact official native
MXFP4 base with BF16 dense/head, FP32 adapted norms/Markov/confidence and FP32
low-rank deltas. The output label is `d2-adapted-native`, explicitly adapted,
with its own drafter AND Markov identities. It cannot advertise an unchanged
source-faithful identity. Native source receipts, exact base configuration,
adapter checkpoint hash, model configuration and runtime sources bind identity.

Deltas use the already QDQ'd input and are added before output rounding. TP2
output shards slice B rows; input shards slice A columns and reduce FP32
partials through the existing collectives. Grouped wo_a has one A/B pair per
owned group. No low-bit matrix is merged/requantized. Fused shared-base bypasses
are disabled for adapted projections. Every checkpoint trainable is validated
and consumed, including norms, Markov factors and confidence. Unknown, missing,
nonfinite or misshaped tensors refuse before installation. A second install or
an install after capture refuses. Legacy `package.py` low-bit merging is NOT
this route and must not qualify this floating-adapter profile.

Existing feature windows and Markov rows are discarded at installation; exact
FP32 hot Markov rows are rebuilt from the adapted factors BEFORE capture.
The dependency guard runs before TP2 backend flush/draft, outside graph replay.
In-place changes or tensor rebinding invalidate feature/Markov state and graphs,
then refuse further use until the worker is recreated. Adapters are immutable
for the worker lifetime.

`worker_eval` first verifies the Amendment-4 sample draw and runs the checkpoint
through the qualified reference evaluator. It holds predictions only in CPU
memory, destroys the reference model, then loads the SAME checkpoint in the
real native TP2 worker/backend. The real child-process proxy and TCP frame path
send each same-sample tap window through graph insertion, Markov cache and
fixed-depth-five drafting. It refuses missing/partial/misaligned inputs or any
proxy/worker fallback. Only aggregate block/position agreement and bonus-excluded
E[A] comparisons leave memory. Every sample has a distinct request ID, so the
protocol exercises reset/window ownership across samples.

The proxy's host handoff uses the CPU host-flag stand-in. This tests the real
process, wire/state and GPU worker, but cannot qualify GB10 CUDA gather/H2D
stream ordering. A coordinator G3 window on the Sparks remains necessary before
production. No universal exact-output or throughput claim follows from this
check. The code reports disagreement; it does not invent a new rounding tolerance
or automatically promote a candidate.

## Preflight and first solo run (not executed by D2-PILOT-R3)

Stage the reviewed complete repository commit immutably on X86host, including
`worker`, `native`, `kernels`, `spark/remote_dspark`, `d2/train` and their existing
build dependencies. A newly merged source snapshot requires a newly prepared
plan; source identity intentionally refuses stale plans. Use the existing pinned
X86host environment. No downloads/installs are assumed. Read fleet placement state,
`nvidia-smi -L`, `free -g` and current progress first. Verify the power logger,
both solo UUIDs and unchanged 225 W caps; do not stop anything to obtain a lock.

First use ONE named solo under the existing supervisor. This is TP2 emulation:
two logical rank shards/graphs on one card, not a two-device performance result.
Use the frozen real candidate checkpoint only after G1 and the approved pilot;
a synthetic checkpoint can only provide a labelled rehearsal.

```bash
source ~/.config/ndm/paths.sh
export D2_TRAIN_ROOT=$HOME/dsv41-offload/d2/pilot/amend6-20261023
export D2_GIT_SHA=$(cat "$D2_TRAIN_ROOT/SOURCE_COMMIT")
export D2_PILOT_GPU=GPU-240876ad-ea42-47de-1b8d-f6153810b7e1
export TMPDIR=/opt/dsv41/x86host-data/d2/pilot/worker-eval-scratch
mkdir -p "$TMPDIR"
free -g
nvidia-smi -L
# Set these to the actual source-bound plan/manifest/checkpoint, never a guess.
: "${D2_EVAL_PLAN:?}" "${D2_EVAL_MANIFEST:?}" "${D2_EVAL_ADAPTER:?}"
: "${D2_EVAL_K2_ACAP:?}" "${D2_EVAL_K2_LEGAL:?}" "${D2_EVAL_OUT:?}"
"$D2_TRAIN_ROOT/repo/offload/dspark/d2/train/run_pilot_locked.sh" worker-eval \
  -m offload.dspark.d2.train.worker_eval \
  --plan "$D2_EVAL_PLAN" --manifest "$D2_EVAL_MANIFEST" --adapter "$D2_EVAL_ADAPTER" \
  --k2-acap "$D2_EVAL_K2_ACAP" --k2-legal "$D2_EVAL_K2_LEGAL" \
  --root /opt/dsv41/x86host-data/dspark --native-root /opt/dsv41/x86host-data/native-1017 \
  --bind 10.202.28.1 --max-model-len 327680 --out "$D2_EVAL_OUT"
```

Run detached in a uniquely named tmux session using this immutable command file.
The wrapper fails a busy flock immediately, verifies 225 W and supervises its own
child's <=128 GiB RSS/no swap. Use a declared total context <=524288 if sample
anchors require it; the command refuses samples beyond the declared limit. This
is not a request for new long-context generation. Stop only this lane's owned
process if necessary. Remove its scratch at the end. The aggregate output is an
evaluation record, not a deployment receipt.

## Two physical solo cards

After the emulated memory/correctness screen, the same command with `--two-solos`
can exercise physical TP2 over PCIe. It must run under a detached supervisor with
BOTH inherited nonblocking individual locks: fd 8 on
`~/dsv41-offload/locks/gpu-240876ad.lock`, fd 9 on
`~/dsv41-offload/locks/gpu-bc62df2a.lock`, and
`CUDA_VISIBLE_DEVICES=GPU-240876ad-ea42-47de-1b8d-f6153810b7e1,GPU-bc62df2a-59cf-9828-7d43-558d1355bab7`,
`D2_ADAPTER_EVAL=1`. `require_evaluation_lease(False)` proves BOTH inherited
flocks are already held and owned, and rejects every other UUID pair before
CUDA. The one-solo `lease.py` intentionally refuses a two-card environment; do
not reuse it as if it supervised both cards. The coordinator must supply the
matching two-card supervision/telemetry window before that run. Check both caps
remain 225 W, no swap, available memory and peer support. No claim is made that
these solos have NVLink or reproduce production-pair timing. The production
c4e7a371/7744b44b pair and its units are never part of this command.

Budget ESTIMATED 1 GPU-hour for worker qualification (e.g. 30 minutes on two
solos) plus reference/model load, process startup, graph capture and I/O. Record
actual per-phase wall/GPU time and peak memory during execution; the plan budget
is not a measurement. If native ranks do not fit emulation or PCIe peer access
fails, preserve the exact error and request a coordinator-qualified window;
never fall back to the production pair. The missing real-G1/candidate and GPU
checks are explicit external prerequisites, not passing CPU tests.
