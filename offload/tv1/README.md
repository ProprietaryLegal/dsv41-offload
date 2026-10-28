# TV1 target-work experiments

Round9 moves the installed SM121 oracle between experiment stop/staging and
candidate model startup, using the same pinned image/overlay mounts. It no longer
competes with a live server. Pre-HOLD admission checks 8GiB plus the largest
free Normal physical buddy block, saving the byte/order snapshot and explicitly
leaving CUDA allocatability unqualified. Read the first section of WINDOW.md. Compact is
CLOSED / NOT A CANDIDATE after its second greedy-gate failure; no rule changed.

Default OFF. Round4 fits feature-specific C/arm/C schedules within85min ESTIMATED including recovery.
Round8 repairs candidate layer binding during memory profiling using the active
forward map, adds no-current-config regressions and a local pre-HOLD CPU import/binding
smoke. Read CANDIDATE-INIT.md and WINDOW.md; GB10 startup/numerics remain unqualified.
Round10 registers NEW experiments as `tv1-r10-near-tie-0p9` (NDM 2026-10-27: 0.9 nats):
the same NBW first-divergence/top-20 rule with the threshold 0.501 -> 0.901 nats passed
explicitly; NBW's own default is unchanged. Round7 (`tv1-r7-nbw-near-tie`, 0.501) and strict
historical runs keep their verdicts. WINDOW.md records the rules and diagnostic receipts.
Round5 fixes recovery log collisions, preserves both-rank startup logs and repairs
two compact startup blockers; read INIT-DIAGNOSIS.md before another window.
Round3 replaces private ownership with the shared OWN1 modules; the original scorer has Ampere reference tests.
Round2 engine mapping/guards and GB10 numerics/speed remain unqualified.
Read EXPLORE.md. Build only under TV1 scratch with TMPDIR there. Never edit image/K1 sources.

```
source ~/.config/ndm/paths.sh
export TMPDIR=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/TV1/tmp
/home/operator/miniconda3/bin/python3 -m offload.tv1.build_overlay --output "$TMPDIR/control"
/home/operator/miniconda3/bin/python3 -m offload.tv1.build_overlay --switch candidate --output "$TMPDIR/candidate"
```

Control overlay is byte-identical to the pinned current indexer and K1 runner. Build
applies only requested zero-fuzz patches and copies our own helper modules. It refuses
drift, pre-existing output and patch offsets. A sidecar manifest binds every output.
The mounted candidate code also requires `VLLM_TV1_CANDIDATE=1` (unset/0 is OFF).
Only decode consumer layers change; source layers, prefill and top-k are unchanged.
The patch uses the actual row page-table expansion, K1 page stride and segregated
FP8 value/scale storage, and retains the existing candidate mask as an extra guard.

For runtime hash verification, emit `verify_runtime.py --emit --manifest <sidecar>`;
only the coordinator pipes it into BOTH experiment containers. This hashes mounted
bytes and imports no CUDA code. It does not certify numerical results.

Qualification so far: CPU layout/hash/build checks; locked x86host 3090 225W synthetic
and recorded geometry selected-top-k oracle checks and captured replay. GB10's actual
DeepGEMM dense scores/top-k, six K1 experimental greedy checks and isolated timing are still
mandatory gates. The CPU mathematical oracle is not the installed SM121 implementation.
A failed candidate selected-index identity remains a hard STOP. W8 restoration
still requires six strictly IDENTICAL references.

S51 Round2 qualification is specified in ROUND2-CANDIDATE.md and WINDOW.md:
288 production-derived cases, actual-forward comparisons before timing, 8GiB
oracle headroom, current OPS1 control equality and an enforced HOLD+98 ceiling.
Ordinary 5–7GiB available memory cannot admit this high-offset oracle. Missing
headroom, coverage, numerical identity or time is STOP; no promotion is implied.

No donor expression is adapted: kernels and policies are original. Donor references
are documented in EXPLORE.md and exploration-pins.json. No donor source is shipped.

Compact arm: `--switch compact`, then `VLLM_TV1_COMPACT=1`;
`VLLM_TV1_COMPACT_TAU=0.5` is the initial, pre-registered survival cutoff. The remote
confidences are staged through an owned pinned buffer/event, associated by request
id, and invalidated on new/preempted/finished requests. Rank zero broadcasts the
chosen vector before any rank builds target geometry. One target row always remains.
The scheduler keeps its original reservation; its rollback and the worker's actual
rejection count end at the same position. Existing acceptance sampling is unchanged.

This initial compact arm deliberately synchronizes the small confidence/width vector
and forces shortened batches eager. Those costs may exceed saved work, especially at
B1, which has no unequal-request padding. Mixed prefill and structured-output batches
keep original widths. CP/DP/PP, LoRA, batch sharding, local/resilience/shadow drafter,
and built-in adaptive verification are refused. Round5 permits the recorded plain
conf depth policy, retaining its masks and rejection checks; other depth modes,
calibration, lookup, context adjustments and file reload are refused. This is a runnable
conservative experiment, not a claim that ragged GB10 graphs are already qualified.

Round3 source/CPU handoff: shared cgroup scopes, exclusive HOLD admission, shared
JIT quiet proof and15min post-close health watch. Primary decision is unprofiled
server request-decode ms/spec request-round with acceptance pooled over four
requests/cell/start. Round4's WINDOW.md fixes separate C/arm/C windows: candidate
150K B1+short B1 (84.25min), compact32K B4+short B1 (57min), including12min recovery
(ESTIMATED). The single ON must pass against EACH OFF separately. Measurement
ends+73; every start reserves all remaining blocks and recovery within+85.
Original85/86/98 safety ceilings remain. Missing qualification, headroom or closing
control refuses; no independent ON repeatability or GB10 performance is claimed.
