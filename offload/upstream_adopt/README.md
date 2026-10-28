Status 2026-10-23: S17 round 2 cleared a274e33 for GB10 testing; merged 7bf6adb. No UP1 production result. 0078 dropped; SO1 shelved and excluded. See [live roadmap](../../documents/plans/roadmap.md) and [measured record](../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# UP1 adoption candidates on image v5

This CPU-qualified adoption set passed S17 round 2 on a274e33 and was merged in 7bf6adb for the coordinator's GB10 window.
It does not change any running service. There is no local speed, memory or quality claim.

Read `CATALOGUE.md`, `PATCH-CATALOGUE.md`, then `GB10-TEST-PLAN.md`.
`catalogue.json` covers all7 commits,28 patch files and72 changed paths in b172664..afe4d5e.
`image-audit.json` proves all40 changed installed-source mirrors reproduce byte-identically from v5 plus the
in-image patch chain (fuzz0): there are no unexplained changes hiding outside patches/.

## Rebuild and check (Controller CPU only)

```bash
source ~/.config/ndm/paths.sh
CUDA_VISIBLE_DEVICES= "$HARNESS_PYTHON" -m offload.upstream_adopt.catalogue --check --audit-image
CUDA_VISIBLE_DEVICES= "$HARNESS_PYTHON" -m offload.upstream_adopt.build_overlay
"$HARNESS_PYTHON" -m offload.upstream_adopt.make_configs --check
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 "$HARNESS_PYTHON" -m pytest -q offload/upstream_adopt/tests
"$HARNESS_PYTHON" -m mypy --strict --explicit-package-bases offload/upstream_adopt/{catalogue,build_overlay,make_configs,verify_runtime}.py
bash offload/kvpack/test_stage_overlay.sh
```

`build_overlay --write` regenerates only this lane's 42 known output files and MANIFEST after pins/zero-fuzz patches
and py_compile. It refuses changed inputs and tampered previous output. It never runs donor code, launchers or JIT.
Source inputs and patch SHA256s are in `inputs.lock.json`; every delivered file hash is in `MANIFEST.json`.

The core tree is v5 + K1 + UP1's process-aware X1 replay port + donor0064 + rebased0067. All
K1 mounted files, including model_runner.py, remain byte-identical. Original K1 source
is untouched. The complete current remote package is pinned/copied so replacing gather/roundcore retains every
async process helper. New code never imports torch in the process proxy.

The stock X1 replay tests in this base FAIL their old gather hash fixture (a0d78... expected vs551324... current).
This is inherited drift after the process proxy moved BatchView into roundcore.py. UP1's0075-process.patch preserves
max_q's positional slot and adds engine-only tap_lens AFTER it. The replay tests here execute the actual composed
files and preserve X1's safety assertions. Do not describe the original X1 suite as passing on this branch.
No contract version/wire-layout change is required: STEP carries actual real row count/first position; partial
prefill cannot request drafting; final128-row suffix rebuilds the worker context. These optional provenance
semantics must accompany the coordinator's CONTRACT update if adopted.

## Configurations

All generated `offload/launch/configs/D-kv6-async-up1-*.env` arms derive from
D-kv6-async-process-nap-lazy THROUGH K1's D-kv6-async-pack: capacity, remote TP2 drafter, remote required Engram,
process proxy,50us spin/20us nap,LAZY loading,6GiB KV/rank,327680 context and four requests remain the baseline.
The local drafter is not reintroduced. No prefix connector or SO1 policy is included.

| Arm suffix | Isolated change | Override (0 disables) |
|---|---|---|
| control | Composed files, all UP1 features off | every UP1_* default0 |
| split | 0064 target33–63 row MoE | UP1_SPLIT=63 |
| vor | 0067 target SWA + replay copy | UP1_VOR=1 (or swa/copy) |
| replay | 0075 target intermediate replay + actual remote taps | UP1_REPLAY=1 |
| k5 | 0040 five drafts at1–4 streams | UP1_K5=1 |
| dual-ring | Target ring over OUR f1 two PCIe functions | UP1_DUAL_RING=1 |
| cpu-pin | Worker process/thread X925 affinity | UP1_CPU_PIN=1 |
| core | split + vor + replay | each independently overrideable |
| all | core + three config candidates | screening only AFTER individual gates |

Bad flags fail before launcher execution. K5 + K1_NODROP is refused until a combined spec is supplied.
These profile files themselves are trusted repo files, not donor launchers. Source them only from repo root as
start_exp does. Defaults in overlay code are OFF. An env change takes effect at a fresh engine start, not live.
The control keeps K1's original O1 Engram root, shim, extension sources and JIT identity in every arm.
Only switch-gated Python changes and inert helper files are added to K1; P0 still verifies runtime equivalence.
The engine/model files and complete remote package get one mount per target, never duplicate mounts.

0078 is dropped this round under S17 FIX-2. Its unmounted reference code remains at
`offload/spark_leads/patches/0078-remote.patch`; it is absent from UP1's inputs, overlay, configs and build steps.
S17 found ungated C++ work, a 2 ms hint timeout that could disable the foreground connection, and 2.4–4.8 ms
row-split overhead. Reconsider only after those defects are fixed and exposed layer-1/14 wait is measured.

## SO1 shelved

SO1 and the core-so1 composition are SHELVED by the coordinator's 17:34Z decision on 2026-10-23: after S19 found a second
liveness defect, the reorder-only SO1-R2 redesign removed the modeled gain. The earlier hold is superseded. No core-so1 config, SO1 lock, composition code or optional build route is shipped in this set.
Reintroducing a combined scheduler requires an independently checked SO1 revision, an explicit new pin and
both running/waiting0075 hooks plus real-KV admission/fairness tests; neither f1a9dcc nor de30a34 is qualified here.

## Runtime source verification (coordinator only)

On Controller emit the stdlib-only script, then in the coordinator's RESERVED window pipe it into each running
experiment container (use its actual container name, not an assumed one):

```bash
source ~/.config/ndm/paths.sh
"$HARNESS_PYTHON" offload/upstream_adopt/verify_runtime.py --emit > /opt/dsv41/campaigns/DSV41-OFFLOAD/UP1-runtime-check.py
# Coordinator, both ranks, after starting the selected arm:
# ssh HOST 'docker exec -i EXPERIMENT_CONTAINER python3 -' < /opt/dsv41/campaigns/DSV41-OFFLOAD/UP1-runtime-check.py
```

The checker hashes actual
mounted engine files and proxy package; it never imports torch/CUDA. Check process launch
environment and JIT module hash/logs separately. Runtime wrong bytes are a hard error.
Starting configs and all hardware work are EXCLUSIVELY coordinator actions; exact start/rollback commands and
ordered pass lines are in GB10-TEST-PLAN.md. Nothing here has been staged on either Spark.

## UP2 finish candidate (CPU only, default OFF)

`SUCCESSOR.md` describes the positive scheduler reservation and recovery.
`CONTRACT-0075-DRAFT.md` proposes the suffix amendment without changing the wire.
The rebuilt package has 42 files: the additional inert `suffix_provenance.py`
helper supports optional `VLLM_DSV41_REPLAY_AUDIT=1` qualification on BOTH ranks.
That audit compares actual engine metadata and owner epochs before the remote
round; normal STEP/reply generation checks remain authoritative. It synchronizes
metadata, so turn it OFF in headline timing controls and replay arms. With replay
ON, RoundCore refuses drafting over an incomplete real suffix even after resync.
No GPU, Spark, live worker or production equivalence result is claimed by UP2.

For the isolated 0075 window use [UP2-GB10-TEST-PLAN.md](UP2-GB10-TEST-PLAN.md)
and the default-off `UP2-0075.env` wrapper; no hardware commands were run.
