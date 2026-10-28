# D2 Phase 2a trainer (local only)

Current Amendment-6 native adapter route: see [PILOT.md](PILOT.md) and
[WORKER-EVAL.md](WORKER-EVAL.md). The candidate runs as a native base plus FP32
low-rank deltas, never merged/requantized into low-bit weights. The legacy
`package.py` merged-FP8 instructions below do not qualify this floating-adapter
profile. Pilot dev/heldout_legal selection populations are development data; a new,
reserved public TEST family set is required for the frozen final candidate.


Scope: frozen native 1017 routed FP4 experts, routers, target embedding/head and main_proj. Train r32/r64 LoRA on
wq_a/wq_b/wkv/wo_a/wo_b and shared w1/w2/w3, drafter norms/main_norm, Markov head, then confidence head. No expert
LoRA or full-dense Phase 2b. Nothing here deploys, serves, sends mail or accesses a Spark.

## D2-TRAIN-PERF: separate fast training candidate

`--reference-math` remains the qualified evaluation/packaging arithmetic. An explicit `--fast-training` backend
adds GPU sorted routing, register-decoded packed expert tensor-core forward/dX, exact fused QDQ/STE, BF16 dense
products with FP32 output, GPU FP32 LoRA/norms/Markov, chunked analytic loss backward and fused CUDA AdamW.
The frozen vocabulary head uses high/low BF16 activation parts with FP32 outputs; its approximation is included in
all fast gates. No decoded expert bank is stored. Existing inference fusedmoe supports only T<=20, so this large
batch trainer uses separate kernels. Native worker/service dispatch and all G2/packaging gates are unchanged.

**The arithmetic agreement candidate failed. Amendment 3 authorizes a separate matched outcome pilot**
(see [PILOT.md](PILOT.md)); it does not turn those agreement failures into passes. See
`results/fast-agreement-v1.json` and the campaign's D2-TRAIN-PERF report for the final candidate. The production CLI
requires the existing reference parity receipt and current amendment-7b binding **dev** G1 PASS plus
`--pilot-receipt` with a PASSING real matched outcome pilot. It rechecks the exact Amendment-3 rule, frozen/source
identity, manifest, the once-only Amendment-4 sample hash/key coverage, all checkpoint hashes and B128/chunk8192/r32 coverage. Old `--fast-agreement` receipts cannot
authorize production. Resume binds the backend, numerical source and pilot receipt digest. A reference checkpoint cannot silently resume through the fast backend.

Gradient reporting includes every tensor separately and every adapter as its A+B group. At zero-initialized LoRA B,
A's gradient is exactly zero; its per-tensor zero/zero diagnostic is explicit, while the adapter group includes its
nonzero B direction. Zero-only groups fail. Every individual nonzero tensor must also meet cosine >=.999. Confidence
direction is exercised with an explicitly reported auxiliary MSE; distillation loss values use the unmodified
top20+tail objective. Agreement uses identical private inputs/seeds/targets with **no optimizer updates** on captures.
The current agreement runner evaluates B1; larger training batches are not qualified by those receipts.

Sustained synthetic measurements on GPU-240876ad at unchanged 225 W: B256/chunk8192 with reusable frozen main
projection, **204.02 blocks/s** over 200 measured steps, **19.66 GB reserved**; uncached B128/chunk8192,
**177.32 blocks/s**, **18.14 GB reserved**, also 200 steps. Five warmup steps are excluded. These project 3.27 and
3.76 compute-hours respectively for 2.4M blocks; dataset I/O and adaptation quality are unmeasured. B384's earlier
23.62 GB reserved screen and B512's OOM screen are rejected. The matched pilot alone may use both named solo cards under `run_pilot_locked.sh`; see PILOT.md.

Frozen projection reuse is explicit through `hidden_from(inputs)` with `inputs['main_projection']`: only the
frozen main_proj result is cached, and main_norm plus adapted KV are recomputed every step. Tests verify norm
gradients and changes survive reuse. The benchmark caches four repeated N1 windows; no full-corpus projection
cache/dataset integration is supplied or claimed. Raw training uses the existing D2 reader. The loss's FP32 logit
cache scales as batch*5*vocab*4 bytes; it replaces the larger chunk autograd/slice-gradient graphs.

On an immutable X86host snapshot, reproduce the sustained measurement under the existing lease supervisor:

```bash
source ~/.config/ndm/paths.sh
export D2_TRAIN_ROOT=$HOME/dsv41-offload/d2/perf/final-v2
bash "$D2_TRAIN_ROOT/repo/offload/dspark/d2/train/run_locked.sh" reproduce-b256 \
  -m offload.dspark.d2.train.benchmark --ckpt /opt/dsv41/x86host-data/native-1017 \
  --fixtures /opt/dsv41/x86host-data/d2/train/n1-solo225 --batch 256 --chunk 8192 \
  --warmup 5 --steps 200 --cache-projection --out /opt/dsv41/x86host-data/d2/perf/reproduce-b256.json
# Uncached: omit --cache-projection and use --batch 128 and a different output/log name.
```

`perf` records synchronized exclusive phase times; optional `--operator-profile` supplies non-additive host-copy
operator evidence. `diagnose` localizes N1 stage error against qualified fixtures. `agreement prepare` fixes the four
N1 plus the exact previously qualified 50 capture inputs; `reference` and `fast` run separately, then `compare`
computes draft agreement, FP64 log-softmax KL at reference previous tokens, loss difference, and gradient cosines.
Private logits/gradients/fixtures stay outside Git. Only aggregate performance/gate receipts may be committed.

No torch.compile/CUDA graphs, new dependencies or downloads were used. Existing Triton JIT compilation is the only
build activity. Full source/log paths and conditional coordinator commands are in D2-TRAIN-PERF.report.md.

## Qualified arithmetic and current limits

`--reference-math` keeps quantized matrix products on the designated GPU and runs the unquantized FP32 operations
on CPU in the same order as V1's pinned official model.py. Routed expert forward reads packed FP4 in registers and
uses FP64 sums, without materializing a matrix. Backward transiently decodes one selected matrix to BF16 for dX;
CUDA dX accumulates in FP32. LoRA is a floating adapter added before output rounding. QDQ has exact official forward
values and an explicit straight-through derivative. Stage checkpointing is stateless; every optimizer step
recomputes the conditioning and context KV from its actual taps and current weights. No adapted KV cache is reused.

The all-GPU FP32 path is an **unqualified experiment**, not the production training default. Tiny rounding changes
at a QDQ boundary amplify. An earlier GPU screen matched 45/50 blocks; even FP64 GEMMs with GPU unquantized math did
not pass strict exact-draft parity. Preserving the official FP32 grouped wo_a sum and using reference-math passed
50/50 captures with maximum logit relative error 4.43e-7. Full-weight CPU oracle matched a captured round bitwise at
every stage and in logits. These results do not establish useful pilot throughput. See D2-TRAIN.report.md for the
latest measurements and unresolved performance work.

Qualified reference CLI training requires a passing D2-DATA `--g1-receipt` (current amendment-7b binding dev tripwire PASS with VALID live reference), `--reference-math`, and a passing unadapted 50-round `--parity-receipt` whose frozen
weight identity matches the loaded source. A-capture is eval-only. Never use its inputs for a training smoke test.
`probe` trains only synthetic N1 benchmark taps and a synthetic target distribution, then discards the weights.

## Dependencies, sources and tests

X86host's existing environment is pinned in `requirements-x86host.lock`; this lane installed/downloaded nothing.
Controller CPU tests use `$HARNESS_PYTHON` (Torch 2.12.1); X86host qualification uses the existing Torch 2.10/cu126 venv.
Reference tests require the V1 worktree and the already present official code, via `D2_V1_ROOT` and `D2_OFFICIAL`.
The D2 interface test reads D2-DATA's `shards.py` via `D2_DATA_READER`. These integration tests fail if their required
sources are absent; they are never silently skipped.

```bash
source ~/.config/ndm/paths.sh
OMP_NUM_THREADS=2 "$HARNESS_PYTHON" -m pytest -q offload/dspark/d2/train/tests
"$HARNESS_PYTHON" -m mypy --strict --explicit-package-bases --follow-imports=silent \
  offload/dspark/d2/train offload/dspark/spark/tools/identity.py
```

Stage **immutable** source snapshots under `~/dsv41-offload/d2/train/<snapshot>/repo/` on X86host. Include d2/train,
native/{formats.py,official.py,NATIVE_MANIFEST.json}, and spark/tools/identity.py. Do not replace code or a shell
wrapper while a queued/running job is using it. Set `D2_TRAIN_ROOT` to that snapshot and `D2_GIT_SHA` to the source
commit. Dumps additionally record per-file source hashes. N1 fixture/probe helpers read the existing N2 source
snapshot via `--n1-repo`; no worker or service is started.

## GPU discipline

Coordinator reassignment 2026-10-23 ~14:05Z supersedes the brief's original shared-card rules.
`run_locked.sh NAME -m ...` runs from detached tmux, pins exactly
`GPU-240876ad-ea42-47de-1b8d-f6153810b7e1` on **X86host**, and acquires
`~/dsv41-offload/locks/gpu-240876ad.lock`. The card is dedicated to D2-TRAIN, with no 14-minute limit and no
TRAIN_OK requirement. Release it when idle. Its coordinator-set cap is **225 W**; never change power or clocks.
Never run D2 work on 7744b44b (production) or c4e7a371. Amendment-3 pilot work may also use bc62df2a under its own nonblocking lock through `run_pilot_locked.sh`. Old-card measurements are historical.

All jobs use the supervisor and validate the inherited exclusive lock. It requires the existing
`dsv41-powerlog` session and records this card's clocks/power/temperature every 5 s. On TERM/INT, missing logger,
>128 GiB child RSS or any swap, it signals only its child process group and allows 45 s for checkpoint/exit.
The loop checks cancellation at matrix boundaries, discards the partial optimizer step, restores its boundary
RNG state, and saves. Checkpoints every 300 s and on termination contain trainables, optimizer/schedule, data
cursor/order seed, precision/base identities, RNG and curve. Saves are private and atomic/fsynced. Each saved
boundary also has an immutable hardlink in `<output-stem>.checkpoints/step-NNNNNNNNN.pt`; use those versions for
dev selection/replay, since the latest pointer advances. Resume refuses changed contracts or a fresh run that
would overwrite an existing checkpoint namespace.

## Data, loss and training

The normative schema is D2-DATA's `offload/dspark/d2/DATASET.md` (`d2.manifest.v1` / `d2.shard.v1`). `D2Blocks` verifies
all shard/sidecar hashes and split/source-document separation. It uses taps at max(0,P-127)..P, anchor P+1, targets
P+2..P+6 and previous tokens P+1..P+5. Sequence-end targets are masked, not invented. Batches left-pad context with
masked slots. Deterministic whole-block permutations are derived from seed+epoch. Training never reads held-out
splits for selection; source documents and held-out chronology/motion_outline families cannot enter train/dev.

KL is forward KL to top20 plus the **actual complementary tail**, computed with streamed logsumexp, not unstable
`1-sum(q_top)`. Each vocabulary chunk is recomputed in backward. Position weights are 1,.8,.65,.5,.4. Hard-label CE
at .1 is used only for missing top20 positions. `--divergence tvd` selects the bucket TVD surrogate. AdamW with cosine
schedule updates only the allowed trainables. `--stage confidence` freezes everything except the confidence
projection, regresses the known target probability at teacher-forced drafter argmax, and skips unlabelled blocks
when argmax is outside top20; it never assigns an unknown probability zero.

After G1 and the genuine second-pass matched pilot PASS, use
[FULL-RUN.md](FULL-RUN.md) for the current full FAST and full reference G2
procedure. The MANIFEST is `/opt/dsv41/x86host-data/d2/decode-d2pass2-r1-c2/shards/MANIFEST.json`;
G1 is `/opt/dsv41/x86host-data/d2/g1-d2pass2-r1/dev.json`; the pilot receipt is
`/opt/dsv41/x86host-data/d2/pilot/decode-d2pass2-r1/receipt.json`. Preserve the
original immutable pilot source. Full training has its own snapshot/output and
uses three epochs, B128/chunk8192/r32, with initialization seeded before load.
The full stager uses the same two permitted 225 W solo leases as the pilot.
Synthetic loop tests remain CPU-only and never qualify production FAST.

## V1 replay and evaluation

`replay` reads V1 captures by default. For D2 use `--captures <MANIFEST.json> --data-reader <D2-DATA/shards.py>
--split dev|heldout_legal|heldout_general|acap`. `--references` names the V1 dump root with ref/mirror. An explicit K2
mapping may be supplied as `--k2-mapping`: JSON `{arm:"k2",split:<split>,manifest_sha256:<hash>,drafts:{<key>:[5 ids]}}`.
The actual D2-DATA `g1.py k2` output `<split>.k2_drafts.jsonl` is accepted directly too, with exact request/position
coverage validation. Its format has no embedded input-manifest hash: supply only the coordinator's output for that
exact shard set. Receipts record both the JSONL digest and target-manifest digest; the import does not invent a
prior cryptographic binding absent from the producer's format.
Placeholders are never a K2 baseline. A supplied adapter is a diagnostic comparison unless `--parity-gate` is set.
Use `--limit 0` for the full set; the default 50 is the pre-training parity screen spread across all requests.

`reference --tool ref|analyze|classify --v1-root <V1/offload/dspark> [D2 options] -- <V1 arguments>` runs V1's unchanged
modules through a process-local bridge. For D2 add `--data-manifest`, `--data-reader`, `--split`, and actual K2 mapping
for analyze. Taps-only D2 windows do not fabricate an engine main_x comparison. For a packaged candidate add
`--package-root`: every receipt file/hash and aggregate identity is verified before allowing new checkpoint hashes;
the pinned official model.py hash remains mandatory. Run ref and mirror in separate processes, then analyze and
classify. No V1 source file is edited. Classify output records its candidate-reference identity.

`evaluate score` consumes candidate, **native unadapted baseline**, and explicit **K2** V1 dump directories plus the
manifest and adapter. `--k2` also accepts the actual G1 JSONL directly. It recomputes teacher-forced logits with each arm's own Markov head and top20+tail KL, and
scores complete five-target blocks on K2's fixed round trajectory (P starts at gen_start-1 and advances by K2's
realized match length+1). Candidate predictions never choose the scored population. The committed-trajectory proxy is paired against K2,
with a request-clustered bootstrap. Missing/mismatched files or placeholder provenance fail loudly.

The historical `evaluate g2 --acap` route predates amendment 7 and
cannot evaluate the second-pass population. The current full evaluator is
`full_eval score` / `full_eval merge`, staged by `full_stage` in FULL-RUN.md.
It freezes the completed three-epoch FAST checkpoint (no heldout selection),
uses full dev and heldout_legal populations, accepted length against explicit
K2 and teacher KL against unadapted reference, and gates heldout_general as D2-TRAIN G2 does.
A full development-set G2 PASS does not establish freshness: the reserved
public TEST families and real-worker/GB10 G3 remain separate requirements.
No existing TEST-ONCE ledger is bypassed or repurposed.

## Offline packaging (Phase 3 preparation only)

`package --source <native-1017> --output <new-sibling> --adapter <selected.pt> --counts <private-counts.json>
--native-manifest <pinned-NATIVE_MANIFEST.json>` creates a new hardlink tree beside the source, merges only the
allowed adapters, quantizes changed dense tensors to E4M3FN/UE8M0 block32, rounds plain stored tensors back to their
official storage dtype, regenerates hot-token IDs and writes a fresh native manifest/receipt. Every replaced file
uses a new inode. Source files, frozen experts/routers and target embedding/head remain byte-identical. The adapter
must identify the exact frozen source manifest. Counts are previous-token frequencies on train only (no held-out
counts); this file, the Markov JSON, candidate and checkpoint are proprietary.
Use `--data-manifest <MANIFEST.json>` instead of `--counts` to derive the actual teacher-forced input counts from
train only, including reuse across overlapping five-target blocks.

`identity.py` now accepts an explicit native source-file list; its existing default EXL3 file list/digest algorithm
is unchanged. Native identities cover config, index, native manifest and actual source shards; the Markov file has
its own digest. Server identity integration remains the coordinator's Phase 3 work; do not advertise a native
candidate using the old EXL3 receipt/identity.

For post-quantization gain, run `replay --ckpt <new-sibling> --package-root <new-sibling> ...` without `--adapter`
or `--native-manifest`. It verifies the package and records both package and originating adapter identities.
Then `evaluate score ... --adapter <selected.pt> --package-root <new-sibling>` uses the package's actual stored
BF16 Markov table for teacher KL, refuses wrong package/adapter dumps, and gives the retained committed-trajectory
gain on the same K2 rounds. Package test attempts have separate immutable ledgers keyed by native package identity.
The dev-selected source adapter must match the package receipt. Compare the two summaries' gains with
`evaluate packaging-gate --gain-before ... --gain-after ...`.

Compare the merged/requantized arm against its **own** independent V1 FP32 candidate reference, using
`reference --package-root ...`. `evaluate packaging-gate --gain-before ... --gain-after ... --classification ...
--package-root ... --out ...` requires ≥75% gain retention, native representation, zero state flags/unexplained
cases, ≥50 classified rounds, matching candidate-reference identity, and every layer's p99 error below 3× its V1
arithmetic band. Promotion still needs the coordinator's E-protocol live G3 A/B and depth recalibration.

EXL3/K2 re-encoding is assessed but not built: the engine-specific dense encoding/calibration and identity path is
separate, and no validated adapter merger for it was supplied. Native FP8 is the primary deliverable; no expert
requantization, service change, or in-place replacement is needed for Phase 2a.
