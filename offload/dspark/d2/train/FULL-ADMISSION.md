# Full-training admission audit: pilot 2

The full runner supports only the reviewed pilot-2 source at
`d1dd41a2744fd78cc9b74f83a98af1765e9cb812`. Amendment 8 preregisters `ace`.
An actual nonsynthetic PASS from that snapshot remains mandatory. No PASS is
claimed by this lane; synthetic CPU fixtures establish contracts only.
The first pilot's actual FAIL and its b6f79e9 source lineage are refused.

`full_admission.PILOT_SOURCES` pins the complete original `pilot.sources()`
map, including loop/loss/fast_loss/pilot* and the full_* files already present
in pilot 2. Admission requires exact map equality, the exact original trainer
Python file set, and every immutable source byte. Package initializers and
Python files at import roots are separately pinned; unbound startup hooks,
changed/missing/extra trainer files and outside-tree symlinks refuse.
Native package initialization is pinned too. Compiled import artifacts and
stdlib/third-party import shadows refuse; -B also prevents creating bytecode.
Every current shared file must still match pilot 2. Only loop.py and the
explicit full bridge modules may differ: FULL_LOOP and FULL_PATCHES pin the
reviewed candidate bytes. full_admission.py itself is the admission policy;
it is the sole self-pin exception, not a blanket full_* exemption.

The bridge invokes that snapshot's **unchanged** `require_pilot` in an isolated
CPU child (`python -I -B`, explicit verified source paths, CUDA hidden). It
recomputes PASS, intervals, population/evaluation/sample identities, matched
units, every arm's recipe and every checkpoint hash. Then that snapshot's
unchanged `verify_sample_draw` redraws both sample populations from the exact
K2 files. Sample keys, populations, hashes, seed and limit must all match.
The parent rechecks the receipt and source after the child. No map, original
source, plan, sample, receipt or verdict implementation is modified.

Both K2 sources travel in full_stage commands as --k2-dev / --k2-legal. Direct
full-train/eval calls without these paths use dev.k2_drafts.jsonl and
heldout_legal.k2_drafts.jsonl beside the G1 receipt; absent files refuse.
Omitting --pilot-source refuses. FAIL, INCONCLUSIVE, synthetic, old-rule,
forged PASS, missing/borrowed recipe or changed evidence cannot authorize FAST.
The original G1/parity/base/shard/shape checks remain mandatory.

The recipe is read only from the admitted receipt, with an explicit recipe
field and exact pilot_run(recipe) contract. There is no full recipe override.
Full FAST uses that Run with only epochs changed to three. ACE/TVD retain
2e-4; LR retains 5e-5 and 10% warm-up then cosine. The schedule uses the actual
full count, 3*ceil(len(train)/128), never the pilot's literal 159 rates.
CLI objective/epoch/shape changes and an initial adapter refuse before load.
After loading the native base, initialization must equal the pilot's seed23
zero-adapter identity. Pilot checkpoints cannot seed or resume this run.

Checkpoints bind full_recipe, the complete Run, the admitted receipt hash,
backend, numerical source map, dataset, base and optimizer/scheduler/RNG.
Resume refuses a missing/changed recipe or any changed run contract. G2
requires an exactly completed three-epoch qualified checkpoint with the same
recipe. Its request journal, three part identities and merged receipt bind
that recipe. Reference KL/acceptance scoring, bootstrap, population rules and
all gates remain unchanged; ACE training does not replace G2's KL metric.

For pilot-2 ACE, use the exact full_args and commands in FULL-RUN.md:
- original snapshot: ~/dsv41-offload/d2/pilot/decode-d2pass2-r1-p2-ace/repo;
- original receipt: /opt/dsv41/x86host-data/d2/pilot/decode-d2pass2-r1-p2-ace/receipt.json;
- NEW full snapshot/output basename: decode-d2pass2-r1-p2-ace-full.
Admission requires the full basename to equal the admitted pilot namespace
plus `-full`, and snapshot/output basenames to agree. STAGED.json binds the
new reviewed commit, input/receipt hashes, recipe and exact commands. Sessions
include the full basename. No pilot snapshot, output or session is reused.
Dry rendering is side-effect-free and reads no receipt; its commands derive
the recipe only when admitted on the coordinator host.

MEASURED historical read-only S36 copies bind MANIFEST SHA256
`450bb71653f3def68cd95ee9579a456c74c469d08a016940cceda24de48b4f30`,
amendment-7b binding dev G1 PASS and VALID live reference. They contain no
real tensor shards, K2 files or complete pilot-2 evidence. All real-data
admission and any eventual full launch remain coordinator-local prerequisites.
No current pilot outcome, GPU speed, adaptation, full G2, fresh TEST, G3,
worker qualification or deployment is established by CPU tests.
