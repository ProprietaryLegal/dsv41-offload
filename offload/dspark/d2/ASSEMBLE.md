# Frozen decode assembly (CPU only, coordinator procedure)

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

This procedure is prepared for x86host and has only been rehearsed with synthetic
CPU data on Controller. It performs no SSH, HTTP, GPU work or service changes. It
never controls the capture writer or generator. Run it locally on x86host only
after transition 2 has finished and published `COPY_RECEIPT.json`. No pilot is
started or cleared by this procedure.

The existing interpreter is `$HOME/dsv41-offload/o2/venv/bin/python`, as used by
`stage_replay_inputs.sh` and `g1.py`; `train/requirements-x86host.lock` records the
existing dependencies. This is repository evidence, not a current host probe.
No dependencies were installed for this lane. The script pins CPU visibility
and two BLAS/OpenMP threads. `D2_ASSEMBLY_PYTHON` is an explicit interpreter
override for an already reviewed environment.

## Inputs and layout

Use a NEW copied tree, for example `/opt/dsv41/x86host-data/d2/decode-RUN/`:

```text
COPY_RECEIPT.json          # decode_copy v1; additive fields allowed
capture/                  # exact copied FLUSH*, headers, ledgers, taps, IDs
generator/LOCK            # exact copied frozen generator, never live state
generator/seqs/*.json
top20/                    # optional exact copied export, if present
inputs/R6-smoke.json       # copy of approved real image-bound option-ii smoke
inputs/prompt-manifest.json
inputs/sequence-context.json
inputs/evidence/...        # every hash-bound context evidence receipt
inputs/identity/...        # hashed tokenizer, encoder and chat-template files
```

The coordinator stages `inputs/` as copies and independently checks their hashes
against their approved originals. Keep the original smoke hash for the required
`--smoke-sha256` argument. Do not rewrite a JSON path or hash to make it work.
The real image-bound smoke and completions validator gates stay mandatory.

`assemble.sh` always exports a fresh `export/` from `generator/`; an optional
copied `top20/` is still fully checked against the copy receipt. For existing
copied top20 exports, `decode_dataset --assembly-root ... --path-map ...` also
supports their absolute `src` and `VERIFIED.json` smoke references without
rewriting either document. Map the old export root to `top20` and the old smoke
file to its copied `inputs/` file. All destinations must already exist.

## Dry command and execution

From the reviewed checkout on x86host (replace RUN, original paths and SHA256):

```bash
ROOT=/opt/dsv41/x86host-data/d2/decode-RUN
bash offload/dspark/d2/assemble.sh \
  --root "$ROOT" --expect-nonce "$COPY_TIME_NONCE" --image-smoke "$ROOT/inputs/R6-smoke.json" \
  --smoke-sha256 APPROVED_ORIGINAL_SHA256 \
  --prompt-manifest "$ROOT/inputs/prompt-manifest.json" \
  --sequence-context "$ROOT/inputs/sequence-context.json" \
  --path-map /ORIGINAL/evidence/root=inputs/evidence \
  --path-map /ORIGINAL/identity/root=inputs/identity
```

Dry is the default: prints the quoted command and order without reading inputs
or invoking Python. To execute, prepend `D2_COORDINATOR=Deepseek/coordinator`
and place `--execute` immediately after `assemble.sh`. Other identities refuse.
There is no automatic remote invocation. This lane has not executed that command
against x86host or any production input.

Order: verify every copy-receipt file -> export -> prepare -> shard build ->
G1(a) with stride 1 -> S25 probability aggregate -> `ASSEMBLED.json` last.
An explicit OLD_ABSOLUTE=ROOT_RELATIVE map resolves immutable identity paths
inside this root, including context evidence receipts. Unmapped external paths,
symlinks, missing/moved files, hash differences and changed inputs refuse.
Prepare compares each exported sequence with its receipt-bound generator state.
`DECODE.json` records maps, original export-plan hash, new decode-plan hash and
copy-receipt hash. Shard manifests bind DECODE.json and the copy receipt.
The copied tree is fully hashed at assembly start, prepare, build and completion
(four passes), retaining every existing
decode row/identity/shard check. Failed outputs remain for inspection; use a NEW
copy/output root for a retry, never edit a receipt or reuse a `.partial` tree.

## Budgets (ESTIMATED, not x86host measurements)

From DECODE-CLOSE.md: 30,720 bytes/tap row + 4 bytes/ID. From OPTION-II.md:
172 bytes/generated position in normalized top20 arrays; saved JSON roughly
0.7–1.3 GB per million generated positions. Let R be captured rows (including
prompt tails) and N generated positions; use actual receipt sizes when available.

| storage | estimate for R=N=1M |
|---|---:|
| copied taps + IDs | 30.724 GB |
| copied generator JSON | 0.7–1.3 GB |
| new frozen export state + arrays | 0.872–1.472 GB |
| new shards (taps, IDs, positions, arrays) | about 30.90 GB |
| total, before ledger, headers, optional copied top20, inputs | about 63.2–64.4 GB |

Reserve at least 40 GB **additional free disk** after the copy per 1M rows,
plus measured ledger/input/optional-top20 sizes and prompt-tail rows; reserve
more for retained failed builds. This is why assembly belongs on x86host rather
than Controller's roughly 49 GB free disk. No additional full tap copy is made by
export; shards and captures coexist. SHA verification streams file contents.

ESTIMATED RAM: reserve 16 GiB available for this CPU job, leaving other workloads
their allocations. The shard tap limit is 2 GiB; SplitWriter buffers one shard
per populated split, and serialization concatenates another shard. Budget
roughly `(active splits + 1) * 2 GiB`, plus the largest sequence, Python/JSON
objects, hash ledgers, and torch runtime. Five populated splits can approach
12 GiB before that overhead; 16 GiB is a planning floor, not a guaranteed bound.
The capture verifier reads one sequence's taps and ledger at a time; extremely
long sequences and JSON expansion can require more. Inspect available memory
and actual sequence sizes locally before executing; do not start when headroom
is uncertain. No live-host capacity claim is made here.

## Handoff and limits

Shards remain at `$ROOT/shards/` with `MANIFEST.json` and `SHA256SUMS`.
`G1A.log`, `AGGREGATE.json` and `ASSEMBLED.json` record CPU checks and hashes.
`g1a_pass` means structural plumbing only. Synthetic rehearsals remain labeled
synthetic and cannot supply a real smoke or pilot authorization.

Roadmap row 11 still requires real-data G1, source-disjoint splits and genuine
provenance; row 12 requires the independently cleared matched-pilot harness;
row 13 consumes these exact G1-approved shards for REF/FAST/FAST-B under the
binding amendments in `documents/plans/2026-10-23-d2-plan.md` (including amendment
6's later decision rule). Original acap held-out data and G1(b) require their
separate existing verified provenance and coordinator workflow; this script
does not relabel acap, fabricate missing splits or turn G1(a) into full G1.
No GPU training or pilot command is part of assembly.

## Second pass d2pass2-r1: context evidence and coordinator command

`pass2_context` is an offline staging tool, not an attestation writer. W12 proves
that the reviewed decode-capture derivative was served, including both ranks'
mounted source hashes and the pinned generator source hashes. It does **not**
establish the weights/quantization manifest hashes, effective sampling defaults,
or the served tokenizer/renderer identity. The INCOMPLETE replay-prep scaffold
cannot establish those facts. Do not fill its nulls with guesses.

The coordinator must provide a reviewed `d2.pass2-attestation.v1` JSON and record
its SHA256 independently. Required fields:

* `coordinator`: `Deepseek/coordinator`; `run`: `d2pass2-r1`.
* `target`: `model_root`, `quantization`, `engine_revision`,
  `weights_manifest_sha256`, `quantization_manifest_sha256`. A separately supplied
  `target-receipt.json` must contain exactly this object. Its hashes must come
  from actual production manifest evidence, not the served alias.
* `decode_capture_target`: those same production identity fields, plus `run`,
  `review_sha256` (the W12 review file digest), and `capture_env_sha256` (from
  that review). The attestation itself is this derivative's receipt; the tool
  records its real SHA256, never a made-up receipt value.
* `sampling`: explicit `temperature: 1.0`, `top_p: 1.0`, `top_k`, `min_p`,
  `structured_outputs`, `stop`, `stop_token_ids`, `seed`. Record effective
  defaults supported by source evidence; null must be an attested null/default,
  not a substitute for an unknown. `formatting`: `chat_template_kwargs` matching
  the prompt manifest, `tools`, `tool_choice`. The coordinator must verify the
  served renderer matches the supplied local renderer, including raw-ID requests.
* `generator`: `revision: "1b8fc1c"`, `priority: 20`, `top20: "generation"`,
  `source_sha256` mapping `generate.py`, `generation_top20.py`, `prompts.py` to
  their full digests. These files must match the W12 `GEN:` records. The tool
  reads the named source files only, never generator state or output.
* `identity`: `tokenizer`, `encoder`, `chat_template`, each an object with an
  absolute `path` and `sha256`. The tokenizer supplies the verified marker IDs.
  Encoder and chat template may be the same file; each gets an explicit map.
* `evidence`: objects with `label`, absolute `path`, `sha256`, exactly one each
  for `w12.log`, `base-hashes.txt`, `staged-hashes.txt`, `mounted-hashes.txt`,
  `rendered-d2pass2-r1.diff`, `review-d2pass2-r1.json`, `start.log`, the three
  generator source files, and `target-receipt.json`. Paths must name the actual
  W12 files and pinned clone sources recorded in its log. No directory scan,
  live-host lookup, source import or inference is performed.

The seven W12 deployment files are `W12/w12.log` and the six named files in
`W12/private/`. Generator sources are under
`W12/private/gen-checkout-1b8fc1c/offload/dspark/d2/`. Nothing in W12 is written.
Missing files, changed pinned hashes, unresolved claims and contradictory
base/derivative/mounted/generator identities refuse before context publication.

On Controller, stage to an existing **offline staging root** with an `inputs/`
directory. This root is separate from production and from the frozen payload.
Use the reviewed interpreter with `PYTHONPATH=offload/dspark`:

```bash
: "${STAGING_ROOT:?offline staging root with inputs directory}"
: "${ATTESTATION:?coordinator-reviewed absolute receipt path}"
: "${ATTESTATION_SHA256:?independently recorded receipt digest}"
: "${PROMPT_MANIFEST:?copied manifest under staging root}"
"$HARNESS_PYTHON" -m d2.pass2_context \
  --root "$STAGING_ROOT" \
  --attestation "$ATTESTATION" --attestation-sha256 "$ATTESTATION_SHA256" \
  --prompt-manifest "$PROMPT_MANIFEST"
```

It creates `inputs/context-inputs/{evidence,identity,sequence-context.json,STAGED.json}`.
Evidence bytes are copied and hashed, with original paths preserved in the
context. `STAGED.json` records exact per-file maps and copied file hashes. Transfer
these inputs through the coordinator's approved copy procedure and independently
compare their hashes on the assembly machine. The COMPLETE context has
`mode: decode_capture`, `target` and `decode_capture_target`; it has no replay
target. Served decode builds explicitly require this mode and hash-bound
receipt. The replay branch still requires both matching target/replay receipts.
A replay-mode context offered for decode data refuses.

The exact local assembly command for this run is below. Only the coordinator
runs it, after the frozen copy has completed. The nonce is a required input
recorded **at copy time**; reading it back from the candidate receipt and calling
that the expected nonce would defeat the check. No copy-time nonce was available
to this lane, so this command deliberately refuses if it has not been supplied.

```bash
ROOT=/opt/dsv41/x86host-data/d2/decode-d2pass2-r1-c2   # verified copy retry; the first attempt has no receipt
: "${COPY_TIME_NONCE:?nonce independently recorded at copy time}"
R6_SHA256=b288080cc44f45c5edb9f5b96e4caff9327cfd29b32558e9741bb9bf1b812dde
# Run in the existing reviewed x86host Python environment, locally on x86host.
PY="$HOME/dsv41-offload/o2/venv/bin/python"
D2_COORDINATOR=Deepseek/coordinator "$PY" - "$ROOT" "$COPY_TIME_NONCE" "$R6_SHA256" <<'PYRUN'
import hashlib, json, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
data = json.loads((root / 'inputs/context-inputs/STAGED.json').read_text())
for relative, expected in data['files'].items():
    path = root / relative
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('staged input differs: ' + relative)
args = ['bash', 'offload/dspark/d2/assemble.sh', '--execute',
        '--root', str(root), '--expect-nonce', sys.argv[2],
        '--image-smoke', str(root / 'inputs/R6-smoke.json'), '--smoke-sha256', sys.argv[3],
        '--prompt-manifest', str(root / 'inputs/prompt-manifest.json'),
        '--sequence-context', str(root / 'inputs/context-inputs/sequence-context.json'),
        '--quarantine-named-errors']
for source, destination in sorted(data['path_maps'].items()):
    args += ['--path-map', source + '=' + destination]
subprocess.run(args, check=True)
PYRUN
```

The quarantine decision in that command is explicit: accept healthy sequences
while excluding exactly the IDs attributed by `capture/ERRORS.jsonl`. Omit the
flag to require every sequence instead (default). IDs and the error-log hash
are recorded in DECODE.json, ASSEMBLED.json and the shard manifest. Unattributed
errors, global refusal, flush/error-count disagreement, and any other surviving
sequence's defect still refuse; `allow_skips` stays false. A run with no surviving
sequences refuses. This is not permission to edit the copied error log or receipt.

Maps must be narrow. A source prefix of `/`, the assembly root, or an ancestor
of that root is rejected up front. Relative paths undergo the same symlink
checks as absolute paths. Build pins DECODE.json captures to `$ROOT/capture`.
A G1(a) failure retains `G1A.log` with its report and publishes no ASSEMBLED.json.

CPU rehearsal: `tests/test_pass2_context.py::test_context_tool_to_real_g1a`
drives the tool's COMPLETE output through served-path provenance checks,
relocation, copy binding, real shards and genuine stride-1 G1(a), using only
synthetic bytes. Only the external image-smoke gate is stubbed. Refusal tests
cover missing evidence, changed evidence hashes and a replay-mode context
supplied for decode data. `test_named_quarantine.py` covers named-only exclusion,
default-off refusal, global errors, count drift and other integrity defects.
S34 probes retain their injected tampering; the four observation probes now
assert the hardened behavior instead of the previous accepted edge cases.
