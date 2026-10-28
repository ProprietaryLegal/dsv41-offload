# Coordinator review and draft generation

This is offline preparation for `d2pass2-r1`, not authority to deploy or collect
from a lane. Read [the evidence map](PASS2-ATTEST-EVIDENCE.md) and
[the assembly schema](ASSEMBLE.md#second-pass-d2pass2-r1-context-evidence-and-coordinator-command).
Only Deepseek/coordinator runs `pass2_collect.sh`; the authoring lane has never
run it, including dry mode. No running generator, W12 files or services change.

## Collection (coordinator only)

From a reviewed checkout, source `~/.config/ndm/paths.sh`. `pass2_collect.sh`
with no arguments prints the read plan without accessing hosts or creating files.
For actual collection, the coordinator sets `D2_COORDINATOR=Deepseek/coordinator`
and passes `--execute` followed by an absolute NEW directory under
`/opt/dsv41/campaigns/DSV41-OFFLOAD/`, outside W12 and scratch. Its parent
must already exist. The script refuses identity/path errors and busy or missing
metrics. It preserves stderr and nonzero exits. An interrupted/busy collection
has no COLLECTED.json success receipt; retain partial metadata for diagnosis and
use another NEW directory for a later attempt.

The fixed reads are `/v1/models`, `docker inspect dsv41-serve` and small allowlisted
manifest/config/tokenizer/renderer/protocol/version files on both ranks. Above
1 MiB it hashes with `dd iflag=direct`; above 32 MiB it refuses. It copies text
only for files at most 1 MiB. There are no weight reads, model requests, tokenizer
requests, GPU imports, x86host operations or remote writes. Idle checks before and
after each bounded operation cannot rule out arrivals between checks. Abort on
any activity; do not claim a quiescent interval from snapshots alone.

Inspect both ranks' image IDs, mounts, command and environment. Bind the model
path and actual candidate/quantization manifests, the engine revision **and**
overlays, generation-config path/overrides, tokenizer mode and renderer sources.
If launch arguments name another config or template outside the fixed allowlist,
retain the gap and prepare another reviewed light-read procedure; do not guess
or expand the collector's paths from untrusted metadata. Manifest hashes identify
manifest bytes, not newly verified weight bytes. Existing S2 evidence has its
stated scope, which does not cover every legal target expert.

## Resolve claims explicitly

The evidence map has one JSON block, `d2.pass2-evidence-map.v1`, whose `claims`
keys are the complete required field set. The generator refuses extra/missing
keys. `verification.*` keys are additional semantic obligations; they do not
become assembly schema fields. Every resolved claim must have:

```json
{
  "value": 0,
  "status": "MEASURED",
  "scope": "Coordinator review: effective top_k after checking request, server generation config and pinned protocol precedence; cite the actual scope and limitations here.",
  "sources": [
    {
      "path": "/absolute/private/reviewed-values.json",
      "sha256": "REPLACE_WITH_ACTUAL_FILE_SHA256",
      "pointer": "/values/sampling.top_k"
    },
    {
      "path": "/absolute/private/sparkhead-files.json",
      "sha256": "REPLACE_WITH_ACTUAL_FILE_SHA256"
    }
  ]
}
```

This is a format example, not production evidence. `reviewed-values.json` is a
coordinator-written semantic review containing a `values` object keyed by the
same flat field names. A pointer may instead reference a value in a genuine
primary JSON receipt, for example an actual collected manifest digest. Every
pointer-bearing source must equal the claim value, every listed source must
match its full digest, and there must be at least two distinct source files.
Include all raw artifacts needed to support the review, not arbitrary second
files. The tool verifies binding and equality, **not** the truth of a human's
semantic conclusion. The coordinator's explicit approval is the trust boundary.
Do not mark a claim MEASURED merely because a file exists or its hash matches.

The resolution document is:

```json
{
  "schema": "d2.pass2-resolution.v1",
  "coordinator": "Deepseek/coordinator",
  "run": "d2pass2-r1",
  "approved": false,
  "claims": {}
}
```

Populate `claims` using exactly the map keys. Approve only after review. The
existing map's MEASURED values are immutable constraints; IMPLIED candidates
may be corrected with evidence. An attested null is a real `value: null` with
MEASURED evidence, not a missing value. For target manifests, select and explain
the actual production documents; never use the baseline or KV overlay manifest
as the legal target identity. `evidence` contains exactly the ten W12/generator
records in the map, excluding target-receipt.json, which the generator creates.
Do not change their original W12/source paths or hashes.

Resolve `verification.served_identity`, `verification.effective_defaults` and
`verification.renderer_raw_ids` to boolean true only with supporting reviewed
evidence. The collector cannot establish renderer equivalence on its own.
The pinned generator compares `/tokenize` to LocalRenderer when local checking
is enabled, then sends raw IDs to completions. A zero mismatch counter lacks
checking-enabled/coverage/identity proof by itself. The collector does not read
rate.jsonl or state; the coordinator supplies a separate scoped review of that
existing evidence if authorized. No new inference or /tokenize request is part
of this lane's procedure. Leave any unresolved obligation missing.

## Coordinator signature (NDM 2026-10-25)

For `d2pass2-r1` NDM decided ("Sign them") that the coordinator signs the open
configuration fields from the configuration of record rather than running
another collection. Only the fields in `pass2_attest.SIGNABLE` may carry
`"status": "COORDINATOR_ATTESTED"`: both `engine_revision` fields,
`sampling.{top_k,min_p,stop_token_ids,structured_outputs}`,
`formatting.{chat_template_kwargs,tools,tool_choice}` and the three
`verification.*` obligations. A signed claim keeps every other proof rule (scope,
two hash-verified sources, a JSON-pointer binding equal to the value, `true` for
an obligation). Integrity fields (target and derivative hashes, identity files,
evidence, generator source, run identity) are refused with that label. The label
is retained in the attestation's `claim_review`, so a reader can tell a signed
value from a measured one.

## Generate and review offline

Use copied local identity files and the actual supplied prompt manifest. Nothing
imports the pinned generator or encoder source. All output paths must be NEW and
outside production and W12; no existing output is overwritten.

```bash
source ~/.config/ndm/paths.sh
: "${MAP_SHA256:?independently recorded map digest}"
: "${RESOLUTION:?absolute approved resolution path}"
: "${RESOLUTION_SHA256:?independently recorded resolution digest}"
: "${PROMPT_MANIFEST:?absolute local supplied manifest}"
: "${DRAFT_OUT:?NEW absolute offline draft directory}"
PYTHONPATH=offload/dspark "$HARNESS_PYTHON" -m d2.pass2_attest \
  --evidence-map "$PWD/offload/dspark/d2/PASS2-ATTEST-EVIDENCE.md" \
  --map-sha256 "$MAP_SHA256" --resolution "$RESOLUTION" \
  --resolution-sha256 "$RESOLUTION_SHA256" \
  --prompt-manifest "$PROMPT_MANIFEST" --out "$DRAFT_OUT"
```

The tool verifies all source hashes and JSON pointer values, measured-map
constraints, target/derivative agreement and every unchanged pass2_context check.
Its temporary context staging is local and removed. Failed validation removes
only its newly created output directory. Success writes target-receipt.json,
pass2-attestation.json and DRAFT.json (private modes via umask 077); the attestation
also retains the hash-bound field review and source references. No production
receipt is generated by the authoring lane. Review the draft and independently
record its digest before the separate ASSEMBLE.md staging procedure. DRAFT is
not approval to copy payloads, assemble production data or run a pilot. The
independently recorded copy-time nonce remains separately required.

## CPU verification

`tests/test_pass2_attest.py` creates synthetic evidence, checks refusals for
missing/inferred-only fields, wrong hashes, contradictory values, null misuse,
missing semantic approvals, source/identity/sampling defects, existing output,
symlinks and duplicate JSON keys. Its successful draft passes the real
`d2.pass2_context` CLI in a subprocess with inherited GIT_* values stripped. The existing context suite also runs real CPU G1(a)
using synthetic bytes. Neither test executes the collector or contacts a host.
