# Original acap audit — D2-ACAP, 2026-10-25

**STOP: the located original artifacts do not supply a provenance-qualified
`d2.shard.v1` acap split for the second-pass pilot.** No joining implementation,
G1(b) execution script or end-to-end admission rehearsal was built. The binding
lane brief requires stopping if the original exports fail provenance; no gaps
were filled with replay taps, reconstructed features or new generations.

This finding does not invalidate the original captures for their existing V1
reference comparisons. They are genuine but incomplete for the requested D2
shard interface. No G1/G2 rule or held-out population was changed.

## Located artifacts and identity (MEASURED, CPU/read-only)

Token exports: Controller `/opt/dsv41/data/d2/acap/acap{0..3}.json`;
identical x86host copies `/opt/dsv41/x86host-data/d2/acap/acap{0..3}.json`.

| Export | Bytes | SHA256 |
|---|---:|---|
| acap0.json | 47251 | 7a4d5f7fbf7e2628e0b695ee16d1583d906ee6e679489d92282d9f0fcc18f6d4 |
| acap1.json | 46669 | 1bddc72146ac004e4bb6cdd248b1514c3e96b4a90c042e63e0d22e424207fbfe |
| acap2.json | 47048 | c6ee5dbd4944a920867e10f895cb35ec9d9ccb450b6b68e77e5dfb63915645ef |
| acap3.json | 3356 | 82344da14d7040cbe4e88eda3a9ee2d8fb331c10216c2dc9fd225ac27e5b347e |

These are `d2.acap.v1` JSON exports from `equiv.cmd_acap_export`, containing
only schema, seq_id, capture_req, tokens, gen_start, prompt_sha256,
tokens_sha256 and decode_tap_rows. They contain no taps or teacher probabilities.
All four complete int32 token hashes revalidate; every committed token matches
the corresponding capture request and local V1 `d1-exact/reqN.request.json`.

Original round tensors:

- Controller `/opt/dsv41/data/v1/captures/A-capture-20261022/round*.pt`.
- X86host `/opt/dsv41/x86host-data/dspark/captures/A-capture-20261022/dspark-captures/round*.pt`.
- 453 files, 363376011 bytes. Every basename and SHA256 agrees across hosts.
- Inventory digest: `0e1cb80d14df15ec458a0359146e841ccf5d9f30e8aa2946948c4dd6b5895978`.
  It hashes UTF-8 concatenated `SHA256  basename\n` lines sorted by basename.
  Complete per-file hashes are retained outside git in the campaign's
  `D2-ACAP.capture-SHA256SUMS`.

## Why the existing files cannot be joined

The actual V1 `verify.captures.CaptureSet` was CPU-loaded on Controller: zero
plumbing errors, 542 samples. DATASET.md sections 3, 6 and 8 require native
BF16 taps for every position `max(0, gen_start-128)..L-1` and target top-20
probabilities/committed-token probabilities across the response span.

| Sequence | L | gen_start | Required tap rows | Native tap rows present | Missing required taps | Complete tap-window samples |
|---|---:|---:|---:|---:|---:|---:|
| acap0 | 8402 | 8000 | 530 | 402 | 128 | 109 |
| acap1 | 8300 | 8000 | 428 | 273 | 155 | 62 |
| acap2 | 8299 | 7999 | 428 | 300 | 128 | 59 |
| acap3 | 523 | 124 | 523 | 399 | 124 | 142 |

Total missing required taps: 535. The original overlay
`capture/speculator.py` stores raw taps only for steps with <=64 tokens;
large prefill steps retain engine `main_x` but omit raw taps. The V1 adapter
explicitly supports this mixed representation. D2's shard tensor schema requires
three concatenated 5120-wide taps; engine `main_x` cannot substitute for them.
The 372 complete-window samples are a subset, not permission to alter the
binding population or fill missing rows. Round records do not store teacher
logprobs/top-20/residual mass. Draft confidence is not that distribution.

Provenance is independently incomplete. X86host's
`d2/provenance/sequence-context.INCOMPLETE.json` has SHA256
`b8aeaef46e9e521692b922cafdbc282a751480bbef7045f519466edc8f7a58a0`;
its status and all four per_sequence acap statuses are
`INCOMPLETE_DO_NOT_BUILD`. Target quantization, engine_revision,
weights_manifest_sha256, quantization_manifest_sha256 and receipt_sha256
are null. The directory listing contains no final sequence-context.json.
`evidence.json` SHA256
`7a4297bc331812e0766c8f92984ef05ed9a48c6dd4d9dca7ea70992e564b69d2`
explicitly lists missing original acap effective defaults/processors and
target/runtime evidence, and a missing coordinator-verified final context.

`ACAP-SAMPLING.md` establishes recorded T=1/top_p=1, high thinking, seed42,
priority10, and explicitly says that evidence is not a complete context receipt.
The D2-REPLAY-PREP report likewise describes its scaffold as incomplete.
DATASET.md's final paragraph prohibits relabeling failed replay taps as acap.
The existing replay tree is consequently not a source of replacement native taps.

The historical x86host `d2/g1-precheck/k2_captures.json` exists (SHA256
`644673d06bfc45e113eb30585a9594e334ea74e6f3f891d76bf9bcf19760d758`).
It reports 372 blocks, 364 scored, proxy_worker=1.4945 and proxy_engine=1.5247;
it is a V1 worker/engine precheck, not a binding acap G1(b) PASS with exact
shard/K2 pairing. It contains neither the required split/binding/pass fields
nor a complete pilot population. No PASS was inferred from it.

## Coordinator disposition

Keep pilot admission blocked. Locate an independently existing complete original
native export plus verified context and teacher evidence, if one exists beyond
the documented locations. The inspected files and receipts do not establish
such an export. No claim is made about uninspected storage. New collection,
representation/schema changes or changes to the binding evaluation population
would require a separate coordinator decision; this lane implements none.

`train/pilot_stage.py:check_inputs` still requires original acap in the manifest,
a genuine binding acap G1(b) PASS, complete K2 files and exact score recomputation.
The mandatory bba0795 integration was a fast-forward from 65ce31d. No production
code changed. Conditional brief items 2–4 were not started after this stop finding.
