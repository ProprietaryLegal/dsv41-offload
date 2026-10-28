# D2 drafter self-distillation data: shard schema `d2.shard.v1`

Written by lane D2-DATA (2026-10-23) before any data exists, so lane D2-TRAIN can build against it. D2-DATA writes the
shards (`d2/dataset.py build`); D2-TRAIN only reads them. A change to anything below bumps the schema version.

Everything in a shard is proprietary legal-v2a data (target outputs, taps, target probabilities): it stays on x86host
`/opt/dsv41/x86host-data/d2/` (directories mode 700) and is never committed or copied off x86host/Controller.

## 1. What a sequence is

One generation request of legal-v2a on production `:8000` at production settings (T=1, top_p=1, thinking on,
`reasoning_effort: high` through `chat_template_kwargs`, `priority: 5`). Its token ids:

```
tokens[0 : gen_start]      the prompt, rendered by the server's own chat template (POST /tokenize), BOS included
tokens[gen_start : L]      the generated span: thinking + answer exactly as sampled (EOS included when it ended on EOS)
```

A request aborted by the idle gate is resumed later from its exact token prefix; the concatenation is still an exact
T=1 sample of the target, so a sequence is one contiguous token list whatever the number of segments.

Every committed token in the span is a sample of p_target(. | tokens[:p]). That is the distribution the engine's
rejection sampler verifies drafts against at T=1, and what the drafter is trained toward.

## 2. Constants

| name | value | source |
|---|---|---|
| `W` (drafter window) | 128 | native config `sliding_window: 128`; the drafter's 128-slot window ring; V1 `captures.WINDOW` |
| `K` (block) | 5 | `dspark_block_size` |
| `NOISE` | 128799 | `dspark_noise_token_id` |
| `TOPK` | 20 | vLLM `prompt_logprobs: 20` (server `max_logprobs` 20) |
| tap layers | 37, 38, 39 | `dspark_target_layer_ids` |
| `H` | 5120 | hidden size |
| vocab | 129280 | |

## 3. Per sequence, what is stored

Let `r0 = max(0, gen_start - W)`.

| item | positions | meaning |
|---|---|---|
| `tokens` | 0 .. L-1 | all token ids (int32) |
| taps rows | r0 .. L-1 | for row p: the three taps the target computed while processing token p, i.e. the engine's `aux_hidden_states` (mean over the 4 mHC streams at the INPUT of target layers 37, 38, 39), BF16 exactly as the engine produced them |
| target top-20 | gen_start .. L-1 | for position p: the 20 most likely tokens of p_target(. \| tokens[:p]) with their natural-log probabilities, sorted by descending probability; the committed token's own logprob and rank; the residual mass outside the top-20 |

Where the taps come from (`taps_source` in the manifest):
* `prefill_replay` (planned): one prefill of the finished sequence on the replay serve (`A-replay.env`), rows saved by
  the replay overlay. Valid only if the equivalence check passes (`d2.equiv`, rule in FALLBACK.md section 1: FP32
  reference drafts on prefill taps vs decode taps agree on >= 99 % of the blocks whose inputs differ, or the
  BF16-noise rule against a same-run noise control holds). `equiv drafts` runs at `--every 1`: all 542 blocks, about
  90 min of CPU in 3 shards.
* `decode_capture` (fallback): rows captured during a capture-overlay generation run. Same layout. `dataset.py build`
  reads the source from the tree, not the label: a replay plan (request ids `d2r.*`) is `prefill_replay` and needs
  the equivalence gate; `decode_capture` is refused until FALLBACK.md option A's overlay records its mode.

The top-20 come from the same replay request (`prompt_logprobs: 20`, raw logprobs = log_softmax of the served logits,
the same distribution the rejection sampler uses at T=1).

## 4. Training blocks (normative)

A block is a pair (sequence, P) with `gen_start - 1 <= P <= L - 3`:

* window rows: positions `max(0, P - W + 1) .. P` (always inside `r0 .. L-1`); the drafter's window KV of row p comes
  from `main_x[p] = main_norm(main_proj(taps[p]))` (taps concatenated 37 | 38 | 39);
* anchor: `tokens[P + 1]` at position P+1; query block = [anchor, NOISE x 4] at positions P+1 .. P+5;
* draft i (i = 0..4) predicts the token at `t_i = P + 2 + i`. It has a target only when `t_i <= L - 1` (the last
  blocks of a sequence are partial: mask the rest). Its target distribution is the top-20 row of position `t_i`, its
  committed token is `tokens[t_i]`, and its teacher-forced Markov input (the previous token) is `tokens[t_i - 1]`
  (the anchor for i = 0).

This is the engine's own round layout (V1 `captures.py`: P = last committed position, anchor at P+1, draft i
predicts P+2+i), so every block is a round the engine could have run on that committed trajectory.

Blocks with P inside the prompt (P < gen_start - 1) have no top-20 and are not part of the data.

## 5. Splits

| split | content | use |
|---|---|---|
| `train` | legal families `holding_summary`, `issue_spotting`, `argument_section`, `discovery_requests`, `sc_family_qa` + general train items | training |
| `dev` | same families, source documents disjoint from train | checkpoint selection only |
| `heldout_legal` | families `chronology` and `motion_outline` (never in train/dev), on source documents never in train/dev; >= 20 sequences | G1, G2 (report once) |
| `heldout_general` | general items never in train/dev; >= 10 sequences | G2 general gate |
| `acap` | the 4 A-capture requests replayed for the equivalence check | never trained on; eval only |

Splits are disjoint by source document (a document, identified by its text sha256, appears in one split only) and by
task family as above. A shard holds one split only.

## 6. Files

Directory `/opt/dsv41/x86host-data/d2/shards/` (mode 700):

```
MANIFEST.json                      schema d2.manifest.v1 (section 7)
SHA256SUMS                         sha256sum -c format, every file below
<split>/<split>-NNNNN.safetensors  tensors (section 6.1)
<split>/<split>-NNNNN.json         sidecar: per-sequence metadata (section 6.2)
```

Each sequence lies entirely in one shard. Shards are cut at <= 2 GiB of taps (about 65K rows).

### 6.1 Tensors (safetensors; header metadata `{"schema": "d2.shard.v1", "split": <split>}`)

N sequences, T = sum L, R = sum (L - r0) rows, S = sum (L - gen_start) span positions.

| name | dtype | shape | content |
|---|---|---|---|
| `seq_index` | int64 | [N, 8] | per sequence: `tok_off, L, gen_start, r0, row_off, n_rows, span_off, n_span` |
| `tokens` | int32 | [T] | sequence i's ids at `tokens[tok_off : tok_off + L]` |
| `taps` | bfloat16 | [R, 15360] | row of position p of sequence i at `row_off + (p - r0)`; columns `[0:5120]` layer 37, `[5120:10240]` layer 38, `[10240:15360]` layer 39 (`view(R, 3, 5120)`) |
| `row_pos` | int32 | [R] | absolute position of each row (redundant; integrity) |
| `top_ids` | int32 | [S, 20] | position p = gen_start + j of sequence i at `span_off + j`; descending probability |
| `top_lp` | float32 | [S, 20] | natural-log probabilities of `top_ids` (non-increasing along the row) |
| `tok_lp` | float32 | [S] | logprob of the committed token `tokens[p]` |
| `tok_rank` | int32 | [S] | its rank (1 = most likely); `<= 20` means it is in `top_ids` |
| `resid` | float32 | [S] | `max(0, 1 - sum(exp(top_lp)))`: the mass outside the top-20 (one tail bucket) |

`n_rows = L - r0`, `n_span = L - gen_start`; offsets are cumulative in sequence order.

### 6.2 Sidecar JSON

```json
{"schema": "d2.shard.v1", "split": "train", "file": "train-00000.safetensors", "sha256": "<of the .safetensors>",
 "W": 128, "K": 5, "topk": 20, "tap_layers": [37, 38, 39], "hidden": 5120, "noise_token": 128799,
 "taps_source": "prefill_replay",
 "sequences": [
   {"index": 0, "seq_id": "p0123", "split": "train", "family": "argument_section", "source": "opinions_cl",
    "source_docs": ["<doc sha256>", "..."], "prompt_sha256": "<sha256 of the canonical prompt JSON>",
    "tokens_sha256": "<sha256 of tokens[0:L] as int32 little-endian>", "L": 9412, "gen_start": 6120,
    "n_gen": 3292, "finish_reason": "stop|length|aborted", "segments": 1, "replay_request_id": "d2r.p0123.5992.9412"}]}
```

`prompt_sha256` and `tokens_sha256` equal the values in the prompt manifest and the generation state, so a shard row
can be traced back to its prompt without opening any text.

## 7. MANIFEST.json (`d2.manifest.v1`)

```json
{"schema": "d2.manifest.v1", "created_utc": "...", "taps_source": "prefill_replay",
 "prompt_manifest_sha256": "...", "generation": {"server": "http://HEAD_IP:8000", "model_root": "...",
   "settings": {"temperature": 1.0, "top_p": 1.0, "thinking": true, "reasoning_effort": "high", "priority": 5}},
 "replay": {"config": "A-replay.env", "overlay_sha256": "...", "window_utc": "..."},
 "splits": {"train": {"n_seq": 0, "n_span": 0, "n_rows": 0, "families": {"argument_section": 0},
                      "files": [{"file": "train/train-00000.safetensors", "sidecar": "train/train-00000.json",
                                 "sha256": "...", "sidecar_sha256": "...", "bytes": 0, "n_seq": 0, "n_rows": 0,
                                 "n_span": 0}]}},
 "checks": {"plumbing_errors": 0, "...": "..."}}
```

## 8. Integrity rules (enforced by `dataset.py build`; a reader may re-check)

1. `tokens_sha256` equals the generation state's; `L`, `gen_start` equal the generation state's.
2. The replay overlay also saves the token id it saw at every row; it must equal `tokens[r0:L]` (proves each tap row is
   keyed to the right sequence and position).
3. One overlay row per position `r0 .. L-1`, none missing, none duplicated with different bytes; all finite; no row
   with any layer's 5120 taps all zero (eager prefill steps leave trimmed rows' taps zero; requests with
   `prompt_logprobs` keep every row, so a zero row means the replay did not run as planned). DONE.json carries
   `zero_rows`, which must be 0.
4. `prompt_logprobs` has L entries; at every span position the committed token's logprob is present.
5. `top_ids` distinct within a row; `top_lp` finite, `<= 0`, non-increasing; `sum(exp(top_lp)) <= 1 + 1e-4`;
   `tok_lp == top_lp[k]` where `top_ids[k] == tokens[p]` whenever `tok_rank <= 20`.
6. Every file's sha256 matches `SHA256SUMS` and `MANIFEST.json`.

## 9. Reference reader

`d2/shards.py` (pure torch + safetensors): `Shard(path)` checks the schema and exposes the tensors, the sidecar and
`blocks()` (every (i, P) of section 4); `ShardCaptureSet` adapts a set of shards to V1's `CaptureSet` interface
(`rows`, `samples`, `plumbing`, `window`, `window_status`, `arm_inputs`), so V1 `run_ref` / `analyze` run on D2 data
unchanged. `dataset.py synth` writes schema-valid synthetic shards for tests (a synthetic set is on x86host at
`/opt/dsv41/x86host-data/d2/synthetic/`).

Notes for V1 consumers:
* sample keys are `r%06d_b%d` with round = P and b = the sequence's index in the set (unique, and V1 analyze's dump
  glob finds them); `s.req` is the seq_id;
* D2 has no engine drafts and no engine main_x. `engine_drafts` are NOISE placeholders (`engine_drafts_placeholder`)
  unless drafts are passed in: `drafts=` keyed by sample key, or `drafts_by_seq=` keyed seq_id -> {P: [5 ids]}, the form
  `g1.py k2` writes (`<work>/<split>.k2_drafts.jsonl`), which gives the K2 baseline for paired comparisons. Rows carry a
  zero `mx`, so run_ref's `mainx_engine_vs_native.json` is meaningless on D2 sets.

## D2-FIX1 split reconciliation (2026-10-23)

MEASURED from the read-only prompts.jsonl and both private/committed manifests: original PLAN built
546 prompts = 480 + 30 + 24 + 12. S14 F8 removed exactly eight train IDs, without renumbering.
Actual counts: **472 train + 30 dev + 24 heldout_legal + 12 heldout_general = 538**.
The private and committed manifests match; JSONL SHA256 is
`07f5a8e31c59b8c0f8032e43163dbc969f95c61c27b601594666cdca4630e1f8`.
The former 480-train statement is the pre-exclusion plan, not the built population.

`dataset build` requires --prompt-manifest. Before reading taps it validates the manifest's exact
rows/counts, unique IDs and hashes, disjoint source families (connected source documents used in
prompts), held-out legal task-family separation, every replay ID/split/family/hash against the
manifest, and each source state against its plan entry. No --allow-skips bypass exists for this
preflight. Task labels can repeat in train/dev; source families cannot cross splits.
The private split-audit receipt records included and not-replayed IDs. Time-capped generation may
produce a subset; missing sequences are recorded, never added to force a planned count.

Amendment 6 makes the historical `acap`, `heldout_legal` and `heldout_general` names development
populations once used for selection. Their on-disk names stay stable for existing replay readers;
they must not be described as untouched final TEST. Only the separately frozen new TEST families
may be evaluated once against the final frozen candidate.

## D2-FIX1 sequence audit and long-position coverage

The tensor schema remains d2.shard.v1. New additive per-sequence metadata has its own schema,
`d2.sequence-audit.v1`, and does not change existing reader tensor meanings. Real `dataset build`
now requires --sequence-context, a coordinator-prepared `d2.sequence-context.v1` JSON receipt:

- `target` and `replay_target`: model_root, quantization, engine_revision,
  weights_manifest_sha256, quantization_manifest_sha256, receipt_sha256. Stable identities must
  agree; the generation and replay evidence receipts may have different hashes.
- tokenizer_path/tokenizer_sha256, encoder_path/encoder_sha256,
  chat_template_path/chat_template_sha256. Supply locally accessible copies. Hashes must match
  files; tokenizer/encoder hashes must match the frozen prompt manifest. The template file can
  be the pinned encoder source when that is the active chat renderer.
- `markers`: think_start, think_end, assistant, user, eos, dsml. IDs must equal those in the hashed
  tokenizer for the actual public special-token strings; no text decoding or usage-counter guess.
- `sampling`: temperature=1, top_p=1; explicitly include top_k, min_p, structured_outputs, stop,
  stop_token_ids, seed (null when absent). Record additional effective processors/defaults and
  generation_config absence/hash in the receipt. Each new generation segment saves its actual
  sampling body; recorded fields must agree with the receipt. Legacy segment settings are sourced
  from the pinned e2b2186 contract, not inferred from a new priority20 restart.
- `formatting`: chat_template_kwargs matching the frozen manifest, tools and tool_choice (explicit
  null for this no-tools pipeline), plus any schema/tool processor identity. For acap, supply its
  original capture context explicitly in `per_sequence[seq_id]`, never silently copy new settings.

The coordinator verifies target receipts against the actual generation/replay windows. This lane
cannot verify live target weights without prohibited service access and does not manufacture that
receipt. A missing context is a build refusal, not a guessed identity. Synthetic-only tests use
an explicit synthetic manifest schema and are labelled not_served_data.

Every sidecar retains the raw committed int32 IDs and their hash via the existing tensor schema.
The audit adds complete identity, half-open absolute reasoning/answer/tool-call/format intervals,
full source segment records, and the deliberate response-only label policy. Reasoning starts in the
prompt's <think> state, continues across cancelled/resumed segments, and stays reasoning if a
partial ends before </think>. EOS and role controls are format positions. DSML marks a generated
tool-call span. Raw IDs preserve all chat/tool formatting even when the pipeline has no tool schema.
No tool output is invented or post-hoc decoded to alter a sequence.

Alignment is tested through the actual shard writer and D2Blocks reader at short and 500,000-token
absolute positions: last usable tap is P; anchor/correction is P+1; draft i targets P+2+i with
teacher-forced predecessor P+1+i. Teacher logprob row t is conditioned on tokens[:t]. Taps after P
are excluded. Prompt labels are excluded. The first response token is an anchor/correction and
has no draft label; draft targets begin gen_start+1. Partial tails have label=-1 and false masks.
These are committed-prefix/teacher-forcing labels; they cannot label a different rollout prefix.

PLANNED long-context DEV/TEST coverage (not generated or serving-qualified here): use independent
source families at 128K, 300K, 450K and up to literal 500,000 prompt tokens, with anchors early and
late in reasoning and the visible answer, recording actual absolute positions and segment counts.
DEV supports checkpoint selection; reserved TEST families are used only for the frozen candidate.
Use coherent public source blocks/tool histories rather than treating repeated filler as representative.
Existing examples stop at 48K and do not establish long-context adaptation. The draft's 128-row
window does not remove target features' long-prefix dependence.

450K/500K collection waits for the coordinator's qualified 524,288-total-context profile; production
is currently 327,680 and the larger profile is HOLD. At 500,000 prompt tokens reserve 16,384 for
reasoning+answer, leaving 7,904 for other material; no 1M-context experiment. Server /tokenize counts
are authoritative when generation is later authorized. The CPU 500K test proves indexing/masks only.

## Reserved final TEST prompt list

`python -m d2.reserved_test --out <new-private-dir> --commit-manifest <text-free-manifest>`
uses the existing public-source loader/SHA checks, V4.1 renderer, escape rules, legal-template
pipeline, source-disjointness and 12-word-shingle screen. It excludes every document already used
by the 538 prompts and all original acap documents. All prior development/historical-heldout prompts
are treated as exclusion material. It refuses duplicate prompt identities and overwrite attempts.

The frozen design is 12 prompts, four new task families × three excerpt lengths, from four new
public source families. Only the text-free manifest goes in git; prompts.jsonl, manifest.json and
SHA256SUMS are frozen read-only in the private campaign directory. This is a prompt-only reservation;
no targets, teacher distributions or model outputs are produced. No TEST traffic is added to the
default generator ORDER or the pilot's selection populations. Final generation/replay/evaluation
requires the coordinator's frozen-candidate identity and explicit final-test collection path. The
current pilot readers retain their historical split registry; do not relabel TEST into dev to bypass
it. Once used for candidate selection, this reservation must be honestly reclassified as development
and replaced before any claim of untouched final testing.

## Decode-capture option A implementation (2026-10-24)

The W9 gate failed; replay taps remain refused. `--taps decode_capture` is an alias
for `--taps-source decode_capture`, using a NEW plan made by `d2.decode_dataset`:

```bash
# CPU on x86host, after collection, durable flush and verified private copy:
python -m d2.decode_dataset --top20 /private/SECOND_PASS_top20_replay \
  --captures /private/SECOND_PASS_decode_copy --out /private/NEW_decode_plan
python -m d2.dataset build --replay /private/NEW_decode_plan --taps decode_capture \
  --prompt-manifest /private/prompts/manifest.json \
  --sequence-context /private/verified-second-pass-context.json --out /private/NEW_shards
```

The first command consumes the second pass's top20-only replay plan/arrays, never
historical first-pass probabilities. It does not submit requests. Its d2c plan IDs
plus DECODE.json are checked against each CAPTURE.json's actual mode; a d2r plan
cannot bypass equivalence by changing a CLI label. Assembly requires FLUSH.json
and matching FLUSHED.json nonce with zero errors and exact ledger hash/file sizes,
no capture ERRORS.jsonl, header mode/origin/module identity, every ledger range and
retained-byte digest, full original r0..L_gen-2 coverage and exact
`ids.bin[0:L_gen-1-r0] == tokens[r0:L_gen-1]`. Any zero layer row refuses. Duplicate
rows retain the first bytes; `dup_rows_differing` is recorded as information.
Aborted/uncheckpointed suffix rows are verified and excluded, never trained on.
No DONE.json is expected or fabricated for a live generation capture.

For these shards only, `L = L_gen - 1`: the final sampled token was never fed back
and has no required tap. tokens, taps and top20 span arrays are shortened together.
The tensor schema remains d2.shard.v1. Sidecars retain generation_L and the full
generation hash, ledger/header/flush/source hashes, top20 source/hash and counters.
The full raw generation sequence audit retains original segment coverage and adds
`shard_interval` and `dropped_final_sample`; segment metadata is never silently
shortened or relabelled. The ordinary split, actual marker/tokenizer/encoder,
T=1/top_p=1, target/quantization and shard/plumbing checks still apply. Real decode
build additionally runs post_replay.context_ready, including accessible SHA-bound
evidence_receipts, before tensor assembly. Missing/incomplete receipts refuse.

Default top20 source is `prefill_replay_top20_only`. Its taps are unused and no
equivalence gate applies to the generation taps. The opt-in `--top20 generation`
collector and frozen exporter are documented in [OPTION-II.md](OPTION-II.md).
`--top20-source generation_verified --verification <TOP20_TREE/VERIFIED.json>`
requires both the real pinned-image/capture smoke and the completions-validation
receipt binding every frozen state, normalized array and plan. A chat smoke alone
cannot certify this different formatter. All existing capture/context and
probability checks remain. `tok_rank=21` means outside the selected top20; the
completion API does not return the exact outside vocabulary rank.

The original acap exports are eval-only and are not new d2c generation states.
This preparation command refuses to relabel them. Keep G1(b)'s original native
decode-capture evaluation separate; do not fill acap gaps with failed replay taps
or claim newly generated answers reproduce the original engine trajectory.
