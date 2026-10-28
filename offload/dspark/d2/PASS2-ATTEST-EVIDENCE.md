# Pass 2 attestation evidence (offline inventory)

No production attestation is issued here. MEASURED means the cited bytes establish the stated narrow fact; IMPLIED is a candidate requiring coordinator review; MISSING has no established value. Digests below are MEASURED on Controller. Historical reads do not prove the current container. No weights or live generator state were read.

## Field map

| Field | Value / candidate | Status | Sources | Reason / limit |
|---|---|---|---|---|
| `schema` | `"d2.pass2-attestation.v1"` | MEASURED | review-d2pass2-r1.json | Required output schema (ASSEMBLE.md), not a deployment fact. |
| `coordinator` | `"Deepseek/coordinator"` | MEASURED | review-d2pass2-r1.json | W12 approved coordinator; fresh resolution approval still required. |
| `run` | `"d2pass2-r1"` | MEASURED | review-d2pass2-r1.json | W12 run. |
| `target.model_root` | `"/srv/dsv41/legal-v2a"` | IMPLIED | git:dsv41.env, S2.report.md, D2-REPLAY-PREP.live-container.json | Historical candidate only. Coordinator must select actual production manifest semantics and bind both current ranks/image/overlays. Baseline manifest and KV overlay MANIFEST are not the legal weights/quantization manifests. |
| `decode_capture_target.model_root` | `"/srv/dsv41/legal-v2a"` | IMPLIED | git:dsv41.env, S2.report.md, D2-REPLAY-PREP.live-container.json | Must equal resolved production target; W12 establishes derivative only. |
| `target.quantization` | `"EXL3 legal-v2a over 2.75bpw"` | IMPLIED | D2-REPLAY-PREP.live-files.txt, offload/results/inventory/INVENTORY-2026-10-22.md, offload/kvpack/MANIFEST.json | Historical candidate only. Coordinator must select actual production manifest semantics and bind both current ranks/image/overlays. Baseline manifest and KV overlay MANIFEST are not the legal weights/quantization manifests. |
| `decode_capture_target.quantization` | `"EXL3 legal-v2a over 2.75bpw"` | IMPLIED | D2-REPLAY-PREP.live-files.txt, offload/results/inventory/INVENTORY-2026-10-22.md, offload/kvpack/MANIFEST.json | Must equal resolved production target; W12 establishes derivative only. |
| `target.engine_revision` | `"0.31.1.dev0+g6e517b15.dsv41gb10"` | IMPLIED | D2-REPLAY-PREP.live-files.txt, offload/results/inventory/INVENTORY-2026-10-22.md, offload/kvpack/MANIFEST.json | Historical candidate only. Coordinator must select actual production manifest semantics and bind both current ranks/image/overlays. Baseline manifest and KV overlay MANIFEST are not the legal weights/quantization manifests. |
| `decode_capture_target.engine_revision` | `"0.31.1.dev0+g6e517b15.dsv41gb10"` | IMPLIED | D2-REPLAY-PREP.live-files.txt, offload/results/inventory/INVENTORY-2026-10-22.md, offload/kvpack/MANIFEST.json | Must equal resolved production target; W12 establishes derivative only. |
| `target.weights_manifest_sha256` | `"7549bb956f6f660d7932bcd5112501f78a2c660b8bcbcbc8905c2673f5ae05ae"` | IMPLIED | D2-REPLAY-PREP.live-files.txt, offload/results/inventory/INVENTORY-2026-10-22.md, offload/kvpack/MANIFEST.json | Historical candidate only. Coordinator must select actual production manifest semantics and bind both current ranks/image/overlays. Baseline manifest and KV overlay MANIFEST are not the legal weights/quantization manifests. |
| `decode_capture_target.weights_manifest_sha256` | `"7549bb956f6f660d7932bcd5112501f78a2c660b8bcbcbc8905c2673f5ae05ae"` | IMPLIED | D2-REPLAY-PREP.live-files.txt, offload/results/inventory/INVENTORY-2026-10-22.md, offload/kvpack/MANIFEST.json | Must equal resolved production target; W12 establishes derivative only. |
| `target.quantization_manifest_sha256` | `"ebf315efa28868de91fb95f5782fbc3a4a75623e0e0cd8a7fd3bae0b1a01830d"` | IMPLIED | D2-REPLAY-PREP.live-files.txt, offload/results/inventory/INVENTORY-2026-10-22.md, offload/kvpack/MANIFEST.json | Historical candidate only. Coordinator must select actual production manifest semantics and bind both current ranks/image/overlays. Baseline manifest and KV overlay MANIFEST are not the legal weights/quantization manifests. |
| `decode_capture_target.quantization_manifest_sha256` | `"ebf315efa28868de91fb95f5782fbc3a4a75623e0e0cd8a7fd3bae0b1a01830d"` | IMPLIED | D2-REPLAY-PREP.live-files.txt, offload/results/inventory/INVENTORY-2026-10-22.md, offload/kvpack/MANIFEST.json | Must equal resolved production target; W12 establishes derivative only. |
| `decode_capture_target.run` | `"d2pass2-r1"` | MEASURED | review-d2pass2-r1.json, rendered-d2pass2-r1.diff | Hash-bound W12 derivative review. |
| `decode_capture_target.review_sha256` | `"0e299348fd67da31a7b4ef240e82a1cd4e62b77c8a7027bbbd7a3c9848d3a2e0"` | MEASURED | review-d2pass2-r1.json, rendered-d2pass2-r1.diff | Hash-bound W12 derivative review. |
| `decode_capture_target.capture_env_sha256` | `"34b0a736a15cd241d0d0778cf9640bd76055eb4a918206be8ab6a2742d023071"` | MEASURED | review-d2pass2-r1.json, rendered-d2pass2-r1.diff | Hash-bound W12 derivative review. |
| `sampling.temperature` | `1.0` | MEASURED | generate.py, completion-protocol, completion-serving, model-config, sampling-source, D2-REPLAY-PREP.live-files.txt | Explicit raw-ID request parameter. |
| `sampling.top_p` | `1.0` | MEASURED | generate.py, completion-protocol, completion-serving, model-config, sampling-source, D2-REPLAY-PREP.live-files.txt | Explicit raw-ID request parameter. |
| `sampling.top_k` | `0` | IMPLIED | generate.py, completion-protocol, completion-serving, model-config, sampling-source, D2-REPLAY-PREP.live-files.txt | Candidate effective default: request omits field. Protocol precedence is user > server generation config > neutral defaults; stop IDs also merge server defaults. Historical absence of generation_config is insufficient to prove current effective behavior. EOS remains tokenizer/model driven; [] here means explicit extra stops. |
| `sampling.min_p` | `0.0` | IMPLIED | generate.py, completion-protocol, completion-serving, model-config, sampling-source, D2-REPLAY-PREP.live-files.txt | Candidate effective default: request omits field. Protocol precedence is user > server generation config > neutral defaults; stop IDs also merge server defaults. Historical absence of generation_config is insufficient to prove current effective behavior. EOS remains tokenizer/model driven; [] here means explicit extra stops. |
| `sampling.structured_outputs` | `null` | IMPLIED | generate.py, completion-protocol, completion-serving, model-config, sampling-source, D2-REPLAY-PREP.live-files.txt | Candidate effective default: request omits field. Protocol precedence is user > server generation config > neutral defaults; stop IDs also merge server defaults. Historical absence of generation_config is insufficient to prove current effective behavior. EOS remains tokenizer/model driven; [] here means explicit extra stops. |
| `sampling.stop` | `[]` | IMPLIED | generate.py, completion-protocol, completion-serving, model-config, sampling-source, D2-REPLAY-PREP.live-files.txt | Candidate effective default: request omits field. Protocol precedence is user > server generation config > neutral defaults; stop IDs also merge server defaults. Historical absence of generation_config is insufficient to prove current effective behavior. EOS remains tokenizer/model driven; [] here means explicit extra stops. |
| `sampling.stop_token_ids` | `[]` | IMPLIED | generate.py, completion-protocol, completion-serving, model-config, sampling-source, D2-REPLAY-PREP.live-files.txt | Candidate effective default: request omits field. Protocol precedence is user > server generation config > neutral defaults; stop IDs also merge server defaults. Historical absence of generation_config is insufficient to prove current effective behavior. EOS remains tokenizer/model driven; [] here means explicit extra stops. |
| `sampling.seed` | `null` | IMPLIED | generate.py, completion-protocol, completion-serving, model-config, sampling-source, D2-REPLAY-PREP.live-files.txt | Candidate effective default: request omits field. Protocol precedence is user > server generation config > neutral defaults; stop IDs also merge server defaults. Historical absence of generation_config is insufficient to prove current effective behavior. EOS remains tokenizer/model driven; [] here means explicit extra stops. |
| `formatting.chat_template_kwargs` | `{"thinking": true, "reasoning_effort": "high"}` | IMPLIED | generate.py, prompts.py | Must bind supplied prompt manifest and served renderer. Raw-ID completions do not invoke the chat renderer; /tokenize checks only compared prompts. |
| `formatting.tools` | `null` | IMPLIED | generate.py, prompts.py | Must bind supplied prompt manifest and served renderer. Raw-ID completions do not invoke the chat renderer; /tokenize checks only compared prompts. |
| `formatting.tool_choice` | `null` | IMPLIED | generate.py, prompts.py | Must bind supplied prompt manifest and served renderer. Raw-ID completions do not invoke the chat renderer; /tokenize checks only compared prompts. |
| `generator.revision` | `"1b8fc1c"` | MEASURED | w12.log, generate.py, generation_top20.py | Pinned source/settings, not proof every runtime segment used them. |
| `generator.priority` | `20` | MEASURED | w12.log, generate.py, generation_top20.py | Pinned source/settings, not proof every runtime segment used them. |
| `generator.top20` | `"generation"` | MEASURED | w12.log, generate.py, generation_top20.py | Pinned source/settings, not proof every runtime segment used them. |
| `generator.source_sha256.generate.py` | `"56c2fd7152e8dd8a8ccbce2a2bf79ad1382a9f68d03f343fba25116c76ddfd7a"` | MEASURED | w12.log, generate.py | Full pinned file digest matching W12 GEN records. |
| `generator.source_sha256.generation_top20.py` | `"984910e53c2229a881e4b6954963fce16670d93032bd0de32dd61c10ba33672d"` | MEASURED | w12.log, generation_top20.py | Full pinned file digest matching W12 GEN records. |
| `generator.source_sha256.prompts.py` | `"117a90dd70f41b03f5abaa0fe9f5e637abad54ddb48adb117b32bf1a75554966"` | MEASURED | w12.log, prompts.py | Full pinned file digest matching W12 GEN records. |
| `identity.tokenizer.path` | `null` | MISSING | D2-REPLAY-PREP.live-files.txt, generate.py, prompts.py, encoder-source, tokenizer-wrapper, tokenize-serving, online-renderer | Local encoder bytes are measured; served correspondence is unresolved. prompts.LocalRenderer calls encode_messages with thinking_mode, drop_thinking=True and reasoning_effort, then tokenizer.encode(add_special_tokens=False). The pinned served wrapper calls the same encoder after message normalization. Candidate encoder/template may use the same file. tokenizer_config alone is insufficient; need absolute local tokenizer and current served-file bindings. |
| `identity.tokenizer.sha256` | `"c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"` | IMPLIED | D2-REPLAY-PREP.live-files.txt, generate.py, prompts.py, encoder-source, tokenizer-wrapper, tokenize-serving, online-renderer | Local encoder bytes are measured; served correspondence is unresolved. prompts.LocalRenderer calls encode_messages with thinking_mode, drop_thinking=True and reasoning_effort, then tokenizer.encode(add_special_tokens=False). The pinned served wrapper calls the same encoder after message normalization. Candidate encoder/template may use the same file. tokenizer_config alone is insufficient; need absolute local tokenizer and current served-file bindings. |
| `identity.encoder.path` | `"/opt/dsv41/data/src-image/vllm/tokenizers/deepseek_v41_encoding.py"` | IMPLIED | D2-REPLAY-PREP.live-files.txt, generate.py, prompts.py, encoder-source, tokenizer-wrapper, tokenize-serving, online-renderer | Local encoder bytes are measured; served correspondence is unresolved. prompts.LocalRenderer calls encode_messages with thinking_mode, drop_thinking=True and reasoning_effort, then tokenizer.encode(add_special_tokens=False). The pinned served wrapper calls the same encoder after message normalization. Candidate encoder/template may use the same file. tokenizer_config alone is insufficient; need absolute local tokenizer and current served-file bindings. |
| `identity.encoder.sha256` | `"abfe2a1aa76d235bad273fe144d18919d8ef77c3709857bb523ef69f75cdfc6b"` | IMPLIED | D2-REPLAY-PREP.live-files.txt, generate.py, prompts.py, encoder-source, tokenizer-wrapper, tokenize-serving, online-renderer | Local encoder bytes are measured; served correspondence is unresolved. prompts.LocalRenderer calls encode_messages with thinking_mode, drop_thinking=True and reasoning_effort, then tokenizer.encode(add_special_tokens=False). The pinned served wrapper calls the same encoder after message normalization. Candidate encoder/template may use the same file. tokenizer_config alone is insufficient; need absolute local tokenizer and current served-file bindings. |
| `identity.chat_template.path` | `"/opt/dsv41/data/src-image/vllm/tokenizers/deepseek_v41_encoding.py"` | IMPLIED | D2-REPLAY-PREP.live-files.txt, generate.py, prompts.py, encoder-source, tokenizer-wrapper, tokenize-serving, online-renderer | Local encoder bytes are measured; served correspondence is unresolved. prompts.LocalRenderer calls encode_messages with thinking_mode, drop_thinking=True and reasoning_effort, then tokenizer.encode(add_special_tokens=False). The pinned served wrapper calls the same encoder after message normalization. Candidate encoder/template may use the same file. tokenizer_config alone is insufficient; need absolute local tokenizer and current served-file bindings. |
| `identity.chat_template.sha256` | `"abfe2a1aa76d235bad273fe144d18919d8ef77c3709857bb523ef69f75cdfc6b"` | IMPLIED | D2-REPLAY-PREP.live-files.txt, generate.py, prompts.py, encoder-source, tokenizer-wrapper, tokenize-serving, online-renderer | Local encoder bytes are measured; served correspondence is unresolved. prompts.LocalRenderer calls encode_messages with thinking_mode, drop_thinking=True and reasoning_effort, then tokenizer.encode(add_special_tokens=False). The pinned served wrapper calls the same encoder after message normalization. Candidate encoder/template may use the same file. tokenizer_config alone is insufficient; need absolute local tokenizer and current served-file bindings. |
| `evidence` | `10 W12/source records (see JSON)` | MEASURED | D2-ASSEMBLE-R2.evidence-inventory.json | Ten measured W12/source files. Generator adds its NEW target-receipt.json; never write W12. |
| `verification.served_identity` | `null` | MISSING | generate.py | Coordinator must explicitly resolve true from hash-bound primary evidence, with scope/limitations recorded in reviewed resolution. |
| `verification.effective_defaults` | `null` | MISSING | generate.py | Coordinator must explicitly resolve true from hash-bound primary evidence, with scope/limitations recorded in reviewed resolution. |
| `verification.renderer_raw_ids` | `null` | MISSING | generate.py | Coordinator must explicitly resolve true from hash-bound primary evidence, with scope/limitations recorded in reviewed resolution. |

## Source hashes

| Source | Path / git object | SHA256 |
|---|---|---|
| git:dsv41.env | `git:21bed313ac803d38327c4de1ff2e5235f329c716:dsv41.env` | `faa9d3e0019a8665f0b61dbf46a79134680731d43b9a9cb99ca2fa5af85373a8` |
| git:manifest.json | `git:21bed313ac803d38327c4de1ff2e5235f329c716:manifest.json` | `70a8516ce28250804c454408c97cdd75be8ff86dfbe6ce7e1c2fee55654a9f73` |
| git:serve.sh | `git:21bed313ac803d38327c4de1ff2e5235f329c716:serve.sh` | `8cf08d3abb6a453dcbe5f108e7c3862f15aac9db384767736dd343709addfb2b` |
| offload/kvpack/MANIFEST.json | `/opt/dsv41/repos/.worktrees/dsv41-d2attest/offload/kvpack/MANIFEST.json` | `559c5867ba42aa7daa7b4a7c8231d15ee1d35be40ae5b715b013f973193c58de` |
| offload/dspark/TENSORS.json | `/opt/dsv41/repos/.worktrees/dsv41-d2attest/offload/dspark/TENSORS.json` | `dc8c78a4529eab861143bfe76496034ec07987544a1791aef30f25389575f03b` |
| offload/dspark/TRANSFER_RECEIPTS.json | `/opt/dsv41/repos/.worktrees/dsv41-d2attest/offload/dspark/TRANSFER_RECEIPTS.json` | `5fdfe4fdab36af5f5445e27a98356659d91d610c21ed06ee3a6410073294a32f` |
| offload/results/inventory/INVENTORY-2026-10-22.md | `/opt/dsv41/repos/.worktrees/dsv41-d2attest/offload/results/inventory/INVENTORY-2026-10-22.md` | `84b887806e8a4dfae0f453d666c418ae11c619c9500b70c49e0cbf26015cd008` |
| S2.report.md | `/opt/dsv41/campaigns/DSV41-OFFLOAD/S2.report.md` | `447e4d0d7def23738b23d0106ead64b3187bd511e06f04239053b3a5dda8c875` |
| V1-VERIFY.report.md | `/opt/dsv41/campaigns/DSV41-OFFLOAD/V1-VERIFY.report.md` | `d6d042663ed0fe667d2b1b76f802198cc9faba20f9b2c80914272f5a97764ea6` |
| K1-PROMOTE-327.report.md | `/opt/dsv41/campaigns/DSV41-OFFLOAD/K1-PROMOTE-327.report.md` | `0dd0e72ef74168294ccb569651737541db6ae114a6ed2cfc6ff2cd627cda1f05` |
| D2-REPLAY-PREP.live-files.txt | `/opt/dsv41/campaigns/DSV41-OFFLOAD/D2-REPLAY-PREP.live-files.txt` | `3c7b95b3fc24b95b2b174b0d4c3df9db417698703ce5c902b0ff9f2d880759d1` |
| D2-REPLAY-PREP.live-container.json | `/opt/dsv41/campaigns/DSV41-OFFLOAD/D2-REPLAY-PREP.live-container.json` | `32be581e4cedfcd0b01dd99092732acd16707a679ab05103ced6e09f879123ea` |
| D2-ASSEMBLE-R2.evidence-inventory.json | `/opt/dsv41/campaigns/DSV41-OFFLOAD/D2-ASSEMBLE-R2.evidence-inventory.json` | `44f17c6884b952d2ba59fb3e922724efa7373055da4f3eb921fbcfcbbcf5cede` |
| sampling-source | `/opt/dsv41/data/src-image/vllm/sampling_params.py` | `3c37c655a0a60851e3d96d946eb10d1cbb347dee6cd143f38a7fc18c6f66dd8b` |
| completion-protocol | `/opt/dsv41/data/src-image/vllm/entrypoints/openai/completion/protocol.py` | `4616cd6887ef0a23fd8a77468e82f2668e3d0b53075112f2d97f3de960c22c81` |
| completion-serving | `/opt/dsv41/data/src-image/vllm/entrypoints/openai/completion/serving.py` | `aac889978efb034b95ab81ec59695d83ba7c879ab48cd5be2448c980f109a3bd` |
| model-config | `/opt/dsv41/data/src-image/vllm/config/model.py` | `c00204dc349ceb402012e32d3837e0db4d6bd896e9aae03ef0b0d16ffca1b69d` |
| w12.log | `/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/w12.log` | `b7b164c0f24ba90198df73aadc4fed446baf48220478b8a490725c370e4ff420` |
| base-hashes.txt | `/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/base-hashes.txt` | `970ad2934afaa8ef4447d7b490f8832ae4a930e0fb7fcaff74aae9ffcce93cfb` |
| staged-hashes.txt | `/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/staged-hashes.txt` | `2a01d2fcf82c7c782f50228b53ffe578cf3e6f3dd9311d91328e60a188ed0f7a` |
| mounted-hashes.txt | `/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/mounted-hashes.txt` | `43b8f7aa67c807756071691aee8ca5c0338eab5e0f6e7dd1adcc28ff35fe04cd` |
| rendered-d2pass2-r1.diff | `/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/rendered-d2pass2-r1.diff` | `e24f19fbd0c253d67f30d665c32badc8947ca067af63570458ad1ac3304b8c91` |
| review-d2pass2-r1.json | `/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/review-d2pass2-r1.json` | `0e299348fd67da31a7b4ef240e82a1cd4e62b77c8a7027bbbd7a3c9848d3a2e0` |
| start.log | `/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/start.log` | `ff55ad2262df50d40a65e8c22a03166db96f29f96707edec981ebc4a9fb6f9bf` |
| generate.py | `/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/gen-checkout-1b8fc1c/offload/dspark/d2/generate.py` | `56c2fd7152e8dd8a8ccbce2a2bf79ad1382a9f68d03f343fba25116c76ddfd7a` |
| generation_top20.py | `/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/gen-checkout-1b8fc1c/offload/dspark/d2/generation_top20.py` | `984910e53c2229a881e4b6954963fce16670d93032bd0de32dd61c10ba33672d` |
| prompts.py | `/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/gen-checkout-1b8fc1c/offload/dspark/d2/prompts.py` | `117a90dd70f41b03f5abaa0fe9f5e637abad54ddb48adb117b32bf1a75554966` |
| W10-R6-console | `/opt/dsv41/campaigns/DSV41-OFFLOAD/W10/private/smoke-r6-20261024T203207Z.console.log` | `78b38214b128464d727f2b652af60d26429116e5aa547960930c06e04988975d` |
| encoder-source | `/opt/dsv41/data/src-image/vllm/tokenizers/deepseek_v41_encoding.py` | `abfe2a1aa76d235bad273fe144d18919d8ef77c3709857bb523ef69f75cdfc6b` |
| tokenizer-wrapper | `/opt/dsv41/data/src-image/vllm/tokenizers/deepseek_v41.py` | `e182e6357f9e2063298e6fbf3c515789cc8e6d1b31fa5f6058e74865a3a8183e` |
| tokenize-serving | `/opt/dsv41/data/src-image/vllm/entrypoints/serve/tokenize/serving.py` | `fb9fd00cc71df5c052f420a959e39434ce5fd3c84556275eb9dc25d43ff59750` |
| online-renderer | `/opt/dsv41/data/src-image/vllm/renderers/online_renderer.py` | `a000d446e3febba540b218a363db1682a359be6773e7e9d69de4ca0b89ae279d` |

## What these records do and do not prove

S2 verifies extracted drafter dependencies and Engram transfers, not every re-encoded target expert. V1 is a numerical/fidelity report, not a current production-target receipt. K1 and its MANIFEST bind KV overlays, not weights or the complete engine. git 21bed31 records launch intent; W12 review/start/mounted files bind that deployment window. The historical replay reads supply candidate CANDIDATE_MANIFEST/exl3-plan/tokenizer digests; their bytes and current rank correspondence still need review.

W10 R6 console reports a genuine smoke-window PASS for 07fc7d6. The assembly runbook pins the separate R6 smoke receipt to b288080cc44f45c5edb9f5b96e4caff9327cfd29b32558e9741bb9bf1b812dde. That receipt was not found in the permitted local metadata inventory; do not substitute the console digest or read production capture directories. R6 validates the capture path, not model manifests or defaults.

Pinned generate.py: ensure_prompt_ids calls /tokenize with messages and chat_template_kwargs, then compares those IDs with local.ids only when a local renderer exists. tokenize_local_mismatch increments once per mismatching prompt. rate.jsonl serializes that counter; its definition is in the pinned source. A zero counter alone proves neither local checking enabled nor prompt coverage nor encoder file identity. This lane did not inspect rate.jsonl or generator state, and will not collect them. Coordinator must provide separately reviewed coverage/identity evidence or retain the gap. No new /tokenize/model requests are authorized by the collector.

## Coordinator handoff

Run the collection script only as coordinator in an idle window, into a NEW private campaign directory. It collects bounded metadata, never weights or model requests; unavailable allowlisted files are recorded explicitly. Review both ranks, generation-config/override launch arguments and mounted renderer sources. Resolve every field with a MEASURED claim bound by a JSON pointer to reviewed primary evidence (or a coordinator semantic-review JSON plus hash-pinned raw sources). Include verified served identity, effective defaults and raw-ID renderer obligations. Missing runtime semantic evidence remains a refusal; collector success is not attestation approval.

The machine map below is consumed by pass2_attest.py. Proposed values marked IMPLIED may be corrected by reviewed evidence; MEASURED values cannot silently change. The independent map and resolution SHA256 arguments bind review inputs. Output is a DRAFT pending coordinator review, never automatic authorization.

```json
{
  "schema": "d2.pass2-evidence-map.v1",
  "claims": {
    "schema": {
      "value": "d2.pass2-attestation.v1",
      "status": "MEASURED",
      "sources": [
        "review-d2pass2-r1.json"
      ],
      "note": "Required output schema (ASSEMBLE.md), not a deployment fact."
    },
    "coordinator": {
      "value": "Deepseek/coordinator",
      "status": "MEASURED",
      "sources": [
        "review-d2pass2-r1.json"
      ],
      "note": "W12 approved coordinator; fresh resolution approval still required."
    },
    "run": {
      "value": "d2pass2-r1",
      "status": "MEASURED",
      "sources": [
        "review-d2pass2-r1.json"
      ],
      "note": "W12 run."
    },
    "target.model_root": {
      "value": "/srv/dsv41/legal-v2a",
      "status": "IMPLIED",
      "sources": [
        "git:dsv41.env",
        "S2.report.md",
        "D2-REPLAY-PREP.live-container.json"
      ],
      "note": "Historical candidate only. Coordinator must select actual production manifest semantics and bind both current ranks/image/overlays. Baseline manifest and KV overlay MANIFEST are not the legal weights/quantization manifests."
    },
    "decode_capture_target.model_root": {
      "value": "/srv/dsv41/legal-v2a",
      "status": "IMPLIED",
      "sources": [
        "git:dsv41.env",
        "S2.report.md",
        "D2-REPLAY-PREP.live-container.json"
      ],
      "note": "Must equal resolved production target; W12 establishes derivative only."
    },
    "target.quantization": {
      "value": "EXL3 legal-v2a over 2.75bpw",
      "status": "IMPLIED",
      "sources": [
        "D2-REPLAY-PREP.live-files.txt",
        "offload/results/inventory/INVENTORY-2026-10-22.md",
        "offload/kvpack/MANIFEST.json"
      ],
      "note": "Historical candidate only. Coordinator must select actual production manifest semantics and bind both current ranks/image/overlays. Baseline manifest and KV overlay MANIFEST are not the legal weights/quantization manifests."
    },
    "decode_capture_target.quantization": {
      "value": "EXL3 legal-v2a over 2.75bpw",
      "status": "IMPLIED",
      "sources": [
        "D2-REPLAY-PREP.live-files.txt",
        "offload/results/inventory/INVENTORY-2026-10-22.md",
        "offload/kvpack/MANIFEST.json"
      ],
      "note": "Must equal resolved production target; W12 establishes derivative only."
    },
    "target.engine_revision": {
      "value": "0.31.1.dev0+g6e517b15.dsv41gb10",
      "status": "IMPLIED",
      "sources": [
        "D2-REPLAY-PREP.live-files.txt",
        "offload/results/inventory/INVENTORY-2026-10-22.md",
        "offload/kvpack/MANIFEST.json"
      ],
      "note": "Historical candidate only. Coordinator must select actual production manifest semantics and bind both current ranks/image/overlays. Baseline manifest and KV overlay MANIFEST are not the legal weights/quantization manifests."
    },
    "decode_capture_target.engine_revision": {
      "value": "0.31.1.dev0+g6e517b15.dsv41gb10",
      "status": "IMPLIED",
      "sources": [
        "D2-REPLAY-PREP.live-files.txt",
        "offload/results/inventory/INVENTORY-2026-10-22.md",
        "offload/kvpack/MANIFEST.json"
      ],
      "note": "Must equal resolved production target; W12 establishes derivative only."
    },
    "target.weights_manifest_sha256": {
      "value": "7549bb956f6f660d7932bcd5112501f78a2c660b8bcbcbc8905c2673f5ae05ae",
      "status": "IMPLIED",
      "sources": [
        "D2-REPLAY-PREP.live-files.txt",
        "offload/results/inventory/INVENTORY-2026-10-22.md",
        "offload/kvpack/MANIFEST.json"
      ],
      "note": "Historical candidate only. Coordinator must select actual production manifest semantics and bind both current ranks/image/overlays. Baseline manifest and KV overlay MANIFEST are not the legal weights/quantization manifests."
    },
    "decode_capture_target.weights_manifest_sha256": {
      "value": "7549bb956f6f660d7932bcd5112501f78a2c660b8bcbcbc8905c2673f5ae05ae",
      "status": "IMPLIED",
      "sources": [
        "D2-REPLAY-PREP.live-files.txt",
        "offload/results/inventory/INVENTORY-2026-10-22.md",
        "offload/kvpack/MANIFEST.json"
      ],
      "note": "Must equal resolved production target; W12 establishes derivative only."
    },
    "target.quantization_manifest_sha256": {
      "value": "ebf315efa28868de91fb95f5782fbc3a4a75623e0e0cd8a7fd3bae0b1a01830d",
      "status": "IMPLIED",
      "sources": [
        "D2-REPLAY-PREP.live-files.txt",
        "offload/results/inventory/INVENTORY-2026-10-22.md",
        "offload/kvpack/MANIFEST.json"
      ],
      "note": "Historical candidate only. Coordinator must select actual production manifest semantics and bind both current ranks/image/overlays. Baseline manifest and KV overlay MANIFEST are not the legal weights/quantization manifests."
    },
    "decode_capture_target.quantization_manifest_sha256": {
      "value": "ebf315efa28868de91fb95f5782fbc3a4a75623e0e0cd8a7fd3bae0b1a01830d",
      "status": "IMPLIED",
      "sources": [
        "D2-REPLAY-PREP.live-files.txt",
        "offload/results/inventory/INVENTORY-2026-10-22.md",
        "offload/kvpack/MANIFEST.json"
      ],
      "note": "Must equal resolved production target; W12 establishes derivative only."
    },
    "decode_capture_target.run": {
      "value": "d2pass2-r1",
      "status": "MEASURED",
      "sources": [
        "review-d2pass2-r1.json",
        "rendered-d2pass2-r1.diff"
      ],
      "note": "Hash-bound W12 derivative review."
    },
    "decode_capture_target.review_sha256": {
      "value": "0e299348fd67da31a7b4ef240e82a1cd4e62b77c8a7027bbbd7a3c9848d3a2e0",
      "status": "MEASURED",
      "sources": [
        "review-d2pass2-r1.json",
        "rendered-d2pass2-r1.diff"
      ],
      "note": "Hash-bound W12 derivative review."
    },
    "decode_capture_target.capture_env_sha256": {
      "value": "34b0a736a15cd241d0d0778cf9640bd76055eb4a918206be8ab6a2742d023071",
      "status": "MEASURED",
      "sources": [
        "review-d2pass2-r1.json",
        "rendered-d2pass2-r1.diff"
      ],
      "note": "Hash-bound W12 derivative review."
    },
    "sampling.temperature": {
      "value": 1.0,
      "status": "MEASURED",
      "sources": [
        "generate.py",
        "completion-protocol",
        "completion-serving",
        "model-config",
        "sampling-source",
        "D2-REPLAY-PREP.live-files.txt"
      ],
      "note": "Explicit raw-ID request parameter."
    },
    "sampling.top_p": {
      "value": 1.0,
      "status": "MEASURED",
      "sources": [
        "generate.py",
        "completion-protocol",
        "completion-serving",
        "model-config",
        "sampling-source",
        "D2-REPLAY-PREP.live-files.txt"
      ],
      "note": "Explicit raw-ID request parameter."
    },
    "sampling.top_k": {
      "value": 0,
      "status": "IMPLIED",
      "sources": [
        "generate.py",
        "completion-protocol",
        "completion-serving",
        "model-config",
        "sampling-source",
        "D2-REPLAY-PREP.live-files.txt"
      ],
      "note": "Candidate effective default: request omits field. Protocol precedence is user > server generation config > neutral defaults; stop IDs also merge server defaults. Historical absence of generation_config is insufficient to prove current effective behavior. EOS remains tokenizer/model driven; [] here means explicit extra stops."
    },
    "sampling.min_p": {
      "value": 0.0,
      "status": "IMPLIED",
      "sources": [
        "generate.py",
        "completion-protocol",
        "completion-serving",
        "model-config",
        "sampling-source",
        "D2-REPLAY-PREP.live-files.txt"
      ],
      "note": "Candidate effective default: request omits field. Protocol precedence is user > server generation config > neutral defaults; stop IDs also merge server defaults. Historical absence of generation_config is insufficient to prove current effective behavior. EOS remains tokenizer/model driven; [] here means explicit extra stops."
    },
    "sampling.structured_outputs": {
      "value": null,
      "status": "IMPLIED",
      "sources": [
        "generate.py",
        "completion-protocol",
        "completion-serving",
        "model-config",
        "sampling-source",
        "D2-REPLAY-PREP.live-files.txt"
      ],
      "note": "Candidate effective default: request omits field. Protocol precedence is user > server generation config > neutral defaults; stop IDs also merge server defaults. Historical absence of generation_config is insufficient to prove current effective behavior. EOS remains tokenizer/model driven; [] here means explicit extra stops."
    },
    "sampling.stop": {
      "value": [],
      "status": "IMPLIED",
      "sources": [
        "generate.py",
        "completion-protocol",
        "completion-serving",
        "model-config",
        "sampling-source",
        "D2-REPLAY-PREP.live-files.txt"
      ],
      "note": "Candidate effective default: request omits field. Protocol precedence is user > server generation config > neutral defaults; stop IDs also merge server defaults. Historical absence of generation_config is insufficient to prove current effective behavior. EOS remains tokenizer/model driven; [] here means explicit extra stops."
    },
    "sampling.stop_token_ids": {
      "value": [],
      "status": "IMPLIED",
      "sources": [
        "generate.py",
        "completion-protocol",
        "completion-serving",
        "model-config",
        "sampling-source",
        "D2-REPLAY-PREP.live-files.txt"
      ],
      "note": "Candidate effective default: request omits field. Protocol precedence is user > server generation config > neutral defaults; stop IDs also merge server defaults. Historical absence of generation_config is insufficient to prove current effective behavior. EOS remains tokenizer/model driven; [] here means explicit extra stops."
    },
    "sampling.seed": {
      "value": null,
      "status": "IMPLIED",
      "sources": [
        "generate.py",
        "completion-protocol",
        "completion-serving",
        "model-config",
        "sampling-source",
        "D2-REPLAY-PREP.live-files.txt"
      ],
      "note": "Candidate effective default: request omits field. Protocol precedence is user > server generation config > neutral defaults; stop IDs also merge server defaults. Historical absence of generation_config is insufficient to prove current effective behavior. EOS remains tokenizer/model driven; [] here means explicit extra stops."
    },
    "formatting.chat_template_kwargs": {
      "value": {
        "thinking": true,
        "reasoning_effort": "high"
      },
      "status": "IMPLIED",
      "sources": [
        "generate.py",
        "prompts.py"
      ],
      "note": "Must bind supplied prompt manifest and served renderer. Raw-ID completions do not invoke the chat renderer; /tokenize checks only compared prompts."
    },
    "formatting.tools": {
      "value": null,
      "status": "IMPLIED",
      "sources": [
        "generate.py",
        "prompts.py"
      ],
      "note": "Must bind supplied prompt manifest and served renderer. Raw-ID completions do not invoke the chat renderer; /tokenize checks only compared prompts."
    },
    "formatting.tool_choice": {
      "value": null,
      "status": "IMPLIED",
      "sources": [
        "generate.py",
        "prompts.py"
      ],
      "note": "Must bind supplied prompt manifest and served renderer. Raw-ID completions do not invoke the chat renderer; /tokenize checks only compared prompts."
    },
    "generator.revision": {
      "value": "1b8fc1c",
      "status": "MEASURED",
      "sources": [
        "w12.log",
        "generate.py",
        "generation_top20.py"
      ],
      "note": "Pinned source/settings, not proof every runtime segment used them."
    },
    "generator.priority": {
      "value": 20,
      "status": "MEASURED",
      "sources": [
        "w12.log",
        "generate.py",
        "generation_top20.py"
      ],
      "note": "Pinned source/settings, not proof every runtime segment used them."
    },
    "generator.top20": {
      "value": "generation",
      "status": "MEASURED",
      "sources": [
        "w12.log",
        "generate.py",
        "generation_top20.py"
      ],
      "note": "Pinned source/settings, not proof every runtime segment used them."
    },
    "generator.source_sha256.generate.py": {
      "value": "56c2fd7152e8dd8a8ccbce2a2bf79ad1382a9f68d03f343fba25116c76ddfd7a",
      "status": "MEASURED",
      "sources": [
        "w12.log",
        "generate.py"
      ],
      "note": "Full pinned file digest matching W12 GEN records."
    },
    "generator.source_sha256.generation_top20.py": {
      "value": "984910e53c2229a881e4b6954963fce16670d93032bd0de32dd61c10ba33672d",
      "status": "MEASURED",
      "sources": [
        "w12.log",
        "generation_top20.py"
      ],
      "note": "Full pinned file digest matching W12 GEN records."
    },
    "generator.source_sha256.prompts.py": {
      "value": "117a90dd70f41b03f5abaa0fe9f5e637abad54ddb48adb117b32bf1a75554966",
      "status": "MEASURED",
      "sources": [
        "w12.log",
        "prompts.py"
      ],
      "note": "Full pinned file digest matching W12 GEN records."
    },
    "identity.tokenizer.path": {
      "value": null,
      "status": "MISSING",
      "sources": [
        "D2-REPLAY-PREP.live-files.txt",
        "generate.py",
        "prompts.py",
        "encoder-source",
        "tokenizer-wrapper",
        "tokenize-serving",
        "online-renderer"
      ],
      "note": "Local encoder bytes are measured; served correspondence is unresolved. prompts.LocalRenderer calls encode_messages with thinking_mode, drop_thinking=True and reasoning_effort, then tokenizer.encode(add_special_tokens=False). The pinned served wrapper calls the same encoder after message normalization. Candidate encoder/template may use the same file. tokenizer_config alone is insufficient; need absolute local tokenizer and current served-file bindings."
    },
    "identity.tokenizer.sha256": {
      "value": "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
      "status": "IMPLIED",
      "sources": [
        "D2-REPLAY-PREP.live-files.txt",
        "generate.py",
        "prompts.py",
        "encoder-source",
        "tokenizer-wrapper",
        "tokenize-serving",
        "online-renderer"
      ],
      "note": "Local encoder bytes are measured; served correspondence is unresolved. prompts.LocalRenderer calls encode_messages with thinking_mode, drop_thinking=True and reasoning_effort, then tokenizer.encode(add_special_tokens=False). The pinned served wrapper calls the same encoder after message normalization. Candidate encoder/template may use the same file. tokenizer_config alone is insufficient; need absolute local tokenizer and current served-file bindings."
    },
    "identity.encoder.path": {
      "value": "/opt/dsv41/data/src-image/vllm/tokenizers/deepseek_v41_encoding.py",
      "status": "IMPLIED",
      "sources": [
        "D2-REPLAY-PREP.live-files.txt",
        "generate.py",
        "prompts.py",
        "encoder-source",
        "tokenizer-wrapper",
        "tokenize-serving",
        "online-renderer"
      ],
      "note": "Local encoder bytes are measured; served correspondence is unresolved. prompts.LocalRenderer calls encode_messages with thinking_mode, drop_thinking=True and reasoning_effort, then tokenizer.encode(add_special_tokens=False). The pinned served wrapper calls the same encoder after message normalization. Candidate encoder/template may use the same file. tokenizer_config alone is insufficient; need absolute local tokenizer and current served-file bindings."
    },
    "identity.encoder.sha256": {
      "value": "abfe2a1aa76d235bad273fe144d18919d8ef77c3709857bb523ef69f75cdfc6b",
      "status": "IMPLIED",
      "sources": [
        "D2-REPLAY-PREP.live-files.txt",
        "generate.py",
        "prompts.py",
        "encoder-source",
        "tokenizer-wrapper",
        "tokenize-serving",
        "online-renderer"
      ],
      "note": "Local encoder bytes are measured; served correspondence is unresolved. prompts.LocalRenderer calls encode_messages with thinking_mode, drop_thinking=True and reasoning_effort, then tokenizer.encode(add_special_tokens=False). The pinned served wrapper calls the same encoder after message normalization. Candidate encoder/template may use the same file. tokenizer_config alone is insufficient; need absolute local tokenizer and current served-file bindings."
    },
    "identity.chat_template.path": {
      "value": "/opt/dsv41/data/src-image/vllm/tokenizers/deepseek_v41_encoding.py",
      "status": "IMPLIED",
      "sources": [
        "D2-REPLAY-PREP.live-files.txt",
        "generate.py",
        "prompts.py",
        "encoder-source",
        "tokenizer-wrapper",
        "tokenize-serving",
        "online-renderer"
      ],
      "note": "Local encoder bytes are measured; served correspondence is unresolved. prompts.LocalRenderer calls encode_messages with thinking_mode, drop_thinking=True and reasoning_effort, then tokenizer.encode(add_special_tokens=False). The pinned served wrapper calls the same encoder after message normalization. Candidate encoder/template may use the same file. tokenizer_config alone is insufficient; need absolute local tokenizer and current served-file bindings."
    },
    "identity.chat_template.sha256": {
      "value": "abfe2a1aa76d235bad273fe144d18919d8ef77c3709857bb523ef69f75cdfc6b",
      "status": "IMPLIED",
      "sources": [
        "D2-REPLAY-PREP.live-files.txt",
        "generate.py",
        "prompts.py",
        "encoder-source",
        "tokenizer-wrapper",
        "tokenize-serving",
        "online-renderer"
      ],
      "note": "Local encoder bytes are measured; served correspondence is unresolved. prompts.LocalRenderer calls encode_messages with thinking_mode, drop_thinking=True and reasoning_effort, then tokenizer.encode(add_special_tokens=False). The pinned served wrapper calls the same encoder after message normalization. Candidate encoder/template may use the same file. tokenizer_config alone is insufficient; need absolute local tokenizer and current served-file bindings."
    },
    "evidence": {
      "value": [
        {
          "label": "w12.log",
          "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/w12.log",
          "sha256": "b7b164c0f24ba90198df73aadc4fed446baf48220478b8a490725c370e4ff420",
          "bytes": 4079
        },
        {
          "label": "base-hashes.txt",
          "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/base-hashes.txt",
          "sha256": "970ad2934afaa8ef4447d7b490f8832ae4a930e0fb7fcaff74aae9ffcce93cfb",
          "bytes": 246
        },
        {
          "label": "staged-hashes.txt",
          "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/staged-hashes.txt",
          "sha256": "2a01d2fcf82c7c782f50228b53ffe578cf3e6f3dd9311d91328e60a188ed0f7a",
          "bytes": 634
        },
        {
          "label": "mounted-hashes.txt",
          "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/mounted-hashes.txt",
          "sha256": "43b8f7aa67c807756071691aee8ca5c0338eab5e0f6e7dd1adcc28ff35fe04cd",
          "bytes": 1030
        },
        {
          "label": "rendered-d2pass2-r1.diff",
          "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/rendered-d2pass2-r1.diff",
          "sha256": "e24f19fbd0c253d67f30d665c32badc8947ca067af63570458ad1ac3304b8c91",
          "bytes": 1072
        },
        {
          "label": "review-d2pass2-r1.json",
          "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/review-d2pass2-r1.json",
          "sha256": "0e299348fd67da31a7b4ef240e82a1cd4e62b77c8a7027bbbd7a3c9848d3a2e0",
          "bytes": 701
        },
        {
          "label": "start.log",
          "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/start.log",
          "sha256": "ff55ad2262df50d40a65e8c22a03166db96f29f96707edec981ebc4a9fb6f9bf",
          "bytes": 3229
        },
        {
          "label": "generate.py",
          "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/gen-checkout-1b8fc1c/offload/dspark/d2/generate.py",
          "sha256": "56c2fd7152e8dd8a8ccbce2a2bf79ad1382a9f68d03f343fba25116c76ddfd7a",
          "bytes": 44983
        },
        {
          "label": "generation_top20.py",
          "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/gen-checkout-1b8fc1c/offload/dspark/d2/generation_top20.py",
          "sha256": "984910e53c2229a881e4b6954963fce16670d93032bd0de32dd61c10ba33672d",
          "bytes": 4118
        },
        {
          "label": "prompts.py",
          "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/gen-checkout-1b8fc1c/offload/dspark/d2/prompts.py",
          "sha256": "117a90dd70f41b03f5abaa0fe9f5e637abad54ddb48adb117b32bf1a75554966",
          "bytes": 40826
        }
      ],
      "status": "MEASURED",
      "sources": [
        "D2-ASSEMBLE-R2.evidence-inventory.json"
      ],
      "note": "Ten measured W12/source files. Generator adds its NEW target-receipt.json; never write W12."
    },
    "verification.served_identity": {
      "value": null,
      "status": "MISSING",
      "sources": [
        "generate.py"
      ],
      "note": "Coordinator must explicitly resolve true from hash-bound primary evidence, with scope/limitations recorded in reviewed resolution."
    },
    "verification.effective_defaults": {
      "value": null,
      "status": "MISSING",
      "sources": [
        "generate.py"
      ],
      "note": "Coordinator must explicitly resolve true from hash-bound primary evidence, with scope/limitations recorded in reviewed resolution."
    },
    "verification.renderer_raw_ids": {
      "value": null,
      "status": "MISSING",
      "sources": [
        "generate.py"
      ],
      "note": "Coordinator must explicitly resolve true from hash-bound primary evidence, with scope/limitations recorded in reviewed resolution."
    }
  },
  "sources": {
    "git:dsv41.env": {
      "path": "git:21bed313ac803d38327c4de1ff2e5235f329c716:dsv41.env",
      "sha256": "faa9d3e0019a8665f0b61dbf46a79134680731d43b9a9cb99ca2fa5af85373a8"
    },
    "git:manifest.json": {
      "path": "git:21bed313ac803d38327c4de1ff2e5235f329c716:manifest.json",
      "sha256": "70a8516ce28250804c454408c97cdd75be8ff86dfbe6ce7e1c2fee55654a9f73"
    },
    "git:serve.sh": {
      "path": "git:21bed313ac803d38327c4de1ff2e5235f329c716:serve.sh",
      "sha256": "8cf08d3abb6a453dcbe5f108e7c3862f15aac9db384767736dd343709addfb2b"
    },
    "offload/kvpack/MANIFEST.json": {
      "path": "/opt/dsv41/repos/.worktrees/dsv41-d2attest/offload/kvpack/MANIFEST.json",
      "sha256": "559c5867ba42aa7daa7b4a7c8231d15ee1d35be40ae5b715b013f973193c58de"
    },
    "offload/dspark/TENSORS.json": {
      "path": "/opt/dsv41/repos/.worktrees/dsv41-d2attest/offload/dspark/TENSORS.json",
      "sha256": "dc8c78a4529eab861143bfe76496034ec07987544a1791aef30f25389575f03b"
    },
    "offload/dspark/TRANSFER_RECEIPTS.json": {
      "path": "/opt/dsv41/repos/.worktrees/dsv41-d2attest/offload/dspark/TRANSFER_RECEIPTS.json",
      "sha256": "5fdfe4fdab36af5f5445e27a98356659d91d610c21ed06ee3a6410073294a32f"
    },
    "offload/results/inventory/INVENTORY-2026-10-22.md": {
      "path": "/opt/dsv41/repos/.worktrees/dsv41-d2attest/offload/results/inventory/INVENTORY-2026-10-22.md",
      "sha256": "84b887806e8a4dfae0f453d666c418ae11c619c9500b70c49e0cbf26015cd008"
    },
    "S2.report.md": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/S2.report.md",
      "sha256": "447e4d0d7def23738b23d0106ead64b3187bd511e06f04239053b3a5dda8c875"
    },
    "V1-VERIFY.report.md": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/V1-VERIFY.report.md",
      "sha256": "d6d042663ed0fe667d2b1b76f802198cc9faba20f9b2c80914272f5a97764ea6"
    },
    "K1-PROMOTE-327.report.md": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/K1-PROMOTE-327.report.md",
      "sha256": "0dd0e72ef74168294ccb569651737541db6ae114a6ed2cfc6ff2cd627cda1f05"
    },
    "D2-REPLAY-PREP.live-files.txt": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/D2-REPLAY-PREP.live-files.txt",
      "sha256": "3c7b95b3fc24b95b2b174b0d4c3df9db417698703ce5c902b0ff9f2d880759d1"
    },
    "D2-REPLAY-PREP.live-container.json": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/D2-REPLAY-PREP.live-container.json",
      "sha256": "32be581e4cedfcd0b01dd99092732acd16707a679ab05103ced6e09f879123ea"
    },
    "D2-ASSEMBLE-R2.evidence-inventory.json": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/D2-ASSEMBLE-R2.evidence-inventory.json",
      "sha256": "44f17c6884b952d2ba59fb3e922724efa7373055da4f3eb921fbcfcbbcf5cede"
    },
    "sampling-source": {
      "path": "/opt/dsv41/data/src-image/vllm/sampling_params.py",
      "sha256": "3c37c655a0a60851e3d96d946eb10d1cbb347dee6cd143f38a7fc18c6f66dd8b"
    },
    "completion-protocol": {
      "path": "/opt/dsv41/data/src-image/vllm/entrypoints/openai/completion/protocol.py",
      "sha256": "4616cd6887ef0a23fd8a77468e82f2668e3d0b53075112f2d97f3de960c22c81"
    },
    "completion-serving": {
      "path": "/opt/dsv41/data/src-image/vllm/entrypoints/openai/completion/serving.py",
      "sha256": "aac889978efb034b95ab81ec59695d83ba7c879ab48cd5be2448c980f109a3bd"
    },
    "model-config": {
      "path": "/opt/dsv41/data/src-image/vllm/config/model.py",
      "sha256": "c00204dc349ceb402012e32d3837e0db4d6bd896e9aae03ef0b0d16ffca1b69d"
    },
    "w12.log": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/w12.log",
      "sha256": "b7b164c0f24ba90198df73aadc4fed446baf48220478b8a490725c370e4ff420"
    },
    "base-hashes.txt": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/base-hashes.txt",
      "sha256": "970ad2934afaa8ef4447d7b490f8832ae4a930e0fb7fcaff74aae9ffcce93cfb"
    },
    "staged-hashes.txt": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/staged-hashes.txt",
      "sha256": "2a01d2fcf82c7c782f50228b53ffe578cf3e6f3dd9311d91328e60a188ed0f7a"
    },
    "mounted-hashes.txt": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/mounted-hashes.txt",
      "sha256": "43b8f7aa67c807756071691aee8ca5c0338eab5e0f6e7dd1adcc28ff35fe04cd"
    },
    "rendered-d2pass2-r1.diff": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/rendered-d2pass2-r1.diff",
      "sha256": "e24f19fbd0c253d67f30d665c32badc8947ca067af63570458ad1ac3304b8c91"
    },
    "review-d2pass2-r1.json": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/review-d2pass2-r1.json",
      "sha256": "0e299348fd67da31a7b4ef240e82a1cd4e62b77c8a7027bbbd7a3c9848d3a2e0"
    },
    "start.log": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/start.log",
      "sha256": "ff55ad2262df50d40a65e8c22a03166db96f29f96707edec981ebc4a9fb6f9bf"
    },
    "generate.py": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/gen-checkout-1b8fc1c/offload/dspark/d2/generate.py",
      "sha256": "56c2fd7152e8dd8a8ccbce2a2bf79ad1382a9f68d03f343fba25116c76ddfd7a"
    },
    "generation_top20.py": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/gen-checkout-1b8fc1c/offload/dspark/d2/generation_top20.py",
      "sha256": "984910e53c2229a881e4b6954963fce16670d93032bd0de32dd61c10ba33672d"
    },
    "prompts.py": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W12/private/gen-checkout-1b8fc1c/offload/dspark/d2/prompts.py",
      "sha256": "117a90dd70f41b03f5abaa0fe9f5e637abad54ddb48adb117b32bf1a75554966"
    },
    "W10-R6-console": {
      "path": "/opt/dsv41/campaigns/DSV41-OFFLOAD/W10/private/smoke-r6-20261024T203207Z.console.log",
      "sha256": "78b38214b128464d727f2b652af60d26429116e5aa547960930c06e04988975d"
    },
    "encoder-source": {
      "path": "/opt/dsv41/data/src-image/vllm/tokenizers/deepseek_v41_encoding.py",
      "sha256": "abfe2a1aa76d235bad273fe144d18919d8ef77c3709857bb523ef69f75cdfc6b"
    },
    "tokenizer-wrapper": {
      "path": "/opt/dsv41/data/src-image/vllm/tokenizers/deepseek_v41.py",
      "sha256": "e182e6357f9e2063298e6fbf3c515789cc8e6d1b31fa5f6058e74865a3a8183e"
    },
    "tokenize-serving": {
      "path": "/opt/dsv41/data/src-image/vllm/entrypoints/serve/tokenize/serving.py",
      "sha256": "fb9fd00cc71df5c052f420a959e39434ce5fd3c84556275eb9dc25d43ff59750"
    },
    "online-renderer": {
      "path": "/opt/dsv41/data/src-image/vllm/renderers/online_renderer.py",
      "sha256": "a000d446e3febba540b218a363db1682a359be6773e7e9d69de4ca0b89ae279d"
    }
  }
}
```
