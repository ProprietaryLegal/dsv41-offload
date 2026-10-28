# Draft-only representation leads: verified public metadata

No weight tensor payload downloaded. Four pinned HF API inventories and fourteen safetensors headers are saved in
`metadata/*.json`; each includes revision, API/metadata hashes, sibling sizes/LFS SHA256, tensor names/shapes/dtypes/
byte offsets and header hash/range bytes. All header_failures are empty. Total safetensors header bytes consumed:
**1,165,560** (255176 official +454968 2.7 +455416 2.8), tensor payload bytes consumed **0**.
`hf_manifest.py` reproduces this with full pinned SHA, capped metadata64MiB/header8MiB; requires206 and exact
Content-Range BEFORE reading, consumes exactly prefix8 + JSON header only. A200/full-file response is closed unread.
Initial metadata8MiB limit rejected sudoingX's large indexes; bounded64MiB rerun succeeded. No model/quant download.

| Source | Pinned revision | Verified inventory / provenance |
|---|---|---|
| [sudoingx 2.7](https://huggingface.co/sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw/tree/026e27a37b0c154380c81ce1ee49fc334b4fa264) | 026e27a37b0c154380c81ce1ee49fc334b4fa264 | 4747 mtp.* map entries; 5 header files. Dense MTP21 linears/84 tensors, K5 trellis width80; 1152 expert matrices, K3 width48, 4608 tensor parts. Main projection remains official FP8, not EXL3. Shared target/draft head K4, no separate MTP-head file. |
| [sudoingx 2.8](https://huggingface.co/sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw/tree/aba08ea002ae271ee2a2724bcc0c6dc6e1c457b8) | aba08ea002ae271ee2a2724bcc0c6dc6e1c457b8 | 4751 mtp.* entries; 6 headers. Same MTP K3 expert and K5 dense FILE hashes as2.7. Adds separate `mtp.head.*` K4 head; target config head_k5. Native main_proj stays FP8.416B `dspark_conf_feat_head.npz` metadata only, no donor policy import. |
| [official1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) | 1bae0e1088993b9b82de08f59de5be5a34e42170 | 2401 mtp.* tensors in `model-00044/45/46-of-00048.safetensors`, respectively801/798/802; three headers. Expert packed I8 nibble storage + F8_E8M0 scales, native MXFP4 semantics per official config/card; dense roles FP8/F32/BF16. Both sudoingX cards pin this same official base revision. |
| [ELVEZ1017](https://huggingface.co/elvezjp/DeepSeek-V4.1-Flash-1017-ELVEZ-Q2-gguf/tree/189d54bce22d30a1563d346e84ff1873855efb88) | 189d54bce22d30a1563d346e84ff1873855efb88 | API file list + README provenance only; no GGUF bytes read. DSpark artifact7935279104B, SHA2564f85722c85591bfc022933947cc19673345aea1a202ace11e96db9f99072813b. Card says three official MTP stages, expert MXFP4 lossless repack, dense Q8_0/F32 roles; IXV-specific and not directly loadable. |

Critical public tensor/file facts:
- Both EXL3 donor expert files: `model-exl3-mtp-experts.safetensors`, **5,113,631,328B**, LFS SHA256
  `39a7331913c5327ad5be97814e2cfdde636325fb18baa0ea7849e824264a049b`. Every one of1152 trellis headers has last
  dimension48=16*K3; w1 example `[320,144,48]` I16 + FP16 suh[5120]/svh[2304] and I32 mul1 (payload NOT read).
  Plan has three128-character K3 rows at absolute layers40/41/42. K3 is a real format lead, not just a card claim.
- Both dense files: `model-exl3-mtp.safetensors`, **241,342,412B**, SHA256
  `d8e3f4394fd95311be89d97bce7acf676e3c1e6a0fca15756a10a9970834510f`. Per stage wq_a,wkv,wq_b,wo_b,shared w1/w2/w3;
  K5 shapes respectively[320,80,80], [320,32,80], [80,2048,80], [512,320,80], [320,144,80], [144,320,80], [320,144,80].
-2.8 separate head: **331,226,052B**, SHA256 `58e845b46b24a92864320fd9044599eb54c43ba33084b46cc457b827751212b0`,
  `mtp.head.trellis` I16[320,8080,64] => logical5120x129280 K4. Header does not certify mul1 value or decoded weights.
- Official expert w1 `[2304,2560]` I8 encodes5120 input values at two4-bit values/byte; scales `[2304,160]` F8_E8M0,
  i.e.32 logical values per scale. The header cannot prove nibble/codebook arithmetic; official representation
  definition plus byte-level decode tests would be required before conversion. No native FP4 arithmetic on SM86.
- Native main_proj: F8_E4M3 `[5120,15360]`, scale F8_E8M0 `[160,480]` (32x32 blocks). Final Markov head BF16
  `[129280,256]`. Official shards44–46 total7,933,129,808B including headers, not the full needed embed/head dependency.

Compatibility with OUR worker (inspection of checkpoint.py/identity.py/model.py/exl3 plan, base21bc3ab):
1. Current checkpoint loader is identity-locked to S2 legal-v2a receipt and required tensor set. ALL donor packages
   are rejected today; changing only a filename or claiming same1017 does not authorize identity bypass.
2. Selective donor K3 EXPERTS are the closest-format experiment. Same shapes/names, version1.6.0/mul1 metadata,
   integerK3 supported by worker planner and decode kernel interfaces. Future adapter must produce a new explicit
   composite drafter manifest: keep current legal-v2a taps, K5 main_proj/dense, embedding, K6 head and Markov/confidence;
   substitute ONLY expert weights+plan rows, verify each source range/hash and actual mul1, new HELLO drafter identity.
   No target weight/config mutation. SM86 K3 kernel/value parity and VRAM budget still need validation.
3. Whole donor dense/head load is incompatible without changes: donor main_proj is FP8 rather than our required
   EXL3 parts. The K5 dense linears otherwise match layout but were calibrated under donor hidden-state streams.
   Keep current FP32-accumulating main_proj. Numerical/acceptance equivalence is not implied by native provenance.
4. Separate K4 head is mathematically compatible with the worker's full-vocab EXL3 head IF mapped as a proposal-only
   head with independent identity/config. Current loader expects `head.*` + x.head_k and shared target identity;
   it has no `drafter_head_k`/`mtp.head.*` route. Do not rename silently or replace target K6. Donor feature-depth
   coefficients are invalid for our head/expert/conditioning combination. Refit locally on disjoint legal tasks.
5. Official/ELVEZ native experts require a new exact MXFP4 decoder/layout/scale adapter. Official tensors are not
   EXL3 trellis and GGUF is not safetensors. Ampere can decode to supported values/software kernels; no hardware
   FP4 claim. A native expert+current rest mixed worker needs its own complete identity and reference tests.

Measurement/stop plan (coordinator + W1; metadata work alone is not acceptance evidence): one representation at a
time after exact worker qualification. Current vs K3-experts-only, current vs K4-head-only, native expert separate;
match target legal-v2a, raw taps, power/mode/shape. Measure decoded-value parity to each representation reference,
round time, committed tokens/round, confidence calibration/acceptance, Markov tie handling, memory/stability and
heldout legal full-task latency. Stop on incompatible/missing parts, unverified tensor provenance, unsupported
codebook/scale corners, VRAM pressure, target mutation, lost required logits or no repeatable useful end-to-end gain.
Provenance/headers do not establish better proposal agreement with our legal-v2a target.
