# Exact DSpark tensor inventory

Source: sparkhead `/srv/dsv41/legal-v2a`; every safetensors header inspected with O_DIRECT. No tensor execution or conversion.

Baseline: HeNryous revision `86f852533ec94d1a537fea41fd9c01d6b0f6eb50`. All selected files share device+inode with `/srv/dsv41/model`, proving byte identity; file SHA256 expectations are from the deployment manifest. Engram originates at DeepSeek revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.

`TENSORS.json` lists every selected tensor's name, shape, dtype, EXL3 K, source absolute half-open byte range, packed resident bytes, baseline file SHA256 and hardlink evidence. After transfer it also records SHA256 of every source byte range and destination file/range. These artifacts are local only.

| Source file | Tensors | Packed bytes |
|---|---:|---:|
| model-00002-of-00046.safetensors | 1 | 1323827200 |
| model-00043-of-00046.safetensors | 1 | 10240 |
| model-00044-of-00046.safetensors | 17 | 38865880 |
| model-00045-of-00046.safetensors | 16 | 38855640 |
| model-00046-of-00046.safetensors | 20 | 171259352 |
| model-exl3-head.safetensors | 4 | 496704004 |
| model-exl3-mtp-experts.safetensors | 4608 | 3414495744 |
| model-exl3-mtp.safetensors | 88 | 290527832 |

Total: 4755 tensors, 5,774,545,892 bytes (5.378 GiB).

Dependencies: `mtp.0.main_proj.*` / `mtp.0.main_norm.weight`; all `mtp.{0,1,2}` attention, MoE router+experts, shared expert and mHC tensors; `mtp.2.norm.weight`, `mtp.2.markov_head.{embed,head}.weight`, `mtp.2.confidence_head.proj.weight`. `embed.weight` and EXL3 `head.*` are the target aliases. `norm.weight` is included as an explicit target dependency for loader inspection; the drafter final norm is `mtp.2.norm.weight`.

EXL3 representations: routed drafter experts K2, dense/attention/shared/main projection K5, target LM head K6; each includes trellis I16, suh/svh F16 and mul1 I32. Codebook: `mul1` (algorithmic runtime codebook, no separate checkpoint tensor). Remaining `attn.wo_a` FP8 and ue8m0 scales are preserved. K values follow the deployed config and drafter plan, not target expert K allocations.

Packed bytes are not a GPU fit claim: O2 must add dequantized runtime representations, padding, caches, scratch, CUDA graphs and bounded communication pools. The live Markov token cache is copied separately and compared byte-for-byte between both Sparks.

## Engram shard entries

| Tensor | Shape | Dtype | File | Absolute byte range |
|---|---|---|---|---|
| layers.1.engram.embed.weight | [384006168, 256] | F8_E4M3 | model-00047-of-00048.safetensors | [664, 98305579672] |
| layers.1.engram.embed.scale | [384006168, 8] | F8_E8M0 | model-00047-of-00048.safetensors | [98305579672, 101377629016] |
| layers.1.engram.q_weight | [4, 5120] | BF16 | model-00047-of-00048.safetensors | [101377629016, 101377669976] |
| layers.1.engram.k_weight | [4, 5120] | BF16 | model-00047-of-00048.safetensors | [101377669976, 101377710936] |
| layers.1.engram.wkv.weight | [25600, 6144] | F8_E4M3 | model-00047-of-00048.safetensors | [101377710936, 101534997336] |
| layers.1.engram.wkv.scale | [800, 192] | F8_E8M0 | model-00047-of-00048.safetensors | [101534997336, 101535150936] |
| layers.14.engram.embed.weight | [384016682, 256] | F8_E4M3 | model-00048-of-00048.safetensors | [672, 98308271264] |
| layers.14.engram.embed.scale | [384016682, 8] | F8_E8M0 | model-00048-of-00048.safetensors | [98308271264, 101380404720] |
| layers.14.engram.q_weight | [4, 5120] | BF16 | model-00048-of-00048.safetensors | [101380404720, 101380445680] |
| layers.14.engram.k_weight | [4, 5120] | BF16 | model-00048-of-00048.safetensors | [101380445680, 101380486640] |
| layers.14.engram.wkv.weight | [25600, 6144] | F8_E4M3 | model-00048-of-00048.safetensors | [101380486640, 101537773040] |
| layers.14.engram.wkv.scale | [800, 192] | F8_E8M0 | model-00048-of-00048.safetensors | [101537773040, 101537926640] |
