# Build record: what the production deployment runs (as of 2026-10-27)

Labels: **MEASURED** on our hardware; **ESTIMATED** by arithmetic. Host names are role names: `sparkhead` = DGX Spark
rank 0 (API), `sparkworker` = DGX Spark rank 1, `x86host` = the x86 offload host with the RTX 3090s, `controller` = the
workstation that runs the measurement harness. Addresses are placeholders (`HEAD_IP`, `LAN_IP`, `TAILNET_IP`); the
`10.202.x.x` point-to-point link addresses are private RoCE links and are kept as they are. No secrets, keys or tokens.

## Pinned versions

| Item | Value |
|---|---|
| Production branch | `production` in this repository = private commit `596ba1cf4e3b`, tag `prod-k1-327-ops1-20261026` |
| Serving image | `ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5`, repo digest `sha256:8fc2c99a3562e99d81d2a05b8345d12726def0a027410d14d0eb0763de8309e4`, image id `sha256:435a5236ce3a7db9d45f5a2c483817d60ea629c4ea0a0b3af33da2ef8122334b` |
| Inside the image | vLLM `0.31.1.dev0+g6e517b15.dsv41gb10` (sudoingX recipe v1.1 + HeNryous patches), torch 2.13.0+cu130, CUDA 13.0.2 |
| Launcher | sudoingX `dgx-spark-deepseek-v41` @ `cc59c36`, through `start.sh` → `offload/promote/k1/start_checked.sh` (K1 start guard) → `serve.sh` |
| Weights | [ProprietaryLegal/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark) (`legal-v2a`); baseline `QUANT_REV=86f852533ec94d1a537fea41fd9c01d6b0f6eb50`, base `BASE_REV=1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Drafter | DSpark (DeepSeek's in-checkpoint block drafter, tensors `mtp.*`; V4.1 has no V3-style MTP module), upstream EXL3 "K2" quantization, worker commit `fb68c52`, TP2 on the `x86host` NVLink RTX 3090 pair |
| Drafter identities | remote drafter `be2c1436…43bd`, Markov head `0b6e71e6…a19a` |
| Restore script (W8) | sha256 `d1c4479dbfbd90bc2036c7bf1561db5b564c2b162a7cde4a121e46bfb9f86a49` |
| Host software (both Sparks) | DGX OS 7.6.0, kernel 7.0.0-1019-nvidia, driver 580.178.04, ConnectX-7 28.50.1002 |

## Serving settings (production branch)

| Setting | Value |
|---|---|
| Max request length | 327,680 tokens |
| Sequences | 4 |
| Prefill chunk (`MAX_BATCH`) | 4,096; `--long-prefill-token-threshold` 1,024 |
| KV | 6,442,450,944 bytes per rank; K1 exact packing → 2,502,600 tokens (46,603 blocks × 138,240 B) |
| Speculation | `{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,2,5],[3,4,4]]}`, per-request depth policy `conf`, adaptive verify window (`VLLM_DSV41_ADAPTIVE_K=1`) |
| Scheduling | `--scheduling-policy priority`; priority ≥ 5 is background (prefill every 24th step while foreground runs) |
| K1 switches | `VLLM_DSV41_KV_PAIR=3`, `VLLM_DSV41_DROP_MTP_BUFFER=1`, `VLLM_EXL3_TF_PROMPT_TRANSIENT=1`, `VLLM_EXL3_TF_PROMPT_ALIAS=1`, `VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4` |
| Tensor-parallel fabric | ConnectX-7 200 Gb/s RoCE between the Sparks (`10.202.26.x`, second PCIe function `10.202.27.x`), NCCL over both functions; RDMA ring poller pinned to one core |
| Remote drafter | `VLLM_DSPARK_REMOTE=o1rdma://10.202.28.1:18620`, policy `capacity` (worker down → correct target-only rounds), timeout 50 ms, async process proxy |
| Remote Engram | both tables in `x86host` RAM (~189 GiB resident) over RDMA (`10.202.28.x`, `10.202.29.x`, 40 Gb/s), timeout 100 ms, exact local fallback |
| GPU clock | repository default 2,150 MHz; production runs 2,300 MHz from a host setting since 2026-10-27 (fold into the repository pending), with a thermal watch that dials back to 2,150 |
| Display reservation (rank 0) | 512 MiB (firmware setting, from 2 GB): +1,532 MiB usable memory |

### Overlays mounted into the image

| Overlay | Content |
|---|---|
| K1 KV packing | 13 engine/plugin files (attention, compressed/quantized KV cache, dspark, flashinfer sparse, model, KV cache utils and interface, worker utils, model runner, memprobe, TF prompt and experts); a zero-fuzz rebuild reproduces all 13 mounted hashes on both ranks |
| Remote drafter | `remote_dspark` package + `spec_decode/__init__.py`; transport libraries |
| Remote Engram | `engram.py` overlay |
| Warm-up | host-side `offload/launch/warmup_serve.sh` + `warmup.py`: 4 waves, 8.3 s |

## Measured performance (control archive)

149 control runs (596 archived metric rows) from 11 measurement windows on 2026-10-26/27, production commit `596ba1c`,
2,150 MHz, temperature 1, top_p 1, seed 42, 512-token answers unless noted. Medians per cell; the mixed long-prefill cell is not shown. MEASURED.

| Cell | Runs | Decode (tok/s per request) | ms per verify round | Accepted drafts per round | Time to first token |
|---|---:|---:|---:|---:|---:|
| short prompt, 1 user | 38 | 57.5 | 39.6 | 1.27 | 0.8 s |
| 32K prompt, 1 user | 38 | 51.8 | 38.9 | 1.01 | 14.5 s |
| 150K prompt, 1 user | 14 | 51.7 | 40.2 | 1.03 | 69.2 s |
| 32K, 3 users | 12 | 30.2 | 80.8 | 1.10 | 51.5 s |
| 32K, 4 users | 5 | 24.8 | 92.1 | 1.11 | 64.6 s |
| 32K, 2 users (1,500-token answers) | 5 | 35.5 | 58.1 | 1.07 | 35.1 s |

Other measured facts:
- Speculation on vs off (remote DSpark drafter, 2026-10-22): 56.0 vs 41.9 tok/s short, 64.6 vs 41.0 at 32K.
- Prefill at 150K: about 2,170 tok/s (time to first token 68.7–69.7 s, cold).
- Drafter wait per single-user round: 2.7–3.0 ms, the whole remote round trip (worker 2.53–2.61 ms, wire 0.04–0.14 ms,
  proxy 0.10–0.15 ms). One graph per drafter step saves 0.11–0.12 ms; no change reaches a measurable 0.62 ms.
- Prefill chunk 16,384 at 6 GiB KV does not fit rank 0: about 6.7 GiB extra, stopped by the memory guard at the first
  32K prompt. Chunk 8,192 fits (lowest MemAvailable 5.8 GiB rank 0, 5.5 GiB rank 1).
- Acceptance depends mostly on the prompt: one 32K summary prompt holds 1.59 accepted drafts per round over 3 days
  (70 runs), one short drafting prompt 1.00 (77 runs).

## Build and test commands

```bash
# serving (rank 0; starts rank 1 through serve.sh)
./start.sh            # K1 start guard → serve.sh; waits for /v1/models
./status.sh           # both ranks, drafter, Engram
./stop.sh
# tests (CPU only, no host contact needed): see reports/tests/TESTS.md for per-branch results
python3 -m pytest -q offload
```
