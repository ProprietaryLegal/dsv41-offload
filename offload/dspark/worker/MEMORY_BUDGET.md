# DSpark worker memory budget (D2)

Measured on x86host with the worker as the systemd unit builds it (`server.build_core`: 8 request slots + 1 scratch,
CUDA graphs for 1-4 requests x n_out 1-5 at k = 5, each with a WANT_MARGIN twin, insert graphs for 8/16/32/64/128
rows). Table: commit bccdbf2 (25 graphs); the margin twins (4e3e9bd, 45 graphs) add 68 MiB: 8,152 MiB in total. Source: `DSparkDrafter.memory_table()`, torch's allocator, `nvidia-smi`, `/proc/<pid>/status`.

## GPU (RTX 3090 `GPU-c4e7a371`, 24 GiB)

| item | representation | MiB |
|---|---|---|
| token embedding [129,280 x 5,120] | bf16 (checkpoint bytes) | 1,262.5 |
| vocabulary head [5,120 -> 129,280] | EXL3 K6 trellis + suh/svh | 473.7 |
| main_proj [15,360 -> 5,120] | EXL3 K5 | 46.9 |
| dense per stage x 3 (wq_a, wkv, wq_b, wo_b, shared w1/w3/w2) | EXL3 K5 | 230.2 |
| wo_a x 3 [8 x 1,024 x 4,096] | FP8 e4m3 bytes + fp32 UE8M0 block scales (checkpoint bytes; dequantized in the kernel) | 96.4 |
| routed experts x 3 (128 experts x w1/w3/w2) | EXL3 K2 banks | 3,256.3 |
| norms, mHC fn/scale/base, gate weight + bias x 3 | fp32 / bf16 | 18.8 |
| Markov embed + head [129,280 x 256] x 2 | bf16 | 126.2 |
| confidence head, final norm, main_norm | bf16 / fp32 | < 0.1 |
| **weights subtotal** | | **5,511.0** |
| Markov bias cache: 8,192 rows x 129,280 | bf16 (+ int32 token -> row map) | 2,020.5 |
| RoPE cos/sin, 329,728 positions x 64 | fp32 | 80.5 |
| per-request state: window KV 3 stages x 128 x 512 + positions | bf16 / int64, 0.38 MiB per slot x 9 | 3.4 |
| coop MoE scratch (exllamav3) | fp16 / fp32 | 6.3 |
| activations + CUDA graph pool (25 graphs share one pool) | | ~18 |
| torch allocator total: allocated / reserved | | 7,639 / 7,700 |
| CUDA context, kernel images (two exllamav3 modules, Triton, cuBLAS), exllamav3 lock buffer | | ~384 |
| **process total (nvidia-smi)** | | **8,084 (7.9 GiB, 33 % of the card); 8,152 with the margin graphs** |

Scaling: the window is fixed at 128 positions, so per-request state does not grow with context length (0.38 MiB per
slot); more slots cost almost nothing. The Markov cache is the one discretionary item: 4,096 rows (the serve's size)
would save 1,010 MiB for a ~6-point lower cache hit rate (PERF.md). Graph memory is shared by all 25 graphs (they are
never replayed concurrently); capture takes 0.5 s.

## Host (x86host)

| item | size |
|---|---|
| pinned staging buffer for inserted rows (512 rows x 30,720 B, the contract's max rows per STEP) | 15 MiB |
| ofl server request / response buffers (CONTRACT 5: max request 16 MiB, max response 64 KiB, inflight 1; libofl registers them for RDMA per connection, max_conns 4; the Spark proxy holds one) | 16 MiB + 64 KiB per connection, at most 4 x |
| TCP stub receive buffer (only with `--tcp-stub`) | 16 MiB |
| process RSS, steady state | 1.28 GB (0.75 GB anonymous, 0.43 GB file-backed) |
| process RSS, peak during start-up (identity verification of the checkpoint bytes and the upload of 5.5 GB of tensors) | 6.9 GB |

x86host had ~305 GB available during these runs; the worker never needs swap.
