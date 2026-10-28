#!/usr/bin/env bash
set -e
exec vllm serve '/srv/dsv41/legal-v2a' --served-model-name 'DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2' --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 --nnodes 2 --node-rank 0 --master-addr 10.202.26.2 --master-port 25000 \
  --gpu-memory-utilization 0.86 --max-model-len 327680 --max-num-seqs 4 \
  --max-num-batched-tokens 4096 --load-format auto \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":false,"disk_offload":true,"disk_threads":256}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --kernel-config '{"enable_flashinfer_autotune":false}' \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,2,5],[3,4,4]]}' \
  --kv-cache-memory-bytes 4026531840 \
  --limit-mm-per-prompt '{"image":4}' --mm-processor-cache-gb 0 --long-prefill-token-threshold 1024 --scheduling-policy priority --enable-prompt-tokens-details
