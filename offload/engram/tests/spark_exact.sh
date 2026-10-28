#!/usr/bin/env bash
# spark_exact.sh RANK: remote (x86host) vs local (this Spark's shards, O_DIRECT) Engram rows for this rank's real row
# ranges, both layers, RDMA then TCP; then refusal cases and a short latency bench. Run on the Spark host under
# idle_gate.sh. Ranges from layout.py (config.json of legal-v2a, TP2).
set -euo pipefail
RANK=$1
B=${B:-/tmp/o1/bin}; M=${M:-/tmp/o1/manifest.json}; D=${D:-/srv/dsv41/legal-v2a}
ITERS=${ITERS:-3}; BITERS=${BITERS:-5}
if [ "$RANK" = 0 ]; then SRV=10.202.28.1 LOC=10.202.28.2 L1="0 192001740" L14="0 192007016"
else SRV=10.202.29.1 LOC=10.202.29.2 L1="192001740 192004428" L14="192007016 192009666"; fi
run() {  # shard rows(start n) backend mode iters label
  set -- "$@"
  local shard=$1 start=$2 n=$3 be=$4 mode=$5 it=$6 label=$7
  "$B/engram_check" --shard "$D/$shard" --manifest "$M" --server "$SRV" --local "$LOC" --port 18610 --backend "$be" \
    --row-start "$start" --num-rows "$n" --mode "$mode" --iters "$it" --seed $((RANK * 100 + 7)) --label "$label"
}
for be in rdma tcp; do
  run model-00047-of-00048.safetensors $L1 $be exact "$ITERS" "rank$RANK-L1"
  run model-00048-of-00048.safetensors $L14 $be exact "$ITERS" "rank$RANK-L14"
  run model-00048-of-00048.safetensors $L14 $be negative 1 "rank$RANK-L14"
done
for be in rdma tcp; do run model-00047-of-00048.safetensors $L1 $be bench "$BITERS" "rank$RANK-L1"; done
