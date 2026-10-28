#!/usr/bin/env bash
# both_links.sh: both Sparks look up from x86host at the same time (each rank over its own link, as in a TP2 step),
# RDMA, real row ranges of layer 1, then the same alone for comparison. Run from Controller; every Spark-side command runs
# under that host's idle_gate.sh. Output: offload/results/o1/both_links_<ts>.jsonl
set -euo pipefail
cd "$(dirname "$0")"
OUT=../../results/o1/both_links_$(date -u +%Y%m%dT%H%M%SZ).jsonl
B=/tmp/o1/bin; M=/tmp/o1/manifest.json; D=/srv/dsv41/legal-v2a
cmd() {  # rank label
  if [ "$1" = 0 ]; then echo "$B/idle_gate.sh -n 3 -w 60 -- $B/engram_check --shard $D/model-00047-of-00048.safetensors --manifest $M --server 10.202.28.1 --local 10.202.28.2 --backend rdma --row-start 0 --num-rows 192001740 --mode bench --no-local --iters 40 --sizes 288,4096,49152 --label $2"
  else echo "$B/idle_gate.sh -n 3 -w 60 -- $B/engram_check --shard $D/model-00047-of-00048.safetensors --manifest $M --server 10.202.29.1 --local 10.202.29.2 --backend rdma --row-start 192001740 --num-rows 192004428 --mode bench --no-local --iters 40 --sizes 288,4096,49152 --label $2"; fi
}
ssh spark "$(cmd 0 alone-rank0)" >> "$OUT"
ssh sparkworker "$(cmd 1 alone-rank1)" >> "$OUT"
ssh spark "$(cmd 0 together-rank0)" >> "$OUT" & a=$!
ssh sparkworker "$(cmd 1 together-rank1)" >> "$OUT" & b=$!
wait $a; wait $b
python3 - "$OUT" <<'PY'
import json, sys
for l in open(sys.argv[1]):
    d = json.loads(l)
    c = d["remote_cold_us"]
    print(f"{d['label']:16} {d['rows_per_batch']:6} rows  p50 {c['p50']:8.1f}  p99 {c['p99']:8.1f}  server p50 {d['server_us']['p50']:7.1f} us")
PY
echo "raw: $OUT"
