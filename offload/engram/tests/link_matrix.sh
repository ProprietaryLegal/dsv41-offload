#!/usr/bin/env bash
# link_matrix.sh: ofl ping-pong over one x86host<->Spark direct link, TCP and RDMA, at the Engram request shapes.
# Run ON the Spark host (client side) under idle_gate.sh; the x86host server must already be up:
#   x86host:  ofl_pingpong server --bind 10.202.28.1 --port 18600 --seconds 120
#   spark:  idle_gate.sh -- link_matrix.sh <pingpong> <server_ip> <local_ip> <label> > results.jsonl
# Shapes: 64 B echo; rows R -> request R*8 B ids, response R*264 B (256 B weights + 8 B scales a row, 2 ranges).
set -euo pipefail
PP=$1 SRV=$2 LOC=$3 LABEL=$4
for be in rdma tcp; do
  "$PP" client --backend "$be" --server "$SRV" --port 18600 --local "$LOC" --req 64 --iters 2000 --warmup 200 --check --label "$LABEL-echo64"
  for rows in 288 4096 16384 49152; do
    iters=$(( rows <= 288 ? 1000 : rows <= 4096 ? 200 : 40 ))
    "$PP" client --backend "$be" --server "$SRV" --port 18600 --local "$LOC" --req $((rows * 8)) --resp $((rows * 264)) \
      --segs 2 --iters "$iters" --warmup 10 --check --label "$LABEL-rows$rows"
  done
done
