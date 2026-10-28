#!/usr/bin/env bash
# Compact matched set for worker A/B on a running serve (coordinator only): short x3, 32K x3, 32K x3-concurrent x1, 150K x2.
#   offload/bench/compact.sh <LABEL> <URL> <OUTDIR>
set -euo pipefail; cd "$(dirname "$0")/../.."
L=${1:?label}; U=${2:?url}; O=${3:?outdir}; mkdir -p "$O"
B="python3 offload/bench/bench2.py --url $U --label $L --thinking --reasoning-effort high --priority 10 --temperature 1 --top-p 1 --data-dir ${BENCH_DATA:-/opt/dsv41/data/bench-data}"
$B --ctx 0 --workload legal --concurrency 1 --reps 3 --max-tokens 1500 --cache-mode cold --out "$O/$L-short.jsonl"
$B --ctx 32000 --workload summary --concurrency 1 --reps 3 --max-tokens 1500 --cache-mode cold --out "$O/$L-c32.jsonl"
$B --ctx 32000 --workload summary --concurrency 3 --reps 1 --max-tokens 1500 --cache-mode cold --out "$O/$L-c32x3.jsonl"
$B --ctx 150000 --workload synth --concurrency 1 --reps 2 --max-tokens 1500 --cache-mode cold --out "$O/$L-c150.jsonl"
echo "== $(date -u +%FT%TZ) $L compact done"
