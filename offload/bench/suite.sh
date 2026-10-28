#!/usr/bin/env bash
# DSV41-OFFLOAD screening suite: the same prompts, sampling (T=1, top_p=1), thinking at effort "high" (= local-law's 75),
# priority 10 (local-law's), cold salts unless warm, fixed 1,500-token budgets plus one full task, for every configuration.
#   offload/bench/suite.sh <LABEL> <URL> <OUTDIR> [stage ...]     stages: short c32 warm32 c150 warm150 full (default: all); finalists c256 c300 cap4
# Coordinator only, inside an open window.
set -euo pipefail; cd "$(dirname "$0")/../.."
L=${1:?label}; U=${2:?url}; O=${3:?outdir}; shift 3
stages=${*:-short c32 warm32 c150 warm150 full}
mkdir -p "$O"
B="python3 offload/bench/bench2.py --url $U --label $L --thinking --reasoning-effort high --priority 10 --temperature 1 --top-p 1 --data-dir ${BENCH_DATA:-/opt/dsv41/data/bench-data} ${BENCH_EXTRA:-}"
for s in $stages; do
  echo "== $(date -u +%FT%TZ) $L stage $s"
  case $s in
    short)   $B --ctx 0 --workload legal --concurrency 1 --reps 3 --max-tokens 1500 --cache-mode cold --out "$O/$L-short.jsonl" ;;
    c32)     $B --ctx 32000 --workload summary --concurrency 1 2 3 --mixed --mixed-short-ctx 8000 --reps 2 --max-tokens 1500 --cache-mode cold --out "$O/$L-c32.jsonl" ;;
    warm32)  $B --ctx 32000 --workload summary --concurrency 1 --reps 3 --max-tokens 1500 --cache-mode warm --out "$O/$L-warm32.jsonl" ;;
    c150)    $B --ctx 150000 --workload synth --concurrency 1 --mixed --mixed-short-ctx 8000 --reps 2 --max-tokens 1500 --cache-mode cold --out "$O/$L-c150.jsonl" ;;
    warm150) $B --ctx 150000 --workload synth --concurrency 1 --reps 2 --max-tokens 1500 --cache-mode warm --out "$O/$L-warm150.jsonl" ;;
    c256)    $B --ctx 256000 --workload synth --concurrency 1 --reps 1 --max-tokens 1500 --cache-mode cold --out "$O/$L-c256.jsonl" ;;
    c300)    $B --ctx 300000 --workload synth --concurrency 1 --reps 1 --max-tokens 1500 --cache-mode cold --out "$O/$L-c300.jsonl" ;;
    cap4)    $B --ctx 300000 --workload synth --concurrency 4 --reps 1 --max-tokens 1500 --cache-mode cold --out "$O/$L-cap4.jsonl" ;;  # 4 x 300K: needs ~1.21M KV tokens (A holds 989,460)
    full)    $B --ctx 32000 --workload summary --concurrency 1 --reps 1 --max-tokens 16384 --cache-mode cold --out "$O/$L-full.jsonl" ;;
    *) echo "unknown stage $s" >&2; exit 2 ;;
  esac
done
echo "== $(date -u +%FT%TZ) $L suite done"
