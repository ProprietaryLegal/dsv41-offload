#!/usr/bin/env bash
set -euo pipefail
source ~/.config/ndm/paths.sh
root=${N2_ROOT:-$HOME/dsv41-offload/n2}
run=$root/repo/offload/dspark/native/tools/n2/run_locked.sh
for rank in 0 1; do
  for arm in k2-tp2 native-hybrid-kx native-hybrid-kx2; do
    "$run" "bench-$arm-r$rank" -m offload.dspark.native.tools.native_bench \
      "$arm" solo "$root/results/bench_${arm}_solo_r$rank.json" "$rank" 200
  done
done
