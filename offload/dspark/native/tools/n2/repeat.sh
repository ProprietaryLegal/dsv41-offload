#!/usr/bin/env bash
# Launch in detached tmux. One lock acquisition per benchmark, <=14min each.
set -euo pipefail
source ~/.config/ndm/paths.sh
root=${N2_ROOT:-$HOME/dsv41-offload/n2}
run=$root/repo/offload/dspark/native/tools/n2/run_locked.sh
out=${N2_REPEAT_OUTPUT:-$root/results/repeated}
mkdir -p "$out"
for rep in 1 2 3; do
  case $rep in
    1) arms=(k2-tp2 native-hybrid-kx native-hybrid-kx2); ranks=(0 1) ;;
    2) arms=(native-hybrid-kx2 native-hybrid-kx k2-tp2); ranks=(1 0) ;;
    3) arms=(native-hybrid-kx k2-tp2 native-hybrid-kx2); ranks=(0 1) ;;
  esac
  for rank in "${ranks[@]}"; do
    for arm in "${arms[@]}"; do
      "$run" "repeat${rep}-${arm}-r${rank}" -m offload.dspark.native.tools.native_bench \
        "$arm" solo "$out/repeat${rep}_${arm}_r${rank}.json" "$rank" 200
      echo "$(date -u +%FT%TZ) complete rep=$rep rank=$rank arm=$arm" >> "$root/logs/repeat-progress.log"
    done
  done
done
printf 'rc=0\n' > "$root/logs/repeats.done"
