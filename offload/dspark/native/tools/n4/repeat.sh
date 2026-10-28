#!/usr/bin/env bash
# Both arithmetic policies on the reassigned solo UUID, separate bounded holds.
set -euo pipefail
source ~/.config/ndm/paths.sh
root=${N4_ROOT:-$HOME/dsv41-offload/n4}
out=${N4_REPEAT_OUTPUT:-$root/results/repeated}
mkdir -p "$out"
export OFL_LIB=$root/lib/libofl.so DSPARK_P2P_PREBUILT=$root/p2p
for rep in 1 2 3; do
  case $rep in
    1) arms=(native-packed native-expanded native-hybrid-kx native-hybrid-kx2); ranks=(0 1); sides=(before repo) ;;
    2) arms=(native-hybrid-kx2 native-hybrid-kx native-expanded native-packed); ranks=(1 0); sides=(repo before) ;;
    3) arms=(native-hybrid-kx native-packed native-hybrid-kx2 native-expanded); ranks=(0 1); sides=(before repo) ;;
  esac
  for rank in "${ranks[@]}"; do
    for arm in "${arms[@]}"; do
      for side in "${sides[@]}"; do
        name=repeat${rep}-${side}-${arm}-r${rank}
        receipt=$out/$name.json
        [[ ! -e $receipt ]] || { echo "existing receipt $receipt: inspect before resuming" >&2; exit 1; }
        N4_TREE=$side bash "$root/repo/offload/dspark/native/tools/n4/run_locked.sh" "$name" \
          -m offload.dspark.native.tools.native_bench "$arm" solo "$receipt" "$rank" 200
        echo "$(date -u +%FT%TZ) complete $name" >> "$root/logs/repeat-progress.log"
      done
    done
  done
done
echo rc=0 > "$root/logs/repeats.done"
