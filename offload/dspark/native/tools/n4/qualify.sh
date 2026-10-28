#!/usr/bin/env bash
# Actual solo shards, all40 worker graph shapes and2000replays each.
set -euo pipefail
source ~/.config/ndm/paths.sh
root=${N4_ROOT:-$HOME/dsv41-offload/n4}
export OFL_LIB=$root/lib/libofl.so DSPARK_P2P_PREBUILT=$root/p2p
for arm in native-packed native-expanded native-hybrid-kx native-hybrid-kx2; do
  for rank in 0 1; do
    name=qualified-${arm}-r${rank}
    out=$root/results/$name.json
    [[ ! -e $out ]] || { echo "existing receipt $out: inspect before resuming" >&2; exit 1; }
    bash "$root/repo/offload/dspark/native/tools/n4/run_locked.sh" "$name" \
      -m offload.dspark.native.tools.n3.shard "$arm" --rank "$rank" --out "$out"
    echo "$(date -u +%FT%TZ) complete $name" >> "$root/logs/qualification-progress.log"
  done
done
echo rc=0 > "$root/logs/qualification.done"
