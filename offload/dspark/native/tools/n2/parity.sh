#!/usr/bin/env bash
# Detached tmux; full same-capture replay on the single authorized card.
set -euo pipefail
source ~/.config/ndm/paths.sh
root=${N2_ROOT:-$HOME/dsv41-offload/n2}
run=$root/repo/offload/dspark/native/tools/n2/run_locked.sh
for arm in k2-tp2 native-hybrid-kx native-hybrid-kx2; do
  "$run" "final-parity-$arm" -m offload.dspark.native.tools.native_parity \
    "$arm" emu "$root/results/final_parity_${arm}_emu.json"
done
printf 'rc=0\n' > "$root/logs/final-parity.done"
