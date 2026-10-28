#!/usr/bin/env bash
set -euo pipefail
source ~/.config/ndm/paths.sh
kx_root="$HOME/dsv41-offload/kx-fusedmoe"
kx_runner="$kx_root/src/offload/dspark/kernels/fusedmoe/run_locked.sh"
for stage in 0 1 2; do
    for rank in 0 1; do
        "$kx_runner" -m offload.dspark.kernels.fusedmoe.measure --stage "$stage" --rank "$rank" --variants 256 \
            --output "$kx_root/results/bench${stage}r${rank}split-final.json" \
            > "$kx_root/results/bench${stage}r${rank}split-final.log" 2>&1
    done
done
