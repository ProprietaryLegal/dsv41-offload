#!/usr/bin/env bash
# Verify the chosen default on the final pointer ABI/source, after stride fixes.
set -euo pipefail
source ~/.config/ndm/paths.sh
kx_root="$HOME/dsv41-offload/kx-fusedmoe"
kx_runner="$kx_root/src/offload/dspark/kernels/fusedmoe/run_locked.sh"
for stage in 0 1 2; do
    for rank in 0 1; do
        "$kx_runner" -m offload.dspark.kernels.fusedmoe.qualify --stage "$stage" --rank "$rank" --variant 131072 \
            --output "$kx_root/results/qual${stage}r${rank}v131072-delivery-final.json" \
            > "$kx_root/results/qual${stage}r${rank}v131072-delivery-final.log" 2>&1
    done
done
