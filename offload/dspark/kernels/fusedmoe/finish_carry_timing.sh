#!/usr/bin/env bash
set -euo pipefail
source ~/.config/ndm/paths.sh
kx_root="$HOME/dsv41-offload/kx-fusedmoe"
kx_runner="$kx_root/src/offload/dspark/kernels/fusedmoe/run_locked.sh"
"$kx_runner" -m pytest -xq offload/dspark/kernels/fusedmoe/test_fusedmoe.py \
    > "$kx_root/results/test-delivery.log" 2>&1
for stage in 0 1 2; do
    for rank in 0 1; do
        "$kx_runner" -m offload.dspark.kernels.fusedmoe.measure --stage "$stage" --rank "$rank" --variants 131072 \
            --output "$kx_root/results/bench${stage}r${rank}carry-final.json" \
            > "$kx_root/results/bench${stage}r${rank}carry-final.log" 2>&1
    done
done
