#!/usr/bin/env bash
set -euo pipefail
source ~/.config/ndm/paths.sh
kx_root="$HOME/dsv41-offload/kx-fusedmoe"
kx_runner="$kx_root/src/offload/dspark/kernels/fusedmoe/run_locked.sh"
"$kx_runner" -m pytest -xq offload/dspark/kernels/fusedmoe/test_fusedmoe.py \
    > "$kx_root/results/test-complete.log" 2>&1
for stage in 0 1 2; do
    for rank in 0 1; do
        for variant in 65536 131072; do
            "$kx_runner" -m offload.dspark.kernels.fusedmoe.qualify --stage "$stage" --rank "$rank" --variant "$variant" \
                --output "$kx_root/results/qual${stage}r${rank}v${variant}-final.json" \
                > "$kx_root/results/qual${stage}r${rank}v${variant}-final.log" 2>&1
        done
    done
    "$kx_runner" -m offload.dspark.kernels.fusedmoe.baseline --stage "$stage" --rank 0 \
        --output "$kx_root/results/reuse-screen${stage}-all.json" \
        > "$kx_root/results/reuse-screen${stage}-all.log" 2>&1
done
