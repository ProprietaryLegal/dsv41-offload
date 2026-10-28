#!/usr/bin/env bash
# Immutable detached queue; each step separately acquires/releases the GPU.
set -euo pipefail
source ~/.config/ndm/paths.sh
kx_root="$HOME/dsv41-offload/kx-fusedmoe"
kx_runner="$kx_root/src/offload/dspark/kernels/fusedmoe/run_locked.sh"
for stage in 0 1 2; do
    for rank in 0 1; do
      for variant in 16 64; do
        "$kx_runner" -m offload.dspark.kernels.fusedmoe.qualify --stage "$stage" --rank "$rank" --variant "$variant" \
            --output "$kx_root/results/qual${stage}r${rank}v${variant}-final.json" \
            > "$kx_root/results/qual${stage}r${rank}v${variant}-final.log" 2>&1
      done
    done
    "$kx_runner" -m offload.dspark.kernels.fusedmoe.measure --stage "$stage" --rank 0 \
        --output "$kx_root/results/bench${stage}-final.json" \
        > "$kx_root/results/bench${stage}-final.log" 2>&1
    "$kx_runner" -m offload.dspark.kernels.fusedmoe.baseline --stage "$stage" --rank 0 \
        --output "$kx_root/results/baseline${stage}-final.json" \
        > "$kx_root/results/baseline${stage}-final.log" 2>&1
done
