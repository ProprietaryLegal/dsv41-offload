#!/usr/bin/env bash
set -euo pipefail
source ~/.config/ndm/paths.sh
kx_root="$HOME/dsv41-offload/kx-fusedmoe"
kx_runner="$kx_root/src/offload/dspark/kernels/fusedmoe/run_locked.sh"
"$kx_runner" -m pytest -xq offload/dspark/kernels/fusedmoe/test_fusedmoe.py \
    > "$kx_root/results/test-final.log" 2>&1
for stage in 0 1 2; do
    "$kx_runner" -m offload.dspark.kernels.fusedmoe.measure --stage "$stage" --rank 1 \
        --output "$kx_root/results/bench${stage}r1-final.json" \
        > "$kx_root/results/bench${stage}r1-final.log" 2>&1
done
