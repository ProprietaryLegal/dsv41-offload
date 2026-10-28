#!/usr/bin/env bash
# Prepared for the COORDINATOR. N4 itself uses only the assigned solo card.
set -euo pipefail
source ~/.config/ndm/paths.sh
[[ ${N4_PAIR_BY_COORD:-0} == 1 ]] || { echo 'coordinator-only N4 pair recipe' >&2; exit 1; }
root=${N4_ROOT:-$HOME/dsv41-offload/n4}
cd "$root/repo"
export PYTHONPATH=$root/repo CUDA_VISIBLE_DEVICES=
"$HOME/dsv41-offload/e/venv/bin/python" -m offload.dspark.native.tools.n4.identity_expectations \
  --receipt "$root/N3-DSPARK_COPY_DONE.json" \
  --expected offload/dspark/native/tools/n4/evidence/identities.json
export N3_ROOT=$root N3_PAIR_BY_COORD=1
exec bash offload/dspark/native/tools/n3/pair.sh
