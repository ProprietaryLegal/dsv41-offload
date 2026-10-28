#!/usr/bin/env bash
# W10 R2: five-repeat median gate; dry by default.
# --execute is for Deepseek/coordinator only; inputs default to the R2 receipt.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
: "${HARNESS_PYTHON:?}"
# D2_SMOKE_REPO can name a coordinator-reviewed clean worktree after integration.
cd "${D2_SMOKE_REPO:-/opt/dsv41/repos/.worktrees/dsv41-smoke}"
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/D2-SMOKE-R2
mkdir -p "$TMPDIR"
exec "$HARNESS_PYTHON" -m offload.dspark.d2.smoke_window "$@"
