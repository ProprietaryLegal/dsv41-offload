#!/bin/bash
# WIN2-R2 (S92 FIX 1): coordinator chain abort. Brings production back after a chain stopped outside a window's own HOLD
# (python -m offload.launch.windows.chain abort). No chain marker: nothing to do (exit 0). A HOLD exists: ALERT, touch
# nothing (exit 4). Otherwise the strict W8 restore start|check|close under the reviewed scoped recovery's own HOLD, then
# the 15-minute post-close watch; the marker is cleared only on RESTORED + watch PASS (exit 0), else kept (exit 3).
# Usage: chain_abort.sh [OUT_DIR]   (default $C/CHAIN-ABORT-<UTC>; its summary.json is a valid PREV_FILE target)
set -euo pipefail
C=${DSV41_CHAIN_CAMPAIGN:-/opt/dsv41/campaigns/DSV41-OFFLOAD}
PY=${DSV41_CHAIN_PY:-/home/operator/miniconda3/bin/python3}
cd "$(dirname "$0")/../../.."
OUT=${1:-$C/CHAIN-ABORT-$(date -u +%Y%m%dT%H%M%SZ)}
echo "chain abort -> $OUT"
exec env PYTHONPATH="$PWD" "$PY" -m offload.launch.windows.chain abort --out "$OUT"
