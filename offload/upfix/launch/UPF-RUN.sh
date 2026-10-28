#!/bin/bash
# R-UPF TEMPLATE (2026-10-27): one UPF window, UPF-W1 or the UPF-0091 / UPF-TOPK fallbacks (offload/upfix/WINDOW.md).
# DO NOT QUEUE until a runner exists. No runner executes R-UPF's config-only arms yet (R-UPF report), and this script
# refuses BEFORE any gate, ssh or docker while offload.upfix.window is absent from the run checkout.
# Usage (copy to the campaign launch/ directory): UPF-RUN.sh WINDOW COMMIT PREV_TMUX PREV_FILE|-
#   WINDOW: UPF-W1 | UPF-0091 | UPF-TOPK. COMMIT: the staged, reviewed commit of the run checkout.
# Gate: offload/launch/windows/chain_prev_ok.sh, always (it admits 'production' or 'chained' mode).
#   exit 2: refused with no chain marker; not started.
#   exit 3: CHAIN STOPPED. Production was down under the chain marker and the gate has ALREADY run the chain abort
#           (strict W8 restore + watch). Not started; the chain is over. Read the abort's summary.json.
# Chain: WINDOW_CHAIN=1 for every window of a chain, WINDOW_CHAIN_LAST=1 on the last one.
set -euo pipefail
W=$1 R=$2 PREV_TMUX=$3 PREV_FILE=$4
case "$W" in UPF-W1|UPF-0091|UPF-TOPK) ;; *) echo "window $W? (UPF-W1 | UPF-0091 | UPF-TOPK)"; exit 2;; esac
C=/opt/dsv41/campaigns/DSV41-OFFLOAD
PY=/home/operator/miniconda3/bin/python3
cd "${UPF_RUN_CHECKOUT:-/opt/dsv41/repos/.worktrees/dsv41-upf-run}"
[ "$(git rev-parse HEAD)" = "$R" ] && [ -z "$(git status --porcelain --untracked-files=no)" ] \
  || { echo "run checkout not clean at $R; $W not started"; exit 2; }
PYTHONPATH=$PWD "$PY" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("offload.upfix.window") is None)' \
  || { echo "UPF runner offload.upfix.window not built (R-UPF report); $W not started"; exit 2; }
rc=0
bash offload/launch/windows/chain_prev_ok.sh "$PREV_TMUX" "$PREV_FILE" || rc=$?
case $rc in
  0) ;;
  3) echo "CHAIN STOPPED (chain_prev_ok exit 3): the chain abort has run; $W not started"; exit 3 ;;
  *) echo "gate refused (chain_prev_ok exit $rc); $W not started"; exit "$rc" ;;
esac
source ~/.config/ndm/paths.sh
unset TMUX TMUX_PANE
export WINDOW_COORDINATOR=Deepseek/coordinator CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
mkdir -p $C/scratch/upfw; export TMPDIR=$C/scratch/upfw TMUX_TMPDIR=$C/scratch/upfw
export PYTHONPATH=$PWD:$PWD/offload/dspark
OUT=$C/$W-W-$(date -u +%Y%m%dT%H%M%SZ); echo "$OUT" > $C/scratch/$W-current.txt
exec "$PY" -m offload.upfix.window --execute --window "$W" --out "$OUT" --expected-experiment-commit "$R" \
  --remote-repo dsv41-deploy-upf
