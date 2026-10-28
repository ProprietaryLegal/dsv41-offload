#!/bin/bash
# WIN2 chain-aware version of the coordinator's prev-ok gate (campaign launch/prev-ok.sh), for adoption by the coordinator.
# Usage: chain_prev_ok.sh PREV_TMUX PREV_FILE|-
# Same as prev-ok: wait for PREV_TMUX, no HOLD, no stale containers. The previous window must have ended EITHER restored
# with watch PASS (normal) OR chained (WINDOW_CHAIN=1, summary 'chain' == 'CHAINED', marker present). Production must be
# serving K1 unless the chain marker says it is intentionally down (python -m offload.launch.windows.chain gate).
set -euo pipefail
PREV_TMUX=$1 PREV_FILE=$2
C=/opt/dsv41/campaigns/DSV41-OFFLOAD
cd "$(dirname "$0")/../../.."
while tmux has-session -t "$PREV_TMUX" 2>/dev/null; do sleep 30; done
if [ "$PREV_FILE" != - ]; then
  test -s "$PREV_FILE" || { echo "previous window never started"; exit 2; }
  python3 - "$(cat "$PREV_FILE")/summary.json" <<'PY' || { echo "previous window ended neither restored+watch PASS nor chained"; exit 2; }
import json, sys
s = json.load(open(sys.argv[1]))
chained = s.get('chain') == 'CHAINED'
restored = str(s.get('restore', {}).get('status', '')).startswith('RESTORED') and str(s.get('post_close_watch', {}).get('status', '')) == 'PASS'
assert chained or restored, (s.get('chain'), s.get('restore'), s.get('post_close_watch'))
PY
fi
MODE=$(PYTHONPATH=$PWD /home/operator/miniconda3/bin/python3 -m offload.launch.windows.chain gate) || exit 2
echo "chain gate: $MODE"
$C/launch/no-stale-containers.sh
