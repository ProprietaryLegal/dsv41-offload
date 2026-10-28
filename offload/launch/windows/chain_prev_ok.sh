#!/bin/bash
# WIN2 chain-aware version of the coordinator's prev-ok gate (campaign launch/prev-ok.sh), for adoption by the coordinator.
# Usage: chain_prev_ok.sh PREV_TMUX PREV_FILE|-
# Same as prev-ok: wait for PREV_TMUX, no HOLD, no stale containers. The previous window must have ended EITHER restored
# with watch PASS (normal) OR chained (WINDOW_CHAIN=1, summary 'chain' == 'CHAINED', marker present). Production must be
# serving K1 unless the chain marker says it is intentionally down (python -m offload.launch.windows.chain gate).
# Exit 0: admitted (prints "chain gate: chained|production").
# Exit 2: refused with NO chain marker (production is not down because of a chain; as prev-ok.sh).
# Exit 3: "CHAIN STOPPED: PRODUCTION DOWN" (S92 FIX 1): any refusal while the chain marker is present. Production was left
#         down by the previous chained window, so the gate runs the chain abort before exiting (chain_abort.sh: strict W8
#         start|check|close + post-close watch when no HOLD exists, marker cleared only on RESTORED + watch PASS; alert and
#         touch nothing when a HOLD exists). The next window must not start; read the abort's summary.json.
# Test-only overrides: DSV41_CHAIN_CAMPAIGN, DSV41_CHAIN_PY, DSV41_CHAIN_ABORT, DSV41_CHAIN_MARKER (chain.py).
set -uo pipefail  # no -e: every refusal goes through refuse()
PREV_TMUX=$1 PREV_FILE=$2
C=${DSV41_CHAIN_CAMPAIGN:-/opt/dsv41/campaigns/DSV41-OFFLOAD}
PY=${DSV41_CHAIN_PY:-/home/operator/miniconda3/bin/python3}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../../.." || exit 2
chain() { PYTHONPATH=$PWD "$PY" -m offload.launch.windows.chain "$@"; }
refuse() {
  # Only an explicit ABSENT counts as "no marker"; an unreadable answer is treated as present (restore, never ignore).
  if [ "$(chain present 2>/dev/null)" = ABSENT ]; then echo "$1"; exit 2; fi
  echo "CHAIN STOPPED: PRODUCTION DOWN ($1)"
  "${DSV41_CHAIN_ABORT:-$HERE/chain_abort.sh}"
  echo "chain abort exit $?; the chain is stopped (exit 3)"
  exit 3
}
while tmux has-session -t "$PREV_TMUX" 2>/dev/null; do sleep 30; done
if [ "$PREV_FILE" != - ]; then
  test -s "$PREV_FILE" || refuse "previous window never started"
  "$PY" - "$(cat "$PREV_FILE")/summary.json" <<'PY' || refuse "previous window ended neither restored+watch PASS nor chained"
import json, sys
s = json.load(open(sys.argv[1]))
chained = s.get('chain') == 'CHAINED'
restored = str(s.get('restore', {}).get('status', '')).startswith('RESTORED') and str(s.get('post_close_watch', {}).get('status', '')) == 'PASS'
assert chained or restored, (s.get('chain'), s.get('restore'), s.get('post_close_watch'))
PY
fi
MODE=$(chain gate) || refuse "chain gate refused"
echo "chain gate: $MODE"
"$C/launch/no-stale-containers.sh" || refuse "stale serve/experiment container check refused"
