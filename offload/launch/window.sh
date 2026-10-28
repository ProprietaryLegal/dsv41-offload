#!/usr/bin/env bash
# DSV41-OFFLOAD maintenance window on the Sparks (run on sparkhead from ~/dsv41-deploy).
#   offload/launch/window.sh open "<purpose>"   wait until the serve is idle, write HOLD (pauses the local-law K24 queue),
#                                               start a placeholder `vllm serve [DSV41-OFFLOAD hold]` process on both nodes
#                                               (the NVO drama3 waiter treats a node with a `vllm serve` process as busy)
#   offload/launch/window.sh close             production must answer on :8000 with /srv/dsv41/legal-v2a loaded; then
#                                               remove the placeholders and HOLD
#   offload/launch/window.sh status
set -euo pipefail; cd "$(dirname "$0")/../.."; set -a; . ./dsv41.env; set +a
HOLDF=$HOME/dsv41-deploy/HOLD   # the production checkout; the K24 driver reads exactly this path
# idle: the metrics fetch must succeed (HTTP 2xx) and BOTH request gauges must be present and zero; anything else is busy
# (S5R fix 3: an empty or partial page used to read as idle)
idle() {
  local m; m=$(curl -sf -m 5 http://localhost:8000/metrics) || return 1
  printf '%s\n' "$m" | awk '/^vllm:num_requests_running\{/{r=1; s+=$NF} /^vllm:num_requests_waiting\{/{w=1; s+=$NF}
    END{exit !(r && w && s==0)}'
}
holdproc() { ssh -o BatchMode=yes "$1" "tmux has-session -t dsv41-hold 2>/dev/null || tmux new-session -d -s dsv41-hold 'exec -a \"vllm serve [DSV41-OFFLOAD hold]\" sleep 86400'"; }
case "${1:-status}" in
  open)
    purpose=${2:?purpose}
    # a running serve must be drained first; a stopped one is idle by definition
    if curl -sf -m 5 http://localhost:8000/v1/models >/dev/null; then
      for i in $(seq 1 240); do idle && break; sleep 15; done
      idle || { echo "serve still busy after 60 min; not opening" >&2; exit 1; }
    fi
    echo "$(date -u +%FT%TZ) DSV41-OFFLOAD (session coordinator): $purpose" > "$HOME/dsv41-deploy/HOLD"
    for n in $HEAD $WORKER; do holdproc $n; done
    echo "window open: $(cat "$HOME/dsv41-deploy/HOLD")" ;;
  close)
    root=$(curl -s -m 5 http://localhost:8000/v1/models | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["root"])' 2>/dev/null || true)
    [ "$root" = /srv/dsv41/legal-v2a ] || { echo "production not up on :8000 with legal-v2a (root='$root'); window stays open" >&2; exit 1; }
    for n in $HEAD $WORKER; do ssh -o BatchMode=yes $n "tmux kill-session -t dsv41-hold 2>/dev/null || true"; done
    rm -f "$HOME/dsv41-deploy/HOLD"; echo "window closed" ;;
  status)
    [ -f "$HOME/dsv41-deploy/HOLD" ] && echo "HOLD: $(cat "$HOME/dsv41-deploy/HOLD")" || echo "no HOLD"
    for n in $HEAD $WORKER; do echo "$n: $(ssh $n 'pgrep -af "vllm serve" | cut -c1-90 | tr "\n" ";"')"; done ;;
esac
