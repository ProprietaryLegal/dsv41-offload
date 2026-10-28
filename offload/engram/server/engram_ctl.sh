#!/usr/bin/env bash
# engram_ctl.sh start|stop|restart|status|build  - the x86host Engram table server (lane O1, DSV41-OFFLOAD).
# Runs the systemd --user unit dsv41-engram.service when it is linked, else a tmux session named dsv41-engram.
# Loading the two tables takes ~70 s (O_DIRECT + sha256 check); status shows "ready" when it serves.
set -euo pipefail
O1=${O1:-$HOME/dsv41-offload/o1}
SRC=${SRC:-$O1/src/offload}
BIN=$O1/bin/engram_server
STATS=$O1/run/engram_server.stats.json
LOG=$O1/logs/engram-server.log
ARGS=(--data /opt/dsv41/x86host-data/engram --manifest "$O1/manifest.json" --wait-marker ENGRAM_COPY_DONE
      --listen 10.202.28.1 --listen 10.202.29.1 --port 18610 --threads 8 --budget-gib 215 --stats "$STATS")
unit_linked() { systemctl --user cat dsv41-engram.service >/dev/null 2>&1; }
pid() {  # only the server started from $BIN (other lanes run their own engram_server copies on this host)
  # A rebuild/install replaces $BIN while the server runs: /proc/PID/exe then reads "<path> (deleted)", still ours.
  # Always returns 0, so `p=$(pid)` under set -e never aborts when only other lanes' servers exist (S4R fix 1).
  local p want exe
  want=$(readlink -f "$BIN")
  for p in $(pgrep -u "$(id -u)" -x engram_server || true); do
    exe=$(readlink /proc/$p/exe 2>/dev/null || true)
    if [ "$exe" = "$want" ] || [ "$exe" = "$want (deleted)" ]; then echo "$p"; return 0; fi
  done
  return 0
}

case "${1:-status}" in
  build)
    [ -z "$(pid)" ] || { echo "refusing to rebuild under a running server (pid $(pid)); stop it first" >&2; exit 1; }
    make -C "$SRC/engram/server" VERBS_INC="$O1/sysroot/usr/include" OUT="$O1/bin" ;;
  start)
    [ -z "$(pid)" ] || { echo "already running (pid $(pid))"; exit 0; }
    free -g | head -2
    mkdir -p "$O1/run" "$O1/logs"
    if unit_linked; then systemctl --user start dsv41-engram.service
    else tmux new-session -d -s dsv41-engram "$BIN ${ARGS[*]} 2>&1 | tee -a $LOG"; fi
    echo "starting; '$0 status' shows ready after the load (~70 s)" ;;
  stop)
    if unit_linked && systemctl --user is-active --quiet dsv41-engram.service; then systemctl --user stop dsv41-engram.service
    else
      p=$(pid); [ -n "$p" ] && kill -TERM "$p"
      for _ in $(seq 60); do [ -z "$(pid)" ] && break; sleep 0.5; done
      tmux kill-session -t =dsv41-engram 2>/dev/null || true
    fi
    [ -z "$(pid)" ] && echo stopped || { echo "still running: $(pid)"; exit 1; } ;;
  restart) "$0" stop; "$0" start ;;
  status)
    p=$(pid)
    if [ -z "$p" ]; then echo "engram_server: not running"; exit 3; fi
    echo "engram_server: pid $p, up $(ps -o etime= -p "$p" | tr -d ' '), RSS $(awk '/VmRSS/ {printf "%.1f GiB", $2/1048576}' /proc/$p/status)"
    ss -ltn | awk '$4 ~ /:18610$/ {print "  listening " $4}'
    if [ -f "$STATS" ]; then python3 - "$STATS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  ready={d['ready']} opens={d['opens']} lookups={d['lookups']} rows={d['rows']} refused={d['refused']} "
      f"lookup_us_mean={d['lookup_us_mean']} rss={d['rss_gib']} GiB huge={d['anon_huge_gib']} GiB "
      f"MemAvailable={d['mem_available_gib']} GiB")
for t in d["tables"]:
    print(f"  {t['shard']} layer {t['layer']} rows {t['rows']} sha256 verified={t['verified']} identity {t['identity'][:16]}...")
PY
    fi
    tail -n 3 "$LOG" 2>/dev/null | sed 's/^/  log: /' ;;
  *) echo "usage: $0 start|stop|restart|status|build"; exit 64 ;;
esac
