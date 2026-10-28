#!/usr/bin/env bash
# Detached, resumable task sequence. Only lock contention retries; any real failure stops.
set -euo pipefail
source ~/.config/ndm/paths.sh
lane_root=$HOME/dsv41-offload/kx-grouped
lane_python=$HOME/dsv41-offload/o2/venv/bin/python
lane_donor=/opt/dsv41/x86host-data/donors/cutlass-v3.9.2
lane_native=/opt/dsv41/x86host-data/native-1017
mkdir -p "$lane_root/logs"
run_item() {
  local name=$1
  shift
  for attempt in 1 2 3 4 5 6; do
    date -u '+%FT%TZ'
    printf 'WAITING %s attempt %s (exclusive development-card lock)\n' "$name" "$attempt"
    printf '%s\n' "waiting $name attempt $attempt" > "$lane_root/queue-state.txt"
    if "$lane_root/src/run_locked.sh" "$@" > "$lane_root/logs/$name.log" 2>&1; then
      printf 'PASS %s\n' "$name"
      printf '%s\n' "passed $name" > "$lane_root/queue-state.txt"
      return
    else
      result=$?
      if [[ $result != 75 ]]; then
        printf 'FAILED %s exit %s; see %s/logs/%s.log\n' "$name" "$result" "$lane_root" "$name" >&2
        printf '%s\n' "failed $name exit $result" > "$lane_root/queue-state.txt"
        return "$result"
      fi
      printf 'Lock still reserved; no GPU process launched for %s\n' "$name"
    fi
  done
  printf 'Development-card lock unavailable for three hours: %s\n' "$name" >&2
  return 75
}
run_item verify-final "$lane_python" -u "$lane_root/src/verify.py" "$lane_donor" "$lane_root/build" "$lane_root/verify-final.json"
run_item routes-final "$lane_python" -u "$lane_root/src/collect_routes.py" "$lane_root/reference" /opt/dsv41/x86host-data/dspark /opt/dsv41/x86host-data/dspark/captures/A-capture-20261022/dspark-captures "$lane_root/routes-final.pt" --rounds 453
run_item dense-final "$lane_python" -u "$lane_root/src/bench.py" dense "$lane_donor" "$lane_root/build" "$lane_native" "$lane_root/routes-final.pt" "$lane_root/dense-final.json" --worker "$lane_root/reference" --checkpoint /opt/dsv41/x86host-data/dspark
run_item experts-final "$lane_python" -u "$lane_root/src/bench.py" experts "$lane_donor" "$lane_root/build" "$lane_native" "$lane_root/routes-final.pt" "$lane_root/experts-final.json"
run_item qualify-final "$lane_python" -u "$lane_root/src/qualify_native.py" "$lane_donor" "$lane_root/build" "$lane_native" "$lane_root/n1-reference/formats.py" "$lane_root/routes-final.pt" "$lane_root/qualify-final.json"
printf 'COMPLETE all GPU items (verify final tmux process exits)\n'
printf '%s\n' 'complete' > "$lane_root/queue-state.txt"
