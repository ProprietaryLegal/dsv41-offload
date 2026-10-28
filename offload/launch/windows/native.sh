#!/usr/bin/env bash
# Pair qualification and GB10 C/x/C; only the coordinator may use --execute.
# shellcheck source=offload/launch/windows/common.sh
source "$(dirname -- "$0")/common.sh"
w_init native "$@"
[[ ${#W_ARGS[@]} == 4 ]] || w_die 'usage: native.sh [--dry-run|--execute] CONTROL_CONFIG CONTROL_KV NATIVE_KV TIMING_JSON (pair: use native_pair.sh)'
control=${W_ARGS[0]} control_kv=${W_ARGS[1]} native_kv=${W_ARGS[2]} timing=${W_ARGS[3]}
w_uint "$control_kv"; w_uint "$native_kv"
w_step timings "$W_PY" "$W_DIR/checks.py" timing "$timing"
w_open_checks
# The adapter must stage a NEW timing-patched config after validating the pair/verify-only cost evidence.
w_gate recalibrate native "$timing" offload/launch/configs/E4-native-kx2-async.env
for rep in 1 2 3; do
  for phase in C1 X C2; do
    arm=native-r$rep-$phase
    if [[ $phase == X ]]; then
      cfg=offload/launch/configs/WR1-E4-native-kx2-async-calibrated.env; kv=$native_kv; worker=native-hybrid-kx2
    else cfg=$control; kv=$control_kv; worker=k2-tp2; fi
    # Gate owns transient worker switch under prod_window.sh pair COORD; validates lock/PID/hash/identity.
    w_gate worker "$arm" "$worker"
    w_start "$arm" "$cfg" "$kv"
    w_greedy "$arm"
    w_measure "$arm"
  done
  w_compare native-r$rep-C1 native-r$rep-C2
  w_compare native-r$rep-C1 native-r$rep-X
 done
w_summary
