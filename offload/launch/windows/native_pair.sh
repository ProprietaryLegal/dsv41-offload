#!/usr/bin/env bash
# No Spark access before execution. prod_window pair provides marker, N4 obtains its own bounded locks.
# shellcheck source=offload/launch/windows/common.sh
source "$(dirname -- "$0")/common.sh"
w_init native-pair "$@"
[[ ${#W_ARGS[@]} == 1 ]] || w_die 'usage: native_pair.sh [--dry-run|--execute] r000008_b0,r000009_b0,...'
keys=${W_ARGS[0]}
[[ $keys =~ ^r[0-9]+_b[0-9]+(,r[0-9]+_b[0-9]+)*$ ]] || w_die 'explicit V1 keys required'
w_open_checks
w_remote pair-open "$W_X86HOST" bash dsv41-offload/prod_window.sh pair COORD
w_remote pair-check "$W_X86HOST" env N4_PAIR_BY_COORD=1 N3_PAIR_ARMS='native-hybrid-kx2 native-packed' N3_V1_KEYS="$keys" bash dsv41-offload/n4/n4_pair.sh
w_gate pair-results native-pair "$keys" # classify V1 and record timings/identities, per-card peaks and fresh source pins
W_SUMMARY_PAIR=1
w_summary
# Never close here: coordinator inspects V1 and restores production worker explicitly.
