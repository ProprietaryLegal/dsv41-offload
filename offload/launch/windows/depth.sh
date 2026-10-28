#!/usr/bin/env bash
# SP2 candidate paths are inputs; never invent or modify a depth policy in this runner.
# shellcheck source=offload/launch/windows/common.sh
source "$(dirname -- "$0")/common.sh"
w_init depth "$@"
[[ ${#W_ARGS[@]} -ge 3 ]] || w_die 'usage: depth.sh [--dry-run|--execute] EXPECTED_KV CONTROL_CONFIG CANDIDATE_CONFIG [CANDIDATE_CONFIG ...]'
kv=${W_ARGS[0]} control=${W_ARGS[1]}
w_uint "$kv"
for candidate in "${W_ARGS[@]:2}"; do [[ $candidate != "$control" ]] || w_die 'candidate must differ from control'; done
w_open_checks
for candidate in "${W_ARGS[@]:2}"; do
  [[ $candidate != "$control" ]] || w_die 'candidate must differ from control'
  tag=$(basename "$candidate" .env)
  [[ $tag =~ ^[a-zA-Z0-9_.-]+$ ]] || w_die 'candidate basename must be a safe label'
  w_gate depth "$tag" "$control" "$candidate" "$kv"
  for rep in 1 2 3; do
    for phase in C1 X C2; do
      arm=depth-$tag-r$rep-$phase
      cfg=$control; [[ $phase != X ]] || cfg=$candidate
      w_start "$arm" "$cfg" "$kv"
      w_greedy "$arm"
      w_measure "$arm"
    done
    w_compare "depth-$tag-r$rep-C1" "depth-$tag-r$rep-C2"
    # Acceptance/depth choices can differ; target greedy identity must remain the same.
    w_compare "depth-$tag-r$rep-C1" "depth-$tag-r$rep-X"
  done
 done
w_summary
