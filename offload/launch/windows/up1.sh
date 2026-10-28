#!/usr/bin/env bash
# S17: qualify C/K, split alone then split+pin, replay, K5/dual-ring/0067 microchecks.
# shellcheck source=offload/launch/windows/common.sh
source "$(dirname -- "$0")/common.sh"
w_init up1 "$@"
[[ ${#W_ARGS[@]} == 1 ]] || w_die 'usage: up1.sh [--dry-run|--execute] EXPECTED_PACKED_KV'
kv=${W_ARGS[0]}
w_uint "$kv"
control=offload/launch/configs/D-kv6-async-up1-control.env
w_open_checks
w_gate up1-prereq before-window # K1/S18 on same build are prerequisites, not inferred from a start
# C/K default-off equivalence, reference repeat on both sides; keep each hash proof.
for rep in 1 2 3; do
  for phase in K1 C K2; do
    arm=p0-r$rep-$phase
    cfg=$control
    [[ $phase == C ]] || cfg=offload/launch/configs/D-kv6-async-pack.env
    w_start "$arm" "$cfg" "$kv"
    w_k1 "$arm"
    [[ $phase != C ]] || w_up1_hashes "$arm"
    w_greedy "$arm"
    w_measure "$arm"
  done
  w_compare p0-r$rep-K1 p0-r$rep-K2
  w_compare p0-r$rep-K1 p0-r$rep-C
 done
w_gate up1-p0 p0-complete
for feature in split split-pin cpu-pin replay k5 dual-ring vor-copy vor-swa vor-both; do
  for rep in 1 2 3; do
    for phase in C1 X C2; do
      arm=up1-$feature-r$rep-$phase
      overrides=(UP1_SPLIT=0 UP1_CPU_PIN=0 UP1_REPLAY=0 UP1_K5=0 UP1_DUAL_RING=0 UP1_VOR=0)
      if [[ $phase == X ]]; then
        case $feature in
          split) overrides+=(UP1_SPLIT=63) ;;
          split-pin) overrides+=(UP1_SPLIT=63 UP1_CPU_PIN=1) ;;
          cpu-pin) overrides+=(UP1_CPU_PIN=1) ;;
          replay) overrides+=(UP1_REPLAY=1) ;;
          k5) overrides+=(UP1_K5=1) ;;
          dual-ring) w_gate dual-gid "$arm"; overrides+=(UP1_DUAL_RING=1) ;;
          vor-copy) overrides+=(UP1_VOR=copy) ;;
          vor-swa) overrides+=(UP1_VOR=swa) ;;
          vor-both) overrides+=(UP1_VOR=1) ;;
        esac
      fi
      w_start "$arm" "$control" "$kv" "${overrides[@]}"
      w_k1 "$arm"
      w_up1_hashes "$arm" # actual emitted hash checker on EACH container, every start
      w_stats "$arm" before
      if [[ $phase == X ]]; then
        case $feature in
          split|split-pin) w_gate split-bitwise "$arm"; [[ $feature == split ]] || w_gate cpu-pin "$arm" ;;
          cpu-pin) w_gate cpu-pin "$arm" ;;
          replay) w_gate replay "$arm" ;;
          k5) w_gate k5 "$arm" ;;
          dual-ring)
            w_rank_logs "$arm"
            for rank in 0 1; do w_step "$arm-dual-rank$rank" "$W_PY" "$W_DIR/checks.py" dual "$W_OUT/$arm-rank$rank-log.log" "$rank"; done
            w_gate dual-ring "$arm" ;;
          vor-*) w_gate vor "$arm" "$feature" ;;
        esac
      fi
      w_greedy "$arm"
      w_measure "$arm"
      w_rank_logs "$arm"
      w_stats "$arm" after
      w_stats_check "$arm"
    done
    w_compare up1-$feature-r$rep-C1 up1-$feature-r$rep-C2
    # Replay/fused numerical gates compare final logits at matched shapes; greedy drift requires diagnosis.
  done
 done
w_summary
