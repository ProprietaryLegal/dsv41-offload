#!/usr/bin/env bash
# Sourced by coordinator-only window entry points. Dry-run performs no commands or writes.
set -euo pipefail
W_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
W_ROOT=$(cd "$W_DIR/../../.." && pwd)
W_DRY=1 W_ACTIVE=0
W_HEAD=${WINDOW_HEAD:-sparkhead}
W_PEER=${WINDOW_PEER:-sparkworker-wired}
W_X86HOST=${WINDOW_X86HOST:-x86host}
W_REMOTE=${WINDOW_REMOTE_REPO:-dsv41-deploy-offload}
W_URL=${WINDOW_URL:-http://HEAD_IP:8001}
W_PY=${HARNESS_PYTHON:-/home/operator/miniconda3/bin/python3}
W_HOOK=${WINDOW_HOOK:-}
W_K1='VLLM_DSV41_KV_PAIR=3 VLLM_DSV41_DROP_MTP_BUFFER=1 VLLM_EXL3_TF_PROMPT_TRANSIENT=1 VLLM_EXL3_TF_PROMPT_ALIAS=1 VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4'
w_die() { echo "STOP: $*" >&2; exit 1; }
w_next() {
  echo "Coordinator next: inspect the run's step logs, STOP file and summary. Do not close the window while an experiment is serving."
  if [[ $W_DRY == 0 && -f $W_OUT/coordinator-next.sh ]]; then
    echo "Exact saved restoration commands (review and run as coordinator):"
    cat "$W_OUT/coordinator-next.sh"
  fi
  echo "On x86host: bash ~/dsv41-offload/prod_window.sh status; stop only the transient worker owned by this window, then bash ~/dsv41-offload/prod_window.sh close; require PRODUCTION HOLD OK."
  echo 'On sparkhead: use the saved pre-window production checkout/revision/environment and recorded worker state to restore production with ./stop.sh then ./start.sh. Verify model/identity, async process proxy, remote Engram, KV, health and smoke workload on :8000; require prod_window.sh check -> PRODUCTION HOLD OK; only then offload/launch/window.sh close and resume paused jobs.'
}
w_exit() { local rc=$?; trap - EXIT; [[ $rc == 0 ]] || echo "STOP rc=$rc; production was not restored." >&2; w_next; exit "$rc"; }
w_init() {
  local lane=$1; shift
  if [[ ${1:-} == --execute ]]; then W_DRY=0; shift
  elif [[ ${1:-} == --dry-run ]]; then shift; fi
  # shellcheck disable=SC2034 # Entry points consume this array.
  W_ARGS=("$@")
  W_OUT=${WINDOW_OUT:-/opt/dsv41/campaigns/DSV41-OFFLOAD/WR1-$lane-$(date -u +%Y%m%dT%H%M%SZ)}
  [[ $W_REMOTE =~ ^[a-zA-Z0-9_./-]+$ && $W_REMOTE != /* && $W_REMOTE != *..* ]] || w_die 'unsafe relative remote checkout'
  for host in "$W_HEAD" "$W_PEER" "$W_X86HOST"; do [[ $host =~ ^[a-zA-Z0-9_.-]+$ && $host != -* ]] || w_die 'unsafe SSH host'; done
  if [[ $W_DRY == 0 ]]; then
    [[ ${WINDOW_COORDINATOR:-} == Deepseek/coordinator ]] || w_die 'execution requires WINDOW_COORDINATOR=Deepseek/coordinator'
    [[ -x $W_HOOK ]] || w_die 'WINDOW_HOOK must name an executable coordinator gate/worker adapter (see README)'
    [[ $W_OUT == /opt/dsv41/campaigns/DSV41-OFFLOAD/* ]] || w_die 'output must be in the private campaign directory'
    mkdir "$W_OUT" # refuse reuse, including stale receipts/STOP files
    : > "$W_OUT/.run-start"
  fi
  trap w_exit EXIT
  echo "RUN $lane: dry_run=$W_DRY output=$W_OUT; MemAvailable stop 5242880 kB/rank; swap growth/monitor failure stops."
  cd "$W_ROOT"
}
w_step() {
  local name=$1; shift
  printf 'STEP %s: ' "$name"; printf '%q ' "$@"; printf '\n'
  [[ $W_DRY == 0 ]] || return 0
  [[ ! -e $W_OUT/STOP ]] || w_die "STOP exists: $W_OUT/STOP"
  echo "$(date -u +%FT%TZ) BEGIN $name" >> "$W_OUT/steps.log"
  local rc=0
  if [[ $W_ACTIVE == 1 ]]; then
    "$W_PY" "$W_DIR/checks.py" supervise --out "$W_OUT" --name "$name" --hosts "$W_HEAD" "$W_PEER" --x86host "$W_X86HOST" -- "$@" > "$W_OUT/$name.log" 2>&1 || rc=$?
  else
    "$@" > "$W_OUT/$name.log" 2>&1 || rc=$?
  fi
  echo "$(date -u +%FT%TZ) END $name rc=$rc" >> "$W_OUT/steps.log"
  [[ $rc == 0 ]] || w_die "$name rc=$rc (see $W_OUT/$name.log)"
}
w_remote() { local name=$1 host=$2; shift 2; local cmd; printf -v cmd '%q ' "$@"; w_step "$name" ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "bash -lc $(printf '%q' "$cmd")"; }
w_head() {
  local name=$1; shift; local cmd
  printf -v cmd '%q ' "$@"
  w_step "$name" ssh -o BatchMode=yes -o ConnectTimeout=5 "$W_HEAD" "cd ~/$W_REMOTE && $cmd"
}
w_gate() {
  local gate=$1 arm=$2; shift 2
  local receipt=$W_OUT/$arm-$gate.json
  w_step "$arm-$gate" "${W_HOOK:-/COORDINATOR/REQUIRED-window-hook}" "$gate" "$arm" "$W_OUT" "$receipt" "$@"
  w_step "$arm-$gate-assert" "$W_PY" "$W_DIR/checks.py" receipt "$receipt" "$gate" "$arm" "$W_OUT"
}
w_open_checks() {
  w_step hold ssh -o BatchMode=yes -o ConnectTimeout=5 "$W_HEAD" 'test -f ~/dsv41-deploy/HOLD'
  w_gate preflight window "$W_URL" "$W_REMOTE"
  w_step restore-plan "$W_PY" "$W_DIR/checks.py" rollback "$W_OUT/coordinator-next.sh"
  # preflight receipt records drained production, saved rollback, D2 pause, approved source hashes and exclusive window.
  W_ACTIVE=1
}
w_uint() { [[ $1 =~ ^[1-9][0-9]*$ ]] || w_die "expected positive integer, got: $1"; }
w_start() {
  local arm=$1 cfg=$2 kv=$3; shift 3
  w_uint "$kv"
  w_gate idle "$arm" "$W_URL"
  w_head "$arm-start" env "$@" bash offload/launch/start_exp.sh "$cfg"
  w_step "$arm-kv" "$W_PY" "$W_DIR/checks.py" kv "$W_OUT/$arm-start.log" "$kv"
  w_remote "$arm-health" "$W_HEAD" curl --fail --silent --show-error --max-time 10 http://localhost:8001/health
  w_gate runtime "$arm" "$cfg" "$kv" "$W_URL" # both ranks' identity/process/HELLO/env/layout + error counters
}
w_k1() { w_head "$1-k1" bash offload/kvpack/stage_overlay.sh verify-running --expect "$W_K1" spark sparkworker; }
w_greedy() {
  w_step "$1-greedy" "$W_PY" offload/engram/tests/logit_ab.py run --url "$W_URL" --label "$1" --sizes 512 --max-tokens 64 --data /opt/dsv41/data/bench-data/opinions_cl.jsonl --out "$W_OUT/$1-greedy.jsonl"
}
w_compare() { w_step "$1-vs-$2-greedy" "$W_PY" offload/engram/tests/logit_ab.py compare "$W_OUT/$1-greedy.jsonl" "$W_OUT/$2-greedy.jsonl"; }
w_measure() {
  local arm=$1 cache reps
  for cache in cold warm; do
    reps=1; [[ $cache == cold ]] || reps=2 # warm mode rep0 primes; rep1 is the measured warm repeat
    w_step "$arm-$cache" "$W_PY" offload/bench/bench2.py --url "$W_URL" --label "$arm" --ctx 0 32000 150000 --workload legal --filler real --data-dir /opt/dsv41/data/bench-data --concurrency 1 3 4 --reps "$reps" --mixed --mixed-short-ctx 8000 --max-tokens 1500 --thinking --reasoning-effort high --temperature 1 --top-p 1 --seed 42 --priority 10 --cache-mode "$cache" --spark-hosts "$W_HEAD" "$W_PEER" --x86host-host "$W_X86HOST" --out "$W_OUT/$arm-$cache.jsonl"
    w_step "$arm-$cache-summary" "$W_PY" "$W_DIR/checks.py" bench "$W_OUT/$arm-$cache.jsonl"
  done
  w_gate clean "$arm" "$W_URL" # current both-rank logs and before/after counters, no fallback/provenance/watchdog
}
w_summary() { local flags=(); [[ ${W_SUMMARY_PAIR:-0} == 0 ]] || flags+=(--pair); w_step summary "$W_PY" "$W_DIR/checks.py" summary "$W_OUT" "${flags[@]}"; echo 'Window steps complete; no production restoration or promotion performed.'; }
w_stdin() {
  local name=$1 input=$2; shift 2
  printf 'STDIN %s < %q\n' "$name" "$input"
  if [[ $W_DRY == 1 ]]; then w_step "$name" "$@"; else w_step "$name" "$@" < "$input"; fi
}
w_up1_hashes() {
  local arm=$1 host rank=0
  w_step "$arm-up1-emit" "$W_PY" offload/upstream_adopt/verify_runtime.py --emit
  for host in "$W_HEAD" "$W_PEER"; do
    w_stdin "$arm-runtime-hash-rank$rank" "$W_OUT/$arm-up1-emit.log" ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" 'docker exec -i dsv41-serve python3 -'
    rank=$((rank+1))
  done
}
w_rank_logs() {
  local arm=$1 host rank=0
  for host in "$W_HEAD" "$W_PEER"; do
    w_remote "$arm-rank$rank-log" "$host" docker logs dsv41-serve
    w_step "$arm-rank$rank-clean-log" "$W_PY" "$W_DIR/checks.py" logs "$W_OUT/$arm-rank$rank-log.log"
    rank=$((rank+1))
  done
}
w_stats() { w_remote "$1-stats-$2" "$W_HEAD" docker exec dsv41-serve cat /tmp/dsv41-remote-dspark.json; }
w_stats_check() { w_step "$1-stats-check" "$W_PY" "$W_DIR/checks.py" counters "$W_OUT/$1-stats-before.log" "$W_OUT/$1-stats-after.log"; }
