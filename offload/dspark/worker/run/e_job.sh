#!/bin/bash
# Lane E: start a detached job on x86host in tmux WITH the GPU selection inside the session's own command (a tmux
# session does not inherit the caller's environment; an unset CUDA_VISIBLE_DEVICES would put the job on cuda:0).
# Usage (on x86host): e_job.sh NAME solo|pair LOGFILE CMD...   (solo = GPU-7744b44b only; pair = c4e7a371,7744b44b)
set -euo pipefail
name=$1; which=$2; log=$3; shift 3
case $which in
  solo) vis=GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc ;;
  pair) vis=GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93,GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc ;;
  *) echo "solo|pair"; exit 2 ;;
esac
cd ~/dsv41-offload/e/repo/offload/dspark
cmd=$(printf '%q ' "$@")
tmux new-session -d -s "$name" "export CUDA_VISIBLE_DEVICES=$vis PYTHONPATH=$PWD EXLLAMAV3_TUNE_CACHE=\$HOME/dsv41-offload/e/run/exl3_tune_cache.bin; $cmd > $log 2>&1; echo EXIT \$? >> $log"
echo "started $name on $vis -> $log"
