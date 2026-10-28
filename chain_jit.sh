#!/usr/bin/env bash
set -uo pipefail; cd ~/dsv41-deploy
IMAGE=ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5
until docker image inspect $IMAGE >/dev/null 2>&1 && ssh sparkworker "docker image inspect $IMAGE >/dev/null 2>&1"; do
  pgrep -f "docker pull" >/dev/null || nohup docker pull $IMAGE >> logs/pull-head3.log 2>&1 &
  ssh sparkworker "pgrep -f 'docker pull' >/dev/null || nohup docker pull $IMAGE >> ~/dsv41-pull.log 2>&1 &" 
  sleep 30; done
echo "== $(date -Is) images: head $(docker image inspect $IMAGE --format '{{index .RepoDigests 0}}') worker $(ssh sparkworker "docker image inspect $IMAGE --format '{{index .RepoDigests 0}}'")"
JIT="mkdir -p \$HOME/.cache/dsv41-serve && docker run --rm --gpus all -v \$HOME/.cache/dsv41-serve:/root/.cache \
-e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions -e MAX_JOBS=4 -e EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1 \
--entrypoint python3 $IMAGE \
-c 'from vllm_exl3_moe import tf_experts as t, mrow, rdma, mhc, router, smoe, l2pf; t._ext(); mrow.ext(); rdma.ext(); mhc.ext(); router.ext(); smoe.ext(); l2pf.ext(); print(\"JIT OK\")'"
echo "== $(date -Is) JIT both"
bash -c "$JIT" > logs/jit-head.log 2>&1 & a=$!
ssh sparkworker "$JIT" > logs/jit-worker.log 2>&1 & b=$!
wait $a; ra=$?; wait $b; rb=$?
echo "== $(date -Is) JIT head rc=$ra worker rc=$rb"; tail -2 logs/jit-head.log logs/jit-worker.log
echo "== $(date -Is) start legal fetch"
python3 -I ~/dsv41-legal/scripts/evict_cache.py ~/dsv41-legal/models/DeepSeek-V4.1-Flash-1017 --interval 20 > logs/legal-evict.log 2>&1 &
RUN=1 ~/dsv41-legal/scripts/run_stage.sh fetch > logs/legal-fetch.log 2>&1 && RUN=1 ~/dsv41-legal/scripts/run_stage.sh verify >> logs/legal-fetch.log 2>&1; echo "legal fetch rc=$?" >> logs/legal-fetch.log
