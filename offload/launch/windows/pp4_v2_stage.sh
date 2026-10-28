#!/usr/bin/env bash
# R-PP4 v2: stage the sparkhead experiment checkout at THIS commit for a carried-qualification measurement. WRITTEN, NOT
# EXECUTED by the lane. Coordinator only, between windows (refuses under HOLD):
#   pp4_v2_stage.sh --execute HEAD_COMMIT QUALIFIED_COMMIT
# Copies the qualified admission tree's prepared ordinary verifier on sparkhead (never re-generated, so the bytes are the
# admitted ones; the runner's preflight re-checks every hash against the admission receipt). The x86host worker tree stays
# the qualified one (--worker-tree .../admission-<qualified12>/repo). No HOLD, service, GPU or production action.
set -euo pipefail
cd "$(dirname "$0")/../../.."
if [ "${1-}" != --execute ]; then
  echo 'DRY: pp4_v2_stage.sh --execute HEAD_COMMIT QUALIFIED_COMMIT'
  echo 'Fresh sparkhead checkout ~/dsv41-deploy-pp4-v2-<HEAD12> + the qualified tree'"'"'s pp4-staged/verifier; nothing else.'
  exit 0
fi
[ "$#" = 3 ] && [ "${WINDOW_COORDINATOR-}" = Deepseek/coordinator ] || exit 2
head=$2 qualified=$3
[[ "$head" =~ ^[0-9a-f]{40}$ && "$qualified" =~ ^[0-9a-f]{40}$ ]] && [ "$(git rev-parse HEAD)" = "$head" ] || exit 2
[ -z "$(git status --porcelain --untracked-files=no)" ] || exit 2
ssh -o BatchMode=yes sparkhead 'test ! -e ~/dsv41-deploy/HOLD' || { echo 'a HOLD exists; not staging' >&2; exit 2; }
remote=dsv41-deploy-pp4-v2-${head:0:12} source=dsv41-deploy-pp4-admission-${qualified:0:12}
staged=offload/launch/windows/pp4-staged/verifier
scratch=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/PP4/v2-staging-${head:0:12}
[ ! -e "$scratch" ] || exit 2
mkdir -p "$scratch"
trap 'rm -rf -- "$scratch"' EXIT
git bundle create "$scratch/source.bundle" HEAD
ssh -o BatchMode=yes sparkhead "test ! -e ~/'$remote' && test ! -e ~/'$remote.bundle' && test -d ~/'$source/$staged'"
scp -q -o BatchMode=yes "$scratch/source.bundle" "sparkhead:$remote.bundle"
ssh -o BatchMode=yes sparkhead "git clone -q ~/'$remote.bundle' ~/'$remote' && git -C ~/'$remote' checkout -q --detach '$head' \
  && rm ~/'$remote.bundle' && mkdir -p ~/'$remote/offload/launch/windows/pp4-staged' \
  && cp -a ~/'$source/$staged' ~/'$remote/$staged' && git -C ~/'$remote' rev-parse HEAD \
  && test -z \"\$(git -C ~/'$remote' status --porcelain --untracked-files=no)\" && cd ~/'$remote/$staged' && find . -type f -exec sha256sum {} +"
echo "STAGED ONLY: sparkhead ~/$remote (verifier copied from ~/$source); x86host tree unchanged."
