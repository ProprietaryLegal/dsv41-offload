#!/usr/bin/env bash
# WRITTEN, NOT EXECUTED. Fresh versioned trees; never changes queued5fdc17a/source.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
cd "$(dirname "$0")/../../.."
if [ "${1-}" != --execute ]; then
  echo 'DRY: pp4_admission_stage.sh --execute REVIEWED_COMMIT ADMISSION_SEED_JSON'
  echo 'Fresh versioned Sparkhead/X86host admission trees; reuse reviewed runtime/tune read-only. No HOLD/service/precheck.'
  exit 0
fi
[ "$#" = 3 ] && [ "${WINDOW_COORDINATOR-}" = Deepseek/coordinator ] || exit 2
commit=$2 seed=$3
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] && [ "$(git rev-parse HEAD)" = "$commit" ] || exit 2
[ -z "$(git status --porcelain --untracked-files=no)" ] || exit 2
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/offload/dspark CUDA_VISIBLE_DEVICES=''
scratch=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/PP4/admission-staging
[ ! -e "$scratch" ] || exit 2
mkdir -p "$scratch"
export TMPDIR=$scratch
trap 'rm -rf -- "$scratch"' EXIT
/home/operator/miniconda3/bin/python3 -m offload.launch.windows.pp4_prepare --source /opt/dsv41/data/src-image --output "$scratch/ordinary"
/home/operator/miniconda3/bin/python3 -m offload.launch.windows.pp4_admission_prepare --source /opt/dsv41/data/src-image --output "$scratch/admission"
/home/operator/miniconda3/bin/python3 - "$seed" "$commit" "$scratch" <<'PY'
import json,sys
from pathlib import Path
from offload.launch.windows.pp4_admission import seed
s=Path(sys.argv[3]); data=seed(Path(sys.argv[1]),sys.argv[2])
assert data['verifier_sha256']==json.loads((s/'ordinary/receipt.json').read_text())['output_sha256']
assert data['admission_verifier_sha256']==json.loads((s/'admission/receipt.json').read_text())['output_sha256']
PY
remote=dsv41-deploy-pp4-admission-${commit:0:12}
x86host_root=/opt/dsv41/x86host-data/pp4/admission-${commit:0:12}
git bundle create "$scratch/source.bundle" HEAD
# Refuse existing destination. Existing source/runtime/tune is only read/hashed;
# no symlink, copy, chmod, cleanup or build beneath the queued source root.
ssh -o BatchMode=yes x86host "test ! -e '$x86host_root' && mkdir '$x86host_root'"
scp -q -o BatchMode=yes "$scratch/source.bundle" "x86host:$x86host_root/source.bundle"
ssh -o BatchMode=yes x86host "git clone '$x86host_root/source.bundle' '$x86host_root/repo' && git -C '$x86host_root/repo' checkout --detach '$commit' && echo '$commit' > '$x86host_root/repo/SOURCE_COMMIT' && rm '$x86host_root/source.bundle'"
ssh -o BatchMode=yes sparkhead "test ! -e ~/'$remote' && test ! -e ~/'$remote.bundle'"
scp -q -o BatchMode=yes "$scratch/source.bundle" "sparkhead:$remote.bundle"
ssh -o BatchMode=yes sparkhead "git clone ~/'$remote.bundle' ~/'$remote' && git -C ~/'$remote' checkout --detach '$commit' && rm ~/'$remote.bundle' && mkdir -p ~/'$remote/offload/launch/windows/pp4-staged' ~/'$remote/offload/launch/windows/pp4-admission-staged'"
scp -rq -o BatchMode=yes "$scratch/ordinary" "sparkhead:$remote/offload/launch/windows/pp4-staged/verifier"
scp -rq -o BatchMode=yes "$scratch/admission" "sparkhead:$remote/offload/launch/windows/pp4-admission-staged/verifier"
echo "STAGED ONLY: $remote, $x86host_root/repo; existing source and queued5fdc17a unchanged."
