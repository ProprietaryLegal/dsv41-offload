#!/usr/bin/env bash
# WRITTEN, NOT RUN by lane PP4. Offline preparation + attended coordinator staging only; never starts services or opens HOLD.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
: "${HARNESS_PYTHON:?}"
cd "$(dirname "$0")/../../.."
if [ "${1-}" != --execute ]; then
  echo 'DRY RUN: coordinator only. Required: --execute REVIEWED_COMMIT STAGING_REVIEW_JSON REVIEWED_TUNE_FILE'
  echo 'Prepares pinned K1 runner + PP3 verifier utilities; verifies qualifications and hashes; bundles clean source;'
  echo 'stages separate Sparkhead experiment and X86host /opt/dsv41/x86host-data/pp4/source/repo. No unit/current/image/HOLD change.'
  exit 0
fi
[ "$#" = 4 ] || { echo 'usage: pp4_stage.sh --execute COMMIT STAGING_REVIEW_JSON REVIEWED_TUNE_FILE' >&2; exit 2; }
[ "${WINDOW_COORDINATOR-}" = Deepseek/coordinator ] || { echo 'STOP: coordinator required, before any remote action' >&2; exit 1; }
commit=$2 qualification=$3 tune=$4
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] && [ "$(git rev-parse HEAD)" = "$commit" ] || { echo 'STOP: reviewed HEAD required' >&2; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo 'STOP: tracked source dirty' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD:$PWD/offload/dspark
"$HARNESS_PYTHON" -c 'from pathlib import Path; import sys; from offload.launch.windows.pp4_prerequisites import staging; staging(Path(sys.argv[1]),sys.argv[2])' "$qualification" "$commit"
# Runner base is the reviewed production K1 overlay; sampler utilities retain their independent image pin.
# All local scratch is on Controller disk; never /tmp, never a campaign window directory.
scratch=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/PP4/staging
[ ! -e "$scratch" ] || { echo 'STOP: fresh staging scratch required' >&2; exit 1; }
mkdir -p "$scratch"
export TMPDIR=$scratch
trap 'rm -rf -- "$scratch"' EXIT
"$HARNESS_PYTHON" -m offload.launch.windows.pp4_prepare --source /opt/dsv41/data/src-image --output "$scratch/verifier"
"$HARNESS_PYTHON" - "$qualification" "$scratch/verifier/receipt.json" "$tune" <<'PY'
import hashlib, json, sys
from offload.launch.windows.pp4_window import TUNE_SHA256
with open(sys.argv[1]) as f: q=json.load(f)
with open(sys.argv[2]) as f: p=json.load(f)
assert p['output_sha256']==q['verifier_sha256'], 'prepared verifier differs from independently reviewed evidence'
with open(sys.argv[3],'rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
assert digest==TUNE_SHA256, 'reviewed tune bytes differ; no rebuilding during HOLD'
PY
git bundle create "$scratch/source.bundle" HEAD
[ "$(git bundle list-heads "$scratch/source.bundle" HEAD | cut -d' ' -f1)" = "$commit" ] || { echo 'STOP: source HEAD changed during bundle' >&2; exit 1; }
remote=dsv41-deploy-pp4-${commit:0:12}
x86host_root=/opt/dsv41/x86host-data/pp4/source
# No production paths are written. Refuse any existing source destination, including a previous partially staged run.
ssh -o BatchMode=yes x86host "test ! -e '$x86host_root' && mkdir -p '$x86host_root'"
scp -q -o BatchMode=yes "$scratch/source.bundle" "x86host:$x86host_root/source.bundle"
scp -q -o BatchMode=yes "$tune" "x86host:$x86host_root/exl3_tune_cache-reviewed.bin"
scp -q -o BatchMode=yes "$qualification" "x86host:$x86host_root/staging-review.json"
ssh -o BatchMode=yes x86host "git clone '$x86host_root/source.bundle' '$x86host_root/repo' && git -C '$x86host_root/repo' checkout --detach '$commit' && echo '$commit' > '$x86host_root/repo/SOURCE_COMMIT' && rm '$x86host_root/source.bundle'"
# Require prebuilt, hash-attested CPython3.13 stock/facc AND source-keyed P2P. No 3.12 fallback or in-window build.
ssh -o BatchMode=yes x86host /home/operator/miniconda3/bin/python3 - "$x86host_root" <<'PP4_RUNTIME'
import hashlib,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]); data=json.loads((root/'staging-review.json').read_text())
assert sys.version_info[:2]==(3,13)
sources=data['runtime_sources']; artifacts=data['runtime_artifact_sha256']
for relative, expected in artifacts.items():
    role, tail=relative.split('/',1)
    source=Path(sources[role])/tail
    with source.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==expected, 'runtime source digest differs: '+relative
for role, source in sources.items():
    shutil.copytree(source,root/'runtime'/role,ignore=shutil.ignore_patterns('build','__pycache__','.git','*.o','*.d'))
for relative, expected in artifacts.items():
    target=root/'runtime'/relative
    with target.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==expected, 'private runtime copy differs: '+relative
    if relative.endswith('/dspark_p2p.so'):
        Path(str(target)+'.sha256').write_text(expected+'  dspark_p2p.so\n')
print('PP4 private CPython3.13 runtime hash verified; no GPU/compile/unit operation')
PP4_RUNTIME
ssh -o BatchMode=yes sparkhead "test ! -e ~/'$remote' && test ! -e ~/'$remote.bundle'"
scp -q -o BatchMode=yes "$scratch/source.bundle" "sparkhead:$remote.bundle"
ssh -o BatchMode=yes sparkhead "git clone ~/'$remote.bundle' ~/'$remote' && git -C ~/'$remote' checkout --detach '$commit' && rm ~/'$remote.bundle' && mkdir -p ~/'$remote/offload/launch/windows/pp4-staged'"
scp -rq -o BatchMode=yes "$scratch/verifier" "sparkhead:$remote/offload/launch/windows/pp4-staged/verifier"
echo "STAGED ONLY: experiment $remote; K2 source $x86host_root/repo. No service/precheck/HOLD operation ran."
echo 'Coordinator: review staged source/binaries/images, then independently qualify; never infer live readiness from staging success.'
