#!/usr/bin/env bash
# PP4-PROMOTE staging (coordinator only, ON Controller, BEFORE the HOLD). Creates NEW immutable directories only; never
# touches a running service, unit, `current` link, dsv41.env or an existing directory (each step refuses one).
#   Sparks (sparkhead + sparkworker-wired): ~/dsv41-offload-prod/pp4/<id>/{remote_dspark,verifier}/ + SHA256SUMS
#       remote_dspark = the PP4 commit's offload/dspark/spark/remote_dspark/*.py (stage_overlay.sh's selection);
#       verifier      = pp4_prepare's ORDINARY output, refused unless it equals the qualified verifier_sha256.
#   X86host: release ~/dsv41-offload/prod/dspark-worker/<short7> (worker/run/prod_tp2.sh stage), runtime copy
#       ~/dsv41-offload/prod/pp4-runtime/<id>/{stock,facc,p2p} + reviewed tune cache + SHA256SUMS + attest.sh,
#       refused unless the three qualified .so digests and the tune digest match.
# Usage: WINDOW_COORDINATOR=Deepseek/coordinator bash offload/launch/windows/pp4_promote_stage.sh --execute
set -euo pipefail
cd "$(dirname "$0")/../../.."
[ "${1-}" = --execute ] || { sed -n 2,11p "$0"; exit 0; }
[ "${WINDOW_COORDINATOR-}" = Deepseek/coordinator ] || { echo "coordinator only" >&2; exit 2; }
PY=/home/operator/miniconda3/bin/python3
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=
SW=offload/launch/windows/pp4_promote_switch.py
R=$($PY -c 'from offload.launch.windows.pp4_promote_switch import PP4_COMMIT as c; print(c)')
ID=${R:0:12}
[ "$(git rev-parse HEAD)" = "$R" ] || [ "$(git merge-base --is-ancestor "$R" HEAD && echo yes)" = yes ] \
  || { echo "run from a checkout containing $R" >&2; exit 2; }
scratch=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/PP4PROMO/stage-$ID
[ ! -e "$scratch" ] || { echo "refusing: $scratch exists" >&2; exit 2; }
mkdir -p "$scratch/src" "$scratch/stage/remote_dspark"
trap 'rm -rf "$scratch"' EXIT
git archive "$R" offload/dspark/spark/remote_dspark offload/kvpack offload/dspark/probabilistic/pp3 \
  offload/launch/windows/pp4_prepare.py offload/launch/windows/pp4_round_trace.py | tar -x -C "$scratch/src"
(cd "$scratch/src/offload/dspark/spark/remote_dspark" && find . -name '*.py' -not -path '*/__pycache__/*' -print0 \
  | cpio -0pdm --quiet "$scratch/stage/remote_dspark")
(cd "$scratch/src" && PYTHONPATH=$scratch/src $PY -m offload.launch.windows.pp4_prepare \
  --source /opt/dsv41/data/src-image --output "$scratch/prepared" >/dev/null)
$PY - "$scratch/prepared" "$scratch/stage/verifier" <<'PY'
import hashlib, shutil, sys
from pathlib import Path
from offload.launch.windows.pp4_promote_switch import VERIFIER_SHA256
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for name, want in VERIFIER_SHA256.items():
    got = hashlib.sha256((src / name).read_bytes()).hexdigest()
    assert got == want, f'{name}: prepared {got} != qualified {want}'
    (dst / name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / name, dst / name)
print('verifier == qualified verifier_sha256')
PY
(cd "$scratch/stage" && find . -type f | LC_ALL=C sort | xargs sha256sum > "$scratch/SHA256SUMS" && mv "$scratch/SHA256SUMS" .)
for h in sparkhead sparkworker-wired; do
  ssh -o BatchMode=yes "$h" "test ! -e ~/dsv41-offload-prod/pp4/$ID && test ! -e ~/dsv41-offload-prod/pp4/$ID.tmp"
  tar -C "$scratch/stage" -cf - . | ssh -o BatchMode=yes "$h" "mkdir -p ~/dsv41-offload-prod/pp4/$ID.tmp && \
    tar -x -C ~/dsv41-offload-prod/pp4/$ID.tmp && cd ~/dsv41-offload-prod/pp4/$ID.tmp && sha256sum -c --quiet SHA256SUMS && \
    chmod -R a-w . && mv ~/dsv41-offload-prod/pp4/$ID.tmp ~/dsv41-offload-prod/pp4/$ID && echo \"\$(hostname): staged PP4 $ID\""
done
bash offload/dspark/worker/run/prod_tp2.sh stage "$R"
# X86host runtime: one locally generated script on stdin (attest.sh embedded base64); every digest is checked remotely.
$PY - "$ID" > "$scratch/x86host.sh" <<'PY'
import base64, shlex, sys
from offload.launch.windows.pp4_promote_switch import SO_SHA256, TUNE, TUNE_SHA256, render_attest
ident = sys.argv[1]
attest = render_attest().encode()
lines = ['set -euo pipefail', 'S=/opt/dsv41/x86host-data/pp4/source', f'D=$HOME/dsv41-offload/prod/pp4-runtime/{ident}',
         'test ! -e "$D" && test ! -e "$D.tmp" && mkdir -p "$D.tmp"',
         f'cp -a "$S/runtime/stock" "$S/runtime/facc" "$S/runtime/p2p" "$D.tmp/" && cp "$S/{TUNE}" "$D.tmp/"',
         'cd "$D.tmp"']
lines += [f'test "$(sha256sum {shlex.quote(name)} | cut -d" " -f1)" = {digest}' for name, digest in sorted(SO_SHA256.items())]
lines += [f'test "$(sha256sum {TUNE} | cut -d" " -f1)" = {TUNE_SHA256}',
          f"find . -type f -not -name {TUNE} -not -path '*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS",
          f'echo {base64.b64encode(attest).decode()} | base64 -d > attest.sh',
          f'test "$(sha256sum < attest.sh | cut -d" " -f1)" = {__import__("hashlib").sha256(attest).hexdigest()}',
          'chmod 0555 attest.sh && chmod -R a-w stock facc p2p SHA256SUMS',
          'cd && mv "$D.tmp" "$D" && sh "$D/attest.sh"',
          f'echo "$(hostname): staged PP4 runtime {ident}"']
print('\n'.join(lines))
PY
ssh -o BatchMode=yes x86host bash -s < "$scratch/x86host.sh"
echo "PP4 STAGED $ID: Sparks ~/dsv41-offload-prod/pp4/$ID, x86host release ${R:0:7} + runtime ~/dsv41-offload/prod/pp4-runtime/$ID"
