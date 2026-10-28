#!/usr/bin/env bash
# Offline tests of stage_overlay.sh's libdsv41_hostflag.so handling (S9 F8): a copy of the script with test pins runs
# against fake ssh (each node = a local directory) and fake docker. No Spark, no network.
#   offload/dspark/spark/test_stage_overlay.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
ROOT=$PWD
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
# a minimal repo copy (the script cds to its repo root)
R=$S/repo
mkdir -p "$R/offload/dspark/spark" "$R/offload/transport"
cp -a offload/dspark/spark/remote_dspark offload/dspark/spark/overlay "$R/offload/dspark/spark/"
cp offload/transport/ofl.py "$R/offload/transport/"
img_sha=$(sed -n 's/^IMAGE_INIT_SHA=//p' offload/dspark/spark/stage_overlay.sh)
# test builds of the two libraries (contents only matter for their sha256)
mkdir -p "$S/run/ofl"
echo "test libofl" > "$S/run/ofl/libofl.so"
cc -O2 -Wall -shared -fPIC -o "$S/run/ofl/libdsv41_hostflag.so" offload/dspark/spark/remote_dspark/hostflag.c -lpthread
ofl_sha=$(sha256sum < "$S/run/ofl/libofl.so" | cut -d' ' -f1)
hf_sha=$(sha256sum < "$S/run/ofl/libdsv41_hostflag.so" | cut -d' ' -f1)
mkscript() {  # $1 = hostflag binary pin
  sed -e "s/^OFL_SO_SHA=.*/OFL_SO_SHA=$ofl_sha/" -e "s/^HOSTFLAG_SO_SHA=.*/HOSTFLAG_SO_SHA=$1/" \
    "$ROOT/offload/dspark/spark/stage_overlay.sh" > "$R/offload/dspark/spark/stage_overlay.sh"
  chmod +x "$R/offload/dspark/spark/stage_overlay.sh"
}
# fake ssh: "ssh [-o X] NODE CMD" runs CMD locally with HOME = the node's directory
mkdir -p "$S/bin" "$S/nodes"
cat > "$S/bin/ssh" <<EOS
#!/usr/bin/env bash
while [ "\${1:-}" = -o ]; do shift 2; done
node=\$1; shift
mkdir -p "$S/nodes/\$node"
HOME="$S/nodes/\$node" PATH="$S/bin:\$PATH" bash -c "\$*"
EOS
cat > "$S/bin/docker" <<EOS
#!/usr/bin/env bash
echo "$img_sha  \${@: -1}"
EOS
chmod +x "$S/bin/ssh" "$S/bin/docker"
if ! command -v cpio >/dev/null; then   # the Sparks have cpio; a pass-through stand-in (cpio -0pdm --quiet DEST)
  cat > "$S/bin/cpio" <<'EOS'
#!/usr/bin/env python3
import os, shutil, sys
dest = sys.argv[-1]
for rel in sys.stdin.buffer.read().split(b"\0"):
    if rel:
        rel = os.fsdecode(rel)
        os.makedirs(os.path.join(dest, os.path.dirname(rel)), exist_ok=True)
        shutil.copy2(rel, os.path.join(dest, rel))
EOS
  chmod +x "$S/bin/cpio"
fi
run() {  # env assignments..., then nodes
  env PATH="$S/bin:$PATH" OFL_SO_SRC="$S/run/ofl/libofl.so" HOSTFLAG_SO_SRC="$S/run/ofl/libdsv41_hostflag.so" \
    "$@" "$R/offload/dspark/spark/stage_overlay.sh" n1 n2
}
staged() { test -f "$S/nodes/$1/dsv41-offload-prod/dspark/ofl/libdsv41_hostflag.so"; }
ASYNC_ENV="OFFLOAD_DSPARK_ENV=VLLM_DSPARK_REMOTE_POLICY=capacity VLLM_DSPARK_REMOTE_ASYNC=1"
SYNC_ENV="OFFLOAD_DSPARK_ENV=VLLM_DSPARK_REMOTE_POLICY=capacity"

mkscript UNPINNED
out=$(run "$SYNC_ENV" 2>&1) || fail "T1 sync config refused with an unpinned hostflag build: $out"
grep -q "note: libdsv41_hostflag.so not staged" <<<"$out" || fail "T1 no note"
! staged n1 && ! staged n2 || fail "T1 an unpinned build was staged"
! out=$(run "$ASYNC_ENV" 2>&1) || fail "T2 async staged with an unpinned build"
grep -q "async rounds requested but .* is not the pinned libdsv41_hostflag.so build" <<<"$out" || fail "T2 message: $out"
! run "$SYNC_ENV" HOSTFLAG_REQUIRED=1 >/dev/null 2>&1 || fail "T3 HOSTFLAG_REQUIRED=1 accepted an unpinned build"
mkscript "$hf_sha"
run "$ASYNC_ENV" >/dev/null || fail "T4 async with the pinned build refused"
staged n1 && staged n2 || fail "T4 pinned build not staged on both nodes"
[ "$(sha256sum < "$S/nodes/n2/dsv41-offload-prod/dspark/ofl/libdsv41_hostflag.so" | cut -d' ' -f1)" = "$hf_sha" ] \
  || fail "T4 staged copy differs"
run "$SYNC_ENV" >/dev/null && staged n1 || fail "T5 a pinned build is staged for sync configs too"
mv "$S/run/ofl/libdsv41_hostflag.so" "$S/hf.so"
! out=$(run "$ASYNC_ENV" 2>&1) && grep -q "is missing" <<<"$out" || fail "T6 missing build not refused: $out"
mv "$S/hf.so" "$S/run/ofl/libdsv41_hostflag.so"
echo "/* changed */" >> "$R/offload/dspark/spark/remote_dspark/hostflag.c"
! out=$(run "$ASYNC_ENV" 2>&1) && grep -q "is not the pinned hostflag source" <<<"$out" || fail "T7 changed source not refused"
echo "stage_overlay.sh hostflag tests: 7/7 passed"
