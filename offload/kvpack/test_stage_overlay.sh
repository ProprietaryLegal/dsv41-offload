#!/usr/bin/env bash
# K1: offline tests of stage_overlay.sh against the image-v5 source tree standing in for the image (no Spark, no docker).
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC=${SRC:-/opt/dsv41/data/src-image}
S=$(mktemp -d); trap 'chmod -R u+w "$S"; rm -rf "$S"' EXIT
mkdir -p "$S/home" "$S/img"
fail() { echo "FAIL: $*" >&2; exit 1; }
K=offload/kvpack/stage_overlay.sh
KVPACK_TEST_IMAGE_ROOT=$SRC $K check local >/dev/null || fail "T1 check against the v5 tree"
cp -a "$SRC/vllm" "$SRC/vllm_exl3_moe" "$S/img/"
echo "# tampered" >> "$S/img/vllm/v1/kv_cache_interface.py"
! KVPACK_TEST_IMAGE_ROOT=$S/img $K check local 2>/dev/null || fail "T2 a changed base file was accepted"
cp "$SRC/vllm/v1/kv_cache_interface.py" "$S/img/vllm/v1/kv_cache_interface.py"
touch "$S/img/vllm/v1/worker/gpu/dsv41_memprobe.py"
! KVPACK_TEST_IMAGE_ROOT=$S/img $K check local 2>/dev/null || fail "T3 an image that already has a new file was accepted"
HOME=$S/home KVPACK_TEST_IMAGE_ROOT=$SRC $K stage local >/dev/null || fail "T4 stage"
HOME=$S/home KVPACK_TEST_IMAGE_ROOT=$SRC $K stage local >/dev/null || fail "T4 re-stage (verify only)"
f=$(ls -d "$S"/home/dsv41-kvpack/*)/vllm/v1/worker/utils.py; chmod u+w "$f"; echo "# corrupted" >> "$f"
! HOME=$S/home KVPACK_TEST_IMAGE_ROOT=$SRC $K stage local 2>/dev/null || fail "T5 a corrupted staged copy was accepted"
n=$($K overlays | wc -w); [ "$n" = 13 ] || fail "T6 overlays prints $n pairs, expected 13"
n=$($K args | grep -o -- ' -v ' | wc -l); [ "$n" = 13 ] || fail "T7 args prints $n mounts, expected 13"
offload/kvpack/build_overlay.sh >/dev/null || fail "T8 overlay != fresh rebuild"
python3 offload/kvpack/kv_arith.py >/dev/null || fail "T9 KV arithmetic checks"
python3 offload/kvpack/make_configs.py --check | grep -qx "verified 6 configs" || fail "T10 configs != fresh generation (sync + async, 6)"
src_cfg() { bash -c 'set -euo pipefail; set -a; . ./dsv41.env; PORT=8001; . "$1"; set +a; echo "$EXP_DOCKER_ENV"' _ "$1"; }
PACKS="offload/launch/configs/D-kv6-pack.env offload/launch/configs/D-kv6-async-pack.env"   # sync control + the plan's async base
for c in $PACKS; do
K1_MEMPROBE=/root/.cache/dsv41-memprobe-R1 src_cfg $c | grep -q "VLLM_DSV41_MEMPROBE=/root/.cache/dsv41-memprobe-R1" || fail "T11 flat memprobe prefix not passed ($c)"
! K1_MEMPROBE=/root/.cache/dsv41-memprobe/R1 src_cfg $c >/dev/null 2>&1 || fail "T12 a memprobe prefix with a subdirectory was accepted ($c)"
! K1_MEMPROBE=/tmp/x src_cfg $c >/dev/null 2>&1 || fail "T13 a memprobe prefix outside /root/.cache was accepted ($c)"
done
offload/kvpack/stage_overlay.sh verify-checkout >/dev/null || fail "T14 verify-checkout on this checkout"
cp -a offload/kvpack "$S/kvcopy"; echo "# tampered" >> "$S/kvcopy/overlay/vllm/v1/kv_cache_interface.py"
! "$S/kvcopy/stage_overlay.sh" verify-checkout >/dev/null 2>&1 || fail "T15 a tampered checkout overlay was accepted"
printf 'VLLM_DSV41_KV_PAIR=3\nVLLM_DSV41_AK_MAX_REQS=4\nVLLM_EXL3_MOE_TF_PROMPT=1\n' > "$S/env.txt"
RUN="KVPACK_TEST_RUNNING_ENV=$S/env.txt"
env KVPACK_TEST_RUNNING_ROOT=offload/kvpack/overlay KVPACK_TEST_RUNNING_ENV="$S/env.txt" $K verify-running local | grep -q "K1 switches: VLLM_DSV41_KV_PAIR=3$" || fail "T16 verify-running on the reviewed files"
env KVPACK_TEST_RUNNING_ROOT=offload/kvpack/overlay KVPACK_TEST_RUNNING_ENV="$S/env.txt" $K verify-running --expect "VLLM_DSV41_KV_PAIR=3" local >/dev/null || fail "T17 --expect matching set"
! env KVPACK_TEST_RUNNING_ROOT=offload/kvpack/overlay KVPACK_TEST_RUNNING_ENV="$S/env.txt" $K verify-running --expect "" local >/dev/null 2>&1 || fail "T18 --expect '' accepted a live switch"
! env KVPACK_TEST_RUNNING_ROOT=$SRC KVPACK_TEST_RUNNING_ENV="$S/env.txt" $K verify-running local >/dev/null 2>&1 || fail "T19 the v5 files were accepted as the running overlay"
n_run=$(grep -c 'on "$n" "docker run' $K); n_never=$(grep -c 'on "$n" "docker run --rm --pull never ' $K)
[ "$n_run" -ge 2 ] || fail "T20 expected the two docker run invocations, found $n_run"
[ "$n_run" = "$n_never" ] || fail "T20 $n_run docker run calls, only $n_never with --pull never"
python3 offload/kvpack/prefix_identity.py --dry-run --n 8000 >/dev/null || fail "T21 prefix_identity self-test / dry build"
for c in $PACKS; do
src_cfg $c | grep -q "VLLM_DSV41_MEMPROBE" && fail "T22 memprobe on without K1_MEMPROBE ($c)"
K1_MEMPROBE=/root/.cache/dsv41-memprobe-x src_cfg $c | grep -q "VLLM_DSV41_MEMPROBE_LAYERS=2,14,20" || fail "T23 memprobe without its layer list ($c)"
[ "$(bash -c 'set -euo pipefail; set -a; . ./dsv41.env; . "$1"; echo "${SPEC_OVERRIDE-unset}"' _ $c)" = unset ] || fail "T24 K1_NODROP is on by default ($c)"
K1_KV_BYTES=1610612736 bash -c 'set -euo pipefail; set -a; . ./dsv41.env; . "$1"; [ "$KV_BYTES_OVERRIDE" = 1610612736 ]' _ $c || fail "T25 K1_KV_BYTES override ($c)"
bash -c 'set -euo pipefail; set -a; . ./dsv41.env; . "$1"; [ "$KV_BYTES_OVERRIDE" = 6442450944 ]' _ $c || fail "T26 default KV bytes ($c)"
done
arith=$(python3 offload/kvpack/kv_arith.py)
grep -q "pressure R1b 0060 .*625,609 tokens; 4 x 240K needles .*(full attention 12,500 .*(preempts: full attention alone > pool).*evicts A" <<<"$arith" &&
  grep -q "pressure R1b v5 .*401,205 tokens; 4 x 240K needles .*(full attention 22,500 .*(preempts: full attention alone > pool).*evicts A" <<<"$arith" ||
  fail "T27 pressure arithmetic (4 x 240K needles preempt on full attention alone, packed and v5; ABA evicts A)"
echo "stage_overlay / build / arithmetic / configs / memprobe guard / verify modes / R-4 / P-1: 27 tests passed"
