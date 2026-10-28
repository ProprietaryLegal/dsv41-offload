#!/bin/bash
# Production layout of the TP = 2 DSpark worker (S10 P-1; R2-RUNTIME). Coordinator-run; nothing here enables, starts or
# stops a unit. Layout (all under ~/dsv41-offload/prod on x86host):
#   dspark-worker/<commit>/offload/{dspark,transport}   code (git archive), shared with the one-card unit
#   dspark-worker/current -> <commit>                   flipped by the coordinator (ln -sfn), never by this script
#   dspark-worker/lib/{libofl.so,ofl.py}                transport pin (sha256 9f11b6e8..., the one-card pin)
#   dspark-worker/p2p/<src16>/dspark_p2p.so(.sha256)    the p2p extension built for one p2p.cu (worker/p2p/prebuilt.py)
#   run/exl3_tune_cache_tp2.bin, run/dspark-worker.status.json, logs/dspark-worker.log
#
#   prod_tp2.sh stage <commit> [repo]     ON Controller: git archive <commit> offload/dspark offload/transport from repo
#                                         (default: this checkout) into x86host:~/dsv41-offload/prod/dspark-worker/<commit>;
#                                         refuses an existing directory
#   prod_tp2.sh build-p2p <commit> [root] ON x86host: compile that commit's p2p.cu with the worker's own build function
#                                         (CPU only, CUDA_VISIBLE_DEVICES empty) and install the .so + its sha256 under
#                                         root (default ~/dsv41-offload/prod/dspark-worker/p2p)/<src16>/
#   prod_tp2.sh install-unit              ON x86host: install dsv41-dspark-worker-tp2-prod.service (the production unit) into
#                                         ~/.config/systemd/user and daemon-reload; seeds the tune cache from lane E's.
#                                         Never enables it
#   prod_tp2.sh check <commit> [root]     ON x86host: everything the unit needs for <commit>, read-only
#
# Promotion, on the decision only (one-card unit stays enabled until then):
#   ln -sfn <commit> ~/dsv41-offload/prod/dspark-worker/current
#   systemctl --user disable --now dsv41-dspark-worker && systemctl --user enable --now dsv41-dspark-worker-tp2-prod
# Rollback:
#   systemctl --user disable --now dsv41-dspark-worker-tp2-prod && systemctl --user enable --now dsv41-dspark-worker
# DSV41_PROD_REL (default dsv41-offload/prod, relative to $HOME on both hosts) points every subcommand at another tree,
# e.g. a lane's rehearsal of the layout.
set -euo pipefail
PROD_REL=${DSV41_PROD_REL:-dsv41-offload/prod}
PROD=$HOME/$PROD_REL
WD=$PROD/dspark-worker
OFL_SHA=9f11b6e8
PY=$HOME/dsv41-offload/o2/venv/bin/python
UNIT=dsv41-dspark-worker-tp2-prod.service
here=$(cd "$(dirname "$0")" && pwd)
cmd=${1:-}
shift || true

src16() { sha256sum "$1" | cut -c1-16; }

case "$cmd" in
  stage)
    commit=${1:?commit}; repo=${2:-$(cd "$here/../../../.." && pwd)}
    full=$(git -C "$repo" rev-parse --verify "$commit^{commit}")
    short=$(git -C "$repo" rev-parse --short=7 "$full")
    rwd="~/$PROD_REL/dspark-worker"
    ssh -o BatchMode=yes x86host "test ! -e $rwd/$short" \
      || { echo "refusing: x86host:$rwd/$short exists" >&2; exit 1; }
    ssh -o BatchMode=yes x86host "rm -rf $rwd/$short.tmp && mkdir -p $rwd/$short.tmp"
    git -C "$repo" archive "$full" offload/dspark offload/transport \
      | ssh -o BatchMode=yes x86host "tar -x -C $rwd/$short.tmp && echo $full > $rwd/$short.tmp/SOURCE_COMMIT && \
          mv $rwd/$short.tmp $rwd/$short"
    echo "staged $full as x86host:$rwd/$short (current unchanged)"
    ;;
  build-p2p)
    commit=${1:?commit}; root=${2:-$WD/p2p}
    tree=$WD/$commit/offload/dspark
    [ -f "$tree/worker/p2p/p2p.cu" ] || { echo "no $tree/worker/p2p/p2p.cu (stage $commit first)" >&2; exit 1; }
    key=$(src16 "$tree/worker/p2p/p2p.cu")
    dst=$root/$key
    if [ -f "$dst/dspark_p2p.so" ]; then echo "already built: $dst"; cat "$dst/dspark_p2p.so.sha256"; exit 0; fi
    build=$(mktemp -d "${TMPDIR:-/tmp}/dspark-p2p-build.XXXXXX")
    trap 'rm -rf "$build"' EXIT
    (cd "$tree" && env -u DSPARK_P2P_PREBUILT CUDA_VISIBLE_DEVICES= DSPARK_P2P_BUILD="$build" \
       "$PY" -c "from worker.p2p import ext; e = ext(); print('built', e.__file__)")
    mkdir -p "$dst.tmp"
    install -m 755 "$build/dspark_p2p.so" "$dst.tmp/dspark_p2p.so"
    (cd "$dst.tmp" && sha256sum dspark_p2p.so > dspark_p2p.so.sha256)
    { echo "source $tree/worker/p2p/p2p.cu sha256 $(sha256sum "$tree/worker/p2p/p2p.cu" | cut -d' ' -f1)"
      echo "commit $commit ($(cat "$WD/$commit/SOURCE_COMMIT" 2>/dev/null || echo unknown))"
      echo "built $(date -u +%FT%TZ) on $(hostname) with $("$PY" -c 'import torch; print("torch", torch.__version__)') and" \
           "$(${DSPARK_P2P_CUDA_HOME:-/usr/local/cuda-12.6}/bin/nvcc --version | tail -1)"; } > "$dst.tmp/BUILD.txt"
    mv "$dst.tmp" "$dst"
    echo "installed $dst"; cat "$dst/dspark_p2p.so.sha256" "$dst/BUILD.txt"
    ;;
  install-unit)
    mkdir -p ~/.config/systemd/user "$PROD/run" "$PROD/logs"
    install -m 644 "$here/../dsv41-dspark-worker-tp2.prod.service" ~/.config/systemd/user/$UNIT
    systemctl --user daemon-reload
    if [ ! -f "$PROD/run/exl3_tune_cache_tp2.bin" ] && [ -f "$HOME/dsv41-offload/e/run/exl3_tune_cache.bin" ]; then
      install -m 644 "$HOME/dsv41-offload/e/run/exl3_tune_cache.bin" "$PROD/run/exl3_tune_cache_tp2.bin"
    fi
    echo "installed $UNIT (enabled: $(systemctl --user is-enabled $UNIT 2>/dev/null || true))"
    ;;
  check)
    commit=${1:?commit}; root=${2:-$WD/p2p}
    tree=$WD/$commit/offload/dspark
    ok=1
    [ -d "$tree/worker" ] || { echo "MISSING code $tree"; ok=0; }
    echo "current -> $(readlink "$WD/current" || echo none)"
    s=$(sha256sum "$WD/lib/libofl.so" | cut -c1-8)
    [ "$s" = "$OFL_SHA" ] && echo "libofl.so $s (pinned)" || { echo "libofl.so $s != pinned $OFL_SHA"; ok=0; }
    if [ -f "$tree/worker/p2p/p2p.cu" ]; then
      key=$(src16 "$tree/worker/p2p/p2p.cu"); so=$root/$key/dspark_p2p.so
      if [ -f "$so" ] && [ "$(sha256sum "$so" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$so.sha256")" ]; then
        echo "p2p prebuilt $so ok"
      else echo "MISSING/BAD p2p prebuilt $so (build-p2p $commit)"; ok=0; fi
    fi
    [ -f ~/.config/systemd/user/$UNIT ] && echo "unit installed (enabled: $(systemctl --user is-enabled $UNIT 2>/dev/null || true))" \
      || { echo "unit not installed (install-unit)"; ok=0; }
    for u in dsv41-dspark-worker.service dsv41-dspark-worker-tp2.service $UNIT; do
      echo "$u: $(systemctl --user is-active $u 2>/dev/null || true), $(systemctl --user is-enabled $u 2>/dev/null || true)"
    done
    [ -f "$PROD/run/exl3_tune_cache_tp2.bin" ] || echo "note: no tune cache yet (created at first start)"
    [ "$ok" = 1 ] && echo "CHECK OK" || { echo "CHECK FAILED"; exit 1; }
    ;;
  *)
    sed -n 2,29p "$0"; exit 2 ;;
esac
