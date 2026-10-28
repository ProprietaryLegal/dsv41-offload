#!/bin/bash
# Pin lane O1's transport for the worker (run ON x86host): copy libofl.so and ofl.py into ~/dsv41-offload/o2/lib/ so a
# later O1 build or install can never replace the library under a running worker. Records sha256s in PINNED.txt.
# Usage: pin_ofl.sh [O1_LIB] [O1_PY]   (defaults: O1's installed bin/libofl.so and src/offload/transport/ofl.py)
# OFL_PIN_DST overrides the destination (lane E's TP = 2 unit: ~/dsv41-offload/e/lib).
set -euo pipefail
LIB=${1:-$HOME/dsv41-offload/o1/bin/libofl.so}
PY=${2:-$HOME/dsv41-offload/o1/src/offload/transport/ofl.py}
DST=${OFL_PIN_DST:-$HOME/dsv41-offload/o2/lib}
mkdir -p "$DST"
install -m 755 "$LIB" "$DST/libofl.so.new" && mv -f "$DST/libofl.so.new" "$DST/libofl.so"
install -m 644 "$PY" "$DST/ofl.py.new" && mv -f "$DST/ofl.py.new" "$DST/ofl.py"
{ echo "pinned $(date -u +%FT%TZ) from $LIB and $PY"; (cd "$DST" && sha256sum libofl.so ofl.py); } > "$DST/PINNED.txt"
cat "$DST/PINNED.txt"
