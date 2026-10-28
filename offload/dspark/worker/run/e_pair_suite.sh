#!/bin/bash
# Lane E: everything that needs the NVLink pair, in order, each step logged; stops at the first failure. Run ON
# x86host through e_job.sh (pair) only after the coordinator's "pair free"; start e_powerwatch.py on the session.
# Budget (coordinator): < 14 GiB of ours on GPU-c4e7a371 (the live one-card worker stays up there, 8 GiB, idle);
# extra models (the emulation, the one-card reference) go to GPU-7744b44b. e_powerwatch.py enforces it.
#   watchdog FIRST (it snapshots the processes already on c4e7a371, then waits for the session), then the suite:
#   nohup python3 worker/run/e_powerwatch.py e-pair ~/dsv41-offload/e/results/e7_pair_power.jsonl > ~/dsv41-offload/e/logs/powerwatch.log 2>&1 &
#   bash worker/run/e_job.sh e-pair pair ~/dsv41-offload/e/logs/pair_suite.log bash worker/run/e_pair_suite.sh
set -euo pipefail
PY=$HOME/dsv41-offload/e/venv/bin/python
R=$HOME/dsv41-offload/e/results
ROOT=/opt/dsv41/x86host-data/dspark
CAP=$ROOT/captures/A-capture-20261022/dspark-captures
step() { echo "=== $(date -u +%FT%TZ) $*"; }
for u in GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93 GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc; do
  nvidia-smi nvlink -e -i $u > $R/e7_nvlink_errors_before_${u:4:8}.txt
done
step "1 p2p tests + TP2 real == emulation on the pair (the single-card TP2 tests ran on 7744b44b: their module
#   fixture would put a 10 GB emulation on GPU-c4e7a371, which hosts the live worker during the window)"
$PY -m pytest -q -s worker/tests/test_p2p.py "worker/tests/test_tp2.py::test_pair_equals_emulation"
step "2 pair bench (timings, collectives, NVLink counters, memory)"
$PY -m worker.tools.tp2_bench pair $ROOT $R/e7_bench_pair.json 300
step "3 server selftest tp=2 over the O1 transport (x86host loopback on 10.202.28.1:18627)"
OFL_PY_DIR=$HOME/dsv41-offload/e/lib OFL_LIB=$HOME/dsv41-offload/e/lib/libofl.so \
  $PY -m worker.tools.server_selftest $ROOT $R/e7_selftest_pair.json 18627 2
step "4 capture parity on the pair"
$PY -m worker.tools.tp2_parity $ROOT $CAP $R/e7_parity_pair.json
for u in GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93 GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc; do
  nvidia-smi nvlink -e -i $u > $R/e7_nvlink_errors_after_${u:4:8}.txt
done
step "done"
