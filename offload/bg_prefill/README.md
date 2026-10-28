# SO2 background scheduling experiment

Read DESIGN.md for the service bound and its explicit scope. Default OFF;
foreground priorities 0/10 and deployed 0031 floor5/cadence24 are unchanged.
Production is not modified. One scheduler mount composes on the pinned installed
scheduler, not SO1 or UP1's scheduler; the builder refuses another base.
UP2/0075 also replaces that scheduler: composition requires one hand-composed
scheduler and a fresh independent check. SO2 stays OFF; no GB10 window is
authorized now. See GB10-TEST-PLAN.md for the blocking preconditions and recorded
restore target. Sustained overlap requires a future non-aborting background
client; D2 already aborts on foreign demand.

CPU checks (from repository root):

```bash
source ~/.config/ndm/paths.sh
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
"$HARNESS_PYTHON" -m pytest -q -p no:cacheprovider \
  --basetemp=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/SO2-BG-PREFILL/cpu \
  offload/bg_prefill/tests/test_scheduler.py offload/bg_prefill/tests/test_build.py \
  offload/bg_prefill/tests/test_cadence.py
/opt/dsv41/data/kvpack-build/venv/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=/opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/SO2-BG-PREFILL/real \
  offload/bg_prefill/tests/test_real_kv.py
MYPYPATH=. "$HARNESS_PYTHON" -m mypy --strict --follow-imports=silent \
  --explicit-package-bases offload/bg_prefill
"$HARNESS_PYTHON" -m offload.bg_prefill.build_overlay \
  --output /opt/dsv41/campaigns/DSV41-OFFLOAD/scratch/SO2-BG-PREFILL/build
```

Build destination must not exist. MANIFEST.json records input and output hashes;
the build creates just scheduler.py, so2_policy.py and that receipt. Mount both
code files at their relative paths under
`/usr/local/lib/python3.12/dist-packages/`, scheduler read-only. The generated
scheduler imports the single policy module. No test shim is mounted.

Experiment switches, read at scheduler import:

- `VLLM_SO2_BG_PREFILL=0` (default): no overlay state/counters or scheduling changes.
- `VLLM_SO2_BG_PREFILL=1`: enable, with `VLLM_SO2_PERIOD=24` and
  `VLLM_SO2_QUANTUM=32`. Invalid switches refuse startup with a useful error.

No hot-file switch, worker-depth override or cadence100000 is used. Each enabled
scheduler owns `so2_stats`: steps, native (unsupported fallback calls), mixed,
admission, maintenance and pauses (request skips). Read these in a controlled
coordinator trace, including a final snapshot. Counters are process-local; they
are not a new service or hidden production log. Async overlap and actual elapsed
stall bounds still need GB10 measurement. A step guarantee is not a time guarantee.

Tests execute D2-FIX1's pinned image Scheduler/AsyncScheduler/Request/queues.
Dependency imports, __init__, KV and synthetic delivery are substituted in the
fast suite. The S19 suite uses the real image KVCacheManager with the pinned
S8R2 cache-spec bootstrap (1.25 GiB, 9709 blocks); remote completion notifications
are synthetic. The exact S19 trace function is unchanged and hash-checked. Two
additional variants assign those holders priority20 through setup. S15's admitted
runner/infeasible waiter state is encoded for sync/async, priority10/20 and both
throttle/release calls. No transport, model forward, GPU or output-equivalence
claim follows from these tests.
