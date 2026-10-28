# GB10 budget tools

CPU-only analysis and preparation:

```bash
source ~/.config/ndm/paths.sh
CUDA_VISIBLE_DEVICES= "$HARNESS_PYTHON" -m offload.gb10_budget.evidence \
  --results offload/results --extra-results /opt/dsv41/repos/dsv41-deploy/offload/results/K1 \
  --output /opt/dsv41/campaigns/DSV41-OFFLOAD/SP1-evidence.json
"$HARNESS_PYTHON" -m offload.gb10_budget.model
"$HARNESS_PYTHON" -m offload.gb10_budget.prepare_profile \
  --base offload/launch/configs/D-kv6-async-pack.env --output /tmp/SP1-profile.env
"$HARNESS_PYTHON" -m offload.gb10_budget.capture --output /tmp/unused-SP1-preview.json
```

The last command previews without any network activity. `capture --execute` is for the
coordinator alone, after opening an isolated window and starting the prepared profiling
configuration on experiment port 8001. It requires loopback forwarding at port 18001.
It never starts/stops/reconfigures a model service. It does send benchmark requests and
start/stop-profile API calls when explicitly executed. Generated content stays in memory
only and is removed from result objects before writing timing/usage artifacts.

`prepare_profile` only writes a fresh .env. It preserves the chosen reviewed base,
adds a bounded PyTorch profiler and a separate host trace directory mount, and does
not execute the file. Both global TP ranks are profiled by vLLM's collective API;
local_rank=0 occurs on both nodes and must not be used to claim a single-rank capture.
No CPU stack, tensor-shape, allocation or frontend tracing is enabled. The trace has
worker iteration annotations and CPU/CUDA activity. Profiled observations are for
attribution, not promotion. Pair them with unprofiled measurements.

Decode capture waits for every concurrent stream to emit text. Prefill capture starts
after prompt construction but before submission, or after `--prefill-delay` seconds
if specified. Delayed capture refuses if any stream has reached decode. Existing
public-corpus summary (32K) / synth (150K) prompts and T=1/top_p=1/high thinking/seed42/
priority10 are reused from bench2. It fails if real corpus or model capacity is missing.

The report, exact coordinator commands, full evidence appendix and source hashes live
outside git in `/opt/dsv41/campaigns/DSV41-OFFLOAD/SP1*`.
`experiments.json` contains ranked hypotheses, never measured speedups.

SP1-WINDOW (one profiling HOLD of today's production): the coordinator sheet is [PROFILE-WINDOW.md](PROFILE-WINDOW.md)
(precheck, runner, analysis commands and the decision table); design and evidence are in PROFILE-WINDOW-DESIGN.md and
PROFILE-WINDOW-RECONCILE.md.

SP1-WINDOW trace budget (after the coordinator's profiling window; CPU only, never a service):

```bash
CUDA_VISIBLE_DEVICES= "$HARNESS_PYTHON" -m offload.gb10_budget.trace_budget \
  --window /opt/dsv41/campaigns/DSV41-OFFLOAD/SP1-W-<tag>
```

It verifies every trace against `traces-manifest.json` (size, sha256), reads both ranks' torch traces as a stream,
and writes `trace-budget.json` (never overwritten): per-step decode and per-chunk prefill GPU budgets by the
pre-registered kernel-name buckets in `buckets.json`, idle split into host gaps / device waits / drafter waits,
coverage, a CUDA-graph visibility flag, client-side profiler overhead from the cell artifacts, and the
`experiments.json` hypotheses each budget supports. Shares are attribution in a profiled run, not promotion timings.

Verification (CPU only):

```bash
source ~/.config/ndm/paths.sh
CUDA_VISIBLE_DEVICES= "$HARNESS_PYTHON" -m pytest -q offload/gb10_budget
"$HARNESS_PYTHON" -m mypy --strict --explicit-package-bases offload/gb10_budget
```

Rate comparisons follow [the repo convention](../evidence/THROUGHPUT.md): `R = (1 + E[A]) / C`, accepted drafts
exclude bonus, complete-round C includes dependencies/host gaps. Component timelines and SSE delivery p50 do not
by themselves measure C. [EV1 sampled timing](../evidence/TIMING.md) adds round correlation and actual row counts.
