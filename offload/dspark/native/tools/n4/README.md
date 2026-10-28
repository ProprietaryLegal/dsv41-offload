Status 2026-10-23: Official attention/head arithmetic port d1dfe13 merged after S20 MERGE; strict attention-band limitation retained, physical-pair timings and GB10 A/B still required. See [live roadmap](../../../../../documents/plans/roadmap.md) and [measured record](../../../../results/COMPARISON.md); campaign review decisions are in `LANES.md` / lane reports outside the repo.

# N4 native arithmetic and reproduction

The four serving arms keep their native weight values and retained EXL3 K6
head. Their attention now follows the official online64 schedule and the
head/base/Markov sum remains FP32. The K2 control is unchanged. Source policy
and pins: `documents/plans/lanes/N4-official-arithmetic.md`.

Strict V1 remains red on stage0 attention excess. All corrected pilots clear
state flags, unexplained disagreements and KL. An unmodified official GPU
attention component control also fails the attention band. Read the campaign
`N4-NATIVE-FIDELITY.report.md` before interpreting these as promotion evidence.
No gate or reference was relaxed. `evidence/v1-summary.json` includes original
source hashes and all historical, same-card baseline and corrected verdicts.

## Hardware and staging

N4 uses only X86host `GPU-bc62df2a-59cf-9828-7d43-558d1355bab7`, fixed 225 W,
under `~/dsv41-offload/locks/gpu-bc62df2a.lock`. The coordinator reassigned it
from 7744b44b; old N2/N3 latency numbers are not comparison baselines. Production
PAIR_OK refers to the separate NVLink pair. `run_locked.sh` owns each bounded
830+5s hold, checks actual occupancy/cap, samples memory and releases between
steps. STOP is `<N4_ROOT>/STOP`. It never changes a service, power or clocks.

The measured root is `~/dsv41-offload/n4` on X86host:

- `repo/`: arithmetic frozen at 55ceecc; later tools/docs do not change it.
- `before/`: 0035319 arithmetic, with only the bc ownership changes in
  `worker/tp2.py` and `kernels/fp8dense/api.py`, and extra benchmark source hashes.
- `lib/`, `p2p/`: existing qualified OFL and matching source-pinned p2p binaries.
- `build/`, `triton/`: owned compilation caches.
- `verify-source/`: read-only V1 package copy; full classification runs use the
  independent Controller V1 worktree and original reference dumps.
- `logs/`, `results/`: private receipts, traces and sampler data. Never commit
  tensor dumps, captures, logits or decoded weights.

Repeat into a fresh root with both frozen trees and their SOURCE_COMMIT files,
receipt, library pins and verification package. Completed receipts intentionally
refuse in-place reuse; a fresh root also preserves sampler files and logs.
Model files are reused at the existing S2 paths; no weight download is needed.

## Checks and measurements

On Controller, from this worktree:

```bash
source ~/.config/ndm/paths.sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 "$HARNESS_PYTHON" -m pytest -q \
  offload/dspark/native/tests/test_fidelity_reference.py \
  offload/dspark/native/tests/test_solo_ownership.py
```

On X86host, launch the GPU scripts through detached tmux with `bash` and the
fresh N4_ROOT set inside its command. The wrapper sets the UUID inside each
process, so it does not depend on tmux inheriting CUDA_VISIBLE_DEVICES.

- `repeat.sh`: 48 before/after acquisitions, all four arms/both ranks/three
  interleaved repetitions. Each B1–B4/n_out4–5 case settles 0.5 s, then records 200
  CUDA-event samples and full kernel maps. It retains the N2 benchmark's 327680
  RoPE allocation and eight graph shapes.
- `qualify.sh`: eight corrected actual solo shards, 1M RoPE allocation, full 40
  worker graph shapes and 2,000 replays per shard.
- `qualify_before.sh`: the same eight full 40/1M baseline checks, including
  matched load and isolated serving-memory peaks.
- `run_locked.sh JOB -m pytest -q offload/dspark/native/tests/test_fidelity_gpu.py`:
  13 synthetic GPU checks, including exact closed-form block/sink rounding,
  mixed short/wrapped requests, graph replay and FP32 Markov dyadic values.
- `prefix_check --out PRIVATE.json`: complete packed TP2 emulation through the
  server factory. Requires DSPARK_WORKER_TP_EMULATE=1 (the wrapper sets it).
  All 40 B/n_out/margin cases must preserve exact IDs/confidence prefixes and
  zero unused tails. This tests the shared FP32 head path, not real P2P.

Aggregate with the collectors only after the cohorts finish:

```bash
source ~/.config/ndm/paths.sh
cd ~/dsv41-offload/n4/repo
CUDA_VISIBLE_DEVICES= ~/dsv41-offload/e/venv/bin/python -m offload.dspark.native.tools.n4.collect \
  --root ../results/repeated --out ../results/repeated-summary.json
CUDA_VISIBLE_DEVICES= ~/dsv41-offload/e/venv/bin/python -m offload.dspark.native.tools.n4.collect_qualification \
  --root ../results --expected offload/dspark/native/tools/n4/evidence/identities.json \
  --out ../results/qualification-summary.json
CUDA_VISIBLE_DEVICES= ~/dsv41-offload/e/venv/bin/python -m offload.dspark.native.tools.n4.collect_qualification \
  --root ../results --expected ../baseline-identities.json --prefix baseline-qualified \
  --out ../results/baseline-qualification-summary.json
```

Collectors reject incomplete scopes, source/identity drift, wrong UUID or cap,
and missing graph/sample coverage. The inherited timing tool's raw serving_peak
counter is **process high-water through load/prewarm/serving/profiling**; it is
not reset after load. Its SMI value is a post-serving snapshot. The separate
full 40 shard receipts reset before serving, and their timestamped phase files
permit actual one-second observed load/prewarm/serve SMI peaks. Keep these
methods separate; all are solo local-collective stand-ins, not physical TP2.

## Reference probes and V1

`probe.py` only observes private raw native attention during the actual traced
serving replay. `reference_probe.py` saves matching pinned CPU inputs.
`compare_probe.py` reports aggregate identical-input errors. `official_probe.py`
hash-checks and runs unmodified official kernel.py on independent 16-head chunks
that fit SM86 shared memory. Its isolated dependencies are pinned in
`reference-requirements.lock`; Torch/NumPy/CUDA remain the existing environment.
`counterfactual.py` injects only the stage0 raw result into the official CPU
pipeline; optional private V1-format output is explicitly labelled a component
control, not a worker receipt. The unchanged N3 wrapper's hardcoded K6 label
does not describe the control's official BF16 head.

Actual worker pilots use the original `native.tools.n3.replay` and unchanged
`native.tools.n3.classify`, with explicit requested keys and original
mirror/mirror64 data. Full commands/logs are in the campaign's N4-*.sh files.
Original V1 classify/ref_kernels/native_ref/official and the N3 wrapper are
byte-checked against their pinned commits in `evidence/v1-summary.json`.

## Coordinator pair and GB10

`pair.sh` is prepared for the coordinator and was never executed by N4. It
requires N4_PAIR_BY_COORD=1 and the original explicit N3_V1_KEYS. It first
verifies all four frozen identities, then delegates to N3's unchanged pinned,
bounded, dual-lock pair script. Alias: `~/dsv41-offload/n4/n4_pair.sh`.
Final integration edits require regenerating identities/configs/expectations.
Physical pair, strict V1 clearance, engine measurements and depth-policy work
remain prerequisites for promotion. No pair or GB10 gain follows from these
solo timings or graph checks.

## Completed same-card cost evidence

`evidence/repeated-summary.json` validates all 48 acquisitions and 64 comparison
cells. Median three-run p50 increases 0.64–7.77%; the arithmetic is retained.
`evidence/memory-comparison.json` records both full40/1M memory cohorts: isolated
serving allocation falls 1006.9–1010.9 MiB per rank. `evidence/lock-audit.json`
checks the 48 timing and 16 memory job intervals: all rc0, no overlap, maximum
logged acquired-to-release 233 seconds, followed by bounded wrapper cleanup.
Strict attention qualification remains unresolved; these costs do not clear it.
