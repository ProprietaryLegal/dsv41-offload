> **This branch:** `lane/upx` — Idle cache release and expert-row splitting (R-UPX battery runner). Code-only snapshot of private `offload/upx` at `d88d2cbf90c5` (2026-10-27).
> Tests on this exact tree: 2,797 passed, 756 failed or errored (distinct tests), 324 skipped (CPU only, no network). Of the failures: 358 also fail on the private source commit in the same sandbox (pre-existing: GPU, serving hosts, deployment state, git history a single-commit tree lacks, or an existing failure); 13 pass when their file runs alone; 377 need data on our hosts; 8 need a withheld prompt set or a file outside the data boundary. Details: [`reports/tests/TESTS.md`](reports/tests/TESTS.md).

# DSV41 offload — full source: DeepSeek-V4.1-Flash on two DGX Spark (GB10) nodes with an x86 RTX 3090 offload host

The complete code of PLI Labs' DeepSeek-V4.1-Flash deployment: tensor-parallel serving on two DGX Spark nodes, the DSpark
drafter moved to two NVLinked RTX 3090s over RDMA, Engram tables served from x86 RAM, packed KV pages, and the fenced
measurement harness used to qualify every change. Code only — see "What is not here".

**Hugging Face:** [collection](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e) · **Weights:** [DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark) · **Stack overview:** [DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · **Progress log:** [deepseek-v41-flash-2x-dgx-spark-progress](https://github.com/ProprietaryLegal/deepseek-v41-flash-2x-dgx-spark-progress) · **Hub:** [pli-labs](https://github.com/ProprietaryLegal/pli-labs)

<!-- pli:blurb:start -->

> Published by [PLI Labs](https://proprietarylegal.ai/labs.html), the open research arm of [Proprietary Legal Intelligence](https://proprietarylegal.ai). PLI builds AI systems around a legal organization's own work product, combining retrieval, staged workflows, validation, and attorney review. Visit [proprietarylegal.ai](https://proprietarylegal.ai) to explore our work and discuss what your organization needs.

<!-- pli:blurb:end -->

## Status — 2026-10-27

Labels: **MEASURED** on our hardware; **ESTIMATED** by arithmetic; **UPSTREAM** someone else's number on someone else's stack
(self-reported, not reproduced here).

| | Value | Label |
|---|---|---|
| Single-stream decode, production (short prompt / 32K / 150K, medians of 38 / 38 / 14 runs) | 57.5 / 51.8 / 51.7 tok/s | MEASURED |
| Decode per request with 3 / 4 concurrent users at 32K | 30.2 / 24.8 tok/s | MEASURED |
| Prefill at 150K tokens | ~2,170 tok/s (time to first token ~69 s) | MEASURED |
| Speculation on vs off (remote DSpark drafter) | 56.0 vs 41.9 tok/s short; 64.6 vs 41.0 at 32K | MEASURED |
| KV pool (packed pages, K1) | 2,502,600 tokens, +62 % vs unpacked; max request 327,680 | MEASURED |
| Remote Engram (x86 RAM over RDMA) | byte-exact; −5.4 to −5.7 % time to first token | MEASURED |
| Rank-0 firmware display reservation 2 GB → 512 MiB | +1,532 MiB usable memory | MEASURED |
| Best public two-Spark V4.1-Flash prose decode | 56.8–59.3 tok/s | UPSTREAM |

Exact versions, serving settings, overlays and the full measured table are in
[`reports/build/BUILD.md`](reports/build/BUILD.md). An outside review on 2026-10-27 placed single-user speed at the public
two-Spark frontier and pointed at three larger levers: a restart-surviving KV cache on the offload host, confidence-gated
drafting priced by experts touched, and a per-token decode time budget. The `lane/matter-cache`, `lane/census` and
drafter lanes below are the first steps on those.

## Layout

The repository mirrors the private deployment tree (same paths, so documents and code references line up):

| Path | What |
|---|---|
| `serve.sh`, `start.sh`, `stop.sh`, `dsv41.env`, `RUNBOOK.md` | two-node TP2 launch on the serving image (rank 0 head, rank 1 worker) |
| `offload/dspark/` | remote DSpark drafter: Spark-side `remote_dspark` speculator, x86 worker (EXL3 K2 experts, CUDA graphs, TP2 over NVLink), native-weight kernels, depth policy, sampled drafting (PP4), drafter-training tooling (D2) |
| `offload/drafter/acc_split/` | DP-2 acceptance split: drafted tokens accepted per round by prompt, token budget, segment, batch and window, with the metric reconciliation (analysis only) |
| `offload/engram/`, `offload/transport/` | Engram tables in x86 RAM; the `ofl` RDMA RC / TCP request–response transport |
| `offload/kvpack/` | packed KV pages and head-memory switches (K1) |
| `offload/prefixstore/` | prefix KV store tiers; on `lane/matter-cache` the MC1 matter tier |
| `offload/launch/` | fenced measurement windows: HOLD admission, scoped remote commands, W8-style restore, chain mode, the WIN2 decision rule (R3: seeded paired cells, smoke start, two-window cap, resolution filter, distributional gate) |
| `offload/gb10_budget/` | window cells, memory floors, budgets |
| `offload/upstream_adopt/`, `offload/upfix/`, `offload/up3/`, `offload/upx/` | adoption of upstream patches (HeNryous series, vLLM fixes, top-k kernels, idle cache release) |
| `offload/promote/` | promotion packages with exact rollback |
| `offload/audit/` | numerics and RNG audits |
| `reports/` | test results for this tree (`reports/tests/`) and the production build record (`reports/build/`) |

**Branches.** Each branch is a fresh snapshot with its own single commit; the source commit in the private repository is
named in the commit message.

| Branch | What it holds |
|---|---|
| `main` | The integration snapshot (all merged work, including the DP-2 acceptance split). |
| `production` | Exactly the code serving today (production tag `prod-k1-327-ops1-20261026`). |
| `lane/win2` | The shared measurement rule WIN2 with its third revision (R3). |
| `lane/census` | The expert census: CEN0 tables (experts and bytes touched per verify round) and the CEN1/CEN3/CEN4 window runners. |
| `lane/lsuite` | The legal evaluation suite: runner, scorers and pre-registered pass rule (task data not included). |
| `lane/matter-cache` | MC1, an opt-in, restart-surviving matter tier for the prefix KV cache (code only). |
| `lane/dw1` | DW1 drafter-wait measurements: per-stage step timer, one CUDA graph per step, fused-pair identity probe. |
| `lane/dp5` | DP-5 locked-clock sweep for the drafter worker GPUs, with its report tabulator. |
| `lane/pp4` | Sampled drafting (PP3/PP4): top-32 sampled drafts with their probabilities carried to the verifier. |
| `lane/pp4-win2` | PP4 admission and measurement on the WIN2 rule (including the admission API fix). |
| `lane/pp4-promote` | The PP4 promotion package with exact rollback. |
| `lane/nbw-win2` | The native-weight drafter (NBW) re-measure prepared under WIN2. |
| `lane/pc1-win2` | PC1 prefill-chunk experiments for time to first token (default-off settings, paired TTFT gates). |
| `lane/sp-prefill` | R-SP, prompt-only sequence-parallel prefill on K1 (CPU-qualified overlay, default off). |
| `lane/tv1` | TV1 verification reductions (default-off compact verification, FP8 candidate-only scorer, 0.9-nat near-tie). |
| `lane/up2-adopt` | Adoption of upstream patches (UP2) under the WIN2-R3 run sheet. |
| `lane/up3-win2` | UP3 upstream arms isolated on pinned production, in fenced two-node windows. |
| `lane/upfix` | Upstream vLLM fixes and top-k kernels (R-UPF). |
| `lane/upx` | Idle cache release and expert-row splitting (R-UPX battery runner). |

## What is not here (data boundary)

No model weights or shards; no prompt sets, captured requests or generated model text; no benchmark or window output files;
no client data of any kind. Specifically:

- **Prompt sets live outside the code.** `bench.py` and `offload/bench/bench2.py` read their workloads and record
  vocabulary from the JSON file named by `DSV41_BENCH_SPEC_FILE`; `offload/engram/tests/logit_ab.py` reads its prompts
  from `DSV41_LOGIT_AB_PROMPTS_FILE`; `offload/dspark/worker/tools/parity_ref.py` reads its text from
  `DSV41_PARITY_TEXT_FILE`; the six greedy reference prompts come from `DSV41_GREEDY_PROMPTS_FILE`. The code paths are
  unchanged; only the literals moved. Still withheld: `quality.py`, `functest.py`, `offload/dspark/d2/prompts.py` and its test.
- **Legal suite (`lane/lsuite`):** runner, scorers and the pre-registered pass rule only. The task file
  (`tasks.v1.json`) and its source data are not published; test fixtures are synthetic.
- **Matter cache (`lane/matter-cache`):** code only — no matter names, namespace keys, exclusion lists or their
  hashes. The exclusion tool fails closed until `--expect-sha256` names the operator's own reviewed list.
- **Host names and paths are role names and placeholders.** `sparkhead` = DGX Spark rank 0 (API), `sparkworker` = rank 1,
  `x86host` = the x86 offload host with the RTX 3090s, `controller` = the workstation that runs the measurement harness;
  `HEAD_IP`, `LAN_IP`, `TAILNET_IP` stand for addresses. Our directories appear as neutral absolute paths:
  `/opt/dsv41/data` (data and state on the controller), `/opt/dsv41/x86host-data` (the offload host),
  `/opt/dsv41/campaigns/DSV41-OFFLOAD` (measurement campaign files), `/opt/dsv41/repos`, and home directories
  `/home/operator`, `/home/rank0user`, `/home/rank1user`. Point them at your own locations before use. `GX10` is the ASUS
  product name. The `10.202.x.x` addresses are private point-to-point RoCE links.
- **Pinned hashes.** Where the public scrub changed a file whose sha256 another file pins (manifests, lock files, tests
  and measured-evidence notes such as `offload/audit/OPS1-STATUS.md`, `offload/dspark/d2/PASS2-ATTEST-EVIDENCE.md`,
  `offload/dspark/probabilistic/pp4/heldout-replay.json` and `heldout-summary.json`, and `offload/up3/manifests/`), the
  pin was updated to the published bytes, so the hash checks hold on this tree. The measured runs those notes record used
  the pre-scrub bytes, which differ from the published ones only by the role-name, path and prompt-loader changes.

## Tests and build

Every branch was tested as published, on CPU, in a network-isolated sandbox: [`reports/tests/TESTS.md`](reports/tests/TESTS.md)
gives passed / failed / skipped per branch and the reason for every failure (no GPU, no serving hosts, the serving
image's vLLM source tree not present, a withheld prompt file, or a whole-tree session artifact that passes when its file
runs alone). [`reports/build/BUILD.md`](reports/build/BUILD.md) records the production image digest, pinned commits,
serving settings, overlays and the measured performance table.

## Method notes worth copying

- Every change runs inside a fenced window: production stops under a HOLD, the candidate and control alternate on the same
  nodes, and production is restored and checked against six greedy references before the window closes.
- A schedule test judged only where the candidate and control are identical is an A/A test; it can still "lose" by chance.
  Measure where the configurations actually differ.
- Exact greedy identity is the right gate for restores and the wrong one for drafting changes: BF16 near-ties flip at
  0.5/1.0-nat logit steps. Use a near-tie tolerance (with the differing token located on the common prefix), and
  distributional checks.
- Paired per-request seeds and many short requests shrink run-to-run noise far more than longer runs.
- Acceptance depends mostly on the prompt (one 32K summary prompt holds 1.59 accepted drafts per round over 70 runs, one
  short drafting prompt 1.00 over 77): compare arms on the same prompt identity, never pooled across prompts.

## Licence

Original PLI Labs code: Apache-2.0 (`LICENSE`). Third-party material keeps its licence and attribution — see `NOTICE` and
`LICENSES/`. DeepSeek supplies the model and the DSpark drafter; no model files are distributed.
