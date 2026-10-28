# X1 Spark-side leads and policy package

All nine brief items are delivered as isolated source drafts, CPU tooling and coordinator test plans. Nothing here
has been deployed or benchmarked on a GPU. No Spark access, service changes, tensor payload downloads, private data
exports, pushes or merges were performed. Branch `offload/x1-leads`.

| Brief item | Deliverable |
|---|---|
| 1 | [0075 replay skip](01-replay-skip.md), v5 patch + actual tap-suffix handoff, pinned bases, CPU simulation/gather gates |
| 2 | [Remote0078 lookahead](02-remote-lookahead.md), O1 single-reader patch, bounded cancellation/cache core, host hash bridge, staging/header-hash changes |
| 3 | [0051/64/66/67 verdicts](03-equivalence-verdicts.md), target/worker equivalents and real residual work |
| 4 | [Bounded replay audit](04-bounded-replay-audit.md), DeepInfer retained-suffix comparison and backtracking counterexample |
| 5 | [Depth table plan](05-depth-cost-plan.md), strict paired-round schema and CPU fitter `depth_cost.py` |
| 6 | [Teacher-forced comparison](06-fixed-prefix-plan.md), dense full-vocabulary capture schema and `fixed_prefix.py` |
| 7 | [Mixed arrivals](07-mixed-arrival-design.md), elapsed decoder budgets, adaptive chunks and long-task starvation gates |
| 8 | [Representation inventory](08-draft-representation-metadata.md), four pinned HF API manifests/14 headers and `hf_manifest.py` |
| 9 | [Donor per-op leads](09-per-operation-donor-leads.md), pinned sudoingX/Deepinfer-SM80 applicability, measurement and stops |

Reproduce CPU checks on Controller:

```bash
source ~/.config/ndm/paths.sh
cd /opt/dsv41/repos/.worktrees/dsv41-x1
"$HARNESS_PYTHON" -m pytest -q offload/spark_leads/tests
"$HARNESS_PYTHON" offload/spark_leads/compose_check.py --out /tmp/X1-composition.json
```

Final X1 suite: **38 passed**. Combined earlier X1+existing O3 suite: **195 passed,1 skipped** in66.41s (before two
additional final X1 cases). The pre-existing optional O1 TCP transport test skips because this worktree has no
`offload/dspark/spark/.build/ofl/libofl.so` + `ofl.py`. This is classified as an unavailable optional native-transport
fixture, not an X1 pass. No test assertions were weakened and no skips added. Native cache core compiled at C++17,
`-O2 -Wall -Wextra -Werror`; patched Python/shell source syntax verified, including actual K1 P1 composition.
CPU fused gather checks use Triton's interpreter, never CUDA. Source reference defaults to
`/opt/dsv41/data/src-image`; the replay fixture accepts `DSV41_V5_SOURCE` for another pinned copy.

Integration:
- K1 P1 landed during X1 at **87b08b5c486e68458223ea38dca6bb911b678400**. `compose_check.py` reads that git tree
  without checkout/merge, validates manifest/base/overlay hashes, applies both drafts with **zero fuzz**, compiles
  all composed Python and checks shell syntax. `patches/K1-composition.json` records resulting23 file hashes.
- K1 owns the deployment mechanism. Coordinator composes its engine-core/worker mounts with X1 runner/model-state/
  InputBatch/scheduler/bridge files, rather than overwriting the K1 overlay or adding competing mount targets.
  O3 gather patch goes into the one authoritative remote_dspark package, coordinated with A2/W1 changes.
- O1 source remains its existing package; draft updates its staging allowlist and JIT header hash. The module bridge
  is a P1 engine mount. Full C++ JIT/linking and GB10 ordering/visibility have NOT been tested. The native cache core
  test does not qualify the torch/CUDA/ofl integration. Both switches default off; do not enable in production.
- Fine paired Engram issue/ready/consumer phase markers are specified in item2; current draft exports row/hit
  counters, not those full markers. Coordinator must add/collect them before attributing a performance result.
- CONTRACT1.8 needs the optional real-tap provenance rule and forward-only/backtracking qualifications at integration.
  Current typed patch makes invalid drafting provenance fail loudly instead of sending dropped zero rows.

Review corrections are included in the final patch: host-wrapper discovery, proxy teardown ordering, partial-cache
miss fetch/scatter (all-hit-only would be ineffective at4096-token chunks), header in the JIT hash/staging allowlist,
and native cache hit statistics. The39/40/…ms R1 gather number is queued target wait as well as extraction, so it is
not used as a remote prep measurement. Historical SSE/top20 JSONL are not mislabelled as paired costs or full KL.

Provenance and licenses: [NOTICE](NOTICE), [Apache2.0](licenses/Apache-2.0.txt); per-item notes pin donor revisions.
Public metadata artifacts contain no tensor values. Campaign progress, raw CPU logs, metadata failures/retry record,
and final report are under `/opt/dsv41/campaigns/DSV41-OFFLOAD/X1*`.
