# STOP — PP3 is an OFF correctness prototype, not a serving-speed candidate

| Brief item | Delivered | Result / remaining gate |
|---|---|---|
|1 Fused producer/K2|dyadic24 top32 primitive, keyed Philox, T0 RNG bypass, optional eager K2 hook, B1–B4/depth4–5 measurements|Two launches/position, not one. B1depth5 .155443ms; B4 .208589ms. Misses absolute .05ms goal. Synthetic conditioned-logit selector only; actual K2/emulated model/pair costs unqualified.|
|2 Async q/verification|Negotiated extension, request seed/T mask, sparse q+CRC, both proxy modes/shared pools, FP32 full-vocab q, pinned standard verifier+failure mask overlays|100K GPU toy-law and100K T0 comparisons PASS; exact PP1 law using actual GPU q PASS; CPU thread/process mixed-T/failure paths PASS. Engine grammar/state and real pair/GB10 qualification remain.|
|3 Estimate/window|Hash-bound PP2 freeze, .182813ms known proxies, explicit unknown costs, short/32K/150K sensitivities,69min ABBA plan capped75min|Live rate remains null.150K acceptance unmeasured. No window opened or recommended for current prototype.|
|4 Validation/handoff|CPU/GPU evidence, strict typed new runtime components, source/hash reconciliation, reproduction notes, cleanup/report|See campaign PP3.validation.log and PP3.report.md for final counts, commits and cleanup.|

Production remains unchanged. Neither switch is enabled by the commits. Ordinary
stage_overlay.sh does not deploy the prepared verifier/runner files. The sampled
startup guard requires both ABI markers and refuses incompatible policies.

The required /home/operator/miniconda3/bin/python3 on x86host isCPython3.13 with
Torch2.12.1/cu130 and Triton3.7.1, but no exllamav3_ext_facc. Production's EXL3
binaries areCPython3.12. PP2 used a cached separateCPython3.12 interpreter;
PP3 did not substitute it or install/rebuild dependencies. Therefore no claim
of full production-K2 TP2 emulation or paired greedy identity is made here.

The eager hook still computes the existing greedy selector, recomputes Markov
logits and copies full vocabulary shards GPU-to-GPU before drawing. The
single-launch/candidate-exchange/graph work must be finished and measured
before any promotion decision. Graph-replayed selector costs and SM86 sampler
costs are not exposed engine timings; PP2's float64 nominal top32/.7 acceptance
is not new dyadic24 replay evidence. [COST.md](COST.md) keeps these distinctions.

Coordinator next action: review the prototype and choose a follow-on optimization
and interpreter qualification lane. Close full K2/pair/GB10/state gates, fill
all missing costs and independently review before considering the pre-registered
[LIVE-AB-PLAN.md](LIVE-AB-PLAN.md). This handoff requests no service mutation.
