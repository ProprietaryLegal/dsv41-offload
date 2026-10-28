# UP1 upstream applicability audit

Scope: b172664 (v5) .. afe4d5e41ec6fa4db17c5cc88758af439a106913 (v24).
Original lane base was 6b99476; UP1-FIX1 merged offload-x86host c704523 and re-pinned all inputs before rebuilding.
This is an adoption candidate set, not a production image upgrade. Donor files were read as data.
No launcher, Docker build, GPU or remote command ran.

`PATCH-CATALOGUE.md` gives EVERY patch's function, upstream README status and measured effect verbatim,
our decision and compatibility implications. `catalogue.json` covers all seven commits, all 72 changed paths,
before/after SHA256s and patch ownership of every installed-source mirror. Run `catalogue.py --check` to refuse
an unclassified change, missing README row, dirty donor or donor revision drift. Added docker/files mirrors
include the recipe's already installed baseline; additions are not treated as 32,000 lines of new algorithm.
No missing numbers are assumed to be missing patches: 0052, 0068, 0069 and 0070 are absent upstream.

## Configuration and other changes outside patches/

| Change | Upstream README evidence/status | Decision for our stack / conflict |
|---|---|---|
| README.md, all seven commits | v12 results, v22 experiments, 512K then 1M results, SSD off/on incident and mitigations, new attribution | Catalogue evidence only. Results used stock target, local drafter and local disk Engram; no legal-v2a/remote improvement claimed. Preserve licensing. |
| docker/Dockerfile | "0062, 0063, 0065 and 0076c are in patches/ but not in the image" (Dockerfile); README says internal versions not published | Keep v5 and selective overlays. Do not rebuild/replace engine, change CUDA or take all copied sources. 0079/0080 are carried upstream but off. |
| Installed docker/files mirrors | Each file is mapped to its patch(es) in catalogue.json | Inspect patch hunks as source; use our exact v5/K1/O1 bases, never replace whole mirrors. Multiple owners on one file require composition. |
| 0040 schedule, serve SPEC | "tok/s +1.27 % (geometric mean), tokens per round +2.37 %, cost per round +1.23 %"; in use `[[1,4,5]]` | Switchable config K5 candidate; target mrow32 supports 24 rows, x86host block5 supports 5 proposals. Keep depth/remote round tables unchanged until measured. Must test 3/4 streams and 0064 interaction. |
| serve EXL3_RDMA_DEVICES both PCIe functions | With 0051: "cost per round −1.20 %"; shares not separated | Switchable config uses OUR f1 fabric `rocep1s0f1,roceP2p1s0f1`, never donor f0. Topology/discovery is coordinator gate. It is Spark-to-Spark target ring, not two x86host links and not a claim of doubled bandwidth. |
| serve EXL3_CPU_PIN=5-9,15-19 | With 0064: "sustained 118 → 129 tok/s (+9 %)"; shares not separated | Separate config candidate. Pin can constrain remote Engram proxy, async helper and communicator threads. Leave defaults unchanged in other arms; audit actual threads and first-consumer waits. No CPU frequency/governor change. |
| serve IMAGE v24 | Upstream 1M in use | Reject wholesale image upgrade; our source pins and custom remote integrations remain v5-based. |
| serve MAX_LEN=1048576, KV_BYTES=3758096384 | 949K needles 3/3; memory made possible by 0076; README long-context table | K1 owns context/memory controls. Keep 327680 and 6GiB/rank for UP1 comparisons, four sequences. Upstream 3.5GiB includes its CPU offload tier, not ours. |
| serve KV connector JSON, CRC, expandable exception, multinode, load fraction, I/O affinity | 0056/0058/0077/0081 in use; stale rank1 produced garbage/NaNs/illegal access before fixes | RP1 owns safety-hardened connector, namespace and load cap. No connector in UP1 arms. Never copy donor .45 cap or shared-memory cleanup instructions. |
| serve cadence100000/background flags | 0071–74 in use, prio10 background upstream | SO1 owns priority5 adaptation and fairness. UP1 preserves cadence24 and current priorities; SO1 composition is on hold after S19; no combined arm is staged. |
| serve KV pairing/indexer factor/drop MTP/transient+alias prompt scratch | 0057/0060/0076 in use | K1 owns; preserve K1 source bytes in composed files and K1 flags. No double mounts. |
| serve MARKOV_FUSED | With 0067 "cost per round −0.66 %" | Drop local drafter entirely; worker already has fused SM86 Markov. Package only target-side 0067. |
| serve REPLAY_SKIP / ENGRAM_LOOKAHEAD4096 | +4.9–6.5% long prefill / +8–9% prefill respectively | Use X1 tap-aware 0075; 0078 is dropped this round (S17). Retain X1 reference code unmounted; no new Engram JIT. Results here unmeasured. |
| serve comments, default fabric/model/alias paths | Example hardware and public target | Never copy donor paths or alter legal-v2a model, alias, image, sampling, reasoning, priority or request cap. JIT recipe stays coordinator-only. |

## Adoption decisions

Build candidates: 0064 target smoe split; 0067 target SWA fusion/replay upload; 0075 via X1 remote tap handoff;
0078 is dropped this round (S17 FIX-2). Config-only independent candidates: K5 at all four streams, target dual-function
RDMA ring, X925 affinity. Keep all UP1 feature switches off in the mounted control. K1 remains an explicit base.

Reject 0062/0063/0065/0076c/0079/0080 on missing gain or measured regression; 0054/0055 and ALL 0066 local-drafter
paths are excluded. NVFP4 0061 is rejected; 0059 carried by K1 remains off. Do not expose rejected switches in UP1
configs. 0078 reader-pool affinity is rejected, even though lookahead is in use upstream. "In the image" does not
mean "in use". 0067's joint result does not establish its isolated gain. 0064's 33–63 row rounding change needs a
numerical gate rather than an identical-text assertion. No tuning table or claimed percentage is imported as measured.

## Shared-file boundaries

* K1 model_runner (0076) retains exact bytes after dropping 0078. Build one composed tree from hash-pinned files. Attention.py / flashinfer/KV records remain K1's unchanged bytes.
* X1 0075 modifies scheduler + input_batch + remote gather + model_state. Without tap provenance the donor patch is
  unsafe. Both engine and remote gather must be mounted on BOTH ranks. Capacity mode still retains target taps.
* 0067 modifies sparse_swa + model_state + a new Triton module. Capacity removes local draft groups: discover actual
  target SWA groups; never hard-code upstream's 15. Mixed/noncausal/capture/microbatch mismatch uses stock metadata.
* X1 0078 is unmounted reference code only; no UP1 Engram Python/C++/header or JIT changes remain.
* RP1 has no shared mounted source with this adoption set. SO1 DOES share scheduler.py with 0075. SO1 composition is
  on hold after S19; no SO1 lock, builder or combined arm is staged.
