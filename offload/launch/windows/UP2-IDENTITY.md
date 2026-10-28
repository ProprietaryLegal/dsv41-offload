# UP2 low-cap identity reconciliation

Coordinator ledger (binding brief 2026-10-24): sparkhead `~/dsv41-deploy`
main `21bed31`, tag `prod-k1-327-20261023`. No Spark access by this lane.
MEASURED CPU source inspection: `git show 21bed31:{dsv41.env,serve.sh,start.sh}`.
MEASURED x86host **file reads only** over SSH: `current` resolves to
`/home/operator/dsv41-offload/prod/dspark-worker/fb68c52`; SOURCE_COMMIT is
`fb68c5291792ab90c0f468bd94010917967799f3`. The installed TP2 unit reads
that tree, uses the original checkpoint, max length1048576 and the production
NVLink pair. Status file reports TP2 and no communication errors at the read.
This is file evidence, not proof of process bytes or native UP2 qualification.

`UP2-low-k1.env` derives from the composed UP2 wrapper; it does **not** change
the worker/checkpoint identity. Original identity be2c1436… is still production's
identity, not an obsolete worker revision. Worker *source revision* fb68c52 and
HELLO *checkpoint identity* are different fields. Inherited noncanonical K1
values are refused. No config/worker overwrite is permitted on a mismatch.

| Setting, equal across every arm | Pinned value / verification |
|---|---|
| Production rollback | main21bed31, tag prod-k1-327-20261023; save actual argv/env/hashes before opening |
| Target/checkpoint | /srv/dsv41/legal-v2a; candidate manifest, quant plan, config/tokenizer hashes identical both ranks and saved production |
| Base/quant revision | 1bae0e1088993b9b82de08f59de5be5a34e42170 / 86f852533ec94d1a537fea41fd9c01d6b0f6eb50 |
| Image/engine/kernels | v5; exact running image SHA and engine/JIT/code hashes from saved production; tag alone insufficient |
| Target TP / fabric | 2; enp1s0f1np1 / rocep1s0f1; NCCL both f1 functions; EXL3 ring rocep1s0f1 |
| Context / KV bytes / pool | 327680 / 6442450944 per rank / 2502600 tokens |
| K1 | pair3, drop-MTP1, transient1, alias1, indexer-factor4; all13 K1 bytes match manifest |
| Admission / batching | 4 sequences, MAX_BATCH4096, threshold1024, priority scheduler, throttle min5 / every24 |
| Speculation | dspark5; K5 at B1–2, K4 at B3–4; adaptive margin1/min2/hi1.6/maxreq4; no K5 override |
| Remote worker | fb68c5291792ab90c0f468bd94010917967799f3; TP2, maximum1048576, checkpoint/adapters unchanged |
| Worker GPUs (identity only) | GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93 / GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc; never changed by runner |
| HELLO identities | be2c1436fa9f9e66b398e1b1da2e140e6fddc82a11939e2ac4240a0ca79543bd / Markov0b6e71e638abd5abce2db433965bb5ff523cbbe92ed7ae54324a50ccaeffa19a |
| Worker transport/cost | o1rdma://10.202.28.1:18620; bind10.202.28.2; timeout50; round28.79,31.86,34.44,37.76,41.56,43.68; draft4.4ms |
| Proxy | capacity, async1/process, spin50us/nap20us, backoff5000ms, respawns3/reset3600s; no local drafter, no proxy torch |
| Engram | original O1 shim/JIT; both RDMA links, timeout100/retry5000/log60; REQUIRED1 in **all experiment arms** |
| Engram production difference | production REQUIRED0 allows fallback; recipe requires REQUIRED1 to exclude fallback from replay ratios; verify remote active both ranks |
| Other experiments | split0, VOR0, dual-ring0, CPU-pin0, SO1/SO2 absent; no connector/PF1/capture overlays |
| Clocks / thermal / loading | existing2150MHz cap, LAZY; same worker clocks; record drift, never adjust worker clocks/units |
| Timing requests | exact150000/300000 complete chat; output1500, thinking true, high/75, T1/top_p1/seed42/priority10; same corpus/order/header |
| Cache / repetitions | cold cached_tokens0, 3 per size per arm; one fixed fresh-window salt prefix; point/rep prompt and salt SHA equal |
| Allowed differences | replay OFF/ON/OFF; audit1 on both diagnostic starts, audit0 on all measured starts; label/run/cache reset/start time |

UP2's additional composed engine bytes are identical across OFF/ON. Runtime
`verify_runtime --emit` checks all42 bytes on **both ranks**. Any source/image,
worker, target, K1 or policy mismatch stops; it never authorizes worker staging.
All live gates remain NOT RUN by this lane. Required-native missing apparatus
must be reported NOT RUN/HOLD. The newer lane brief sets the resource floor to
3GiB; the older recipe's5GiB is superseded only for this low-cap runner.
