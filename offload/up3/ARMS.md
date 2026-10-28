# UP3: current-production upstream arms

CPU source qualification only. No GB10 result or production adoption.
Production is 596ba1cf4e3b11be29b41a57e7f69a5d8efa4799,
`prod-k1-327-ops1-20261026`: K1 327680, KV 2502600, 4 sequences,
TP2 K2 fb68c52, process proxy, OPS1 warm-up. Preserve exact production
Engram settings (including REQUIRED=0), respawn policy, tensors and JIT.

| Arm | Only env/argv change | Only replacement engine files |
|---|---|---|
| C | experiment port 8001 | none; production mounts/bytes |
| split | VLLM_EXL3_MOE_SPLIT_MAX_TOKENS=63 | vllm_exl3_moe/{forward,smoe}.py |
| cpu-pin | EXL3_CPU_PIN=5-9,15-19 | vllm_exl3_moe/{rdma,up3_rdma_pin}.py (startup identity only; Round8) |
| vor | VLLM_DSV41_VOR_FUSED=1 | sparse_swa.py, nvidia/model_state.py, common/ops/swa_fused.py |
| k5 | speculative schedule [[1,4,5]], five drafts | none; noncausal block stays intact |
| dual-ring | EXL3_RDMA_DEVICES=rocep1s0f1,roceP2p1s0f1 | none |

`inputs.json` pins production git objects, every K1/remote file, image bases
and vendored donor patch bytes. `manifests/*.json` pins each rebuilt arm.
0064 and original 0067 both apply to those bases with zero fuzz AND zero
line offsets. No replay/X1 changes, scheduler, remote package replacement,
K1 package change, whole donor mirror or image upgrade. Current development's
remote speculator differs from production: retain production mounts and
verify against production objects, never substitute that development tree.
OFF has no overlay files or added default-off env keys; it reuses production
bytes. This is stronger than claiming disabled branches are equivalent.
Round8's CPU-pin-only startup identity overlay is documented in `RDMA-PIN.md`.
It changes no arithmetic; only this arm writes private container receipts.

Run `python3 -m offload.up3.build` on CPU under the prescribed private TMPDIR.
The command verifies, never deploys. `--output FRESH_DIR` rebuilds into scratch.

Round4 admission policy supersedes the historical config-arm native gates:
**k5, dual-ring and cpu-pin** use six experimental greedy checks per start,
with the unchanged NBW/DEP1 top20/0.501nat first-divergence rule, then OPS1 and
the full counter baseline. Dual ring also requires both exact rank/device/GID3
logs and host sysfs GID evidence. CPU pin requires a complete stable census
and the Round8 poller startup receipt; unavailable required evidence refuses.
Other native/Engram birth attribution remains explicitly UNMEASURED.
No resident-worker adapter is invoked for config arms. See `UP3-WINDOW.md`.
**split and vor are parked at execute admission BEFORE HOLD** until reviewed
resident-worker integration can run their exact gates on both ranks.

Historical S17/native adapter matrix (code-arm requirements remain; config
arms follow the Round4 decision above):

* 0064: every real row at m=33/42 equals unchanged exl3_smoe; m=43/63
  equals per-row decode (never raise the 42-row limit). Same inputs, weights,
  routing, permutation, padding and activation limit; byte equality before
  TP reduction. Rows32/64 prove unchanged dispatch. Then qualify changed
  band-vs-tier rounding separately using teacher-forced equal batch shapes.
* 0067: every valid/lens/index byte, padded/invalid rows, replay bounds,
  contexts through327680, actual target SWA groups, mixed/capture/noncausal/
  microbatch fallbacks. Prove fused launch and no hidden exception disable.
* K5: five proposals available, B4 24 target rows supported, unchanged
  noncausal SWA block; actual proposal/verification dimensions in receipt.
* Dual ring: IPv4 RoCE v2 GID3 on BOTH f1 devices/BOTH nodes; exact per-rank
  prefix `EXL3 RDMA all-reduce: rank R of 2 on rocep1s0f1,roceP2p1s0f1`;
  no NCCL-only fallback; collective byte equality and traffic on both devices.
* CPU pin: all real thread/TID affinities, preserve ring poller CPU14;
  inspect inherited async/Engram affinity and tails before crediting pin.

UPSTREAM CLAIM, not a prediction: 0064+pin +9% sustained jointly; K5
+1.27% geometric mean with B1 unchanged; dual ring with0051 -1.20% round;
0067 with0066 -0.66% round. RS1 §6 ESTIMATED effects here: B1 ~0%,
K5 loaded1–3%, dual0.5–1.5%, vor0–0.7%. Most may not beat UP3's 2% floor.
No combined arm is provided; only individual passing arms may enter a later window.
