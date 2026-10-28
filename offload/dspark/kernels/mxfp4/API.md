# N1 integration handoff

Use `offload.dspark.kernels.mxfp4.PackedExperts` (no worker monkeypatch).

1. Select rank-local native group-32 checkpoint shards on CPU. For w1/w3,
   output rows `[rank*1152:(rank+1)*1152]`; for w2, input bytes
   `[rank*576:(rank+1)*576]`, scale columns `[rank*36:(rank+1)*36]`.
2. `PackedExperts.from_checkpoint(bytes_u8, scales_u8, torch.device('cuda',r))`
   validates exact normal BF16 values and performs a lossless transpose. Preserve
   native source SHA/transform receipts in the loader. BF16 subnormal/nonfinite/
   overflow operands refuse admission and need the separate wider oracle.
3. Preallocate contiguous BF16 x `[rows,K]`, int32 IDs `[rows,3]`, FP32
   projection outputs `[rows,3,N]`, and int32 status `[1]` on that device.
   `linear_into(x,ids,out,status,up=up_bank,up_out=up_out,mode='mma')` fuses
   gate/up; omit `up` for one projection; `mode='simt'` is a measured control.
4. This is W4A16 linear algebra. Apply official block-32 E4M3 activation QDQ
   **before** passing x to the packed kernel. Decode QDQ values to exactly
   representable BF16. The packed kernel does not substitute raw BF16 for QDQ.
5. Linear output is FP32 deliberately. Round to BF16 where the official linear
   returns BF16. The optional `activation_into(g,u,routing_weights,a,limit)`
   helper does BF16 projection rounding, FP32 clamp/SwiGLU, applies routing
   weights, then BF16 down-input rounding. It does **not** apply down-input
   QDQ. Apply that QDQ before calling down `linear_into(...,per_slot=True)`.
6. `combine_into(parts,unit_weights,out)` sums the three routed down outputs in
   fixed FP32 slot order. Unit weights are essential if routing was applied in
   step 5. Round w2 outputs to BF16 first if required by the selected official
   reference contract. TP2 reduction and shared-expert addition belong to N1.
7. Zero status once at proposal entry. Invalid device IDs set bit 0 and write
   NaNs without out-of-bounds reads. Preserve/check this sticky status at the
   proposal completion boundary; never clear it per layer or return bad proposals.
8. Prewarm each shape/mode before graph capture. Retain disjoint weights,
   inputs, outputs and workspaces until stream completion. API calls allocate
   no device memory and perform no CPU tensor-value readback. Each CUDA device
   has its own weights/workspaces; there is no static global GPU pointer cache.

The benchmark pipeline deliberately isolates kernel machinery and excludes model
activation QDQ, attention, routing selection, TP communication, shared experts,
head and network. It is not a complete native MoE or draft timing. Numerical
projection checks use actual native weights and independent CPU FP32/FP64 dot
products; full captured-tap fidelity and accepted-prefix measurements remain
with N1/V1 and coordinator, respectively.

## Faster register-fed path

`from offload.dspark.kernels.mxfp4.regfed import RegisterExperts`
uses the same canonical uint8 checkpoint input and source-admission checks.
It requires N%16=0,K%64=0 (all actual TP2 shapes qualify), and E8M0 code zero
is refused because the donor scale-multiplier shortcut cannot represent it.
Prepack streams one expert at a time into its final fragment layout; packed
resident size is unchanged. No decoded 16-bit bank is allocated.

Allocate an additional int32 schedule `[rows*top_k,8]` on the same device and
pass it as the fifth argument of `linear_into(x,ids,out,status,schedule,...)`.
At most 64 routed slots are admitted per launch (actual B1–4/block5/top3 <=60).
Each call rebuilds the bounded device schedule from current IDs; no cached
routing metadata can go stale across graph replays. Groups of eight rows
sharing an expert reuse its decoded register fragments. IDs may be unsorted;
per-slot results stay in the original caller order. Fused gate/up is the
measured preferred register variant; masked Triton remains the padding control.

For the official model's final expert sum, after per-route TP FP32 reduction,
use `combine_after_tp_into(tp_reduced_parts,ids,out)`: BF16-round each complete
w2 result and sum in ascending expert ID (stable slot order for duplicate IDs).
Do not use the unrounded slot-order benchmark helper as the native model's
final combine. This distinction is pinned by `test_native_combine_order`.

The JIT module is prebuilt under the existing lane-E environment with nvcc
12.6.85, `-gencode=arch=compute_86,code=sm_86`, `--fmad=false`, MAX_JOBS=2.
The module has no global GPU pointers or device-specific allocation caches;
launches use an explicit current-device guard and the caller's current stream.
Keep all arrays disjoint and live through completion. Build/warmup precedes
capture; the scheduling and GEMM calls allocate no memory.
