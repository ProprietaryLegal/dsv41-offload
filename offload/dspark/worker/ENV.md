# DSpark worker environment (x86host, RTX 3090 `GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93`)

| item | value |
|---|---|
| host | x86host (`x86host`), x86_64, EPYC 7302P, driver 580.126.20 (CUDA 13.0 capable) |
| GPU | RTX 3090, SM 8.6, 24 GiB, selected **only** by UUID: `CUDA_VISIBLE_DEVICES=GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93` |
| venv | `~/dsv41-offload/o2/venv` (uv, CPython 3.12.3) |
| torch | `2.10.0+cu126` from `https://download.pytorch.org/whl/cu126`, triton 3.6.0 (bundled) |
| CUDA toolkit for extensions | `/usr/local/cuda-12.6` (nvcc 12.6.85) |
| exllamav3 | **1.6.0**, `turboderp-org/exllamav3@151539c77abc7ab7425d30da7a4e8e3c5c154e7b` (the commit the Spark plugin names), cloned to `~/dsv41-offload/o2/src/exllamav3`, built for `sm_86` only; `exllamav3_ext.cpython-312-x86_64-linux-gnu.so` sha256 `a955d042af50c0b38d4476032e1fdfda37c8286d422af73b7b7521dae2e25a1d` |
| exllamav3_ext_facc | same commit + `exl3/exllamav3_facc.patch` (fp32 MMA accumulation, own module name and pybind11 internals), `exllamav3_ext_facc.cpython-312-x86_64-linux-gnu.so` sha256 `df5bdc2406591d7b5197ee84f445c1ff3e144ae0f090556792c9dbe0707783a6`, unzipped into the venv's site-packages from the wheel (not pip-installed) |
| O1 transport | pinned copy in `~/dsv41-offload/o2/lib/` by `run/pin_ofl.sh` (`libofl.so` sha256 `9f11b6e8...`, `ofl.py` `7c9b99ed...`; `PINNED.txt`) |
| pins | `requirements.lock` (pip freeze of the venv) |

The container on the Sparks runs torch `2.13.0+cu130` and exllamav3 1.6.0 built for sm_121. x86host's toolkit is CUDA 12.6,
so the worker uses a CUDA 12.x torch build; the exllamav3 kernels are the same source at the same commit.

## Rebuild

```bash
cd ~/dsv41-offload/o2
~/.local/bin/uv venv --python 3.12 venv
VIRTUAL_ENV=$PWD/venv ~/.local/bin/uv pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.10.0+cu126"
VIRTUAL_ENV=$PWD/venv ~/.local/bin/uv pip install -r <(grep -v -e '^torch==' -e '^exllamav3' -e '^nvidia-' -e '^triton==' requirements.lock)
git clone https://github.com/turboderp-org/exllamav3.git src/exllamav3 && git -C src/exllamav3 checkout 151539c77abc7ab7425d30da7a4e8e3c5c154e7b
cd src/exllamav3 && PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=20 \
  ../../venv/bin/python -m pip install --no-build-isolation --no-deps .      # ~4 min, 190 objects
# second module exllamav3_ext_facc: the same commit with fp32 MMA accumulation in exl3_gemm (main_proj; see below)
cd ~/dsv41-offload/o2 && git -C src/exllamav3 worktree add --detach ../exllamav3-facc 151539c77abc7ab7425d30da7a4e8e3c5c154e7b
git -C src/exllamav3-facc apply $REPO/offload/dspark/worker/exl3/exllamav3_facc.patch
cd src/exllamav3-facc && PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=20 \
  ../../venv/bin/python -m pip wheel --no-build-isolation --no-deps -w ../../wheels-facc .               # ~6 min
# install ONLY the module (the wheel's exllamav3 package would replace the stock one): unzip
# exllamav3_ext_facc.cpython-312-x86_64-linux-gnu.so from the wheel into the venv's site-packages
```

### `exllamav3_ext_facc` (fp32 accumulation for main_proj)
exllamav3 1.6.0 compiles `exl3_gemm` with fp16 MMA accumulators on sm_86 only (`EXL3_GEMM_H_ACC`, quant/exl3_gemm_inner.cuh:
"~14 % faster at bsz 1 on RTX 3090 ... max observed error ~1 % of output RMS"); every other arch, including the Sparks'
sm_121, accumulates in fp32. D3b found the effect where it matters: main_proj (15,360 inputs, three nearly equal target
residual streams that cancel) put the worker's main_x 0.37 % from an fp32 computation of the same taps, the engine's
0.11 %. `exl3/exllamav3_facc.patch` (vs 151539c7) turns the fp16 accumulators off, renames the module to
`exllamav3_ext_facc` and gives it its own pybind11 internals (`PYBIND11_STDLIB="_o2facc"`; the stock module registers
the same class names), so both load in one process. model.py runs main_proj on it (`Exl3Linear(ext=facc_ext())`, a
direct `exl3_gemm` call) and everything else on the stock module. main_proj alone: 0.37 % -> 0.036 % from fp64 (the
fp16 input rounding floor is 0.025 %); main_x 0.125 % from fp32 (engine 0.106 %). Building the whole step on fp32
accumulation instead cost +130-140 us per draft step for no measurable draft-parity gain (D3b, captured main_x:
93.2 % vs 92.8 % blocks exact).

## Forced runtime environment (`worker/env.py`, applied at import)

| variable | value | why |
|---|---|---|
| `EXL3_INT8_GEMV` | `0` | the Spark's setting. exllamav3 1.6.0 otherwise runs one-row mul1 GEMVs through a plain-int8 activation path: 0.66-0.77 % relative error on these tensors at 1 row vs 0.09-0.16 % with it off (`offload/results/o2/d1_exl3_identity.json`, first run vs final run) |
| `EXL3_MOE_COOP_WIDE` | `1` | the Spark's setting; exllamav3's default below Blackwell anyway (wide 128-column coop tile) |

## D1 result: EXL3 on SM86 vs a CPU reference (`tools/exl3_identity.py`)

`offload/results/o2/d1_exl3_identity.json`, run 2026-10-22 on the 3090 above, all 4,754 drafter tensors first
identity-checked (sha256 of every byte range against S2's `TENSORS.json`, 1.0 s from page cache).

* **Trellis decode is bit-identical** (fp16 bits) between `exllamav3_ext.reconstruct` on the 3090 and the plain-torch CPU
  decoder `exl3/exl3_ref.py` for all 13 tensors checked: `mtp.0.attn.{wkv,wq_a,wq_b,wo_b}`, `mtp.0.main_proj`,
  `mtp.1.ffn.shared_experts.w{1,2}`, `mtp.2.ffn.shared_experts.w3`, `mtp.2.attn.wq_b` (K5), drafter experts
  `mtp.0.ffn.experts.0.w1`, `mtp.1.ffn.experts.77.w2`, `mtp.2.ffn.experts.127.w3` (K2), and the head (K6, first 4,096
  columns; every column of the others).
* **Linear outputs** (fp16 input, fp32 output) vs the fp64 CPU linear `H(H(x*suh) @ W) * svh` on the same input:
  relative Frobenius error 0.029 % on exllamav3's reconstruct + hgemm path (> 144 rows: the fp16 input rounding alone),
  0.05-0.30 % on its small-batch `exl3_gemm` kernel (1-20 rows; head 0.26 %). `had_r_128` alone: 0.029 %.
* **Routed experts**: `exl3_moe_coop` (one K2/K2 group, 128 experts) on 5 tokens x top-3 vs a CPU fp64 SwiGLU-limit-10
  reference: 0.087-0.089 % relative error (layers mtp.0 and mtp.2).

The Spark's decode-size dense path is the plugin's own `exl3_mrow` kernel (TMA, sm_90+; not available on Ampere), so the
3090 runs exllamav3's `exl3_gemm` for the same tensors: same weights bit for bit, a different accumulation (the
0.1-0.3 % class above). Parity with the Spark drafter is therefore judged on draft tokens and near-ties (D3), not bits.
