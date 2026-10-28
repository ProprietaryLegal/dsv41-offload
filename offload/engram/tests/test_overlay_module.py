#!/usr/bin/env python3
"""The engram.py overlay inside the engine's own code, on a GPU host without vLLM: A/B/C module-level outputs.

vLLM's imports in common/engram.py are stubbed (TP group, config, platform, set_weight_attrs, the UVA view op, ...)
so that ParallelEngramEmbedding and its Triton lookup kernel run unchanged:
  A  the image's engram.py, disk offload (EngramDiskTable, spin mode)              = production today
  B  the overlay, VLLM_ENGRAM_REMOTE set (EngramRemoteTable, spin mode, x86host)     = remote Engram
  C  the overlay, VLLM_ENGRAM_REMOTE unset                                         = must behave as A
For one table rank's head shard (real head sizes from layout.py, real shard and resident server table), the bf16
outputs of lookup() for random hash ids (incl. DEAD_ID rows) must be bit-identical across A, B and C.

    CUDA_VISIBLE_DEVICES=GPU-... python3 test_overlay_module.py --image-common DIR --overlay .../spark/engram.py \\
      --root REPO --config config.json --shard .../model-00047-of-00048.safetensors --layer 1 --tp-rank 0 \\
      --endpoint rdma://10.202.28.1:18610@10.202.28.1
"""

import argparse
import ctypes
import importlib.util
import json
import os
import sys
import types

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--image-common", required=True)
ap.add_argument("--overlay", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--shard", required=True)
ap.add_argument("--layer", type=int, default=1)
ap.add_argument("--tp-rank", type=int, default=0)
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--endpoint", required=True)
ap.add_argument("--tokens", type=int, nargs="+", default=[24, 341, 4096])
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--out", default="")
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from layout import layout  # noqa: E402

# ---- stubs for the vLLM imports of common/engram.py -----------------------------------------------------------
cudart = ctypes.CDLL("libcudart.so.13")


def uva_view(t: torch.Tensor) -> torch.Tensor:
    """vllm.utils.torch_utils.get_accelerator_view_from_cpu_tensor: a CUDA tensor aliasing pinned host memory."""
    assert t.is_pinned(), "UVA view of unpinned memory"
    dptr = ctypes.c_void_p()
    rc = cudart.cudaHostGetDevicePointer(ctypes.byref(dptr), ctypes.c_void_p(t.data_ptr()), 0)
    assert rc == 0, f"cudaHostGetDevicePointer: {rc}"

    class CAI:
        __cuda_array_interface__ = {"data": (dptr.value, False), "shape": (t.numel() * t.element_size(),),
                                    "typestr": "|u1", "version": 3, "strides": None}

    return torch.as_tensor(CAI(), device="cuda").view(t.dtype).view(t.shape)


def mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


TP_RANK, TP = a.tp_rank, a.tp


class _Cfg:  # get_current_vllm_config(): only model_config / scheduler_config / engram_config are read
    pass


import logging  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for n in ("vllm", "vllm.models", "vllm.models.deepseek_v41", "vllm.compilation", "vllm.distributed",
          "vllm.distributed.device_communicators", "vllm.model_executor", "vllm.model_executor.layers",
          "vllm.utils"):
    mod(n, __path__=[])
mod("vllm.models.deepseek_v41.common", __path__=[a.image_common],
    __file__=os.path.join(a.image_common, "__init__.py"))
mod("vllm.compilation.breakable_cudagraph", eager_break_during_capture=lambda f: f)
mod("vllm.config", VllmConfig=object, get_current_vllm_config=lambda: _Cfg)
mod("vllm.distributed", get_dp_group=lambda: None, get_engram_dp_group=lambda: None, get_engram_dp_size=lambda: 1,
    get_tensor_model_parallel_rank=lambda: TP_RANK, get_tensor_model_parallel_world_size=lambda: TP,
    tensor_model_parallel_all_gather=None, __path__=[])
mod("vllm.distributed.device_communicators.shm_broadcast", SHM_PATH="/dev/shm", check_shm_free_space=None)
mod("vllm.distributed.parallel_state", GroupCoordinator=object)
mod("vllm.forward_context", get_forward_context=None, is_forward_context_available=lambda: False)
mod("vllm.logger", init_logger=logging.getLogger)
mod("vllm.model_executor.layers.linear", ColumnParallelLinear=object)
mod("vllm.model_executor.layers.quantization", QuantizationConfig=object)
mod("vllm.model_executor.utils",
    set_weight_attrs=lambda p, attrs: [setattr(p, k, v) for k, v in (attrs or {}).items()])


class _Platform:
    @staticmethod
    def is_cuda():
        return True

    @staticmethod
    def is_rocm():
        return False

    @staticmethod
    def is_device_capability_family(*_):
        return False


mod("vllm.platforms", current_platform=_Platform())
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

mod("vllm.triton_utils", tl=tl, triton=triton)
mod("vllm.utils.platform_utils", is_uva_available=lambda: True)
mod("vllm.utils.torch_utils", get_accelerator_view_from_cpu_tensor=uva_view)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ---- the three variants -----------------------------------------------------------------------------------------
tc = json.load(open(a.config))
tc = tc.get("text_config", tc)
L = layout(tc, TP)[str(a.layer)]
head_sizes = tuple(L["head_sizes"])
num_emb = L["num_embeddings"]


def tensor_offsets(path):
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return (path, 8 + n + hdr[f"layers.{a.layer}.engram.embed.weight"]["data_offsets"][0],
            path, 8 + n + hdr[f"layers.{a.layer}.engram.embed.scale"]["data_offsets"][0])


disk_source = tensor_offsets(a.shard)
os.environ["VLLM_ENGRAM_DISK_SPIN_MS"] = "1000"  # the serve's mode
os.environ["VLLM_ENGRAM_REMOTE_ROOT"] = a.root
os.environ["VLLM_ENGRAM_REMOTE_MANIFEST"] = os.path.join(a.root, "manifest.json")
torch.cuda.init()
max_rows = a.max_tokens * (-(-len(head_sizes) // TP))


def make(module, remote: bool):
    if remote:
        os.environ["VLLM_ENGRAM_REMOTE"] = ",".join([a.endpoint] * TP)
    else:
        os.environ.pop("VLLM_ENGRAM_REMOTE", None)
    with torch.device("cuda"):  # vLLM initializes the model under the CUDA default device (_disk_rows/_dst land there)
        emb = module.ParallelEngramEmbedding(num_emb, 256, head_sizes, cpu_offload=True, disk_source=disk_source,
                                             disk_threads=64, disk_cache_rows=0)
        emb.init_disk_staging(max_rows)
    return emb


class RefLookupKernel:
    """torch version of _engram_lookup_kernel's SORTED path (the one _disk_lookup launches): out[dst] =
    bf16(float32(fp8 weight[local]) * 2^(ue8m0 scale - 127)), unowned rows 0. Triton cannot compile the image's kernel
    on SM86 (no fp8e4nv on Ampere), so A, B and C all use this; on GB10 the image's kernel runs unchanged."""

    def __getitem__(self, grid):
        def run(weight, scales, ids, sorted_dst, out, vocab_start, vocab_end, num_rows, ids_stride_t, ids_stride_h,
                HEAD_START, LOCAL_HEADS, TOTAL_HEADS, DIM, QUANT_BLOCK, BLOCK_R, GRID, SORTED):
            assert SORTED, "the disk / remote path launches the kernel in sorted mode"
            n = int(num_rows)
            local = ids[:n].to(torch.int64)
            dst = sorted_dst[:n].to(torch.int64)
            owned = (local >= 0) & (local < int(vocab_end) - int(vocab_start))
            local = torch.where(owned, local, torch.zeros_like(local))
            vals = weight[local].to(torch.float32)
            sc = (scales[local].to(torch.int32) << 23).view(torch.float32).repeat_interleave(QUANT_BLOCK, dim=1)
            res = torch.where(owned[:, None], vals * sc, torch.zeros_like(vals))
            out.view(-1, DIM)[dst] = res.to(torch.bfloat16)
        return run


image = load_module(os.path.join(a.image_common, "engram.py"), "vllm.models.deepseek_v41.common.engram_image")
image._engram_lookup_kernel = RefLookupKernel()
A = make(image, False)
overlay = load_module(a.overlay, "vllm.models.deepseek_v41.common.engram")
overlay._engram_lookup_kernel = RefLookupKernel()
B = make(overlay, True)
# C: the overlay's disk path must resolve to the image's engram_disk.cpp; reuse the extension A already loaded (a
# second import of the same pybind11 module in one process would re-register its types)
assert os.path.samefile(overlay._IMAGE_DIR, a.image_common), overlay._IMAGE_DIR
overlay._ENGRAM_DISK_EXT = image._ENGRAM_DISK_EXT
C = make(overlay, False)
print("A:", A._disk.mode(), flush=True)
print("B:", B._disk.mode(), "|", B._disk.describe(), flush=True)
print("C:", C._disk.mode(), flush=True)
assert type(A._disk).__name__ == "EngramDiskTable" and type(C._disk).__name__ == "EngramDiskTable"
assert type(B._disk).__name__ == "EngramRemoteTable"

# hash ids as the engine produces them: column c in [offset_c, offset_c + prime_c), DEAD_ID (-1) rows (dummy hashes)
offsets = [sum(head_sizes[:c]) for c in range(len(head_sizes))]
g = torch.Generator().manual_seed(77 + TP_RANK)
results, fails = [], 0
for T in a.tokens:
    for rep in range(3):
        idx = torch.empty(T, len(head_sizes), dtype=torch.int64)
        for c in range(len(head_sizes)):
            idx[:, c] = offsets[c] + torch.randint(0, head_sizes[c], (T,), generator=g)
        dead = torch.rand(T, generator=g) < 0.03
        idx[dead] = -1
        if T > 8:
            idx[T // 2] = idx[0]  # a repeated position
        idx = idx.to(torch.int32).cuda()
        outs = []
        for emb in (A, B, C):
            out = torch.full((T, emb.part_n_hash_cols, 256), float("nan"), dtype=torch.bfloat16, device="cuda")
            emb.lookup(idx, out)
            torch.cuda.synchronize()
            outs.append(out.view(torch.int16).cpu())
        ab, ac = torch.equal(outs[0], outs[1]), torch.equal(outs[0], outs[2])
        nonzero = int((outs[0] != 0).any(dim=-1).sum())
        ok = ab and ac and nonzero > 0
        fails += 0 if ok else 1
        results.append({"tokens": T, "rep": rep, "rows": T * A.part_n_hash_cols, "A_eq_B": ab, "A_eq_C": ac,
                        "nonzero_rows": nonzero})
        print(("PASS " if ok else "FAIL ") + json.dumps(results[-1]), flush=True)
st = {k: v for k, v in B._disk.remote_stats().items() if not k.startswith("hist")}
print("B remote_stats:", json.dumps(st), flush=True)
if st["fallbacks"]:
    fails += 1
    print("FAIL: B used the local fallback", flush=True)
if a.out:
    json.dump({"layer": a.layer, "tp_rank": TP_RANK, "results": results, "remote_stats": st,
               "A_mode": A._disk.mode(), "B_mode": B._disk.mode()}, open(a.out, "w"), indent=1)
print("RESULT:", "ALL PASS" if fails == 0 else f"{fails} FAILED", flush=True)
os._exit(1 if fails else 0)  # the spin proxies are detached threads
