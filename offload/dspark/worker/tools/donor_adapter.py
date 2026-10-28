"""Standalone, explicit-dispatch donor adapter; never imports SGLang or vLLM.

Experimental tooling only. No worker switch is added before measured full-step
parity. The raw trellis donor returns fp16/bf16, while the current worker asks its
EXL3 kernels for fp32 output: decoded-weight identity does not remove that difference.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import ModuleType
from typing import TYPE_CHECKING

from .profile_support import require_gpu_window

if TYPE_CHECKING:
    import torch

TRELLIS_SHA = "a1f72be2ff6319b9b2ea9bf6768c99664b1ca892"
CUDA_EXL3_SHA = "6a1ffc34866e23f484574ce1922a8bca93eb33b2"


def verify_checkout(root: Path, expected: str) -> None:
    actual = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    if actual != expected:
        raise RuntimeError(f"donor revision mismatch: {actual} != {expected}")
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", expected, "--"], check=True,
                   capture_output=True, text=True)


@dataclass(frozen=True)
class Dispatch:
    backend: str
    bits: int
    k: int
    n: int
    wide_override: bool = False


def trellis_dispatch(bits: int, k: int, n: int, *, moe: bool = False, allow_wide: bool = False) -> Dispatch:
    supported = (3, 4) if moe else (3, 4, 5, 6)
    if bits not in supported:
        raise ValueError(f"trellis {'MoE' if moe else 'dense'} does not implement K{bits}; supported={supported}")
    if k <= 0 or n <= 0 or k % 128 or n % 128:
        raise ValueError(f"trellis requires positive 128-aligned dimensions, got {k}x{n}")
    if n > 65536 and not allow_wide:
        raise ValueError(f"N={n} exceeds donor default 65536; explicit allow_wide is required (no fallback)")
    return Dispatch("trellis_marlin_moe" if moe else "trellis_marlin_dense", bits, k, n, n > 65536)


def load_verified_extension(build: Path, module_name: str, revision: str) -> ModuleType:
    """Only load a W1 build with a matching source receipt and binary digest."""
    require_gpu_window()
    import torch  # noqa: F401 -- load libtorch dependencies before importing a raw extension
    receipt = json.loads((build / "w1-build.json").read_text())
    if receipt["revision"] != revision or receipt["module"] != module_name:
        raise RuntimeError("extension receipt does not match pinned donor/module")
    path = (build / receipt["binary"]).resolve()
    if not path.is_relative_to(build.resolve()) or not path.is_file():
        raise RuntimeError("extension must be a file inside its W1 build directory")
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
        raise RuntimeError("extension binary digest mismatch")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load extension {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrellisDense:
    """Raw pybind entry points: no engine package, width cutoff or silent fallback.

    Own caller buffers are reusable in CUDA graphs. Initialize and pack before
    capture. `linear` always logs its explicit dispatch through the returned
    plan; the head must opt into testing above the donor's default width cutoff.
    """

    def __init__(self, donor: Path, build: Path):
        verify_checkout(donor, TRELLIS_SHA)
        self.ext = load_verified_extension(build, "trellis_exl3_kernels", TRELLIS_SHA)

    def prepare(self, trellis: torch.Tensor, *, allow_wide: bool = False) -> tuple[torch.Tensor, Dispatch]:
        require_gpu_window()
        if trellis.ndim != 3 or trellis.shape[2] % 16:
            raise ValueError("expected trellis [K/16,N/16,16*bits]")
        plan = trellis_dispatch(trellis.shape[2] // 16, trellis.shape[0] * 16, trellis.shape[1] * 16,
                                allow_wide=allow_wide)
        self.ext.init_device(trellis.device.index)
        return self.ext.repack_trellis(trellis), plan

    def unpack(self, packed: torch.Tensor) -> torch.Tensor:
        require_gpu_window()
        return self.ext.unpack_trellis(packed)

    def rotated(self, xh: torch.Tensor, packed: torch.Tensor, out: torch.Tensor) -> None:
        """Identity input rows recover decoded weights, independently of Hadamard arithmetic."""
        require_gpu_window()
        self.ext.exl3_gemm_marlin(xh, packed, out, 2, -1, -1)  # actual MUL1 id; no auto backend

    def linear(self, x: torch.Tensor, packed: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor,
               scratch: torch.Tensor, out: torch.Tensor) -> None:
        require_gpu_window()
        self.ext.exl3_linear_marlin_out(x, packed, suh, svh, 2, scratch, out)


class CudaExl3Dense:
    """Native registered op only; deterministic split-K off, no oracle fallback."""

    def __init__(self, donor: Path, build: Path):
        verify_checkout(donor, CUDA_EXL3_SHA)
        self.ext = load_verified_extension(build, "_C", CUDA_EXL3_SHA)

    def linear(self, x: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor,
               widths: list[int]) -> torch.Tensor:
        require_gpu_window()
        import torch
        if sum(widths) != trellis.shape[1] * 16 or suh.shape != (len(widths), x.shape[-1]):
            raise ValueError("per-shard input scales/output widths disagree")
        return torch.ops.cuda_exl3_C.exl3_linear(x, trellis, suh, svh, widths, 2, False)
