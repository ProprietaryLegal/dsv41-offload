"""Load the image's real speculative-verification kernels on CPU (Triton interpreter).

AUD1 item 1. Production's verifier is vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
in the serving image (read-only copy: /opt/dsv41/data/src-image). This module imports
that file and the gumbel.py it uses unchanged, under TRITON_INTERPRET=1 with CPU tensors, so a
CPU test exercises the exact verification code production runs. Only two dependencies are
replaced: vllm.triton_utils (re-exports triton / tl / libdevice, as the image does) and the
watermark module (watermarking is off in production; its kernel is never reached because
WATERMARK is a False constexpr). Nothing here touches a GPU, a network or the serving hosts.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

DEFAULT_SRC = Path("/opt/dsv41/data/src-image")
_GUMBEL = "vllm/v1/worker/gpu/sample/gumbel.py"
_REJECTION = "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
_PACKAGES = (
    "vllm",
    "vllm.v1",
    "vllm.v1.worker",
    "vllm.v1.worker.gpu",
    "vllm.v1.worker.gpu.sample",
    "vllm.v1.worker.gpu.spec_decode",
)


class LoaderError(RuntimeError):
    pass


def _require_interpreter() -> None:
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise LoaderError("set TRITON_INTERPRET=1 before importing triton (CPU interpreter only)")
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise LoaderError("set CUDA_VISIBLE_DEVICES= (empty): this audit never uses a GPU")


def _load_file(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LoaderError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load(rejection_path: Path | None = None, src_root: Path = DEFAULT_SRC) -> Any:
    """Return the rejection_sampler_utils module (the image's, or a given variant of it).

    A variant path (e.g. a copy with vLLM #61161's one-line change, used as a positive control)
    is loaded under the same module name with the image's gumbel.py.
    """
    _require_interpreter()
    gumbel_path = src_root / _GUMBEL
    rejection = rejection_path if rejection_path is not None else src_root / _REJECTION
    for p in (gumbel_path, rejection):
        if not p.is_file():
            raise LoaderError(f"missing kernel source {p}")
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
    import triton.language.extra.libdevice as tldevice  # type: ignore[import-untyped]

    for name in _PACKAGES:
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        sys.modules[name] = pkg
    tu = types.ModuleType("vllm.triton_utils")
    tu.triton = triton  # type: ignore[attr-defined]
    tu.tl = tl  # type: ignore[attr-defined]
    tu.tldevice = tldevice  # type: ignore[attr-defined]
    tu.HAS_TRITON = True  # type: ignore[attr-defined]
    sys.modules["vllm.triton_utils"] = tu
    wm = types.ModuleType("vllm.v1.worker.gpu.sample.watermark")

    def philox_gumbel_block_argmax(*_args: object, **_kwargs: object) -> None:
        raise LoaderError("watermark path reached; production has watermarking off")

    wm.philox_gumbel_block_argmax = philox_gumbel_block_argmax  # type: ignore[attr-defined]
    sys.modules["vllm.v1.worker.gpu.sample.watermark"] = wm
    _load_file("vllm.v1.worker.gpu.sample.gumbel", gumbel_path)
    return _load_file("vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils", rejection)
