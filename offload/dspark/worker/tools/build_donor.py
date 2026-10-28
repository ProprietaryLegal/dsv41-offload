"""Build reviewed donor kernels into a W1-owned directory, without installing engines.

Requires the GPU window even for compilation. Reuses O2's locked torch/toolchain;
does not pip install, run upstream setup/launch scripts, or alter the donor tree.
Usage: python -m worker.tools.build_donor trellis|cuda-exl3 DONOR BUILD
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from .donor_adapter import CUDA_EXL3_SHA, TRELLIS_SHA, verify_checkout
from .profile_support import require_gpu_window


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("donor", choices=("trellis", "cuda-exl3"))
    p.add_argument("source", type=Path)
    p.add_argument("build", type=Path)
    p.add_argument("--resume", action="store_true", help="reuse this lane's partial compiler outputs")
    args = p.parse_args()
    require_gpu_window()
    args.source = args.source.resolve()
    args.build = args.build.resolve()
    revision = TRELLIS_SHA if args.donor == "trellis" else CUDA_EXL3_SHA
    verify_checkout(args.source, revision)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    os.environ["MAX_JOBS"] = "2"
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    from torch.utils.cpp_extension import load
    args.build.mkdir(parents=True, exist_ok=args.resume)
    flags = ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17"]
    if args.donor == "trellis":
        name = "trellis_exl3_kernels"
        csrc = args.source / "cuda/csrc"
        sources = [str(csrc / "exl3_marlin.cu")]
        variants = []
        # Our dense rows are 1..20. MUL1 only, K5 dense and K6 head.
        for mb in (0, 1, 2):
            for bits in (5, 6):
                variants.append(f"X({mb}, 2, {bits})")
                path = args.build / f"inst_mb{mb}_cb2_k{bits}.cu"
                path.write_text(f"#define TRELLIS_INST_MB {mb}\n#define TRELLIS_INST_CB 2\n"
                                f"#define TRELLIS_INST_KB {bits}\n#include \"exl3_marlin_inst.h\"\n")
                sources.append(str(path))
        (args.build / "trellis_families.h").write_text(
            "#pragma once\n#define TRELLIS_VARIANTS(X) " + " ".join(variants) + "\n")
        includes = [str(csrc), str(csrc / "third_party/marlin"), str(args.build)]
        help_text = subprocess.run([os.environ["CUDA_HOME"] + "/bin/nvcc", "--help"],
                                   capture_output=True, text=True, check=True).stdout
        if "static-global-template-stub" in help_text:
            flags.append("-static-global-template-stub=false")
        flags += ["-DTRELLIS_WRAP_LOAD=1", "-DTRELLIS_PROBES=0", "-DTRELLIS_SLOT_REDUCE=1",
                  "-DTRELLIS_MOE_MINBLOCKS_M8=2", "-DTRELLIS_MCG_SELFADD=1", "-DTRELLIS_K3_IMAD_SHIFTS=0"]
    else:
        name = "_C"
        csrc = args.source / "src/cuda_exl3/csrc"
        # MLA uses SM89 FP8 conversion instructions; it is unrelated to EXL3 GEMM.
        # Remove only its declaration/registration from a lane-owned binding copy.
        bindings = (csrc / "bindings.cpp").read_text()
        for pattern in (r"at::Tensor mla_decode\([\s\S]*?;", r'    m.def\(\n        "mla_decode[\s\S]*?;\n',
                        r'    m.impl\("mla_decode", &cuda_exl3::mla_decode\);\n'):
            bindings, count = re.subn(pattern, "", bindings)
            if count != 1:
                raise RuntimeError(f"pinned MLA binding exclusion matched {count}, expected 1: {pattern}")
        dense_binding = args.build / "bindings_dense.cpp"
        dense_binding.write_text(bindings)
        sources = [str(dense_binding)] + [str(csrc / f) for f in ("hadamard.cu", "gemm.cu")]
        # Separate diagnostic binding; donor sources stay unchanged and pinned.
        binding = args.build / "decode_binding.cpp"
        binding.write_text('#include <torch/extension.h>\n'
                           'at::Tensor w1_decode(const at::Tensor&, bool);\n'
                           'TORCH_LIBRARY(w1_donor, m) { m.def("decode(Tensor tr, bool vector) -> Tensor"); }\n'
                           'TORCH_LIBRARY_IMPL(w1_donor, CUDA, m) { m.impl("decode", &w1_decode); }\n')
        sources += [str(Path(__file__).with_name("cuda_exl3_decode.cu")), str(binding)]
        includes = [str(csrc)]
        flags += ["--expt-relaxed-constexpr", "--expt-extended-lambda", "-U__CUDA_NO_HALF_OPERATORS__",
                  "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
                  "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"]
    require_gpu_window()
    module = load(name, sources=sources, extra_include_paths=includes, extra_cflags=["-O3", "-std=c++17"],
                  extra_cuda_cflags=flags, build_directory=str(args.build), verbose=True)
    require_gpu_window()
    verify_checkout(args.source, revision)
    if module.__file__ is None:
        raise RuntimeError("build returned a module without a binary path")
    binary = Path(module.__file__).resolve()
    if not binary.is_relative_to(args.build.resolve()):
        raise RuntimeError("build returned an extension outside W1's directory")
    receipt = {"module": name, "revision": revision, "binary": binary.name,
               "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "arch": "8.6",
               "sources": {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},
               "nvcc_flags": flags}
    (args.build / "w1-build.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
