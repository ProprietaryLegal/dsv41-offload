"""Build only sm_86 against pinned CUTLASS in an explicit lane-owned build directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import ModuleType

UUID = "GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc"
SHA = "ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e"


def require_lock() -> None:
    import fcntl
    if os.environ.get("CUDA_VISIBLE_DEVICES") != UUID:
        raise RuntimeError(f"must pin CUDA_VISIBLE_DEVICES={UUID}")
    fd = int(os.environ["KX_GROUPED_LOCK_FD"])
    lock = Path.home()/"dsv41-offload/locks/gpu-7744b44b.lock"
    if os.fstat(fd).st_ino != lock.stat().st_ino or os.fstat(fd).st_dev != lock.stat().st_dev:
        raise RuntimeError("inherited fd does not refer to the development-card lock")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with lock.open("a") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError("exclusive GPU lock is not held")


def load_extension(donor: Path, build: Path) -> ModuleType:
    require_lock()
    rev = subprocess.check_output(["git", "-C", str(donor), "rev-parse", "HEAD"], text=True).strip()
    if rev != SHA or subprocess.check_output(["git", "-C", str(donor), "status", "--porcelain"], text=True):
        raise RuntimeError(f"donor must be clean at {SHA}, got {rev}")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    os.environ["MAX_JOBS"] = "1"
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    from torch.utils.cpp_extension import load
    build.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("grouped.cu")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    ext = load("kx_grouped", [str(source)], extra_include_paths=[str(donor/"include")],
               extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo"],
               extra_cflags=["-O3", "-std=c++17"], build_directory=str(build), verbose=True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_digest:
        raise RuntimeError("source changed during compilation")
    if ext.__file__ is None:
        raise RuntimeError("extension has no binary")
    binary = Path(ext.__file__)
    (build/"receipt.json").write_text(json.dumps({
        "cutlass_sha": SHA, "source_sha256": source_digest,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "cuda_arch": "sm_86", "fast_math": False, "accumulator": "FP32",
    }, indent=2)+"\n")
    return ext


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("donor", type=Path)
    p.add_argument("build", type=Path)
    a = p.parse_args()
    load_extension(a.donor, a.build)


if __name__ == "__main__":
    main()
