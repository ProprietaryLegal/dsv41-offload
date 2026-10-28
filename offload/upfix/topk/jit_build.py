"""Build the fixed persistent top-k op (torch.ops.dsv41_topk) for SM121 inside the serving image.

Run INSIDE the production image (same torch/CUDA as the server), before a window, with the
vendored sources mounted read-only and a fresh private build directory:

    python3 -m offload.upfix.topk.jit_build --execute --sources /opt/dsv41-upfix/topk/csrc \
        --build-dir /opt/dsv41-upfix/topk/build --receipt /opt/dsv41-upfix/topk/build/receipt.json

It compiles once (no GPU needed; nvcc only), records the library's sha256, the exact flags, torch
and CUDA versions, and source hashes. The topk arm mounts the library and sets
VLLM_DSV41_TOPK_FIX=1 / VLLM_DSV41_TOPK_FIX_SO. Without --execute it prints the plan only.
Flags mirror vLLM's release build: torch's common nvcc flags (cpp_extension defaults) plus -O3
-DNDEBUG; no --use_fast_math (vLLM's top-k build does not use it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SOURCES = ('dsv41_topk.cu',)
HEADERS = ('persistent_topk.cuh', 'sampled_topk.cuh', 'topk_histogram_4096.cuh',
           'persistent_topk_overflow.cuh', 'topk_histogram_4096_overflow.cuh')
NAME = 'dsv41_topk'
ARCH = '12.1'  # GB10 / SM121
CUDA_FLAGS = ('-O3', '-DNDEBUG', '--threads=4')
CXX_FLAGS = ('-O3', '-DNDEBUG')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(sources: Path, build_dir: Path) -> dict[str, object]:
    return {'name': NAME, 'arch': ARCH, 'sources': [str(sources / s) for s in SOURCES],
            'include': str(sources), 'build_dir': str(build_dir),
            'extra_cuda_cflags': list(CUDA_FLAGS), 'extra_cflags': list(CXX_FLAGS),
            'source_sha256': {name: sha256(sources / name) for name in (*SOURCES, *HEADERS)}}


def execute(sources: Path, build_dir: Path, receipt: Path) -> dict[str, object]:
    import torch
    from torch.utils import cpp_extension

    if build_dir.exists() and any(build_dir.iterdir()):
        raise ValueError(f'{build_dir} must be a fresh empty build directory')
    if receipt.exists():
        raise ValueError(f'{receipt} already exists')
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_CUDA_ARCH_LIST'] = ARCH
    details = plan(sources, build_dir)
    library = cpp_extension.load(name=NAME, sources=[str(sources / s) for s in SOURCES],
                                 extra_include_paths=[str(sources)], extra_cuda_cflags=list(CUDA_FLAGS),
                                 extra_cflags=list(CXX_FLAGS), build_directory=str(build_dir),
                                 is_python_module=False, verbose=True)
    so = Path(str(library)) if library else build_dir / f'{NAME}.so'
    if not so.is_file():
        found = sorted(build_dir.glob(f'{NAME}*.so'))
        if len(found) != 1:
            raise ValueError(f'expected one built {NAME} library in {build_dir}, found {found}')
        so = found[0]
    if not hasattr(torch.ops, NAME) or not hasattr(getattr(torch.ops, NAME), 'persistent_topk'):
        raise ValueError('built library did not register torch.ops.dsv41_topk.persistent_topk')
    details.update(library=str(so), library_sha256=sha256(so), torch=torch.__version__,
                   cuda=torch.version.cuda, python=sys.version.split()[0])
    receipt.write_text(json.dumps(details, indent=1, sort_keys=True) + '\n')
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--sources', type=Path, default=Path(__file__).resolve().parent / 'csrc')
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(plan(args.sources, args.build_dir), indent=1, sort_keys=True))
        return
    if args.receipt is None:
        parser.error('--receipt is required with --execute')
    print(json.dumps(execute(args.sources, args.build_dir, args.receipt), indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
