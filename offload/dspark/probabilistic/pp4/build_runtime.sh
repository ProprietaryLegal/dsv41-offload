#!/usr/bin/env bash
# CPU-only prerequisite build. Run in an owned named tmux on x86host; never a unit.
set -euo pipefail
source ~/.config/ndm/paths.sh
umask 077
if [[ $(hostname -s) != x86host ]]; then
  echo 'PP4 runtime builds only on x86host.' >&2
  exit 3
fi
if [[ -z ${TMUX:-} ]]; then
  echo 'Run this long build in an owned named tmux with private TMUX_TMPDIR.' >&2
  exit 3
fi
PP4_BUILD_HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PP4_BUILD_ROOT=/opt/dsv41/x86host-data/pp4/runtime
PP4_BUILD_PYTHON=/home/operator/miniconda3/bin/python3
if [[ ${1:-} != --bounded-child ]]; then
  exec taskset -c 12-15,28-31 nice -n 19 bash "$0" --bounded-child
fi
export TMPDIR="$PP4_BUILD_ROOT/tmp" TMUX_TMPDIR="$PP4_BUILD_ROOT/tmux" PIP_CACHE_DIR="$PP4_BUILD_ROOT/pip-cache"
export TORCH_EXTENSIONS_DIR="$PP4_BUILD_ROOT/torch_extensions" TRITON_CACHE_DIR="$PP4_BUILD_ROOT/triton"
export CUDA_CACHE_PATH="$PP4_BUILD_ROOT/cuda_cache" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST=8.6 CUDA_HOME="$PP4_BUILD_ROOT/cuda13/nvidia/cu13"
export DSPARK_P2P_CUDA_HOME="$CUDA_HOME"
mkdir -p "$TMPDIR" "$TMUX_TMPDIR" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
ulimit -v 50331648
"$PP4_BUILD_PYTHON" - <<'PY'
import importlib.metadata, os, resource, sys
import torch
if sys.version_info[:2] != (3, 13) or torch.__version__ != '2.12.1+cu130':
    raise RuntimeError('requires existing CPython3.13/Torch2.12.1+cu130; do not substitute')
if importlib.metadata.version('nvidia-cuda-runtime') != '13.0.96':
    raise RuntimeError('requires pinned existing nvidia-cuda-runtime13.0.96')
if os.sched_getaffinity(0) != {12,13,14,15,28,29,30,31} or os.getpriority(os.PRIO_PROCESS,0) != 19:
    raise RuntimeError('reserved affinity/nice19 missing')
if resource.getrlimit(resource.RLIMIT_AS)[0] > 48*1024**3:
    raise RuntimeError('48GiB virtual-memory cap missing')
PY
"$PP4_BUILD_PYTHON" -m pip download --no-deps --require-hashes \
  -r "$PP4_BUILD_HERE/requirements-runtime.lock" --dest "$PP4_BUILD_ROOT/wheels"
cp -- "$PP4_BUILD_HERE/runtime_reuse.py" "$PP4_BUILD_ROOT/runtime_reuse.py"
"$PP4_BUILD_PYTHON" - <<'PY'
from pathlib import Path
from zipfile import ZipFile
import shutil, subprocess
root=Path('/opt/dsv41/x86host-data/pp4/runtime')
for wheel in sorted((root/'wheels').glob('*.whl')):
    with ZipFile(wheel) as archive:
        for entry in archive.infolist():
            target=root/'cuda13'/entry.filename
            if entry.is_dir(): target.mkdir(parents=True,exist_ok=True); continue
            data=archive.read(entry)
            if not target.is_file() or target.read_bytes()!=data:
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
cuda=root/'cuda13/nvidia/cu13'
installed=Path('/home/operator/miniconda3/lib/python3.13/site-packages/nvidia/cu13')
def same_bytes_copy(source,destination):
    destination=Path(destination)
    if not destination.is_file() or Path(source).read_bytes()!=destination.read_bytes():
        return shutil.copy2(source,destination)
    return str(destination)
shutil.copytree(installed/'include',cuda/'include',dirs_exist_ok=True,copy_function=same_bytes_copy)
(cuda/'lib64').mkdir(exist_ok=True)
for source in (installed/'lib').iterdir():
    target=cuda/'lib64'/source.name
    if not target.exists(): target.symlink_to(source)
link=cuda/'lib64/libcudart.so'
if not link.exists(): link.symlink_to(installed/'lib/libcudart.so.13')
for directory in (cuda/'bin',cuda/'nvvm/bin'):
    for path in directory.iterdir():
        if path.is_file(): path.chmod(path.stat().st_mode|0o111)
for name in ('exllamav3','exllamav3-facc'):
    source=Path.home()/'dsv41-offload/o2/src'/name
    revision=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],text=True,capture_output=True,check=True).stdout.strip()
    if revision!='151539c77abc7ab7425d30da7a4e8e3c5c154e7b':
        raise RuntimeError('public EXL3 source revision changed')
    destination=root/name
    if not destination.exists():
        shutil.copytree(source,destination,ignore=shutil.ignore_patterns('.git','build','*.egg-info','__pycache__'))
setup=root/'exllamav3-facc/setup.py'
original=root/'exllamav3-facc/setup.original.py'
if not original.exists(): original.write_bytes(setup.read_bytes())
setup.write_text('import sys\nsys.path.insert(0, "/opt/dsv41/x86host-data/pp4/runtime")\nimport runtime_reuse\nruntime_reuse.install()\n'+original.read_text())
PY
for PP4_BUILD_EXTENSION in exllamav3 exllamav3-facc; do
  (
    cd "$PP4_BUILD_ROOT/$PP4_BUILD_EXTENSION"
    "$PP4_BUILD_PYTHON" setup.py build_ext --inplace
  ) > "$PP4_BUILD_ROOT/build-$PP4_BUILD_EXTENSION.log" 2>&1
done
# Compile/import only. This does not create a CUDA context or touch either pair GPU.
export PP4_BUILD_HERE
export PYTHONPATH="$PP4_BUILD_ROOT/exllamav3:$PP4_BUILD_ROOT/exllamav3-facc"
"$PP4_BUILD_PYTHON" "$PP4_BUILD_HERE/build_p2p.py" \
  --source "$PP4_BUILD_HERE/../../worker/p2p/p2p.cu"
