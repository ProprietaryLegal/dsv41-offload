# Whole-worker qualification runtime

PP4 rebuilt the public EXL3 dependencies for the required x86host interpreter:
`/home/operator/miniconda3/bin/python3`, CPython 3.13.12, Torch 2.12.1+cu130
and Triton 3.7.1. Stock EXL3, FP32-accumulation EXL3 and the worker's P2P
extension all import together successfully. This is an ABI qualification;
the P2P binary has not executed on either production-pair GPU.

[runtime-build.json](runtime-build.json) records the module, compiler wheel,
source-tree and object-reuse certificate hashes. Every binary remains in PP4
scratch and must be deleted when the lane closes. The coordinator must rebuild
the prerequisites before a future window. No installed package, production
file, service or power setting was changed.

## Reproduce the prerequisite build

[build_runtime.sh](build_runtime.sh) is the CPU-only build entry point. Run it
from an owned named tmux session on x86host, with `TMUX_TMPDIR` set to its private
PP4 directory **before starting the tmux server**. It applies the reserved
CPU affinity 12–15,28–31, nice 19, eight-job/thread limits and a 48 GiB virtual
limit for the CPU build. No GPU is visible. GPU execution uses measured RSS
and the 48 GiB resident-memory guard instead; a virtual limit would reject
CUDA's large address mappings even when resident use is small.

The two public source trees are copied from
`~/dsv41-offload/o2/src/{exllamav3,exllamav3-facc}`. Both are upstream commit
`151539c77abc7ab7425d30da7a4e8e3c5c154e7b`; the facc tree retains O2's existing
FP32-accumulation change and distinct pybind internals. PP4 changes no numerical
C++/CUDA source. Its scratch setup.py loads only the build reuse hook.

The installed CUDA 12.6 compiler does not match Torch's CUDA 13.0 build. Another
existing environment's CUDA 13.3 compiler also fails the compiler/header
compatibility check with its mixed headers. The build instead uses the pinned
public compiler packages in [requirements-runtime.lock](requirements-runtime.lock):

| Package | Version |
|---|---|
| nvidia-cuda-nvcc | 13.0.88 |
| nvidia-nvvm | 13.0.88 |
| nvidia-cuda-crt | 13.0.88 |
| nvidia-cuda-cccl | 13.0.85 |

The existing interpreter's nvidia-cuda-runtime 13.0.96 headers and libraries
remain read-only dependencies. A scratch `libcudart.so` symlink supplies the
linker name for the installed `libcudart.so.13`. All downloads, toolchain files,
build outputs and caches stay under `/opt/dsv41/x86host-data/pp4/runtime`.
The build targets SM86 only. [build_p2p.py](build_p2p.py) builds/imports P2P
without creating a CUDA context and writes its normal source-key/hash pin.

[runtime_reuse.py](runtime_reuse.py) certified 82 facc objects for reuse;
108 were rebuilt. It compares the actual Ninja compile commands and all
transitive source/header bytes. The module and pybind-internals definitions
may differ only when neither identifier appears in any input. A changed
numerical header, a pybind dependency, a missing input or any other flag change
requires compilation. At use it rechecks every input and object hash, writes
the complete dependency file and lets Ninja record the real wrapper command.
Reused objects retain their original stock paths in diagnostic metadata.
The complete certificate is preserved outside scratch as campaign artifact
`PP4.runtime-reuse-manifest.json.gz`; its uncompressed hash is in the receipt.

A reproduction attempt exposed an idempotence defect: extracting unchanged
wheel files updated header timestamps and triggered an unnecessary rebuild.
That owned job was stopped before linking; both qualified module hashes stayed
unchanged. The script now preserves unchanged file contents and timestamps.
The final CPU P2P helper executed successfully; the revised full build entry
point has shell validation but has not been rerun from an empty directory.

## Whole-worker measurement

[full_worker_run_locked.sh](full_worker_run_locked.sh) takes only the assigned
solo UUID's lock with `flock -w 3600`, checks its existing 225 W cap and
occupancy, and applies the CPU limits. Set `PP4_SOURCE_ROOT` to the staged
PP4 source and `PP4_EXL3_PATH` to the stock/facc module directories, separated
by a colon. Private CUDA, Triton and Torch caches are mandatory.

[full_worker.py](full_worker.py) executes both complete logical K2 TP2 ranks
on the solo card with production weights, query block 5 and output depths
4/5. The matched timing fixture contains four real N4 request states. It
interleaves greedy and sampled graphs over five repetitions, reversing cell
order each repetition. CUDA events bracket each complete worker call and
include host gaps inside it. Physical-pair collectives, transport and target
verification are excluded.

On all 24 NB1/N4 states, both output depths and every batch size 1–4, the
switch-enabled all-T0 request must match the original graph's proposals and
confidence exactly. Batches 2–4 also use both alternating T0/T1 masks, checking
every T0 row's proposals/confidence and zero meaningful q payload. Unused q
padding beyond the output depth is outside that check.

The 24 states comprise twelve frozen synthetic states and twelve real N4
captures; captures stay on x86host. A separate B1 comparison checks both depths
against the archived NB1 K2 digests in
[n4-greedy-reference.json](n4-greedy-reference.json). That file contains only
hashes, with the archived source receipt pinned. The comparison detects runtime
arithmetic drift; matching ON/OFF outputs in the rebuilt environment alone
cannot establish identity with production's older binary/runtime.

## Incidental writes and cleanup

The first failed build briefly created its own tmux socket at
`/tmp/tmux-1000/pp4-runtime`. The exact owned process group was stopped and its
socket removed; no bulk data was written there. Replacement servers use the
private socket directory from startup.

Two initial pip index queries created four HTTP metadata cache files outside
scratch, totaling 10,427 bytes. Their exact package names were verified and
only those four owned files were moved into scratch for cleanup. Subsequent
pip commands use the private cache. Early local mypy/ruff commands also used
the worktree caches; later checks use the prescribed private cache/no-cache
options. The root lane owns final worktree-cache and scratch removal.
