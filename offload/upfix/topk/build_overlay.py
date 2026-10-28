"""Build the default-OFF indexer_topk.py overlay that routes persistent top-k to the fixed JIT op.

UPS1 #5: our SM12x indexer resolves ``auto`` to ``persistent`` (cooperative top-k is disabled on the
SM12x family), so vLLM's persistent_topk kernels are live in production without the upstream fixes
#62467, #62936, #63475 (and #63118). The vendored files at 3090266d8 also carry #59645 (cooperative candidate
overflow, the prerequisite of #62936) and #60721 (header consolidation): six commits touch them since the image's
6e517b15. The fixed kernels are built as ``torch.ops.dsv41_topk``
(csrc/dsv41_topk.cu). This overlay is the image's own file plus exactly two anchored edits:
a module-level switch (``VLLM_DSV41_TOPK_FIX``, default "0") that loads the built library only
when set, and the persistent branch calling the fixed op only when set. With the switch unset the
behaviour is the image's, and the file is only mounted by the topk arm (offload/upfix/configs.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REL = 'vllm/model_executor/layers/indexer_topk.py'
IMAGE_SRC = Path(os.environ.get('UPFIX_IMAGE_SRC', '/opt/dsv41/data/src-image'))
OVERLAY = HERE / 'overlay' / REL
MANIFEST = HERE / 'MANIFEST-topk.json'
VLLM_FIX_COMMIT = '3090266d88b6c3abbe18483fa5bbc7de293d8ef0'
# git blob ids at VLLM_FIX_COMMIT; the vendored bytes must match them exactly.
VENDORED = {
    'persistent_topk.cuh': 'bd18483b48d0ae6aed27a9829baa4283a1212cc6',
    'sampled_topk.cuh': 'c25f8ed8e73137b37c94170bd4852dafa3bbafb6',
    'topk_histogram_4096.cuh': 'b5aeb2d78cd6b0419c291d58b03ba96c306df555',
    'persistent_topk_overflow.cuh': '9ae116998e89eec5c4740a737e722505791d8f8c',
    'topk_histogram_4096_overflow.cuh': '7433a0362acf866b27dbec76375bea131236df2a',
}
LAUNCHER_UPSTREAM = {'path': 'csrc/topk.cu', 'blob': 'd5ff2a8386ebdb10fb263ecd6a08f6bcf53ff4c2'}

SWITCH_ANCHOR = 'RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024\n'
SWITCH = '''
# dsv41 R-UPF (UPS1 #5): VLLM_DSV41_TOPK_FIX=1 routes the persistent backend to
# torch.ops.dsv41_topk.persistent_topk (vLLM #62467/#62936/#63475/#63118 at 3090266d8, offload/upfix/topk),
# loaded once from VLLM_DSV41_TOPK_FIX_SO. Default "0": the image's _C kernel, unchanged.
import os as _dsv41_os

_DSV41_TOPK_FIX = _dsv41_os.environ.get("VLLM_DSV41_TOPK_FIX", "0") == "1"
if _DSV41_TOPK_FIX:
    _dsv41_topk_so = _dsv41_os.environ.get("VLLM_DSV41_TOPK_FIX_SO", "")
    if not _dsv41_topk_so or not _dsv41_os.path.isfile(_dsv41_topk_so):
        raise RuntimeError(
            "VLLM_DSV41_TOPK_FIX=1 needs VLLM_DSV41_TOPK_FIX_SO naming the built dsv41_topk library"
        )
    torch.ops.load_library(_dsv41_topk_so)
'''
CALL_ANCHOR = '            torch.ops._C.persistent_topk(\n'
CALL = ('            (\n'
        '                torch.ops.dsv41_topk.persistent_topk  # dsv41 R-UPF, VLLM_DSV41_TOPK_FIX=1\n'
        '                if _DSV41_TOPK_FIX\n'
        '                else torch.ops._C.persistent_topk\n'
        '            )(\n')


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def build(base: str) -> str:
    for anchor in (SWITCH_ANCHOR, CALL_ANCHOR):
        if base.count(anchor) != 1:
            raise ValueError(f'indexer_topk.py anchor {anchor.strip()!r} must occur exactly once')
    return base.replace(SWITCH_ANCHOR, SWITCH_ANCHOR + SWITCH, 1).replace(CALL_ANCHOR, CALL, 1)


def manifest(base: bytes, overlay: bytes) -> dict[str, object]:
    csrc = HERE / 'csrc'
    return {'schema': 'dsv41-upfix-topk-v1', 'vllm_fix_commit': VLLM_FIX_COMMIT,
            'upstream_prs': ['#62467', '#62936', '#63475', '#63118'],
            'also_in_vendored_files': {'#59645': 'cooperative candidate overflow, prerequisite of #62936',
                                       '#60721': 'header consolidation'},
            'vendored': {name: {'git_blob': blob, 'sha256': sha256((csrc / name).read_bytes())}
                         for name, blob in VENDORED.items()},
            'launcher': {'file': 'offload/upfix/topk/csrc/dsv41_topk.cu', 'ported_from': LAUNCHER_UPSTREAM,
                         'sha256': sha256((csrc / 'dsv41_topk.cu').read_bytes())},
            'op': 'torch.ops.dsv41_topk.persistent_topk',
            'schema': ('persistent_topk(Tensor logits, Tensor lengths, Tensor! output, '
                       'Tensor workspace, int k, int max_seq_len) -> ()'),
            'overlay': {'path': REL, 'container_path': '/usr/local/lib/python3.12/dist-packages/' + REL,
                        'base_sha256': sha256(base), 'overlay_sha256': sha256(overlay)},
            'switches': {'VLLM_DSV41_TOPK_FIX': "'0' default (image kernel); '1' fixed op",
                         'VLLM_DSV41_TOPK_FIX_SO': 'path of the built library inside the container'}}


def check() -> None:
    for name, blob in VENDORED.items():
        if git_blob((HERE / 'csrc' / name).read_bytes()) != blob:
            raise ValueError(f'vendored {name} differs from vLLM {VLLM_FIX_COMMIT[:9]}')
    base = (IMAGE_SRC / REL).read_bytes()
    built = build(base.decode()).encode()
    if OVERLAY.read_bytes() != built:
        raise ValueError('indexer_topk.py overlay differs from a fresh build')
    if json.loads(MANIFEST.read_text()) != manifest(base, built):
        raise ValueError('MANIFEST-topk.json differs from a fresh build')


def write() -> None:
    base = (IMAGE_SRC / REL).read_bytes()
    built = build(base.decode()).encode()
    OVERLAY.parent.mkdir(parents=True, exist_ok=True)
    OVERLAY.write_bytes(built)
    MANIFEST.write_text(json.dumps(manifest(base, built), indent=1, sort_keys=True) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'write'))
    args = parser.parse_args()
    if args.action == 'check':
        check()
    else:
        write()
    print(f'topk overlay {args.action}: OK')


if __name__ == '__main__':
    main()
