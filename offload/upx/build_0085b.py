"""Build the default-OFF 0085b idle cache release overlay (UPAUDIT B33, ranked fix 5) from the v5 image tree.

henryous patch 0085b (24.10.2026, image v29): after a long prefill the engine core, once idle for
VLLM_DSV41_RELEASE_GRACE_S (5 s) with nothing in flight, calls collective_rpc("dsv41_release_if_due"); each rank empties
its allocator cache once if at least the threshold of prefill tokens ran since its last release. On GB10 GPU memory is
system RAM, so the released cache comes back as MemAvailable. Switches: VLLM_DSV41_RELEASE_TOKENS (threshold, 0 = off)
and VLLM_DSV41_RELEASE_FILE (a file holding the threshold, read at every idle check, overriding it live). Neither set =
the image's behaviour.

engine/core.py: the donor hunks apply to v5 with ``patch --fuzz=0`` unchanged. worker/gpu_worker.py: hunk 2 (the
worker method) applies unchanged at an offset; hunks 1 and 3 anchor on the donor's patch-0066 lines, which v5 does not
have, so their added lines are placed at the equivalent v5 positions (WORKER_PORT): the switch block right after
``logger = init_logger(__name__)`` (where the donor has it, below 0066's block), and the prefill-token count right
before ``with self.annotate_profile(scheduler_output):`` (the donor has 0066's MARKOV_TRIM block in between).

Neither file is mounted by production (K1's 13 files, the remote drafter and Engram mounts), UP1/UP2, R-UPF's 0091 or
R-UP3's arms (tests/test_build_0085b.py). "Default OFF" means nothing changes unless an arm mounts these files
(UPX.env, x0085b.env) and sets one of the switches.

    python3 -m offload.upx.build_0085b check    # overlay, port patch and MANIFEST-0085b.json equal a fresh build
    python3 -m offload.upx.build_0085b write    # regenerate them (refuses changed donor bytes)
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PATCH = HERE / 'patches/0085b-dsv41-idle-cache-release.patch'
PATCH_SHA256 = '73a293320e13d0d9135fe203f352b925c1b8c77e859cb82c1a58b7fe3444662d'
DONOR = {'repo': 'github.com/HeNryous/deepseek-v41-1017-dgx-spark',
         'tip': '382071728ce69700b28e7a5ea197ab0a48b2232c',
         'patch_commit': '49473699d0d60acec3cdd4fe83c332da2f51766e'}
IMAGE = 'ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5'
IMAGE_SRC = Path(os.environ.get('UPX_IMAGE_SRC', '/opt/dsv41/data/src-image'))
CONTAINER = '/usr/local/lib/python3.12/dist-packages/'
CORE = 'vllm/v1/engine/core.py'
WORKER = 'vllm/v1/worker/gpu_worker.py'
FILES = (CORE, WORKER)
OVERLAY = HERE / 'overlay'
MANIFEST = HERE / 'MANIFEST-0085b.json'
PORT_PATCH = HERE / 'patches/0085b-v5.patch'
SCHEMA = 'dsv41-upx-0085b-v1'
# gpu_worker.py hunks without a v5 anchor: (a line only that hunk adds, v5 text before the insertion, v5 text after it).
# The before+after pair must occur exactly once in the base; the hunk's added lines go between them, unchanged.
WORKER_PORT = (
    ('_DSV41_RELEASE_TOKENS = _dsv41_int_env("VLLM_DSV41_RELEASE_TOKENS")\n',
     'logger = init_logger(__name__)\n', '\n\ndef _num_workspace_lanes('),
    ('            # dsv41 patch 0085b: count prefill tokens only; the release happens at idle\n',
     '                comm_postprocess=comm_postprocess,\n            )\n\n',
     '        with self.annotate_profile(scheduler_output):\n'),
)
SWITCHES = {
    'VLLM_DSV41_RELEASE_TOKENS': "prefill-token threshold; '' or 0 = off (both files)",
    'VLLM_DSV41_RELEASE_FILE': 'path of a file holding the threshold, read at every idle check (0 = off); set = engine '
                               'core idles through the grace period',
    'VLLM_DSV41_RELEASE_GRACE_S': 'idle seconds before the release (default 5; parsed at import, used only when a '
                                  'switch above is set)',
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def donor_bytes() -> bytes:
    patch = PATCH.read_bytes()
    if sha256(patch) != PATCH_SHA256:
        raise ValueError('vendored 0085b patch bytes changed')
    return patch


def sections(text: str) -> dict[str, str]:
    """Per-file sections of the donor's ``diff -ru`` patch (the description lines before the first one are dropped)."""
    found: dict[str, str] = {}
    for part in re.split(r'(?m)^(?=diff -ru )', text):
        match = re.match(r'diff -ru a/(\S+) b/\S+\n', part)
        if match:
            found[match.group(1)] = part
    if sorted(found) != sorted(FILES):
        raise ValueError(f'0085b patch touches {sorted(found)}, expected {sorted(FILES)}')
    return found


def hunks(section: str) -> tuple[str, list[str]]:
    head, *rest = re.split(r'(?m)^(?=@@ )', section)
    return head, rest


def added(hunk: str) -> list[str]:
    return [line[1:] + '\n' for line in hunk.splitlines() if line.startswith('+')]


def apply(section: str, rel: str, base: bytes) -> bytes:
    """``patch --fuzz=0`` in a scratch tree (offsets allowed, fuzz refused)."""
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
        target = Path(tmp) / rel
        target.parent.mkdir(parents=True)
        target.write_bytes(base)
        result = subprocess.run(['patch', '-p1', '--batch', '--no-backup-if-mismatch', '--fuzz=0', '-r', '-'],
                                cwd=tmp, input=section, text=True, capture_output=True)
        if result.returncode != 0:
            raise ValueError(f'0085b hunks do not apply to {rel}: {result.stdout}{result.stderr}')
        return target.read_bytes()


def split_worker(section: str) -> tuple[str, list[list[str]]]:
    """The worker section without its two unanchored hunks, and those hunks' added lines in WORKER_PORT order."""
    head, rest = hunks(section)
    if len(rest) != 3:
        raise ValueError(f'gpu_worker.py: {len(rest)} hunks, expected 3')
    ported: list[list[str]] = []
    keep: list[str] = []
    for hunk in rest:
        lines = added(hunk)
        if any(lines.count(marker) == 1 for marker, _, _ in WORKER_PORT):
            ported.append(lines)
        else:
            keep.append(hunk)
    order = [next((p for p in ported if marker in p), None) for marker, _, _ in WORKER_PORT]
    if len(ported) != len(WORKER_PORT) or any(o is None for o in order) or len(keep) != 1:
        raise ValueError('gpu_worker.py hunks are not the expected switch block, worker method and token count')
    if any(line.startswith('-') for hunk in rest for line in hunk.splitlines() if not line.startswith('---')):
        raise ValueError('gpu_worker.py: the donor hunks were expected to add lines only')
    return head + keep[0], [o for o in order if o is not None]


def build_file(rel: str, base: bytes, section: str) -> bytes:
    if rel == CORE:
        return apply(section, rel, base)
    rest, ported = split_worker(section)
    text = apply(rest, rel, base).decode()
    for (_, before, after), lines in zip(WORKER_PORT, ported):
        if text.count(before + after) != 1:
            raise ValueError(f'{rel}: expected exactly one v5 anchor {before.strip()!r} + {after.strip()!r}')
        text = text.replace(before + after, before + ''.join(lines) + after, 1)
    return text.encode()


def port_diff(rel: str, base: bytes, built: bytes) -> str:
    return ''.join(difflib.unified_diff(base.decode().splitlines(keepends=True), built.decode().splitlines(keepends=True),
                                        f'a/{rel}', f'b/{rel}', n=3))


def build() -> tuple[dict[str, object], dict[Path, bytes]]:
    found = sections(donor_bytes().decode())
    entries: list[dict[str, object]] = []
    outputs: dict[Path, bytes] = {}
    diffs: list[str] = []
    for rel in FILES:
        base = (IMAGE_SRC / rel).read_bytes()
        built = build_file(rel, base, found[rel])
        outputs[OVERLAY / rel] = built
        diffs.append(port_diff(rel, base, built))
        entries.append({'path': rel, 'container_path': CONTAINER + rel, 'base': 'image:' + rel,
                        'base_sha256': sha256(base), 'overlay': str((OVERLAY / rel).relative_to(ROOT)),
                        'overlay_sha256': sha256(built),
                        'port': 'donor hunks unchanged (patch --fuzz=0)' if rel == CORE else
                                'hunk 2 unchanged (patch --fuzz=0, offset); hunks 1 and 3 placed at v5 anchors (WORKER_PORT)'})
    port = ''.join(diffs).encode()
    outputs[PORT_PATCH] = port
    manifest: dict[str, object] = {
        'schema': SCHEMA, 'donor': DONOR, 'patch': str(PATCH.relative_to(ROOT)), 'patch_sha256': PATCH_SHA256,
        'port_patch': str(PORT_PATCH.relative_to(ROOT)), 'port_patch_sha256': sha256(port),
        'image': IMAGE, 'vllm_base': '6e517b15',
        'default': 'OFF: nothing changes unless an arm mounts these files (offload/upx/configs.py) and sets a switch',
        'switches': SWITCHES, 'files': entries}
    return manifest, outputs


def check() -> None:
    manifest, outputs = build()
    for path, data in outputs.items():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f'{path.relative_to(ROOT)} differs from a fresh 0085b build')
    if json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError('MANIFEST-0085b.json differs from a fresh build')


def write() -> None:
    manifest, outputs = build()
    for path, data in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('check', 'write'))
    args = parser.parse_args()
    if args.action == 'check':
        check()
    else:
        write()
    print(f'0085b overlay {args.action}: OK')


if __name__ == '__main__':
    main()
