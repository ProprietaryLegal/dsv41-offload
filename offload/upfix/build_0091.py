"""Build the default-OFF 0091 robustness overlay (UPS1 #4) from a base vLLM tree.

henryous patch 0091 backports six vLLM fixes onto vLLM 6e517b15 (the base of our v5 image):
#59096 file-tier mark_miss assert, #59880 spec-decode step blocks per KV group, #61174 stale
block-table rows, #63165 + #61000 head-node shm_broadcast growth/ownership, #59504 KV zeroing next
to async loads. Our image predates the donor's 0071+ scheduler header, so hunk 1 of scheduler.py
(the VLLM_DSV41_SPEC_RESERVE_PER_GROUP switch) has no anchor; it is placed right after the
existing ``import os as _dsv41_os`` line instead. Every other hunk applies with ``patch`` unchanged.

None of the four files is in K1's production mounts. scheduler.py is also replaced by the
upstream_adopt overlay (0075 replay skip), so a second variant composes 0091 onto that file.
"Default OFF" means the overlay changes nothing until an arm mounts it (configs.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PATCH = HERE / 'patches/0091-dsv41-upstream-fixes.patch'
PATCH_SHA256 = '25627e5094f006895a199b4f32b9f381669d79766a66e7163533b313ed16dbf6'
DONOR = {'repo': 'github.com/HeNryous/deepseek-v41-1017-dgx-spark',
         'tip': '382071728ce69700b28e7a5ea197ab0a48b2232c',
         'patch_commit': 'f1b6207c5c774f3bdefa686ae6150f4042a4fd06'}
IMAGE_SRC = Path(os.environ.get('UPFIX_IMAGE_SRC', '/opt/dsv41/data/src-image'))
CONTAINER = '/usr/local/lib/python3.12/dist-packages/'
SCHEDULER = 'vllm/v1/core/sched/scheduler.py'
FILES = ('vllm/distributed/device_communicators/shm_broadcast.py', SCHEDULER,
         'vllm/v1/kv_offload/tiering/async_lookup.py', 'vllm/v1/worker/gpu/block_table.py')
UPSTREAM = {'vllm/distributed/device_communicators/shm_broadcast.py': ['#63165', '#61000'],
            SCHEDULER: ['#59880', '#59096'],
            'vllm/v1/kv_offload/tiering/async_lookup.py': ['#59096'],
            'vllm/v1/worker/gpu/block_table.py': ['#61174', '#59504']}
# Variant -> (overlay dir, base of each file). 'image' bases every file on the image; '0075' bases
# scheduler.py on the upstream_adopt overlay that the 0075 replay-skip arm mounts.
OVERLAY = HERE / 'overlay'
OVERLAY_0075 = HERE / 'overlay-0075'
ADOPT_SCHEDULER = ROOT / 'offload/upstream_adopt/overlay' / SCHEDULER
ANCHOR = 'import os as _dsv41_os\n'
MANIFEST = HERE / 'MANIFEST-0091.json'


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sections(text: str) -> dict[str, str]:
    """Per-file sections of the donor's ``diff -ruN`` patch."""
    parts = re.split(r'(?m)^(?=diff -ruN )', text)
    found: dict[str, str] = {}
    for part in parts:
        match = re.match(r'diff -ruN a/(\S+) b/\S+\n', part)
        if match:
            found[match.group(1)] = part
    if sorted(found) != sorted(FILES):
        raise ValueError(f'0091 patch touches {sorted(found)}, expected {sorted(FILES)}')
    return found


def split_spec_switch(section: str) -> tuple[str, list[str]]:
    """Remove scheduler hunk 1 (the unanchored switch) and return its added lines."""
    head, *hunks = re.split(r'(?m)^(?=@@ )', section)
    first = hunks[0]
    if 'VLLM_DSV41_SPEC_RESERVE_PER_GROUP' not in first:
        raise ValueError('scheduler hunk 1 is not the #59880 switch')
    added = [line[1:] + '\n' for line in first.splitlines() if line.startswith('+')]
    if len(added) != 6:
        raise ValueError(f'#59880 switch has {len(added)} added lines, expected 6')
    return head + ''.join(hunks[1:]), added


def apply(section: str, rel: str, base: bytes) -> bytes:
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
        target = Path(tmp) / rel
        target.parent.mkdir(parents=True)
        target.write_bytes(base)
        result = subprocess.run(['patch', '-p1', '--batch', '--no-backup-if-mismatch', '--fuzz=0',
                                 '-r', '-'], cwd=tmp, input=section, text=True, capture_output=True)
        if result.returncode != 0:
            raise ValueError(f'0091 hunks do not apply to {rel}: {result.stdout}{result.stderr}')
        return target.read_bytes()


def build_file(rel: str, base: bytes, section: str) -> bytes:
    if rel != SCHEDULER:
        return apply(section, rel, base)
    rest, switch = split_spec_switch(section)
    text = apply(rest, rel, base).decode()
    if text.count(ANCHOR) != 1:
        raise ValueError(f'{rel}: expected exactly one {ANCHOR.strip()!r} anchor for the #59880 switch')
    return text.replace(ANCHOR, ANCHOR + ''.join(switch), 1).encode()


def plan() -> list[tuple[str, str, Path, Path]]:
    """(variant, relative path, base file, overlay file)."""
    rows = [('image', rel, IMAGE_SRC / rel, OVERLAY / rel) for rel in FILES]
    rows.append(('0075', SCHEDULER, ADOPT_SCHEDULER, OVERLAY_0075 / SCHEDULER))
    return rows


def build() -> tuple[dict[str, object], dict[Path, bytes]]:
    patch = PATCH.read_bytes()
    if sha256(patch) != PATCH_SHA256:
        raise ValueError('vendored 0091 patch bytes changed')
    found = sections(patch.decode())
    entries: list[dict[str, object]] = []
    outputs: dict[Path, bytes] = {}
    for variant, rel, base_path, out in plan():
        base = base_path.read_bytes()
        built = build_file(rel, base, found[rel])
        outputs[out] = built
        entries.append({'variant': variant, 'path': rel, 'container_path': CONTAINER + rel,
                        'base': str(base_path.relative_to(ROOT)) if base_path.is_relative_to(ROOT) else 'image:' + rel,
                        'base_sha256': sha256(base), 'overlay': str(out.relative_to(ROOT)),
                        'overlay_sha256': sha256(built), 'upstream': UPSTREAM[rel]})
    manifest: dict[str, object] = {
        'schema': 'dsv41-upfix-0091-v1', 'donor': DONOR, 'patch': str(PATCH.relative_to(ROOT)),
        'patch_sha256': PATCH_SHA256, 'vllm_base': '6e517b15',
        'default': 'OFF: nothing changes unless an arm mounts these files (offload/upfix/configs.py)',
        'switch_when_mounted': {'VLLM_DSV41_SPEC_RESERVE_PER_GROUP': "'1' default (#59880 on); '0' restores the old count"},
        'files': entries}
    return manifest, outputs


def check() -> None:
    manifest, outputs = build()
    for path, data in outputs.items():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f'{path.relative_to(ROOT)} differs from a fresh 0091 build')
    if json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError('MANIFEST-0091.json differs from a fresh build')


def write() -> None:
    manifest, outputs = build()
    for path, data in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'write'))
    args = parser.parse_args()
    if args.action == 'check':
        check()
    else:
        write()
    print(f'0091 overlay {args.action}: OK')


if __name__ == '__main__':
    main()
