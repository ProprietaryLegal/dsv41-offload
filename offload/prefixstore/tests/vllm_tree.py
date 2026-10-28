"""Build a CPU-importable vLLM tree = image v5 source + K1 overlay + RP1/MC1 overlay (symlink farm).

The image source is never modified: every unchanged file is a symlink into it, and only the mounted overlay
paths are real copies. Used by test_vllm_cases.py, which runs the cases under the kvpack-build venv (it has
vLLM's CPU-side dependencies; the miniconda interpreter does not).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any, cast

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
SRC = Path(os.environ.get('SRC', '/opt/dsv41/data/src-image'))
VENV_PYTHON = Path(os.environ.get('MC1_VLLM_PYTHON', '/opt/dsv41/data/kvpack-build/venv/bin/python'))


def _replace(link: Path, source: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, link)


def overlay_bytes(dest: Path, rel: str, data: bytes) -> None:
    """Mount extra bytes at a container-relative path (e.g. a pending package's file, for a composition run)."""
    target = dest / rel
    if target.is_symlink() or target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def build(dest: Path) -> Path:
    """Create <dest>/vllm with the overlays applied; returns dest (the PYTHONPATH entry)."""
    if not (SRC / 'vllm').is_dir():
        raise FileNotFoundError(f'image source snapshot missing: {SRC}/vllm')
    shutil.copytree(SRC / 'vllm', dest / 'vllm', symlinks=True, copy_function=os.symlink,
                    ignore=shutil.ignore_patterns('__pycache__'))
    k1 = cast(dict[str, Any], json.loads((ROOT / 'offload/kvpack/MANIFEST.json').read_text()))
    for entry in k1['files']:
        if entry['path'].startswith('vllm/'):
            _replace(dest / entry['path'], ROOT / 'offload/kvpack/overlay' / entry['path'])
    mc1 = cast(dict[str, Any], json.loads((HERE / 'MANIFEST.json').read_text()))
    for entry in list(mc1['files']) + list(mc1.get('new_files', [])):
        _replace(dest / entry['path'], HERE / 'overlay' / entry['path'])
    return dest
